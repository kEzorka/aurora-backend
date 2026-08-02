"""Манифест артефакта: поля из DATA_CONTRACT.md §3 и флаг `published`.

Проверяется набор ключей целиком, а не выборочно: манифест — контракт с
читателем, и поле, которого в нём не оказалось, никто не заметит до разбора
инцидента.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from storage.manifest import (
    Degraded,
    Input,
    Model,
    build_manifest,
    mark_published,
    read_manifest,
    write_manifest,
)

#: `source` здесь — провенанс из `canon.SOURCES`, а не имя потока скачивания:
#: в docs/DATA_CONTRACT.md §3 пример манифеста и таблица ниже него расходятся,
#: и словарь берётся один — тот, которым уже помечены наборы адаптеров.
INPUTS = (
    Input("ifs-analysis", "2026-07-31T18:00Z", "sha256:" + "a" * 64, "ifs/0p25/oper", ("2t",)),
    Input("ifs-analysis", "2026-08-01T00:00Z", "sha256:" + "b" * 64, "ifs/0p25/oper", ("2t",)),
)
MODEL = Model("aurora", "aurora-0.25-v1.5", "9f2c1ab")
TIMINGS = {"ingest": 480, "normalize": 190, "inference": 260, "write": 520}


def _manifest(**over: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "inputs": INPUTS,
        "model": MODEL,
        "steps": 40,
        "timings_sec": TIMINGS,
        "created_at": datetime(2026, 8, 1, 8, 14, 3, tzinfo=UTC),
    }
    kwargs.update(over)
    return build_manifest("forecast/2026-08-01T00Z", **kwargs)  # type: ignore[arg-type]


def test_manifest_has_every_field_of_the_contract() -> None:
    """docs/DATA_CONTRACT.md §3: набор ключей верхнего уровня — ровно этот."""
    assert set(_manifest()) == {
        "artifact",
        "created_at",
        "inputs",
        "model",
        "steps",
        "timings_sec",
        "degraded",
        "validation",
        "published",
    }


def test_manifest_fields_carry_the_values_they_were_given() -> None:
    manifest = _manifest()
    assert manifest["artifact"] == "forecast/2026-08-01T00Z"
    assert manifest["created_at"] == "2026-08-01T08:14:03Z"
    assert manifest["model"] == {
        "name": "aurora",
        "checkpoint": "aurora-0.25-v1.5",
        "revision": "9f2c1ab",
    }
    assert manifest["inputs"] == [
        {
            "source": "ifs-analysis",
            "valid_time": "2026-07-31T18:00Z",
            "checksum": INPUTS[0].checksum,
            "stream": "ifs/0p25/oper",
            "fields": ["2t"],
        },
        {
            "source": "ifs-analysis",
            "valid_time": "2026-08-01T00:00Z",
            "checksum": INPUTS[1].checksum,
            "stream": "ifs/0p25/oper",
            "fields": ["2t"],
        },
    ]
    assert manifest["validation"] == "validation.json"


def test_a_fresh_manifest_is_not_published() -> None:
    """Флаг ставится последним действием публикации, а не при сборке
    (docs/STORAGE.md §5)."""
    assert _manifest()["published"] is False


def test_unknown_input_source_is_refused() -> None:
    # Имя потока скачивания — не провенанс: в манифест оно попасть не должно.
    bad = (Input("ecmwf-opendata-ifs", "2026-08-01T00:00Z", "sha256:" + "c" * 64, "oper", ("2t",)),)
    with pytest.raises(ValueError, match="source"):
        _manifest(inputs=bad)


def test_checksum_without_algorithm_is_refused() -> None:
    """`sha256:` не украшение: без имени алгоритма контрольную сумму не с чем
    сравнить, когда алгоритм сменится."""
    bad = (Input("era5t", "2026-08-01T00:00Z", "d" * 64, "reanalysis-era5-single-levels", ("ci",)),)
    with pytest.raises(ValueError, match="checksum"):
        _manifest(inputs=bad)


def test_an_input_without_a_stream_is_refused() -> None:
    """`ifs/0p25/oper` и `aifs-single/0p25/oper` — один провенанс `ifs-analysis`,
    и облачность приходит только из второго (BACKLOG 1.10). Без потока манифест
    не отвечает на вопрос, откуда взялось поле."""
    bad = (Input("ifs-analysis", "2026-08-01T00:00Z", "sha256:" + "e" * 64, "", ("2t",)),)
    with pytest.raises(ValueError, match="stream"):
        _manifest(inputs=bad)


def test_an_input_without_fields_is_refused() -> None:
    """Вход на срок пятидневной давности без списка полей читается как «весь
    срез пятидневный», а устарела там одна сплочённость льда."""
    bad = (Input("era5t", "2026-07-27T00:00Z", "sha256:" + "f" * 64, "era5", ()),)
    with pytest.raises(ValueError, match="fields"):
        _manifest(inputs=bad)


def test_a_field_outside_the_canon_is_refused() -> None:
    """Опечатка в имени поля иначе доедет до манифеста и останется там навсегда:
    манифест никто не перечитывает, пока не случится разбор полётов."""
    bad = (Input("era5t", "2026-07-27T00:00Z", "sha256:" + "f" * 64, "era5", ("sithick",)),)
    with pytest.raises(ValueError, match="fields"):
        _manifest(inputs=bad)


def test_manifest_without_inputs_is_refused() -> None:
    with pytest.raises(ValueError, match="inputs"):
        _manifest(inputs=())


def test_timings_must_carry_all_four_stages() -> None:
    """BACKLOG 6.2 считает по этим ключам; недостающий этап там станет нулём,
    а не пропуском."""
    with pytest.raises(ValueError, match="timings_sec"):
        _manifest(timings_sec={"ingest": 480, "inference": 260, "write": 520})


def test_unexpected_timing_key_is_refused() -> None:
    with pytest.raises(ValueError, match="timings_sec"):
        _manifest(timings_sec={**TIMINGS, "upload": 12})


def test_zero_steps_is_refused() -> None:
    with pytest.raises(ValueError, match="steps"):
        _manifest(steps=0)


def test_a_clean_run_says_so_instead_of_staying_silent() -> None:
    """`degraded: null` есть всегда. Отсутствие ключа читатель не отличит от
    манифеста, написанного старым кодом, и «деградации не было» пришлось бы
    предполагать ровно там, где предполагать нельзя."""
    assert _manifest()["degraded"] is None


def test_a_fallback_to_gfs_says_it_was_a_fallback() -> None:
    """`inputs[].source == "gfs-analysis"` на это не отвечает: GFS бывает в
    манифесте и по выбору (отладка без ключей ECMWF), и по откату, и алерта
    заслуживает только второе (docs/PIPELINE.md §2)."""
    degraded = _manifest(
        degraded=Degraded("late_input", ("ifs/0p25/oper",), "2026-08-01T08:30:00Z")
    )["degraded"]

    assert degraded == {
        "reason": "late_input",
        "waited_for": ["ifs/0p25/oper"],
        "deadline": "2026-08-01T08:30:00Z",
    }


def test_a_degradation_without_a_reason_from_the_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match=r"degraded\.reason"):
        _manifest(degraded=Degraded("slow", ("ifs/0p25/oper",), "2026-08-01T08:30:00Z"))


def test_a_degradation_must_name_what_it_waited_for() -> None:
    """«Прогон деградировал» без потока — это жалоба без адресата: дежурному
    дальше некуда идти."""
    with pytest.raises(ValueError, match=r"degraded\.waited_for"):
        _manifest(degraded=Degraded("late_input", (), "2026-08-01T08:30:00Z"))


def test_manifest_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.json", _manifest())
    assert read_manifest(path) == _manifest()
    assert json.loads(path.read_text(encoding="utf-8"))["published"] is False


def test_mark_published_flips_the_flag_and_touches_nothing_else(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.json", _manifest())
    published = mark_published(path)
    assert published["published"] is True
    assert read_manifest(path)["published"] is True
    assert {k: v for k, v in published.items() if k != "published"} == {
        k: v for k, v in _manifest().items() if k != "published"
    }


def test_write_manifest_refuses_a_manifest_already_marked_published(tmp_path: Path) -> None:
    """Опубликованным артефакт делает `mark_published` после того, как всё
    легло на диск. Записать флаг сразу — значит объявить срез целым до того,
    как он таким стал."""
    manifest = {**_manifest(), "published": True}
    with pytest.raises(ValueError, match="published"):
        write_manifest(tmp_path / "manifest.json", manifest)

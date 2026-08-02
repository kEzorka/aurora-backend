"""Атомарная публикация прогона (BACKLOG 2.3).

Читатель ходит через указатель `forecast/current`. Пока указатель не
переставлен, прогон для него не существует — поэтому прерванная публикация
проверяется не «упало красиво», а «указатель не сдвинулся».

Сетка маленькая: проверяется порядок действий, а не объём.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from contracts import canon
from storage import publish
from storage.manifest import Input, Model, build_manifest, read_manifest
from storage.publish import (
    current_run,
    previous_run,
    publish_run,
    stage_path,
)
from storage.write import write_layer

#: Слой на восемь часовых переменных: из него получается и `coarse` теста, и
#: `hourly`, и сокращение в `previous` — оно берёт ровно эти имена.
STAGED_COARSE = canon.Layer("coarse", canon.HOURLY_VARS, (), canon.STEP_HOURS, 2)
STAGED_HOURLY = canon.Layer("hourly", canon.HOURLY_VARS, (), canon.FINE_STEP_HOURS, 2)


def _tiny(value: float, steps: int = 2) -> xr.Dataset:
    ny, nx = 6, 8
    return xr.Dataset(
        {
            name: (("time", "lat", "lon"), np.full((steps, ny, nx), value, dtype=np.float32))
            for name in canon.HOURLY_VARS
        },
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)],
                dtype="datetime64[ns]",
            ),
            "lat": np.linspace(90.0, -90.0, ny),
            "lon": np.linspace(-180.0, 179.75, nx),
        },
        attrs={"init_time": "2026-08-01T00:00:00Z", "kind": "forecast"},
    )


def _manifest(run_id: str) -> dict[str, object]:
    return build_manifest(
        f"forecast/{run_id}",
        inputs=(
            Input(
                "ifs-analysis",
                "2026-08-01T00:00Z",
                "sha256:" + "a" * 64,
                "ifs/0p25/oper",
                canon.SURFACE_INGESTED_VARS,
            ),
        ),
        model=Model("aurora", "aurora-0.25-v1.5", "9f2c1ab"),
        steps=2,
        timings_sec={"ingest": 1, "normalize": 1, "inference": 1, "write": 1},
        created_at=datetime(2026, 8, 1, 8, 0, 0, tzinfo=UTC),
    )


def _stage(
    root: Path,
    run_id: str,
    value: float = 288.0,
    *,
    layers: tuple[str, ...] = ("coarse", "hourly"),
    validated: bool = True,
) -> Path:
    """Разложить прогон в scratch так, как это сделал бы конвейер."""
    staged = stage_path(root, run_id)
    if "coarse" in layers:
        write_layer(_tiny(value), staged / "coarse", STAGED_COARSE)
    if "hourly" in layers:
        write_layer(_tiny(value), staged / "hourly", STAGED_HOURLY)
    if validated:
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "validation.json").write_text('{"ok": true, "checks": []}', encoding="utf-8")
    return staged


def test_published_run_is_reachable_through_the_current_pointer(tmp_path: Path) -> None:
    _stage(tmp_path, "2026-08-01T00Z")
    final = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    assert current_run(tmp_path) == final
    assert (tmp_path / "forecast" / "current" / "coarse").is_dir()
    assert read_manifest(final / "manifest.json")["published"] is True


def test_publication_builds_the_point_layer(tmp_path: Path) -> None:
    """`points` собирается публикацией и до подъёма флага: `published`
    означает «всё, что читатель спросит, лежит на диске». Прогон без него
    читается через отступ на `coarse` — молча и в сто раз медленнее
    (docs/STORAGE.md §3)."""
    _stage(tmp_path, "2026-08-01T00Z")
    final = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    assert (final / "points").is_dir()
    assert not (final / "points.tmp").exists()
    np.testing.assert_allclose(
        xr.open_zarr(final / "points")["2t"].values, xr.open_zarr(final / "coarse")["2t"].values
    )


def test_the_layers_lie_on_disk_in_the_layouts_they_were_promised(tmp_path: Path) -> None:
    """Проверяется артефакт, а не таблица `layout_for`. Раскладки различает
    чанк по времени: у карт он равен единице (срок за раз), у рядов — всей оси.
    Слой, записанный не в своей раскладке, проходит все прочие тесты и молча
    отвечает в сто раз дольше (docs/STORAGE.md §3)."""
    _stage(tmp_path, "2026-08-01T00Z")
    final = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    def time_chunk(layer: str) -> int:
        return int(zarr.open_group(str(final / layer))["2t"].chunks[0])

    assert time_chunk("coarse") == 1  # карты
    assert time_chunk("points") == 2  # ряды: вся ось времени тестового прогона
    assert time_chunk("hourly") == 2


def test_every_artifact_carries_both_files(tmp_path: Path) -> None:
    """Приёмка 2.4: манифест и отчёт валидатора лежат рядом с данными, и
    манифест ссылается на отчёт по имени — ссылка обязана вести в файл."""
    _stage(tmp_path, "2026-08-01T00Z")
    final = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    manifest = read_manifest(final / "manifest.json")
    assert (final / str(manifest["validation"])).is_file()


def test_a_manifest_naming_a_missing_report_is_refused(tmp_path: Path) -> None:
    _stage(tmp_path, "2026-08-01T00Z")
    manifest = {**_manifest("2026-08-01T00Z"), "validation": "checks.json"}
    with pytest.raises(ValueError, match=r"checks\.json"):
        publish_run(tmp_path, "2026-08-01T00Z", manifest=manifest)


def test_staged_directory_is_gone_after_publication(tmp_path: Path) -> None:
    """Прогон переезжает переименованием, а не копированием: копия удвоила бы
    17.4 ГБ на диске и оставила бы окно, в котором есть обе половины."""
    _stage(tmp_path, "2026-08-01T00Z")
    publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    assert not stage_path(tmp_path, "2026-08-01T00Z").exists()


def test_an_unpublished_run_is_invisible_to_the_reader(tmp_path: Path) -> None:
    """Приёмка 2.3: прерванная на середине запись читателю не видна."""
    _stage(tmp_path, "2026-08-01T00Z")
    assert current_run(tmp_path) is None


def test_a_failed_publication_leaves_the_pointer_where_it_was(tmp_path: Path) -> None:
    """Второй прогон разложен наполовину — читатель обязан видеть первый."""
    _stage(tmp_path, "2026-08-01T00Z")
    first = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    _stage(tmp_path, "2026-08-01T12Z", layers=("coarse",))
    with pytest.raises(ValueError, match="hourly"):
        publish_run(tmp_path, "2026-08-01T12Z", manifest=_manifest("2026-08-01T12Z"))

    assert current_run(tmp_path) == first


def test_a_run_that_moved_but_was_not_pointed_at_stays_invisible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Настоящее окно приёмки 2.3: прогон уже переехал в `runs/`, лежит целиком
    и с поднятым `published`, — но пока указатель не переставлен, читатель
    обязан видеть прошлый прогон, а не новый."""
    _stage(tmp_path, "2026-08-01T00Z", value=280.0)
    first = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    def _die(link: Path, target: Path) -> None:
        raise OSError("диск кончился на перестановке указателя")

    monkeypatch.setattr(publish, "_point", _die)
    _stage(tmp_path, "2026-08-01T12Z", value=290.0)
    with pytest.raises(OSError, match="указател"):
        publish_run(tmp_path, "2026-08-01T12Z", manifest=_manifest("2026-08-01T12Z"))

    second = tmp_path / "runs" / "2026-08-01T12Z"
    assert read_manifest(second / "manifest.json")["published"] is True
    assert current_run(tmp_path) == first
    np.testing.assert_allclose(
        xr.open_zarr(tmp_path / "forecast" / "current" / "coarse")["2t"].values, 280.0
    )


def test_an_interrupted_reduction_is_not_taken_for_a_finished_one(tmp_path: Path) -> None:
    """Свёртка прошлого прогона пишется минутами внутрь того каталога, который
    читатель видит. Оборванная, она не должна пережить публикацию под именем
    `previous`: следующий прогон отдал бы её как готовую."""
    _stage(tmp_path, "2026-08-01T00Z", value=280.0)
    first = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    (first / "previous.tmp" / "2t").mkdir(parents=True)

    _stage(tmp_path, "2026-08-01T12Z", value=290.0)
    publish_run(tmp_path, "2026-08-01T12Z", manifest=_manifest("2026-08-01T12Z"))

    assert not (first / "previous.tmp").exists()
    np.testing.assert_allclose(xr.open_zarr(tmp_path / "forecast" / "previous")["2t"].values, 280.0)


def test_publication_without_a_validation_report_is_refused(tmp_path: Path) -> None:
    """Приёмка 2.4: у артефакта оба файла. Отчёт пишет валидатор до публикации —
    иначе «проверено» означает «никто не смотрел»."""
    _stage(tmp_path, "2026-08-01T00Z", validated=False)
    with pytest.raises(ValueError, match=r"validation\.json"):
        publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    assert current_run(tmp_path) is None


def test_republishing_the_same_run_id_is_refused(tmp_path: Path) -> None:
    """Тот же ключ второй раз — это запись поверх опубликованного, то есть
    ровно то, от чего защищает публикация в новый ключ (docs/STORAGE.md §5)."""
    _stage(tmp_path, "2026-08-01T00Z")
    publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))

    _stage(tmp_path, "2026-08-01T00Z")
    with pytest.raises(FileExistsError, match="2026-08-01T00Z"):
        publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))


def test_second_publication_moves_current_and_fills_previous(tmp_path: Path) -> None:
    _stage(tmp_path, "2026-08-01T00Z", value=280.0)
    first = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    _stage(tmp_path, "2026-08-01T12Z", value=290.0)
    second = publish_run(tmp_path, "2026-08-01T12Z", manifest=_manifest("2026-08-01T12Z"))

    assert current_run(tmp_path) == second
    assert previous_run(tmp_path) == first / "previous"
    np.testing.assert_allclose(
        xr.open_zarr(tmp_path / "forecast" / "current" / "coarse")["2t"].values, 290.0
    )
    np.testing.assert_allclose(xr.open_zarr(tmp_path / "forecast" / "previous")["2t"].values, 280.0)


def test_previous_keeps_only_the_eight_variables(tmp_path: Path) -> None:
    """docs/STORAGE.md §2: прошлый прогон — 1.3 ГБ, а не 15.0. Указатель на
    старый `coarse` целиком удвоил бы шестичасовой слой в ядре."""
    _stage(tmp_path, "2026-08-01T00Z")
    publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    _stage(tmp_path, "2026-08-01T12Z")
    publish_run(tmp_path, "2026-08-01T12Z", manifest=_manifest("2026-08-01T12Z"))

    previous = xr.open_zarr(tmp_path / "forecast" / "previous")
    assert set(previous.data_vars) == set(canon.HOURLY_VARS)


def test_the_pointer_is_a_symlink_and_survives_repeated_switching(tmp_path: Path) -> None:
    """Указатель переставляется `replace`, а не `unlink` + `symlink`: между
    ними есть момент, когда `forecast/current` не существует вовсе."""
    link = tmp_path / "forecast" / "current"
    for run_id in ("2026-08-01T00Z", "2026-08-01T12Z", "2026-08-02T00Z"):
        _stage(tmp_path, run_id)
        publish_run(tmp_path, run_id, manifest=_manifest(run_id))
        assert link.is_symlink()
    assert current_run(tmp_path) == tmp_path / "runs" / "2026-08-02T00Z"
    assert not any(p.name.endswith(".tmp") for p in (tmp_path / "forecast").iterdir())


def test_a_real_directory_in_place_of_the_pointer_is_refused(tmp_path: Path) -> None:
    """Каталог вместо ссылки — след ручного вмешательства. Публиковать поверх
    него нельзя: `replace` каталог не заменит, а частичная публикация хуже
    отказа."""
    (tmp_path / "forecast" / "current").mkdir(parents=True)
    _stage(tmp_path, "2026-08-01T00Z")
    with pytest.raises(RuntimeError, match="current"):
        publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))


def test_the_first_publication_does_not_care_about_the_previous_pointer(tmp_path: Path) -> None:
    """Прошлого прогона нет — второй указатель не переставляется, и хлам на его
    месте к первой публикации отношения не имеет."""
    (tmp_path / "forecast" / "previous").mkdir(parents=True)
    _stage(tmp_path, "2026-08-01T00Z")
    final = publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    assert current_run(tmp_path) == final


def test_pointers_are_relative_so_the_store_can_be_moved(tmp_path: Path) -> None:
    """Хранилище переезжает вместе с диском; ссылка с абсолютным путём после
    переезда указывает в никуда."""
    _stage(tmp_path, "2026-08-01T00Z")
    publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    assert not os.path.isabs(os.readlink(tmp_path / "forecast" / "current"))


def test_manifest_of_a_run_that_was_never_published_stays_false(tmp_path: Path) -> None:
    """Разложенный, но не опубликованный прогон отличим от опубликованного по
    самому артефакту, а не только по указателю."""
    staged = _stage(tmp_path, "2026-08-01T00Z", layers=("coarse",))
    with pytest.raises(ValueError):
        publish_run(tmp_path, "2026-08-01T00Z", manifest=_manifest("2026-08-01T00Z"))
    assert not (staged / "manifest.json").exists()

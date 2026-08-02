"""Отметка о пропуске — приёмка BACKLOG 1.7.

Критерий: «пропуск прогона виден в манифесте, а не замаскирован». Виден он
двумя вещами, и обе проверяются здесь: документом на диске, который называет
причину, и журналом прогонов, который этот документ находит. Без второго первое
пишется в пустоту: `coverage` и `published_run` смотрят только на текущий
прогон и дырку показать не могут по построению.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from storage import read
from storage.manifest import (
    MANIFEST_NAME,
    SKIP_NAME,
    build_manifest,
    build_skip,
    mark_published,
    write_manifest,
    write_skip,
)
from storage.publish import RUNS_DIR
from tests.storage.test_manifest import INPUTS, MODEL, TIMINGS

MOMENT = datetime(2026, 8, 1, 9, 30, 0, tzinfo=UTC)


def _skip(**over: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "init_time": "2026-08-01T00:00:00Z",
        "reason": "no_input",
        "waited_for": ["ifs/0p25/oper", "gfs/0p25"],
        "deadline": "2026-08-01T09:30:00Z",
        "attempts": 5,
        "created_at": MOMENT,
    }
    kwargs.update(over)
    return build_skip("forecast/2026-08-01T00Z", **kwargs)  # type: ignore[arg-type]


def _run(root: Path, run_id: str) -> Path:
    run = root / RUNS_DIR / run_id
    run.mkdir(parents=True)
    return run


def test_a_skip_says_which_run_why_and_until_when_it_waited() -> None:
    """Три вопроса, на которые обязана отвечать отметка. «Не получилось» без
    них отвечает на тот вопрос, который и так был ясен."""
    record = _skip()

    assert record["artifact"] == "forecast/2026-08-01T00Z"
    assert record["init_time"] == "2026-08-01T00:00:00Z"
    assert record["reason"] == "no_input"
    assert record["waited_for"] == ["ifs/0p25/oper", "gfs/0p25"]
    assert record["deadline"] == "2026-08-01T09:30:00Z"
    assert record["attempts"] == 5
    assert record["created_at"] == "2026-08-01T09:30:00Z"


def test_a_skip_is_not_a_manifest_with_the_flag_down() -> None:
    """Ключ назван `skipped`, а не `published`. Совпади они, отметка прошла бы
    через `published_run` как прогон с `published: false`, то есть как
    оборванная публикация, — а это другая история и чинится иначе."""
    record = _skip()

    assert record["skipped"] is True
    assert "published" not in record
    assert "steps" not in record
    assert "inputs" not in record


def test_a_reason_outside_the_vocabulary_is_refused() -> None:
    """Набор причин закрыт: «источник молчит» лечится ожиданием, а «валидатор
    отверг срез» — чтением отчёта, и свободный текст стирает эту разницу."""
    with pytest.raises(ValueError, match="reason"):
        _skip(reason="whatever")


def test_a_skip_blamed_on_the_source_must_name_the_source() -> None:
    """`no_input` без потоков — это «данных нет», сказанное так, что непонятно,
    чьих. Дежурному дальше некуда идти."""
    with pytest.raises(ValueError, match="waited_for"):
        _skip(reason="no_input", waited_for=[])
    # Отвергнутый валидатором срез источники не обвиняет: там ждать нечего.
    assert _skip(reason="invalid_input", waited_for=[])["waited_for"] == []


def test_zero_attempts_is_a_legal_and_different_story() -> None:
    """Ноль отличает «источник молчал полтора часа» от «не пробовали вовсе,
    потому что воркер лежал»."""
    assert _skip(attempts=0)["attempts"] == 0
    with pytest.raises(ValueError, match="attempts"):
        _skip(attempts=-1)


def test_a_skip_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    path = write_skip(tmp_path / SKIP_NAME, _skip())

    assert json.loads(path.read_text(encoding="utf-8")) == _skip()
    with pytest.raises(ValueError, match="skipped"):
        write_skip(tmp_path / "other.json", {"artifact": "forecast/2026-08-01T00Z"})


def test_the_run_log_tells_a_skip_from_an_interrupted_publication(tmp_path: Path) -> None:
    """Три состояния, а не два. Прогона нет по двум разным причинам, и лечатся
    они по-разному: пропуск — решение расписания, незавершённый прогон —
    публикация, оборванная на середине."""
    good = _run(tmp_path, "2026-07-31T18Z")
    mark_published(
        write_manifest(
            good / MANIFEST_NAME,
            build_manifest(
                "forecast/2026-07-31T18Z",
                inputs=INPUTS,
                model=MODEL,
                steps=40,
                timings_sec=TIMINGS,
            ),
        )
    )
    torn = _run(tmp_path, "2026-08-01T06Z")
    write_manifest(
        torn / MANIFEST_NAME,
        build_manifest(
            "forecast/2026-08-01T06Z",
            inputs=INPUTS,
            model=MODEL,
            steps=40,
            timings_sec=TIMINGS,
        ),
    )
    write_skip(_run(tmp_path, "2026-08-01T00Z") / SKIP_NAME, _skip())

    log = read.run_log(tmp_path)

    assert [(entry.run_id, entry.state) for entry in log] == [
        ("2026-07-31T18Z", read.PUBLISHED),
        ("2026-08-01T00Z", read.SKIPPED),
        ("2026-08-01T06Z", read.UNFINISHED),
    ]
    assert log[1].reason == "no_input"
    assert log[1].init_time == "2026-08-01T00:00:00Z"
    # У прогона срок берётся по самому позднему входу: своего `init_time` в
    # манифесте нет, а у входа с переносом вперёд он чужой.
    assert log[0].init_time == "2026-08-01T00:00Z"
    assert log[0].reason is None


def test_a_storage_without_runs_has_an_empty_log(tmp_path: Path) -> None:
    """Пустой журнал — это «прогонов не было», а не отказ: свежее хранилище
    выглядит именно так, и падать на нём нечему."""
    assert read.run_log(tmp_path) == ()
    (tmp_path / RUNS_DIR).mkdir()
    assert read.run_log(tmp_path) == ()

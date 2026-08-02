"""Кто пишет отметку о пропуске — вторая половина приёмки BACKLOG 1.7.

Первая половина (`tests/storage/test_skip.py`) проверяет схему и журнал: если
`skip.json` на диске есть, он говорит всё, что нужно, и виден. Здесь
проверяется, что он там окажется. Схема без писателя — это критерий «пропуск
виден, а не замаскирован», выполненный на бумаге: пропущенный прогон писал бы
ровно ничего, как и до неё.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline import schedule
from pipeline.cycle import artifact, run_id, skip_if_missed
from storage.manifest import MANIFEST_NAME, SKIP_NAME
from storage.publish import run_path

MIDNIGHT = "2026-08-01T00:00:00Z"
RUN = schedule.cycle(MIDNIGHT)
#: Дедлайн публикации 00Z: `HH + 9:30` (docs/PIPELINE.md §2).
TOO_LATE = datetime(2026, 8, 1, 9, 30, tzinfo=UTC)


def test_the_run_directory_is_named_after_the_hour_it_forecasts() -> None:
    """`runs/<run_id>` и `forecast/<run_id>` обязаны совпадать до символа:
    иначе отметка ложится в каталог, который никто больше не называет."""
    assert run_id(MIDNIGHT) == "2026-08-01T00Z"
    assert artifact(MIDNIGHT) == "forecast/2026-08-01T00Z"
    assert run_id("2026-08-01T18:00:00Z") == "2026-08-01T18Z"


def test_a_run_id_from_a_time_in_another_format_is_refused() -> None:
    with pytest.raises(schedule.ScheduleError, match="init_time"):
        run_id("2026-08-01 00:00")


def test_a_run_id_for_an_hour_that_has_no_run_is_refused() -> None:
    """`03Z` разбирается форматной строкой без ошибок и даёт каталог
    `runs/2026-08-01T03Z`, которого не ждёт ни один другой компонент. Имя
    каталога — то самое место, где неверный час остаётся навсегда."""
    with pytest.raises(schedule.ScheduleError, match="init_time"):
        run_id("2026-08-01T03:00:00Z")


def test_a_missed_run_leaves_a_document_behind(tmp_path: Path) -> None:
    mark = skip_if_missed(
        tmp_path,
        RUN,
        now=TOO_LATE,
        reason="no_input",
        waited_for=["ifs/0p25/oper", "gfs/0p25"],
        attempts=5,
    )

    assert mark == run_path(tmp_path, "2026-08-01T00Z") / SKIP_NAME
    assert json.loads(mark.read_text(encoding="utf-8")) == {
        "artifact": "forecast/2026-08-01T00Z",
        "created_at": "2026-08-01T09:30:00Z",
        "init_time": MIDNIGHT,
        "reason": "no_input",
        "waited_for": ["ifs/0p25/oper", "gfs/0p25"],
        # Срок из расписания, а не с потолка: ждали до дедлайна публикации.
        "deadline": "2026-08-01T09:30:00Z",
        "attempts": 5,
        "skipped": True,
    }


def test_a_run_that_can_still_be_saved_is_not_declared_dead(tmp_path: Path) -> None:
    """До `HH + 9:30` прогон ещё спасается откатом на GFS. Отметка, записанная
    раньше, — ложная тревога, которую дежурный увидит вместо настоящей."""
    early = datetime(2026, 8, 1, 8, 45, tzinfo=UTC)
    assert schedule.phase(RUN, early) == schedule.FALLBACK

    assert (
        skip_if_missed(tmp_path, RUN, now=early, reason="no_input", waited_for=["gfs/0p25"]) is None
    )
    assert not run_path(tmp_path, "2026-08-01T00Z").exists()


def test_a_run_that_happened_is_never_marked_skipped(tmp_path: Path) -> None:
    """Манифест в каталоге — прогон состоялся или оборвался на середине. И то и
    другое видно журналом, и «пропущен» про них неправда."""
    directory = run_path(tmp_path, "2026-08-01T00Z")
    directory.mkdir(parents=True)
    (directory / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    assert skip_if_missed(tmp_path, RUN, now=TOO_LATE, reason="no_forecast") is None
    assert not (directory / SKIP_NAME).exists()


def test_the_second_call_keeps_the_first_time_of_death(tmp_path: Path) -> None:
    """У прогона в фазе `missed` попыток больше не будет, и второе `created_at`
    сдвинуло бы время отказа вперёд — то самое число, по которому потом считают,
    сколько сервис молчал."""
    first = skip_if_missed(
        tmp_path, RUN, now=TOO_LATE, reason="no_input", waited_for=["gfs/0p25"], attempts=5
    )
    assert first is not None
    written = first.read_text(encoding="utf-8")

    again = skip_if_missed(
        tmp_path,
        RUN,
        now=datetime(2026, 8, 1, 11, 0, tzinfo=UTC),
        reason="no_input",
        waited_for=["gfs/0p25"],
        attempts=9,
    )

    assert again == first
    assert first.read_text(encoding="utf-8") == written


def test_a_skip_blamed_on_the_source_still_has_to_name_it(tmp_path: Path) -> None:
    """Проверки схемы конвейер не обходит: `build_skip` вызывается как есть."""
    with pytest.raises(ValueError, match="waited_for"):
        skip_if_missed(tmp_path, RUN, now=TOO_LATE, reason="no_input")

"""Расписание цикла и ретраи — приёмка BACKLOG 1.7, часть про время.

Вторая часть приёмки («пропуск прогона виден в манифесте, а не замаскирован»)
проверяется в `tests/storage/test_manifest.py`: отметка о пропуске — документ на
диске, а не арифметика.

Сети тут нет и быть не может: расписание — это функция от `init_time` и `now`,
и проверяется оно подстановкой момента, а не ожиданием.
"""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from pipeline import schedule

NOON = "2026-08-01T12:00:00Z"


def _at(stamp: str) -> datetime:
    return datetime.strptime(stamp, schedule.TIME_FORMAT).replace(tzinfo=UTC)


def test_the_four_moments_of_a_cycle_come_from_the_run_time() -> None:
    """Расписание из docs/PIPELINE.md §2, поле в поле."""
    run = schedule.cycle(NOON)

    assert run.poll_from == _at("2026-08-01T19:00:00Z")
    assert run.fallback_at == _at("2026-08-01T20:30:00Z")
    assert run.publish_by == _at("2026-08-01T21:30:00Z")


def test_a_run_that_does_not_exist_has_no_schedule() -> None:
    """Прогонов четыре в сутки. Посчитать дедлайны прогону на 03Z значит завести
    цикл, которого никто не запускает и чей пропуск поэтому никогда не всплывёт."""
    with pytest.raises(schedule.ScheduleError, match="init_time"):
        schedule.cycle("2026-08-01T03:00:00Z")
    with pytest.raises(schedule.ScheduleError, match="init_time"):
        schedule.cycle("2026-08-01 12:00")


def test_the_phase_changes_exactly_on_the_deadline() -> None:
    """Граница принадлежит следующей фазе. Иначе последний опрос и откат
    назначены на одну секунду, и порядок решает планировщик, а не расписание."""
    run = schedule.cycle(NOON)

    assert schedule.phase(run, run.poll_from - timedelta(seconds=1)) == schedule.WAIT
    assert schedule.phase(run, run.poll_from) == schedule.POLL
    assert schedule.phase(run, run.fallback_at - timedelta(seconds=1)) == schedule.POLL
    assert schedule.phase(run, run.fallback_at) == schedule.FALLBACK
    assert schedule.phase(run, run.publish_by) == schedule.MISSED


def test_a_naive_moment_is_read_as_utc() -> None:
    """Умолчание Python — локальная зона, и на сервере в московской она сдвинула
    бы все четыре дедлайна на три часа. Заметно это стало бы по пропущенным
    прогонам, а не по ошибке."""
    run = schedule.cycle(NOON)
    naive = datetime(2026, 8, 1, 19, 0, 0)

    assert schedule.phase(run, naive) == schedule.POLL


def test_polling_stops_at_the_fallback_deadline() -> None:
    """Полтора часа по пять минут — восемнадцать опросов, и последний строго до
    дедлайна. Опрос «пока не появится» на источнике, который сегодня может не
    появиться вовсе, держит прогон до следующего прогона."""
    times = schedule.poll_times(schedule.cycle(NOON))

    assert len(times) == 18
    assert times[0] == _at("2026-08-01T19:00:00Z")
    assert times[-1] == _at("2026-08-01T20:25:00Z")
    assert all(later - earlier == schedule.POLL_EVERY for earlier, later in pairwise(times))


def test_the_pause_between_attempts_grows_and_then_stops_growing() -> None:
    """Экспонента без потолка даёт шестую паузу за полчаса — она одна съедает
    дедлайн отката, не добавив ни одной попытки."""
    assert schedule.delays(5, base_sec=15.0, cap_sec=300.0) == (15.0, 30.0, 60.0, 120.0)
    assert schedule.delays(8, base_sec=15.0, cap_sec=100.0) == (
        15.0,
        30.0,
        60.0,
        100.0,
        100.0,
        100.0,
        100.0,
    )


def test_the_last_attempt_is_not_followed_by_a_pause() -> None:
    """Пауз на одну меньше, чем попыток: после последней ждать нечего."""
    assert len(schedule.delays(schedule.MAX_ATTEMPTS)) == schedule.MAX_ATTEMPTS - 1
    assert schedule.delays(1) == ()
    with pytest.raises(schedule.ScheduleError, match="attempts"):
        schedule.delays(0)


def test_the_run_that_should_be_on_disk_is_not_the_last_hour_that_passed() -> None:
    """В 13:00 UTC срок 12Z прошёл, но в Open Data его ещё нет: анализ приезжает
    через семь часов с лишним (§1). Требовать его — считать нормальную задержку
    источника пропуском."""
    assert schedule.latest_init_time(_at("2026-08-01T13:00:00Z")) == "2026-08-01T06:00:00Z"
    assert schedule.latest_init_time(_at("2026-08-01T19:00:00Z")) == "2026-08-01T12:00:00Z"
    # Через полночь окно уезжает во вчера, а не обнуляется.
    assert schedule.latest_init_time(_at("2026-08-02T02:00:00Z")) == "2026-08-01T18:00:00Z"

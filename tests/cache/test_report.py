"""Отчёт по кэшу — приёмка BACKLOG 3.5.

Критерий дословно: «отчёт показывает hit rate и топ запрашиваемых точек».
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cache.index import Key, default_index, log_query, open_index, record
from cache.report import DAY_S, HOUR_S, collect, hit_rate, main, miss_latency, render, top_spots

NOW = 1_800_000_000.0


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _ask(
    index: sqlite3.Connection,
    *,
    hit: bool,
    area: str = "55.75,37.62",
    endpoint: str = "/v1/history/point",
    latency_ms: int = 50,
    age_s: float = 0.0,
) -> None:
    log_query(
        index,
        endpoint=endpoint,
        cache_hit=hit,
        latency_ms=latency_ms,
        area=area,
        now=NOW - age_s,
    )


def test_the_report_shows_the_hit_rate_and_the_top_points(
    index: sqlite3.Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    """Сам критерий приёмки, но через `render`: `main` берёт `time.time()`, и
    привязывать проверку к настоящим часам — значит проверять часы."""
    for _ in range(3):
        _ask(index, hit=True, area="55.75,37.62")
    _ask(index, hit=False, area="55.75,37.62")
    _ask(index, hit=True, area="59.94,30.31")

    text = render(collect(index, now=NOW))

    assert "hit rate за сутки: 80% (4 из 5)" in text
    assert "55.75,37.62: 4 запросов" in text
    assert "59.94,30.31: 1 запросов" in text


def test_the_hour_window_does_not_hide_a_collapse(index: sqlite3.Connection) -> None:
    """Одно суточное окно прячет обвал после выкладки: сотня свежих промахов
    растворяется во вчерашних попаданиях."""
    for _ in range(100):
        _ask(index, hit=True, age_s=HOUR_S * 5)
    for _ in range(10):
        _ask(index, hit=False, age_s=60)

    report = collect(index, now=NOW)

    assert report.day.rate == pytest.approx(100 / 110)
    assert report.hour == (10, 0)
    assert report.hour.poor is True


def test_points_and_maps_are_counted_apart(index: sqlite3.Connection) -> None:
    """За ними разные источники: ряд в точке идёт из очереди CDS и стоит минуты,
    карта — из ARCO и стоит ~200 мс. Общий hit rate — среднее между «терпимо» и
    «пользователь ушёл»."""
    for _ in range(4):
        _ask(index, hit=False, endpoint="/v1/history/point")
    for _ in range(4):
        _ask(index, hit=True, endpoint="/v1/history/grid")

    report = collect(index, now=NOW)

    assert (report.point.rate, report.grid.rate) == (0.0, 1.0)
    assert report.day == (8, 4)


def test_only_misses_count_towards_the_latency(index: sqlite3.Connection) -> None:
    """Попадание с диска — не то, ради чего смотрят на перцентиль."""
    for _ in range(19):
        _ask(index, hit=False, latency_ms=100)
    _ask(index, hit=False, latency_ms=9000)
    for _ in range(50):
        _ask(index, hit=True, latency_ms=1)

    measured = miss_latency(index, since=NOW - DAY_S)

    assert measured.misses == 20
    # Ближайший ранг, а не интерполяция: девятнадцатое значение из двадцати.
    assert measured.p95_ms == 9000
    assert measured.mean_ms == pytest.approx((19 * 100 + 9000) / 20)


def test_an_empty_log_is_not_a_broken_cache(index: sqlite3.Connection) -> None:
    """Ноль процентов на пустом журнале выглядит как отказ кэша, а это отсутствие
    трафика. Делить на ноль тут тоже нечего."""
    report = collect(index, now=NOW)

    assert (report.day.rate, report.day.poor) == (0.0, False)
    assert report.miss == (0, 0.0, 0)
    assert "нет запросов" in render(report)
    assert "журнал за сутки пуст" in render(report)


def test_what_is_older_than_a_day_is_out_of_the_report(index: sqlite3.Connection) -> None:
    """Окно — это окно. Позавчерашние запросы к сегодняшнему кэшу отношения не
    имеют: между ними была и выкладка, и чистка."""
    for _ in range(5):
        _ask(index, hit=True, age_s=DAY_S + 60)

    report = collect(index, now=NOW)

    assert report.day == (0, 0)
    assert top_spots(index, since=NOW - DAY_S) == ()


def test_the_report_counts_what_lies_on_disk(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Заполненность берётся из индекса, а не со `stat`: по этим же байтам
    решает вытеснение, и расхождение между отчётом и чисткой скрывало бы, почему
    место не находится."""
    record(
        index,
        Key("era5", "2t", "era5-final", "v1", "1"),
        path=str(tmp_path / "1.chunk"),
        size=3 * 2**30,
        origin="arco",
        cost_ms=200,
    )
    record(
        index,
        Key("era5", "2t", "era5-final", "v1", "2"),
        path=str(tmp_path / "2.chunk"),
        size=2**30,
        origin="arco",
        cost_ms=200,
        pinned=True,
    )

    report = collect(index, now=NOW)

    assert (report.objects, report.occupied_bytes) == (2, 4 * 2**30)
    assert report.pinned_bytes == 2**30
    assert "занято 4.00 ГБ в 2 объектах, из них запиннено 1.00 ГБ" in render(report)


def test_a_poor_hit_rate_is_said_out_loud(index: sqlite3.Connection) -> None:
    """Ниже порога кэш не выполняет свою работу, и причина одна из двух —
    чинятся они по-разному (docs/CACHE.md §6)."""
    for _ in range(4):
        _ask(index, hit=False)
    _ask(index, hit=True)

    text = render(collect(index, now=NOW))

    assert "либо кэшируются запросы вместо чанков" in text


def test_the_top_is_limited(index: sqlite3.Connection) -> None:
    """Топ читает человек: сотня строк — это не отчёт, а дамп журнала."""
    for number in range(30):
        for _ in range(number + 1):
            _ask(index, hit=True, area=f"point-{number}")

    top = top_spots(index, since=NOW - DAY_S, limit=3)

    assert [spot.area for spot in top] == ["point-29", "point-28", "point-27"]
    assert top[0].asked == 30


def test_a_query_without_an_area_does_not_make_a_row(index: sqlite3.Connection) -> None:
    """`/v1/meta/coverage` и `/v1/health` площади не имеют. Пустая строка в топе
    самых спрашиваемых точек — это не точка."""
    log_query(index, endpoint="/v1/meta/coverage", cache_hit=True, latency_ms=1, now=NOW)
    _ask(index, hit=True, area="55.75,37.62")

    assert [spot.area for spot in top_spots(index, since=NOW - DAY_S)] == ["55.75,37.62"]
    assert hit_rate(index, since=NOW - DAY_S) == (2, 2)


def test_the_command_says_which_index_it_did_not_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Пустая схема, разложенная по неверному пути, отчиталась бы «занято 0
    байт» — то есть спрятала бы незаданную переменную за исправным видом."""
    code = main([str(tmp_path / "нет.sqlite")])

    assert code == 1
    assert "индекса нет" in capsys.readouterr().out
    assert not (tmp_path / "нет.sqlite").exists()


def test_the_command_reads_the_index_it_was_given(
    index: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`make cache-report` без аргументов идёт в `$AURORA_ROOT`, но дежурный
    смотрит и чужие копии индекса."""
    _ask(index, hit=True, age_s=1.0)
    index.close()

    code = main([str(tmp_path / "cache.sqlite")])

    assert code == 0
    assert "hit rate за сутки" in capsys.readouterr().out


def test_the_default_index_follows_the_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один корень на все данные: индекс, разъехавшийся с хранилищем, — это
    отчёт про кэш, которого нет."""
    monkeypatch.setenv("AURORA_ROOT", "/tmp/aurora-корень")

    assert default_index() == Path("/tmp/aurora-корень/cache/index.sqlite")

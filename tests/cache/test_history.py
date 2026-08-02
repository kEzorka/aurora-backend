"""История поверх кэша: склейка, обрезка, окно и честный `hit` (BACKLOG 5.5).

Критерий приёмки дословно: «холодный запрос отрабатывает, `cache.hit` в ответе
честный». Поэтому здесь считаются походы наружу, а не время ответа, и оба
источника берутся настоящие — `CdsOrigin` поверх подставной очереди и
`ArcoOrigin` поверх того же архива, каким проверялся адаптер. Склейка, которая
работает на выдуманных чанках, но не на этих, — это проверка кодека, а не
истории.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from adapters.era5_arco import FINAL, FINAL_AFTER, PRELIMINARY
from adapters.era5_cds import build_request
from cache.history import grid_window, point_series
from cache.index import open_index
from cache.origins import CDS_GRID, ArcoOrigin, CdsOrigin
from cache.proxy import Grid
from tests.fixtures.arco import STEPS, build
from tests.fixtures.cds import MOSCOW, Service

#: «Сейчас» для рядов: далеко позади и фикстуры ARCO, и дат CDS ниже, поэтому
#: версия источника всюду финальная и не мешает смотреть на склейку.
NOW = datetime(2026, 3, 1, tzinfo=UTC)

#: Сутки данных в чанке вместо года: длина чанка на склейку не влияет, а год
#: почасовых строк — это 8760 строк CSV в каждом тесте.
DAY_GRID = Grid(epoch=CDS_GRID.epoch, step=timedelta(hours=1), span=24, max_chunks=40)

#: Двое суток, попадающие в разные чанки: на одном чанке обрезка периода
#: неотличима от её отсутствия.
FIRST = datetime(1990, 5, 1, tzinfo=UTC)
SECOND = FIRST + timedelta(days=1)

#: Секунды жизни записей индекса — к календарю данных отношения не имеют.
CLOCK = 1_800_000_000.0


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("arco") / "sfc.zarr", "2m_temperature")


def _series_origin(service: Service, *, now: datetime = NOW) -> CdsOrigin:
    return CdsOrigin(grid=DAY_GRID, retriever=service, clock=lambda: now)


def test_the_period_is_cut_out_of_the_chunks(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Наружу просили период, а с диска приехали чанки, выровненные по границам
    источника. Без обрезки запрос «с вечера до утра» вернул бы двое суток — и
    тем больше лишнего, чем длиннее чанк: у настоящей нарезки CDS он годовой.
    """
    service = Service()
    series = point_series(
        index,
        _series_origin(service),
        ["2t"],
        *MOSCOW,
        FIRST + timedelta(hours=18),
        SECOND + timedelta(hours=5),
        root=tmp_path / "cache",
    )

    assert len(service.requests) == 2  # период лёг на два чанка
    assert series.times[0] == "1990-05-01T18:00:00Z"
    assert series.times[-1] == "1990-05-02T05:00:00Z"
    assert len(series.times) == 12
    # Значение равно 250 + час от начала чанка: видно, что взят хвост первых
    # суток и начало вторых, а не первые двенадцать часов подряд.
    assert series.values["2t"][0] == pytest.approx(268.0)
    assert series.values["2t"][6] == pytest.approx(250.0)
    assert (series.lat, series.lon) == MOSCOW
    assert series.source == FINAL and not series.preliminary


def test_the_second_request_does_not_reach_the_queue(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """`cache.hit` в ответе честный — это и есть критерий приёмки 5.5. Честный
    он тогда, когда за ним не стоит похода в очередь CDS, общую на весь сервис.
    """
    service = Service()
    origin = _series_origin(service)
    root = tmp_path / "cache"
    span = (FIRST, FIRST + timedelta(hours=5))

    cold = point_series(index, origin, ["2t"], *MOSCOW, *span, root=root)
    warm = point_series(index, origin, ["2t"], *MOSCOW, *span, root=root)

    assert (cold.hit, warm.hit) == (False, True)
    assert warm.origin_latency_ms == 0
    assert len(service.requests) == 1
    assert warm.values == cold.values and warm.times == cold.times


def test_fields_are_joined_by_time_and_not_by_length(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Растущий чанк, взятый для `10u` утром, а для `10v` в полдень, короче на
    несколько часов. Сложенные по длине, они соединили бы составляющие ветра за
    разные сроки — и `wind` наверху получился бы из разных моментов времени.
    """
    late = build_request("10v", *MOSCOW, FIRST, FIRST)["variable"][0]
    service = Service(available={late: FIRST + timedelta(hours=17)})

    series = point_series(
        index,
        _series_origin(service),
        ["10u", "10v"],
        *MOSCOW,
        FIRST,
        FIRST + timedelta(hours=23),
        root=tmp_path / "cache",
    )

    assert len(series.times) == 24  # ось общая, по более длинному полю
    assert series.values["10u"][23] == pytest.approx(273.0)
    assert series.values["10v"][17] == pytest.approx(267.0)
    # Недостающие часы — пропуск, а не сдвиг: `null` в JSON (§1).
    assert series.values["10v"][18:] == [None] * 6


def test_a_map_window_is_averaged_over_the_period(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Окно карты: срез по рамке, прореживание и среднее по сроку.

    Значение в фикстуре равно номеру узла по долготе плюс тысяча за каждый
    срок, поэтому среднее по трём срокам считается в уме: `узел + 1000`.
    Геометрия проверяется вместе со значениями — рамка, съехавшая на узел,
    иначе не отличима от правильной.
    """
    window = grid_window(
        index,
        ArcoOrigin(archive, clock=lambda: datetime(2020, 12, 1, tzinfo=UTC)),
        ["2t"],
        (50.0, 0.0, 51.0, 2.0),
        STEPS[0],
        STEPS[-1],
        root=tmp_path / "cache",
        stride=2,
    )

    assert window.shape == (3, 5)  # 5 узлов по широте и 9 по долготе через один
    assert (window.lat0, window.dlat) == (51.0, -0.5)
    assert (window.lon0, window.dlon) == (0.0, 0.5)
    assert (window.first, window.last, window.steps) == (
        "2020-06-01T00:00:00Z",
        "2020-06-01T02:00:00Z",
        3,
    )
    assert len(window.values["2t"]) == 15
    assert window.values["2t"][0] == pytest.approx(1000.0)
    assert window.values["2t"][1] == pytest.approx(1002.0)  # +0.5° долготы — это два узла
    assert window.source == FINAL and not window.hit


def test_one_preliminary_hour_makes_the_whole_answer_preliminary(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Слабейшее звено решает (docs/API_CONTRACT.md §5.3): «значения могут
    измениться задним числом» верно для всего ответа, если верно хоть для
    одного его срока. Момент подобран так, что предварительный ровно один.
    """
    origin = ArcoOrigin(archive, clock=lambda: STEPS[1] + FINAL_AFTER + timedelta(seconds=1))

    window = grid_window(
        index,
        origin,
        ["2t"],
        (50.0, 0.0, 51.0, 2.0),
        STEPS[0],
        STEPS[-1],
        root=tmp_path / "cache",
    )

    versions = [origin.source_version(origin.grid.holding(step)) for step in STEPS]
    assert versions == [FINAL, FINAL, PRELIMINARY]
    assert window.source == PRELIMINARY and window.preliminary


def test_a_frame_between_the_nodes_is_an_error(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Рамка уже шага сетки отдаёт пустой срез, а не отказ: `xarray` считает
    это законным ответом. Наружу такое обязано ехать ошибкой — карта из нуля
    узлов неотличима от «данных нет», и `200` с пустым списком отправил бы
    клиента искать причину не там."""
    with pytest.raises(LookupError):
        grid_window(
            index,
            ArcoOrigin(archive, clock=lambda: datetime(2020, 12, 1, tzinfo=UTC)),
            ["2t"],
            (50.05, 0.05, 50.10, 0.10),
            STEPS[0],
            STEPS[0],
            root=tmp_path / "cache",
        )


def test_a_request_without_fields_is_refused_before_the_index(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Пустой список полей — ошибка вызывающего, а не пустой ответ: иначе
    `Series` без значений доедет до API и станет `200` с пустым телом."""
    origin = _series_origin(Service())

    with pytest.raises(ValueError):
        point_series(index, origin, [], *MOSCOW, FIRST, FIRST, root=tmp_path / "cache")
    with pytest.raises(ValueError):
        grid_window(
            index, origin, [], (50.0, 0.0, 51.0, 2.0), FIRST, FIRST, root=tmp_path / "cache"
        )

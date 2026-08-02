"""Ряды CDS через кэш: нарезка, версия источника и точка в ключе (BACKLOG 5.5).

Адаптер проверен отдельно (`tests/adapters/test_era5_cds.py`) — здесь
проверяется то, чего он про себя не знает: как ряд нарезается на чанки, что
попадает в ключ и когда поход в очередь CDS не случается вовсе.

Ретривер подставной: в сеть тесты не ходят. Нарезка при этом настоящая — стуб
отвечает ровно на тот период, который у него спросили, поэтому обрезка чанка
по «сейчас» и отказ на пустом ответе проверяются на том же пути, каким поедет
боевой запрос.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from adapters.era5_arco import FINAL, FINAL_AFTER, PRELIMINARY
from cache.index import Key, absent, open_index
from cache.origins import CDS_GRID, CdsOrigin, at_point, split_point
from cache.proxy import Grid, NotYetError, serve

#: Москва, уже округлённая до узла сетки: тест сравнивает с числом, а не
#: повторяет в себе `snap`.
MOSCOW = (55.75, 37.5)

#: Нарезка на сутки вместо года. Мельче настоящей нарочно: год почасовых строк
#: на чанк — это 8760 строк CSV в каждом тесте ради свойства, которое от длины
#: чанка не зависит. Настоящая нарезка проверяется там, где речь именно о ней.
DAY_GRID = Grid(epoch=CDS_GRID.epoch, step=timedelta(hours=1), span=24, max_chunks=40)

#: Секунды жизни индекса — к календарю данных отношения не имеют.
CLOCK = 1_800_000_000.0


class Service:
    """Очередь CDS: отдаёт почасовые строки за спрошенный период.

    Значение равно числу часов от начала суток запроса — по нему видно, какой
    именно кусок ряда доехал до ответа, а какой обрезала обрезка периода.
    """

    def __init__(self, *, available: datetime | None = None) -> None:
        self.available = available
        self.requests: list[Mapping[str, Any]] = []

    def __call__(self, dataset: str, request: Mapping[str, Any]) -> str:
        self.requests.append(request)
        first, _, last = str(request["date"][0]).partition("/")
        start = datetime.fromisoformat(first).replace(tzinfo=UTC)
        end = datetime.fromisoformat(last).replace(hour=23, tzinfo=UTC)
        if self.available is not None:
            end = min(end, self.available)
        name = str(request["variable"][0])
        lines = [f"valid_time,latitude,longitude,{name}"]
        moment, hour = start, 0
        while moment <= end:
            lines.append(f"{moment:%Y-%m-%d %H:%M:%S},{MOSCOW[0]},{MOSCOW[1]},{250.0 + hour}")
            moment, hour = moment + timedelta(hours=1), hour + 1
        return "\n".join(lines[: 1 if len(lines) == 1 else None]) + "\n"


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _origin(service: Service | None = None, *, now: datetime, grid: Grid = DAY_GRID) -> CdsOrigin:
    return CdsOrigin(grid=grid, retriever=service or Service(), clock=lambda: now)


def test_the_point_rides_inside_the_variable_name() -> None:
    """У `cache.index.Key` поля под точку нет, и заводить его нельзя: у карт
    точки не бывает вовсе. Точка едет в имени переменной — а `dataset` при этом
    остаётся один на все точки, иначе отчёт по кэшу превращается в список
    координат, которые кто-то когда-то спросил."""
    name = at_point("2t", 55.76, 37.61)

    assert name == "2t@55.75,37.5"  # ближайший узел, а не то, что прислали
    assert split_point(name) == ("2t", 55.75, 37.5)
    assert CdsOrigin().dataset == "era5-series"


def test_the_same_place_written_two_ways_is_one_key() -> None:
    """359.5 и -0.5 — одно место, и минус у нуля — тоже: два ключа на один
    узел означают два похода в очередь CDS за одними и теми же данными."""
    assert at_point("2t", 0.0, 359.5) == at_point("2t", 0.0, -0.5)
    assert at_point("2t", -0.0, -0.0) == "2t@0,0"


def test_the_version_is_taken_at_the_end_of_the_chunk() -> None:
    """Шов, которого нет у ARCO: там чанк равен сроку, а здесь — году, и
    версия по началу чанка обещала бы финальный ERA5 на 364 днях из 365.

    Момент подобран так, что начало чанка старше трёх месяцев, а конец — нет.
    """
    chunk = CDS_GRID.holding(datetime(2025, 6, 1, tzinfo=UTC))
    ends = CDS_GRID.begins(chunk + 1) - CDS_GRID.step
    now = ends + timedelta(days=10)
    origin = _origin(now=now, grid=CDS_GRID)

    assert now - CDS_GRID.begins(chunk) > FINAL_AFTER  # по началу вышло бы `era5-final`
    assert origin.source_version(chunk) == PRELIMINARY


def test_a_decade_keeps_the_blast_radius_at_one_chunk() -> None:
    """То, ради чего чанк длиной в год (заметка в `era5_cds.read_series`).

    Метка `era5t` идёт на весь чанк, а инвалидация выбрасывает помеченное
    целиком. При чанке в год замена предварительных данных финальными стоит
    одного чанка из десяти лет ряда, а не всего ряда.
    """
    latest = CDS_GRID.holding(datetime(2025, 1, 1, tzinfo=UTC))
    now = CDS_GRID.begins(latest) + timedelta(days=200)
    origin = _origin(now=now, grid=CDS_GRID)

    versions = [origin.source_version(chunk) for chunk in range(latest - 9, latest + 1)]

    assert sum(version.startswith(PRELIMINARY) for version in versions) == 1
    assert versions[:-1] == [FINAL] * 9


def test_a_growing_chunk_is_a_new_object_each_day() -> None:
    """Пока конец чанка в будущем, источник каждый день отдаёт на сутки больше.
    Ключ без даты означал бы навсегда замороженный обрубок года."""
    chunk = CDS_GRID.holding(datetime(2026, 3, 1, tzinfo=UTC))
    today = CDS_GRID.begins(chunk) + timedelta(days=30)

    first = _origin(now=today, grid=CDS_GRID).source_version(chunk)
    second = _origin(now=today + timedelta(days=1), grid=CDS_GRID).source_version(chunk)

    assert first.startswith(PRELIMINARY) and first != second
    assert "/" not in first  # разделитель частей ключа (`cache.index.SEPARATOR`)


def test_the_chunk_is_cut_at_now_before_the_queue(index: sqlite3.Connection) -> None:
    """Заявка на будущие даты — это отказ CDS через очередь длиной в минуты, а
    не пустой ответ. Обрезка стоит до похода, а не после."""
    day = datetime(2026, 3, 1, tzinfo=UTC)
    now = day + timedelta(hours=10)
    service = Service(available=now)
    origin = _origin(service, now=now)

    origin.fetch(at_point("2t", *MOSCOW), DAY_GRID.holding(day))

    (request,) = service.requests
    assert request["date"] == ["2026-03-01/2026-03-01"]
    assert request["location"] == {"latitude": MOSCOW[0], "longitude": MOSCOW[1]}


def test_a_day_that_has_not_happened_is_refused_without_asking(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Чанк целиком в будущем наружу не идёт, а отказ запоминается на шесть
    часов: человек, листающий календарь вперёд, иначе устраивает поход в общую
    очередь CDS на каждое движение."""
    now = datetime(2026, 3, 1, tzinfo=UTC)
    ahead = now + timedelta(days=2)
    service = Service()
    origin = _origin(service, now=now)
    variable = at_point("2t", *MOSCOW)

    with pytest.raises(NotYetError):
        serve(index, origin, variable, ahead, ahead, root=tmp_path / "cache", now=CLOCK)

    chunk = DAY_GRID.holding(ahead)
    key = Key(origin.dataset, variable, origin.source_version(chunk), origin.version, str(chunk))
    assert service.requests == []
    assert absent(index, key, now=CLOCK) is not None


def test_an_empty_answer_is_a_gap_and_not_a_failure(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Реанализ отстаёт от календаря на ~5 суток, и чанк, начавшийся внутри
    этого отставания, приезжает пустым. Это `404` с отрицательным кэшем, а не
    `503`: источник не упал, ему просто нечего сказать про вчера."""
    day = datetime(2026, 3, 1, tzinfo=UTC)
    now = day + timedelta(hours=5)
    # Данные кончились до начала суток запроса — строк не будет ни одной.
    origin = _origin(Service(available=day - timedelta(hours=1)), now=now)

    with pytest.raises(NotYetError):
        serve(
            index,
            origin,
            at_point("2t", *MOSCOW),
            day,
            day + timedelta(hours=4),
            root=tmp_path / "cache",
            now=CLOCK,
        )

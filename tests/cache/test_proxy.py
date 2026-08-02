"""Прокси к origin — приёмка BACKLOG 3.2.

Критерий дословно: «повторный запрос того же периода — попадание в кэш».
Считаются обращения к origin, а не время ответа: «стало быстрее» — это не
проверка, а впечатление, и на подставном источнике оно вообще ничего не значит.

Наружу тесты не ходят: origin здесь подставной и умеет одно — считать, сколько
раз его дёрнули, и падать по команде.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cache.index import Key, entries, lookup, open_index, total_bytes
from cache.proxy import Grid, Served, TooManyChunksError, align, cold, moment, serve

#: Сетка ARCO: чанк равен шагу времени (docs/CACHE.md §1). Предел маленький
#: намеренно — на нём и проверяется отказ.
MAPS = Grid(epoch=datetime(1940, 1, 1, tzinfo=UTC), step=timedelta(hours=1), span=6, max_chunks=8)

DAY = "1990-06-01T00:00:00Z"


class FakeOrigin:
    """Источник, который считает походы к себе.

    `broken` — номер чанка, на котором источник падает: очередь CDS отдаёт
    отказы штатно, и половина выкачанного не должна остаться на диске.
    """

    name = "arco"
    dataset = "era5"
    version = "v1"
    grid = MAPS

    def __init__(self, *, broken: int | None = None, since: int | None = None) -> None:
        self.calls: list[int] = []
        self.broken = broken
        # Номер чанка, начиная с которого данные ещё предварительные.
        self.since = since

    def source_version(self, chunk: int) -> str:
        return "era5t" if self.since is not None and chunk >= self.since else "era5-final"

    def fetch(self, variable: str, chunk: int) -> bytes:
        self.calls.append(chunk)
        if chunk == self.broken:
            raise TimeoutError("очередь CDS не ответила")
        return f"{variable}:{chunk}".encode()


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _serve(
    index: sqlite3.Connection,
    origin: FakeOrigin,
    root: Path,
    start: str,
    stop: str,
    *,
    now: float = 1_800_000_000.0,
) -> Served:
    return serve(index, origin, "2t", moment(start), moment(stop), root=root, now=now)


def test_the_same_period_asked_twice_comes_from_disk(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Сам критерий приёмки."""
    origin = FakeOrigin()

    first = _serve(index, origin, tmp_path, DAY, "1990-06-01T12:00:00Z")
    again = _serve(index, origin, tmp_path, DAY, "1990-06-01T12:00:00Z")

    assert first.misses == 3 and first.hits == 0
    assert again.hits == 3 and again.misses == 0
    assert again.hit is True
    # Три похода на два запроса: второй раз наружу не ходили вовсе.
    assert origin.calls == [73656, 73657, 73658]


def test_an_overlapping_period_only_pulls_what_is_new(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Ради этого кэшируется чанк, а не запрос (docs/CACHE.md §3.1): два
    пересекающихся периода делят общее, а не кладут его дважды."""
    origin = FakeOrigin()
    _serve(index, origin, tmp_path, DAY, "1990-06-01T12:00:00Z")

    later = _serve(index, origin, tmp_path, "1990-06-01T06:00:00Z", "1990-06-01T18:00:00Z")

    assert (later.hits, later.misses) == (2, 1)
    assert origin.calls == [73656, 73657, 73658, 73659]
    assert len(list(entries(index))) == 4


def test_a_request_inside_one_chunk_pulls_the_whole_chunk(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Просим шире нужного намеренно: цельный чанк переиспользуется, огрызок —
    нет. Соседний запрос внутри того же чанка наружу уже не идёт."""
    origin = FakeOrigin()

    _serve(index, origin, tmp_path, "1990-06-01T01:00:00Z", "1990-06-01T02:00:00Z")
    neighbour = _serve(index, origin, tmp_path, "1990-06-01T04:00:00Z", "1990-06-01T05:00:00Z")

    assert origin.calls == [73656]
    assert neighbour.hits == 1


def test_the_end_of_the_period_is_included(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Граница решает, сколько чанков тянуть, и ошибка здесь удваивает походы
    наружу на каждом выровненном запросе."""
    first = align(MAPS, moment(DAY), moment("1990-06-01T05:59:59Z"))
    boundary = align(MAPS, moment(DAY), moment("1990-06-01T06:00:00Z"))

    assert first == (73656,)
    # Момент лежит уже в следующем чанке — значит он в ответе, а не рядом.
    assert boundary == (73656, 73657)


def test_a_backwards_period_is_refused() -> None:
    """`stop` раньше `start` — это опечатка в запросе, а не пустой ответ."""
    with pytest.raises(ValueError, match="задом наперёд"):
        align(MAPS, moment("1990-06-02T00:00:00Z"), moment(DAY))


def test_decades_of_maps_are_refused_before_the_first_request(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Та самая ловушка из docs/CACHE.md §1: на неделе при отладке всё быстро,
    а на десятилетиях это сотни тысяч обращений на один ответ. Отказ обязан
    случиться до первого похода наружу, иначе он бесполезен."""
    origin = FakeOrigin()

    with pytest.raises(TooManyChunksError, match="при пределе 8"):
        _serve(index, origin, tmp_path, DAY, "1990-06-30T00:00:00Z")

    assert origin.calls == []
    assert total_bytes(index) == 0


def test_a_failed_download_leaves_nothing_behind(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Origin отказал на середине. На диске не должно остаться ни огрызка, ни
    строки в индексе: строка без файла позже спишет байты, которых нет."""
    origin = FakeOrigin(broken=73657)

    with pytest.raises(TimeoutError):
        _serve(index, origin, tmp_path, DAY, "1990-06-01T12:00:00Z")

    assert [found.chunk for found in entries(index)] == ["73656"]
    assert total_bytes(index) == len(b"2t:73656")
    # Ни `.part`, ни полуфабриката под настоящим именем.
    assert sorted(item.name for item in tmp_path.rglob("*.chunk*")) == ["73656.chunk"]


def test_two_writers_do_not_share_a_draft(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Черновик у каждого писателя свой.

    `docs/API_CONTRACT.md` §7 разрешает четыре холодных запроса разом, и API с
    воркером упреждения нередко тянут один и тот же чанк одновременно. С общим
    именем черновика они пишут вперемешку в один файл, а `finally` первого
    стирает недописанное второго. Здесь второй писатель заходит внутрь первого:
    в этот момент на диске обязаны лежать два разных черновика.
    """
    seen: list[str] = []
    other = open_index(tmp_path / "cache.sqlite")

    class Inner(FakeOrigin):
        def fetch(self, variable: str, chunk: int) -> bytes:
            seen.extend(sorted(item.name for item in tmp_path.rglob("*.chunk*")))
            return super().fetch(variable, chunk)

    class Outer(FakeOrigin):
        def fetch(self, variable: str, chunk: int) -> bytes:
            _serve(other, Inner(), tmp_path, DAY, DAY)
            return super().fetch(variable, chunk)

    _serve(index, Outer(), tmp_path, DAY, DAY)

    # Оба черновика недописаны, готового файла ещё нет ни у одного.
    assert len(set(seen)) == 2
    assert all(name.startswith(".73656.chunk") for name in seen)


def test_the_index_records_the_file_that_is_already_in_place(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Порядок на промахе: сначала файл, потом строка. Обратный оставляет после
    падения запись, указывающую в никуда."""
    origin = FakeOrigin()
    _serve(index, origin, tmp_path, DAY, DAY)

    row = lookup(index, Key("era5", "2t", "era5-final", "v1", "73656"))

    assert row is not None
    lying = Path(row.path)
    assert lying.is_file() and lying.read_bytes() == b"2t:73656"
    assert row.bytes == len(b"2t:73656") and row.origin == "arco"


def test_a_chunk_deleted_by_hand_is_pulled_again(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Диск чистят мимо индекса — руками, чужим скриптом, переездом. Строка,
    пережившая свой файл, отдала бы пользователю несуществующий путь."""
    origin = FakeOrigin()
    served = _serve(index, origin, tmp_path, DAY, DAY)
    served.paths[0].unlink()

    again = _serve(index, origin, tmp_path, DAY, DAY)

    assert (again.hits, again.misses) == (0, 1)
    assert origin.calls == [73656, 73656]
    assert len(list(entries(index))) == 1
    assert served.paths[0].is_file()


def test_a_hit_counts_as_an_access(index: sqlite3.Connection, tmp_path: Path) -> None:
    """По времени доступа вытесняет 3.3. Попадание, не отмеченное в индексе, —
    это чанк, который вылетит первым именно потому, что его читают."""
    origin = FakeOrigin()
    _serve(index, origin, tmp_path, DAY, DAY, now=1_800_000_000.0)

    _serve(index, origin, tmp_path, DAY, DAY, now=1_800_000_300.0)
    _serve(index, origin, tmp_path, DAY, DAY, now=1_800_000_600.0)

    row = lookup(index, Key("era5", "2t", "era5-final", "v1", "73656"))
    assert row is not None
    # Считаются попадания, а не запросы: первый был промахом, и записывать его
    # в популярность чанка не за что — чанка тогда ещё не было.
    assert (row.access_count, row.last_access) == (2, 1_800_000_600.0)


def test_provisional_data_lands_under_its_own_key(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """ERA5T и финальный ERA5 — разные ключи и разные каталоги (docs/CACHE.md
    §5). Иначе приход финальных данных пришлось бы вылавливать обходом, а до
    того пользователь получал бы предварительные под видом окончательных."""
    origin = FakeOrigin(since=73657)

    _serve(index, origin, tmp_path, DAY, "1990-06-01T12:00:00Z")

    assert sorted(found.source_version for found in entries(index)) == [
        "era5-final",
        "era5t",
        "era5t",
    ]
    assert (tmp_path / "era5" / "2t" / "era5t" / "v1" / "73657.chunk").is_file()


def test_the_price_of_a_download_is_measured(index: sqlite3.Connection, tmp_path: Path) -> None:
    """`cost_ms` — не украшение: по нему 3.4 решает, что дороже восстановить, и
    держит это в кэше дольше при равной давности."""
    origin = FakeOrigin()
    ticks = iter([10.0, 10.4, 20.0, 20.1])

    served = serve(
        index,
        origin,
        "2t",
        moment(DAY),
        moment("1990-06-01T06:00:00Z"),
        root=tmp_path,
        clock=lambda: next(ticks),
    )

    assert served.origin_latency_ms == 500
    assert sorted(found.cost_ms for found in entries(index)) == [100, 400]


def test_a_cold_period_is_seen_before_the_request(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Счётчик холодных запросов к истории (docs/API_CONTRACT.md §3, четыре
    разом) обязан решать до похода наружу: после `serve` решать уже поздно.

    Один недостающий чанк делает холодным весь период: наружу пойдут за ним, а
    ждать в очереди будет весь запрос."""
    origin = FakeOrigin()
    _serve(index, origin, tmp_path, DAY, "1990-06-01T06:00:00Z")

    assert cold(origin, "2t", moment(DAY), moment(DAY), root=tmp_path) is False
    assert cold(origin, "2t", moment(DAY), moment("1990-06-01T12:00:00Z"), root=tmp_path) is True
    assert cold(origin, "2t", moment(DAY), moment(DAY), root=tmp_path / "empty") is True


def test_looking_for_a_cold_period_does_not_warm_it(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Смотреть на диск, а не в индекс — не экономия: `hit` двигает счётчики
    доступа, по которым 3.3 решает, что вытеснять. Спрошенный дважды на каждый
    запрос, он сделал бы историю вдвое «горячее», чем она есть."""
    origin = FakeOrigin()
    _serve(index, origin, tmp_path, DAY, DAY)
    before = lookup(index, Key("era5", "2t", "era5-final", "v1", "73656"))

    cold(origin, "2t", moment(DAY), moment(DAY), root=tmp_path)

    after = lookup(index, Key("era5", "2t", "era5-final", "v1", "73656"))
    assert before is not None and after is not None
    assert (after.access_count, after.last_access) == (before.access_count, before.last_access)
    assert origin.calls == [73656]

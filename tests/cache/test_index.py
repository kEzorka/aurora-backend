"""Индекс кэша — приёмка BACKLOG 3.1.

Критерий дословно: «все поля из `CACHE.md`; WAL включён; параллельные чтение и
запись не блокируются». Все три проверяются буквально: набор столбцов — через
`pragma table_info`, режим журнала — чтением режима у настоящего файла, а не
предположением, что `pragma` сработала, и одновременность — двумя открытыми
соединениями, у которых чтение действительно не закрыто на время записи.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cache.index import (
    Key,
    entries,
    forget,
    hit,
    log_query,
    lookup,
    open_index,
    pin,
    record,
    total_bytes,
    touch,
)

#: Поля из docs/CACHE.md §4, дословно. Свои столбцы сверх этого — дело наше,
#: недостающий — невыполненный критерий.
REQUIRED = {
    "key",
    "path",
    "bytes",
    "last_access",
    "access_count",
    "origin",
    "cost_ms",
    "pinned",
    "source_version",
}

NOW = 1_800_000_000.0

MOSCOW = Key("era5", "2t", "era5-final", "v1", "t=2024-01-01,lat=6,lon=12")
OCEAN = Key("era5", "10u", "era5-final", "v1", "t=2024-01-01,lat=6,lon=12")


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    """Индекс на настоящем файле: WAL в памяти невозможен, и тест на нём
    проверял бы не то, что работает в бою."""
    return open_index(str(tmp_path / "cache.sqlite"))


def _record(conn: sqlite3.Connection, key: Key, **over: object) -> None:
    fields: dict[str, object] = {
        "path": f"/cache/{key.variable}.zarr",
        "size": 1 << 20,
        "origin": "arco",
        "cost_ms": 200,
        "now": NOW,
    }
    fields.update(over)
    record(conn, key, **fields)  # type: ignore[arg-type]


def test_the_table_has_every_field_the_document_names(index: sqlite3.Connection) -> None:
    """Сам критерий приёмки. Проверяется список, а не рассуждение о нём:
    столбец, забытый здесь, всплывёт в 3.3 как вытеснение, которому нечем
    считать цену восстановления."""
    columns = {row["name"] for row in index.execute("pragma table_info(cache_index)")}

    assert columns >= REQUIRED


def test_the_journal_is_wal(index: sqlite3.Connection) -> None:
    """`pragma journal_mode` умеет молча не примениться — например, внутри
    открытой транзакции. Тогда всё поедет на старом журнале, где читатель
    блокирует писателя, и заметят это под нагрузкой через полгода."""
    assert index.execute("pragma journal_mode").fetchone()[0].lower() == "wal"


def test_a_write_goes_through_while_a_read_is_open(tmp_path: Path) -> None:
    """Второй критерий приёмки, и проверять его надо именно так.

    Наивная проверка «прочитали, потом записали» проходит и на откатном
    журнале: читатель к моменту записи уже отпустил базу. Здесь чтение
    **остаётся открытым** — воркер пишет, пока API держит транзакцию, — и
    короткий `busy_timeout` у писателя превращает блокировку в быстрый отказ,
    а не в зависший прогон.
    """
    path = str(tmp_path / "cache.sqlite")
    reader = open_index(path)
    writer = open_index(path)
    writer.execute("pragma busy_timeout=200")
    _record(writer, MOSCOW)

    reader.execute("begin deferred")
    before = list(entries(reader))
    _record(writer, OCEAN)

    # Открытая транзакция читателя видит свой снимок — это и есть WAL, а не
    # везение с порядком блокировок.
    assert [found.key for found in entries(reader)] == [found.key for found in before]
    reader.execute("commit")
    assert len(list(entries(reader))) == 2


def test_a_repeated_download_updates_the_file_but_keeps_the_history(
    index: sqlite3.Connection,
) -> None:
    """Чанк могли выкачать заново после вытеснения. Путь и цена при этом новые,
    а счётчик обращений и пиннинг — про читателей, и обнулять их незачем."""
    _record(index, MOSCOW)
    touch(index, MOSCOW, now=NOW + 60)
    pin(index, MOSCOW)

    _record(index, MOSCOW, path="/cache/2t.new.zarr", size=2 << 20, cost_ms=900, now=NOW + 120)

    again = lookup(index, MOSCOW)
    assert again is not None
    assert (again.path, again.bytes, again.cost_ms) == ("/cache/2t.new.zarr", 2 << 20, 900)
    assert (again.access_count, again.pinned) == (1, True)


def test_reading_the_index_is_not_an_access(index: sqlite3.Connection) -> None:
    """`lookup` двигал бы время доступа и у обслуживающих обходов: отчёт и
    вытеснение тоже читают строки, и LRU показывал бы их, а не пользователей."""
    _record(index, MOSCOW)

    lookup(index, MOSCOW)
    assert lookup(index, MOSCOW) is not None
    quiet = lookup(index, MOSCOW)
    assert quiet is not None and quiet.access_count == 0

    served = hit(index, MOSCOW, now=NOW + 300)
    assert served is not None
    noted = lookup(index, MOSCOW)
    assert noted is not None
    assert (noted.access_count, noted.last_access) == (1, NOW + 300)


def test_a_miss_is_a_miss(index: sqlite3.Connection) -> None:
    """Ключа нет — это `None`, а не пустая запись: 3.2 по этому ответу решает,
    идти ли наружу."""
    assert hit(index, MOSCOW) is None
    assert not touch(index, MOSCOW)
    assert not forget(index, MOSCOW)


def test_a_new_adapter_version_is_a_different_key(index: sqlite3.Connection) -> None:
    """Инвалидация по версии — это промах, а не обход и удаление
    (docs/CACHE.md §5). Иначе исправление в адаптере тихо оставило бы в кэше
    данные, собранные по старым правилам."""
    _record(index, MOSCOW)

    assert hit(index, MOSCOW._replace(adapter_version="v2")) is None
    assert hit(index, MOSCOW._replace(source_version="era5t")) is None
    assert hit(index, MOSCOW) is not None


def test_what_era5t_wrote_can_be_listed(index: sqlite3.Connection) -> None:
    """Приход финального ERA5 требует найти «всё за период с
    `source_version = era5t`». Это запрос по столбцу; в мешанине из ключа его
    пришлось бы искать подстрокой."""
    _record(index, MOSCOW._replace(source_version="era5t"))
    _record(index, OCEAN)

    provisional = [found.key for found in entries(index, source_version="era5t")]

    assert provisional == [MOSCOW._replace(source_version="era5t").text()]


def test_the_walk_starts_from_the_oldest_access(index: sqlite3.Connection) -> None:
    """Порядок обхода — тот, в котором 3.3 будет вытеснять."""
    _record(index, MOSCOW)
    _record(index, OCEAN)
    touch(index, MOSCOW, now=NOW + 600)

    assert [found.variable for found in entries(index)] == ["10u", "2t"]


def test_the_size_is_counted_by_the_index(index: sqlite3.Connection) -> None:
    """С этим числом сверяются ватермарки 3.3, и берётся оно из учёта: обход
    диска на каждую проверку — это чтение всего кэша ради одного сравнения."""
    assert total_bytes(index) == 0

    _record(index, MOSCOW, size=3)
    _record(index, OCEAN, size=4)

    assert total_bytes(index) == 7
    forget(index, OCEAN)
    assert total_bytes(index) == 3


def test_a_typo_in_the_origin_does_not_reach_the_table(index: sqlite3.Connection) -> None:
    """Неверный origin — это неверная цена восстановления, а по ней 3.3 решает,
    что удалять. Ошибка тихая, поэтому её ловят на входе."""
    with pytest.raises(ValueError):
        _record(index, MOSCOW, origin="s3")

    assert lookup(index, MOSCOW) is None


def test_a_separator_inside_a_key_is_refused(index: sqlite3.Connection) -> None:
    """Иначе два разных чанка сложились бы в один ключ, и кэш отдавал бы чужие
    данные — молча и правдоподобно."""
    with pytest.raises(ValueError):
        Key("era5", "2t", "era5-final", "v1", "t=0/lat=6").text()
    with pytest.raises(ValueError):
        Key("era5", "", "era5-final", "v1", "t=0").text()


def test_the_query_log_does_not_touch_the_cache(index: sqlite3.Connection) -> None:
    """Журнал нужен для решений, а не для работы кэша (docs/CACHE.md §4)."""
    assert log_query(
        index,
        endpoint="/v1/history/point",
        cache_hit=False,
        latency_ms=1_200,
        area="55.75,37.62",
        variable="2t",
        start="2020-01-01",
        stop="2020-12-31",
        now=NOW,
    )

    logged = index.execute("select endpoint, cache_hit, latency_ms from query_log").fetchall()

    assert [tuple(row) for row in logged] == [("/v1/history/point", 0, 1_200)]
    assert total_bytes(index) == 0


def test_a_broken_log_does_not_break_a_read(index: sqlite3.Connection) -> None:
    """Журнал пишется на горячем пути. Запертая база или кончившееся место
    превратили бы «нечего записать в отчёт» в «пятьсот пользователю»."""
    index.execute("drop table query_log")

    assert not log_query(index, endpoint="/v1/history/point", cache_hit=True, latency_ms=3)


def test_the_index_refuses_to_pretend_it_has_wal() -> None:
    """WAL в памяти невозможен. Молчаливое согласие тут — индекс, который
    исчезает при перезапуске, оставляя файлы-сироты на диске."""
    with pytest.raises(RuntimeError, match="WAL"):
        open_index(":memory:")

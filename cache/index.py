"""Индекс кэша в SQLite (BACKLOG 3.1).

Учёт ведётся свой, а не файловой системой: чтобы вытеснять, нужно знать время
последнего обращения, а `atime` на боевых машинах отключают ради скорости
(docs/CACHE.md §4). Плюс к времени нужны цена восстановления и флаг пиннинга —
их файловой системе взять негде вовсе.

Ключ — чанк, а не запрос (§3.1): `(датасет, переменная, версия источника,
версия адаптера, индекс чанка)`. Версии внутри ключа превращают инвалидацию в
промах: после смены версии адаптера старое просто перестаёт находиться и
уходит вытеснением, вместо обхода и удаления (§5). Версия источника при этом
ещё и отдельный столбец: «заменить всё, что записано ERA5T» — это запрос
`where source_version = 'era5t'`, а не поиск подстроки в ключе.

Таблиц две, и путать их не надо (§4). `cache_index` — что физически лежит на
диске; без неё вытеснение невозможно. `query_log` — что спрашивали люди; в
работе кэша она не участвует вовсе, поэтому неудача записи в журнал не имеет
права уронить чтение из кэша.

Режим журнала — WAL: воркер пишет, API читает, и они друг друга не ждут.
Отсюда два следствия. Первое: файл индекса обязан лежать на локальном диске —
WAL держит рядом `-wal` и `-shm` и по сетевой файловой системе не работает.
Второе: режим проверяется при открытии, а не предполагается — `PRAGMA` умеет
молча не примениться, и тогда всё едет на старом журнале, где читатель
блокирует писателя. Обнаруживается это через полгода под нагрузкой.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NamedTuple

#: Откуда взят чанк (docs/CACHE.md §4). Список закрытый и проверяется базой:
#: опечатка в origin — это неверная цена восстановления, а по ней 3.3 решает,
#: что вытеснять. Ошибка тихая и всплывёт месяцы спустя.
ORIGINS: Final = ("arco", "cds", "local")

#: Сколько ждать освобождения базы, прежде чем признать её занятой. В WAL
#: писатели всё равно сериализуются между собой, и пять секунд — это про
#: очередь из двух воркеров, а не про блокировку читателем.
BUSY_TIMEOUT_MS: Final = 5_000

#: Разделитель частей ключа. Ни одна часть его содержать не вправе — иначе два
#: разных чанка сложатся в один ключ и кэш начнёт отдавать чужие данные.
SEPARATOR: Final = "/"

#: Список origin для `check` — собран, а не взят из `repr` кортежа: `repr`
#: пишет висячую запятую, когда значение остаётся одно, и схема перестаёт
#: создаваться в тот день, когда список сократят.
_ORIGIN_LIST: Final = ", ".join(f"'{origin}'" for origin in ORIGINS)

_SCHEMA: Final = f"""
create table if not exists cache_index (
    key            text primary key,
    dataset        text not null,
    variable       text not null,
    source_version text not null,
    adapter_version text not null,
    chunk          text not null,
    path           text not null,
    bytes          integer not null,
    last_access    real not null,
    access_count   integer not null default 0,
    origin         text not null check (origin in ({_ORIGIN_LIST})),
    cost_ms        integer not null default 0,
    pinned         integer not null default 0,
    created_at     real not null
);
create index if not exists cache_index_source on cache_index (source_version);
create index if not exists cache_index_access on cache_index (last_access);

create table if not exists query_log (
    id         integer primary key autoincrement,
    ts         real not null,
    endpoint   text not null,
    area       text,
    variable   text,
    start      text,
    stop       text,
    cache_hit  integer not null,
    latency_ms integer not null
);
create index if not exists query_log_ts on query_log (ts);
"""


class Key(NamedTuple):
    """Ключ кэша: чанк источника, а не запрос пользователя.

    Версии — часть ключа намеренно (docs/CACHE.md §5). Один и тот же чанк,
    прочитанный адаптером другой версии, — другой объект: если бы версия в
    ключ не входила, исправление в адаптере тихо оставляло бы в кэше данные,
    собранные по старым правилам, и обнаруживалось бы это разбором жалоб.
    """

    dataset: str
    variable: str
    source_version: str
    adapter_version: str
    chunk: str

    def text(self) -> str:
        """Ключ строкой — то, что лежит в `cache_index.key`."""
        parts = tuple(self)
        for part in parts:
            if not part or SEPARATOR in part:
                raise ValueError(f"часть ключа не годится: {part!r}")
        return SEPARATOR.join(parts)


class Entry(NamedTuple):
    """Строка `cache_index` — то, что физически лежит на диске."""

    key: str
    dataset: str
    variable: str
    source_version: str
    adapter_version: str
    chunk: str
    path: str
    bytes: int
    last_access: float
    access_count: int
    origin: str
    cost_ms: int
    pinned: bool
    created_at: float


def open_index(path: str | Path) -> sqlite3.Connection:
    """Открыть индекс, включить WAL и убедиться, что он включился.

    `isolation_level=None` — не мелочь: драйвер иначе сам открывает транзакцию
    перед первым же изменением, а сменить режим журнала внутри транзакции
    нельзя, и `PRAGMA` тихо ничего не сделает. Транзакции здесь ставятся
    руками, там, где они нужны.

    `path` — файл, не `:memory:`: WAL в памяти невозможен, а индекс, живущий
    до перезапуска, оставит после себя файлы-сироты на диске (docs/CACHE.md §4).
    """
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    mode = str(conn.execute("pragma journal_mode=wal").fetchone()[0]).lower()
    if mode != "wal":
        conn.close()
        raise RuntimeError(f"WAL не включился, режим журнала: {mode!r}")
    conn.execute(f"pragma busy_timeout={BUSY_TIMEOUT_MS}")
    # `normal` вместо `full`: в WAL это теряет последние транзакции только при
    # отказе питания, и потерянные — это запись о чанке, который лежит на
    # диске. Худшее последствие — лишний поход в origin.
    conn.execute("pragma synchronous=normal")
    # `executescript` перед выполнением сам делает `commit`. Здесь это безвредно
    # — транзакции ещё нет, — но вызывать его внутри своей транзакции нельзя:
    # она молча закроется на середине.
    conn.executescript(_SCHEMA)
    return conn


def record(
    conn: sqlite3.Connection,
    key: Key,
    *,
    path: str,
    size: int,
    origin: str,
    cost_ms: int,
    pinned: bool = False,
    now: float | None = None,
) -> None:
    """Записать, что чанк лёг на диск.

    Повторная запись того же ключа — не ошибка, а перезаливка: чанк могли
    выкачать заново после вытеснения. Тогда обновляются путь, размер и цена, а
    счётчик обращений и пиннинг остаются: они про поведение читателей, а не
    про файл.
    """
    if origin not in ORIGINS:
        raise ValueError(f"неизвестный origin: {origin!r}")
    moment = _moment(now)
    conn.execute(
        """
        insert into cache_index (
            key, dataset, variable, source_version, adapter_version, chunk,
            path, bytes, last_access, access_count, origin, cost_ms, pinned, created_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        on conflict(key) do update set
            path = excluded.path,
            bytes = excluded.bytes,
            cost_ms = excluded.cost_ms,
            origin = excluded.origin,
            last_access = excluded.last_access
        """,
        (
            key.text(),
            key.dataset,
            key.variable,
            key.source_version,
            key.adapter_version,
            key.chunk,
            path,
            size,
            moment,
            origin,
            cost_ms,
            int(pinned),
            moment,
        ),
    )


def lookup(conn: sqlite3.Connection, key: Key) -> Entry | None:
    """Что известно про ключ. Обращением это не считается — см. `touch`."""
    row = conn.execute("select * from cache_index where key = ?", (key.text(),)).fetchone()
    return _entry(row) if row is not None else None


def touch(conn: sqlite3.Connection, key: Key, *, now: float | None = None) -> bool:
    """Отметить обращение: время и счётчик.

    Отдельно от `lookup` потому, что смотрят в индекс не только читатели:
    отчёт, вытеснение и чистка тоже читают строки, и если бы чтение двигало
    время доступа, LRU показывал бы популярность обслуживающих обходов, а не
    пользователей.
    """
    changed = conn.execute(
        "update cache_index set last_access = ?, access_count = access_count + 1 where key = ?",
        (_moment(now), key.text()),
    )
    return changed.rowcount > 0


def hit(conn: sqlite3.Connection, key: Key, *, now: float | None = None) -> Entry | None:
    """Чтение из кэша: найти и отметить обращение. Промах — `None`."""
    found = lookup(conn, key)
    if found is None:
        return None
    touch(conn, key, now=now)
    return found


def pin(conn: sqlite3.Connection, key: Key, *, pinned: bool = True) -> bool:
    """Поставить или снять запрет на вытеснение (docs/CACHE.md §2)."""
    changed = conn.execute(
        "update cache_index set pinned = ? where key = ?", (int(pinned), key.text())
    )
    return changed.rowcount > 0


def forget(conn: sqlite3.Connection, key: Key) -> bool:
    """Убрать запись из индекса. Файл на диске удаляет вызвавший.

    Порядок именно такой: сначала уходит файл, потом запись. Обратный порядок
    оставляет файл, про который никто не знает, — место занято навсегда.
    """
    return conn.execute("delete from cache_index where key = ?", (key.text(),)).rowcount > 0


def entries(conn: sqlite3.Connection, *, source_version: str | None = None) -> Iterator[Entry]:
    """Пройти по индексу, от давнего обращения к свежему.

    Порядок — тот, в котором 3.3 будет вытеснять, если цену восстановления не
    учитывать. `source_version` отвечает на вопрос из §5: «что тут записано
    ERA5T» — это столбец, а не подстрока в ключе.
    """
    if source_version is None:
        rows = conn.execute("select * from cache_index order by last_access")
    else:
        rows = conn.execute(
            "select * from cache_index where source_version = ? order by last_access",
            (source_version,),
        )
    for row in rows:
        yield _entry(row)


def total_bytes(conn: sqlite3.Connection) -> int:
    """Сколько кэш занимает по учёту. С ватермарками 3.3 сверяется это число."""
    return int(conn.execute("select coalesce(sum(bytes), 0) from cache_index").fetchone()[0])


def log_query(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    cache_hit: bool,
    latency_ms: int,
    area: str | None = None,
    variable: str | None = None,
    start: str | None = None,
    stop: str | None = None,
    now: float | None = None,
) -> bool:
    """Записать вопрос пользователя в журнал (docs/CACHE.md §4).

    Журнал в работе кэша не участвует, поэтому его отказ не имеет права стать
    отказом ответа: пишется он на пути горячего чтения, и переполненный диск
    или запертая база превратили бы «нечего записать в отчёт» в «пятьсот
    пользователю». Возвращается признак записи — для отчёта, а не для решений.
    """
    try:
        conn.execute(
            """
            insert into query_log (ts, endpoint, area, variable, start, stop, cache_hit, latency_ms)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _moment(now),
                endpoint,
                area,
                variable,
                start,
                stop,
                int(cache_hit),
                latency_ms,
            ),
        )
    except sqlite3.Error:
        return False
    return True


def _entry(row: sqlite3.Row) -> Entry:
    return Entry(
        key=row["key"],
        dataset=row["dataset"],
        variable=row["variable"],
        source_version=row["source_version"],
        adapter_version=row["adapter_version"],
        chunk=row["chunk"],
        path=row["path"],
        bytes=row["bytes"],
        last_access=row["last_access"],
        access_count=row["access_count"],
        origin=row["origin"],
        cost_ms=row["cost_ms"],
        pinned=bool(row["pinned"]),
        created_at=row["created_at"],
    )


def _moment(now: float | None) -> float:
    """Время — секундами эпохи, а не строкой.

    По нему считается `score` вытеснения (docs/CACHE.md §3.2), то есть его
    вычитают и делят прямо в запросе. Строка ISO сортировалась бы верно, но
    арифметику пришлось бы тащить в Python и вычитывать ради неё всю таблицу.
    """
    return time.time() if now is None else now

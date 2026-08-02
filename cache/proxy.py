"""Прокси к origin с выравниванием по чанкам (BACKLOG 3.2).

Кэшируется чанк источника, а не запрос пользователя (docs/CACHE.md §3.1).
Разница видна на втором запросе: два пересекающихся периода, положенные в кэш
как запросы, дают два перекрывающихся куска — вдвое больше места при том же
покрытии. Выровненные по границам чанков, они делят общее.

Поэтому внешний запрос всегда шире нужного: период расширяется до границ
чанков origin, и на диск ложится цельный чанк. Лишние байты окупаются первым
же соседним запросом.

Предел на число чанков в одном запросе задаёт сам origin, и это не
перестраховка. Чанк ARCO равен одному шагу времени: ряд в точке за 80 лет по
часам — это сотни тысяч обращений на один ответ (docs/CACHE.md §1). На неделе
при отладке всё быстро, поэтому предел стоит в `align`, до первого похода
наружу, а не в середине выкачивания.

Отказ источника кэшируется тоже (docs/CACHE.md §3.3). Реанализ отстаёт от
сегодняшнего дня на ~5 суток, и дата из этой слепой зоны — не ошибка, а
нормальный вопрос, на который нет ответа. Без отрицательного кэша один человек,
листающий календарь на фронтенде, превращает каждое движение в поход наружу.

Порядок на промахе: сначала файл, потом запись в индекс. Обратный порядок
оставляет после падения строку, указывающую в никуда, — а вытеснение по ней
позже спишет байты, которых на диске не было. Удаление зеркально: сперва
файл, потом строка (`cache.index.forget`).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, Protocol

from cache.index import Key, absent, forget, forget_absent, hit, mark_absent, record

#: Расширение файла чанка. Своё, а не `.zarr`/`.grib`: в кэше лежит то, что
#: отдал origin, и разбирать это — дело того, кто просил, а не хранилища.
CHUNK_SUFFIX: Final = ".chunk"

#: Имя незаконченного файла. Точка в начале — чтобы обход каталога отличал
#: недописанное от готового и без обращения к индексу.
PART_PREFIX: Final = "."


class TooManyChunksError(ValueError):
    """Запрос, который развернулся бы в сотни тысяч обращений к origin."""


class NotYetError(LookupError):
    """Данных в источнике ещё нет: спросили дату из слепой зоны реанализа.

    Отдельно от «источник упал»: реанализ отстаёт на ~5 суток (docs/CACHE.md
    §3.3), и это не сбой, а нормальное состояние, ответ на которое — сказать
    пользователю про доступный период, а не повторять поход наружу.
    """


class OriginError(RuntimeError):
    """Источник ответил сбоем или данными, которые адаптер не понял.

    Тип принадлежит кэшу намеренно: API запрещено импортировать адаптеры, но
    именно API решает, что штатная слепая зона — это `404`, а настоящий отказ
    origin — `503` с `Retry-After`.
    """


class Grid(NamedTuple):
    """Как origin нарезан по времени.

    `span` — шагов данных в одном чанке. У ARCO он равен единице (чанк — это
    шаг времени), у рядов CDS — годам; отсюда и разные пределы `max_chunks`:
    восемь тысяч чанков для точечных рядов — обычный запрос за год, а для карт
    ARCO это и есть та самая катастрофа.
    """

    epoch: datetime
    step: timedelta
    span: int
    max_chunks: int

    def holding(self, moment: datetime) -> int:
        """Номер чанка, в который попадает момент.

        Не `index`: `Grid` — это кортеж, а у кортежа `index` уже занят поиском
        значения. Метод с тем же именем — это подмена, которую заметят не здесь.
        """
        return int((moment - self.epoch) // (self.step * self.span))

    def begins(self, chunk: int) -> datetime:
        """Момент, с которого чанк начинается."""
        return self.epoch + self.step * self.span * chunk


class Origin(Protocol):
    """Внешний источник: ARCO для карт, CDS для рядов (docs/CACHE.md §1).

    Версию источника origin сообщает **до** выкачивания и по номеру чанка:
    она входит в ключ, а ключ нужен, чтобы понять, надо ли вообще идти наружу.
    Взять её из ответа было бы поздно — ради ключа пришлось бы качать всегда.
    Знание тут нехитрое: последние несколько суток — ERA5T, остальное —
    финальный ERA5.
    """

    name: str
    dataset: str
    version: str
    grid: Grid

    def source_version(self, chunk: int) -> str: ...

    def fetch(self, variable: str, chunk: int) -> bytes:
        """Отдать чанк. `NotYetError`, когда этих данных в источнике ещё нет."""
        ...


class Served(NamedTuple):
    """Чем закончился запрос: файлы и то, что API кладёт в блок `cache`."""

    paths: tuple[Path, ...]
    hits: int
    misses: int
    origin_latency_ms: int

    @property
    def hit(self) -> bool:
        """Ответ целиком собран из кэша (docs/API_CONTRACT.md §3)."""
        return self.misses == 0


def align(grid: Grid, start: datetime, stop: datetime) -> tuple[int, ...]:
    """Номера чанков, покрывающих период. Границы включительно с обеих сторон.

    Включительно потому, что просят так: «с 1990-01-01 по 1990-12-31» — это
    вместе с последним днём. Момент, попавший ровно на границу чанка, тянет за
    собой следующий чанк: он там и лежит.
    """
    if stop < start:
        raise ValueError(f"период задом наперёд: {start.isoformat()}..{stop.isoformat()}")
    first = grid.holding(start)
    last = grid.holding(stop)
    count = last - first + 1
    if count > grid.max_chunks:
        raise TooManyChunksError(
            f"{count} чанков на один запрос при пределе {grid.max_chunks}: "
            f"{start.isoformat()}..{stop.isoformat()}"
        )
    return tuple(range(first, last + 1))


def serve(
    conn: sqlite3.Connection,
    origin: Origin,
    variable: str,
    start: datetime,
    stop: datetime,
    *,
    root: str | Path,
    now: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Served:
    """Отдать период: что есть — с диска, чего нет — из origin и на диск.

    `conn` — соединение `cache.index`. `clock` — секундомер цены восстановления
    (`cost_ms` в индексе); отдельным аргументом, потому что по этой цене 3.3
    решает, что переживёт вытеснение, и проверять её надо не «на глаз».
    """
    base = Path(root)
    chunks = align(origin.grid, start, stop)

    paths: list[Path] = []
    hits = 0
    latency_ms = 0
    for chunk in chunks:
        key = _key(origin, variable, chunk)
        path = _path(base, key)
        found = hit(conn, key, now=now)
        if found is not None and path.is_file():
            hits += 1
            paths.append(path)
            continue
        if found is not None:
            # Строка есть, файла нет: кто-то почистил диск руками мимо индекса.
            # Строку убираем, чанк качаем заново — иначе вытеснение потом будет
            # списывать байты, которых нет, и место так и не найдётся.
            forget(conn, key)
        # Отказ спрашивается после кэша, а не до: данные на диске старше любой
        # записи о том, что их нет.
        refused = absent(conn, key, now=now)
        if refused is not None:
            raise NotYetError(f"{key.text()}: {refused.reason}")
        cost_ms = _download(conn, origin, key, variable, chunk, path, now=now, clock=clock)
        latency_ms += cost_ms
        paths.append(path)
    return Served(tuple(paths), hits, len(chunks) - hits, latency_ms)


def cold(
    origin: Origin, variable: str, start: datetime, stop: datetime, *, root: str | Path
) -> bool:
    """Похоже ли, что запрос пойдёт наружу. Ничего не меняет и не считает.

    Нужно ровно одному месту — счётчику холодных запросов к истории
    (docs/API_CONTRACT.md §3, четыре разом): решать, занимать ли место в
    очереди, надо **до** запроса, а `serve` к этому моменту уже сходил бы
    наружу.

    Смотрит на диск, а не в индекс, и это не экономия: `hit` двигает счётчики
    доступа, по которым 3.3 решает, что вытеснять, и спрашивать его дважды
    значит сделать все запросы к истории вдвое «горячее», чем они были.
    Расхождение возможно (файл есть, строки в индексе нет) — стоит оно лишнего
    места в очереди, а решает всё равно `serve`.
    """
    base = Path(root)
    return any(
        not _path(base, _key(origin, variable, chunk)).is_file()
        for chunk in align(origin.grid, start, stop)
    )


def _download(
    conn: sqlite3.Connection,
    origin: Origin,
    key: Key,
    variable: str,
    chunk: int,
    path: Path,
    *,
    now: float | None,
    clock: Callable[[], float],
) -> int:
    """Выкачать чанк, положить целиком, записать в индекс. Вернуть цену в мс.

    Файл кладётся переименованием: половина чанка, застигнутая падением,
    неотличима от целого — и отдастся пользователю как данные.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Имя черновика своё у каждого пишущего. Общее имя рушит ровно то, ради
    # чего черновик и заводился: API на холодном запросе и воркер на упреждении
    # тянут один чанк одновременно (docs/API_CONTRACT.md §7 разрешает четыре
    # холодных запроса разом), пишут вперемешку в один файл и переименовывают
    # смесь — а `finally` одного стирает черновик другого на середине.
    handle, draft = tempfile.mkstemp(dir=path.parent, prefix=f"{PART_PREFIX}{path.name}.")
    os.close(handle)
    part = Path(draft)
    began = clock()
    try:
        try:
            data = origin.fetch(variable, chunk)
        except NotYetError as refusal:
            # Отказ запоминается на шесть часов (docs/CACHE.md §3.3). Без этого
            # один любопытный человек, ткнувший во вчерашнюю дату на фронтенде,
            # превращает каждый свой скролл в поход наружу — а отвечает на них
            # очередь CDS, общая на весь сервис.
            mark_absent(conn, key, reason=str(refusal), now=now)
            raise
        part.write_bytes(data)
        part.replace(path)
        cost_ms = max(0, int((clock() - began) * 1000))
        # Запись в индекс — здесь, после переименования: файл уже на месте, и
        # порядок «сначала файл, потом строка» держится сам, а не по памяти
        # того, кто будет править эту функцию следующим.
        record(
            conn,
            key,
            path=str(path),
            size=len(data),
            origin=origin.name,
            cost_ms=cost_ms,
            now=now,
        )
        # Данные пришли — протухший отказ по этому ключу больше не нужен. Сам
        # он никого не задержит (`absent` смотрит на срок), но остался бы в
        # таблице навсегда: повторно про эту дату уже не спросят.
        forget_absent(conn, key)
    finally:
        part.unlink(missing_ok=True)
    return cost_ms


def _key(origin: Origin, variable: str, chunk: int) -> Key:
    return Key(
        dataset=origin.dataset,
        variable=variable,
        source_version=origin.source_version(chunk),
        adapter_version=origin.version,
        chunk=str(chunk),
    )


def _path(root: Path, key: Key) -> Path:
    """Раскладка кэша на диске повторяет ключ.

    Не хэш: когда дежурный смотрит, чем занят диск, каталоги обязаны отвечать
    на вопрос «чего тут столько» без обращения к индексу.
    """
    return (
        root
        / key.dataset
        / key.variable
        / key.source_version
        / key.adapter_version
        / f"{key.chunk}{CHUNK_SUFFIX}"
    )


def moment(text: str) -> datetime:
    """Момент из ISO-строки, всегда в UTC.

    Наивное время здесь — это молчаливое «по местному»: период, съехавший на
    три часа, попадёт не в те чанки и промахнётся мимо кэша, ничем себя не
    выдав.
    """
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

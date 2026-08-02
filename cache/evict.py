"""Вытеснение по ватермаркам, цене восстановления и пиннингу (BACKLOG 3.3, 3.4).

Чистка начинается на 85% заполнения и идёт до 70% (docs/CACHE.md §2). Два
порога, а не один, здесь не украшение: с одним порогом чистка срабатывает на
каждом следующем чанке — то есть ровно в момент записи нового прогона, когда
диск и так занят.

Чистый LRU неверен (docs/CACHE.md §3.2). Чанк ARCO возвращается за ~200 мс, ряд
CDS — за минуты, потому что там очередь. По одной давности первым вылетит
именно дорогое, и следующий запрос за ним встанет в ту же очередь. Поэтому
оценка учитывает три величины: давность, цену восстановления и размер.

Запиннённое не вытесняется вовсе. Прогноз спрашивают реже популярной
исторической точки, и чистый LRU однажды удалит именно его — а он не
восстанавливается походом наружу, его надо считать заново на GPU.

Удаление зеркально записи: сперва файл, потом строка. Обратный порядок
оставляет после падения строку, указывающую в никуда, и вытеснение по ней
позже спишет байты, которых на диске нет.

Оценка считается здесь, а не в SQL: половинное затухание требует возведения в
степень, а математические функции SQLite — опция сборки. Кэш, который на
стандартной сборке молча перестаёт чиститься, хуже кэша, читающего свой индекс.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Final, NamedTuple

from cache.index import (
    Entry,
    Key,
    default_index,
    entries,
    expire_absent,
    forget,
    open_index,
    stale_absent,
    total_bytes,
)

#: Порог запуска чистки — доля занятого места (docs/CACHE.md §2).
HIGH_WATER: Final = 0.85

#: Порог остановки. Разрыв с `HIGH_WATER` — это запас на один прогон вперёд.
LOW_WATER: Final = 0.70

#: За это время польза чанка падает вдвое. Сутки взяты по локальности запросов
#: из docs/CACHE.md §2: спросили про Москву — скоро спросят ещё, но интерес к
#: конкретной дате живёт часы, а не недели. Значение подбирается по `query_log`
#: (BACKLOG 3.5), поэтому оно параметр, а не число внутри формулы.
HALF_LIFE_S: Final = 86_400.0

#: Сколько места отведено кэшу на диске (docs/STORAGE.md §2, строка `cache/*`).
#: Значение по умолчанию, а не обязательный аргумент: чистку запускает
#: расписание, и байты, которые дежурный вводит руками раз в полгода, он введёт
#: неверно — а чистка с заниженной ёмкостью снесёт полкэша молча и штатно.
CAPACITY_BYTES: Final = 195 * (1 << 30)


class Limits(NamedTuple):
    """Сколько места отведено кэшу и когда его чистить."""

    capacity_bytes: int = CAPACITY_BYTES
    high: float = HIGH_WATER
    low: float = LOW_WATER

    def start_at(self) -> int:
        """Байты, после которых чистка запускается."""
        return int(self.capacity_bytes * self.high)

    def stop_at(self) -> int:
        """Байты, до которых чистка идёт."""
        return int(self.capacity_bytes * self.low)


class Swept(NamedTuple):
    """Чем закончилась чистка. `after` — занято после неё."""

    removed: tuple[str, ...]
    freed: int
    before: int
    after: int
    target: int
    triggered: bool

    @property
    def enough(self) -> bool:
        """Дочистили до нижней ватермарки.

        Ложь означает, что вытеснять больше нечего: остаток запиннён. Это не
        ошибка чистки, а повод для тревоги в метриках (docs/CACHE.md §6) —
        кэшу отвели меньше места, чем занимает несменяемое.
        """
        return self.after <= self.target


def score(entry: Entry, *, now: float, half_life_s: float = HALF_LIFE_S) -> float:
    """Полезность объекта: чем меньше, тем раньше вытесняется.

    `cost_ms / bytes` — сколько миллисекунд ожидания экономит каждый занятый
    байт; затухание вдвое за `half_life_s` добавляет давность. Отсюда три
    свойства, которых требует docs/CACHE.md §3.2: при равной давности дорогое
    живёт дольше дешёвого, при равной цене мелкое дольше крупного, а нетронутое
    неделю уходит первым при любых прочих.
    """
    age = max(0.0, now - entry.last_access)
    decay = math.pow(0.5, age / half_life_s)
    # Плюс единица — от нулевого размера: пустой файл в кэше означает сломанный
    # origin, и делить на него нельзя, а вытеснять его нужно наравне с прочими.
    return entry.cost_ms / (entry.bytes + 1) * decay


def sweep(
    conn: sqlite3.Connection,
    *,
    limits: Limits,
    now: float,
    half_life_s: float = HALF_LIFE_S,
) -> Swept:
    """Почистить кэш, если занято больше верхней ватермарки.

    Ниже порога запуска не делает ничего — в этом и смысл двух порогов.
    """
    before = total_bytes(conn)
    target = limits.stop_at()
    if before <= limits.start_at():
        return Swept((), 0, before, before, target, triggered=False)

    freed = 0
    removed: list[str] = []
    for found in doomed(conn, limits=limits, now=now, half_life_s=half_life_s, occupied=before):
        freed += _drop(conn, found)
        removed.append(found.key)
    # `after` перечитывается из индекса, а не считается как `before - freed`.
    # Разойтись эти два числа могут запросто: `_drop` списывает байты строки, а
    # файла под ней могло уже не быть — и тогда вычитание отчитается об
    # освобождённом месте, которого на диске не появилось. Занято ровно
    # столько, сколько говорит индекс; по нему же решает следующая чистка.
    return Swept(tuple(removed), freed, before, total_bytes(conn), target, triggered=True)


def doomed(
    conn: sqlite3.Connection,
    *,
    limits: Limits,
    now: float,
    half_life_s: float = HALF_LIFE_S,
    occupied: int | None = None,
) -> tuple[Entry, ...]:
    """Кого унесла бы чистка. Тот же отбор, но без удаления.

    Нужна ровно для `--dry-run`: команда, сносящая гигабайты, обязана уметь
    сначала показать, что именно, — иначе первый её запуск на боевом диске и
    есть проверка.

    `occupied` передаёт `sweep`, чтобы отбор шёл от того же числа, по которому
    чистка решила запуститься: индекс пишут во время чистки (прокси кладёт
    чанки, WAL это и разрешает), и второе чтение занятого дало бы отбор под
    другое заполнение, чем то, что попадёт в отчёт.
    """
    before = total_bytes(conn) if occupied is None else occupied
    if before <= limits.start_at():
        return ()

    target = limits.stop_at()
    # Запиннённое не рассматривается вовсе: оно не восстанавливается походом
    # наружу (docs/STORAGE.md §2), и «почти освободить место» за его счёт
    # означает потерять прогон.
    candidates = [found for found in entries(conn) if not found.pinned]
    candidates.sort(key=lambda found: score(found, now=now, half_life_s=half_life_s))

    chosen: list[Entry] = []
    left = before
    for found in candidates:
        if left <= target:
            break
        chosen.append(found)
        left -= found.bytes
    return tuple(chosen)


def _drop(conn: sqlite3.Connection, entry: Entry) -> int:
    """Убрать объект: сперва файл, потом строка. Вернуть освобождённые байты.

    Байты берутся из индекса, а не со `stat`: файла может уже не быть — диск
    чистят руками, чужим скриптом, переездом, — и тогда строка всё равно должна
    уйти, иначе она будет вечно занимать место, которого не занимает.
    """
    Path(entry.path).unlink(missing_ok=True)
    forget(conn, key_of(entry))
    return entry.bytes


def main(argv: list[str] | None = None, *, now: float | None = None) -> int:
    """`python -m cache.evict <индекс> --capacity N` — чистка по расписанию.

    Отдельной командой, а не проверкой внутри `serve`: чистка фоновая
    (docs/CACHE.md §2), и запускать её на пути запроса — это отдавать
    пользователю ответ после удаления сотен файлов. Ватермарки и придуманы,
    чтобы чистка не совпадала с записью.

    Код возврата ненулевой, когда до нижней ватермарки не дочистили: остаток
    запиннён, и кэшу отвели меньше места, чем занимает несменяемое.

    `now` подставляется тестами. Без него оценка бралась бы от настоящих часов,
    и проверка порядка вытеснения проверяла бы календарь: строки с давностью,
    заданной от круглой даты, при переходе через неё сравняются в оценке и
    порядок выберет `entries`, а не `score`.
    """
    parser = argparse.ArgumentParser(prog="cache.evict")
    parser.add_argument(
        "index",
        type=Path,
        nargs="?",
        default=None,
        help="файл индекса SQLite; по умолчанию — $AURORA_ROOT/cache/index.sqlite",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=CAPACITY_BYTES,
        help=f"ёмкость кэша в байтах; по умолчанию {CAPACITY_BYTES} (docs/STORAGE.md §2)",
    )
    parser.add_argument("--high", type=float, default=HIGH_WATER, help="порог запуска, доля")
    parser.add_argument("--low", type=float, default=LOW_WATER, help="порог остановки, доля")
    parser.add_argument(
        "--dry-run", action="store_true", help="показать, что ушло бы, и ничего не удалять"
    )
    args = parser.parse_args(argv)

    limits = Limits(capacity_bytes=args.capacity, high=args.high, low=args.low)
    moment = time.time() if now is None else now
    path = args.index if args.index is not None else default_index()
    if not Path(path).exists():
        # Не создавать: `open_index` разложил бы пустую схему по неверному пути,
        # отчитался «чистить нечего» и оставил боевой кэш расти дальше.
        print(f"индекса нет: {path}")
        return 1

    conn = open_index(path)
    try:
        if args.dry_run:
            for found in doomed(conn, limits=limits, now=moment):
                print(f"снесла бы {found.key} ({found.bytes} байт)")
            print(f"протухших отказов сняла бы {stale_absent(conn, now=moment)}")
            return 0

        # Отказы чистятся до вытеснения и независимо от ватермарок: они не
        # занимают места на диске, и чистка по заполнению их бы не тронула
        # никогда. Уборка тут потому, что это единственная команда кэша,
        # которую запускает расписание.
        expired = expire_absent(conn, now=moment)
        swept = sweep(conn, limits=limits, now=moment)
        for key in swept.removed:
            print(f"снесено {key}")
        print(f"занято {swept.after} из {limits.capacity_bytes} байт, освобождено {swept.freed}")
        print(f"протухших отказов снято {expired}")
        if not swept.enough:
            print(f"до {swept.target} байт не дочистили: остаток запиннён")
            return 1
        return 0
    finally:
        conn.close()


def key_of(entry: Entry) -> Key:
    """Ключ строки индекса. Все пять частей лежат в ней отдельными полями."""
    return Key(
        dataset=entry.dataset,
        variable=entry.variable,
        source_version=entry.source_version,
        adapter_version=entry.adapter_version,
        chunk=entry.chunk,
    )


if __name__ == "__main__":
    sys.exit(main())

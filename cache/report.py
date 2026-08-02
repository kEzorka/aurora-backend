"""Отчёт по кэшу: заполненность, hit rate, топ точек (BACKLOG 3.5).

Отчёт нужен ради одного числа. Hit rate ниже ~50% на устоявшемся трафике
означает одно из двух: кэшируются запросы вместо чанков либо чанки origin
слишком крупные для наших запросов (docs/CACHE.md §6). Обе причины чинятся, но
по-разному, и различить их можно только глядя на журнал.

Окна два — час и сутки. Одно суточное прячет обвал, случившийся после выкладки:
за сутки сотня свежих промахов растворяется в тысячах вчерашних попаданий.

Точечные и карточные запросы считаются отдельно, потому что за ними разные
источники: ряд в точке идёт из очереди CDS и стоит минуты, карта — из ARCO и
стоит ~200 мс (docs/CACHE.md §1). Общий hit rate по обоим сразу — это среднее
между «терпимо» и «пользователь ушёл».

Задержка меряется только на промахах: попадание с диска — это не то, ради чего
смотрят на 95-й перцентиль. Перцентиль считается здесь, а не в SQL, — в SQLite
нет оконных функций для перцентилей, а сортировать выборку за сутки в памяти
дешевле, чем заводить ради этого расширение.

Чего в отчёте нет и почему: количество вытеснений за сутки и число отказов
origin (docs/CACHE.md §6). Ни то, ни другое в `query_log` не пишется — журнал
хранит вопросы пользователя, а не события кэша. Появятся они с метриками
эпика 6, и врать про них здесь нулями хуже, чем не показывать вовсе.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Final, NamedTuple

from cache.index import default_index, open_index

#: Окна отчёта в секундах.
HOUR_S: Final = 3_600.0
DAY_S: Final = 86_400.0

#: Порог тревоги из docs/CACHE.md §6.
POOR_HIT_RATE: Final = 0.5

#: Сколько точек показывать. Десяток — это то, что укладывается в один экран
#: дежурного и решает вопрос «что пиннить» (docs/CACHE.md §3.4).
TOP: Final = 10

#: Как отличить точечный запрос от карточного. Совпадает с путями
#: `/v1/forecast/point` и `/v1/history/point` (docs/API_CONTRACT.md §2).
POINT_LIKE: Final = "%point%"
GRID_LIKE: Final = "%grid%"


class Rate(NamedTuple):
    """Попадания за окно."""

    total: int
    hits: int

    @property
    def rate(self) -> float:
        """Доля попаданий. Пустое окно — ноль, а не деление на ноль."""
        return self.hits / self.total if self.total else 0.0

    @property
    def poor(self) -> bool:
        """Тот самый порог. Пустое окно не тревога: трафика просто не было."""
        return self.total > 0 and self.rate < POOR_HIT_RATE


class Latency(NamedTuple):
    """Задержка на промахах — лицо системы для пользователя.

    `misses`, а не `count`: у кортежа `count` уже занят подсчётом значений, и
    поле с тем же именем — подмена, которую заметят не здесь.
    """

    misses: int
    mean_ms: float
    p95_ms: int


class Spot(NamedTuple):
    """Строка топа: что спрашивали и сколько раз попали."""

    area: str
    asked: int
    hits: int

    @property
    def rate(self) -> float:
        return self.hits / self.asked if self.asked else 0.0


class Report(NamedTuple):
    """Всё, что отчёт знает о кэше на момент запуска."""

    occupied_bytes: int
    pinned_bytes: int
    objects: int
    hour: Rate
    day: Rate
    point: Rate
    grid: Rate
    miss: Latency
    top: tuple[Spot, ...]


def hit_rate(conn: sqlite3.Connection, *, since: float, endpoint_like: str | None = None) -> Rate:
    """Попадания и запросы за окно, опционально по виду эндпоинта."""
    where = "ts >= ?"
    params: list[object] = [since]
    if endpoint_like is not None:
        where += " and endpoint like ?"
        params.append(endpoint_like)
    row = conn.execute(
        f"select count(*) as total, sum(cache_hit) as hits from query_log where {where}",
        params,
    ).fetchone()
    total = int(row["total"])
    # `sum` по пустой выборке — это `NULL`, а не ноль.
    return Rate(total, int(row["hits"] or 0))


def miss_latency(conn: sqlite3.Connection, *, since: float) -> Latency:
    """Средняя и 95-й перцентиль задержки на промахах за окно."""
    rows = conn.execute(
        "select latency_ms from query_log where ts >= ? and cache_hit = 0 order by latency_ms",
        (since,),
    ).fetchall()
    if not rows:
        return Latency(0, 0.0, 0)
    values = [int(row["latency_ms"]) for row in rows]
    # Индекс ближайшего ранга: при десяти значениях 95-й перцентиль — десятое,
    # а не «между девятым и десятым». Интерполяция на выборках такого размера
    # рисует число, которого в журнале не было.
    rank = min(len(values) - 1, int(len(values) * 0.95))
    return Latency(len(values), sum(values) / len(values), values[rank])


def top_spots(conn: sqlite3.Connection, *, since: float, limit: int = TOP) -> tuple[Spot, ...]:
    """Самые спрашиваемые точки за окно.

    По ним решают, что пиннить и предрассчитывать: популярную точку не надо
    ждать промаха, чтобы положить в кэш (docs/CACHE.md §3.4).
    """
    rows = conn.execute(
        """
        select area, count(*) as asked, sum(cache_hit) as hits
        from query_log
        where ts >= ? and area is not null
        group by area
        order by asked desc, area
        limit ?
        """,
        (since, limit),
    ).fetchall()
    return tuple(Spot(str(row["area"]), int(row["asked"]), int(row["hits"] or 0)) for row in rows)


def collect(conn: sqlite3.Connection, *, now: float, limit: int = TOP) -> Report:
    """Собрать отчёт целиком."""
    hour = now - HOUR_S
    day = now - DAY_S
    occupied = conn.execute(
        "select coalesce(sum(bytes), 0) as total, count(*) as objects from cache_index"
    ).fetchone()
    pinned = conn.execute(
        "select coalesce(sum(bytes), 0) as total from cache_index where pinned = 1"
    ).fetchone()
    return Report(
        occupied_bytes=int(occupied["total"]),
        pinned_bytes=int(pinned["total"]),
        objects=int(occupied["objects"]),
        hour=hit_rate(conn, since=hour),
        day=hit_rate(conn, since=day),
        point=hit_rate(conn, since=day, endpoint_like=POINT_LIKE),
        grid=hit_rate(conn, since=day, endpoint_like=GRID_LIKE),
        miss=miss_latency(conn, since=day),
        top=top_spots(conn, since=day, limit=limit),
    )


def render(report: Report) -> str:
    """Отчёт текстом. Читает человек, поэтому байты — в гигабайтах."""
    lines = [
        f"занято {report.occupied_bytes / 2**30:.2f} ГБ в {report.objects} объектах, "
        f"из них запиннено {report.pinned_bytes / 2**30:.2f} ГБ",
        f"hit rate за час: {_percent(report.hour)}",
        f"hit rate за сутки: {_percent(report.day)}",
        f"  точки: {_percent(report.point)}",
        f"  карты: {_percent(report.grid)}",
        f"промахи за сутки: {report.miss.misses}, "
        f"в среднем {report.miss.mean_ms:.0f} мс, 95-й перцентиль {report.miss.p95_ms} мс",
    ]
    if report.top:
        lines.append("чаще всего спрашивали:")
        lines += [
            f"  {spot.area}: {spot.asked} запросов, попаданий {spot.rate:.0%}"
            for spot in report.top
        ]
    else:
        lines.append("журнал за сутки пуст")
    # Порог — не украшение отчёта: ниже него кэш не выполняет свою работу, и
    # причина у этого одна из двух, а чинятся они по-разному.
    if report.day.poor:
        lines.append(
            f"hit rate ниже {POOR_HIT_RATE:.0%}: либо кэшируются запросы вместо чанков, "
            "либо чанки origin слишком крупные для наших запросов"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """`make cache-report` — статистика кэша для дежурного.

    Код возврата ненулевой при hit rate ниже порога: отчёт, который читают
    глазами, читают не каждый день, а расписание — каждый.
    """
    parser = argparse.ArgumentParser(prog="cache.report")
    parser.add_argument("index", type=Path, nargs="?", default=None, help="файл индекса SQLite")
    parser.add_argument("--top", type=int, default=TOP, help="сколько точек показывать")
    args = parser.parse_args(argv)

    path = args.index if args.index is not None else default_index()
    if not Path(path).exists():
        # Не создавать: `open_index` разложил бы пустую схему по неверному пути
        # и отчитался «занято 0 байт» вместо того, чтобы сказать про переменную.
        print(f"индекса нет: {path}")
        return 1

    conn = open_index(path)
    try:
        report = collect(conn, now=time.time(), limit=args.top)
    finally:
        conn.close()
    print(render(report))
    return 1 if report.day.poor else 0


def _percent(rate: Rate) -> str:
    """Доля словами. Пустое окно так и пишется — «нет запросов»: ноль процентов
    на пустом журнале выглядит как сломанный кэш."""
    if not rate.total:
        return "нет запросов"
    return f"{rate.rate:.0%} ({rate.hits} из {rate.total})"


if __name__ == "__main__":
    sys.exit(main())

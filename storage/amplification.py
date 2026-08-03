"""Воспроизводимый отчёт о раздувании чтения для BACKLOG 2.7.

Это расчёт по фактическим формам чанков, не оценка времени: сжатый файл читает
меньше физических байтов, но декодер всё равно восстанавливает каждый элемент
затронутого чанка. Wall-clock числа с настоящего диска хранятся отдельно в
``bench/results/read.json`` и не подменяются синтетическим таймером CI.
"""

from typing import Final, NamedTuple

from contracts import canon
from storage.layout import BYTES_PER_VALUE, LAYOUT_A, LAYOUT_B, Chunking, read_values


class Query(NamedTuple):
    name: str
    shape: tuple[int, int, int, int]


class Cost(NamedTuple):
    needed_bytes: int
    read_bytes: int
    amplification: float


QUERIES: Final = (
    Query("карта, один срок", (1, 1, *canon.GRID_SHAPE)),
    Query("карта, сутки", (4, 1, *canon.GRID_SHAPE)),
    Query("точка, один срок", (1, 1, 1, 1)),
    Query("точка, 10 суток", (40, 1, 1, 1)),
    Query("точка, год", (1460, 1, 1, 1)),
    Query("точка, 10 лет", (14600, 1, 1, 1)),
)


def cost(layout: Chunking, query: Query) -> Cost:
    """Логические нужные и распакованные байты одного float32-поля."""
    needed_values = _product(query.shape)
    decoded_values = read_values(layout, query.shape)
    return Cost(
        needed_bytes=needed_values * BYTES_PER_VALUE,
        read_bytes=decoded_values * BYTES_PER_VALUE,
        amplification=decoded_values / needed_values,
    )


def markdown() -> str:
    """Таблица, которую можно дословно сверить с ``report.md``."""
    lines = [
        "| Запрос одного поля | Нужно | Maps: распаковано / × | Series: распаковано / × |",
        "|---|---:|---:|---:|",
    ]
    for query in QUERIES:
        maps = cost(LAYOUT_A, query)
        series = cost(LAYOUT_B, query)
        lines.append(
            f"| {query.name} | {_bytes(maps.needed_bytes)} | "
            f"{_bytes(maps.read_bytes)} / {_factor(maps.amplification)} | "
            f"{_bytes(series.read_bytes)} / {_factor(series.amplification)} |"
        )
    return "\n".join(lines)


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _bytes(value: int) -> str:
    if value >= 10**9:
        return f"{value / 10**9:.2f} GB"
    if value >= 10**6:
        return f"{value / 10**6:.2f} MB"
    if value >= 10**3:
        return f"{value / 10**3:.2f} kB"
    return f"{value} B"


def _factor(value: float) -> str:
    return f"×{value:,.0f}".replace(",", " ")


def main() -> None:
    print(markdown())


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

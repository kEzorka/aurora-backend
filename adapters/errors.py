"""Отказ адаптера всегда называет поле, полученное значение и ожидаемое.

docs/DATA_CONTRACT.md §4: «Формулировка „validation failed“ без деталей — не
принимается на ревью». К адаптеру это относится в первую очередь: он читает
чужие файлы, и почти всякий его отказ — это разговор о том, что именно отдал
источник.
"""


class AdapterError(ValueError):
    """Источник отдал не то, что адаптер умеет привести к канону."""

    def __init__(self, field: str, got: object, expected: object) -> None:
        super().__init__(f"{field}: получено {got!r}, ожидалось {expected!r}")
        self.field = field
        self.got = got
        self.expected = expected

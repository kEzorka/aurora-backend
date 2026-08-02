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


class NotYetInSourceError(LookupError):
    """Данных за этот срок в источнике ещё нет.

    Не `AdapterError`: тот про «источник отдал не то», а это про «источник
    честно отдал всё, что у него есть». Реанализ отстаёт от сегодняшнего дня
    (docs/PIPELINE.md §1), и дата из этой слепой зоны — нормальный вопрос без
    ответа, а не сбой. Наверху из него делают `cache.proxy.NotYetError`, и
    делают это на стороне кэша: адаптеру про кэш знать не положено
    (`tests/test_boundaries.py`).
    """

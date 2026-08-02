"""Очередь CDS на подмену: почасовой CSV за спрошенный период.

Живёт рядом с фикстурой ARCO и по той же причине (`tests/fixtures/arco.py`):
один и тот же стуб нужен и тестам origin (`tests/cache/test_cds_origin.py`), и
тестам склейки истории (`tests/cache/test_history.py`). Две копии разошлись бы
на первой же правке разбора, и «работает через origin, но не через историю»
поймать было бы негде.

Отвечает ровно на тот период, который у него спросили, — поэтому обрезка чанка
по «сейчас» и пустой ответ проверяются на том же пути, каким поедет боевой
запрос, а не подстановкой готового текста.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

#: Москва, уже округлённая до узла сетки: тесты сравнивают с числом, а не
#: повторяют в себе `era5_cds.snap`.
MOSCOW = (55.75, 37.5)


class Service:
    """Ретривер CDS: почасовые строки за период запроса.

    Значение равно `250 + число часов от начала периода` — по нему видно, какой
    именно кусок ряда доехал до ответа, а какой обрезала обрезка периода.

    `available` — момент, дальше которого у сервиса данных нет. Отдельно за
    каждое поле (`Mapping` по имени переменной CDS) он нужен растущему чанку:
    ряд, взятый для `10u` утром, а для `10v` в полдень, разной длины по-честному,
    и склейка полей обязана это пережить.
    """

    def __init__(self, *, available: datetime | Mapping[str, datetime] | None = None) -> None:
        self.available = available
        self.requests: list[Mapping[str, Any]] = []

    def __call__(self, dataset: str, request: Mapping[str, Any]) -> str:
        self.requests.append(request)
        first, _, last = str(request["date"][0]).partition("/")
        start = datetime.fromisoformat(first).replace(tzinfo=UTC)
        end = datetime.fromisoformat(last).replace(hour=23, tzinfo=UTC)
        name = str(request["variable"][0])
        limit = self.available.get(name) if isinstance(self.available, Mapping) else self.available
        if limit is not None:
            end = min(end, limit)
        lines = [f"valid_time,latitude,longitude,{name}"]
        moment, hour = start, 0
        while moment <= end:
            lines.append(f"{moment:%Y-%m-%d %H:%M:%S},{MOSCOW[0]},{MOSCOW[1]},{250.0 + hour}")
            moment, hour = moment + timedelta(hours=1), hour + 1
        # Шапка без строк — это пустой ответ, а не обрезанный: `read_series`
        # обязан отличить «данных ещё нет» от «мы потеряли столбец».
        return "\n".join(lines) + "\n"

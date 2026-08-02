"""История для полосы 1: агрегаты, лимиты и вход в кэш (BACKLOG 5.5).

Разделение с `cache.history` проходит там же, где и у прогноза с `storage.read`:
внизу — данные под каноническими именами и в СИ, здесь — то, как их назвали
снаружи. Кэшу не нужны ни `t2m`, ни градусы Цельсия, ни `daily`; API не нужны
ни номера чанков, ни версия адаптера.

Порядок действий над рядом закреплён и переставлять его нельзя: сначала
склейка составляющих (`wind` — это `hypot(10u, 10v)`), потом перевод единиц,
и только потом усреднение. Средним первые два шага безразличны, а `min` и
`max` — нет: скорость ветра, посчитанная из суточных средних составляющих,
меньше настоящей, а множитель со знаком минус поменял бы местами минимум и
максимум. Множителей со знаком минус в `api.fields` сегодня нет, но порядок
здесь стоит не поэтому.

Чего тут нет: месячных средних по сетке. Их считает слой 2.5, и до него
`agg=monthly` для карты — отказ `400`, а не `413`: `413` означает «уменьшите
запрос и повторите», а уменьшать нечего.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from api.fields import Field
from cache import index as cache_index
from cache import origins, proxy
from contracts import canon

#: Агрегаты из docs/API_CONTRACT.md §2. `raw6h` — не «всё, что есть»: реанализ
#: почасовой, и отдавать час за час означало бы 8760 значений на год при
#: потолке в 500 шагов.
RAW: Final = "raw6h"
DAILY: Final = "daily"
MONTHLY: Final = "monthly"
AGGREGATIONS: Final[tuple[str, ...]] = (RAW, DAILY, MONTHLY)

#: Что считается по каждому полю в агрегате. Суточная средняя без минимума и
#: максимума скрывает ровно то, ради чего суточные данные и смотрят.
STATS: Final[tuple[str, ...]] = ("mean", "min", "max")

#: Потолок периода: 100 лет для точки, год для сетки (§3). Для сетки он тут не
#: нужен — карта отдаётся на один срок, — но записан рядом, чтобы при появлении
#: диапазона у карт его не пришлось выдумывать заново.
MAX_YEARS_POINT: Final = 100
MAX_YEARS_GRID: Final = 1

#: Холодных запросов к истории разом (§3). Считаются идущие наружу, а не все:
#: ответ из кэша не занимает места в очереди CDS.
COLD_LIMIT: Final = 4

#: `Retry-After` при `429`. Сравним с временем одного холодного запроса: раньше
#: возвращаться незачем, а позже — значит держать клиента дольше, чем нужно.
COLD_RETRY_SEC: Final = 30

#: TTL ответа истории (§5.5). Данные старше трёх месяцев неизменны — сутки;
#: предварительные перепишут задним числом, и час здесь не осторожность, а
#: обещание не отдавать вчерашнее число ещё сутки после его замены.
SETTLED_TTL_SEC: Final = 24 * 3600
PRELIMINARY_TTL_SEC: Final = 3600


class UnknownAggregationError(ValueError):
    """`agg`, которого в контракте нет."""


class TooManyColdError(RuntimeError):
    """Холодных запросов к истории больше, чем разрешено (§3)."""


class ColdGate:
    """Счётчик одновременных походов наружу.

    Считается вход, а не время ответа: очередь CDS общая на весь сервис, и пять
    запросов, стоящих в ней одновременно, замедляют друг друга, а не только
    себя. Пятый честнее развернуть с `429` и `Retry-After`, чем поставить в
    очередь, из которой он выйдет через минуты.

    Блокировка настоящая, а не «на всякий случай»: ручки полосы 1 синхронные,
    и FastAPI выполняет их в пуле потоков — счётчик без замка теряет
    инкременты ровно под той нагрузкой, ради которой заведён.
    """

    def __init__(self, limit: int = COLD_LIMIT) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._busy = 0

    @property
    def busy(self) -> int:
        return self._busy

    @contextmanager
    def hold(self, cold: bool) -> Iterator[None]:
        """Занять место в очереди, если запрос холодный. Иначе — пропустить."""
        if not cold:
            yield
            return
        with self._lock:
            if self._busy >= self._limit:
                raise TooManyColdError(
                    f"холодных запросов к истории разом: {self._busy} при пределе {self._limit}"
                )
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1


@dataclass(frozen=True)
class History:
    """Долгоживущая часть истории: источники, кэш и счётчик холодных запросов.

    Origin создаётся один на приложение, а не на запрос: `ArcoOrigin` держит
    открытый архив и переоткрывает его раз в час (`cache.origins.REFRESH_AFTER`),
    а созданный заново читает описание всех восьмидесяти лет на каждый запрос —
    то есть платит за метаданные больше, чем за данные.

    Соединение с индексом, наоборот, берётся на запрос: `sqlite3` привязывает
    его к потоку, в котором оно открыто, а синхронные ручки FastAPI выполняются
    в пуле потоков. Общее соединение падало бы `ProgrammingError` не сразу, а на
    втором потоке — то есть под нагрузкой.
    """

    root: Path
    index_path: Path
    maps: proxy.Origin
    series: proxy.Origin
    gate: ColdGate = field(default_factory=ColdGate)

    def connect(self) -> sqlite3.Connection:
        # Первый холодный запрос обязан работать на пустом корне. Сам
        # `open_index` каталог не создаёт намеренно: инструмент дежурного не
        # должен молча разложить новый индекс по ошибочному пути. Здесь путь —
        # часть уже собранного состояния приложения, поэтому неоднозначности нет.
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_index.open_index(self.index_path)


def from_env(
    root: str | Path | None = None,
    *,
    index_path: str | Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> History:
    """История над боевым кэшем: `$AURORA_ROOT/cache` и индекс рядом с ним."""
    base = Path(root) if root is not None else _default_root()
    clock = now if now is not None else (lambda: datetime.now(UTC))
    return History(
        root=base,
        index_path=Path(index_path) if index_path is not None else cache_index.default_index(),
        maps=origins.ArcoOrigin(clock=clock),
        series=origins.CdsOrigin(clock=clock),
    )


def _default_root() -> Path:
    base = Path(os.environ.get(cache_index.ROOT_ENV, cache_index.DEFAULT_ROOT)).expanduser()
    return base / "cache"


def check_aggregation(agg: str, *, grid: bool = False) -> str:
    """Проверить `agg` и вернуть его же.

    Месячные средние по карте — отдельный случай: слоя 2.5 нет, и посчитать их
    на лету значит прочитать 720 карт на один ответ. Это `400`, а не `413`.
    """
    if agg not in AGGREGATIONS:
        raise UnknownAggregationError(f"agg: получено {agg!r}, ожидалось одно из {AGGREGATIONS}")
    if grid and agg == MONTHLY:
        raise UnknownAggregationError(
            "agg=monthly для карты не поддерживается: месячных средних нет (BACKLOG 2.5)"
        )
    return agg


def span(moment: datetime, agg: str) -> tuple[datetime, datetime]:
    """Период, по которому усредняется карта на срок `moment`.

    Карта отдаётся одна (§1, `values` плоский), поэтому `agg` у сетки решает не
    разбиение ряда, а ширину окна: срок, сутки или месяц.
    """
    if agg == RAW:
        rounded = moment.replace(minute=0, second=0, microsecond=0)
        rounded -= timedelta(hours=rounded.hour % canon.STEP_HOURS)
        return rounded, rounded
    if agg == DAILY:
        day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return day, day + timedelta(hours=23)
    raise UnknownAggregationError(f"agg={agg!r} для карты не поддерживается")


def buckets(times: Sequence[str], agg: str) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Разбить сроки на группы: подписи и номера сроков в каждой.

    Подпись агрегата короче срока намеренно (`1990-01-01`, `1990-01`): суточное
    среднее, подписанное полуночью, читается как значение в полночь.
    """
    if agg == RAW:
        chosen = [
            position
            for position, stamp in enumerate(times)
            if int(stamp[11:13]) % canon.STEP_HOURS == 0
        ]
        return tuple(times[position] for position in chosen), tuple((p,) for p in chosen)
    width = 10 if agg == DAILY else 7
    if agg not in (DAILY, MONTHLY):
        raise UnknownAggregationError(f"agg: получено {agg!r}, ожидалось одно из {AGGREGATIONS}")
    labels: list[str] = []
    groups: list[list[int]] = []
    for position, stamp in enumerate(times):
        label = stamp[:width]
        if not labels or labels[-1] != label:
            labels.append(label)
            groups.append([])
        groups[-1].append(position)
    return tuple(labels), tuple(tuple(group) for group in groups)


def series(
    wanted: Sequence[Field],
    values: Mapping[str, Sequence[float | None]],
    times: Sequence[str],
    agg: str,
    units: str,
) -> tuple[tuple[str, ...], dict[str, list[float | None]]]:
    """Канонические ряды в ответ: подписи и `{имя_статистика: значения}`.

    У `raw6h` суффикса нет: `t2m_mean` за один срок — это то же значение под
    именем, обещающим усреднение.
    """
    labels, groups = buckets(times, agg)
    answer: dict[str, list[float | None]] = {}
    for item in wanted:
        # Сначала склейка и единицы, потом усреднение — см. docstring модуля.
        converted = item.values(values, units)
        digits = item.digits(units)
        if agg == RAW:
            answer[item.name] = [_round(converted[group[0]], digits) for group in groups]
            continue
        for stat in STATS:
            answer[f"{item.name}_{stat}"] = [
                _reduce(converted, group, stat, digits) for group in groups
            ]
    return labels, answer


def ttl(preliminary: bool) -> int:
    """Сколько ответ можно не перепроверять (§5.5)."""
    return PRELIMINARY_TTL_SEC if preliminary else SETTLED_TTL_SEC


def step_count(start: datetime, stop: datetime, agg: str) -> int:
    """Сколько меток выйдет после агрегации, не читая origin.

    Лимит проверяется до холодного запроса: скачать десятилетие из CDS, а
    затем сообщить, что 14 610 шестичасовых значений не помещаются в потолок
    500, — это не лимит, а дорогая ошибка после факта.
    """
    if agg == RAW:
        first = start.replace(minute=0, second=0, microsecond=0)
        first += timedelta(hours=(-first.hour) % canon.STEP_HOURS)
        if first < start:
            first += timedelta(hours=canon.STEP_HOURS)
        if first > stop:
            return 0
        return int((stop - first) // timedelta(hours=canon.STEP_HOURS)) + 1
    if agg == DAILY:
        return (stop.date() - start.date()).days + 1
    if agg == MONTHLY:
        return (stop.year - start.year) * 12 + stop.month - start.month + 1
    raise UnknownAggregationError(f"agg: получено {agg!r}, ожидалось одно из {AGGREGATIONS}")


def grid_shape(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Число узлов канонической сетки в `bbox`, до похода в ARCO."""
    south, west, north, east = bbox
    ny = sum(south <= float(lat) <= north for lat in canon.LAT)
    nx = sum(west <= float(lon) <= east for lon in canon.LON)
    return ny, nx


def _reduce(
    values: Sequence[float | None], group: Sequence[int], stat: str, digits: int
) -> float | None:
    """Статистика по группе. Пропуски выброшены, пустая группа — `null`.

    Выброшены, а не заменены нулём: суточное среднее из двадцати трёх часов
    честнее, чем из двадцати четырёх с нулём вместо пропуска, и заметно лучше,
    чем `null` на весь день из-за одного битого часа.
    """
    known = [values[position] for position in group if values[position] is not None]
    if not known:
        return None
    numbers = [value for value in known if value is not None]
    if stat == "min":
        return _round(min(numbers), digits)
    if stat == "max":
        return _round(max(numbers), digits)
    return _round(sum(numbers) / len(numbers), digits)


def _round(value: float | None, digits: int) -> float | None:
    """Округление под единицы (`api.fields.Field.digits`): `21.340000000000003`
    в ответе — это 18 байт на значение вместо пяти."""
    return None if value is None else round(value, digits)

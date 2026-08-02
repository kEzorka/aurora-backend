"""Имена и единицы полосы 1: что пользователь просит и в чём это отдаётся.

Единственное место, где живут `t2m`, `wind_speed`, градусы Цельсия и
гектопаскали. В хранилище всё в СИ и под каноническими именами — константа
273.15 в `storage/` или в `adapters/` это симптом (contracts/canon.py §UNITS).

Публичное имя совпадает с каноническим везде, кроме двух случаев из
docs/API_CONTRACT.md §2: `t2m` — потому что имя `2t` начинается с цифры и
контракт закрепил `t2m`, и `wind` — потому что скорости ветра в хранилище
нет вовсе, есть две составляющие.
"""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, NamedTuple

import numpy as np

from contracts import canon


class UnknownFieldError(ValueError):
    """Переменная, которой в контракте нет."""


class Field(NamedTuple):
    """Поле ответа: под каким именем отдаётся, из чего считается, в чём."""

    name: str
    inputs: tuple[str, ...]
    unit_si: str
    unit_human: str
    #: `human = si * scale + offset`. Линейного хватает: перевод единиц —
    #: это либо сдвиг нуля (кельвины), либо множитель (паскали, метры).
    scale: float = 1.0
    offset: float = 0.0
    combine: Callable[[Sequence[np.ndarray]], np.ndarray] | None = None

    def unit(self, units: str) -> str:
        return self.unit_human if units == "human" else self.unit_si

    def values(
        self, source: Mapping[str, Sequence[float | None]], units: str
    ) -> list[float | None]:
        """Собрать ряд из канонических рядов и перевести в запрошенные единицы.

        Пропуски остаются пропусками: `null` в одном из слагаемых даёт `null`
        в результате, а не тихий ноль (docs/API_CONTRACT.md §1).
        """
        arrays = [
            np.array([np.nan if v is None else v for v in source[name]]) for name in self.inputs
        ]
        combined = self.combine(arrays) if self.combine is not None else arrays[0]
        if units == "human":
            combined = combined * self.scale + self.offset
        return [None if np.isnan(value) else float(value) for value in combined]

    def digits(self, units: str) -> int:
        """Сколько знаков после запятой имеет смысл в этих единицах."""
        return _DIGITS.get(self.unit(units), DEFAULT_DIGITS)

    def compact(
        self, source: Mapping[str, Sequence[float | None]], units: str
    ) -> list[float | None]:
        """То же, что `values`, но округлённое под компактный формат сетки.

        `21.340000000000003` — это 18 байт на точку вместо пяти, то есть 290 КБ
        вместо 90 на окно в 16 000 точек (docs/API_CONTRACT.md §1). Знаки
        отсчитываются от единицы, а не от поля: 0.1 °C и 0.1 гПа осмысленны,
        0.1 Па — нет, и в СИ то же поле округляется иначе.
        """
        digits = self.digits(units)
        return [
            None if value is None else round(value, digits) for value in self.values(source, units)
        ]


def _speed(arrays: Sequence[np.ndarray]) -> np.ndarray:
    speed: np.ndarray = np.hypot(arrays[0], arrays[1])
    return speed


#: Ярлык той же величины, а не пересчёт: СИ пишет `m s-1`, человек — `м/с`.
_HUMAN_LABEL: Final[Mapping[str, str]] = MappingProxyType({"m s-1": "m/s", "1": "1"})

#: Перевод в человеческие единицы: ярлык, множитель, сдвиг.
_HUMAN: Final[Mapping[str, tuple[str, float, float]]] = MappingProxyType(
    {
        "2t": ("degC", 1.0, -273.15),
        "2d": ("degC", 1.0, -273.15),
        "skt": ("degC", 1.0, -273.15),
        "stl1": ("degC", 1.0, -273.15),
        "msl": ("hPa", 0.01, 0.0),
        "sp": ("hPa", 0.01, 0.0),
        # Осадки хранятся в метрах водного слоя, а показываются в миллиметрах:
        # 0.001 в ответе фронтенд покажет как ноль.
        "tp_1h": ("mm", 1000.0, 0.0),
        "sf_1h": ("mm", 1000.0, 0.0),
        "tp": ("mm", 1000.0, 0.0),
        "tcc": ("%", 100.0, 0.0),
        "lcc": ("%", 100.0, 0.0),
        "mcc": ("%", 100.0, 0.0),
        "hcc": ("%", 100.0, 0.0),
        "ci": ("%", 100.0, 0.0),
    }
)

#: Знаков после запятой в компактном формате сетки. Свойство единицы, а не
#: поля: 0.1 °C — это уже вдвое мельче ошибки модели, а 0.1 Па — шум, который
#: стоит трёх лишних байт на каждой из 16 000 точек.
_DIGITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "degC": 1,
        "K": 1,
        "hPa": 1,
        "Pa": 0,
        "m/s": 1,
        "m s-1": 1,
        "mm": 2,
        "%": 1,
    }
)

#: Для всего прочего — доли и метры водного эквивалента, где значащее начинается
#: с третьего знака.
DEFAULT_DIGITS: Final = 3

#: Публичные имена, которых в каноне нет.
_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"t2m": "2t", "d2m": "2d"})


def _plain(name: str, canonical: str) -> Field:
    si = canon.UNITS[canonical]
    human, scale, offset = _HUMAN.get(canonical, (_HUMAN_LABEL.get(si, si), 1.0, 0.0))
    return Field(
        name=name, inputs=(canonical,), unit_si=si, unit_human=human, scale=scale, offset=offset
    )


def _catalogue() -> dict[str, Field]:
    fields = {name: _plain(name, name) for name in canon.UNITS}
    fields.update({alias: _plain(alias, canonical) for alias, canonical in _ALIASES.items()})
    fields["wind"] = Field(
        name="wind_speed",
        inputs=("10u", "10v"),
        unit_si=canon.UNITS["10u"],
        unit_human="m/s",
        combine=_speed,
    )
    fields["wind_speed"] = fields["wind"]
    return fields


#: Что можно спросить у полосы 1. Ключ — имя в запросе, `Field.name` — ключ
#: в ответе: `vars=wind` возвращается как `wind_speed` (docs/API_CONTRACT.md §2).
FIELDS: Final[Mapping[str, Field]] = MappingProxyType(_catalogue())

#: `vars` по умолчанию — docs/API_CONTRACT.md §2.
DEFAULT_VARS: Final = "t2m,wind,msl"


def resolve(names: str | Sequence[str]) -> tuple[Field, ...]:
    """Список из запроса в поля ответа. Порядок сохраняется, повторы — нет."""
    wanted = names.split(",") if isinstance(names, str) else list(names)
    fields: list[Field] = []
    for raw in wanted:
        name = raw.strip()
        if not name:
            continue
        if name not in FIELDS:
            raise UnknownFieldError(f"vars: неизвестная переменная {name!r}")
        field = FIELDS[name]
        if field not in fields:
            fields.append(field)
    if not fields:
        raise UnknownFieldError("vars: список пуст")
    return tuple(fields)


def canonical_names(fields: Sequence[Field]) -> tuple[str, ...]:
    """Что придётся прочитать из хранилища: `wind` — это два поля, а не одно."""
    names: list[str] = []
    for field in fields:
        for name in field.inputs:
            if name not in names:
                names.append(name)
    return tuple(names)

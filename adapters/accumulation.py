"""Осадки за шаг: четвёртая ловушка docs/DOMAIN.md §6.

Правило «осадки накоплены от начала прогона, вычитай соседние шаги» верно для
ECMWF и **портит** поле GFS. ECMWF отдаёт `tp` со `stepRange` `0-6` и `0-12`,
GFS отдаёт `APCP` со `stepRange` `0-6` и `6-12`: у второго каждое сообщение
уже относится к своему интервалу, и разность даёт разность двух независимых
шестичасовок — местами отрицательную (см. `tests/fixtures/test_fixtures.py`).

Поэтому правило выводится не из имени источника, а из заголовка сообщения.
Это не «так аккуратнее», а необходимость: у GFS шаг накопления меняется
внутри одного прогона — до 120 ч сообщения идут парами `0-3`/`3-6`, дальше
`0-6`/`6-12`. Адаптер, зашивший «GFS не вычитаем», ошибётся ровно там, где
источник переключит правило.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from adapters.errors import AdapterError


@dataclass(frozen=True)
class StepRange:
    """Интервал накопления сообщения в часах от начала прогона."""

    start: int
    end: int

    @classmethod
    def of(cls, field: xr.DataArray) -> StepRange:
        for key in ("start_step", "end_step"):
            if key not in field.attrs:
                raise AdapterError(key, "missing", "attribute set by the adapter")
        return cls(int(field.attrs["start_step"]), int(field.attrs["end_step"]))

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


def interval_amount(previous: xr.DataArray, current: xr.DataArray) -> xr.DataArray:
    """Накопление за интервал `current` — из пары соседних сообщений.

    Два случая, и оба определяются началом интервала:

    * `0-6` и `0-12` — накопление от начала прогона, шаг равен разности;
    * `0-6` и `6-12` — накопление уже за интервал, брать как есть.

    Всё остальное — отказ. Пара сообщений, которая не является ни тем, ни
    другим (пропущенный шаг, перепутанный порядок, сообщения из разных
    прогонов), молча даёт правдоподобное, но неверное поле осадков.
    """
    before, after = StepRange.of(previous), StepRange.of(current)
    if after.end <= before.end:
        raise AdapterError("step_range", f"{before} then {after}", "strictly increasing endStep")
    _same_grid(previous, current)

    if after.start == before.start:
        # Вычитание идёт по массивам, а не по DataArray: у соседних шагов
        # разное `time`, и xarray, выравнивая операнды по общей координате,
        # вернул бы поле нулевого размера вместо осадков. Пустой результат
        # прошёл бы проверку «осадки неотрицательны» — там нечего проверять.
        amount = current.copy(data=current.values - previous.values)
    elif after.start == before.end:
        amount = current.copy()
    else:
        raise AdapterError(
            "step_range",
            f"{before} then {after}",
            f"either {before.start}-{after.end} (from the start of the run) "
            f"or {before.end}-{after.end} (over the interval)",
        )

    amount.attrs = {
        **current.attrs,
        "start_step": before.end,
        "step_range": f"{before.end}-{after.end}",
    }
    return amount


def _same_grid(previous: xr.DataArray, current: xr.DataArray) -> None:
    """Раз выравнивание отключено, совпадение сеток проверяется руками.

    Иначе вычитание полей с разных сеток либо упало бы на форме, либо — при
    случайно совпавшей форме — дало бы разность полей из разных точек мира.
    """
    if previous.dims != current.dims or previous.shape != current.shape:
        raise AdapterError(
            "shape",
            f"{previous.dims}{previous.shape}",
            f"{current.dims}{current.shape}",
        )
    for axis in ("lat", "lon", "level"):
        if axis in current.coords and not current[axis].equals(previous[axis]):
            raise AdapterError(axis, "differs between the messages", "identical axes")

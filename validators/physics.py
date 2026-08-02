"""Уровни «физика» и «здравый смысл» из docs/DATA_CONTRACT.md §4.

Эти две группы ловят то, что не ловится структурой: данные правильной формы
и с правильными подписями, но неверные по существу — °C вместо K, осадки,
вычтенные в обратном порядке, поле, распакованное в константу.

Про память. Валидируется настоящий срез: 69 полей, 40 шагов, 11.5 ГБ float32
(docs/PIPELINE.md §2). Ни одно поле здесь не материализуется целиком: все
проверки выражены редукциями xarray, которые считаются по чанкам и дают
скаляр; `.values` берётся только с этого скаляра. `.values` с самого поля
не просто медленный — он тянет в память 2.2 ГБ одной температуры на 40 шагов
и удваивает их приведением к float64.
"""

import numpy as np
import xarray as xr

from contracts import canon
from validators.result import Check, fail, ok

MAX_NAN_FRACTION = 0.001
MAX_STEP_JUMP_2T = 15.0
GLOBAL_MEAN_2T = (283.0, 292.0)


def check_physics(ds: xr.Dataset) -> list[Check]:
    checks: list[Check] = []
    for raw_name in ds.data_vars:
        name = str(raw_name)
        da = ds[name]
        checks.append(_nan_fraction(name, da))
        bounds = canon.PHYSICAL_RANGES.get(name)
        if bounds is not None:
            checks.append(_range(name, da, bounds))
    return checks


def check_sanity(ds: xr.Dataset) -> list[Check]:
    checks: list[Check] = []
    for raw_name in ds.data_vars:
        name = str(raw_name)
        checks.append(_not_constant(name, ds[name]))
    if "2t" in ds.data_vars:
        checks.append(_global_mean_2t(ds))
        if ds.sizes.get("time", 1) > 1:
            checks.append(_step_jump_2t(ds))
    return checks


def _extremes(da: xr.DataArray) -> tuple[float, float] | None:
    """(min, max) по конечным значениям или None, если конечных нет.

    min/max в xarray пропускают NaN сами; отдельная маска потребовала бы
    копии массива, а значения бесконечностей ловит проверка диапазона.
    """
    lo = float(da.min().values)
    hi = float(da.max().values)
    if np.isnan(lo) or np.isnan(hi):
        return None
    return lo, hi


def _nan_fraction(name: str, da: xr.DataArray) -> Check:
    # Не isnull(): бесконечность — тоже дыра в поле, и у переменной без
    # объявленного диапазона её больше некому поймать. np.isfinite на
    # DataArray остаётся ленивым и считается по чанкам.
    fraction = 1.0 - float(xr.ufuncs.isfinite(da).mean().values)
    if fraction > MAX_NAN_FRACTION:
        return fail("nan_fraction", "physics", name, round(fraction, 6), f"<= {MAX_NAN_FRACTION}")
    return ok("nan_fraction", "physics", f"{name}: {fraction:.6f}")


def _range(name: str, da: xr.DataArray, bounds: tuple[float, float]) -> Check:
    extremes = _extremes(da)
    if extremes is None:
        return fail("range", "physics", name, "all values are NaN", bounds)
    lo, hi = extremes
    if lo < bounds[0] or hi > bounds[1]:
        return fail("range", "physics", name, (round(lo, 4), round(hi, 4)), bounds)
    return ok("range", "physics", f"{name}: [{lo:.4g}, {hi:.4g}] within {bounds}")


def _not_constant(name: str, da: xr.DataArray) -> Check:
    """Константное поле — обычная форма молчаливой поломки: неверная распаковка
    scale/offset или заполнение одним значением при ошибке чтения."""
    extremes = _extremes(da)
    if extremes is not None and extremes[0] == extremes[1]:
        return fail("not_constant", "sanity", name, f"constant {extremes[0]}", "a varying field")
    return ok("not_constant", "sanity", name)


def _global_mean_2t(ds: xr.Dataset) -> Check:
    """Средняя приземная температура Земли ~288 K. Около 15 — это °C."""
    weights = np.cos(np.deg2rad(np.asarray(ds["lat"].values)))
    mean = float(ds["2t"].weighted(xr.DataArray(weights, dims="lat")).mean().values)
    if not GLOBAL_MEAN_2T[0] <= mean <= GLOBAL_MEAN_2T[1]:
        return fail("global_mean_2t", "sanity", "2t", round(mean, 3), GLOBAL_MEAN_2T)
    return ok("global_mean_2t", "sanity", f"{mean:.2f} K")


def _step_jump_2t(ds: xr.Dataset) -> Check:
    """Разница соседних шагов больше 15 K в точке — признак перепутанных шагов."""
    largest = float(abs(ds["2t"].diff("time")).max().values)
    if largest > MAX_STEP_JUMP_2T:
        return fail(
            "step_jump_2t", "sanity", "2t", round(largest, 3), f"<= {MAX_STEP_JUMP_2T} K per 6 h"
        )
    return ok("step_jump_2t", "sanity", f"max jump {largest:.2f} K")

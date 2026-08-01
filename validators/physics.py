"""Уровни «физика» и «здравый смысл» из docs/DATA_CONTRACT.md §4.

Эти две группы ловят то, что не ловится структурой: данные правильной формы
и с правильными подписями, но неверные по существу — °C вместо K, осадки,
вычтенные в обратном порядке, поле, распакованное в константу.
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
        values = np.asarray(ds[name].values, dtype="float64")
        checks.append(_nan_fraction(name, values))
        bounds = canon.PHYSICAL_RANGES.get(name)
        if bounds is not None:
            checks.append(_range(name, values, bounds))
    return checks


def check_sanity(ds: xr.Dataset) -> list[Check]:
    checks: list[Check] = []
    for raw_name in ds.data_vars:
        name = str(raw_name)
        checks.append(_not_constant(name, np.asarray(ds[name].values, dtype="float64")))
    if "2t" in ds.data_vars:
        checks.append(_global_mean_2t(ds))
        if ds.sizes.get("time", 1) > 1:
            checks.append(_step_jump_2t(ds))
    return checks


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _nan_fraction(name: str, values: np.ndarray) -> Check:
    fraction = float(np.count_nonzero(~np.isfinite(values)) / values.size)
    if fraction > MAX_NAN_FRACTION:
        return fail("nan_fraction", "physics", name, round(fraction, 6), f"<= {MAX_NAN_FRACTION}")
    return ok("nan_fraction", "physics", f"{name}: {fraction:.6f}")


def _range(name: str, values: np.ndarray, bounds: tuple[float, float]) -> Check:
    finite = _finite(values)
    if finite.size == 0:
        return fail("range", "physics", name, "all values are NaN", bounds)
    lo, hi = float(finite.min()), float(finite.max())
    if lo < bounds[0] or hi > bounds[1]:
        return fail("range", "physics", name, (round(lo, 4), round(hi, 4)), bounds)
    return ok("range", "physics", f"{name}: [{lo:.4g}, {hi:.4g}] within {bounds}")


def _not_constant(name: str, values: np.ndarray) -> Check:
    """Константное поле — обычная форма молчаливой поломки: неверная распаковка
    scale/offset или заполнение одним значением при ошибке чтения."""
    finite = _finite(values)
    if finite.size and float(finite.min()) == float(finite.max()):
        return fail(
            "not_constant", "sanity", name, f"constant {float(finite.min())}", "a varying field"
        )
    return ok("not_constant", "sanity", name)


def _global_mean_2t(ds: xr.Dataset) -> Check:
    """Средняя приземная температура Земли ~288 K. Около 15 — это °C."""
    weights = np.cos(np.deg2rad(np.asarray(ds["lat"].values)))
    mean = float(ds["2t"].weighted(xr.DataArray(weights, dims="lat")).mean().item())
    if not GLOBAL_MEAN_2T[0] <= mean <= GLOBAL_MEAN_2T[1]:
        return fail("global_mean_2t", "sanity", "2t", round(mean, 3), GLOBAL_MEAN_2T)
    return ok("global_mean_2t", "sanity", f"{mean:.2f} K")


def _step_jump_2t(ds: xr.Dataset) -> Check:
    """Разница соседних шагов больше 15 K в точке — признак перепутанных шагов."""
    diff = np.abs(np.diff(np.asarray(ds["2t"].values, dtype="float64"), axis=0))
    largest = float(np.nanmax(diff))
    if largest > MAX_STEP_JUMP_2T:
        return fail(
            "step_jump_2t", "sanity", "2t", round(largest, 3), f"<= {MAX_STEP_JUMP_2T} K per 6 h"
        )
    return ok("step_jump_2t", "sanity", f"max jump {largest:.2f} K")

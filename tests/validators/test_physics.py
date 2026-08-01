"""Диапазоны и здравый смысл: данные верной формы и с верными подписями,
но неверные по существу."""

import numpy as np
import xarray as xr

from contracts import canon
from tests.helpers import canonical_dataset, varying_field
from validators.physics import check_physics, check_sanity


def _with_varying_2t(base: float = 288.0, times: int = 1) -> xr.Dataset:
    ds = canonical_dataset(times=times)
    ny, nx = canon.GRID_SHAPE
    field = base + varying_field((times, ny, nx)) * 0.5
    return ds.assign({"2t": (("time", "lat", "lon"), field.astype(np.float32))})


def test_plausible_fields_pass_physics_and_sanity() -> None:
    ds = _with_varying_2t()
    assert [c.name for c in check_physics(ds) if not c.passed] == []
    assert [c.message for c in check_sanity(ds) if not c.passed] == []


def test_temperature_in_celsius_is_caught_by_the_range_check() -> None:
    ds = _with_varying_2t(base=15.0)
    failure = next(c for c in check_physics(ds) if c.name == "range" and not c.passed)
    assert failure.details["field"] == "2t"
    assert failure.details["expected"] == (180.0, 340.0)


def test_global_mean_catches_celsius_even_within_the_wide_range() -> None:
    """Глобальное среднее около 15 — это °C, а не K. docs/DATA_CONTRACT.md §4."""
    ds = _with_varying_2t(base=185.0)
    failure = next(c for c in check_sanity(ds) if c.name == "global_mean_2t" and not c.passed)
    assert failure.details["expected"] == (283.0, 292.0)


def test_constant_field_is_rejected() -> None:
    """Распаковка scale/offset мимо — и поле превращается в одно число."""
    ds = canonical_dataset()
    ny, nx = canon.GRID_SHAPE
    flat = np.broadcast_to(np.float32(288.0), (1, ny, nx))
    ds = ds.assign({"2t": (("time", "lat", "lon"), flat)})
    failure = next(c for c in check_sanity(ds) if c.name == "not_constant" and not c.passed)
    assert failure.details["field"] == "2t"


def test_step_to_step_jump_over_15_kelvin_is_rejected() -> None:
    ds = _with_varying_2t(times=2)
    values = ds["2t"].values.copy()
    values[1] = values[0] + 40.0
    ds = ds.assign({"2t": (("time", "lat", "lon"), values)})
    failure = next(c for c in check_sanity(ds) if c.name == "step_jump_2t" and not c.passed)
    assert "40" in failure.message


def test_too_many_nans_are_rejected() -> None:
    ds = _with_varying_2t()
    values = ds["2t"].values.copy()
    values[:, :100, :] = np.nan
    ds = ds.assign({"2t": (("time", "lat", "lon"), values)})
    failure = next(c for c in check_physics(ds) if c.name == "nan_fraction" and not c.passed)
    assert failure.details["field"] == "2t"


def test_negative_interval_precipitation_is_rejected() -> None:
    """Четвёртая ловушка: осадки вычли в неверном порядке."""
    ds = _with_varying_2t()
    ny, nx = canon.GRID_SHAPE
    tp = np.full((1, ny, nx), -0.001, dtype=np.float32)
    ds = ds.assign(tp=(("time", "lat", "lon"), tp))
    ds["tp"].attrs["units"] = "m"
    failure = next(c for c in check_physics(ds) if c.name == "range" and not c.passed)
    assert failure.details["field"] == "tp"

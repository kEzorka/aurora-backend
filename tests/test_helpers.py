"""Помощник обязан давать настоящую сетку и не отъедать под неё 286 МБ."""

import numpy as np
import xarray as xr

from contracts import canon
from tests.helpers import canonical_dataset, plausible_dataset, varying_field


def test_helper_builds_the_real_grid() -> None:
    ds = canonical_dataset()
    assert ds["2t"].shape == (1, 721, 1440)
    assert ds["t"].shape == (1, 13, 721, 1440)


def test_helper_does_not_allocate_the_grid() -> None:
    ds = canonical_dataset()
    values = ds["t"].variable.data
    assert values.base is not None, "поле материализовано, а должно быть broadcast-видом"
    assert values.strides[-1] == 0


def test_helper_fields_are_not_constant() -> None:
    """not_constant — законная проверка; помощник обязан её проходить."""
    ds = canonical_dataset()
    for name in ("2t", "msl", "t", "q"):
        values = ds[name].values
        assert values.min() < values.max(), name


def test_helper_2t_keeps_the_global_mean_at_288_kelvin() -> None:
    ds = canonical_dataset()
    weights = np.cos(np.deg2rad(ds["lat"].values))
    mean = float(ds["2t"].weighted(xr.DataArray(weights, dims="lat")).mean().item())
    assert 287.0 < mean < 289.0


def test_helper_dataset_carries_all_69_fields_and_units() -> None:
    ds = canonical_dataset()
    assert set(canon.SURFACE_VARS) | set(canon.ATMOS_VARS) == {str(v) for v in ds.data_vars}
    assert ds["2t"].attrs["units"] == "K"
    assert ds["msl"].attrs["units"] == "Pa"


def test_helper_coordinates_are_canonical() -> None:
    ds = canonical_dataset()
    assert np.array_equal(ds["lat"].values, canon.LAT)
    assert np.array_equal(ds["lon"].values, canon.LON)
    assert tuple(int(v) for v in ds["level"].values) == canon.PRESSURE_LEVELS


def test_time_axis_steps_by_six_hours() -> None:
    ds = canonical_dataset(times=3)
    deltas = np.diff(ds["time"].values).astype("timedelta64[h]").astype(int)
    assert deltas.tolist() == [6, 6]


def test_varying_field_is_not_constant() -> None:
    field = varying_field((2, 4, 4))
    assert field.min() < field.max()


def test_plausible_dataset_has_a_varying_temperature_around_288_kelvin() -> None:
    ds = plausible_dataset()
    values = ds["2t"].values
    assert values.min() < values.max()
    assert 287.0 < float(values.mean()) < 289.0
    assert ds["2t"].attrs["units"] == "K"

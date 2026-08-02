"""Месячные средние ERA5: агрегация, две раскладки и публикация (2.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from contracts import canon
from storage.monthly import (
    MAPS_LAYER,
    MONTHLY_VARS,
    SERIES_LAYER,
    aggregate,
    current,
    grid_window,
    logical_bytes,
    point_series,
    publish,
)


def _hourly(*, complete: bool = True) -> xr.Dataset:
    stop = np.datetime64("2020-03-01T00") if complete else np.datetime64("2020-02-29T23")
    times = np.arange(np.datetime64("2020-01-01T00"), stop, np.timedelta64(1, "h")).astype(
        "datetime64[ns]"
    )
    lat = np.array([1.0, 0.0, -1.0])
    lon = np.array([-1.0, 0.0, 1.0, 2.0])
    month = np.array([1.0 if str(stamp)[5:7] == "01" else 3.0 for stamp in times])
    data = {}
    for offset, name in enumerate(MONTHLY_VARS):
        samples = 101_000.0 + month * 10.0 if name == "msl" else month + offset
        values = np.broadcast_to(samples[:, None, None], (times.size, 3, 4)).copy()
        data[name] = (("time", "lat", "lon"), values.astype(np.float32))
    ds = xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon})
    for name in MONTHLY_VARS:
        ds[name].attrs["units"] = canon.UNITS[name]
    return ds


@pytest.fixture
def published(tmp_path: Path) -> Path:
    publish(aggregate(_hourly()), tmp_path, version="2020-02", require_canonical_grid=False)
    return tmp_path


def test_complete_calendar_months_are_averaged_and_labelled_by_their_start() -> None:
    monthly = aggregate(_hourly())

    assert tuple(monthly.data_vars) == MONTHLY_VARS
    assert np.array_equal(
        monthly["time"].values,
        np.array(["2020-01-01T00:00:00", "2020-02-01T00:00:00"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(monthly["2t"].isel(lat=0, lon=0), [1.0, 3.0])
    np.testing.assert_allclose(monthly["msl"].isel(lat=0, lon=0), [101_010.0, 101_030.0])
    assert monthly["2t"].dtype == np.float16
    assert monthly["msl"].dtype == np.float32
    assert monthly.attrs["source"] == "era5-final"
    assert monthly["2t"].attrs["units"] == "K"


def test_a_partial_last_month_is_refused_before_it_can_be_published() -> None:
    with pytest.raises(ValueError, match="последний месяц неполон"):
        aggregate(_hourly(complete=False))


def test_a_gap_in_the_hourly_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="непрерывная почасовая"):
        aggregate(_hourly().isel(time=[*range(10), *range(11, 100)]))


def test_an_hourly_axis_shifted_by_half_an_hour_is_refused() -> None:
    shifted = _hourly().assign_coords(time=_hourly()["time"] + np.timedelta64(30, "m"))

    with pytest.raises(ValueError, match="границе часа"):
        aggregate(shifted)


def test_publication_refuses_a_reversed_latitude_axis(tmp_path: Path) -> None:
    monthly = aggregate(_hourly()).sortby("lat")

    with pytest.raises(ValueError, match="убывающая"):
        publish(monthly, tmp_path, version="bad", require_canonical_grid=False)


def test_publication_switches_one_pointer_only_after_both_layouts_exist(
    published: Path,
) -> None:
    run = current(published)

    assert run is not None
    assert (published / "history" / "monthly").is_symlink()
    assert (run / MAPS_LAYER).is_dir() and (run / SERIES_LAYER).is_dir()
    assert zarr.open_array(str(run / MAPS_LAYER / "2t"), mode="r").chunks == (1, 3, 4)
    assert zarr.open_array(str(run / SERIES_LAYER / "2t"), mode="r").chunks == (2, 3, 4)


def test_publishing_the_same_version_twice_is_refused(published: Path) -> None:
    with pytest.raises(FileExistsError, match="2020-02"):
        publish(
            aggregate(_hourly()),
            published,
            version="2020-02",
            require_canonical_grid=False,
        )


def test_a_point_reads_the_series_layout(published: Path) -> None:
    result = point_series(
        published,
        ["2t", "msl"],
        0.1,
        0.1,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 2, 1, tzinfo=UTC),
    )

    assert (result.lat, result.lon) == (0.0, 0.0)
    assert result.times == ("2020-01-01T00:00:00Z", "2020-02-01T00:00:00Z")
    assert result.values == {"2t": [1.0, 3.0], "msl": [101_010.0, 101_030.0]}


def test_a_map_reads_the_maps_layout_in_compact_order(published: Path) -> None:
    result = grid_window(
        published,
        ["2t"],
        (-1.0, -1.0, 1.0, 2.0),
        datetime(2020, 2, 15, tzinfo=UTC),
        stride=2,
    )

    assert result.time == "2020-02-01T00:00:00Z"
    assert result.shape == (2, 2)
    assert (result.lat0, result.dlat) == (1.0, -2.0)
    assert (result.lon0, result.dlon) == (-1.0, 2.0)
    assert result.values["2t"] == [3.0] * 4


def test_the_uncompressed_dual_layout_is_about_twenty_one_gigabytes() -> None:
    size = logical_bytes(86 * 12)

    assert 21 * 10**9 < size < 22 * 10**9

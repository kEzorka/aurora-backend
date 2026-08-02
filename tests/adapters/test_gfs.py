"""Адаптер GFS на настоящих файлах из `tests/fixtures/grib/`."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")

import xarray as xr

from adapters import gfs
from contracts import canon

GRIB = Path(__file__).resolve().parents[1] / "fixtures" / "grib"
RETRIEVED = "2026-08-01T07:41:12Z"


def _read(name: str) -> xr.Dataset:
    return gfs.read_message(GRIB / name, source_url=f"test://{name}", retrieved_at=RETRIEVED)


def test_temperature_message_becomes_canonical() -> None:
    ds = _read("gfs_t2m.grib2")
    assert list(ds.data_vars) == ["2t"]
    assert ds["2t"].dims == ("time", "lat", "lon")
    assert ds["2t"].dtype == np.float32
    assert ds["2t"].attrs["units"] == "K"


def test_longitude_arrives_at_the_canonical_axis() -> None:
    """Исходный файл идёт `0..359.75` — это проверено в `tests/fixtures`."""
    ds = _read("gfs_t2m.grib2")
    assert np.array_equal(ds["lon"].values, canon.LON)
    assert np.array_equal(ds["lat"].values, canon.LAT)


def test_greenwich_stays_where_it_was() -> None:
    """Диапазон оси совпал бы и у поля, сдвинутого на полглобуса. Поэтому
    сравниваются значения в точке: нулевой меридиан в обоих представлениях
    один и тот же, а 359.75 обязано оказаться в -0.25."""
    with xr.open_dataset(
        GRIB / "gfs_t2m.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as raw:
        at_zero = float(raw["t2m"].sel(latitude=55.75, longitude=0.0))
        at_last = float(raw["t2m"].sel(latitude=55.75, longitude=359.75))
    ds = _read("gfs_t2m.grib2")
    assert float(ds["2t"].sel(lat=55.75, lon=0.0).isel(time=0)) == at_zero
    assert float(ds["2t"].sel(lat=55.75, lon=-0.25).isel(time=0)) == at_last


def test_time_is_the_valid_time_of_the_message() -> None:
    """Ловушка 1: файл — шаг +6 ч прогона 2026-07-31 00 UTC."""
    ds = _read("gfs_t2m.grib2")
    assert ds["time"].values.tolist() == [np.datetime64("2026-07-31T06:00:00", "ns").tolist()]
    assert ds.attrs["init_time"] == "2026-07-31T00:00:00Z"


def test_precipitation_is_converted_from_millimetres_to_metres() -> None:
    """Ловушка 3: GFS отдаёт `kg m-2`, канон — метры."""
    with xr.open_dataset(
        GRIB / "gfs_apcp_f006.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as raw:
        peak_mm = float(raw["tp"].max())
    ds = _read("gfs_apcp_f006.grib2")
    assert ds["tp"].attrs["units"] == "m"
    assert float(ds["tp"].max()) == pytest.approx(peak_mm * 1e-3, rel=1e-6)
    assert float(ds["tp"].min()) >= 0.0


def test_provenance_says_what_it_is() -> None:
    ds = _read("gfs_t2m.grib2")
    assert ds.attrs["source"] == "gfs-analysis"
    assert ds.attrs["grid"] == canon.GRID_NAME
    assert ds.attrs["retrieved_at"] == RETRIEVED
    assert ds.attrs["adapter_version"] == gfs.ADAPTER_VERSION


def test_reading_a_message_leaves_no_index_beside_the_fixture() -> None:
    """Сырой слой неизменяем (docs/PIPELINE.md §3), а `.idx` в репозитории —
    это изменение дерева тестом."""
    _read("gfs_t2m.grib2")
    assert not list(GRIB.glob("*.idx"))

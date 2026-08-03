"""Адаптер ECMWF Open Data на настоящих файлах из `tests/fixtures/grib/`."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")

import xarray as xr

from adapters import ecmwf
from adapters.fetch import write_messages
from contracts import canon

GRIB = Path(__file__).resolve().parents[1] / "fixtures" / "grib"
RETRIEVED = "2026-08-01T07:41:12Z"


def _read(name: str) -> xr.Dataset:
    return ecmwf.read_message(GRIB / name, source_url=f"test://{name}", retrieved_at=RETRIEVED)


def test_surface_message_becomes_canonical() -> None:
    ds = _read("ecmwf_2t_6h.grib2")
    assert list(ds.data_vars) == ["2t"]
    assert ds["2t"].dims == ("time", "lat", "lon")
    assert ds["2t"].attrs["units"] == "K"
    assert np.array_equal(ds["lon"].values, canon.LON)
    assert np.array_equal(ds["lat"].values, canon.LAT)


def test_axis_already_canonical_is_left_alone() -> None:
    """У ECMWF долгота приходит `-180..179.75`. Перекладка обязана быть
    тождественной, а не «на всякий случай» сдвинуть поле ещё раз."""
    with xr.open_dataset(
        GRIB / "ecmwf_2t_6h.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as raw:
        at_zero = float(raw["t2m"].sel(latitude=55.75, longitude=0.0))
    ds = _read("ecmwf_2t_6h.grib2")
    assert float(ds["2t"].sel(lat=55.75, lon=0.0).isel(time=0)) == at_zero


def test_pressure_level_message_keeps_the_level_as_an_axis() -> None:
    """Скалярный уровень не даёт склеить тринадцать сообщений в поле на
    уровнях, не полагаясь на порядок файлов."""
    ds = _read("ecmwf_t850_6h.grib2")
    assert ds["t"].dims == ("time", "level", "lat", "lon")
    assert ds["level"].values.tolist() == [850]
    assert int(ds["level"].dtype.itemsize) == 4


def test_time_is_the_valid_time_of_the_message() -> None:
    ds = _read("ecmwf_2t_6h.grib2")
    assert ds["time"].values.tolist() == [np.datetime64("2026-07-31T06:00:00", "ns").tolist()]
    assert ds.attrs["init_time"] == "2026-07-31T00:00:00Z"


def test_precipitation_stays_in_metres() -> None:
    ds = _read("ecmwf_tp_6h.grib2")
    assert ds["tp"].attrs["units"] == "m"
    assert float(ds["tp"].min()) >= 0.0


def test_provenance_says_what_it_is() -> None:
    ds = _read("ecmwf_2t_6h.grib2")
    assert ds.attrs["source"] == "ifs-analysis"
    assert ds.attrs["kind"] == "analysis"
    assert ds.attrs["adapter_version"] == ecmwf.ADAPTER_VERSION


def test_a_multi_message_download_keeps_every_compatible_group(tmp_path: Path) -> None:
    combined = write_messages(
        tmp_path / "analysis.grib2",
        (
            (GRIB / "ecmwf_2t_6h.grib2").read_bytes(),
            (GRIB / "ecmwf_t850_6h.grib2").read_bytes(),
        ),
    )

    ds = ecmwf.read_messages(combined, source_url="test://combined", retrieved_at=RETRIEVED)

    assert set(ds.data_vars) == {"2t", "t"}
    assert ds["2t"].dims == ("time", "lat", "lon")
    assert ds["t"].dims == ("time", "level", "lat", "lon")
    assert ds["t"]["level"].values.tolist() == [850]

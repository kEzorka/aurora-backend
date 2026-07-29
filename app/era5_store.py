"""Random access to the ERA5 archive.

Two layouts are supported and both answer the same questions, so nothing
downstream knows which one it is reading:

* **zarr** (`~/data/global.zarr`, written by `scripts/to_zarr.py`) — one store,
  one time axis, one chunk per (timestamp, level) field. This is the format to
  serve from: a read touches exactly the two timestamps it needs.
* **NetCDF** (`~/data/global/<month>/{surface,pressure}.nc`) — the raw CDS
  download, kept working as a fallback and as something to check zarr against.

The NetCDF layout is what forces the indexing to exist at all: a forecast
initialised at the first timestamp of a month needs the previous timestamp,
which lives in the previous month's file. So we map every timestamp to
(surface source, pressure source, position) across the whole archive and read
6-hourly slices on demand. For zarr both sources are the same store and the
month boundary stops being a special case.

We deliberately never concatenate into memory: one month of pressure-level data
is ~32 GB uncompressed and a rollout only ever needs two timestamps.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

# Names as they appear in the CDS files -> names Aurora expects.
SURF_RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl"}
ATMOS_VARS = ("z", "q", "t", "u", "v")
STATIC_VARS = ("z", "lsm", "slt")

TIME_COORD = "valid_time"


def is_zarr(path: Path) -> bool:
    """A zarr store is a directory carrying group metadata (v2 or v3)."""
    return path.is_dir() and (
        (path / ".zgroup").exists() or (path / "zarr.json").exists()
    )


class ERA5Store:
    def __init__(self, data_root: Path | None = None, static_file: Path | None = None):
        self.data_root = Path(data_root or config.DATA_ROOT)
        self.static_file = Path(static_file or config.STATIC_FILE)

        # timestamp -> (surface path, pressure path, index within those files)
        self._index: dict[dt.datetime, tuple[Path, Path, int]] = {}
        self._open: dict[Path, xr.Dataset] = {}

        self._build_index()
        self._read_grid()

    # ------------------------------------------------------------------ setup

    def _build_index(self) -> None:
        if is_zarr(self.data_root):
            self._build_index_zarr()
        else:
            self._build_index_netcdf()

    def _build_index_zarr(self) -> None:
        """One store, one time axis: surface and pressure sources coincide."""
        ds = self._dataset(self.data_root)
        missing = [v for v in (*SURF_RENAME, *ATMOS_VARS) if v not in ds]
        if missing:
            raise ValueError(f"{self.data_root} is missing variables {missing}")
        for i, t in enumerate(pd.to_datetime(ds[TIME_COORD].values)):
            self._index[t.to_pydatetime()] = (self.data_root, self.data_root, i)

    def _build_index_netcdf(self) -> None:
        month_dirs = sorted(p for p in self.data_root.iterdir() if p.is_dir())
        if not month_dirs:
            raise FileNotFoundError(f"no month directories under {self.data_root}")

        for month in month_dirs:
            surf, press = month / "surface.nc", month / "pressure.nc"
            if not (surf.exists() and press.exists()):
                continue

            st = self._times(surf)
            pt = self._times(press)
            if not np.array_equal(st, pt):
                raise ValueError(f"{month}: surface and pressure timestamps differ")

            for i, t in enumerate(st):
                self._index[t] = (surf, press, i)

        if not self._index:
            raise FileNotFoundError(f"no usable surface/pressure pairs under {self.data_root}")

    def _times(self, path: Path) -> list[dt.datetime]:
        ds = self._dataset(path)
        return [t.to_pydatetime() for t in pd.to_datetime(ds[TIME_COORD].values)]

    def _read_grid(self) -> None:
        any_surf = next(iter(self._index.values()))[0]
        ds = self._dataset(any_surf)
        self.lat = ds["latitude"].values.astype("float32")
        self.lon = ds["longitude"].values.astype("float32")

        any_press = next(iter(self._index.values()))[1]
        raw_levels = self._dataset(any_press)["pressure_level"].values

        # The files store levels descending (1000 -> 50); Aurora's own examples
        # use ascending levels. Keep one canonical ascending order and remember
        # the permutation that gets us there.
        self._level_order = np.argsort(raw_levels)
        self.levels = tuple(int(v) for v in raw_levels[self._level_order])

    def _dataset(self, path: Path) -> xr.Dataset:
        if path not in self._open:
            if is_zarr(path):
                # chunks=None keeps this lazy without pulling dask in: xarray
                # reads a slice straight out of the zarr array. dask is only
                # needed to write the store, not to serve from it.
                self._open[path] = xr.open_dataset(path, engine="zarr", chunks=None)
            else:
                self._open[path] = xr.open_dataset(path, engine="netcdf4")
        return self._open[path]

    # ------------------------------------------------------------------ query

    @property
    def timestamps(self) -> list[dt.datetime]:
        return sorted(self._index)

    def has(self, when: dt.datetime) -> bool:
        return when in self._index

    def surface_slice(self, when: dt.datetime) -> dict[str, np.ndarray]:
        """{aurora name: (lat, lon)} at a single timestamp."""
        surf, _, i = self._require(when)
        ds = self._dataset(surf)
        return {
            aurora: ds[cds].isel({TIME_COORD: i}).values.astype("float32")
            for cds, aurora in SURF_RENAME.items()
        }

    def atmos_slice(self, when: dt.datetime) -> dict[str, np.ndarray]:
        """{name: (level, lat, lon)} at a single timestamp, levels ascending."""
        _, press, i = self._require(when)
        ds = self._dataset(press)
        return {
            name: ds[name].isel({TIME_COORD: i}).values.astype("float32")[self._level_order]
            for name in ATMOS_VARS
        }

    def static_fields(self) -> dict[str, np.ndarray]:
        """{name: (lat, lon)} — the leading length-1 time axis is squeezed away."""
        ds = self._dataset(self.static_file)
        out = {}
        for name in STATIC_VARS:
            arr = ds[name].values.astype("float32")
            while arr.ndim > 2:
                arr = arr[0]
            out[name] = arr
        return out

    def _require(self, when: dt.datetime) -> tuple[Path, Path, int]:
        try:
            return self._index[when]
        except KeyError:
            first, last = self.timestamps[0], self.timestamps[-1]
            raise KeyError(
                f"{when:%Y-%m-%d %H:%M} not in archive (have {first:%Y-%m-%d %H:%M} "
                f"through {last:%Y-%m-%d %H:%M}, 6-hourly)"
            ) from None

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()

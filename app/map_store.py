"""Read access to a map-major store — the layout map queries are served from.

The mirror image of `series_store`. One chunk holds one whole map, so "the world
at one moment" is a single decode: `series_probe.json` measures 0.005 s here
against 0.23–2.5 s from the time-major copy, and `read.json` has a global 2t map
at 4.1 ms.

The archive is this shape already, so nothing new gets written — this is a reader
over `config.DATA_ROOT`. It carries the CDS names (`t2m`, `u10`), and the API
speaks Aurora's (`2t`, `10u`), so the rename lives here rather than leaking into
every caller.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from . import contracts

# CDS name -> the name the contract, the model and the API all use.
RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl"}
TIME_COORD = "valid_time"


class MapStore:
    def __init__(self, path: Path, time_coord: str = TIME_COORD):
        self.path = Path(path)
        self.time_coord = time_coord
        self._g = zarr.open(str(self.path), mode="r")

        self.lat = np.asarray(self._g["latitude"][:], dtype="float64")
        self.lon = np.asarray(self._g["longitude"][:], dtype="float64")

        units = self._g[time_coord].attrs.get("units", "hours since 1970-01-01")
        step, origin = units.split(" since ")
        self.times = pd.to_datetime(
            pd.Series(np.asarray(self._g[time_coord][:])),
            unit={"hours": "h", "seconds": "s", "days": "D", "minutes": "m"}[step.strip()],
            origin=pd.Timestamp(origin.strip()),
        ).to_numpy("datetime64[ns]")

        # Only what the store actually holds, under the name callers use. The
        # ERA5 archive carries CDS names; a published forecast is written with
        # the contract's names already, so both spellings resolve here.
        self._name = {}
        for cds, name in RENAME.items():
            if cds in self._g:
                self._name[name] = cds
            elif name in self._g:
                self._name[name] = name
        for name in ("z", "q", "t", "u", "v"):
            if name in self._g:
                self._name[name] = name

        self.levels = (tuple(int(v) for v in self._g["pressure_level"][:])
                       if "pressure_level" in self._g else ())
        if self.levels:
            # Files store levels descending; one canonical ascending order.
            self._level_order = np.argsort(self._g["pressure_level"][:])
            self.levels = tuple(int(self._g["pressure_level"][:][i]) for i in self._level_order)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._name)

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        if len(self.times) == 0:
            return None
        return (pd.Timestamp(self.times[0]).to_pydatetime(),
                pd.Timestamp(self.times[-1]).to_pydatetime())

    def index_of(self, when: dt.datetime) -> int:
        i = int(np.searchsorted(self.times, np.datetime64(when), "left"))
        if i >= len(self.times) or self.times[i] != np.datetime64(when):
            raise KeyError(when)
        return i

    def units(self, name: str) -> str | None:
        """The archive carries no units attribute, so fall back to the contract
        — which is the definition anyway, not a guess about the file."""
        attr = self._g[self._name[name]].attrs.get("units")
        if attr:
            return attr
        spec = contracts.SURFACE.get(name) or contracts.ATMOS.get(name)
        return spec[0] if spec else None

    def check_level(self, name: str, level: int | None) -> None:
        """Validate the level argument on its own, so callers can do it early."""
        on_levels = self._g[self._name[name]].ndim == 4
        if on_levels and level is None:
            raise ValueError(f"{name} is on pressure levels; pass one of {list(self.levels)}")
        if on_levels and level not in self.levels:
            raise ValueError(f"level {level} not in {list(self.levels)}")
        if not on_levels and level is not None:
            raise ValueError(f"{name} is a surface field and has no pressure level")

    def map_at(self, name: str, when: dt.datetime, level: int | None = None) -> np.ndarray:
        """One whole map. The query this store exists for: one chunk, one decode."""
        self.check_level(name, level)
        arr = self._g[self._name[name]]
        i = self.index_of(when)
        if arr.ndim == 4:
            j = int(self._level_order[self.levels.index(level)])
            return np.asarray(arr[i, j], dtype="float32")
        return np.asarray(arr[i], dtype="float32")

    def window(self, name: str, when: dt.datetime, lat_sel: slice, lon_sel: slice,
               level: int | None = None) -> np.ndarray:
        """A cut of one map. Costs the same chunk as the whole map — the saving
        is in what crosses the wire, not in what gets decoded."""
        return self.map_at(name, when, level)[lat_sel, lon_sel]

    def nearest_point(self, lat: float, lon: float) -> tuple[int, int, float, float]:
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude {lat} outside [-90, 90]")
        ilat = int(np.abs(self.lat - lat).argmin())
        ilon = int(np.abs(self.lon - (lon % 360.0)).argmin())
        return ilat, ilon, float(self.lat[ilat]), float(self.lon[ilon])

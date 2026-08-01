"""Read access to a time-major store — the layout point queries are served from.

The archive keeps one chunk per map, which is right for "the whole world at one
moment" and catastrophic for "one place over time": `bench/results/read.json` has
a point series touching 1441 MB to return 1.4 KB. This store holds the same
fields with the **time axis inside the chunk**, and `series_probe.json` measures
what that buys — a four-variable point series over the archive goes from 2.036 s
to 0.011 s, amplification from 1038240 to 1024.

It is a second copy, not a replacement. The same probe shows a map read costing
0.23–2.5 s here against 0.005 s map-major, so both stores exist and each answers
the queries it was shaped for.

Two things this module owns because they are wrong everywhere else:

* **Longitude.** The grid runs 0…359.75 ascending, because that is what Aurora
  wants. Callers say −180…180 because that is what people say. The conversion
  happens here, at the edge, and never inside the model path.
* **Packing.** A store may hold int16 with a scale and offset (that is how ERA5
  itself is stored, and it costs 1.42x the size for nothing). Callers always get
  float32 in physical units.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Not a coordinate we invent: the archive names its time axis this way, and so
# does everything CDS hands us.
TIME_COORD = "valid_time"


class SeriesStore:
    """One time-major zarr group. Opened once, read many times, never written."""

    def __init__(self, path: Path, time_coord: str = TIME_COORD):
        self.path = Path(path)
        self.time_coord = time_coord
        self._g = zarr.open(str(self.path), mode="r")

        self.lat = np.asarray(self._g["latitude"][:], dtype="float64")
        self.lon = np.asarray(self._g["longitude"][:], dtype="float64")

        raw = self._g[time_coord][:]
        # zarr hands back the raw int64; the units live in the attrs the writer
        # copied over, and pandas is the only thing that reads CF properly.
        units = self._g[time_coord].attrs.get("units")
        if units:
            self.times = pd.to_datetime(
                pd.Series(np.asarray(raw)), unit=_cf_unit(units), origin=_cf_origin(units)
            ).to_numpy(dtype="datetime64[ns]")
        else:
            self.times = np.asarray(raw, dtype="datetime64[ns]")

        self.variables = tuple(
            n for n in self._g.array_keys()
            if n not in ("latitude", "longitude", time_coord)
        )

    # ------------------------------------------------------------------ coords

    def nearest_point(self, lat: float, lon: float) -> tuple[int, int, float, float]:
        """(ilat, ilon, node lat, node lon) — the node, not what was asked for.

        The grid is 0.25 degrees, which is up to 14 km. Silently answering a
        different place than the caller named is exactly the kind of thing that
        looks fine until somebody compares against a station.
        """
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude {lat} outside [-90, 90]")
        ilat = int(np.abs(self.lat - lat).argmin())
        ilon = int(np.abs(self.lon - (lon % 360.0)).argmin())
        return ilat, ilon, float(self.lat[ilat]), float(self.lon[ilon])

    def time_slice(self, start: dt.datetime | None, end: dt.datetime | None) -> slice:
        """Half-open in index space, inclusive in wall-clock, as callers expect."""
        t = self.times
        i0 = 0 if start is None else int(np.searchsorted(t, np.datetime64(start), "left"))
        i1 = len(t) if end is None else int(np.searchsorted(t, np.datetime64(end), "right"))
        return slice(i0, max(i0, i1))

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        if len(self.times) == 0:
            return None
        return (pd.Timestamp(self.times[0]).to_pydatetime(),
                pd.Timestamp(self.times[-1]).to_pydatetime())

    def units(self, name: str) -> str | None:
        """Physical units as the writer recorded them. Never inferred."""
        return self._g[name].attrs.get("units")

    def map_at(self, name: str, index: int) -> np.ndarray:
        """One whole map. Present so callers do not reach into the group, but
        this store is the wrong shape for it — see `series_probe.json`."""
        return self._physical(name, self._g[name][index])

    # ------------------------------------------------------------------- reads

    def point(self, lat: float, lon: float, variables=None,
              start=None, end=None) -> dict:
        """One grid node over a time range. The query this store exists for."""
        ilat, ilon, node_lat, node_lon = self.nearest_point(lat, lon)
        sel = self.time_slice(start, end)
        names = self._names(variables)

        values = {n: self._physical(n, self._g[n][sel, ilat, ilon]) for n in names}
        return {
            "point": {"lat": node_lat, "lon": _to_signed(node_lon),
                      "requested": {"lat": lat, "lon": lon}},
            "times": [pd.Timestamp(t).to_pydatetime() for t in self.times[sel]],
            "values": values,
        }

    def box(self, lat0: float, lat1: float, lon0: float, lon1: float,
            variables=None, start=None, end=None) -> dict:
        """A small area over a time range — a city, not a continent.

        Deliberately not the way to ask for a map: this store reads a whole
        chunk of time for every tile it touches, so a continent-sized box here
        costs the whole archive. Maps come from the map-major store.
        """
        ilat0, ilon0, _, _ = self.nearest_point(max(lat0, lat1), lon0)
        ilat1, ilon1, _, _ = self.nearest_point(min(lat0, lat1), lon1)
        if ilon1 < ilon0:
            # The box crosses the prime meridian on a 0..360 grid. Two reads,
            # joined on the longitude axis, rather than a wrong empty answer.
            return self._box_wrapped(ilat0, ilat1, ilon0, ilon1, variables, start, end)

        sel = self.time_slice(start, end)
        names = self._names(variables)
        lat_sel, lon_sel = slice(ilat0, ilat1 + 1), slice(ilon0, ilon1 + 1)
        values = {n: self._physical(n, self._g[n][sel, lat_sel, lon_sel]) for n in names}
        return self._box_result(sel, lat_sel, lon_sel, values)

    def _box_wrapped(self, ilat0, ilat1, ilon0, ilon1, variables, start, end) -> dict:
        sel = self.time_slice(start, end)
        names = self._names(variables)
        lat_sel = slice(ilat0, ilat1 + 1)
        east, west = slice(ilon0, len(self.lon)), slice(0, ilon1 + 1)
        values = {
            n: np.concatenate(
                [self._physical(n, self._g[n][sel, lat_sel, east]),
                 self._physical(n, self._g[n][sel, lat_sel, west])], axis=-1)
            for n in names
        }
        lons = np.concatenate([self.lon[east], self.lon[west]])
        return {
            "times": [pd.Timestamp(t).to_pydatetime() for t in self.times[sel]],
            "lat": self.lat[lat_sel].tolist(),
            "lon": [_to_signed(v) for v in lons],
            "values": values,
        }

    def _box_result(self, sel, lat_sel, lon_sel, values) -> dict:
        return {
            "times": [pd.Timestamp(t).to_pydatetime() for t in self.times[sel]],
            "lat": self.lat[lat_sel].tolist(),
            "lon": [_to_signed(v) for v in self.lon[lon_sel]],
            "values": values,
        }

    # ------------------------------------------------------------------ helpers

    def _names(self, variables) -> list[str]:
        if variables is None:
            return list(self.variables)
        unknown = [v for v in variables if v not in self.variables]
        if unknown:
            raise KeyError(f"{unknown} not in store (have {sorted(self.variables)})")
        return list(variables)

    def _physical(self, name: str, raw: np.ndarray) -> np.ndarray:
        """int16 back to physical units, if this store is packed."""
        attrs = self._g[name].attrs
        if "scale_factor" not in attrs:
            return np.asarray(raw, dtype="float32")
        return (np.asarray(raw, dtype="float32") * float(attrs["scale_factor"])
                + float(attrs["add_offset"]))


def _to_signed(lon: float) -> float:
    """0..360 -> -180..180. Only ever at the boundary, never in the model path."""
    lon = float(lon) % 360.0
    return lon - 360.0 if lon > 180.0 else lon


def _cf_unit(units: str) -> str:
    return {"seconds": "s", "hours": "h", "days": "D",
            "minutes": "m", "nanoseconds": "ns"}[units.split(" since ")[0].strip()]


def _cf_origin(units: str) -> pd.Timestamp:
    return pd.Timestamp(units.split(" since ")[1].strip())

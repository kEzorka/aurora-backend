"""Deep-history maps, from the copy of ERA5 that is already chunked as maps.

Google's ARCO ERA5 is `[1, 721, 1440]` per chunk: one chunk *is* one global map,
which is precisely the layout `build_maps.py` writes locally and precisely what
`/v1/map` asks for. Nothing here has to be rearranged, decompressed twice, or
regridded — the remote store and the local one are the same shape, and this
class exists to make `read_api` unable to tell them apart.

Reads like a `MapStore`, for the same reason `OpenMeteoSeries` reads like a
`SeriesStore`: the planner, the segment bookkeeping and the bbox windowing were
written once and should not learn about the network.

Measured on this box against 1990-03-15T12:00Z: 1.30 s for a cold global map,
values 214.4..313.2 K, `msl` 100 974 Pa — both inside the contract, which is how
we know the epoch decoded right. That last point is not rhetorical: ARCO counts
`hours since 1900-01-01` where every local store counts from 1970, and a
seventy-year offset produces a perfectly plausible map of the wrong day.

**What is cached and why.** 1.3 s is fine once and awful for the tenth caller
asking about the same famous storm. Maps land in `config.CACHE_ROOT/arco` as raw
float32 — 4.15 MB each, read back in about 4 ms — under a byte ceiling this
class enforces itself on every write. The ceiling is not decoration: the disk is
shared with other people's jobs, and an unbounded cache in front of eighty years
of hourly maps would fill it in a weekend.

**What is not covered.** ARCO's time axis runs 1900 to 2050 because the array is
pre-allocated, not because it holds a map for 1912. Coverage is declared here
from ERA5's own start rather than read off the axis, and a moment outside it is
a 404 rather than a screen of NaN.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, contracts

URL = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
       "full_37-1h-0p25deg-chunk-1.zarr-v3")

# ERA5's start. The store's own axis begins in 1900 and is empty until here.
START = dt.datetime(1940, 1, 1)

# ERA5 final is roughly three months behind; ERA5T fills the gap and is what the
# local archive holds anyway, so the proxy is trimmed rather than trusted.
LAG_DAYS = 10

# ARCO's names, which are neither the contract's nor CDS's short names. A third
# naming convention, kept here rather than in `map_store.RENAME`, because it
# belongs to this source and not to the backend.
NAMES = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
VARIABLES = tuple(NAMES)

CELL_BYTES = 721 * 1440 * 4


class ArcoMaps:
    """A `MapStore` over ERA5 in Google's bucket, for the years we do not hold."""

    name = "proxied-history"

    def __init__(self, until: dt.datetime | None = None,
                 cache_root: Path | None = None,
                 cache_bytes: int = None,
                 step_hours: int = config.STEP_HOURS):
        self.step_hours = int(step_hours)
        self.cache = MapCache(
            Path(cache_root or Path(config.CACHE_ROOT) / "arco"),
            cache_bytes if cache_bytes is not None else config.ARCO_CACHE_BYTES,
        )

        # Exclusive, like `OpenMeteoSeries.__init__` — /v1/map takes the first
        # source that holds the moment, and this one sorts first, so an
        # inclusive bound would answer one moment a second over the network that
        # the archive has on disk in five milliseconds.
        hard = dt.datetime.utcnow() - dt.timedelta(days=LAG_DAYS)
        cut = until - dt.timedelta(hours=self.step_hours) if until else None
        self.end = min(cut, hard) if cut else hard
        self.end = self.end.replace(minute=0, second=0, microsecond=0)
        self.end -= dt.timedelta(hours=self.end.hour % self.step_hours)

        self.variables = VARIABLES
        self.levels: tuple[int, ...] = ()

        # The analysis grid, stated rather than fetched. Opening the remote store
        # costs four seconds and this is the one fact about it that cannot
        # change without the dataset becoming a different dataset; it is checked
        # against the real coordinates the first time a map is actually read.
        self.lat = np.linspace(90.0, -90.0, contracts.LAT_SIZE)
        self.lon = np.arange(0.0, 360.0, contracts.RESOLUTION)

        self._ds = None
        self._times: np.ndarray | None = None

    # ------------------------------------------------------------------- axis

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            if self.end <= START:
                self._times = np.empty(0, dtype="datetime64[ns]")
            else:
                self._times = np.arange(
                    np.datetime64(START, "ns"),
                    np.datetime64(self.end, "ns") + np.timedelta64(1, "ns"),
                    np.timedelta64(self.step_hours, "h"),
                ).astype("datetime64[ns]")
        return self._times

    def time_slice(self, start: dt.datetime | None, end: dt.datetime | None) -> slice:
        t = self.times
        i0 = 0 if start is None else int(np.searchsorted(t, np.datetime64(start), "left"))
        i1 = len(t) if end is None else int(np.searchsorted(t, np.datetime64(end), "right"))
        return slice(i0, max(i0, i1))

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        t = self.times
        if len(t) == 0:
            return None
        return (pd.Timestamp(t[0]).to_pydatetime(), pd.Timestamp(t[-1]).to_pydatetime())

    def index_of(self, when: dt.datetime) -> int:
        i = int(np.searchsorted(self.times, np.datetime64(when), "left"))
        if i >= len(self.times) or self.times[i] != np.datetime64(when):
            raise KeyError(f"{when:%Y-%m-%d %H:%M} is not in {self.name}")
        return i

    def units(self, name: str) -> str | None:
        spec = contracts.SURFACE.get(name)
        return spec[0] if spec else None

    def check_level(self, name: str, level: int | None) -> None:
        if level is not None:
            raise ValueError(f"{self.name} carries surface fields only; "
                             f"{sorted(self.variables)} take no pressure level")

    def nearest_point(self, lat: float, lon: float) -> tuple[int, int, float, float]:
        ilat = int(np.abs(self.lat - lat).argmin())
        ilon = int(np.abs(self.lon - (lon % 360.0)).argmin())
        return ilat, ilon, float(self.lat[ilat]), float(self.lon[ilon])

    # ------------------------------------------------------------------ reads

    def map_at(self, name: str, when: dt.datetime, level: int | None = None) -> np.ndarray:
        """One global map, from the cache if it is there and the bucket if not."""
        self.check_level(name, level)
        if name not in NAMES:
            raise KeyError(f"{name} not available from {self.name}; "
                           f"have {sorted(self.variables)}")
        self.index_of(when)

        hit = self.cache.get(name, when)
        if hit is not None:
            return hit

        arr = self._fetch(name, when)
        self.cache.put(name, when, arr)
        return arr

    def window(self, name: str, when: dt.datetime, lat_sel: slice, lon_sel: slice,
               level: int | None = None) -> np.ndarray:
        """Present because `MapStore` has it. There is no cheaper path for a
        subset here — the chunk is the whole map either way, so a bbox saves
        bytes on the wire only if the remote store is cut finer than it is."""
        return self.map_at(name, when, level)[lat_sel, lon_sel]

    def _fetch(self, name: str, when: dt.datetime) -> np.ndarray:
        import xarray as xr  # ~2 s of import; not paid unless a proxy read happens

        if self._ds is None:
            self._ds = xr.open_zarr(URL, chunks=None)
            got_lat = np.asarray(self._ds.latitude.values, dtype="float64")
            got_lon = np.asarray(self._ds.longitude.values, dtype="float64")
            # The one place the stated grid is confronted with the real one. If
            # this ever fails, every cached map under this root is on a grid the
            # backend was not told about.
            if not (np.allclose(got_lat, self.lat) and np.allclose(got_lon, self.lon)):
                self._ds = None
                raise ValueError(f"{URL} is not on the analysis grid this backend "
                                 f"assumes: lat {got_lat[0]}..{got_lat[-1]} "
                                 f"({got_lat.size}), lon {got_lon[0]}..{got_lon[-1]}")
            self.lat, self.lon = got_lat, got_lon

        arr = np.asarray(
            self._ds[NAMES[name]].sel(time=np.datetime64(when, "ns")).values,
            dtype="float32")
        # Cheap, and it is what proves the 1900 epoch decoded correctly: a map
        # read seventy years off would still be a map, but a plausible-looking
        # one. A range check does not prove the date, and nothing does from one
        # field alone — but it does catch the epoch mistakes that land in space.
        problems = contracts.check_values(name, arr)
        if problems:
            raise ValueError(f"{self.name} {name} at {when:%Y-%m-%d %H:%M} failed "
                             f"the value contract: {problems}")
        return arr

    # ------------------------------------------------------------------- meta

    def describe(self) -> dict:
        c = self.coverage
        return {
            "name": self.name,
            "via": "arco-era5",
            "available": c is not None,
            "kind": "proxy",
            "upstream": URL,
            "dataset": "ARCO ERA5, chunk [1, 721, 1440] — one chunk is one map",
            "grid": {"shape": [len(self.lat), len(self.lon)], "kind": "analysis",
                     "latitude": f"{self.lat[0]} to {self.lat[-1]}",
                     "longitude": f"{self.lon[0]} to {self.lon[-1]}"},
            "variables": sorted(self.variables),
            "levels": [],
            "from": c[0].isoformat() + "Z" if c else None,
            "to": c[1].isoformat() + "Z" if c else None,
            "step_hours": self.step_hours,
            "layout": "map-major upstream, answers /v1/map",
            "cache": self.cache.describe(),
        }


class MapCache:
    """Whole maps on local disk, under a ceiling this class keeps itself.

    One file per (variable, moment), raw little-endian float32, no header — the
    shape is the grid and the grid is in the contract. Eviction is oldest-access
    first and runs on write, so the ceiling is a property of the directory
    rather than a promise somebody has to remember to check.

    Deliberately not `functools.lru_cache` and deliberately not in-process: the
    read API runs four workers, and a cache that each of them fills separately
    is four times the disk for the same hit rate.
    """

    def __init__(self, root: Path, limit_bytes: int):
        self.root = Path(root)
        self.limit = int(limit_bytes)
        self.hits = 0
        self.misses = 0

    def _path(self, name: str, when: dt.datetime) -> Path:
        return self.root / name / f"{when:%Y%m%dT%H%M}.f32"

    def get(self, name: str, when: dt.datetime) -> np.ndarray | None:
        path = self._path(name, when)
        try:
            raw = path.read_bytes()
        except OSError:
            self.misses += 1
            return None
        if len(raw) != CELL_BYTES:
            # A short file is a write that was interrupted. Drop it rather than
            # reshaping whatever arrived into a map-shaped lie.
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        # Access time drives eviction, so touch it. Failing to is not fatal —
        # a read-only cache root just evicts by write order instead.
        try:
            os.utime(path)
        except OSError:
            pass
        return np.frombuffer(raw, dtype="<f4").reshape(
            contracts.LAT_SIZE, contracts.LON_SIZE).copy()

    def put(self, name: str, when: dt.datetime, arr: np.ndarray) -> None:
        if self.limit <= 0:
            return
        path = self._path(name, when)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed in, for the reason every other writer here
        # does it: four workers can miss the same map at the same moment, and a
        # reader must never see half of one.
        tmp = path.with_suffix(f".{os.getpid()}.partial")
        try:
            tmp.write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            return
        self.evict()

    def evict(self) -> int:
        """Bring the directory back under the ceiling. Returns bytes dropped."""
        files = [(p.stat(), p) for p in self.root.rglob("*.f32")]
        total = sum(st.st_size for st, _ in files)
        if total <= self.limit:
            return 0
        files.sort(key=lambda pair: pair[0].st_atime)
        dropped = 0
        for st, path in files:
            if total - dropped <= self.limit:
                break
            try:
                path.unlink()
            except OSError:
                continue
            dropped += st.st_size
        return dropped

    def total_bytes(self) -> int:
        try:
            return sum(p.stat().st_size for p in self.root.rglob("*.f32"))
        except OSError:
            return 0

    def describe(self) -> dict:
        used = self.total_bytes()
        return {
            "root": self.root.name,
            "used_gb": round(used / 1024**3, 2),
            "limit_gb": round(self.limit / 1024**3, 2),
            "maps": used // CELL_BYTES,
            "hits": self.hits,
            "misses": self.misses,
        }

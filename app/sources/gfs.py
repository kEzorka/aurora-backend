"""The operational feed: NOAA GFS analyses, pulled from S3 as GRIB2.

The brief asks for a backend that takes fresh weather from the centres that
publish it and recomputes, rather than sitting on a static archive. This is the
half that goes and gets it. It reads like `ERA5Store` — `has`, `surface_slice`,
`atmos_slice`, `static_fields`, `lat`, `lon`, `levels` — so `batch_builder` and
`produce.py` do not learn that the input came off a network.

**Why the poll is three-hourly when the cycles are six-hourly.** Global analysis
is six-hourly because the assimilation window is: `gfs.20260731/` holds `00`,
`06`, `12`, `18` and nothing between them. That is not a limitation to route
around, it is what "an analysis" means. What a three-hour poll buys is latency:
00z lands on the bucket around 03:30 UTC and 06z around 09:30, so a tick every
three hours catches each cycle within about ninety minutes of publication, where
a six-hourly tick can be five hours and fifty-nine minutes late. `produce.py` is
idempotent, so the ticks that find nothing new exit without touching the GPU.

**Bytes, and why the .idx file is the whole trick.** One `pgrb2.0p25` file is
~500 MB and holds several hundred fields. Aurora needs 69 of them: four at the
surface and five on each of thirteen pressure levels. The sidecar `.idx` gives
the byte offset of every record, so each field is an HTTP range request of about
0.8 MB. Measured for one cycle: **54.6 MB in 43 requests**, against 500 MB for
the file — the records for one level sit next to each other, so adjacent ones
merge into a single range.

**The unit trap this file exists to survive.** GFS publishes `HGT`, geopotential
*height* in geopotential metres. Aurora wants `z`, geopotential in m² s⁻². They
differ by exactly `contracts.G = 9.80665` and both are plausible-looking numbers
in the thousands. `contracts.check_values` has a branch that names this mistake
by hand; every field here goes through it before it reaches a tensor.

**What this is not.** `AuroraPretrained` is the ERA5 checkpoint. GFS analysis is
a different centre's model and a different assimilation, so initialising the
pretrained weights from it is out of distribution and the forecast is worse than
the same run from ERA5 — how much worse is not something this file can claim
without a scored comparison. ECMWF's open IFS HRES data is the closer match to
the `Aurora` fine-tuned checkpoint and is the upgrade path; GFS is here because
it is open, indexed, and on S3 without credentials.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .. import config, contracts

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# GFS keeps a rolling window; the bucket's oldest prefix on 2026-07-31 was
# gfs.20210101. This is an operational feed, not an archive — anything older
# than a few days belongs to the proxy in arco.py.
CYCLE_HOURS = 6

# The bucket's `.idx` uses NCEP's names; cfgrib decodes the same records into
# ECMWF short names. Both appear here because both are needed: the first to pick
# byte ranges without downloading the file, the second to read what came back.
IDX_SURFACE = [("TMP", "2 m above ground"), ("UGRD", "10 m above ground"),
               ("VGRD", "10 m above ground"), ("PRMSL", "mean sea level")]
IDX_ATMOS = ("TMP", "UGRD", "VGRD", "SPFH", "HGT")

# cfgrib's short name -> the contract's. Nearly an identity, which is exactly
# what makes the two exceptions dangerous: `prmsl` is our `msl`, and `gh` is
# geopotential HEIGHT in gpm where we want geopotential in m^2 s^-2. See
# `_convert`, and the module docstring for why that one is the trap.
RENAME = {"2t": "2t", "10u": "10u", "10v": "10v", "prmsl": "msl",
          "t": "t", "u": "u", "v": "v", "q": "q", "gh": "z"}

SURFACE_OUT = ("2t", "10u", "10v", "msl")
ATMOS_OUT = ("t", "u", "v", "q", "z")

# What cfgrib should say the units are, per contract name. Checked rather than
# assumed: if NCEP ever publishes geopotential instead of height under the same
# record, multiplying by G again would be a silent factor of 9.8.
EXPECT_UNITS = {"2t": "K", "10u": "m s**-1", "10v": "m s**-1", "msl": "Pa",
                "t": "K", "u": "m s**-1", "v": "m s**-1", "q": "kg kg**-1",
                "z": "gpm"}

LEVELS = contracts.PRESSURE_LEVELS


class GFSStore:
    """One cycle's analysis, and the one before it, read like a local archive.

    Nothing is held on disk between runs. Two timesteps of 69 fields is ~110 MB
    of GRIB and ~800 MB decoded, which is a rollout's worth of input and not
    worth an eviction policy; the forecast it produces is what gets published.
    """

    name = "gfs"

    def __init__(self, init: dt.datetime | None = None,
                 static_file: Path | None = None,
                 workers: int = 8, timeout: float = 300.0):
        self.timeout = timeout
        self.workers = workers
        self._session = requests.Session()
        self._static_file = Path(static_file or config.STATIC_FILE)
        self._cache: dict[dt.datetime, dict[str, np.ndarray]] = {}
        self._idx: dict[dt.datetime, list[tuple[int, int | None, str, str]]] = {}

        self.lat = np.linspace(90.0, -90.0, contracts.LAT_SIZE).astype("float32")
        self.lon = np.arange(0.0, 360.0, contracts.RESOLUTION).astype("float32")
        self.levels = tuple(LEVELS)

        self.init = init or self.newest_cycle()

    # --------------------------------------------------------------- discovery

    def _key(self, cycle: dt.datetime) -> str:
        return (f"gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
                f"gfs.t{cycle:%H}z.pgrb2.0p25.f000")

    def newest_cycle(self, now: dt.datetime | None = None, back: int = 8) -> dt.datetime:
        """The newest cycle whose analysis is actually on the bucket.

        Walks backwards rather than computing from the clock, because a cycle
        exists as a directory some time before its files are complete. The
        `.idx` is written after the GRIB, so its presence is the signal that the
        record offsets — and therefore every range request below — are valid.
        """
        now = now or dt.datetime.utcnow()
        c = now.replace(minute=0, second=0, microsecond=0)
        c -= dt.timedelta(hours=c.hour % CYCLE_HOURS)
        for _ in range(back):
            if self._has_cycle(c) and self._has_cycle(c - dt.timedelta(hours=CYCLE_HOURS)):
                return c
            c -= dt.timedelta(hours=CYCLE_HOURS)
        raise RuntimeError(
            f"no GFS cycle with a predecessor in the last {back * CYCLE_HOURS} h "
            f"under {BUCKET}; the feed is down or the bucket layout changed")

    def _has_cycle(self, cycle: dt.datetime) -> bool:
        try:
            r = self._session.head(f"{BUCKET}/{self._key(cycle)}.idx", timeout=30)
            return r.status_code == 200
        except OSError:
            return False

    # --------------------------------------------------------------- the store

    @property
    def timestamps(self) -> list[dt.datetime]:
        return [self.init - dt.timedelta(hours=CYCLE_HOURS), self.init]

    def has(self, when: dt.datetime) -> bool:
        return when in self.timestamps

    def _require(self, when: dt.datetime):
        raise KeyError(f"{when:%Y-%m-%d %H:%M} is not a cycle this store holds; "
                       f"have {[f'{t:%Y-%m-%d %H:%M}' for t in self.timestamps]}")

    def surface_slice(self, when: dt.datetime) -> dict[str, np.ndarray]:
        return {k: v for k, v in self._fields(when).items() if v.ndim == 2}

    def atmos_slice(self, when: dt.datetime) -> dict[str, np.ndarray]:
        return {k: v for k, v in self._fields(when).items() if v.ndim == 3}

    def static_fields(self) -> dict[str, np.ndarray]:
        """`lsm`, `z` and `slt` from the local ERA5 static file, on purpose.

        These do not change with the weather, and two of them are not in
        `pgrb2.0p25` in a usable form at all — soil type is not published there.
        Taking the invariants from the archive keeps the model's static inputs
        exactly what the checkpoint was trained with, which is one fewer
        distribution shift on top of the one this source already introduces.
        """
        from ..era5_store import ERA5Store

        return ERA5Store(static_file=self._static_file).static_fields()

    # ------------------------------------------------------------------ decode

    def _fields(self, when: dt.datetime) -> dict[str, np.ndarray]:
        if when not in self._cache:
            if not self.has(when):
                self._require(when)
            self._cache[when] = self._download(when)
        return self._cache[when]

    def _records(self, cycle: dt.datetime):
        """(start, end, variable, level) for every record, ends resolved.

        The last record has no successor, so its end is None and the range
        request is open-ended — which S3 answers with "to the end of the object".
        """
        if cycle in self._idx:
            return self._idx[cycle]
        r = self._session.get(f"{BUCKET}/{self._key(cycle)}.idx", timeout=60)
        r.raise_for_status()
        rows = []
        for line in r.text.splitlines():
            parts = line.split(":")
            if len(parts) < 5:
                continue
            rows.append((int(parts[1]), parts[3], parts[4]))
        out = [(s, (rows[i + 1][0] - 1 if i + 1 < len(rows) else None), v, lv)
               for i, (s, v, lv) in enumerate(rows)]
        self._idx[cycle] = out
        return out

    def _wanted(self, cycle: dt.datetime):
        """The 69 records Aurora needs, as merged byte ranges.

        Adjacent records become one request: measured on 2026-07-31T06Z, 69
        records collapse to 43 ranges and 54.6 MB, against 500 MB for the file.
        """
        recs = self._records(cycle)
        want = set(IDX_SURFACE) | {(v, f"{lv} mb") for v in IDX_ATMOS for lv in LEVELS}
        picked = [(i, *r) for i, r in enumerate(recs) if (r[2], r[3]) in want]
        if len(picked) != len(want):
            found = {(v, lv) for _, _, _, v, lv in picked}
            raise ValueError(f"{cycle:%Y-%m-%d %H:%M}: {sorted(want - found)} not in "
                             f"the index; this cycle cannot initialise the model")

        merged = []
        for i, start, end, _, _ in sorted(picked):
            if merged and merged[-1][2] == i - 1 and merged[-1][1] is not None:
                lo, _, _ = merged[-1]
                merged[-1] = (lo, end, i)
            else:
                merged.append((start, end, i))
        return [(lo, hi) for lo, hi, _ in merged]

    def _download(self, cycle: dt.datetime) -> dict[str, np.ndarray]:
        ranges = self._wanted(cycle)
        url = f"{BUCKET}/{self._key(cycle)}"

        def fetch(lo_hi):
            lo, hi = lo_hi
            rng = f"bytes={lo}-{hi}" if hi is not None else f"bytes={lo}-"
            r = self._session.get(url, headers={"Range": rng}, timeout=self.timeout)
            r.raise_for_status()
            return r.content

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            blobs = list(pool.map(fetch, ranges))

        # cfgrib wants a file. The concatenation of complete GRIB messages is
        # itself a valid GRIB file, which is the property that makes the whole
        # range-request approach work.
        Path(config.CACHE_ROOT).mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".grib2", dir=str(config.CACHE_ROOT))
        try:
            with os.fdopen(fd, "wb") as fh:
                for b in blobs:
                    fh.write(b)
            return self._decode(path, cycle)
        finally:
            os.unlink(path)

    def _decode(self, path: str, cycle: dt.datetime) -> dict[str, np.ndarray]:
        """GRIB in, contract-shaped arrays out.

        cfgrib splits a mixed file into one dataset per (level type, step) and
        stacks the pressure levels onto a dimension of their own, so this walks
        datasets rather than records and reorders the level axis by hand.
        """
        import cfgrib

        out: dict[str, np.ndarray] = {}
        for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
            self._check_grid(ds, cycle)
            for da in ds.data_vars.values():
                short = str(da.attrs.get("GRIB_shortName", ""))
                name = RENAME.get(short)
                if name is None:
                    continue
                got = da.attrs.get("units")
                if got != EXPECT_UNITS[name]:
                    raise ValueError(
                        f"GFS {cycle:%Y-%m-%d %H:%M}: {short} arrived in {got!r}, "
                        f"expected {EXPECT_UNITS[name]!r}. The conversion for "
                        f"{name} is written for the second and is now wrong.")

                if "isobaricInhPa" in da.dims:
                    have = [int(v) for v in np.atleast_1d(da.coords["isobaricInhPa"].values)]
                    missing = [lv for lv in LEVELS if lv not in have]
                    if missing:
                        raise ValueError(f"{cycle:%Y-%m-%d %H:%M}: {name} missing "
                                         f"levels {missing}")
                    # Reindexed explicitly to ascending order, not taken as it
                    # came: GRIB files are free to order levels either way, and
                    # a level axis silently reversed is a forecast with the
                    # stratosphere at the bottom. `Metadata.atmos_levels` then
                    # tells the model which order this is.
                    take = [have.index(lv) for lv in LEVELS]
                    arr = np.asarray(da.values, dtype="float32")[take]
                else:
                    arr = np.asarray(da.values, dtype="float32")
                out[name] = _convert(name, arr)

        missing = [n for n in (*SURFACE_OUT, *ATMOS_OUT) if n not in out]
        if missing:
            raise ValueError(f"{cycle:%Y-%m-%d %H:%M}: decoded no {missing}")

        problems = [p for n, a in out.items() for p in contracts.check_values(n, a)]
        if problems:
            raise ValueError(f"GFS {cycle:%Y-%m-%d %H:%M} failed the value "
                             f"contract: {problems}")
        return out

    def _check_grid(self, ds, cycle: dt.datetime) -> None:
        lat = np.asarray(ds["latitude"].values, dtype="float64")
        lon = np.asarray(ds["longitude"].values, dtype="float64")
        if not (np.allclose(lat, self.lat) and np.allclose(lon, self.lon)):
            raise ValueError(
                f"GFS {cycle:%Y-%m-%d %H:%M} is not on the analysis grid: "
                f"lat {lat[0]}..{lat[-1]} ({lat.size}), lon {lon[0]}..{lon[-1]} "
                f"({lon.size}). Expected {contracts.LAT_SIZE}x{contracts.LON_SIZE} "
                "descending from 90. Regridding is not done silently.")

    def close(self) -> None:
        self._cache.clear()
        self._session.close()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "upstream": BUCKET,
            "init_time": self.init.isoformat(),
            "history": [t.isoformat() for t in self.timestamps],
            "cycle_hours": CYCLE_HOURS,
            "poll_hours": 3,
            "grid": {"shape": [len(self.lat), len(self.lon)], "kind": "analysis"},
            "levels": list(self.levels),
            "converted": ["HGT gpm -> z m**2 s**-2 (x9.80665)"],
            "caveat": "AuroraPretrained is the ERA5 checkpoint; GFS input is out "
                      "of distribution for it",
        }


def _convert(name: str, arr: np.ndarray) -> np.ndarray:
    """Into the contract's units. One branch, and it is the dangerous one."""
    if name == "z":
        # Geopotential height (gpm) to geopotential (m^2 s^-2). Skipping this
        # leaves every level off by a factor of 9.8 — numbers that still look
        # like altitudes, which is why contracts.check_values names it by hand.
        return arr * contracts.G
    return arr

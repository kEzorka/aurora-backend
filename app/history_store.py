"""The preloaded history window: 220 days of all 69 fields, time-major.

`SeriesStore` reads one zarr group with one time axis, which is right for the
surface-only copy that sits beside the archive and wrong here. This window is
172 GB, it slides — every six hours a moment arrives at the front and an old one
has to leave the back — and zarr has no way to drop the head of an array. Doing
it inside one array means rewriting the array.

So the window is **a directory of blocks**, each block a zarr group of
`config.HISTORY_TIME_CHUNK` (88) moments, each array chunked `[88, 32, 32]`:

    preload.zarr/manifest.json
    preload.zarr/b0000246/     <- 88 moments, 25 GB, sealed
    preload.zarr/b0000247/
    ...
    preload.zarr/b0000256/     <- the newest, being written into

Eviction is `rm -rf` of the oldest directory. That is the entire reason for the
layout: 22 days leave in one syscall instead of a 172 GB rewrite.

**Ordinals.** A moment is identified by `ordinal = epoch_hours // 6` — absolute,
gapless, monotonic since 1970, independent of what the store happens to hold.
The block is `ordinal // 88` and the slot inside it is `ordinal % 88`, so a
timestamp maps to a file position by arithmetic alone, with no index to keep in
sync and no lookup to get wrong. It also means block ids never repeat, so an
evicted directory name is never reused and a stale reader gets ENOENT rather
than somebody else's weather.

**Time is not stored.** It is `ordinal * 6 h` since the epoch. A `valid_time`
array per block would be 88 numbers that can disagree with the directory name,
and there is no third party to check them against.

**Past only.** The ten forecast days live under `config.FORECAST_ROOT`, which
publishes a run directory and swaps a symlink, so a reader never sees half a
forecast. Folding the forecast tail in here would mean rewriting chunks that
hold real analysis every six hours on a merge that must not drop a truth
moment — a transaction problem in exchange for a routing one. A query spanning
"last week through next week" is stitched in `read_api`, which is the cheaper
side to pay on. The name for the boundary is in the manifest as
`truth_through`: everything at or below it is analysis, and this store holds
nothing above it.

**Appends go in place.** Writing moment 50 into an `[88, 32, 32]` chunk makes
zarr decompress, merge and recompress it, for every tile of every field. That is
25 GB of read-modify-write and it measures 98 s at 69 variables
(`bench/results/edge_probe.json`) against a cycle 21 600 s long. There is no
separate store for the fresh edge and no sealing pass.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from . import config
from .contracts import (ATMOS, LAT_FIRST, LAT_LAST, LAT_SIZE, LON_SIZE,
                        PRESSURE_LEVELS, RESOLUTION, STEP_HOURS, SURFACE)
from .publish import BLOSC, TILE

EPOCH = dt.datetime(1970, 1, 1)
SECONDS_PER_STEP = STEP_HOURS * 3600

# 4 + 5 x 13 = 69. Every array in a block is 3-D `[time, lat, lon]`; a level is
# part of the name rather than a fourth axis, because the chunk shape that the
# probes settled on is 3-D and a level axis would put 13 levels in every chunk
# to answer a query that names one.
FIELDS: tuple[str, ...] = tuple(SURFACE) + tuple(
    f"{name}{level}" for name in ATMOS for level in PRESSURE_LEVELS
)


def field_key(name: str, level: int | None = None) -> str:
    """('t', 500) -> 't500'; ('2t', None) -> '2t'. The one place names are made."""
    if name in SURFACE:
        if level is not None:
            raise KeyError(f"{name} is a surface field and takes no level")
        return name
    if name not in ATMOS:
        raise KeyError(f"unknown variable {name!r}")
    if level not in PRESSURE_LEVELS:
        raise KeyError(f"level {level} not in {PRESSURE_LEVELS}")
    return f"{name}{level}"


# ---------------------------------------------------------------------- time

def ordinal_of(when: dt.datetime) -> int:
    """Floor to the six-hour grid. Reads want the floor; writes want `aligned`."""
    # Converted, not stripped. `replace(tzinfo=None)` on 00:00+03:00 yields an
    # aligned 00:00 that is three hours from where the field belongs — and
    # `aligned_ordinal` would accept it, which is the one thing it exists to
    # stop. `read_api` gets tz-aware datetimes whenever a client sends an offset.
    naive = when.astimezone(dt.timezone.utc).replace(tzinfo=None) if when.tzinfo else when
    return int((naive - EPOCH).total_seconds()) // SECONDS_PER_STEP


def time_of(ordinal: int) -> dt.datetime:
    return EPOCH + dt.timedelta(seconds=ordinal * SECONDS_PER_STEP)


def aligned_ordinal(when: dt.datetime) -> int:
    """As `ordinal_of`, but refuses a timestamp that is not on the grid.

    The write path takes this one. A moment silently floored to 06:00 would be
    stored under a name that says 06:00 and hold the field for 07:13, and
    nothing downstream could ever notice.
    """
    # Converted, not stripped. `replace(tzinfo=None)` on 00:00+03:00 yields an
    # aligned 00:00 that is three hours from where the field belongs — and
    # `aligned_ordinal` would accept it, which is the one thing it exists to
    # stop. `read_api` gets tz-aware datetimes whenever a client sends an offset.
    naive = when.astimezone(dt.timezone.utc).replace(tzinfo=None) if when.tzinfo else when
    if int((naive - EPOCH).total_seconds()) % SECONDS_PER_STEP:
        raise ValueError(f"{when.isoformat()} is not on the {STEP_HOURS}h grid")
    return ordinal_of(when)


def block_of(ordinal: int) -> int:
    return ordinal // config.HISTORY_TIME_CHUNK


def _block_name(block_id: int) -> str:
    return f"b{block_id:07d}"


# ------------------------------------------------------------------ manifest

class Manifest:
    """What the store holds, as a fact separate from what is on disk.

    A block directory exists as soon as its first chunk is written, and the
    six-hourly append writes 69 arrays one after another. So "the directory is
    there" says nothing about whether the moment inside it is complete. The
    manifest is the commit point: it is rewritten only after every field of a
    moment has landed, so a crash halfway leaves `last` where it was and the
    partial moment is simply overwritten on the next attempt.
    """

    def __init__(self, path: Path, first: int | None = None,
                 last: int | None = None, missing: tuple[int, ...] = ()):
        self.path = Path(path)
        self.first = first
        self.last = last
        self.missing = tuple(missing)

    @classmethod
    def load(cls, root: Path) -> "Manifest":
        path = Path(root) / "manifest.json"
        if not path.exists():
            return cls(path)
        rec = json.loads(path.read_text())
        return cls(path, rec.get("first"), rec.get("last"),
                   tuple(rec.get("missing", ())))

    def save(self) -> None:
        rec = {
            "first": self.first,
            "last": self.last,
            "missing": list(self.missing),
            # Derived, and written down anyway: a human reading this file should
            # not have to multiply by six to find out what it covers.
            "first_time": time_of(self.first).isoformat() if self.first is not None else None,
            "truth_through": time_of(self.last).isoformat() if self.last is not None else None,
            "step_hours": STEP_HOURS,
            "block_moments": config.HISTORY_TIME_CHUNK,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        os.replace(tmp, self.path)

    @property
    def empty(self) -> bool:
        return self.first is None or self.last is None

    def __contains__(self, ordinal: int) -> bool:
        if self.empty:
            return False
        return self.first <= ordinal <= self.last and ordinal not in self.missing


# ---------------------------------------------------------------------- read

class HistoryStore:
    """The window, opened for reading. Blocks are opened lazily and kept."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.PRELOAD_ROOT)
        self.manifest = Manifest.load(self.root)
        self.block_moments = config.HISTORY_TIME_CHUNK
        self._blocks: dict[int, zarr.Group] = {}

        # Regular by construction, so it is generated rather than stored. The
        # archive's own axes are checked against these by `contracts.check_axes`
        # on the way in, which is where a grid mismatch belongs.
        self.lat = np.linspace(LAT_FIRST, LAT_LAST, LAT_SIZE)
        self.lon = np.arange(LON_SIZE) * RESOLUTION
        self.fields = FIELDS

    # -------------------------------------------------------------- coverage

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        if self.manifest.empty:
            return None
        return time_of(self.manifest.first), time_of(self.manifest.last)

    @property
    def blocks(self) -> list[int]:
        if self.manifest.empty:
            return []
        return list(range(block_of(self.manifest.first),
                          block_of(self.manifest.last) + 1))

    def units(self, name: str) -> str:
        return (SURFACE.get(name) or ATMOS[name])[0]

    def _group(self, block_id: int) -> zarr.Group:
        g = self._blocks.get(block_id)
        if g is None:
            path = self.root / _block_name(block_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"block {block_id} ({time_of(block_id * self.block_moments)}) "
                    f"is not on disk, though the manifest claims it")
            g = self._blocks[block_id] = zarr.open(str(path), mode="r")
        return g

    # ------------------------------------------------------------ time range

    def ordinals(self, start: dt.datetime | None,
                 end: dt.datetime | None) -> list[int]:
        """The moments this store can actually answer for, in order.

        Clamped to what is held rather than raising: a caller asking for the
        last month when the window starts three weeks ago wants three weeks,
        not an error. What it must not get is silence about the difference —
        the returned list is the honest answer and `coverage` says why.
        """
        if self.manifest.empty:
            return []
        lo = self.manifest.first if start is None else max(
            self.manifest.first, ordinal_of(start))
        hi = self.manifest.last if end is None else min(
            self.manifest.last, ordinal_of(end))
        return [o for o in range(lo, hi + 1) if o not in self.manifest.missing]

    def times(self, ordinals: list[int]) -> list[dt.datetime]:
        return [time_of(o) for o in ordinals]

    # ----------------------------------------------------------------- reads

    def point(self, lat: float, lon: float, fields: list[str] | None = None,
              start: dt.datetime | None = None,
              end: dt.datetime | None = None) -> dict:
        """One grid node over a time range. The query this layout exists for.

        A 220-day series at one point reads one `[88, 32, 32]` chunk per block,
        ten of them, and throws away everything but one column of each. The
        overread is 1024x and it does not matter: the whole thing is 3.6 MB and
        measures 11 ms, against 2.0 s map-major (`bench/results/series.json`).
        """
        ilat, ilon, node_lat, node_lon = self.nearest_point(lat, lon)
        names = self._names(fields)
        ords = self.ordinals(start, end)
        values = self._gather(names, ords, lambda a, s: a[s, ilat, ilon])
        return {
            "point": {"lat": node_lat, "lon": _to_signed(node_lon),
                      "requested": {"lat": lat, "lon": lon}},
            "times": self.times(ords),
            "values": values,
        }

    def box(self, lat0: float, lat1: float, lon0: float, lon1: float,
            fields: list[str] | None = None,
            start: dt.datetime | None = None,
            end: dt.datetime | None = None) -> dict:
        """A region over a time range — a country, not a hemisphere.

        Cost is set by how many 32x32 tiles the box covers, times how many
        blocks the range spans, and every touched chunk is read whole. A box
        that spans the globe here reads the globe for every moment in the
        range; that query belongs to the map-major store.
        """
        ilat0, ilon0, _, _ = self.nearest_point(max(lat0, lat1), lon0)
        ilat1, ilon1, _, _ = self.nearest_point(min(lat0, lat1), lon1)
        names = self._names(fields)
        ords = self.ordinals(start, end)
        lat_sel = slice(ilat0, ilat1 + 1)
        nlat = ilat1 + 1 - ilat0

        if ilon1 < ilon0:
            # The box crosses the prime meridian on a 0..360 grid. Two reads
            # joined on the longitude axis, rather than a wrong empty answer.
            east, west = slice(ilon0, LON_SIZE), slice(0, ilon1 + 1)
            ne, nw = LON_SIZE - ilon0, ilon1 + 1
            values = {
                n: np.concatenate(
                    [self._gather([n], ords, lambda a, s: a[s, lat_sel, east],
                                  (nlat, ne))[n],
                     self._gather([n], ords, lambda a, s: a[s, lat_sel, west],
                                  (nlat, nw))[n]],
                    axis=-1)
                for n in names
            }
            lons = np.concatenate([self.lon[east], self.lon[west]])
        else:
            lon_sel = slice(ilon0, ilon1 + 1)
            values = self._gather(names, ords, lambda a, s: a[s, lat_sel, lon_sel],
                                  (nlat, ilon1 + 1 - ilon0))
            lons = self.lon[lon_sel]

        return {
            "times": self.times(ords),
            "lat": self.lat[lat_sel].tolist(),
            "lon": [_to_signed(v) for v in lons],
            "values": values,
        }

    def map_at(self, field: str, when: dt.datetime) -> np.ndarray:
        """One whole map. Here so callers need not reach into a block, but this
        is the wrong store for it: a global map touches all 23 x 45 tiles and
        each carries 88 moments to return one. `/v1/map` reads the map-major
        forecast store, and historical maps come through the ARCO proxy."""
        o = ordinal_of(when)
        if o not in self.manifest:
            raise KeyError(f"{when.isoformat()} is outside the window {self.coverage}")
        arr = self._group(block_of(o))[field]
        return np.asarray(arr[o % self.block_moments], dtype="float32")

    # --------------------------------------------------------------- helpers

    def nearest_point(self, lat: float, lon: float) -> tuple[int, int, float, float]:
        """(ilat, ilon, node lat, node lon) — the node, not what was asked for.

        Same contract as `SeriesStore.nearest_point`, and for the same reason:
        the grid is 0.25 degrees, up to 14 km at the equator, and answering for
        a different place than the caller named looks fine right up until
        somebody compares against a station.
        """
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude {lat} outside [-90, 90]")
        ilat = int(np.abs(self.lat - lat).argmin())
        ilon = int(np.abs(self.lon - (lon % 360.0)).argmin())
        return ilat, ilon, float(self.lat[ilat]), float(self.lon[ilon])

    def _names(self, fields) -> list[str]:
        if fields is None:
            return list(self.fields)
        unknown = [f for f in fields if f not in self.fields]
        if unknown:
            raise KeyError(f"{unknown} not in store (69 fields, see FIELDS)")
        return list(fields)

    def _gather(self, names: list[str], ordinals: list[int], take,
                trailing: tuple[int, ...] = ()) -> dict:
        """Run `take` per block and join on the time axis.

        The whole point of the block layout shows up here: a range crossing a
        block boundary is two reads and a concatenate, and the caller never
        learns that boundaries exist. `take` gets the zarr array and a slice in
        *block-local* index space.

        `trailing` is the shape after the time axis, and it is only ever used
        for the empty answer: a range wholly outside the window still has to
        come back with the caller's rank, or `box` would return a 1-D array
        against a 3-D `lat`/`lon` contract and the client would see the failure
        as a malformed response rather than as an empty one.
        """
        if not ordinals:
            return {n: np.empty((0, *trailing), dtype="float32") for n in names}

        runs = _contiguous_runs(ordinals, self.block_moments)
        out: dict[str, np.ndarray] = {}
        for name in names:
            pieces = []
            for block_id, lo, hi in runs:
                arr = self._group(block_id)[name]
                pieces.append(np.asarray(take(arr, slice(lo, hi)), dtype="float32"))
            out[name] = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        return out


def _contiguous_runs(ordinals: list[int], block_moments: int):
    """Ordinals -> [(block_id, lo, hi)] with lo:hi block-local and contiguous.

    Two things get split: a block boundary, and a gap where a cycle failed and
    the ordinal is in `missing`. Both produce a separate zarr read, so a run of
    missed moments costs extra requests and never returns a slot nobody wrote.
    """
    runs = []
    for o in ordinals:
        block_id, slot = divmod(o, block_moments)
        if runs and runs[-1][0] == block_id and runs[-1][2] == slot:
            runs[-1][2] = slot + 1
        else:
            runs.append([block_id, slot, slot + 1])
    return [tuple(r) for r in runs]


def _to_signed(lon: float) -> float:
    """0..360 -> -180..180. Only ever at the boundary, never in the model path."""
    lon = float(lon) % 360.0
    return lon - 360.0 if lon > 180.0 else lon


# --------------------------------------------------------------------- write

class HistoryWriter:
    """The six-hourly append, and the fill that creates the window.

    Read and write live in one module because they share the ordinal scheme and
    the manifest format, and a second copy of either is a second thing to get
    wrong. They do not share a zarr handle: a writer opens `mode="a"`.
    """

    def __init__(self, root: Path | None = None, keep_days: int | None = None):
        self.root = Path(root or config.PRELOAD_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest.load(self.root)
        self.block_moments = config.HISTORY_TIME_CHUNK
        self.keep = (keep_days or config.PRELOAD_DAYS) * 24 // STEP_HOURS

    def _ensure_block(self, block_id: int) -> zarr.Group:
        """Create the block if it is not there, atomically.

        Built under a `.tmp` name and renamed once every array exists, so a
        crash during creation cannot leave a directory that a reader would take
        for a real block. Renaming a directory within a filesystem is atomic;
        this is the same trick `publish.py` uses for a forecast run.
        """
        final = self.root / _block_name(block_id)
        if final.exists():
            return zarr.open(str(final), mode="a")

        tmp = self.root / f"{_block_name(block_id)}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        g = zarr.open_group(str(tmp), mode="w")
        for name in FIELDS:
            g.create_array(
                name,
                shape=(self.block_moments, LAT_SIZE, LON_SIZE),
                chunks=(self.block_moments, TILE, TILE),
                dtype="float32",
                fill_value=float("nan"),
                compressors=[BLOSC],
            )
        g.attrs["block_id"] = block_id
        g.attrs["first_time"] = time_of(block_id * self.block_moments).isoformat()
        os.replace(tmp, final)
        return zarr.open(str(final), mode="a")

    def write(self, when: dt.datetime, fields: dict[str, np.ndarray]) -> int:
        """One moment, all 69 fields. Returns the ordinal it landed on.

        **The six-hourly append, and only that.** Writing one slot of an
        `[88, 32, 32]` chunk makes zarr decompress, merge and recompress it, for
        every tile of every field: 25 GB of read-modify-write, measured at 98 s
        (`bench/results/edge_probe.json`). That is nothing against a 21 600 s
        cycle and ruinous against 880 of them — the initial fill goes through
        `write_block`, which compresses each chunk exactly once.

        All 69 or none: a moment with a missing field would be indistinguishable
        from a complete one on read, because the gap would be NaN in a store
        where NaN already means "never written". The manifest advances only
        after the last array is in.
        """
        o = aligned_ordinal(when)
        missing = [f for f in FIELDS if f not in fields]
        if missing:
            raise KeyError(f"moment {when.isoformat()} is missing {len(missing)} "
                           f"fields, first {missing[:3]}")

        g = self._ensure_block(block_of(o))
        slot = o % self.block_moments
        for name in FIELDS:
            arr = np.asarray(fields[name], dtype="float32")
            if arr.shape != (LAT_SIZE, LON_SIZE):
                raise ValueError(f"{name} is {arr.shape}, want {(LAT_SIZE, LON_SIZE)}")
            g[name][slot] = arr

        self._commit([o])
        return o

    def write_block(self, block_id: int, stacks: dict[str, np.ndarray]) -> list[int]:
        """A whole block in one pass — the fill path. Returns the ordinals written.

        `stacks[name]` is `[n, lat, lon]` starting at the block's first slot,
        with `n <= HISTORY_TIME_CHUNK`. Because the block is created here and
        every chunk is written once from slot 0, zarr has nothing to read back
        and merge: each `[88, 32, 32]` tile is compressed exactly once. That is
        the difference between an eight-hour fill and a day of it.

        A short `n` is allowed for the tail of the fill, and it costs nothing
        extra on this pass — the missing slots stay at the fill value, which is
        NaN, and land in `missing`. It does mean the next append into those
        slots pays the ordinary read-modify-write, which is what `write` is for.

        Refuses to touch an existing block: this is a create-and-fill, and
        letting it overwrite would silently discard whatever a previous run had
        already put there.
        """
        final = self.root / _block_name(block_id)
        if final.exists():
            raise FileExistsError(
                f"block {block_id} ({time_of(block_id * self.block_moments)}) "
                f"already exists; use write() to append into it")

        missing = [f for f in FIELDS if f not in stacks]
        if missing:
            raise KeyError(f"block {block_id} is missing {len(missing)} fields, "
                           f"first {missing[:3]}")

        n = int(np.asarray(stacks[FIELDS[0]]).shape[0])
        if not 0 < n <= self.block_moments:
            raise ValueError(f"{n} moments, want 1..{self.block_moments}")

        g = self._ensure_block(block_id)
        for name in FIELDS:
            arr = np.asarray(stacks[name], dtype="float32")
            if arr.shape != (n, LAT_SIZE, LON_SIZE):
                raise ValueError(
                    f"{name} is {arr.shape}, want {(n, LAT_SIZE, LON_SIZE)}")
            g[name][:n] = arr

        base = block_id * self.block_moments
        ords = list(range(base, base + n))
        self._commit(ords)
        return ords

    def _commit(self, ordinals: list[int]) -> None:
        m = self.manifest
        lo, hi = min(ordinals), max(ordinals)
        if m.empty:
            m.first, m.last = lo, hi
            m.missing = tuple(o for o in range(lo, hi + 1) if o not in set(ordinals))
        else:
            # A backfill lands inside the window and only clears a gap; the
            # six-hourly append extends it and opens one if a cycle was lost.
            if hi > m.last:
                gap = tuple(range(m.last + 1, hi))
                m.missing = tuple(sorted(set(m.missing) | set(gap)))
                m.last = hi
            m.first = min(m.first, lo)
            written = set(ordinals)
            m.missing = tuple(o for o in m.missing if o not in written)
        m.save()

    def evict(self) -> list[int]:
        """Drop whole blocks that have fallen out of the window. Returns their ids.

        Only whole blocks, because a partial one would have to be rewritten to
        free anything and rewriting 17 GB of compressed chunks to reclaim
        200 MB is not a trade.

        So the window overshoots, and by a full block rather than by an average:
        block `B-10` only goes once `m.last` passes the last ordinal of block
        `B`, so at the instant `B` fills, eleven whole blocks are on disk. That
        is 968 moments, not 880 — **189 GB at the peak against 172 GB nominal**,
        and the disk budget carries the peak. The cache is the elastic term:
        300 - 189 - 8.7 leaves it about 102 GB.
        """
        m = self.manifest
        if m.empty:
            return []
        cutoff = m.last - self.keep
        dropped = []
        for block_id in list(range(block_of(m.first), block_of(m.last) + 1)):
            last_in_block = (block_id + 1) * self.block_moments - 1
            if last_in_block >= cutoff:
                break
            shutil.rmtree(self.root / _block_name(block_id), ignore_errors=True)
            dropped.append(block_id)

        if dropped:
            new_first = (dropped[-1] + 1) * self.block_moments
            m.first = max(m.first, new_first)
            m.missing = tuple(o for o in m.missing if o >= m.first)
            m.save()
        return dropped

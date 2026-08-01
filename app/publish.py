"""Turn one rollout into a forecast the read API can serve, and make it live.

`postprocess.ForecastWriter` writes the *analysis* artefact: every variable, every
level, one chunk per (lead, level), named after the job that produced it. That is
the right shape for downloading a forecast and the wrong shape for serving one —
it has no valid-time axis, so a caller asking "what is the temperature in Moscow
on Thursday" has to know which run and which lead that is.

This writes the *serving* artefact instead, in the two layouts the measurements
picked (`bench/results/series_probe.json`):

    <root>/run_20260701T0000/maps.zarr     chunk (1, 721, 1440)  -> /v1/map
    <root>/run_20260701T0000/series.zarr   chunk (44, 32, 32)    -> /v1/point
    <root>/latest -> run_20260701T0000

Three properties this module exists to guarantee:

**Valid time, not lead time.** The axis is `valid_time` in the same units as the
history stores, so `MapStore` and `SeriesStore` read a forecast without knowing
it is one. The seam between history and forecast then lives in one place — the
query planner — instead of in every reader.

**Nothing half-written is ever visible.** The run is built under a `.partial`
name and renamed into place, and `latest` is swapped by renaming a symlink over
itself. Both are atomic on a POSIX filesystem, so a reader either sees the whole
previous run or the whole new one. It never sees a store with nineteen of its
forty-four steps.

**Surface variables only.** Adding the five atmospheric fields on thirteen levels
multiplies the run by 65: 731 MB becomes 48 GB, and at eight runs a day the 300
GB budget is gone before lunch. The full state stays in the job outputs for the
runs somebody asked for by hand.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import zarr
from aurora import Batch

from . import config, contracts

# What a person asks a weather backend for. See the module docstring for why the
# atmospheric fields are not here.
SURFACE_VARS = ("2t", "10u", "10v", "msl")

TIME_COORD = "valid_time"
TIME_UNITS = "hours since 1970-01-01"
EPOCH = np.datetime64("1970-01-01")

BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

# 32x32 from series_probe.json: 16x16 buys 4 ms on a point query and costs four
# times the files, which matters more here than in the archive because a run is
# rewritten from nothing every three hours.
TILE = 32


def _hours(times: np.ndarray) -> np.ndarray:
    return ((np.asarray(times) - EPOCH) / np.timedelta64(1, "h")).astype("int64")


class RunPublisher:
    """One rollout in, one published run out.

    Not a context manager, for the same reason `ForecastWriter` is not: a run
    that died at step 19 should be left on disk under its `.partial` name for
    somebody to look at, not tidied away by an `__exit__`.
    """

    def __init__(self, root: Path, init_time: dt.datetime, steps: int,
                 variables=SURFACE_VARS, tile: int = TILE, source: str = "archive"):
        self.root = Path(root)
        self.init_time = init_time
        self.steps = steps
        self.variables = tuple(variables)
        self.tile = tile
        # Which feed initialised this. Not decoration: the same checkpoint run
        # from ERA5 and from a GFS analysis produces two different forecasts,
        # and a run whose manifest does not say which is a run nobody can
        # compare against anything.
        self.source = source

        self.name = f"run_{init_time:%Y%m%dT%H%M}"
        self.staging = self.root / f".{self.name}.partial"
        self.final = self.root / self.name

        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=True)

        self._maps: zarr.Group | None = None
        self._written = 0
        self._problems: list[str] = []

    # ---------------------------------------------------------------- writing

    def add(self, pred: Batch) -> None:
        """Append one rollout step to the map-major store.

        Map-major only, on purpose. The time-major copy has the whole lead axis
        inside one chunk, so appending to it a step at a time would rewrite the
        entire store forty-four times; it is built once in `finish()` from what
        is already on disk.
        """
        lead = (self._written + 1) * config.STEP_HOURS
        valid = np.datetime64(self.init_time + dt.timedelta(hours=lead))
        fields = {
            name: np.asarray(pred.surf_vars[name].detach().cpu().numpy()[0, -1],
                             dtype="float32")
            for name in self.variables
        }

        if self._maps is None:
            self._maps = self._create_maps(pred)

        report = contracts.validate(
            np.asarray(self._maps["latitude"][:]),
            np.asarray(self._maps["longitude"][:]),
            np.asarray([valid]),
            {k: v[None] for k, v in fields.items()},
        )
        self._problems += report["problems"]
        contracts.require(report)

        n0 = self._maps[TIME_COORD].shape[0]
        self._maps[TIME_COORD].resize((n0 + 1,))
        self._maps[TIME_COORD][n0] = _hours(np.asarray([valid]))[0]
        for name, arr in fields.items():
            a = self._maps[name]
            a.resize((n0 + 1, *a.shape[1:]))
            a[n0] = arr
        self._written += 1

    def _create_maps(self, pred: Batch) -> zarr.Group:
        lat = np.asarray(pred.metadata.lat, dtype="float64")
        lon = np.asarray(pred.metadata.lon, dtype="float64")
        g = zarr.open_group(str(self.staging / "maps.zarr"), mode="w")
        for name in self.variables:
            a = g.create_array(name, shape=(0, len(lat), len(lon)),
                               chunks=(1, len(lat), len(lon)),
                               dtype="float32", compressors=[BLOSC])
            a.attrs["units"] = contracts.SURFACE[name][0]
        g.create_array("latitude", shape=lat.shape, chunks=lat.shape,
                       dtype="float64")[:] = lat
        g.create_array("longitude", shape=lon.shape, chunks=lon.shape,
                       dtype="float64")[:] = lon
        t = g.create_array(TIME_COORD, shape=(0,), chunks=(1024,), dtype="int64")
        t.attrs.update({"units": TIME_UNITS, "calendar": "proleptic_gregorian"})
        return g

    def _build_series(self) -> None:
        """The time-major copy, written once from the map-major one.

        A whole run is 183 MB per variable, so this reads and writes one field
        at a time rather than holding the four together.
        """
        maps = zarr.open(str(self.staging / "maps.zarr"), mode="r")
        lat = np.asarray(maps["latitude"][:])
        lon = np.asarray(maps["longitude"][:])
        g = zarr.open_group(str(self.staging / "series.zarr"), mode="w")
        for name in self.variables:
            src = maps[name]
            a = g.create_array(name, shape=src.shape,
                               chunks=(src.shape[0], self.tile, self.tile),
                               dtype="float32", compressors=[BLOSC])
            a.attrs["units"] = contracts.SURFACE[name][0]
            a[:] = src[:]
        g.create_array("latitude", shape=lat.shape, chunks=lat.shape,
                       dtype="float64")[:] = lat
        g.create_array("longitude", shape=lon.shape, chunks=lon.shape,
                       dtype="float64")[:] = lon
        t = g.create_array(TIME_COORD, shape=maps[TIME_COORD].shape,
                           chunks=(1024,), dtype="int64")
        t[:] = maps[TIME_COORD][:]
        t.attrs.update({"units": TIME_UNITS, "calendar": "proleptic_gregorian"})

    # -------------------------------------------------------------- publishing

    def finish(self, keep: int = 4) -> Path:
        """Build the second layout, write the sidecars, make the run live."""
        if self._written == 0:
            raise ValueError("rollout produced no steps")
        self._build_series()

        maps = zarr.open(str(self.staging / "maps.zarr"), mode="r")
        valid = EPOCH + np.asarray(maps[TIME_COORD][:]).astype("timedelta64[h]")
        # Read back rather than restated: Aurora's output grid is 720 rows, not
        # the archive's 721, and a manifest that says otherwise is worse than one
        # that says nothing.
        shape = list(maps[self.variables[0]].shape)
        manifest = {
            "run": self.name,
            "init_time": self.init_time.isoformat(),
            "model": config.MODEL_NAME,
            "source": self.source,
            "precision": config.AUTOCAST,
            "steps": self._written,
            "lead_hours": self._written * config.STEP_HOURS,
            "variables": list(self.variables),
            "valid_from": str(valid[0]),
            "valid_to": str(valid[-1]),
            "grid": {"shape": shape[1:], "kind": contracts.grid_name(
                np.asarray(maps["latitude"][:]))},
            "layouts": {
                "maps.zarr": {"chunks": [1, *shape[1:]], "answers": "/v1/map"},
                "series.zarr": {"chunks": [self._written, self.tile, self.tile],
                                "answers": "/v1/point"},
            },
            "size_mb": round(_size_mb(self.staging), 1),
            "published_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        (self.staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.staging / "validation.json").write_text(json.dumps({
            "ok": not self._problems,
            "problems": self._problems,
            "checked": "axes, time monotonicity and step, per-variable physical range",
        }, indent=2))

        if self.final.exists():
            shutil.rmtree(self.final)
        os.rename(self.staging, self.final)
        publish_latest(self.root, self.final)
        evict(self.root, keep)
        return self.final


def publish_latest(root: Path, run: Path) -> None:
    """Point `latest` at `run` without ever being unset.

    `ln -sfn` unlinks and recreates, so a reader between the two calls finds no
    `latest` at all. Renaming a symlink over the old one is a single atomic
    operation and has no such window.
    """
    root = Path(root)
    tmp = root / f".latest.{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path(run).name)
    os.replace(tmp, root / "latest")


def runs(root: Path) -> list[Path]:
    """Published runs, oldest first. Partial ones are not published."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("run_") and not p.name.startswith("."))


def evict(root: Path, keep: int) -> list[Path]:
    """Drop the oldest runs, never the one `latest` points at.

    Keeping a few is not sentiment: comparing the new run against the previous
    one is how you notice that the input feed changed shape, and a store with
    only ever one run in it cannot answer "was it like this three hours ago".
    """
    root = Path(root)
    live = (root / "latest").resolve() if (root / "latest").exists() else None
    old = runs(root)[:-keep] if keep > 0 else []
    dropped = []
    for path in old:
        if live is not None and path.resolve() == live:
            continue
        shutil.rmtree(path)
        dropped.append(path)
    return dropped


def _size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1024**2

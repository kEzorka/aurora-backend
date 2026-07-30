"""Turn rollout steps into a forecast on disk.

zarr is the default and it is written a step at a time: each prediction is
appended along `lead_time` and dropped, so peak host memory is one step (~290
MB) instead of the whole forecast. NetCDF has no append story worth the
complexity, so that path still assembles everything in memory first — it is
kept for consumers that want a single file.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import zarr
from aurora import Batch

from . import config

TIME_DIM = "lead_time"
LEVEL_DIM = "pressure_level"


def _last_step(t) -> np.ndarray:
    """Drop the batch axis and keep the final history entry.

    A prediction comes back as (batch, time, ...) with time of length 1, but we
    index from the end so a model that returns more history still works.
    """
    arr = t.detach().cpu().numpy()
    return arr[0, -1]


def step_dataset(pred: Batch, lead_hours: int) -> xr.Dataset:
    """One rollout step as a Dataset with a length-1 `lead_time` axis."""
    coords = {
        TIME_DIM: (TIME_DIM, np.asarray([lead_hours], dtype="int32")),
        LEVEL_DIM: (LEVEL_DIM, np.asarray(pred.metadata.atmos_levels, dtype="int32")),
        "latitude": ("latitude", pred.metadata.lat.numpy()),
        "longitude": ("longitude", pred.metadata.lon.numpy()),
    }
    data = {
        name: ((TIME_DIM, "latitude", "longitude"), _last_step(t)[None].astype("float32"))
        for name, t in pred.surf_vars.items()
    }
    data |= {
        name: (
            (TIME_DIM, LEVEL_DIM, "latitude", "longitude"),
            _last_step(t)[None].astype("float32"),
        )
        for name, t in pred.atmos_vars.items()
    }
    return xr.Dataset(data, coords=coords)


def _tag(ds: xr.Dataset, init_time: dt.datetime) -> xr.Dataset:
    # Must not read as a CF time unit: anything containing "since" makes
    # xarray try to decode this axis into datetimes and fail on reopen.
    ds[TIME_DIM].attrs["units"] = "hours"
    ds[TIME_DIM].attrs["long_name"] = "forecast lead time from init_time"
    ds.attrs["init_time"] = init_time.isoformat()
    ds.attrs["model"] = config.MODEL_NAME
    return ds


# The forecast codec, and the only place in app/ that names one. It has to be
# spelled out: zarr 2 defaulted to exactly this, so the old code got it by saying
# nothing, and zarr 3 defaults to Zstd without shuffle. Saying nothing here again
# would quietly re-compress every forecast the backend writes — the same trap as
# the archive conversion, in the opposite direction.
FORECAST_CODEC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

# zarr 3 writes chunks through an async pipeline, 10 at a time by default. A
# rollout step is 69 small writes to a local disk, which gets nothing from that
# fan-out and pays for the contention; 4 measured faster on this box. Set at
# import because the setting is global to zarr, not per-store.
zarr.config.set({"async.concurrency": 4})


def _zarr_encoding(ds: xr.Dataset) -> dict:
    """One chunk per (lead, level) field — the shape a reader asks for."""
    return {
        name: {
            "chunks": tuple(
                1 if d in (TIME_DIM, LEVEL_DIM) else var.sizes[d] for d in var.dims
            ),
            "compressors": [FORECAST_CODEC],
        }
        for name, var in ds.data_vars.items()
    }


def output_path(
    init_time: dt.datetime,
    steps: int,
    out_dir: Path | None = None,
    precision: str | None = None,
) -> Path:
    """Where a forecast lands.

    Everything that changes the contents has to be in the name. The lead does
    for the obvious reason — two jobs with the same init and different step
    counts would write to one path, and the shorter one silently truncates a
    finished forecast another job still holds a download URL for. The precision
    does for the same reason and it is easier to miss: an fp32 reference run and
    the fp16 exploration it is meant to be compared against share both the init
    and the lead, so without it the reference overwrites the thing it scores.
    With both in the name, a collision means the contents match.
    """
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    lead = steps * config.STEP_HOURS
    prec = precision or config.AUTOCAST
    suffix = "zarr" if config.OUTPUT_FORMAT == "zarr" else "nc"
    return out_dir / f"forecast_{init_time:%Y%m%dT%H%M}_{lead:03d}h_{prec}.{suffix}"


class ForecastWriter:
    """Consume rollout steps, leave a forecast on disk.

    Deliberately not a context manager: a failed rollout should leave the
    partial store behind for inspection rather than tidy the evidence away.
    """

    def __init__(
        self,
        init_time: dt.datetime,
        steps: int,
        out_dir: Path | None = None,
        precision: str | None = None,
    ):
        self.init_time = init_time
        self.path = output_path(init_time, steps, out_dir, precision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.format = config.OUTPUT_FORMAT
        self._written = 0
        self._pending: list[xr.Dataset] = []

        if self.path.exists():
            # Same init and same lead, so the contents would match — but a
            # half-written store from a crashed run would not. Start clean.
            shutil.rmtree(self.path) if self.path.is_dir() else self.path.unlink()

    def add(self, pred: Batch) -> None:
        lead = (self._written + len(self._pending) + 1) * config.STEP_HOURS
        ds = _tag(step_dataset(pred, lead), self.init_time)
        if self.format != "zarr":
            self._pending.append(ds)
            return
        if self._written == 0:
            ds.to_zarr(self.path, mode="w", encoding=_zarr_encoding(ds))
        else:
            # Encoding is fixed by the first write; appends inherit it.
            ds.to_zarr(self.path, append_dim=TIME_DIM)
        self._written += 1

    def finish(self) -> Path:
        if self.format == "zarr":
            if self._written == 0:
                raise ValueError("rollout produced no steps")
            return self.path
        if not self._pending:
            raise ValueError("rollout produced no steps")
        ds = xr.concat(self._pending, dim=TIME_DIM, combine_attrs="override")
        encoding = {name: {"zlib": True, "complevel": 1} for name in ds.data_vars}
        ds.to_netcdf(self.path, encoding=encoding)
        self._pending.clear()
        return self.path


def open_forecast(path: str | Path) -> xr.Dataset:
    """Reopen a forecast written by either backend.

    `lead_time` is an offset in hours, not a date and not a duration; both
    decoders have to be off or xarray converts the axis behind your back.
    """
    path = Path(path)
    kwargs = dict(decode_times=False, decode_timedelta=False)
    if path.is_dir():
        return xr.open_dataset(path, engine="zarr", chunks=None, **kwargs)
    return xr.open_dataset(path, **kwargs)


def write_all(
    preds: Iterable[Batch], init_time: dt.datetime, steps: int, out_dir: Path | None = None
) -> Path:
    """Convenience wrapper for callers that just want the whole thing written."""
    writer = ForecastWriter(init_time, steps, out_dir)
    for pred in preds:
        writer.add(pred)
    return writer.finish()

"""The 0.22 s that zarr 3 adds per step, and whether it is a fixed cost.

py312_zarr3.json split the write penalty into ~0.43 s of zstd-instead-of-lz4
and ~0.22 s of zarr 3's own path. The first is a codec choice we already
reversed. The second was recorded as if it were unavoidable, but zarr 3 writes
through an async pipeline with a configurable concurrency limit, so it may just
be a default.

Also settles region writes, the untested half of the warning in
requirements.txt: whether a store can be allocated once and filled step by step
at an index, which is what a parallel writer would need.

No GPU: the steps are replayed from an existing forecast, so this measures the
write path and nothing else.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

MB = 1024**2
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
SRC = "bench/scratch/py312-blosc/forecast_20260405T0000_024h.zarr"
OUT = Path("bench/scratch/writepath")
TIME_DIM, LEVEL_DIM = "lead_time", "pressure_level"

CONCURRENCY = (None, 4, 16, 64)


def encoding(ds):
    return {
        name: {
            "chunks": tuple(1 if d in (TIME_DIM, LEVEL_DIM) else var.sizes[d]
                            for d in var.dims),
            "compressors": [BLOSC],
        }
        for name, var in ds.data_vars.items()
    }


def append_run(src: xr.Dataset, path: Path) -> list[float]:
    """First write plus appends, exactly what ForecastWriter.add does."""
    if path.exists():
        shutil.rmtree(path)
    times = []
    for i in range(src.sizes[TIME_DIM]):
        step = src.isel({TIME_DIM: [i]})
        t0 = time.perf_counter()
        if i == 0:
            step.to_zarr(path, mode="w", consolidated=False, encoding=encoding(step))
        else:
            step.to_zarr(path, append_dim=TIME_DIM, consolidated=False)
        times.append(round(time.perf_counter() - t0, 3))
    return times


def region_run(src: xr.Dataset, path: Path) -> tuple[list[float], bool]:
    """Allocate the whole store first, then fill one lead_time at a time.

    This is what a writer that does not own the time axis has to do — several
    processes, or steps arriving out of order. `compute=False` writes metadata
    and coordinates only; each later call fills its own slice.
    """
    if path.exists():
        shutil.rmtree(path)
    n = src.sizes[TIME_DIM]
    template = src.copy()
    for name, var in template.data_vars.items():
        template[name] = (var.dims, np.zeros(var.shape, dtype=var.dtype))
    template.to_zarr(path, mode="w", consolidated=False, compute=False,
                     encoding=encoding(template))

    times = []
    for i in range(n):
        step = src.isel({TIME_DIM: slice(i, i + 1)}).drop_vars(
            [TIME_DIM, LEVEL_DIM, "latitude", "longitude"]
        )
        t0 = time.perf_counter()
        step.to_zarr(path, region={TIME_DIM: slice(i, i + 1)}, consolidated=False)
        times.append(round(time.perf_counter() - t0, 3))

    back = xr.open_zarr(path, consolidated=False)
    ok = bool(np.allclose(back["t"].values, src["t"].values, equal_nan=True))
    back.close()
    return times, ok


def main() -> int:
    src = xr.open_zarr(SRC, consolidated=True).load()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"{'async.concurrency':<20} {'per-step append s':>32}  mean")
    for c in CONCURRENCY:
        ctx = zarr.config.set({"async.concurrency": c}) if c else _null()
        with ctx:
            times = append_run(src, OUT / f"append_c{c}")
        label = "default" if c is None else str(c)
        # First step allocates the store; the appends are the steady state.
        mean = sum(times[1:]) / len(times[1:])
        results[f"append concurrency={label}"] = {"per_step_s": times,
                                                  "steady_mean_s": round(mean, 3)}
        print(f"{label:<20} {str(times):>32}  {mean:.3f}s")

    times, ok = region_run(src, OUT / "region")
    mean = sum(times) / len(times)
    results["region writes"] = {"per_step_s": times, "mean_s": round(mean, 3),
                                "roundtrip_matches": ok}
    print(f"\nregion write   {times}  mean {mean:.3f}s  values match: {ok}")

    print(f"\nzarr {zarr.__version__}  default async.concurrency = "
          f"{zarr.config.get('async.concurrency')}")

    out = Path("bench/results/zarr3_writepath.json")
    out.write_text(json.dumps({
        "source": SRC,
        "zarr": zarr.__version__,
        "default_async_concurrency": zarr.config.get("async.concurrency"),
        "runs": results,
    }, indent=2))
    print(f"wrote {out}")
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())

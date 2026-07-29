"""Is sharding slow, or is dask slow?

shard_bench read through `xr.open_zarr`, which hands back dask arrays chunked
to match the store. Tiling the store 16x also multiplies the dask graph by 16,
so a slowdown there could be scheduler overhead rather than anything zarr does.
This reads the same stores three ways — dask, xarray without dask, and the zarr
array directly — so the two costs can be told apart.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

ROOT = Path("bench/scratch/shards")
STORES = {
    "map chunk (today)": ROOT / "map_chunk_(today)",
    "shard 180x360": ROOT / "shard_180x360_tiles",
    "shard 90x180": ROOT / "shard_90x180_tiles",
}
# Europe in index space: latitude 72..34 is rows 72..224, longitude -25..45
# wraps, so take the eastern part only — the point here is decode volume, not
# the wrap, which both layouts pay equally.
LAT = slice(72, 224)
LON = slice(0, 180)


def timeit(fn, reps=3) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


print(f"{'store':<20} {'via':<10} {'full t':>9} {'EU t':>9} {'one map':>9}")
for label, path in STORES.items():
    dsk = xr.open_zarr(path, consolidated=False)
    eager = xr.open_zarr(path, consolidated=False, chunks=None)
    raw = zarr.open(str(path), mode="r")["t"]

    rows = {
        "dask": (
            lambda: dsk["t"].values,
            lambda: dsk["t"][:, :, LAT, LON].values,
            lambda: dsk["2t"].isel(lead_time=0).values,
        ),
        "xarray": (
            lambda: eager["t"].values,
            lambda: eager["t"][:, :, LAT, LON].values,
            lambda: eager["2t"].isel(lead_time=0).values,
        ),
        "zarr": (
            lambda: raw[:],
            lambda: raw[:, :, LAT, LON],
            lambda: zarr.open(str(path), mode="r")["2t"][0],
        ),
    }
    for via, (full, eu, one) in rows.items():
        print(f"{label:<20} {via:<10} {timeit(full):>8.2f}s {timeit(eu):>8.2f}s "
              f"{timeit(one, reps=5) * 1e3:>7.0f}ms")
    dsk.close()
    eager.close()

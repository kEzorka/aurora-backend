"""Does the preload fill go faster with more workers, or is the link already full?

`preload_probe.py` measured 34.2 s to pull one moment's 69 fields and found why:
ARCO chunks the pressure-level arrays `[1, 37, 721, 1440]`, so the 13 levels
Aurora wants arrive inside all 37. Per moment that is 784 MB on the wire to keep
286 MB — and 784 MB in 34.2 s is 23 MB/s, which looks much more like a link than
like a per-request cost.

That distinction decides the shape of the backfill. If the 23 MB/s is one
connection's share of a fatter link, workers multiply and 220 days is an hour.
If it is the link, workers do nothing and 8.4 h serial is the real number, so
the fill has to survive being interrupted — a manifest, resumability, and a
sane story about what a half-filled window serves.

Runs the same fetch at several worker counts and reports the aggregate rate.
Distinct moments per worker, so nothing is answered from a cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .preload_probe import ATMOS, LEVELS, SURFACE, URL


def one_moment(ds, t: dt.datetime) -> float:
    """Fetch all 69 fields for one moment. Returns megabytes off the wire."""
    stamp = np.datetime64(t, "ns")
    wire = 0.0
    for full in SURFACE.values():
        np.asarray(ds[full].sel(time=stamp).values, dtype="float32")
        wire += 721 * 1440 * 4 / 1e6
    for full in ATMOS.values():
        np.asarray(ds[full].sel(time=stamp, level=list(LEVELS)).values, dtype="float32")
        # 37, not 13: the chunk is the unit of transfer.
        wire += 37 * 721 * 1440 * 4 / 1e6
    return wire


def main(args) -> dict:
    import xarray as xr

    ds = xr.open_zarr(URL, chunks=None)
    base = dt.datetime.fromisoformat(args.when)

    rows = []
    offset = 0
    for n in args.workers:
        # Fresh moments for every trial: repeating a moment would measure the
        # HTTP cache in front of the bucket rather than the fill.
        moments = [base + dt.timedelta(hours=6 * (offset + i)) for i in range(n)]
        offset += n

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            wire = sum(pool.map(lambda t: one_moment(ds, t), moments))
        elapsed = time.perf_counter() - t0

        row = {
            "workers": n,
            "moments": n,
            "seconds": round(elapsed, 1),
            "seconds_per_moment": round(elapsed / n, 1),
            "megabytes": round(wire, 1),
            "megabytes_per_second": round(wire / elapsed, 1),
            "hours_for_220_days": round(elapsed / n * 880 / 3600, 1),
        }
        rows.append(row)
        print(f"  w={n:2d}  {row['seconds']:6.1f} s  "
              f"{row['megabytes_per_second']:6.1f} MB/s  "
              f"{row['hours_for_220_days']:5.1f} h for 220 d", flush=True)

    return {"url": URL, "trials": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--when", default="1990-06-01T00:00")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--out", default="bench/results/preload_parallel.json")
    a = p.parse_args()
    Path(a.out).write_text(json.dumps(main(a), indent=2))

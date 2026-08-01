"""What one day of the preloaded window actually costs to fetch from ARCO.

`PRELOAD_DAYS = 220` is a number in `config.py` and nothing reads it yet, because
nothing knows what filling it costs. This probe answers that with one day.

Two things are in question and only one of them is obvious.

The obvious one is wall clock: 220 days at the six-hourly cadence is 880 moments,
and a cold ARCO map measured 1.30 s. If every one of the 69 fields is a separate
chunk read, the fill is 880 x 69 x 1.30 s = 22 hours before any parallelism, and
the preload is not a script somebody runs — it is a resumable backfill with a
manifest.

The one that decides it is the chunk shape on the pressure-level arrays. ARCO's
surface fields are `[1, 721, 1440]`, one chunk per map, which is why
`app/sources/arco.py` can proxy them cheaply. If `temperature` is chunked
`[1, 37, 721, 1440]` then asking for the 13 levels Aurora wants still transfers
all 37 — 3x the bytes for the same data — and the fill is bandwidth-bound rather
than request-bound. That is a different bottleneck with a different fix, so it is
measured rather than assumed.

Reports per moment, per variable group, and extrapolates to `PRELOAD_DAYS`.
Writes nothing to the archive: this measures the read side only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

URL = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
       "full_37-1h-0p25deg-chunk-1.zarr-v3")

# The contract's 69: four surface fields, five atmospheric ones on 13 levels.
SURFACE = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def probe(args) -> dict:
    import xarray as xr

    t0 = time.perf_counter()
    ds = xr.open_zarr(URL, chunks=None)
    open_s = time.perf_counter() - t0

    # The fact the extrapolation turns on. `chunks=None` gives numpy-backed
    # arrays, so the encoding is where the chunk shape survives.
    shapes = {}
    for short, full in list(SURFACE.items()) + list(ATMOS.items()):
        enc = ds[full].encoding
        chunks = enc.get("chunks") or list(enc.get("preferred_chunks", {}).values())
        shapes[short] = {
            "dims": list(ds[full].dims),
            "shape": [int(n) for n in ds[full].shape],
            "chunks": [int(n) for n in (chunks or [])],
            "dtype": str(ds[full].dtype),
            "compressor": str(enc.get("compressor") or enc.get("compressors")),
        }

    when = dt.datetime.fromisoformat(args.when)
    marks = []
    for step in range(args.moments):
        t = when + dt.timedelta(hours=6 * step)
        stamp = np.datetime64(t, "ns")

        row = {"time": t.isoformat(), "fields": {}}
        for short, full in SURFACE.items():
            a = time.perf_counter()
            arr = np.asarray(ds[full].sel(time=stamp).values, dtype="float32")
            row["fields"][short] = {
                "seconds": round(time.perf_counter() - a, 3),
                "megabytes": round(arr.nbytes / 1e6, 2),
                "finite": bool(np.isfinite(arr).all()),
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
            }

        for short, full in ATMOS.items():
            a = time.perf_counter()
            arr = np.asarray(
                ds[full].sel(time=stamp, level=list(LEVELS)).values, dtype="float32")
            row["fields"][short] = {
                "seconds": round(time.perf_counter() - a, 3),
                "megabytes": round(arr.nbytes / 1e6, 2),
                "levels": int(arr.shape[0]),
                "finite": bool(np.isfinite(arr).all()),
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
            }

        row["seconds"] = round(sum(f["seconds"] for f in row["fields"].values()), 3)
        row["megabytes"] = round(sum(f["megabytes"] for f in row["fields"].values()), 2)
        marks.append(row)
        print(f"  {t:%Y-%m-%d %H:%M}  {row['seconds']:7.1f} s  "
              f"{row['megabytes']:7.1f} MB", file=sys.stderr, flush=True)

    # Extrapolation. Deliberately the median and not the mean: the first moment
    # pays for connection setup and would flatter or spoil the estimate
    # depending on which way it went.
    per = sorted(m["seconds"] for m in marks)[len(marks) // 2]
    moments = args.days * 4
    return {
        "url": URL,
        "open_seconds": round(open_s, 2),
        "encoding": shapes,
        "moments": marks,
        "extrapolation": {
            "preload_days": args.days,
            "moments_total": moments,
            "median_seconds_per_moment": per,
            "serial_hours": round(per * moments / 3600, 1),
            "megabytes_per_moment": marks[0]["megabytes"],
            "gigabytes_total_raw": round(marks[0]["megabytes"] * moments / 1000, 1),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--when", default="1990-03-15T12:00")
    p.add_argument("--moments", type=int, default=2)
    p.add_argument("--days", type=int, default=220)
    p.add_argument("--out", default="bench/results/preload_probe.json")
    args = p.parse_args()

    rec = probe(args)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    e = rec["extrapolation"]
    print(f"\n{e['median_seconds_per_moment']} s per moment (69 fields, "
          f"{e['megabytes_per_moment']} MB)")
    print(f"{e['preload_days']} days = {e['moments_total']} moments = "
          f"{e['serial_hours']} h serial, {e['gigabytes_total_raw']} GB raw")

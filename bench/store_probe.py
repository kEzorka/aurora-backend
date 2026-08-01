"""What the preloaded window costs on disk, and how many moments go in one chunk.

Two numbers block `history_store.py`, and neither has been measured.

**The compression ratio.** `config.py` says 220 days is "180 GB in fp32 after
compression", which is 252.2 GB raw divided by 1.4. But the 1.4 came from
`bench_compress.py`, and that measured ERA5's *int16 packing* — the gain from
halving the dtype — not what lz4 does to raw fp32 geophysical data. Those are
different numbers and only one of them belongs in a budget where `bytes = 4`.
If the true ratio is nearer 1.15 the window is 219 GB, the budget is 237 GB
against 249 GB free, and 220 days stops being a choice made with real numbers.

**The time-chunk extent.** The time-major layout is `[N moments, 32, 32]`, and N
is baked into the store: changing it later means refilling. Four things pull on
it. A block has to fit in RAM to be written without a staging pass — measured on
this box, 251 GB total and 243 GB available, so even 88 moments of all 69 fields
(25 GB) fits and the staging store the design assumed is unnecessary. Eviction
works by deleting whole chunks, so N sets the granularity the window slides at:
at N = 32 the window is "220 days give or take 8". Every six-hourly append
rewrites the whole active time-chunk across every tile. And N divides into the
file count: 69 fields x 1035 tiles is 71 415 chunks per block.

Phase A takes one moment and all 69 fields and measures the ratio per variable,
map-major, with the production codec. Phase B takes many moments of a few
representative fields and writes them time-major at several N, measuring write
throughput, on-disk size, file count, and what a point series and a region cost
to read back out.

Writes to a scratch directory and removes it. Touches no production store.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from .preload_probe import ATMOS, LEVELS, SURFACE, URL

# Exactly what `app/publish.py` and `app/postprocess.py` write with. Measuring a
# different codec here would produce a number no production store ever reaches.
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

TILE = 32

# Phase B's subset. Not arbitrary: two surface fields that differ in smoothness
# (msl is nearly flat, 2t has coastlines), and the three atmospheric fields whose
# compression behaviour is furthest apart — geopotential is almost a function of
# latitude alone, specific humidity is spiky and near zero over half the globe,
# temperature sits between them. A ratio measured on t alone would flatter q.
SUBSET = (("2t", None), ("msl", None), ("t", 850), ("q", 850), ("z", 500))


def _du(path: Path) -> int:
    """Bytes actually on disk. `stat().st_size` and not `st_blocks`: zarr chunk
    files are small enough that block rounding would swamp the ratio."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _files(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.is_file())


# ------------------------------------------------------------------ phase A

def phase_a(ds, when: dt.datetime, scratch: Path) -> dict:
    """One moment, all 69 fields, map-major. Answers the ratio question."""
    stamp = np.datetime64(when, "ns")
    root = scratch / "maps.zarr"
    g = zarr.open_group(str(root), mode="w")

    rows = {}
    raw_total = 0
    for short, full in SURFACE.items():
        arr = np.asarray(ds[full].sel(time=stamp).values, dtype="float32")
        a = g.create_array(short, shape=arr.shape, chunks=arr.shape,
                           dtype="float32", compressors=[BLOSC])
        t0 = time.perf_counter()
        a[:] = arr
        write_s = time.perf_counter() - t0
        disk = _du(root / short)
        rows[short] = {
            "kind": "surface", "levels": 1,
            "raw_megabytes": round(arr.nbytes / 1e6, 2),
            "disk_megabytes": round(disk / 1e6, 2),
            "ratio": round(arr.nbytes / disk, 3),
            "write_seconds": round(write_s, 3),
        }
        raw_total += arr.nbytes

    for short, full in ATMOS.items():
        arr = np.asarray(ds[full].sel(time=stamp, level=list(LEVELS)).values,
                         dtype="float32")
        a = g.create_array(short, shape=arr.shape, chunks=(1, *arr.shape[1:]),
                           dtype="float32", compressors=[BLOSC])
        t0 = time.perf_counter()
        a[:] = arr
        write_s = time.perf_counter() - t0
        disk = _du(root / short)
        # Per level too: the window is chunked one level at a time, and a single
        # figure over 13 levels would hide that the ratio varies with height.
        per_level = []
        for i, lev in enumerate(LEVELS):
            one = arr[i]
            sub = scratch / f"_lev_{short}_{lev}.zarr"
            b = zarr.create_array(str(sub), shape=one.shape, chunks=one.shape,
                                  dtype="float32", compressors=[BLOSC])
            b[:] = one
            d = _du(sub)
            per_level.append({"level": lev, "ratio": round(one.nbytes / d, 3)})
            shutil.rmtree(sub)
        rows[short] = {
            "kind": "atmospheric", "levels": len(LEVELS),
            "raw_megabytes": round(arr.nbytes / 1e6, 2),
            "disk_megabytes": round(disk / 1e6, 2),
            "ratio": round(arr.nbytes / disk, 3),
            "write_seconds": round(write_s, 3),
            "per_level": per_level,
        }
        raw_total += arr.nbytes

    disk_total = _du(root)
    shutil.rmtree(root)
    return {
        "fields": rows,
        "moment_raw_megabytes": round(raw_total / 1e6, 2),
        "moment_disk_megabytes": round(disk_total / 1e6, 2),
        "ratio": round(raw_total / disk_total, 3),
    }


# ------------------------------------------------------------------ phase B

def _fetch_subset(ds, when: dt.datetime) -> dict:
    """One moment of the subset. Returns short-name (or short@level) to array."""
    stamp = np.datetime64(when, "ns")
    out = {}
    for short, lev in SUBSET:
        if lev is None:
            out[short] = np.asarray(ds[SURFACE[short]].sel(time=stamp).values,
                                    dtype="float32")
        else:
            out[f"{short}@{lev}"] = np.asarray(
                ds[ATMOS[short]].sel(time=stamp, level=lev).values, dtype="float32")
    return out


def phase_b(ds, when: dt.datetime, moments: int, workers: int,
            extents: list[int], scratch: Path) -> dict:
    """Many moments of a few fields, time-major at several N."""
    times = [when + dt.timedelta(hours=6 * i) for i in range(moments)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        blocks = list(pool.map(lambda t: _fetch_subset(ds, t), times))
    fetch_s = time.perf_counter() - t0

    names = list(blocks[0])
    # [moments, 721, 1440] per field, held in RAM. The point of the phase: on a
    # box with 243 GB available this is what production does instead of staging
    # map-major to disk and rechunking.
    stacks = {n: np.stack([b[n] for b in blocks]) for n in names}
    del blocks
    ram_gigabytes = sum(a.nbytes for a in stacks.values()) / 1e9

    ny, nx = stacks[names[0]].shape[1:]
    trials = []
    for n in extents:
        if n > moments:
            continue
        root = scratch / f"series_{n}.zarr"
        g = zarr.open_group(str(root), mode="w")

        t0 = time.perf_counter()
        for name in names:
            a = g.create_array(name, shape=(moments, ny, nx),
                               chunks=(n, TILE, TILE),
                               dtype="float32", compressors=[BLOSC])
            a[:] = stacks[name]
        write_s = time.perf_counter() - t0

        raw = sum(a.nbytes for a in stacks.values())
        disk = _du(root)

        # Reads. A point series is the query the time-major layout exists for;
        # a region is the one the contract has to support over 69 variables.
        r = zarr.open(str(root), mode="r")
        lat_i, lon_i = 300, 700
        t0 = time.perf_counter()
        for name in names:
            _ = r[name][:, lat_i, lon_i]
        point_s = time.perf_counter() - t0

        # 10 deg x 10 deg at 0.25 deg is 41 x 41 cells.
        t0 = time.perf_counter()
        for name in names:
            _ = r[name][:, lat_i:lat_i + 41, lon_i:lon_i + 41]
        region_s = time.perf_counter() - t0

        # The reads above take the whole time axis, which flatters a large N for
        # a reason that has nothing to do with the query people actually send: a
        # full series reads the same total bytes at any N, so only the per-chunk
        # overhead shows. The short spans are where N costs something — asking
        # for one day out of a store chunked 88 moments deep still decompresses
        # all 88, and that amplification is the whole argument against a large N.
        spans = {}
        for span in (4, 45):
            t0 = time.perf_counter()
            for name in names:
                _ = r[name][0:span, lat_i, lon_i]
            pt = time.perf_counter() - t0
            t0 = time.perf_counter()
            for name in names:
                _ = r[name][0:span, lat_i:lat_i + 41, lon_i:lon_i + 41]
            rg = time.perf_counter() - t0
            spans[f"span_{span}"] = {
                "point_seconds": round(pt, 4),
                "region_41x41_seconds": round(rg, 4),
                # Bytes decompressed over bytes wanted, on the region read.
                "time_amplification": round(min(n, moments) / min(span, moments), 1),
            }

        # What a six-hourly append costs: one time-chunk rewritten across every
        # tile, for every field. Measured on the same store, overwriting block 0.
        t0 = time.perf_counter()
        w = zarr.open(str(root), mode="a")
        for name in names:
            w[name][0:n] = stacks[name][0:n]
        append_s = time.perf_counter() - t0

        trials.append({
            "time_chunk": n,
            "fields": len(names),
            "raw_megabytes": round(raw / 1e6, 1),
            "disk_megabytes": round(disk / 1e6, 1),
            "ratio": round(raw / disk, 3),
            "chunk_files": _files(root),
            "write_seconds": round(write_s, 2),
            "write_megabytes_per_second": round(raw / 1e6 / write_s, 1),
            "point_series_seconds": round(point_s, 4),
            "region_41x41_seconds": round(region_s, 4),
            "append_block_seconds": round(append_s, 2),
            "window_slack_days": round(n * 6 / 24, 1),
            "short_spans": spans,
        })
        print(f"  N={n:3d}  ratio {trials[-1]['ratio']:.2f}  "
              f"write {trials[-1]['write_megabytes_per_second']:6.1f} MB/s  "
              f"full point {point_s * 1000:6.1f} ms  region {region_s * 1000:6.1f} ms  "
              f"| 1d region {spans['span_4']['region_41x41_seconds'] * 1000:6.1f} ms  "
              f"11d region {spans['span_45']['region_41x41_seconds'] * 1000:6.1f} ms  "
              f"append {append_s:5.1f} s  files {trials[-1]['chunk_files']}",
              flush=True)
        shutil.rmtree(root)

    return {
        "moments": moments,
        "subset": [n for n in names],
        "fetch_seconds": round(fetch_s, 1),
        "fetch_workers": workers,
        "ram_gigabytes_held": round(ram_gigabytes, 2),
        "trials": trials,
    }


def main(args) -> dict:
    import xarray as xr

    ds = xr.open_zarr(URL, chunks=None)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    when = dt.datetime.fromisoformat(args.when)

    if args.skip_a:
        # Phase A is 69 fields off the wire for a number that does not change
        # between runs; `--skip-a` reads it back from a previous result so the
        # read trials can be re-run on their own.
        a = json.loads(Path(args.out).read_text())["phase_a"]
        print(f"phase A - reused, ratio {a['ratio']}", flush=True)
    else:
        print("phase A - compression ratio, 69 fields, one moment", flush=True)
        a = phase_a(ds, when, scratch)
        print(f"  moment: {a['moment_raw_megabytes']} MB raw, "
              f"{a['moment_disk_megabytes']} MB on disk, ratio {a['ratio']}", flush=True)

    print(f"\nphase B - time-chunk extent, {args.moments} moments", flush=True)
    b = phase_b(ds, when, args.moments, args.workers, args.extents, scratch)

    shutil.rmtree(scratch, ignore_errors=True)

    # The budget, restated with the measured ratio instead of the assumed one.
    moments_220 = 880
    raw_gb = a["moment_raw_megabytes"] * moments_220 / 1000
    return {
        "url": URL,
        "codec": "blosc lz4 clevel=5 shuffle",
        "phase_a": a,
        "phase_b": b,
        "budget": {
            "preload_days": 220,
            "moments": moments_220,
            "raw_gigabytes": round(raw_gb, 1),
            "assumed_ratio": 1.4,
            "assumed_gigabytes": round(raw_gb / 1.4, 1),
            "measured_ratio": a["ratio"],
            "measured_gigabytes": round(raw_gb / a["ratio"], 1),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--when", default="1990-03-15T12:00")
    p.add_argument("--moments", type=int, default=88)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extents", type=int, nargs="+", default=[8, 32, 88])
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--scratch", default="/tmp/store_probe")
    p.add_argument("--out", default="bench/results/store_probe.json")
    args = p.parse_args()

    rec = main(args)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    bud = rec["budget"]
    print(f"\n220 days: {bud['raw_gigabytes']} GB raw")
    print(f"  assumed ratio {bud['assumed_ratio']} -> {bud['assumed_gigabytes']} GB")
    print(f"  measured ratio {bud['measured_ratio']} -> {bud['measured_gigabytes']} GB")

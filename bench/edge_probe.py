"""What the fresh edge of the history window costs, six hours at a time.

`store_probe.py` picked N = 88 by measuring reads over *sealed* blocks — blocks
written once from RAM and never touched again. That is the right measurement for
the ten blocks the initial fill produces, and the wrong one for the block the
six-hourly cycle is currently writing into, which is where "yesterday" and "last
week" live. Those are the most-asked queries in the whole design and no number
covers them yet.

Two shapes are possible and the module's structure hangs off which one is taken.

**In place.** One chunk extent everywhere. Every six hours the new moment is
written into the active `[88, 32, 32]` block, which means zarr reads the whole
chunk back, merges one moment into it, and writes it again — for every tile of
every field. Nothing is ever rechunked, the reader sees one layout, and the cost
is a read-modify-write of 25 GB every cycle.

**A separate edge.** The newest moments accumulate in a small-N store where the
append is cheap, and are sealed into a `[88, 32, 32]` block once 88 of them have
arrived. The append is cheap and the rechunk happens once every 22 days, but the
reader has to union two stores with different chunk shapes, and the hot queries
are served by the shape that measured *worst* on long spans.

This measures both: the true read-modify-write append at several N (writing into
a chunk that is already half full, which is the case the cycle actually hits),
the sealing pass, and what the hot queries cost when they land on the edge.
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

from .preload_probe import URL
from .store_probe import BLOSC, TILE, _du, _fetch_subset


def _build(root: Path, stacks: dict, n: int) -> zarr.Group:
    g = zarr.open_group(str(root), mode="w")
    for name, arr in stacks.items():
        a = g.create_array(name, shape=arr.shape, chunks=(n, TILE, TILE),
                           dtype="float32", compressors=[BLOSC])
        a[:] = arr
    return g


def main(args) -> dict:
    import xarray as xr

    ds = xr.open_zarr(URL, chunks=None)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    when = dt.datetime.fromisoformat(args.when)
    times = [when + dt.timedelta(hours=6 * i) for i in range(args.moments)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        blocks = list(pool.map(lambda t: _fetch_subset(ds, t), times))
    fetch_s = time.perf_counter() - t0
    names = list(blocks[0])
    stacks = {name: np.stack([b[name] for b in blocks]) for name in names}
    del blocks

    # Scale to the real store. The subset is 5 fields; the window carries 69.
    scale = 69 / len(names)

    rows = []
    for n in args.extents:
        root = scratch / f"edge_{n}.zarr"
        _build(root, stacks, n)
        disk = _du(root)

        # The append the cycle actually performs. Not "write a fresh chunk":
        # moment 50 lands in a chunk that already holds 49 others, so zarr has to
        # decompress the chunk, merge, and compress it back. Picking an index in
        # the middle of a chunk is the whole point — an aligned write would
        # measure the easy case that only happens once every N cycles.
        w = zarr.open(str(root), mode="a")
        idx = min(50, args.moments - 1)
        one = {name: stacks[name][idx:idx + 1] for name in names}
        t0 = time.perf_counter()
        for name in names:
            w[name][idx:idx + 1] = one[name]
        rmw_s = time.perf_counter() - t0

        # The hot queries, landing on this shape. "Yesterday" is 4 moments and
        # "last week" is 28 — and both are read off the *end* of the axis, which
        # is where the edge actually is. On a store this size any offset costs
        # the same, but reading the head would describe a query nobody sends.
        r = zarr.open(str(root), mode="r")
        lat_i, lon_i = 300, 700
        hot = {}
        for label, span in (("yesterday", 4), ("last_week", 28)):
            lo = args.moments - span
            t0 = time.perf_counter()
            for name in names:
                _ = r[name][lo:lo + span, lat_i:lat_i + 41, lon_i:lon_i + 41]
            hot[label] = round(time.perf_counter() - t0, 4)

        rows.append({
            "time_chunk": n,
            "disk_megabytes": round(disk / 1e6, 1),
            "rmw_append_seconds": round(rmw_s, 2),
            "rmw_append_seconds_69": round(rmw_s * scale, 1),
            # What the read-modify-write actually moves at 69 variables: one
            # chunk-deep slice of every tile, decompressed and written back.
            "rmw_gigabytes_69": round(n * 286.6 / 1000, 1),
            "hot_region_seconds": hot,
            "hot_region_seconds_69": {k: round(v * scale, 3) for k, v in hot.items()},
        })
        print(f"  N={n:3d}  RMW append {rmw_s:6.2f} s (x69: {rmw_s * scale:6.1f} s)  "
              f"вчера {hot['yesterday'] * 1000:6.1f} ms  "
              f"неделя {hot['last_week'] * 1000:6.1f} ms", flush=True)

    # The sealing pass: read the whole small-N edge and write it as one big block.
    # Only paid by the separate-edge shape, once every 88 cycles.
    seal = None
    if args.seal_from in args.extents:
        src = zarr.open(str(scratch / f"edge_{args.seal_from}.zarr"), mode="r")
        dst_root = scratch / "sealed.zarr"
        t0 = time.perf_counter()
        g = zarr.open_group(str(dst_root), mode="w")
        for name in names:
            a = g.create_array(name, shape=src[name].shape,
                               chunks=(args.moments, TILE, TILE),
                               dtype="float32", compressors=[BLOSC])
            a[:] = src[name][:]
        seal_s = time.perf_counter() - t0
        shutil.rmtree(dst_root)
        seal = {
            "from_time_chunk": args.seal_from,
            "seconds": round(seal_s, 2),
            "seconds_69": round(seal_s * scale, 1),
            "every_days": round(args.moments * 6 / 24, 1),
        }
        print(f"  sealing N={args.seal_from} -> N={args.moments}: {seal_s:.1f} s "
              f"(x69: {seal_s * scale:.0f} s), once per {seal['every_days']} days",
              flush=True)

    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "url": URL,
        "moments": args.moments,
        "fields_measured": names,
        "scale_to_69": round(scale, 2),
        "fetch_seconds": round(fetch_s, 1),
        "trials": rows,
        "sealing": seal,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--when", default="1990-03-15T12:00")
    p.add_argument("--moments", type=int, default=88)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extents", type=int, nargs="+", default=[8, 88])
    p.add_argument("--seal-from", type=int, default=8)
    p.add_argument("--scratch", default="/tmp/edge_probe")
    p.add_argument("--out", default="bench/results/edge_probe.json")
    args = p.parse_args()

    rec = main(args)
    Path(args.out).write_text(json.dumps(rec, indent=2))

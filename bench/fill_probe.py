"""How long the initial fill takes, one pass against slot-at-a-time.

`HistoryWriter.write` is the six-hourly append and costs a full read-modify-write
of the block: 98 s at 69 fields, measured in `edge_probe.py`. The question here
is what the *fill* costs, where the block is created and written once from slot
0 and zarr has nothing to read back and merge.

Real ERA5, five fields, one block of 88, fetched the same way `edge_probe.py`
fetches it and scaled to 69 by the same ratio of field counts. Synthetic data is
not an option here: the bulk leg is pure compression with no decompress to
dominate it, so it moves with how hard the bytes are to compress, and a smooth
made-up field compresses several times better than weather does. The slot leg
would survive synthetic data — read-modify-write is dominated by the pass over
uncompressed bytes — but there is no reason to run the two legs on different
inputs.

Costs one ARCO fetch of ~3.5 minutes before it measures anything.

Writes `bench/results/fill_probe.json`.
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "aurora-backend"))

from app import config, history_store as hs  # noqa: E402
from app.contracts import LAT_SIZE, LON_SIZE  # noqa: E402
from bench.preload_probe import URL  # noqa: E402
from bench.store_probe import _fetch_subset  # noqa: E402

N = config.HISTORY_TIME_CHUNK
ROOT = Path("/tmp/fill_probe.zarr")

p = argparse.ArgumentParser()
p.add_argument("--when", default="1990-03-15T12:00")
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()

import xarray as xr  # noqa: E402

ds = xr.open_zarr(URL, chunks=None)
when = dt.datetime.fromisoformat(args.when)
times = [when + dt.timedelta(hours=6 * i) for i in range(N)]

t = time.perf_counter()
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    moments = list(pool.map(lambda x: _fetch_subset(ds, x), times))
fetch_s = time.perf_counter() - t

FIELDS = tuple(moments[0])
SCALE = 69 / len(FIELDS)
hs.FIELDS = FIELDS
stack = {n: np.stack([m[n] for m in moments]) for n in FIELDS}
del moments

out = {"url": URL, "when": args.when, "moments": N,
       "fields_measured": list(FIELDS), "scale_to_69": round(SCALE, 1),
       "fetch_seconds": round(fetch_s, 1)}

# One pass.
shutil.rmtree(ROOT, ignore_errors=True)
w = hs.HistoryWriter(ROOT, keep_days=220)
t = time.perf_counter()
w.write_block(hs.block_of(w.keep), stack)
bulk = time.perf_counter() - t
out["bulk_seconds"] = round(bulk, 2)
out["bulk_seconds_69"] = round(bulk * SCALE, 1)
# Five fields, not 69, and five that happen to compress well — this is a sanity
# check on the input, not an estimate of what a block costs on disk.
out["subset_disk_megabytes"] = round(
    sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file()) / 1e6, 1)
out["subset_ratio"] = round(
    sum(a.nbytes for a in stack.values()) / 1e6 / out["subset_disk_megabytes"], 3)

# Slot at a time, same data, same block. Ten slots is enough to price one.
shutil.rmtree(ROOT, ignore_errors=True)
w = hs.HistoryWriter(ROOT, keep_days=220)
b = hs.block_of(w.keep)
t0 = hs.time_of(b * N)
t = time.perf_counter()
for k in range(10):
    w.write(t0 + k * (hs.time_of(1) - hs.time_of(0)),
            {n: stack[n][k] for n in FIELDS})
per_slot = (time.perf_counter() - t) / 10
out["slot_seconds"] = round(per_slot, 2)
out["slot_seconds_69"] = round(per_slot * SCALE, 1)
shutil.rmtree(ROOT, ignore_errors=True)

# What the 880-moment fill costs each way, single-threaded.
out["fill_hours_bulk_69"] = round(880 / N * bulk * SCALE / 3600, 2)
out["fill_hours_slot_69"] = round(880 * per_slot * SCALE / 3600, 2)

dest = Path.home() / "aurora-backend" / "bench" / "results" / "fill_probe.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

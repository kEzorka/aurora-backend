"""Rewrite the ERA5 archive from zarr v2 to zarr v3.

Nothing about the data changes: same dtype, same chunk grid (one map per
chunk), same Blosc-lz4-5-shuffle, same consolidated metadata. Only the
metadata format and the codec spelling move to v3.

The target is allocated once with compute=False and then filled by region, a
time block at a time, so peak memory is one block rather than the 104 GB the
archive occupies uncompressed.

    .venv312/bin/python scripts/archive_to_v3.py SRC DST [--block 8]

Verify afterwards with --check, which compares random slices of both stores.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

TIME_COORD = "valid_time"
LEVEL_COORD = "pressure_level"

# The archive's own codec, kept rather than reconsidered: reads of this store
# are already at 3.2 GB/s and the zarr 3 default (Zstd, no shuffle) writes
# slower for a size win we are not short of disk for.
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")


# Keys that describe how v2 stored a variable. They travel with the opened
# dataset and xarray hands them to zarr 3's array constructor, which rejects a
# numcodecs.Blosc where it wants a BytesBytesCodec. Everything else in
# `encoding` — units, calendar, dtype, _FillValue — is about what the values
# mean and must survive.
V2_LAYOUT_KEYS = (
    "compressor",
    "compressors",
    "filters",
    "serializer",
    "chunks",
    "preferred_chunks",
    "shards",
)


def strip_v2_layout(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for var in list(ds.variables.values()):
        for key in V2_LAYOUT_KEYS:
            var.encoding.pop(key, None)
    return ds


def encoding(ds: xr.Dataset) -> dict:
    """One full map per chunk, matching the v2 store, spelled the v3 way."""
    enc = {}
    for name, var in ds.data_vars.items():
        enc[name] = {
            "chunks": tuple(
                1 if d in (TIME_COORD, LEVEL_COORD) else var.sizes[d] for d in var.dims
            ),
            "compressors": [BLOSC],
        }
    return enc


def convert(src: Path, dst: Path, block: int) -> None:
    ds = strip_v2_layout(xr.open_zarr(src, consolidated=True))
    n = ds.sizes[TIME_COORD]
    print(f"{src} -> {dst}: {n} times, {len(ds.data_vars)} variables, block {block}")

    # Allocate. compute=False writes metadata and coordinates but no chunks;
    # write_empty_chunks is False in zarr 3, so nothing is materialised yet.
    ds.to_zarr(dst, mode="w", compute=False, consolidated=True, encoding=encoding(ds))

    started = time.monotonic()
    for i in range(0, n, block):
        stop = min(i + block, n)
        chunk = ds.isel({TIME_COORD: slice(i, stop)}).load()
        # A region write must not carry the store's own index coordinates.
        chunk = chunk.drop_vars([TIME_COORD, LEVEL_COORD, "latitude", "longitude"])
        chunk.to_zarr(dst, region={TIME_COORD: slice(i, stop)}, consolidated=True)
        done = stop
        rate = done / (time.monotonic() - started)
        eta = (n - done) / rate
        print(f"  {done}/{n} times  {rate:.2f} t/s  eta {eta / 60:.1f} min", flush=True)

    print(f"done in {(time.monotonic() - started) / 60:.1f} min")


def check(src: Path, dst: Path, samples: int) -> bool:
    a = xr.open_zarr(src, consolidated=True)
    b = xr.open_zarr(dst, consolidated=True)

    if a.sizes != b.sizes:
        print(f"FAIL sizes differ: {a.sizes} vs {b.sizes}")
        return False
    for coord in (TIME_COORD, LEVEL_COORD, "latitude", "longitude"):
        if not np.array_equal(a[coord].values, b[coord].values):
            print(f"FAIL coordinate {coord} differs")
            return False
    if set(a.data_vars) != set(b.data_vars):
        print(f"FAIL variables differ: {set(a.data_vars) ^ set(b.data_vars)}")
        return False
    fmt = zarr.open(str(dst))["t"].metadata.zarr_format
    if fmt != 3:
        print(f"FAIL target is zarr_format {fmt}, not 3")
        return False

    rng = random.Random(0)
    n = a.sizes[TIME_COORD]
    names = sorted(a.data_vars)
    # Endpoints first: an off-by-one in the region loop hides in the middle.
    picks = [(names[i % len(names)], t) for i, t in enumerate((0, n - 1, n // 2))]
    picks += [(rng.choice(names), rng.randrange(n)) for _ in range(samples)]

    for name, t in picks:
        va = a[name].isel({TIME_COORD: t}).values
        vb = b[name].isel({TIME_COORD: t}).values
        if not np.array_equal(va, vb, equal_nan=True):
            bad = int(np.sum(va != vb))
            print(f"FAIL {name} at {TIME_COORD}={t}: {bad} values differ")
            return False
        print(f"  ok {name} {TIME_COORD}={t}")

    print(f"checked {len(picks)} slices, all identical")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--block", type=int, default=8, help="times per region write")
    p.add_argument("--samples", type=int, default=8, help="random slices to compare")
    p.add_argument("--check", action="store_true", help="only verify, do not write")
    p.add_argument("--force", action="store_true", help="remove an existing target")
    args = p.parse_args()

    # 4 beats zarr 3's default of 10: 69 small writes to a local disk get
    # nothing from async fan-out and pay for the contention. On object storage
    # the optimum would sit the other way.
    zarr.config.set({"async.concurrency": 4})

    if args.check:
        return 0 if check(args.src, args.dst, args.samples) else 1

    if args.dst.exists():
        if not args.force:
            print(f"{args.dst} exists; pass --force to replace it")
            return 1
        shutil.rmtree(args.dst)

    convert(args.src, args.dst, args.block)
    return 0 if check(args.src, args.dst, args.samples) else 1


if __name__ == "__main__":
    sys.exit(main())

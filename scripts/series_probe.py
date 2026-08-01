"""Time-major layouts for the archive, measured against the one we serve from.

`read.json` has a point query over the whole archive taking 0.735 s to return
1.4 KB: it touches 1441 MB because the store keeps one chunk per map, so a series
of N moments is N whole maps. That is the number the serving design turns on, and
this is the experiment that decides whether a second layout fixes it.

`tile_probe` already answered the other half — cutting a *map* into tiles buys
nothing, because a map is read whole either way. So nothing here tiles for the
sake of tiling: every candidate puts the **time axis inside the chunk** and the
tiles exist only to keep the chunk small enough to be worth reading.

Two things are reported per layout and they are not the same measurement:

* **wall time**, which is what a caller feels, and
* **bytes touched**, computed from the chunk grid rather than timed — the honest
  amplification figure, because a warm page cache makes wall time flatter than
  the work actually done.

The map reads are here as a regression check, not as a hope: this layout is
supposed to lose on them. The question is by how much, since the map queries keep
their own store and only pay this cost if we ever consider dropping it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import zarr

MB = 1024**2
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

SRC = Path("/home/kostya/data/global.zarr")
OUT = Path("bench/scratch/series")
RESULT = Path("bench/results/series_probe.json")

# CDS names -> the names Aurora and our API use. Surface only: the pressure-level
# variables are 13x the volume and nobody asks for a five-year series of q at
# 850 hPa from a phone.
SURF = {"t2m": "2t", "msl": "msl", "u10": "10u", "v10": "10v"}

# (label, tile, time chunk or None for "whole axis", pack to int16)
CASES: list[tuple[str, tuple[int, int], int | None, bool]] = [
    ("time x 128x128", (128, 128), None, False),
    ("time x 64x64", (64, 64), None, False),
    ("time x 32x32", (32, 32), None, False),
    ("time x 16x16", (16, 16), None, False),
    ("quarter x 64x64", (64, 64), 91, False),
    ("time x 64x64 int16", (64, 64), None, True),
]

POINT_LAT, POINT_LON = 55.75, 37.62      # Moscow, and any city would do
CITY_CELLS = 8                            # ~2 degrees, a metropolitan area
EUROPE = (slice(72, 224), slice(0, 180))  # same index box tile_probe used


def touched(chunks: tuple[int, ...], shape: tuple[int, ...],
            sel: tuple[slice, ...], itemsize: int = 4) -> int:
    """Uncompressed bytes zarr must decode to answer `sel`.

    Not a timing. A chunk is the unit of decode, so a selection costs whole
    chunks whatever fraction of them it actually wants, and that ratio is the
    thing this whole experiment is about.
    """
    n = 1
    for c, size, s in zip(chunks, shape, sel):
        lo, hi, _ = s.indices(size)
        if hi <= lo:
            return 0
        n *= (hi - 1) // c - lo // c + 1
    return n * math.prod(chunks) * itemsize


def timeit(fn, reps: int = 3):
    """Minimum of several runs: we want the cost of the read, not of a stall."""
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def store_size(path: Path) -> tuple[float, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / MB, len(files)


def quantize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """float32 -> int16 with a scale and offset, the way ERA5 itself is stored.

    int16 rather than fp16 on purpose: geopotential is ~5e4 m2/s2, where fp16
    spacing is 32, while a fitted scale spends all 65535 steps inside the range
    the variable actually occupies.
    """
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    scale = (hi - lo) / 65534.0 or 1.0
    offset = lo + 32767.0 * scale
    return np.rint((arr - offset) / scale).astype("int16"), scale, offset


def build(path: Path, src: zarr.Group, tile, tchunk, pack: bool) -> float:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dst = zarr.open_group(str(path), mode="w")
    for cds, name in SURF.items():
        a = src[cds]
        nt = a.shape[0]
        chunks = (tchunk or nt, *tile)
        # One variable at a time: the transpose needs the whole time axis in
        # memory, which is 1.4 GB per surface variable and 5.6 GB for all four.
        data = a[:]
        attrs = {}
        if pack:
            data, scale, offset = quantize(data)
            attrs = {"scale_factor": scale, "add_offset": offset}
        out = dst.create_array(
            name, shape=data.shape, chunks=chunks, dtype=data.dtype,
            compressors=[BLOSC],
        )
        out[:] = data
        out.attrs.update(attrs)
        del data
    for coord in ("latitude", "longitude", "valid_time"):
        c = src[coord][:]
        dst.create_array(coord, shape=c.shape, chunks=c.shape, dtype=c.dtype)[:] = c
    return time.perf_counter() - t0


def probe(path: Path, ilat: int, ilon: int, nt: int) -> dict:
    g = zarr.open(str(path), mode="r")
    # The archive still carries CDS names; the layouts we write use Aurora's.
    # Both are the same fields, so the probe reads whichever the store has.
    names = list(SURF.values()) if "2t" in g else list(SURF)
    a = g[names[0]]
    chunks, shape = tuple(a.chunks), tuple(a.shape)
    packed = "scale_factor" in a.attrs
    scale = a.attrs.get("scale_factor", 1.0)
    offset = a.attrs.get("add_offset", 0.0)
    itemsize = a.dtype.itemsize

    def deq(x):
        # A packed store is only comparable to a float one if the caller's cost
        # of getting back to physical units is inside the measurement.
        return x.astype("float32") * scale + offset if packed else x

    city = (slice(ilat, ilat + CITY_CELLS), slice(ilon, ilon + CITY_CELLS))
    queries = {
        "point_series": (
            lambda: deq(a[:, ilat, ilon]),
            (slice(0, nt), slice(ilat, ilat + 1), slice(ilon, ilon + 1)),
        ),
        "point_series_4var": (
            lambda: [deq(g[n][:, ilat, ilon]) for n in names],
            None,
        ),
        "city_series": (
            lambda: deq(a[(slice(None), *city)]),
            (slice(0, nt), *city),
        ),
        "one_map": (
            lambda: deq(a[0]),
            (slice(0, 1), slice(0, shape[1]), slice(0, shape[2])),
        ),
        "europe_map": (
            lambda: deq(a[(0, *EUROPE)]),
            (slice(0, 1), *EUROPE),
        ),
    }

    out = {}
    for label, (fn, sel) in queries.items():
        secs, val = timeit(fn, reps=5 if "map" in label else 3)
        returned = sum(v.nbytes for v in val) if isinstance(val, list) else val.nbytes
        row = {"s": round(secs, 4), "returned_kb": round(returned / 1024, 2)}
        if sel is not None:
            tb = touched(chunks, shape, sel, itemsize)
            row["touched_mb"] = round(tb / MB, 2)
            row["amplification"] = round(tb / max(returned, 1))
        else:
            tb = len(SURF) * touched(chunks, shape,
                                     (slice(0, nt), slice(ilat, ilat + 1),
                                      slice(ilon, ilon + 1)), itemsize)
            row["touched_mb"] = round(tb / MB, 2)
            row["amplification"] = round(tb / max(returned, 1))
        out[label] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", help="labels to run, default all")
    ap.add_argument("--keep", action="store_true", help="do not delete the layouts")
    args = ap.parse_args()

    src = zarr.open(str(args.src), mode="r")
    lat, lon = src["latitude"][:], src["longitude"][:]
    ilat = int(np.abs(lat - POINT_LAT).argmin())
    ilon = int(np.abs(lon - POINT_LON).argmin())
    nt = int(src["t2m"].shape[0])
    print(f"archive: {nt} moments, point ({lat[ilat]:.2f}, {lon[ilon]:.2f}) "
          f"at index ({ilat}, {ilon})")

    results: dict[str, dict] = {}

    # The store we serve from today is the baseline, measured the same way.
    base_mb, base_files = store_size(args.src)
    results["source (map-major)"] = {
        "chunks": list(src["t2m"].chunks), "mb": round(base_mb, 1),
        "files": base_files, "build_s": None,
        **{"queries": probe(args.src, ilat, ilon, nt)},
    }
    report(results)

    cases = [c for c in CASES if not args.only or c[0] in args.only]
    for label, tile, tchunk, pack in cases:
        path = args.out / label.replace(" ", "_").replace("x", "")
        build_s = build(path, src, tile, tchunk, pack)
        mb, files = store_size(path)
        results[label] = {
            "chunks": list(zarr.open(str(path), mode="r")["2t"].chunks),
            "mb": round(mb, 1), "files": files, "build_s": round(build_s, 1),
            "queries": probe(path, ilat, ilon, nt),
        }
        report(results)
        if not args.keep:
            shutil.rmtree(path)

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({
        "source": str(args.src),
        "moments": nt,
        "variables": list(SURF.values()),
        "point": [round(float(lat[ilat]), 2), round(float(lon[ilon]), 2)],
        "note": "read through zarr directly, no dask and no xarray cache. "
                "touched_mb is computed from the chunk grid, not timed.",
        "layouts": results,
    }, indent=2))
    print(f"\nwrote {RESULT}")
    return 0


def report(results: dict) -> None:
    label, row = next(reversed(results.items()))
    q = row["queries"]
    print(f"\n{label}  chunks={tuple(row['chunks'])}  {row['mb']:.0f} MB  "
          f"{row['files']} files  build {row['build_s']}s")
    print(f"  {'query':<18} {'wall':>9} {'touched':>11} {'returned':>11} {'amp':>10}")
    for name, r in q.items():
        print(f"  {name:<18} {r['s']:>8.4f}s {r['touched_mb']:>9.2f} MB "
              f"{r['returned_kb']:>9.2f} KB {r['amplification']:>10,}")


if __name__ == "__main__":
    raise SystemExit(main())

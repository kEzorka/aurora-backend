"""Build the time-major store the point queries are served from.

`bench/results/series_probe.json` picked the layout: the whole time axis inside
the chunk, tiles of 32x32. This writes it, and can extend it as new moments
arrive rather than rebuilding from scratch.

**The time chunk is the one number to think about**, because it trades the two
things this store exists between:

* Large (a year, 1460 moments at 6 h) — a five-year series is 5 chunk reads.
  But appending one moment rewrites the whole open chunk: 6 MB per tile column,
  ~6 GB across the grid, for one 8 MB timestep. Unusable on the cycle.
* Small (a day, 4 moments) — appending is cheap and exact, but a five-year
  series is 1826 chunk reads and we are most of the way back to where we started.

So this is meant to be run in two roles, and `--time-chunk` is how you say which:
a **recent** store on a small chunk that the producer appends to every cycle, and
an **archive** store on a large chunk, rebuilt in batches when the recent store
has accumulated enough to fold in. `app/history.py` reads across both.

Writes `manifest.json` and `validation.json` next to the store, because a store
whose provenance is only in somebody's shell history is not a store you can serve
from.
"""

from __future__ import annotations

import argparse
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import contracts  # noqa: E402

MB = 1024**2
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

# CDS names in the archive -> the names the contract and the API use.
RENAME = {"t2m": "2t", "msl": "msl", "u10": "10u", "v10": "10v"}

TIME_COORD = "valid_time"
TIME_UNITS = "hours since 1970-01-01"


def open_source(path: Path):
    g = zarr.open(str(path), mode="r")
    units = g[TIME_COORD].attrs.get("units", TIME_UNITS)
    origin = pd.Timestamp(units.split(" since ")[1].strip())
    step = {"hours": "h", "seconds": "s", "days": "D", "minutes": "m"}[
        units.split(" since ")[0].strip()]
    times = pd.to_datetime(pd.Series(np.asarray(g[TIME_COORD][:])),
                           unit=step, origin=origin).to_numpy("datetime64[ns]")
    return g, times


def to_hours(times: np.ndarray) -> np.ndarray:
    epoch = np.datetime64(TIME_UNITS.split(" since ")[1].strip())
    return ((times - epoch) / np.timedelta64(1, "h")).astype("int64")


def quantize(arr: np.ndarray, lo: float, hi: float):
    """float32 -> int16 over a fixed range, so appended blocks stay comparable.

    The range comes from the contract rather than from the data: a per-block
    min/max would give every append its own scale, and the store would stop
    meaning one thing.
    """
    scale = (hi - lo) / 65534.0
    offset = lo + 32767.0 * scale
    packed = np.rint((np.clip(arr, lo, hi) - offset) / scale).astype("int16")
    return packed, scale, offset


def create(dst: Path, names, lat, lon, tile, time_chunk, pack) -> zarr.Group:
    dst.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(dst), mode="w")
    for name in names:
        spec = contracts.SURFACE.get(name) or contracts.ATMOS[name]
        arr = g.create_array(
            name, shape=(0, len(lat), len(lon)),
            chunks=(time_chunk, *tile),
            dtype="int16" if pack else "float32",
            compressors=[BLOSC],
        )
        arr.attrs["units"] = spec[0]
        if pack:
            lo, hi = spec[1]
            _, scale, offset = quantize(np.zeros(1, "float32"), lo, hi)
            arr.attrs.update({"scale_factor": float(scale), "add_offset": float(offset),
                              "valid_range": [lo, hi]})
    g.create_array("latitude", shape=lat.shape, chunks=lat.shape, dtype="float64")[:] = lat
    g.create_array("longitude", shape=lon.shape, chunks=lon.shape, dtype="float64")[:] = lon
    t = g.create_array(TIME_COORD, shape=(0,), chunks=(max(time_chunk, 1024),), dtype="int64")
    t.attrs.update({"units": TIME_UNITS, "calendar": "proleptic_gregorian"})
    return g


def existing_times(dst: Path) -> np.ndarray:
    g = zarr.open(str(dst), mode="r")
    epoch = np.datetime64(TIME_UNITS.split(" since ")[1].strip())
    hours = np.asarray(g[TIME_COORD][:])
    return epoch + hours.astype("timedelta64[h]")


def append(dst: Path, src, src_times, want: np.ndarray, names, block: int) -> dict:
    """Write `want` (already known to be new and in order) into the store."""
    g = zarr.open_group(str(dst), mode="a")
    src_pos = {t: i for i, t in enumerate(src_times)}
    written, problems = 0, []

    for start in range(0, len(want), block):
        chunk_times = want[start:start + block]
        idx = [src_pos[t] for t in chunk_times]
        fields = {}
        for cds, name in RENAME.items():
            if name not in names:
                continue
            # One variable at a time and one block at a time: the transpose
            # needs the whole block in memory, and a year of one surface field
            # is already 5.8 GB.
            fields[name] = np.stack([src[cds][i] for i in idx]).astype("float32")

        report = contracts.validate(g["latitude"][:], g["longitude"][:],
                                    chunk_times, fields)
        problems += report["problems"]
        contracts.require(report)

        n0 = g[TIME_COORD].shape[0]
        n1 = n0 + len(chunk_times)
        g[TIME_COORD].resize((n1,))
        g[TIME_COORD][n0:n1] = to_hours(chunk_times)
        for name, arr in fields.items():
            a = g[name]
            a.resize((n1, *a.shape[1:]))
            if "scale_factor" in a.attrs:
                lo, hi = a.attrs["valid_range"]
                packed, _, _ = quantize(arr, float(lo), float(hi))
                a[n0:n1] = packed
            else:
                a[n0:n1] = arr
        written += len(chunk_times)
        print(f"  wrote {written}/{len(want)} moments", flush=True)

    return {"written": written, "problems": problems}


def store_size(path: Path) -> tuple[float, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / MB, len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="map-major source store")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=32, help="lat/lon tile, 32 from series_probe")
    ap.add_argument("--time-chunk", type=int, default=1460,
                    help="moments per chunk; 1460 = a year at 6 h (archive role), "
                         "4 = a day (recent role, cheap to append)")
    ap.add_argument("--vars", nargs="*", default=list(RENAME.values()))
    ap.add_argument("--block", type=int, default=None,
                    help="moments held in memory at once; defaults to the time chunk")
    ap.add_argument("--rebuild", action="store_true", help="discard an existing store")
    args = ap.parse_args()

    src, src_times = open_source(args.src)
    lat = np.asarray(src["latitude"][:], dtype="float64")
    lon = np.asarray(src["longitude"][:], dtype="float64")
    pack = False  # measured at 1.42x; enable per store once the API is settled

    fresh = args.rebuild or not (args.dst / "zarr.json").exists()
    if fresh:
        create(args.dst, args.vars, lat, lon, (args.tile, args.tile),
               args.time_chunk, pack)
        have = np.array([], dtype="datetime64[ns]")
    else:
        have = existing_times(args.dst)

    want = np.array([t for t in src_times if t not in set(have.tolist())],
                    dtype="datetime64[ns]")
    want.sort()
    if len(want) == 0:
        print(f"{args.dst}: already covers the source, nothing to do")
        return 0

    print(f"{args.dst}: {len(have)} moments present, appending {len(want)} "
          f"({want[0]} .. {want[-1]}), chunk ({args.time_chunk}, {args.tile}, {args.tile})")
    t0 = time.perf_counter()
    result = append(args.dst, src, src_times, want, args.vars,
                    args.block or args.time_chunk)
    build_s = time.perf_counter() - t0

    mb, files = store_size(args.dst)
    covered = existing_times(args.dst)
    manifest = {
        "store": str(args.dst),
        "role": "archive" if args.time_chunk >= 365 else "recent",
        "source": str(args.src),
        "layout": {"chunks": [args.time_chunk, args.tile, args.tile],
                   "dtype": "int16" if pack else "float32",
                   "compressor": "blosc lz4 clevel=5 shuffle"},
        "variables": args.vars,
        "coverage": [str(covered[0]), str(covered[-1])],
        "moments": int(len(covered)),
        "step_hours": contracts.STEP_HOURS,
        "size_mb": round(mb, 1),
        "files": files,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "build_s": round(build_s, 1),
    }
    # Beside the store, not inside it: zarr walks its own directory and warns
    # about anything it does not recognise as part of the hierarchy.
    sidecar = args.dst.with_suffix(args.dst.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2))
    args.dst.with_suffix(args.dst.suffix + ".validation.json").write_text(json.dumps({
        "ok": not result["problems"],
        "problems": result["problems"],
        "checked": "axes, time monotonicity and step, per-variable physical range",
    }, indent=2))

    print(f"\n{args.dst}: {manifest['moments']} moments, {mb:.0f} MB, {files} files, "
          f"{build_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

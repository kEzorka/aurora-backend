"""What chunk shape and sharding do to a regional read.

The codec question is settled and it moved the Europe cut from 0.97 s to
0.49 s. This asks the bigger one. The store is chunked one full global map per
chunk, so a cut over Europe — 1.6% of the area — decodes 100% of every field it
touches. Splitting the map into tiles fixes that and costs an inode explosion:
16 tiles per map times ~2760 maps per 40-step forecast is 44k files. Sharding
is zarr 3's answer and has no v2 equivalent: shards are the unit of writing,
chunks the unit of reading, so the file count stays where it is today while the
read granularity gets 16x finer.

No GPU and no model. An existing 4-step forecast is rewritten in each candidate
layout, which measures exactly the encode and decode this changes and nothing
else. Whether the winner also survives the real write path is a separate run.

    python -m scripts.shard_bench [--src STORE] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

MB = 1024**2
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

# Europe, the cut the common case looks like: 152 x 280 points of 720 x 1440.
EU = dict(latitude=slice(72, 34), longitude=slice(-25, 45))
# A city, to see where tiling stops paying.
POINT = dict(latitude=55.75, longitude=37.6)

# label -> (lat, lon) tile inside one (lead, level) map. None = whole map.
LAYOUTS: dict[str, tuple[int, int] | None] = {
    "map chunk (today)": None,
    "shard, 180x360 tiles": (180, 360),
    "shard, 90x180 tiles": (90, 180),
    "shard, 60x120 tiles": (60, 120),
}

TIME_DIM = "lead_time"
LEVEL_DIM = "pressure_level"


def encoding(ds: xr.Dataset, tile: tuple[int, int] | None) -> dict:
    """Today's encoding, or the same map split into tiles inside a shard."""
    enc = {}
    for name, var in ds.data_vars.items():
        shard = tuple(1 if d in (TIME_DIM, LEVEL_DIM) else var.sizes[d] for d in var.dims)
        spec: dict = {"compressors": [BLOSC]}
        if tile is None:
            spec["chunks"] = shard
        else:
            # The shard keeps one whole map, so the file count is unchanged;
            # only the addressable unit inside it gets smaller.
            spec["shards"] = shard
            spec["chunks"] = shard[:-2] + tile
        enc[name] = spec
    return enc


def du(path: Path) -> tuple[float, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / MB, len(files)


def timeit(fn, reps=3) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def codec_names(path: Path, var: str) -> list[str]:
    """What zarr actually wrote, not what we asked for.

    xarray silently ignores encoding keys it does not know, so `shards` in the
    dict proves nothing. The store's own metadata does.
    """
    meta = json.loads((path / var / "zarr.json").read_text())
    return [c.get("name", "?") for c in meta.get("codecs", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bench/scratch/py312-blosc/forecast_20260405T0000_024h.zarr")
    ap.add_argument("--out", default="bench/scratch/shards")
    ap.add_argument("--json", default="bench/results/shard_bench.json")
    args = ap.parse_args()

    src = xr.open_zarr(args.src, consolidated=True).load()
    raw = sum(v.nbytes for v in src.data_vars.values()) / MB
    steps = src.sizes[TIME_DIM]
    print(f"source: {steps} steps, {len(src.data_vars)} vars, {raw:.1f} MB in memory")

    out_dir = Path(args.out)
    results = {}

    print(f"\n{'layout':<22} {'MB':>7} {'files':>6} {'write s':>8} "
          f"{'full':>7} {'Europe':>7} {'1 map':>7} {'point':>7}")
    for label, tile in LAYOUTS.items():
        path = out_dir / label.replace(" ", "_").replace(",", "")
        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        write_s = timeit(
            lambda: src.to_zarr(path, mode="w", encoding=encoding(src, tile),
                                consolidated=False),
            reps=1,
        )
        mb, nfiles = du(path)

        codecs = codec_names(path, "t")
        sharded = "sharding_indexed" in codecs
        if (tile is not None) != sharded:
            raise SystemExit(f"{label}: asked shards={tile}, store says {codecs}")

        ds = xr.open_zarr(path, consolidated=False)
        full = timeit(lambda: [ds[v].values for v in ds.data_vars])
        eu = timeit(lambda: [ds[v].sel(**EU).values for v in ds.data_vars])
        one = timeit(lambda: ds["2t"].isel(lead_time=0).values, reps=5)
        pt = timeit(lambda: [ds[v].sel(**POINT, method="nearest").values
                             for v in ds.data_vars], reps=5)
        ds.close()

        results[label] = {
            "mb": round(mb, 1), "files": nfiles, "write_s": round(write_s, 2),
            "full_s": round(full, 3), "europe_s": round(eu, 3),
            "one_map_ms": round(one * 1e3, 1), "point_s": round(pt, 3),
            "codecs": codecs,
        }
        print(f"{label:<22} {mb:>7.1f} {nfiles:>6} {write_s:>8.2f} "
              f"{full:>6.2f}s {eu:>6.2f}s {one * 1e3:>6.0f}ms {pt:>6.2f}s")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source_store": args.src,
        "steps": int(steps),
        "raw_mb_in_memory": round(raw, 1),
        "europe": "latitude 72..34, longitude -25..45",
        "layouts": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

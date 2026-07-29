"""Tiles with and without a shard around them.

shard_bench found tiling made every read slower, which is the opposite of what
tiling is for. Two candidate causes: the sharding codec's own per-access cost,
or the plain fact that N small blosc calls cost more Python than one big one.
This separates them by writing the same tile sizes twice — once inside a shard,
once as ordinary chunks — and reading both through zarr directly, no dask and
no xarray cache in the way.

The file count is the other half of the answer, so it is counted too: plain
tiles are what sharding exists to avoid.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import xarray as xr
import zarr

MB = 1024**2
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
SRC = "bench/scratch/py312-blosc/forecast_20260405T0000_024h.zarr"
OUT = Path("bench/scratch/tiles")

# (label, tile, sharded)
CASES: list[tuple[str, tuple[int, int] | None, bool]] = [
    ("whole map", None, False),
    ("180x360 plain", (180, 360), False),
    ("180x360 shard", (180, 360), True),
    ("90x180 plain", (90, 180), False),
    ("90x180 shard", (90, 180), True),
    ("60x120 plain", (60, 120), False),
]

LAT, LON = slice(72, 224), slice(0, 180)   # Europe in index space
TIME_DIM, LEVEL_DIM = "lead_time", "pressure_level"


def encoding(ds, tile, sharded):
    enc = {}
    for name, var in ds.data_vars.items():
        whole = tuple(1 if d in (TIME_DIM, LEVEL_DIM) else var.sizes[d] for d in var.dims)
        spec = {"compressors": [BLOSC]}
        if tile is None:
            spec["chunks"] = whole
        elif sharded:
            spec["shards"] = whole
            spec["chunks"] = whole[:-2] + tile
        else:
            spec["chunks"] = whole[:-2] + tile
        enc[name] = spec
    return enc


def timeit(fn, reps=3):
    return min(_t(fn) for _ in range(reps))


def _t(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> int:
    src = xr.open_zarr(SRC, consolidated=True).load()
    results = {}
    print(f"{'layout':<16} {'MB':>7} {'files':>7} {'write':>7} "
          f"{'full t':>8} {'EU t':>8} {'1 map':>8} {'EU all':>8}")
    for label, tile, sharded in CASES:
        path = OUT / label.replace(" ", "_")
        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_s = _t(lambda: src.to_zarr(path, mode="w", consolidated=False,
                                         encoding=encoding(src, tile, sharded)))
        files = [p for p in path.rglob("*") if p.is_file()]
        mb = sum(p.stat().st_size for p in files) / MB

        codecs = [c.get("name", "?") for c in
                  json.loads((path / "t" / "zarr.json").read_text())["codecs"]]
        if sharded != ("sharding_indexed" in codecs):
            raise SystemExit(f"{label}: store says {codecs}")

        g = zarr.open(str(path), mode="r")
        names = [n for n in src.data_vars]
        full = timeit(lambda: g["t"][:])
        eu = timeit(lambda: g["t"][:, :, LAT, LON])
        one = timeit(lambda: g["2t"][0], reps=5)
        eu_all = timeit(lambda: [g[n][..., LAT, LON] for n in names])

        results[label] = {
            "mb": round(mb, 1), "files": len(files), "write_s": round(write_s, 2),
            "full_t_s": round(full, 3), "eu_t_s": round(eu, 3),
            "one_map_ms": round(one * 1e3, 1), "eu_all_vars_s": round(eu_all, 3),
        }
        print(f"{label:<16} {mb:>7.1f} {len(files):>7} {write_s:>6.2f}s "
              f"{full:>7.3f}s {eu:>7.3f}s {one * 1e3:>6.0f}ms {eu_all:>7.3f}s")

    out = Path("bench/results/tile_probe.json")
    out.write_text(json.dumps({
        "source": SRC, "steps": 4,
        "europe_index": "lat 72:224, lon 0:180 of 720x1440",
        "note": "read through zarr directly - dask and the xarray cache both "
                "distort this measurement",
        "layouts": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

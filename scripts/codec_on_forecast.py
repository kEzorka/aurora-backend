"""Re-run the codec bench on Aurora's own output instead of on ERA5.

bench_compress.py measures ERA5 fields as a stand-in for forecast output. This
checks whether that substitution holds, by encoding the 69 maps of one real
rollout step with the same codecs.
"""
import json
import time
from pathlib import Path

import numcodecs
import numcodecs.blosc
import numpy as np
import xarray as xr
import zarr

MB = 1024**2
PY310 = Path("bench/scratch/py310-fp16/forecast_20260405T0000_024h.zarr")
PY312 = Path("bench/scratch/py312-fp16/forecast_20260405T0000_024h.zarr")

# What the py310 store actually says it used, rather than what zarr 2's default
# is assumed to be.
meta = json.loads((PY310 / "2t" / ".zarray").read_text())
print("py310 store 2t .zarray compressor:", meta["compressor"], " chunks", meta["chunks"])
print("py310 store 2t .zarray filters:   ", meta["filters"])

ds = xr.open_zarr(PY312, consolidated=True)
fields = {}
for name, da in ds.data_vars.items():
    a = da.isel(lead_time=0).values
    if a.ndim == 2:
        fields[name] = np.ascontiguousarray(a, dtype="float32")
    else:
        for i in range(a.shape[0]):
            fields[f"{name}@{i}"] = np.ascontiguousarray(a[i], dtype="float32")
raw = sum(a.nbytes for a in fields.values())
print(f"\none forecast step: {len(fields)} fields, {raw / MB:.1f} MB raw")

numcodecs.blosc.use_threads = True
numcodecs.blosc.set_nthreads(1)

CODECS = {
    "lz4-5 shuffle": ("lz4", 5, numcodecs.Blosc.SHUFFLE),
    "zstd-3 shuffle": ("zstd", 3, numcodecs.Blosc.SHUFFLE),
    "zstd-3 noshuffle": ("zstd", 3, numcodecs.Blosc.NOSHUFFLE),
    "zstd-5 bitshuffle": ("zstd", 5, numcodecs.Blosc.BITSHUFFLE),
}
print(f"\n{'codec':<20} {'MB':>7} {'ratio':>6} {'enc ms':>7} {'dec ms':>7}")
out = {}
for label, (cname, clevel, shuffle) in CODECS.items():
    c = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
    t0 = time.perf_counter()
    blobs = [c.encode(a) for a in fields.values()]
    enc = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    for b in blobs:
        np.frombuffer(c.decode(b), dtype="float32")
    dec = (time.perf_counter() - t0) * 1e3
    n = sum(len(b) for b in blobs)
    out[label] = {"mb": n / MB, "ratio": raw / n, "encode_ms": enc, "decode_ms": dec}
    print(f"{label:<20} {n / MB:>7.1f} {raw / n:>6.2f} {enc:>7.0f} {dec:>7.0f}")

# The configuration that is actually the migration target, which neither store
# on disk represents: zarr 3 writing through Blosc.
scratch = Path("bench/scratch/z3blosc")
for name, arr in list(fields.items())[:1]:
    pass
t0 = time.perf_counter()
g = zarr.open_group(str(scratch), mode="w", zarr_format=3)
blosc3 = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
for name, arr in fields.items():
    z = g.create_array(name.replace("@", "_"), shape=arr.shape, chunks=arr.shape,
                       dtype="float32", compressors=[blosc3])
    z[:] = arr
write_s = time.perf_counter() - t0
size = sum(f.stat().st_size for f in scratch.rglob("*") if f.is_file())
print(f"\nzarr 3 + BloscCodec(lz4,5,shuffle): {size / MB:.1f} MB  ratio {raw / size:.2f}  "
      f"write {write_s:.2f}s for one step")
out["zarr3_blosc_write_s"] = write_s
out["zarr3_blosc_mb"] = size / MB
Path("bench/results/codec_on_forecast.json").write_text(json.dumps(
    {"raw_mb": raw / MB, "fields": len(fields), "codecs": out}, indent=2))

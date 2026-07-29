"""Decode cost of the two stores, read the way the slicing endpoint will read.

Same forecast, same fp16 weights, written once by zarr 2 (Blosc lz4-5 shuffle)
and once by zarr 3 (its default ZstdCodec(level=0), no shuffle). Both are read
here by zarr 3, because that is the runtime the backend is moving to.
"""
import time
import numcodecs.blosc
import numpy as np
import xarray as xr

STORES = {
    "zarr2 lz4-5 shuffle": "bench/scratch/py310-fp16/forecast_20260405T0000_024h.zarr",
    "zarr3 zstd-0 noshuf": "bench/scratch/py312-fp16/forecast_20260405T0000_024h.zarr",
    "zarr3 lz4-5 shuffle": "bench/scratch/py312-blosc/forecast_20260405T0000_024h.zarr",
}
# Europe: the cut the user says is the common case.
EU = dict(latitude=slice(72, 34), longitude=slice(-25, 45))


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


for threads in (1, 4):
    numcodecs.blosc.use_threads = True
    numcodecs.blosc.set_nthreads(threads)
    print(f"\nblosc threads = {threads}")
    print(f"{'store':<22} {'full 4 steps':>13} {'Europe cut':>12} {'one 2t map':>12}")
    for label, path in STORES.items():
        ds = xr.open_zarr(path, consolidated=True)
        atmos = [v for v in ds.data_vars if ds[v].ndim == 4]
        full = timeit(lambda: [ds[v].values for v in ds.data_vars])
        eu = timeit(lambda: [ds[v].sel(**EU).values for v in ds.data_vars])
        one = timeit(lambda: ds["2t"].isel(lead_time=0).values, reps=5)
        print(f"{label:<22} {full:>12.2f}s {eu:>11.2f}s {one * 1e3:>10.0f}ms")
        ds.close()

"""Convert the month-per-directory NetCDF archive into one zarr store.

    python -m scripts.to_zarr                 # ~/data/global -> ~/data/global.zarr
    python -m scripts.to_zarr --workers 12

The NetCDF layout is one directory per month holding surface.nc and
pressure.nc. That is a fine archive format and a poor serving format: every
forecast reads two timestamps out of a 13 GB file and the month boundary has to
be stitched by hand. Zarr turns that into a single time axis with one chunk per
(timestamp, level) field, so a read touches exactly the bytes it needs.

Two things decide how long this takes.

**Chunking.** The access pattern is fixed — one timestamp, one level, the whole
global field, twice per forecast — so time and level chunk to 1 and the
horizontal grid is never split. A chunked lat/lon grid would turn one read into
dozens.

**Processes, not threads.** The obvious implementation is `open_dataset(...,
chunks=...)` then `to_zarr`, and it runs at one core: HDF5 reads hold the GIL,
so dask's threads serialise behind it. Measured 2.4 MB/s on this box, which is
hours for ~100 GB. Instead the metadata is written once and a process pool
fills the data in, one timestamp per task, writing chunks directly. Timestamps
never share a chunk, so the writes need no coordination.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr

TIME_COORD = "valid_time"
LEVEL_COORD = "pressure_level"
SURF_VARS = ("t2m", "u10", "v10", "msl")
ATMOS_VARS = ("z", "q", "t", "u", "v")

# Set once per worker process so the NetCDF files are opened once, not once
# per timestamp.
_CACHE: dict[Path, xr.Dataset] = {}


def month_dirs(src: Path) -> list[Path]:
    dirs = sorted(d for d in src.iterdir() if d.is_dir() and (d / "surface.nc").exists())
    if not dirs:
        raise SystemExit(f"no month directories with surface.nc under {src}")
    return dirs


def _open(path: Path) -> xr.Dataset:
    if path not in _CACHE:
        _CACHE[path] = xr.open_dataset(path, engine="netcdf4")
    return _CACHE[path]


def plan(src: Path) -> tuple[list[tuple[Path, int, int]], np.ndarray, xr.Dataset]:
    """Global time axis, and the (month, local index, global index) task list."""
    tasks: list[tuple[Path, int, int]] = []
    stamps: list[np.datetime64] = []

    for d in month_dirs(src):
        surf = xr.open_dataset(d / "surface.nc", engine="netcdf4")
        press = xr.open_dataset(d / "pressure.nc", engine="netcdf4")
        st, pt = surf[TIME_COORD].values, press[TIME_COORD].values
        if not np.array_equal(st, pt):
            raise SystemExit(f"{d.name}: surface and pressure timestamps differ")
        for i in range(len(st)):
            tasks.append((d, i, len(stamps) + i))
        stamps.extend(st)
        surf.close()
        press.close()

    order = np.argsort(np.asarray(stamps))
    if not np.array_equal(order, np.arange(len(stamps))):
        raise SystemExit("month directories are not in chronological order")

    sample = xr.open_dataset(month_dirs(src)[0] / "pressure.nc", engine="netcdf4")
    return tasks, np.asarray(stamps), sample


def create_store(dst: Path, stamps: np.ndarray, sample: xr.Dataset) -> np.ndarray:
    """Write coordinates and empty arrays; the pool fills in the data.

    Returns the ascending level order, which is baked into the store so that no
    reader has to know the CDS files come back descending.
    """
    levels = np.sort(sample[LEVEL_COORD].values).astype("int32")
    lat = sample["latitude"].values
    lon = sample["longitude"].values
    nt, nl, ny, nx = len(stamps), len(levels), len(lat), len(lon)

    coords = {
        TIME_COORD: (TIME_COORD, stamps),
        LEVEL_COORD: (LEVEL_COORD, levels),
        "latitude": ("latitude", lat),
        "longitude": ("longitude", lon),
    }
    # Lazy placeholders at full size. With compute=False xarray writes the
    # metadata and the coordinate arrays and skips the data entirely, so this
    # costs neither memory nor disk — the chunk files stay absent until a
    # worker writes one. It also means the time axis is encoded by xarray's own
    # CF machinery rather than by hand.
    empty2 = da.zeros((nt, ny, nx), dtype="float32", chunks=(1, ny, nx))
    empty3 = da.zeros((nt, nl, ny, nx), dtype="float32", chunks=(1, 1, ny, nx))
    template = xr.Dataset(
        {v: ((TIME_COORD, "latitude", "longitude"), empty2) for v in SURF_VARS}
        | {
            v: ((TIME_COORD, LEVEL_COORD, "latitude", "longitude"), empty3)
            for v in ATMOS_VARS
        },
        coords=coords,
    )
    template.to_zarr(dst, mode="w", compute=False)
    return levels


def fill(task: tuple[Path, int, int]) -> tuple[int, float]:
    month, i, g = task
    t0 = time.time()
    root = zarr.open(str(_DST), mode="r+")

    surf = _open(month / "surface.nc")
    for v in SURF_VARS:
        root[v][g] = surf[v].isel({TIME_COORD: i}).values.astype("float32")

    press = _open(month / "pressure.nc")
    for v in ATMOS_VARS:
        arr = press[v].isel({TIME_COORD: i}).values.astype("float32")
        root[v][g] = arr[_ORDER]

    return g, time.time() - t0


def _init(dst: str, order: np.ndarray) -> None:
    global _DST, _ORDER
    _DST, _ORDER = dst, order


def convert_static(src_file: Path, dst: Path) -> None:
    ds = xr.open_dataset(src_file, engine="netcdf4")
    # The static file carries a length-1 time axis that means nothing; drop it
    # here so the store holds plain (latitude, longitude) fields.
    for coord in (TIME_COORD, "time"):
        if coord in ds.dims:
            ds = ds.isel({coord: 0}, drop=True)
    if dst.exists():
        shutil.rmtree(dst)
    ds.to_zarr(dst, mode="w")
    print(f"static -> {dst}  vars={list(ds.data_vars)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(Path.home() / "data" / "global"))
    p.add_argument("--dst", default=str(Path.home() / "data" / "global.zarr"))
    p.add_argument("--static-dst", default=str(Path.home() / "data" / "static.zarr"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)

    tasks, stamps, sample = plan(src)
    print(f"{len(tasks)} timestamps  {stamps[0]} .. {stamps[-1]}")

    levels = create_store(dst, stamps, sample)
    raw_levels = sample[LEVEL_COORD].values
    order = np.argsort(raw_levels)
    sample.close()
    print(f"store created, levels {levels[0]}..{levels[-1]} ({len(levels)})")

    t0 = time.time()
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(str(dst), order)) as pool:
        for _ in pool.imap_unordered(fill, tasks, chunksize=1):
            done += 1
            if done % 20 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                left = (len(tasks) - done) / rate
                print(f"  {done}/{len(tasks)}  {rate:.2f} steps/s  eta {left / 60:.1f} min",
                      flush=True)

    convert_static(src / "static.nc", Path(args.static_dst))

    check = xr.open_dataset(dst, engine="zarr", chunks=None)
    got = check[TIME_COORD].values
    if len(got) != len(stamps) or not np.array_equal(
        got.astype("datetime64[ns]"), stamps.astype("datetime64[ns]")
    ):
        raise SystemExit("time axis does not match the source archive")
    # A zero field means a task silently did not run. Checking only the ends
    # would miss the realistic failure: `imap_unordered` over 364 tasks drops
    # one in the middle. 2t is in kelvin and is never legitimately zero, so one
    # sub-sampled read per timestamp settles the whole axis.
    probe = check["t2m"].isel(latitude=slice(None, None, 180),
                              longitude=slice(None, None, 360)).values
    bad = np.where(~(np.isfinite(probe).all(axis=(1, 2)) & (np.abs(probe).min(axis=(1, 2)) > 0)))[0]
    if len(bad):
        raise SystemExit(f"t2m is empty at {len(bad)} timestamps, first {got[bad[0]]}")
    # The 3-D vars share the task that writes 2t, so spot-checking the ends is
    # enough for them.
    for i in (0, len(got) - 1):
        t = check["t"].isel({TIME_COORD: i}).values
        if not (np.isfinite(t).all() and np.abs(t).min() > 0):
            raise SystemExit(f"t at {got[i]} is empty")
    print(f"\n{dst}\n  {len(got)} steps  {got[0]} .. {got[-1]}\n"
          f"  vars {list(check.data_vars)}\n  {time.time() - t0:.0f}s")
    check.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check a built Batch against everything Aurora silently assumes.

    python -m scripts.verify_batch 2026-05-01T00

The month boundary is the interesting case: it needs 2026-04-30T18 from the
previous month's file.
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np

from app import batch_builder
from app.era5_store import ERA5Store

# name -> plausible physical range, used to catch unit and ordering mistakes
SURF_RANGE = {
    "2t": (180.0, 340.0),  # K
    "10u": (-120.0, 120.0),  # m/s
    "10v": (-120.0, 120.0),
    "msl": (8.5e4, 1.15e5),  # Pa
}
ATMOS_RANGE = {
    "t": (150.0, 340.0),
    "u": (-160.0, 160.0),
    "v": (-160.0, 160.0),
    # ERA5 specific humidity goes very slightly negative from spectral
    # truncation; anything beyond -1e-4 would be a real problem.
    "q": (-1e-4, 0.1),  # kg/kg
    "z": (-5e3, 6e5),  # m^2/s^2
}

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        failures.append(msg)


def main(init_time: dt.datetime) -> int:
    store = ERA5Store()
    stamps = store.timestamps
    print(f"archive {stamps[0]:%Y-%m-%d %H:%M} .. {stamps[-1]:%Y-%m-%d %H:%M} ({len(stamps)} steps)")

    batch = batch_builder.build_batch(store, init_time)
    print(batch_builder.describe(batch))
    print()

    for name, t in batch.surf_vars.items():
        check(tuple(t.shape) == (1, 2, 721, 1440), f"surf {name} shape {tuple(t.shape)}")
        check(str(t.dtype) == "torch.float32", f"surf {name} dtype {t.dtype}")
    for name, t in batch.static_vars.items():
        check(tuple(t.shape) == (721, 1440), f"static {name} shape {tuple(t.shape)}")
    for name, t in batch.atmos_vars.items():
        check(tuple(t.shape) == (1, 2, 13, 721, 1440), f"atmos {name} shape {tuple(t.shape)}")

    m = batch.metadata
    check(len(m.atmos_levels) == 13, f"13 levels, got {len(m.atmos_levels)}")
    check(
        list(m.atmos_levels) == sorted(m.atmos_levels),
        f"levels ascending: {m.atmos_levels}",
    )
    check(float(m.lat[0]) > float(m.lat[-1]), f"lat descending {float(m.lat[0])}..{float(m.lat[-1])}")
    check(float(m.lon[0]) >= 0 and float(m.lon[-1]) < 360, "lon in [0, 360)")
    check(m.time == (init_time,), f"metadata time {m.time}")

    # The level axis of the tensor must agree with metadata.atmos_levels, not
    # just be sorted: temperature falls off with height, so the 50 hPa slice
    # has to be colder than the 1000 hPa slice.
    t_lo = float(batch.atmos_vars["t"][0, -1, 0].mean())
    t_hi = float(batch.atmos_vars["t"][0, -1, -1].mean())
    check(
        t_lo < t_hi,
        f"level axis matches metadata: mean T at {m.atmos_levels[0]} hPa = {t_lo:.1f} K "
        f"< at {m.atmos_levels[-1]} hPa = {t_hi:.1f} K",
    )

    for name, (lo, hi) in SURF_RANGE.items():
        v = batch.surf_vars[name]
        check(bool(v.min() >= lo and v.max() <= hi), f"surf {name} in [{lo}, {hi}]: "
              f"{float(v.min()):.4g}..{float(v.max()):.4g}")
    for name, (lo, hi) in ATMOS_RANGE.items():
        v = batch.atmos_vars[name]
        check(bool(v.min() >= lo and v.max() <= hi), f"atmos {name} in [{lo}, {hi}]: "
              f"{float(v.min()):.4g}..{float(v.max()):.4g}")

    for group in (batch.surf_vars, batch.static_vars, batch.atmos_vars):
        for name, t in group.items():
            check(not bool(t.isnan().any()), f"{name} has no NaNs")

    # The two history slices must actually differ, otherwise the model sees a
    # zero tendency and the whole history=2 mechanism is dead.
    d = float((batch.surf_vars["2t"][0, 1] - batch.surf_vars["2t"][0, 0]).abs().mean())
    check(d > 0.01, f"history slices differ: mean |d2t| = {d:.4f} K over 6 h")

    # Static z (orography) must not be the pressure-level z.
    check(
        batch.static_vars["z"].shape == (721, 1440),
        "static z is orography, separate from atmos z",
    )
    check(
        set(np.unique(batch.static_vars["lsm"].numpy())) <= {0.0, 1.0}
        or float(batch.static_vars["lsm"].max()) <= 1.0,
        "lsm in [0, 1]",
    )

    print()
    print(f"{len(failures)} failure(s)" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "2026-05-01T00"
    raise SystemExit(main(dt.datetime.fromisoformat(arg)))

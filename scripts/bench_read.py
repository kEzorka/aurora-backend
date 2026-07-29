"""How fast the archive answers the queries a user actually asks.

    python -m scripts.bench_read

Prints wall time and the bytes each query touches, because the ratio is the
whole story: this store is chunked one (timestamp, level) map per chunk, so
"the whole world at one moment" reads exactly what it returns and "one point
over three months" reads 364 full maps to hand back 364 floats.

Goes at the zarr arrays directly rather than through xarray on purpose.
`open_dataset` caches a variable once it has been read whole, so a benchmark
that reads the full series and then a single point through xarray times the
cache, not the store — the point read comes back in 0.00 s and the number is
a lie.
"""

from __future__ import annotations

import time

import numpy as np
import zarr

from app import config

MB = 1 << 20


def bench(label: str, fn, touched_mb: float) -> None:
    t0 = time.time()
    out = fn()
    n = np.asarray(out).size
    dt = time.time() - t0
    got = n * 4 / MB
    print(f"{label:<44} {dt:>7.2f}s  read {touched_mb:>8.1f} MB  "
          f"returned {got:>8.2f} MB  waste {touched_mb / max(got, 1e-9):>7.1f}x")


def main() -> int:
    r = zarr.open(str(config.DATA_ROOT), mode="r")
    n = r["t2m"].shape[0]
    # 721*1440*4 bytes is one chunk: one map, one moment, one level.
    chunk = 721 * 1440 * 4 / MB
    print(f"{n} timestamps, one chunk = {chunk:.2f} MB, "
          f"compressor {r['t2m'].compressor}\n")

    bench("2t, one moment, whole globe",
          lambda: r["t2m"][n - 1], chunk)
    bench("2t, one day (4 moments)",
          lambda: r["t2m"][n - 4:n], 4 * chunk)
    bench("2t, 30 days (120 moments)",
          lambda: r["t2m"][n - 120:n], 120 * chunk)
    bench("2t, whole archive (364 moments)",
          lambda: r["t2m"][:], n * chunk)

    bench("2t at ONE grid point, whole archive",
          lambda: r["t2m"][:, 360, 720], n * chunk)

    bench("t at 500 hPa, 30 days",
          lambda: r["t"][n - 120:n, 7], 120 * chunk)
    bench("t all 13 levels, one moment",
          lambda: r["t"][n - 1], 13 * chunk)

    # The full Aurora input: 4 surface + 5 atmospheric x 13 levels = 69 maps.
    def full_slice():
        out = [r[v][n - 1] for v in ("t2m", "u10", "v10", "msl")]
        out += [r[v][n - 1] for v in ("z", "q", "t", "u", "v")]
        return np.concatenate([np.asarray(a).ravel() for a in out])

    bench("full 69-channel slice, one moment", full_slice, 69 * chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

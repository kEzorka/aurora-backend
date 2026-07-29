"""What a codec costs and what it saves, measured on one real step.

    python -m scripts.bench_compress [--out bench/results/compress.json]

The store currently writes with zarr's v2 default, Blosc(lz4, clevel=5,
shuffle), because nothing ever chose otherwise. This measures the alternatives
on the data we actually write rather than on one 2t field: all 69 maps of a
single timestamp — 4 surface plus 5 atmospheric on 13 levels — which is exactly
what one rollout step puts on disk.

No GPU and no model. ERA5 fields stand in for Aurora's output: same grid, same
variables, same physics. Aurora's fields are marginally smoother than the
reanalysis, so the ratios here are a slight under-estimate of what the forecast
will get, not an over-estimate.

The lossy section is separate and secondary. It reports what quantization does
to a derivative, not only to the field, because a diagnostic that differentiates
amplifies rounding noise while a plain RMSE against ERA5 hides it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numcodecs
import numcodecs.blosc
import numpy as np

from app import config
from app.era5_store import ERA5Store

REPS = 3
MB = 1024**2
GB = 1024**3
STEPS_PER_RUN = 40

# name -> (cname, clevel, shuffle)
CODECS: dict[str, tuple[str, int, int]] = {
    "lz4-5 shuffle": ("lz4", 5, numcodecs.Blosc.SHUFFLE),      # today's default
    "lz4-5 bitshuffle": ("lz4", 5, numcodecs.Blosc.BITSHUFFLE),
    "lz4-9 shuffle": ("lz4", 9, numcodecs.Blosc.SHUFFLE),
    "lz4hc-5 shuffle": ("lz4hc", 5, numcodecs.Blosc.SHUFFLE),
    "zstd-1 shuffle": ("zstd", 1, numcodecs.Blosc.SHUFFLE),
    "zstd-3 shuffle": ("zstd", 3, numcodecs.Blosc.SHUFFLE),
    "zstd-5 shuffle": ("zstd", 5, numcodecs.Blosc.SHUFFLE),
    "zstd-3 bitshuffle": ("zstd", 3, numcodecs.Blosc.BITSHUFFLE),
    "zstd-5 bitshuffle": ("zstd", 5, numcodecs.Blosc.BITSHUFFLE),
    "zstd-3 noshuffle": ("zstd", 3, numcodecs.Blosc.NOSHUFFLE),  # control
}

THREAD_SWEEP = ("lz4-5 shuffle", "zstd-5 bitshuffle")
THREAD_COUNTS = (1, 4, 8)


def load_step(store: ERA5Store) -> dict[str, np.ndarray]:
    """The 69 maps of one timestamp, cropped to Aurora's 720x1440 grid."""
    when = store.timestamps[len(store.timestamps) // 2]
    fields: dict[str, np.ndarray] = {}
    for name, arr in store.surface_slice(when).items():
        fields[name] = np.ascontiguousarray(arr[:720])
    for name, arr in store.atmos_slice(when).items():
        for i in range(arr.shape[0]):
            fields[f"{name}@{i}"] = np.ascontiguousarray(arr[i, :720])
    return fields


def time_codec(codec, fields: dict[str, np.ndarray]) -> dict:
    """Encode and decode every field REPS times, keep the median pass."""
    enc_ms, dec_ms, nbytes = [], [], 0
    per_var: dict[str, float] = {}

    for rep in range(REPS):
        blobs = {}
        t0 = time.perf_counter()
        for name, arr in fields.items():
            blobs[name] = codec.encode(arr)
        enc_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        for name, blob in blobs.items():
            out = np.frombuffer(codec.decode(blob), dtype="float32")
        dec_ms.append((time.perf_counter() - t0) * 1e3)

        if rep == 0:
            nbytes = sum(len(b) for b in blobs.values())
            # One number per variable family, so a bad ratio can be traced to
            # the field that caused it rather than averaged away.
            for name, blob in blobs.items():
                var = name.split("@")[0]
                per_var.setdefault(var, [0, 0])
                per_var[var][0] += fields[name].nbytes
                per_var[var][1] += len(blob)

    raw = sum(a.nbytes for a in fields.values())
    return {
        "bytes": nbytes,
        "ratio": raw / nbytes,
        "encode_ms": float(np.median(enc_ms)),
        "decode_ms": float(np.median(dec_ms)),
        "per_var": {v: round(r / c, 2) for v, (r, c) in per_var.items()},
    }


def lossy_report(fields: dict[str, np.ndarray]) -> list[dict]:
    """What each lossy filter costs in the field and in its derivatives.

    Two derivatives, because they fail differently. The horizontal gradient
    subtracts neighbours ~28 km apart and is the common diagnostic. The
    difference between adjacent pressure levels subtracts two large, almost
    equal numbers, which is where rounding noise is worst — a lapse rate or a
    thickness is exactly that subtraction.
    """
    # One representative of each magnitude class: temperature ~3e2,
    # geopotential ~1e5, humidity ~1e-2, wind ~1e1.
    probes = {
        "t@6": fields["t@6"],
        "z@6": fields["z@6"],
        "q@6": fields["q@6"],
        "u@6": fields["u@6"],
    }
    vertical = {
        "t": (fields["t@6"], fields["t@7"]),
        "z": (fields["z@6"], fields["z@7"]),
    }

    filters = {
        "quantize d=1": numcodecs.Quantize(digits=1, dtype="float32"),
        "quantize d=2": numcodecs.Quantize(digits=2, dtype="float32"),
        "quantize d=3": numcodecs.Quantize(digits=3, dtype="float32"),
        "bitround keep=9": numcodecs.BitRound(keepbits=9),
        "bitround keep=12": numcodecs.BitRound(keepbits=12),
        "bitround keep=15": numcodecs.BitRound(keepbits=15),
    }
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    def roundtrip(filt, arr: np.ndarray) -> np.ndarray:
        # BitRound.encode rewrites its input in place. Without the copy the
        # comparison is the array against itself — zero error, every time.
        out = filt.decode(filt.encode(arr.copy()))
        return np.asarray(out, dtype="float32").reshape(arr.shape)

    rows = []
    for label, filt in filters.items():
        row = {"filter": label, "fields": {}}
        for name, arr in probes.items():
            enc = roundtrip(filt, arr)
            err = enc - arr
            grad_err = np.diff(err, axis=1)
            grad = np.diff(arr, axis=1)
            row["fields"][name] = {
                "rms_err": float(np.sqrt((err**2).mean())),
                "max_err": float(np.abs(err).max()),
                "grad_rel": float(
                    np.sqrt((grad_err**2).mean()) / np.sqrt((grad**2).mean())
                ),
                "ratio": arr.nbytes / len(codec.encode(enc)),
            }
        for name, (lo, hi) in vertical.items():
            a = roundtrip(filt, lo)
            b = roundtrip(filt, hi)
            d_true, d_enc = hi - lo, b - a
            row["fields"][f"{name} level-diff"] = {
                "rms_err": float(np.sqrt(((d_enc - d_true) ** 2).mean())),
                "vert_rel": float(
                    np.sqrt(((d_enc - d_true) ** 2).mean())
                    / np.sqrt((d_true**2).mean())
                ),
            }
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/compress.json")
    args = ap.parse_args()

    store = ERA5Store(config.DATA_ROOT, config.STATIC_FILE)
    fields = load_step(store)
    raw = sum(a.nbytes for a in fields.values())
    print(f"one step: {len(fields)} fields, {raw / MB:.1f} MB raw, "
          f"grid {next(iter(fields.values())).shape}")

    numcodecs.blosc.use_threads = True
    numcodecs.blosc.set_nthreads(1)

    results = {}
    print(f"\n{'codec':<20} {'MB':>7} {'ratio':>6} {'enc ms':>7} {'dec ms':>7} "
          f"{'GB/run':>7} {'+s/run':>7}")
    for label, (cname, clevel, shuffle) in CODECS.items():
        codec = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
        r = time_codec(codec, fields)
        results[label] = r
        # The decision is wall clock against disk: a rollout is ~130 s and the
        # write happens 40 times, once per step.
        gb_run = r["bytes"] * STEPS_PER_RUN / GB
        add_s = r["encode_ms"] * STEPS_PER_RUN / 1e3
        print(f"{label:<20} {r['bytes'] / MB:>7.1f} {r['ratio']:>6.2f} "
              f"{r['encode_ms']:>7.0f} {r['decode_ms']:>7.0f} "
              f"{gb_run:>7.2f} {add_s:>7.1f}")

    print("\nper-variable ratio (zstd-5 bitshuffle):")
    print("  " + "  ".join(f"{k}={v}" for k, v in results["zstd-5 bitshuffle"]["per_var"].items()))
    print("per-variable ratio (lz4-5 shuffle):")
    print("  " + "  ".join(f"{k}={v}" for k, v in results["lz4-5 shuffle"]["per_var"].items()))

    threads = {}
    print(f"\n{'codec / threads':<24} {'enc ms':>7} {'dec ms':>7}")
    for label in THREAD_SWEEP:
        cname, clevel, shuffle = CODECS[label]
        for n in THREAD_COUNTS:
            numcodecs.blosc.set_nthreads(n)
            codec = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
            r = time_codec(codec, fields)
            threads[f"{label} x{n}"] = r
            print(f"{label + ' x' + str(n):<24} {r['encode_ms']:>7.0f} {r['decode_ms']:>7.0f}")
    numcodecs.blosc.set_nthreads(1)

    lossy = lossy_report(fields)
    print("\nlossy filters — error in the field and in its derivatives")
    for row in lossy:
        print(f"  {row['filter']}")
        for name, m in row["fields"].items():
            if "vert_rel" in m:
                print(f"    {name:<16} rms {m['rms_err']:.3e}  "
                      f"vertical diff noise {m['vert_rel'] * 100:.3f}%")
            else:
                print(f"    {name:<16} rms {m['rms_err']:.3e}  max {m['max_err']:.3e}  "
                      f"gradient noise {m['grad_rel'] * 100:.3f}%  ratio {m['ratio']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "raw_bytes_per_step": raw,
        "fields": len(fields),
        "steps_per_run": STEPS_PER_RUN,
        "codecs": results,
        "threads": threads,
        "lossy": lossy,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

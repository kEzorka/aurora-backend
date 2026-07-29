"""End-to-end forecast without the HTTP layer.

    python -m scripts.run_forecast 2026-05-01T00 --steps 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import torch

from app import batch_builder, config, postprocess
from app.era5_store import ERA5Store
from app.inference import AuroraEngine


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("init_time", type=dt.datetime.fromisoformat)
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()

    store = ERA5Store()
    batch = batch_builder.build_batch(store, args.init_time)
    print(batch_builder.describe(batch))

    t0 = time.time()
    engine = AuroraEngine()
    print(f"{config.MODEL_NAME} loaded on {config.DEVICE} in {time.time() - t0:.1f}s")

    t0 = time.time()
    writer = postprocess.ForecastWriter(args.init_time, args.steps)
    for i, pred in enumerate(engine.rollout(batch, args.steps), start=1):
        writer.add(pred)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"  step {i}/{args.steps}  +{i * config.STEP_HOURS}h  "
              f"{time.time() - t0:.1f}s  peak {peak:.1f} GiB")

    print(f"wrote {writer.finish()}  ({config.OUTPUT_FORMAT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

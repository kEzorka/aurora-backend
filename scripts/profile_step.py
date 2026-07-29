"""Where the wall clock of one forecast actually goes.

    python -m scripts.profile_step 2026-05-01T00 --steps 4

Splits a rollout step into the three things it does — GPU forward, device to
host copy, encode and write — because the answer decides what is worth
optimising. If the forward dominates, output selection and faster disks buy
nothing and only the model matters.

The copy has to be timed with a CUDA sync around it or the forward's time
lands on whichever line first touches the result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import torch
from aurora import rollout as aurora_rollout

from app import batch_builder, config, postprocess
from app.era5_store import ERA5Store
from app.inference import AuroraEngine


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("init_time")
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()

    init = dt.datetime.fromisoformat(args.init_time)

    t0 = time.time()
    store = ERA5Store()
    t_index = time.time() - t0

    t0 = time.time()
    batch = batch_builder.build_batch(store, init)
    t_read = time.time() - t0

    t0 = time.time()
    engine = AuroraEngine()
    t_load = time.time() - t0

    writer = postprocess.ForecastWriter(init, args.steps)
    batch = batch.to(engine.device)

    totals = {"forward": 0.0, "to_cpu": 0.0, "write": 0.0}
    print(f"{'step':>4}  {'forward':>8}  {'to_cpu':>8}  {'write':>8}")

    with torch.inference_mode(), engine._autocast():
        it = aurora_rollout(engine.model, batch, steps=args.steps)
        for i in range(args.steps):
            sync()
            t0 = time.time()
            pred = next(it)
            sync()
            t1 = time.time()
            pred = pred.to("cpu")
            t2 = time.time()
            writer.add(pred)
            t3 = time.time()
            totals["forward"] += t1 - t0
            totals["to_cpu"] += t2 - t1
            totals["write"] += t3 - t2
            print(f"{i + 1:>4}  {t1 - t0:>8.2f}  {t2 - t1:>8.2f}  {t3 - t2:>8.2f}")

    path = writer.finish()
    wall = sum(totals.values())
    print()
    print(f"index          {t_index:>7.2f}s   once per process")
    print(f"read batch     {t_read:>7.2f}s   once per request")
    print(f"load model     {t_load:>7.2f}s   once per process")
    for k, v in totals.items():
        print(f"{k:<14} {v:>7.2f}s   {100 * v / wall:>5.1f}% of the rollout")
    print(f"rollout        {wall:>7.2f}s   {args.steps} steps")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The scheduled producer: run the model once, publish the result, exit.

    python -m scripts.produce                      # newest GFS cycle, 44 steps
    python -m scripts.produce --source archive     # newest moment in the archive
    python -m scripts.produce --init 2026-06-30T18 --steps 44

This is the only thing in the backend that runs Aurora on a timer, and the whole
point of it is that nothing else runs Aurora at all. A request for "the forecast
for Thursday" reads what this already wrote; it never starts a rollout, never
waits three minutes, and never depends on a GPU being free.

Written to be safe under `cron`, which means three things:

* **Idempotent.** If the newest input already has a published run, it exits 0
  without touching the GPU. Two overlapping cron firings do not produce two
  rollouts of the same init time.
* **Single instance.** A lock file with the pid in it, checked for liveness, so
  a run that takes longer than the interval delays the next one instead of
  racing it onto the same card.
* **All-or-nothing.** The run is built under a `.partial` name and renamed in.
  A crash at step 19 leaves the previous forecast serving, untouched.

The input is the live operational feed by default: `app.sources.gfs` pulls the
newest NOAA GFS analysis and the one before it straight off S3, 54.6 MB of GRIB
per timestep instead of the 500 MB file, and hands `batch_builder` the same
interface the local archive does. `--source archive` still runs from ERA5 on
disk, which is what the checkpoint was trained on and what a comparison should
use.

**Why cron fires every three hours when the cycles are six-hourly.** Global
analysis is six-hourly because the assimilation window is; there is no 03z. What
the extra tick buys is latency — 00z lands on the bucket around 03:30 UTC, so a
three-hourly poll picks it up within about ninety minutes, where a six-hourly one
can be nearly six hours late. The ticks that find nothing new hit
`already_published` and exit 0 without loading torch. See `deploy/crontab`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import batch_builder, config, publish  # noqa: E402
from app.era5_store import ERA5Store  # noqa: E402


class Lock:
    """One producer at a time, without a daemon to hold the lock.

    `O_EXCL` is the atomic part. The pid inside is what makes a lock left by a
    killed process recoverable — otherwise the first crash stops the cycle until
    somebody notices and deletes a file.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if self._alive():
                raise SystemExit(f"another producer holds {self.path}; nothing to do")
            print(f"{self.path}: stale lock, taking it")
            self.path.unlink()
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)

    def _alive(self) -> bool:
        try:
            pid = int(self.path.read_text().strip())
        except (ValueError, OSError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def open_source(kind: str, init: dt.datetime | None):
    """The input store, and the init time it can actually support.

    Two shapes of the same question. The archive is a fixed set of moments and
    the newest one is the init; the operational feed has to go and find out
    which cycle is on the bucket, which is a network call and the reason this
    returns the pair rather than making the caller ask twice.
    """
    if kind == "archive":
        store = ERA5Store()
        return store, (init or newest_init(store))
    if kind == "gfs":
        from app.sources.gfs import GFSStore

        store = GFSStore(init=init)
        return store, store.init
    raise SystemExit(f"unknown --source {kind!r}; use archive or gfs")


def newest_init(store: ERA5Store) -> dt.datetime:
    """The latest moment the model can actually be initialised from.

    That is the newest moment itself: Aurora takes the init time *and* the step
    before it as history, so the archive needs two moments and the newest one is
    the init, not the history.
    """
    stamps = store.timestamps
    if len(stamps) < 2:
        raise SystemExit("archive has fewer than two moments; cannot build a batch")
    return stamps[-1]


def already_published(root: Path, init_time: dt.datetime) -> bool:
    run = root / f"run_{init_time:%Y%m%dT%H%M}"
    link = root / "latest"
    return run.exists() and link.exists() and link.resolve() == run.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=dt.datetime.fromisoformat, default=None,
                    help="init time; defaults to the newest the archive supports")
    ap.add_argument("--steps", type=int, default=config.FORECAST_STEPS)
    ap.add_argument("--root", type=Path, default=config.FORECAST_ROOT)
    ap.add_argument("--keep", type=int, default=config.FORECAST_KEEP)
    ap.add_argument("--vars", nargs="*", default=list(publish.SURFACE_VARS))
    ap.add_argument("--source", choices=("gfs", "archive"), default="gfs",
                    help="gfs: the live operational feed. archive: ERA5 on disk")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if this init time is already published")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    with Lock(args.root / "producer.lock"):
        store, init = open_source(args.source, args.init)
        # Checked before a single field is downloaded or decoded. At the
        # three-hourly cadence half the ticks land on a cycle that is already
        # published, and those must cost nothing.
        if not args.force and already_published(args.root, init):
            print(f"{init:%Y-%m-%d %H:%M} already published; nothing to do")
            return 0

        print(f"source {args.source}, init {init:%Y-%m-%d %H:%M}, {args.steps} steps "
              f"(+{args.steps * config.STEP_HOURS} h), vars {args.vars}")
        t0 = time.perf_counter()
        batch = batch_builder.build_batch(store, init)
        print(f"input ready in {time.perf_counter() - t0:.1f}s")

        # Imported here, not at the top: the checkpoint pulls in torch and
        # ~40 s of load, and `--help` should not pay for that.
        from app.inference import AuroraEngine

        t0 = time.perf_counter()
        engine = AuroraEngine()
        print(f"{config.MODEL_NAME} loaded on {config.DEVICE} "
              f"in {time.perf_counter() - t0:.1f}s")

        pub = publish.RunPublisher(args.root, init, args.steps, variables=args.vars,
                                   source=args.source)
        t0 = time.perf_counter()
        for i, pred in enumerate(engine.rollout(batch, args.steps), start=1):
            pub.add(pred)
            if i % 4 == 0 or i == args.steps:
                print(f"  step {i}/{args.steps}  +{i * config.STEP_HOURS}h  "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
        roll_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        path = pub.finish(keep=args.keep)
        print(f"published {path} in {time.perf_counter() - t0:.1f}s "
              f"(rollout {roll_s:.1f}s)")
        for old in publish.runs(args.root):
            print(f"  kept {old.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

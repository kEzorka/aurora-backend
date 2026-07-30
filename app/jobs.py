"""Single-worker forecast queue.

A rollout takes minutes, so HTTP requests hand work to one background thread
and poll for the result. One worker, because one model instance owns one GPU;
scaling out means one process per GPU, not more threads.

Job state lives in `registry.Registry`, not in this process, so that the
several processes those GPUs imply can answer for each other's jobs — and so
that a request for a forecast that already exists is answered from disk
instead of computed a second time.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
import traceback
import uuid

from . import batch_builder, config, postprocess
from .era5_store import ERA5Store
from .inference import AuroraEngine
from .registry import Job, Registry, Status, store_size, utc_now

__all__ = ["ForecastService", "Job", "Status"]


class ForecastService:
    def __init__(self) -> None:
        self.store = ERA5Store()
        self.registry = Registry()
        self.engine: AuroraEngine | None = None  # loaded lazily on first job
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------- API

    def submit(self, init_time: dt.datetime, steps: int) -> tuple[Job, bool]:
        """Return the job for this request and whether it was already there.

        A forecast is identified by what it contains — the init time, the lead
        and the precision it was computed in — so an identical request is
        answered with the existing job. That matters more than it sounds:
        `ForecastWriter` deletes the store at the output path before writing, so
        recomputing a duplicate used to destroy the finished copy another caller
        still held a download link for.
        """
        ready = self.registry.find_ready(init_time, steps, config.AUTOCAST)
        if ready is not None:
            self.registry.touch(ready.id)
            return ready, True

        # A job somebody else queued a second ago is just as good as a
        # finished one; two callers asking for the same forecast at once
        # should wait on one rollout, not occupy two cards with the same work.
        for job in self.registry.all():
            if (
                job.status in ("queued", "running")
                and job.init_time == init_time
                and job.steps == steps
                and job.precision == config.AUTOCAST
            ):
                return job, True

        prev = init_time - dt.timedelta(hours=config.STEP_HOURS)
        for t in (prev, init_time):
            if not self.store.has(t):
                self.store._require(t)

        job = self.registry.add(
            Job(
                id=uuid.uuid4().hex[:12],
                init_time=init_time,
                steps=steps,
                precision=config.AUTOCAST,
            )
        )
        self._queue.put(job.id)
        return job, False

    def get(self, job_id: str) -> Job | None:
        return self.registry.get(job_id)

    def all(self) -> list[Job]:
        return self.registry.all()

    def touch(self, job_id: str) -> None:
        self.registry.touch(job_id)

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.registry.get(job_id)
            if job is None:
                continue
            try:
                self.registry.update(job_id, status="running")
                if self.engine is None:
                    self.engine = AuroraEngine()

                batch = batch_builder.build_batch(self.store, job.init_time)

                # Each step is written and released before the next one is
                # computed, so a 40-step job costs one step of host memory.
                writer = postprocess.ForecastWriter(
                    job.init_time, job.steps, precision=job.precision
                )
                done = 0
                for pred in self.engine.rollout(batch, job.steps):
                    writer.add(pred)
                    done += 1
                    self.registry.update(job_id, progress=done)

                path = writer.finish()
                self.registry.update(
                    job_id,
                    status="done",
                    output=str(path),
                    size_bytes=store_size(path),
                    last_access=utc_now(),
                )
                # Checked after every finished forecast rather than on a timer:
                # this is the only moment the total can have grown.
                self.registry.evict()
            except Exception:
                self.registry.update(
                    job_id, status="failed", error=traceback.format_exc(limit=3)
                )

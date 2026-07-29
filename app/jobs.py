"""Single-worker forecast queue.

A rollout takes minutes, so HTTP requests hand work to one background thread
and poll for the result. One worker, because one model instance owns one GPU;
scaling out means one process per GPU, not more threads.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Literal

from . import batch_builder, config, postprocess
from .era5_store import ERA5Store
from .inference import AuroraEngine

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    init_time: dt.datetime
    steps: int
    status: Status = "queued"
    progress: int = 0
    output: str | None = None
    error: str | None = None
    created: dt.datetime = field(default_factory=dt.datetime.utcnow)


class ForecastService:
    def __init__(self) -> None:
        self.store = ERA5Store()
        self.engine: AuroraEngine | None = None  # loaded lazily on first job
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------- API

    def submit(self, init_time: dt.datetime, steps: int) -> Job:
        prev = init_time - dt.timedelta(hours=config.STEP_HOURS)
        for t in (prev, init_time):
            if not self.store.has(t):
                self.store._require(t)

        job = Job(id=uuid.uuid4().hex[:12], init_time=init_time, steps=steps)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            job = self.get(self._queue.get())
            if job is None:
                continue
            try:
                job.status = "running"
                if self.engine is None:
                    self.engine = AuroraEngine()

                batch = batch_builder.build_batch(self.store, job.init_time)

                # Each step is written and released before the next one is
                # computed, so a 40-step job costs one step of host memory.
                writer = postprocess.ForecastWriter(job.init_time, job.steps)
                for pred in self.engine.rollout(batch, job.steps):
                    writer.add(pred)
                    job.progress += 1

                job.output = str(writer.finish())
                job.status = "done"
            except Exception:
                job.error = traceback.format_exc(limit=3)
                job.status = "failed"

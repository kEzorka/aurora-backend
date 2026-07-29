"""HTTP surface.

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .jobs import ForecastService
from .registry import Job

service: ForecastService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Indexing the archive touches every month file, so do it once at startup
    # rather than on the first request. The model itself stays lazy: loading
    # the checkpoint costs ~40 s and a server that is only ever asked for
    # /health should not pay it.
    global service
    service = ForecastService()
    yield
    service.store.close()
    service.registry.close()


app = FastAPI(title="Aurora forecast backend", lifespan=lifespan)


class ForecastRequest(BaseModel):
    init_time: dt.datetime = Field(description="Forecast start, must be a 6-hourly archive timestamp")
    # 60 steps is 15 days. Past ten days the forecast has drifted well away from
    # anything verifiable, but watching it come apart is itself the analysis this
    # backend is for. The cap is disk, not skill: 60 steps is ~11 GB.
    steps: int = Field(default=4, ge=1, le=60, description="Number of 6 h steps to roll forward")


def _svc() -> ForecastService:
    if service is None:
        raise HTTPException(503, "service still starting")
    return service


@app.get("/health")
def health() -> dict:
    s = _svc()
    stamps = s.store.timestamps
    return {
        "model": config.MODEL_NAME,
        "device": config.DEVICE,
        "model_loaded": s.engine is not None,
        "archive": str(config.DATA_ROOT),
        "output_format": config.OUTPUT_FORMAT,
        "archive_from": stamps[0].isoformat(),
        "archive_to": stamps[-1].isoformat(),
        "archive_steps": len(stamps),
        "levels": list(s.store.levels),
        "stored_gb": round(s.registry.total_bytes() / 1024**3, 2),
        "cap_gb": round(config.DISK_CAP_BYTES / 1024**3, 1),
    }


def _job_json(job: Job) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "init_time": job.init_time.isoformat(),
        "steps": job.steps,
        "lead_hours": job.lead_hours,
        "progress": job.progress,
        "output": job.output,
        "error": job.error,
    }


@app.post("/forecast")
def create(req: ForecastRequest) -> dict:
    """Answer with the forecast if it is quick, with a job id if it is not.

    Handing back a job id for work that took three seconds makes every client
    implement polling for nothing, and a cache hit costs no time at all. So
    the request waits for its own result up to INLINE_WAIT_S and only then
    falls back to the asynchronous path. Callers do not have to care which
    happened: `status` says `done` or it does not.
    """
    try:
        job, reused = _svc().submit(req.init_time.replace(tzinfo=None), req.steps)
    except KeyError as e:
        raise HTTPException(400, str(e)) from None

    deadline = time.monotonic() + config.INLINE_WAIT_S
    while job.status not in ("done", "failed") and time.monotonic() < deadline:
        time.sleep(0.25)
        job = _svc().get(job.id) or job

    # `reused` is not decoration: a client whose request matched an
    # already-running job knows why the progress bar starts at 12 of 40.
    return {**_job_json(job), "reused": reused, "events": f"/forecast/{job.id}/events"}


@app.get("/forecast/{job_id}")
def status(job_id: str) -> dict:
    job = _svc().get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return _job_json(job)


# A connection that says nothing for two minutes is a connection nginx closes
# after sixty seconds; the timeout is on silence, not on duration. A step
# lands every 2.65 s, so this stream is never quiet for long, and the
# heartbeat covers the gap before the first one while the checkpoint loads.
HEARTBEAT_S = 15.0
POLL_S = 0.5


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/forecast/{job_id}/events")
async def events(job_id: str, request: Request) -> StreamingResponse:
    """Progress as it happens, instead of the client asking over and over.

    The client opens this once and reads until `done`. What it costs the
    server is one SQLite read every POLL_S — the polling did not disappear,
    it moved to where it is cheap and stopped being the client's problem.
    """
    if _svc().get(job_id) is None:
        raise HTTPException(404, "no such job")

    async def stream():
        last: tuple | None = None
        last_sent = 0.0
        while True:
            # A client that closed the tab should not keep this loop alive.
            if await request.is_disconnected():
                return

            job = _svc().get(job_id)
            if job is None:
                yield _sse("error", {"job_id": job_id, "error": "job disappeared"})
                return

            now = time.monotonic()
            state = (job.status, job.progress)
            if state != last:
                eta = round((job.steps - job.progress) * config.STEP_WALL_S, 1)
                yield _sse(
                    "progress",
                    {
                        "job_id": job.id,
                        "status": job.status,
                        "done": job.progress,
                        "total": job.steps,
                        "eta_s": eta if job.status in ("queued", "running") else 0,
                    },
                )
                last, last_sent = state, now
            elif now - last_sent > HEARTBEAT_S:
                yield ": keepalive\n\n"
                last_sent = now

            if job.status == "done":
                yield _sse("done", {**_job_json(job), "data": f"/forecast/{job.id}/download"})
                return
            if job.status == "failed":
                yield _sse("failed", {"job_id": job.id, "error": job.error})
                return

            await asyncio.sleep(POLL_S)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Without this an nginx in front of the service buffers the whole
        # stream and delivers it at the end, which is exactly not the point.
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.get("/forecast/{job_id}/download")
def download(job_id: str):
    """Stream the forecast.

    A zarr forecast is a directory, so it cannot be served as a file. Rather
    than packing a 700 MB copy on disk first, tar it straight to the socket —
    the client unpacks and opens the store. NetCDF output is still a plain
    file and is served as one.
    """
    job = _svc().get(job_id)
    if job is None or job.status != "done" or job.output is None:
        raise HTTPException(404, "no finished output for this job")

    # Eviction ranks by last read, so a forecast being downloaded has to say
    # so — otherwise the one everybody actually uses looks idle and goes first.
    _svc().touch(job_id)
    path = Path(job.output)
    if path.is_file():
        return FileResponse(path, media_type="application/x-netcdf", filename=path.name)

    def tar_stream():
        proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(path.parent), path.name],
            stdout=subprocess.PIPE,
        )
        try:
            while chunk := proc.stdout.read(1 << 20):
                yield chunk
        finally:
            # A client that disconnects mid-download would otherwise leave tar
            # blocked on a full pipe forever.
            proc.stdout.close()
            proc.terminate()
            proc.wait()

    return StreamingResponse(
        tar_stream(),
        media_type="application/x-tar",
        headers={"content-disposition": f'attachment; filename="{path.name}.tar"'},
    )


@app.post("/forecast/{job_id}/pin")
def pin(job_id: str, pinned: bool = True) -> dict:
    """Keep this forecast until it is unpinned.

    For the reference runs an experiment is scored against: eviction ranks by
    last read, and a baseline nobody has opened this week is exactly the thing
    that must not disappear.
    """
    job = _svc().get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    _svc().registry.pin(job_id, pinned)
    return {"job_id": job_id, "pinned": pinned}


@app.get("/jobs")
def jobs() -> list[dict]:
    return [{"job_id": j.id, "status": j.status, "progress": j.progress} for j in _svc().all()]

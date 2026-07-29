"""HTTP surface.

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime as dt
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .jobs import ForecastService

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
    steps: int = Field(default=4, ge=1, le=40, description="Number of 6 h steps to roll forward")


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


@app.post("/forecast")
def create(req: ForecastRequest) -> dict:
    try:
        job, reused = _svc().submit(req.init_time.replace(tzinfo=None), req.steps)
    except KeyError as e:
        raise HTTPException(400, str(e)) from None
    # `reused` is not decoration: a client that sees `done` immediately can
    # skip polling entirely, and a client that sees its request matched an
    # already-running job knows why the progress bar starts at 12 of 40.
    return {"job_id": job.id, "status": job.status, "reused": reused}


@app.get("/forecast/{job_id}")
def status(job_id: str) -> dict:
    job = _svc().get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {
        "job_id": job.id,
        "status": job.status,
        "init_time": job.init_time.isoformat(),
        "steps": job.steps,
        "progress": job.progress,
        "output": job.output,
        "error": job.error,
    }


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


@app.get("/jobs")
def jobs() -> list[dict]:
    return [{"job_id": j.id, "status": j.status, "progress": j.progress} for j in _svc().all()]

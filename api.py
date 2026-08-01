"""HTTP surface.

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
        "archive_from": stamps[0].isoformat(),
        "archive_to": stamps[-1].isoformat(),
        "archive_steps": len(stamps),
        "levels": list(s.store.levels),
    }


@app.post("/forecast")
def create(req: ForecastRequest) -> dict:
    try:
        job = _svc().submit(req.init_time.replace(tzinfo=None), req.steps)
    except KeyError as e:
        raise HTTPException(400, str(e)) from None
    return {"job_id": job.id, "status": job.status}


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
def download(job_id: str) -> FileResponse:
    job = _svc().get(job_id)
    if job is None or job.status != "done" or job.output is None:
        raise HTTPException(404, "no finished output for this job")
    return FileResponse(job.output, media_type="application/x-netcdf")


@app.get("/jobs")
def jobs() -> list[dict]:
    return [{"job_id": j.id, "status": j.status, "progress": j.progress} for j in _svc().all()]

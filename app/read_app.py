"""The read-only server: `/v1` and a health check, and nothing that owns a GPU.

    uvicorn app.read_app:app --host 127.0.0.1 --port 8078 --workers 8

This exists as a second ASGI app rather than a flag on `app.api` because the two
have opposite scaling shapes.

`app.api` cannot run with `--workers`. Its lifespan builds a `ForecastService`
per worker, each of which would claim the same GPU, and `registry.py` keeps its
queue in process memory, so four workers means four independent queues and the
job dedup quietly stops working. One process is not a limitation there — it is
what keeps the accounting honest.

The read path has none of that. It opens two zarr stores and never writes, so
every worker is an identical, independent reader and the operating system's page
cache is shared between them anyway. The endpoints are `def`, not `async def`, so
one process already serves them from anyio's thread pool; that is enough while
the work releases the GIL (blosc decode does) and not enough when it does not
(JSON serialising does not). `bench/results/concurrency.json` measures where that
line falls.

Splitting the apps is what the brief asks for in the first place: the model runs
once every three hours, on a schedule, and nothing on the request path should be
able to start it. A caller pointed at this port cannot.
"""

from __future__ import annotations

from fastapi import FastAPI

from . import config, read_api

app = FastAPI(
    title="Aurora read API",
    description="Forecasts and history that a producer has already written. "
                "No model runs here.",
)
app.include_router(read_api.router)


@app.get("/health")
def health() -> dict:
    """Deliberately cheap: no store is opened, so this stays honest as a
    liveness check even when the archive is missing or being rebuilt."""
    return {
        "ok": True,
        "role": "read-only",
        "history_store": str(config.HISTORY_STORE),
        "map_store": str(config.DATA_ROOT),
    }

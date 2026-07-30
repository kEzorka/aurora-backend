"""Exercise the progress stream and the inline wait without a GPU.

    python -m scripts.check_sse

The service is replaced by a stub that advances a job on a timer, so what is
under test is the HTTP behaviour: that a short request answers with its own
result, that a long one falls back to a job id, and that the stream reports
every step and then closes.
"""

from __future__ import annotations

import datetime as dt
import threading
import time

from fastapi.testclient import TestClient

from app import api, config
from app.registry import Job


class StubService:
    """A rollout that costs no GPU: one step every `step_s` seconds."""

    def __init__(self, steps: int, step_s: float):
        self.job = Job(
            id="stub01",
            init_time=dt.datetime(2026, 5, 1),
            steps=steps,
            status="queued",
        )
        self.step_s = step_s
        self.registry = self
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        self.job.status = "running"
        for i in range(self.job.steps):
            time.sleep(self.step_s)
            self.job.progress = i + 1
        self.job.output = "/tmp/stub.zarr"
        self.job.status = "done"

    # the parts of ForecastService the endpoints touch
    def get(self, job_id):
        return self.job if job_id == self.job.id else None

    def submit(self, init_time, steps):
        return self.job, False

    def touch(self, job_id):
        pass

    def total_bytes(self):
        return 0


def check_inline_hit() -> None:
    """A request shorter than the inline wait comes back finished."""
    config.INLINE_WAIT_S = 5.0
    api.service = StubService(steps=2, step_s=0.3)
    client = TestClient(api.app)
    t0 = time.monotonic()
    body = client.post("/forecast", json={"init_time": "2026-05-01T00:00", "steps": 2}).json()
    waited = time.monotonic() - t0
    assert body["status"] == "done", f"short request did not finish inline: {body}"
    print(f"inline: finished in {waited:.1f} s, no job id needed by the client")


def check_inline_miss() -> None:
    """A request longer than the wait gives up and hands back the id."""
    config.INLINE_WAIT_S = 1.0
    api.service = StubService(steps=40, step_s=0.2)
    client = TestClient(api.app)
    t0 = time.monotonic()
    body = client.post("/forecast", json={"init_time": "2026-05-01T00:00", "steps": 40}).json()
    waited = time.monotonic() - t0
    assert body["status"] in ("queued", "running"), body
    assert waited < 2.0, f"gave up late: {waited:.1f} s"
    assert body["events"].endswith("/events"), body
    print(f"fallback: gave up after {waited:.1f} s, pointed at {body['events']}")


def check_stream() -> None:
    """Every step is reported once, and the stream ends by itself."""
    api.service = StubService(steps=6, step_s=0.4)
    client = TestClient(api.app)
    seen: list[str] = []
    progress = 0
    with client.stream("GET", "/forecast/stub01/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream"), r.headers
        for line in r.iter_lines():
            if line.startswith("event:"):
                seen.append(line.split(": ", 1)[1])
            if line.startswith("data:") and '"done":' in line:
                progress += 1
    assert seen[-1] == "done", f"stream did not end on done: {seen}"
    assert seen.count("progress") >= 6, f"steps missing from the stream: {seen}"
    print(f"stream: {seen.count('progress')} progress events then {seen[-1]}, "
          f"connection closed by the server")


def check_evicted_stream() -> None:
    """A job whose store was reclaimed ends the stream instead of hanging.

    `evicted` is a terminal state the client has never heard of. If the stream
    only closed on `done` and `failed`, this connection would sit there sending
    keepalives until the client gave up.
    """
    api.service = StubService(steps=6, step_s=0.2)
    api.service.job.status = "evicted"
    api.service.job.error = "output evicted"
    client = TestClient(api.app)
    seen: list[str] = []
    with client.stream("GET", "/forecast/stub01/events") as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                seen.append(line.split(": ", 1)[1])
    assert seen[-1] == "failed", f"evicted job did not end the stream: {seen}"
    print(f"stream: evicted job closes as {seen[-1]}, not left open")


def check_precision_paths() -> None:
    """An fp32 reference and its fp16 counterpart do not share a store.

    Same init, same lead, different precision: before the precision went into
    the name these were one path, and `ForecastWriter` deletes what is there
    before writing — so computing the reference destroyed the run it was meant
    to score. Checked here rather than in check_registry because `postprocess`
    pulls in aurora and torch.
    """
    from app.postprocess import output_path

    init = dt.datetime(2026, 5, 1)
    p16 = output_path(init, 7, config.OUTPUT_DIR, "fp16")
    p32 = output_path(init, 7, config.OUTPUT_DIR, "off")
    assert p16 != p32, f"fp16 and fp32 would write to one path: {p16}"
    print(f"paths: {p16.name} vs {p32.name}")


def check_404() -> None:
    api.service = StubService(steps=1, step_s=0.1)
    client = TestClient(api.app)
    assert client.get("/forecast/nope/events").status_code == 404
    print("stream: unknown job is 404, not an empty stream")


if __name__ == "__main__":
    check_inline_hit()
    check_inline_miss()
    check_stream()
    check_evicted_stream()
    check_precision_paths()
    check_404()
    print("ok")

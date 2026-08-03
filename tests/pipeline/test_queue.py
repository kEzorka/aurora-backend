"""SQLite-очередь: идемпотентность, lease и восстановление после смерти воркера."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pipeline.queue import (
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    claim,
    complete,
    enqueue,
    fail,
    heartbeat,
    list_jobs,
    open_queue,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
INIT = "2026-08-03T00:00:00Z"


def test_enqueue_is_idempotent_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    first = open_queue(path)
    job = enqueue(first, INIT, now=NOW)
    assert enqueue(first, INIT, now=NOW + timedelta(hours=1)).id == job.id
    first.close()

    reopened = open_queue(path)
    assert list_jobs(reopened) == (job,)
    assert reopened.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_claim_is_oldest_first_and_exclusive(tmp_path: Path) -> None:
    queue = open_queue(tmp_path / "queue.sqlite")
    first = enqueue(queue, INIT, now=NOW)
    enqueue(queue, "2026-08-03T06:00:00Z", now=NOW + timedelta(seconds=1))

    claimed = claim(queue, "gpu-1", now=NOW, lease_seconds=60)
    assert claimed is not None
    assert (claimed.id, claimed.status, claimed.worker, claimed.attempts) == (
        first.id,
        RUNNING,
        "gpu-1",
        1,
    )
    second = claim(queue, "gpu-2", now=NOW, lease_seconds=60)
    assert second is not None and second.init_time == "2026-08-03T06:00:00Z"
    assert claim(queue, "gpu-3", now=NOW, lease_seconds=60) is None


def test_expired_lease_returns_after_worker_restart(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    queue = open_queue(path)
    enqueue(queue, INIT, now=NOW)
    abandoned = claim(queue, "dead-worker", now=NOW, lease_seconds=60)
    assert abandoned is not None
    queue.close()  # процесс умер вместе с connection

    queue = open_queue(path)
    assert claim(queue, "early-worker", now=NOW + timedelta(seconds=59)) is None
    recovered = claim(queue, "new-worker", now=NOW + timedelta(seconds=60))
    assert recovered is not None
    assert (recovered.id, recovered.worker, recovered.attempts) == (
        abandoned.id,
        "new-worker",
        2,
    )


def test_two_connections_cannot_claim_the_same_job(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    first = open_queue(path)
    second = open_queue(path)
    enqueue(first, INIT, now=NOW)

    claimed = claim(first, "gpu-1", now=NOW)
    assert claimed is not None
    assert claim(second, "gpu-2", now=NOW) is None


def test_heartbeat_extends_only_the_owners_lease(tmp_path: Path) -> None:
    queue = open_queue(tmp_path / "queue.sqlite")
    enqueue(queue, INIT, now=NOW)
    job = claim(queue, "gpu-1", now=NOW, lease_seconds=60)
    assert job is not None

    renewed = heartbeat(queue, job.id, "gpu-1", now=NOW + timedelta(seconds=30), lease_seconds=90)
    assert renewed.lease_until == "2026-08-03T12:02:00.000000Z"
    with pytest.raises(LookupError, match="does not own"):
        heartbeat(queue, job.id, "gpu-2", now=NOW)


def test_success_failure_and_retry_are_explicit_states(tmp_path: Path) -> None:
    queue = open_queue(tmp_path / "queue.sqlite")
    enqueue(queue, INIT, now=NOW)
    job = claim(queue, "gpu-1", now=NOW)
    assert job is not None
    retried = fail(queue, job.id, "gpu-1", "CUDA OOM", retry=True, now=NOW)
    assert (retried.status, retried.last_error, retried.worker) == (PENDING, "CUDA OOM", None)

    again = claim(queue, "gpu-2", now=NOW)
    assert again is not None and again.attempts == 2
    finished = complete(queue, again.id, "gpu-2", now=NOW)
    assert (finished.status, finished.worker, finished.lease_until) == (DONE, None, None)

    enqueue(queue, "2026-08-03T06:00:00Z", now=NOW)
    broken = claim(queue, "gpu-2", now=NOW)
    assert broken is not None
    final = fail(queue, broken.id, "gpu-2", "invalid output", retry=False, now=NOW)
    assert (final.status, final.last_error) == (FAILED, "invalid output")


def test_wrong_worker_cannot_finish_someone_elses_job(tmp_path: Path) -> None:
    queue = open_queue(tmp_path / "queue.sqlite")
    enqueue(queue, INIT, now=NOW)
    job = claim(queue, "gpu-1", now=NOW)
    assert job is not None

    with pytest.raises(LookupError, match="does not own"):
        complete(queue, job.id, "gpu-2", now=NOW)


@pytest.mark.parametrize("init_time", ["2026-08-03T03:00:00Z", "not-a-time"])
def test_invalid_cycles_never_enter_the_queue(tmp_path: Path, init_time: str) -> None:
    queue = open_queue(tmp_path / "queue.sqlite")
    with pytest.raises(ValueError):
        enqueue(queue, init_time, now=NOW)
    assert list_jobs(queue) == ()

"""Устойчивая SQLite-очередь инференса (BACKLOG 4.3).

Задача раз в шесть часов не оправдывает отдельный брокер, но обычный список в
памяти теряется при рестарте. Lease делает ``running`` временным обещанием:
если воркер умер, другой заберёт задачу после ``lease_until`` и увеличит число
попыток. Все переходы выполняются под ``BEGIN IMMEDIATE`` — двух владельцев у
одной GPU-задачи быть не может даже при двух процессах.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

from pipeline.schedule import cycle

PENDING: Final = "pending"
RUNNING: Final = "running"
DONE: Final = "done"
FAILED: Final = "failed"
STATUSES: Final = (PENDING, RUNNING, DONE, FAILED)
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"
DEFAULT_LEASE_SEC: Final = 30 * 60


class Job(NamedTuple):
    id: int
    init_time: str
    status: str
    attempts: int
    worker: str | None
    lease_until: str | None
    last_error: str | None
    created_at: str
    updated_at: str


def open_queue(path: str | Path) -> sqlite3.Connection:
    """Открыть очередь, включить WAL и создать схему идемпотентно."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        connection.close()
        raise RuntimeError(f"queue journal_mode: got {mode!r}, expected 'wal'")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            init_time TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            worker TEXT,
            lease_until TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (
                (status = 'running' AND worker IS NOT NULL AND lease_until IS NOT NULL)
                OR (status != 'running' AND worker IS NULL AND lease_until IS NULL)
            )
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status, lease_until, created_at)"
    )
    return connection


def enqueue(connection: sqlite3.Connection, init_time: str, *, now: datetime | None = None) -> Job:
    """Поставить срок один раз; повтор возвращает ту же задачу."""
    cycle(init_time)  # неверный 03Z не должен навсегда попасть в очередь
    stamp = _stamp(now)
    connection.execute(
        """
        INSERT INTO jobs(init_time, status, created_at, updated_at)
        VALUES (?, 'pending', ?, ?)
        ON CONFLICT(init_time) DO NOTHING
        """,
        (init_time, stamp, stamp),
    )
    return _by_init(connection, init_time)


def claim(
    connection: sqlite3.Connection,
    worker: str,
    *,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_LEASE_SEC,
) -> Job | None:
    """Атомарно забрать старейшую pending или просроченную running-задачу."""
    if not worker.strip():
        raise ValueError("worker: expected a non-empty identifier")
    if lease_seconds <= 0:
        raise ValueError(f"lease_seconds: got {lease_seconds}, expected positive")
    moment = _utc(now)
    stamp = _stamp(moment)
    lease = _stamp(moment + timedelta(seconds=lease_seconds))
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """
            SELECT id FROM jobs
            WHERE status = 'pending'
               OR (status = 'running' AND lease_until <= ?)
            ORDER BY created_at, id
            LIMIT 1
            """,
            (stamp,),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        job_id = int(row["id"])
        connection.execute(
            """
            UPDATE jobs
            SET status = 'running', attempts = attempts + 1,
                worker = ?, lease_until = ?, updated_at = ?
            WHERE id = ?
            """,
            (worker, lease, stamp, job_id),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return get(connection, job_id)


def heartbeat(
    connection: sqlite3.Connection,
    job_id: int,
    worker: str,
    *,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_LEASE_SEC,
) -> Job:
    """Продлить lease только его текущему владельцу."""
    if lease_seconds <= 0:
        raise ValueError(f"lease_seconds: got {lease_seconds}, expected positive")
    moment = _utc(now)
    changed = connection.execute(
        """
        UPDATE jobs SET lease_until = ?, updated_at = ?
        WHERE id = ? AND status = 'running' AND worker = ?
        """,
        (_stamp(moment + timedelta(seconds=lease_seconds)), _stamp(moment), job_id, worker),
    ).rowcount
    if changed != 1:
        raise LookupError(f"job {job_id}: worker {worker!r} does not own a running lease")
    return get(connection, job_id)


def complete(
    connection: sqlite3.Connection,
    job_id: int,
    worker: str,
    *,
    now: datetime | None = None,
) -> Job:
    """Завершить только свою running-задачу."""
    return _finish(connection, job_id, worker, DONE, None, now=now)


def fail(
    connection: sqlite3.Connection,
    job_id: int,
    worker: str,
    error: str,
    *,
    retry: bool,
    now: datetime | None = None,
) -> Job:
    """Вернуть задачу в pending либо окончательно отметить failed."""
    if not error.strip():
        raise ValueError("error: expected a non-empty reason")
    status = PENDING if retry else FAILED
    return _finish(connection, job_id, worker, status, error, now=now)


def get(connection: sqlite3.Connection, job_id: int) -> Job:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise LookupError(f"job {job_id}: not found")
    return _job(row)


def list_jobs(connection: sqlite3.Connection) -> tuple[Job, ...]:
    return tuple(_job(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id"))


def _finish(
    connection: sqlite3.Connection,
    job_id: int,
    worker: str,
    status: str,
    error: str | None,
    *,
    now: datetime | None,
) -> Job:
    changed = connection.execute(
        """
        UPDATE jobs
        SET status = ?, worker = NULL, lease_until = NULL,
            last_error = ?, updated_at = ?
        WHERE id = ? AND status = 'running' AND worker = ?
        """,
        (status, error, _stamp(now), job_id, worker),
    ).rowcount
    if changed != 1:
        raise LookupError(f"job {job_id}: worker {worker!r} does not own a running lease")
    return get(connection, job_id)


def _by_init(connection: sqlite3.Connection, init_time: str) -> Job:
    row = connection.execute("SELECT * FROM jobs WHERE init_time = ?", (init_time,)).fetchone()
    if row is None:  # pragma: no cover - INSERT и SELECT в одной connection
        raise RuntimeError(f"job {init_time}: insert disappeared")
    return _job(row)


def _job(row: sqlite3.Row) -> Job:
    return Job(
        id=int(row["id"]),
        init_time=str(row["init_time"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        worker=None if row["worker"] is None else str(row["worker"]),
        lease_until=None if row["lease_until"] is None else str(row["lease_until"]),
        last_error=None if row["last_error"] is None else str(row["last_error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _utc(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(moment: datetime | None) -> str:
    return _utc(moment).strftime(TIME_FORMAT)

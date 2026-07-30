"""The job registry: one SQLite file that outlives the process.

Three things live here that a dict in process memory cannot hold.

A restart used to orphan every finished forecast — the store was still on
disk, but the job id that named it was gone. A file survives the restart.

Four GPUs mean four processes, and a load balancer sends the POST to one and
the status poll to another. A dict in process 0 is invisible to process 2,
which answers 404 for a job it never saw. One file, opened by all of them,
is the cheapest fix that actually works; Redis would be a second daemon to
run for a few writes a day.

And a forecast is addressable by what it contains, not only by its id, so a
repeat request can be answered from disk instead of recomputed. That lookup
is a row, not a directory walk.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import config

# `evicted` is not a kind of failure and is kept apart from one. A forecast that
# rolled out correctly and whose store was later reclaimed for disk has nothing
# wrong with it; calling that `failed` loses the only record that the box ever
# produced it, and makes "how many forecasts has this run?" unanswerable.
Status = Literal["queued", "running", "done", "failed", "evicted"]

# Nothing more happens to a job in these states, so a caller may stop waiting.
TERMINAL: tuple[Status, ...] = ("done", "failed", "evicted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    init_time   TEXT NOT NULL,
    steps       INTEGER NOT NULL,
    precision   TEXT NOT NULL DEFAULT 'fp16',
    status      TEXT NOT NULL,
    progress    INTEGER NOT NULL DEFAULT 0,
    output      TEXT,
    error       TEXT,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    pinned      INTEGER NOT NULL DEFAULT 0,
    created     TEXT NOT NULL,
    last_access TEXT NOT NULL
);
-- The dedup lookup and the eviction scan are the only two queries that run
-- often enough to care about.
CREATE INDEX IF NOT EXISTS jobs_content ON jobs (init_time, steps, precision, status);
CREATE INDEX IF NOT EXISTS jobs_access  ON jobs (status, last_access);
"""


@dataclass
class Job:
    id: str
    init_time: dt.datetime
    steps: int
    # Part of what identifies a forecast, not a note about how it was made: an
    # fp16 store is not the answer to a request made in fp32. The default is read
    # once, when this class is defined — it is not live, so a test that reassigns
    # `config.AUTOCAST` at runtime will not see it here. Callers that care pass
    # the value; `jobs.submit` does.
    precision: str = config.AUTOCAST
    status: Status = "queued"
    progress: int = 0
    output: str | None = None
    error: str | None = None
    size_bytes: int = 0
    pinned: bool = False
    created: dt.datetime = dt.datetime.min
    last_access: dt.datetime = dt.datetime.min

    @property
    def lead_hours(self) -> int:
        return self.steps * config.STEP_HOURS


def utc_now() -> str:
    """UTC, spelled without `utcnow()`, and naive on purpose.

    `datetime.utcnow()` is deprecated as of the Python 3.12 this service now
    runs on. The replacement is tz-aware, and an aware `isoformat()` carries a
    `+00:00` the rows already in registry.db do not have — which would compare
    unequal to itself and defeat the dedup lookup on `init_time`, whose value
    comes from a request the API deliberately makes naive. So drop the tzinfo
    again: same string as before, no warning.
    """
    return dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        init_time=dt.datetime.fromisoformat(row["init_time"]),
        steps=row["steps"],
        precision=row["precision"],
        status=row["status"],
        progress=row["progress"],
        output=row["output"],
        error=row["error"],
        size_bytes=row["size_bytes"],
        pinned=bool(row["pinned"]),
        created=dt.datetime.fromisoformat(row["created"]),
        last_access=dt.datetime.fromisoformat(row["last_access"]),
    )


def store_size(path: Path) -> int:
    """Bytes on disk under `path`.

    A zarr forecast is a directory of thousands of files, so this walks. It is
    called once when a job finishes, never on a read path — the running total
    lives in the `size_bytes` column precisely so that eviction does not have
    to walk anything.
    """
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass  # a concurrent eviction got there first
    return total


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.STATE_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread is off because the worker thread and the HTTP
        # threadpool both touch this; the lock below serialises them.
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL is what makes several processes on one file workable: readers do
        # not block the writer, and a poll from another worker sees committed
        # rows immediately.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        """Bring an older registry.db up to the current schema.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a column added above would be missing from every file written
        before it — and the first `SELECT precision` would raise. Runs before the
        lock exists because `__init__` is the only caller.

        Existing rows keep the column default, `fp16`: it is the default the
        service runs in, so it is the precision an undated row was almost
        certainly computed at. Rows that predate the column and were in fact
        fp32 will be re-rolled rather than matched — a wasted rollout, which is
        the safe direction to be wrong in.
        """
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(jobs)")}
        if "precision" in cols:
            return
        self._db.execute("ALTER TABLE jobs ADD COLUMN precision TEXT NOT NULL DEFAULT 'fp16'")
        # The index was created over the old column list under this same name,
        # so `CREATE INDEX IF NOT EXISTS` above left it alone. Replace it.
        self._db.execute("DROP INDEX IF EXISTS jobs_content")
        self._db.execute(
            "CREATE INDEX jobs_content ON jobs (init_time, steps, precision, status)"
        )

    # ------------------------------------------------------------------ write

    def add(self, job: Job) -> Job:
        now = utc_now()
        job.created = dt.datetime.fromisoformat(now)
        job.last_access = job.created
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, init_time, steps, precision, status, progress,"
                " created, last_access) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    job.id,
                    job.init_time.isoformat(),
                    job.steps,
                    job.precision,
                    job.status,
                    now,
                    now,
                ),
            )
            self._db.commit()
        return job

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(
                f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id)
            )
            self._db.commit()

    def pin(self, job_id: str, pinned: bool = True) -> None:
        """Exempt a forecast from eviction.

        This backend exists to measure Aurora, and a measurement needs its
        reference to stay put. The fp32 baselines that every fp16 comparison
        is scored against are not "recomputable in 130 s" in any useful sense:
        recomputing them changes what the older numbers meant. Everything else
        here is disposable; these are not, so they are marked rather than
        trusted to stay recently-read.
        """
        self.update(job_id, pinned=int(pinned))

    def touch(self, job_id: str) -> None:
        """Record that somebody read this forecast.

        Eviction ranks by this column, not by creation time: a week-old run
        that is read every day is worth more than yesterday's that nobody
        asked for.
        """
        self.update(job_id, last_access=utc_now())

    # ------------------------------------------------------------------- read

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def all(self) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created DESC").fetchall()
        return [_row_to_job(r) for r in rows]

    def find_ready(self, init_time: dt.datetime, steps: int, precision: str) -> Job | None:
        """A finished job for exactly this request, if its output still exists.

        Precision is part of "exactly this request". `AURORA_AUTOCAST` is
        process-level, so a service restarted in fp32 over the same registry.db
        used to be handed back the fp16 store from before the restart — the one
        case fp32 exists to serve. `bench/testset.py` restarts precisely that
        way, which made the defect a routine event rather than a corner case.

        The disk check is not paranoia: eviction deletes stores, and a row that
        outlived its directory would hand the caller a 404 dressed as a 200.
        Such a row is corrected on the spot rather than left to lie again.

        No default for `precision` on purpose. A default would be
        `config.AUTOCAST`, and a caller that forgot the argument would silently
        get the process-global precision substituted — which is the exact defect
        this parameter exists to remove.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE init_time = ? AND steps = ? AND precision = ?"
                " AND status = 'done' ORDER BY last_access DESC",
                (init_time.isoformat(), steps, precision),
            ).fetchall()
        for row in rows:
            job = _row_to_job(row)
            if job.output and Path(job.output).exists():
                return job
            self.update(job.id, status="evicted", error="output evicted", size_bytes=0)
        return None

    def total_bytes(self) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS n FROM jobs WHERE status = 'done'"
            ).fetchone()
        return int(row["n"])

    # -------------------------------------------------------------- eviction

    def evict(self, cap_bytes: int | None = None, low_bytes: int | None = None) -> list[str]:
        """Delete least-recently-read forecasts until under the low mark.

        Two marks, not one. With a single threshold the store sits exactly on
        it and every new forecast triggers another deletion; the gap between
        cap and low is what makes eviction a rare event instead of a per-job
        one.

        Only forecasts are ever deleted. The ERA5 archive is not in this table
        and must not be: it cost a CDS quota and cannot be recomputed, while
        any forecast here can. Pinned rows are skipped for the same reason at
        a smaller scale — see `pin`.
        """
        cap = cap_bytes if cap_bytes is not None else config.DISK_CAP_BYTES
        low = low_bytes if low_bytes is not None else config.DISK_LOW_BYTES
        if self.total_bytes() <= cap:
            return []

        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE status = 'done' AND pinned = 0"
                " ORDER BY last_access ASC"
            ).fetchall()

        removed: list[str] = []
        # Counts pinned bytes too, deliberately: if the pinned references
        # alone exceed the low mark, this deletes everything it may and stops
        # above it. Running out of evictable data is the honest outcome there,
        # not a reason to start deleting references.
        total = self.total_bytes()
        for row in rows:
            if total <= low:
                break
            job = _row_to_job(row)
            if job.output:
                path = Path(job.output)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink(missing_ok=True)
            total -= job.size_bytes
            # `evicted`, not `failed`: the rollout was fine, the disk was not.
            self.update(job.id, status="evicted", error="evicted", output=None, size_bytes=0)
            removed.append(job.id)
        return removed

    def close(self) -> None:
        with self._lock:
            self._db.close()

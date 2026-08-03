"""Durable asynchronous Zarr exports (API contract lane 3).

The queue and its artifacts live below ``scratch/``.  SQLite makes a submitted
job visible after an API restart; a lease makes an interrupted ``running`` job
claimable again without allowing two healthy workers to write it at once.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, cast

import numpy as np
import xarray as xr

from contracts import canon

QUEUED: Final = "queued"
RUNNING: Final = "running"
READY: Final = "ready"
FAILED: Final = "failed"
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"
DEFAULT_TTL_SEC: Final = 24 * 60 * 60
DEFAULT_QUOTA_BYTES: Final = 10 * 1024**3
DEFAULT_LEASE_SEC: Final = 30 * 60
POLL_SEC: Final = 0.25


class Spec(NamedTuple):
    variables: tuple[str, ...]
    bbox: tuple[float, float, float, float]
    start: str
    stop: str
    stride: int
    format: str = "zarr"


class Job(NamedTuple):
    job_id: str
    status: str
    source: str
    init_time: str
    spec: Spec
    created_at: str
    updated_at: str
    expires_at: str | None
    artifact: str | None
    error: str | None


class ExportManager:
    """One local worker over a process-safe durable queue."""

    def __init__(
        self,
        root: str | Path,
        *,
        ttl_seconds: int = DEFAULT_TTL_SEC,
        quota_bytes: int = DEFAULT_QUOTA_BYTES,
        lease_seconds: int = DEFAULT_LEASE_SEC,
    ) -> None:
        if ttl_seconds <= 0 or quota_bytes <= 0 or lease_seconds <= 0:
            raise ValueError("export ttl, quota and lease must be positive")
        self.root = Path(root)
        self.scratch = self.root / "scratch"
        self.files = self.scratch / "exports"
        self.work = self.scratch / "work"
        self.database = self.scratch / "exports.sqlite3"
        self.ttl_seconds = ttl_seconds
        self.quota_bytes = quota_bytes
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="aurora-export", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    def submit(
        self,
        layer: str | Path,
        spec: Spec,
        *,
        source: str,
        init_time: str,
        now: datetime | None = None,
    ) -> Job:
        validate_spec(layer, spec)
        stamp = _stamp(now)
        job_id = "exp_" + secrets.token_hex(13)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO exports(
                    job_id, status, source, init_time, layer, spec,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source,
                    init_time,
                    str(Path(layer).resolve()),
                    json.dumps(spec._asdict(), separators=(",", ":")),
                    stamp,
                    stamp,
                ),
            )
            job = get(connection, job_id)
        self._wake.set()
        return job

    def lookup(self, job_id: str, *, now: datetime | None = None) -> Job:
        with self._connect() as connection:
            job = get(connection, job_id)
        if job.status == READY and is_expired(job, now=now):
            raise ExpiredError(job_id)
        if job.status == READY and (job.artifact is None or not Path(job.artifact).is_file()):
            raise ExpiredError(job_id)
        return job

    def artifact(self, job_id: str, *, now: datetime | None = None) -> Path:
        job = self.lookup(job_id, now=now)
        if job.status != READY or job.artifact is None:
            raise NotReadyError(job_id)
        return Path(job.artifact)

    def process_once(self, *, now: datetime | None = None) -> Job | None:
        moment = _utc(now)
        self.cleanup(now=moment)
        owner = f"{os.getpid()}:{threading.get_ident()}"
        with self._connect() as connection:
            claimed = claim(
                connection,
                owner,
                now=moment,
                lease_seconds=self.lease_seconds,
            )
        if claimed is None:
            return None
        try:
            artifact = self._build(claimed)
        except Exception as error:
            with self._connect() as connection:
                return finish_failed(connection, claimed.job_id, owner, str(error), now=moment)
        finished = moment if now is not None else datetime.now(UTC)
        expires = finished + timedelta(seconds=self.ttl_seconds)
        with self._connect() as connection:
            return finish_ready(
                connection,
                claimed.job_id,
                owner,
                artifact,
                expires_at=expires,
                now=finished,
            )

    def cleanup(self, *, now: datetime | None = None) -> int:
        moment = _stamp(now)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT artifact FROM exports
                WHERE status = 'ready' AND expires_at <= ? AND artifact IS NOT NULL
                """,
                (moment,),
            ).fetchall()
        removed = 0
        for row in rows:
            path = Path(str(row["artifact"]))
            if path.parent == self.files and path.is_file():
                path.unlink()
                removed += 1
        return removed

    def _build(self, job: Job) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT layer FROM exports WHERE job_id = ?", (job.job_id,)
            ).fetchone()
        if row is None:  # pragma: no cover - claimed row cannot disappear
            raise RuntimeError(f"export {job.job_id}: queue row disappeared")
        layer = Path(str(row["layer"]))
        # The source chunk encoding describes the full published layer.  A
        # bbox/stride slice has different dask boundaries; carrying the old
        # encoding into ``to_zarr`` would make two tasks target one chunk.
        subset = select(layer, job.spec).drop_encoding()
        if int(subset.nbytes) + used_bytes(self.files) > self.quota_bytes:
            raise QuotaError(
                f"export needs {int(subset.nbytes)} bytes; scratch quota is {self.quota_bytes}"
            )

        self.files.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        staging = self.work / job.job_id
        archive_base = self.work / f"{job.job_id}.zarr"
        temporary_archive = Path(str(archive_base) + ".zip")
        target = self.files / f"{job.job_id}.zarr.zip"
        # These paths are private to this job.  If a lease is reclaimed after a
        # process crash, its partial work must not turn a durable job into a
        # permanent failure.
        if staging.is_dir():
            shutil.rmtree(staging)
        if temporary_archive.is_file():
            temporary_archive.unlink()
        subset.attrs.update(
            source=job.source,
            init_time=job.init_time,
            export_job_id=job.job_id,
            export_format="zarr-v3-zip",
        )
        try:
            subset.to_zarr(staging, mode="w-", zarr_format=3, consolidated=True)
            made = Path(
                shutil.make_archive(
                    str(archive_base), "zip", root_dir=staging.parent, base_dir=staging.name
                )
            )
            os.replace(made, target)
        finally:
            subset.close()
            if staging.is_dir():
                shutil.rmtree(staging)
            if temporary_archive.is_file():
                temporary_archive.unlink()
        return target

    def _connect(self) -> sqlite3.Connection:
        return open_queue(self.database)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.process_once() is None:
                self._wake.wait(POLL_SEC)
                self._wake.clear()


class ExpiredError(LookupError):
    pass


class NotReadyError(LookupError):
    pass


class QuotaError(RuntimeError):
    pass


def open_queue(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS exports (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'ready', 'failed')),
            source TEXT NOT NULL,
            init_time TEXT NOT NULL,
            layer TEXT NOT NULL,
            spec TEXT NOT NULL,
            owner TEXT,
            lease_until TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT,
            artifact TEXT,
            error TEXT,
            CHECK (
                (status = 'running' AND owner IS NOT NULL AND lease_until IS NOT NULL)
                OR (status != 'running' AND owner IS NULL AND lease_until IS NULL)
            )
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS exports_claim ON exports(status, lease_until, created_at)"
    )
    return connection


def claim(
    connection: sqlite3.Connection,
    owner: str,
    *,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_LEASE_SEC,
) -> Job | None:
    moment = _utc(now)
    stamp = _stamp(moment)
    lease = _stamp(moment + timedelta(seconds=lease_seconds))
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """
            SELECT job_id FROM exports
            WHERE status = 'queued' OR (status = 'running' AND lease_until <= ?)
            ORDER BY created_at, job_id LIMIT 1
            """,
            (stamp,),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        job_id = str(row["job_id"])
        connection.execute(
            """
            UPDATE exports SET status = 'running', owner = ?, lease_until = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (owner, lease, stamp, job_id),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return get(connection, job_id)


def finish_ready(
    connection: sqlite3.Connection,
    job_id: str,
    owner: str,
    artifact: Path,
    *,
    expires_at: datetime,
    now: datetime | None = None,
) -> Job:
    changed = connection.execute(
        """
        UPDATE exports
        SET status = 'ready', owner = NULL, lease_until = NULL,
            artifact = ?, expires_at = ?, error = NULL, updated_at = ?
        WHERE job_id = ? AND status = 'running' AND owner = ?
        """,
        (str(artifact), _stamp(expires_at), _stamp(now), job_id, owner),
    ).rowcount
    if changed != 1:
        raise LookupError(f"export {job_id}: owner {owner!r} lost its lease")
    return get(connection, job_id)


def finish_failed(
    connection: sqlite3.Connection,
    job_id: str,
    owner: str,
    error: str,
    *,
    now: datetime | None = None,
) -> Job:
    changed = connection.execute(
        """
        UPDATE exports
        SET status = 'failed', owner = NULL, lease_until = NULL,
            error = ?, updated_at = ?
        WHERE job_id = ? AND status = 'running' AND owner = ?
        """,
        (error or "export failed", _stamp(now), job_id, owner),
    ).rowcount
    if changed != 1:
        raise LookupError(f"export {job_id}: owner {owner!r} lost its lease")
    return get(connection, job_id)


def get(connection: sqlite3.Connection, job_id: str) -> Job:
    row = connection.execute("SELECT * FROM exports WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise LookupError(f"export {job_id}: not found")
    raw = json.loads(str(row["spec"]))
    spec = Spec(
        tuple(str(name) for name in raw["variables"]),
        tuple(float(value) for value in raw["bbox"]),  # type: ignore[arg-type]
        str(raw["start"]),
        str(raw["stop"]),
        int(raw["stride"]),
        str(raw["format"]),
    )
    return Job(
        str(row["job_id"]),
        str(row["status"]),
        str(row["source"]),
        str(row["init_time"]),
        spec,
        str(row["created_at"]),
        str(row["updated_at"]),
        None if row["expires_at"] is None else str(row["expires_at"]),
        None if row["artifact"] is None else str(row["artifact"]),
        None if row["error"] is None else str(row["error"]),
    )


def validate_spec(layer: str | Path, spec: Spec) -> None:
    if spec.format != "zarr":
        raise ValueError(f"format: got {spec.format!r}, expected 'zarr'")
    if not spec.variables:
        raise ValueError("vars: expected at least one canonical variable")
    unknown = [name for name in spec.variables if name not in canon.UNITS]
    if unknown:
        raise ValueError(f"vars: unknown canonical variables: {', '.join(unknown)}")
    if len(set(spec.variables)) != len(spec.variables):
        raise ValueError("vars: duplicates are not allowed")
    if spec.stride < 1:
        raise ValueError(f"stride: got {spec.stride}, expected >= 1")
    south, west, north, east = spec.bbox
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("bbox: expected south,west,north,east inside the global grid")
    first, last = _time(spec.start), _time(spec.stop)
    if first > last:
        raise ValueError("from: must be <= to")
    with xr.open_zarr(layer, chunks=None) as dataset:
        missing = [name for name in spec.variables if name not in dataset]
        if missing:
            raise ValueError(f"vars: absent from forecast layer: {', '.join(missing)}")
        times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        if first < times[0] or last > times[-1]:
            raise ValueError(f"time: outside forecast coverage {times[0]}..{times[-1]}")


def select(layer: str | Path, spec: Spec) -> xr.Dataset:
    dataset = xr.open_zarr(layer, chunks={})
    south, west, north, east = spec.bbox
    return cast(
        xr.Dataset,
        dataset[list(spec.variables)]
        .sel(
            time=slice(_time(spec.start), _time(spec.stop)),
            lat=slice(north, south),
            lon=slice(west, east),
        )
        .isel(lat=slice(None, None, spec.stride), lon=slice(None, None, spec.stride)),
    )


def used_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.glob("*.zip") if path.is_file())


def is_expired(job: Job, *, now: datetime | None = None) -> bool:
    return job.expires_at is not None and job.expires_at <= _stamp(now)


def public(job: Job) -> Mapping[str, object]:
    result: dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "source": job.source,
        "init_time": job.init_time,
    }
    if job.status == READY:
        result.update(
            url=f"/v1/export/{job.job_id}/download",
            expires_at=job.expires_at,
        )
    elif job.status == FAILED:
        result["error"] = job.error
    return result


def _time(value: str) -> np.datetime64:
    try:
        return np.datetime64(value.replace("Z", ""), "ns")
    except ValueError as error:
        raise ValueError(f"time: invalid ISO 8601 {value!r}") from error


def _utc(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(moment: datetime | None) -> str:
    return _utc(moment).strftime(TIME_FORMAT)

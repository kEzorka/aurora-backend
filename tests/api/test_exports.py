"""Lane 3 acceptance: durable jobs, provenance, expiry and downloads."""

from __future__ import annotations

import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api import exports
from api.app import create_app

REQUEST = {
    "vars": ["2t", "msl"],
    "bbox": [-18.0, -90.0, 54.0, 45.0],
    "from": "2026-08-01T00:00:00Z",
    "to": "2026-08-01T12:00:00Z",
    "stride": 2,
    "format": "zarr",
}


def _spec() -> exports.Spec:
    return exports.Spec(
        ("2t", "msl"),
        (-18.0, -90.0, 54.0, 45.0),
        "2026-08-01T00:00:00Z",
        "2026-08-01T12:00:00Z",
        2,
    )


def _layer(root: Path) -> Path:
    return (root / "forecast" / "current" / "coarse").resolve()


def test_job_survives_manager_restart_and_archive_has_provenance(
    published: Path, tmp_path: Path
) -> None:
    first = exports.ExportManager(published)
    submitted = first.submit(
        _layer(published),
        _spec(),
        source="aurora-forecast",
        init_time="2026-08-01T00:00:00Z",
    )
    assert submitted.status == "queued"

    # A fresh object has no shared memory with the submitter; SQLite is the
    # only way it can discover and finish the task.
    restarted = exports.ExportManager(published)
    ready = restarted.process_once()

    assert ready is not None
    assert ready.job_id == submitted.job_id
    assert ready.status == "ready"
    archive = restarted.artifact(submitted.job_id)
    assert archive.name.endswith(".zarr.zip")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(tmp_path / "unpacked")
    with xr.open_zarr(tmp_path / "unpacked" / submitted.job_id, chunks=None) as dataset:
        assert tuple(dataset.data_vars) == ("2t", "msl")
        assert dataset.sizes == {"time": 3, "lat": 2, "lon": 2}
        assert dataset.attrs["source"] == "aurora-forecast"
        assert dataset.attrs["init_time"] == "2026-08-01T00:00:00Z"
        assert dataset.attrs["export_job_id"] == submitted.job_id


def test_running_lease_is_recovered_after_expiry(published: Path) -> None:
    moment = datetime(2026, 8, 3, tzinfo=UTC)
    manager = exports.ExportManager(published, lease_seconds=10)
    submitted = manager.submit(
        _layer(published),
        _spec(),
        source="aurora-forecast",
        init_time="2026-08-01T00:00:00Z",
        now=moment,
    )
    with exports.open_queue(manager.database) as connection:
        claimed = exports.claim(connection, "dead-worker", now=moment, lease_seconds=10)
    assert claimed is not None and claimed.status == "running"

    assert manager.process_once(now=moment + timedelta(seconds=9)) is None
    recovered = manager.process_once(now=moment + timedelta(seconds=10))

    assert recovered is not None
    assert recovered.job_id == submitted.job_id
    assert recovered.status == "ready"


def test_expired_archive_is_gone_not_missing(published: Path) -> None:
    moment = datetime(2026, 8, 3, tzinfo=UTC)
    manager = exports.ExportManager(published, ttl_seconds=10)
    submitted = manager.submit(
        _layer(published),
        _spec(),
        source="aurora-forecast",
        init_time="2026-08-01T00:00:00Z",
        now=moment,
    )
    ready = manager.process_once(now=moment)
    assert ready is not None and ready.artifact is not None

    with pytest.raises(exports.ExpiredError):
        manager.lookup(submitted.job_id, now=moment + timedelta(seconds=10))
    assert manager.cleanup(now=moment + timedelta(seconds=10)) == 1
    assert not Path(ready.artifact).exists()


def test_post_poll_and_download(published: Path) -> None:
    with TestClient(create_app(published)) as client:
        response = client.post("/v1/export", json=REQUEST)
        assert response.status_code == 202
        queued = response.json()
        assert queued["status"] == "queued"
        assert queued["job_id"].startswith("exp_")
        assert queued["poll"] == f"/v1/export/{queued['job_id']}"

        status = {}
        for _ in range(100):
            status_response = client.get(queued["poll"])
            assert status_response.status_code == 200
            status = status_response.json()
            if status["status"] in ("ready", "failed"):
                break
            time.sleep(0.01)

        assert status["status"] == "ready", status
        assert status["source"] == "aurora-forecast"
        assert status["init_time"] == "2026-08-01T00:00:00Z"
        assert status["expires_at"].endswith("Z")
        download = client.get(status["url"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        assert download.content.startswith(b"PK")


@pytest.mark.parametrize(
    "change",
    [
        {"format": "netcdf"},
        {"vars": ["not-a-field"]},
        {"stride": 0},
        {"to": "2026-09-01T00:00:00Z"},
    ],
)
def test_invalid_export_is_rejected_before_queueing(
    published: Path, change: dict[str, object]
) -> None:
    body = {**REQUEST, **change}
    with TestClient(create_app(published)) as client:
        response = client.post("/v1/export", json=body)

    assert response.status_code == 400
    assert response.json()["error"] == "bad_export"


def test_unknown_export_is_not_found(published: Path) -> None:
    with TestClient(create_app(published)) as client:
        response = client.get("/v1/export/exp_missing")

    assert response.status_code == 404
    assert response.json()["error"] == "export_not_found"


def test_missing_ready_file_is_gone(published: Path) -> None:
    manager = exports.ExportManager(published)
    submitted = manager.submit(
        _layer(published),
        _spec(),
        source="aurora-forecast",
        init_time="2026-08-01T00:00:00Z",
    )
    ready = manager.process_once()
    assert ready is not None and ready.artifact is not None
    Path(ready.artifact).unlink()

    with TestClient(create_app(published)) as client:
        response = client.get(f"/v1/export/{submitted.job_id}")

    assert response.status_code == 410
    assert response.json()["error"] == "export_expired"

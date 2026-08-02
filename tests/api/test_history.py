"""`GET /v1/history/*` — приёмка BACKLOG 5.5.

Критерий — холодный запрос работает, а `cache.hit` честный. Поэтому основной
тест повторяет один вопрос и считает обращения к подставной очереди CDS.
Остальные закрепляют границы контракта до похода наружу: лимиты, агрегацию,
коды отказов и предварительный ERA5T.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api.history import COLD_LIMIT, History
from cache.origins import CDS_GRID, ArcoOrigin, CdsOrigin
from cache.proxy import Grid
from storage.monthly import MONTHLY_VARS
from storage.monthly import publish as publish_monthly
from tests.fixtures.arco import STEPS, build
from tests.fixtures.cds import MOSCOW, Service

POINT = "/v1/history/point"
GRID = "/v1/history/grid"
FIRST = datetime(1990, 5, 1, tzinfo=UTC)
NOW = datetime(2026, 3, 1, tzinfo=UTC)
DAY_GRID = Grid(epoch=CDS_GRID.epoch, step=timedelta(hours=1), span=24, max_chunks=40)


def _state(tmp_path: Path, service: Service, *, now: datetime = NOW) -> History:
    origin = CdsOrigin(grid=DAY_GRID, retriever=service, clock=lambda: now)
    return History(
        root=tmp_path / "cache",
        index_path=tmp_path / "index.sqlite",
        maps=origin,
        series=origin,
    )


def _client(tmp_path: Path, state: History) -> TestClient:
    from api.app import create_app

    return TestClient(create_app(tmp_path, history=state))


def _point_params(**extra: object) -> dict[str, object]:
    return {
        "lat": MOSCOW[0],
        "lon": MOSCOW[1],
        "vars": "t2m",
        "from": "1990-05-01T00:00:00Z",
        "to": "1990-05-01T23:00:00Z",
        **extra,
    }


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("history-api-arco") / "sfc.zarr", "2m_temperature")


def test_a_cold_point_request_becomes_a_real_warm_hit(tmp_path: Path) -> None:
    service = Service()
    client = _client(tmp_path, _state(tmp_path, service))

    cold = client.get(POINT, params=_point_params())
    warm = client.get(POINT, params=_point_params())

    assert cold.status_code == warm.status_code == 200
    assert cold.json()["cache"]["hit"] is False
    assert warm.json()["cache"] == {"hit": True, "origin_latency_ms": 0}
    assert len(service.requests) == 1
    body = warm.json()
    assert body["query"]["nearest_grid"] == {"lat": MOSCOW[0], "lon": MOSCOW[1]}
    assert body["source"] == "era5-final"
    assert body["agg"] == "raw6h"
    assert body["units"] == {"t2m": "degC"}
    assert body["times"] == [
        "1990-05-01T00:00:00Z",
        "1990-05-01T06:00:00Z",
        "1990-05-01T12:00:00Z",
        "1990-05-01T18:00:00Z",
    ]
    assert body["series"]["t2m"] == [-23.1, -17.1, -11.1, -5.1]


def test_daily_aggregation_has_mean_min_and_max(tmp_path: Path) -> None:
    response = _client(tmp_path, _state(tmp_path, Service())).get(
        POINT, params=_point_params(agg="daily")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["times"] == ["1990-05-01"]
    assert body["series"] == {
        "t2m_mean": [-11.6],
        "t2m_min": [-23.1],
        "t2m_max": [-0.1],
    }


def test_raw_range_without_a_six_hour_term_is_empty_before_the_origin(tmp_path: Path) -> None:
    service = Service()
    params = _point_params(
        **{
            "from": "1990-05-01T00:30:00Z",
            "to": "1990-05-01T00:45:00Z",
        }
    )

    response = _client(tmp_path, _state(tmp_path, service)).get(POINT, params=params)

    assert response.status_code == 404
    assert service.requests == []


def test_preliminary_history_says_that_it_can_change(tmp_path: Path) -> None:
    now = FIRST + timedelta(days=30)
    response = _client(tmp_path, _state(tmp_path, Service(), now=now)).get(
        POINT, params=_point_params()
    )

    assert response.status_code == 200
    assert response.json()["source"] == "era5t"
    assert response.json()["preliminary"] is True
    assert response.headers["Cache-Control"] == "public, max-age=3600"


def test_history_response_has_a_working_etag(tmp_path: Path) -> None:
    client = _client(tmp_path, _state(tmp_path, Service()))
    client.get(POINT, params=_point_params())  # холодное тело честно отличается `cache.hit`
    warm = client.get(POINT, params=_point_params())

    repeated = client.get(
        POINT, params=_point_params(), headers={"If-None-Match": warm.headers["ETag"]}
    )

    assert repeated.status_code == 304 and repeated.content == b""
    assert repeated.headers["Cache-Control"] == "public, max-age=86400"


def test_a_grid_uses_the_same_compact_geometry_as_the_forecast(
    archive: Path, tmp_path: Path
) -> None:
    origin = ArcoOrigin(archive, clock=lambda: datetime(2020, 12, 1, tzinfo=UTC))
    state = History(tmp_path / "cache", tmp_path / "index.sqlite", origin, origin)

    response = _client(tmp_path, state).get(
        GRID,
        params={
            "bbox": "50,0,51,2",
            "var": "t2m",
            "time": "2020-06-01T01:00:00Z",
            "stride": 2,
            "units": "si",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["time"] == "2020-06-01T00:00:00Z"  # raw6h округлён и показан
    assert body["grid"] == {
        "lat0": 51.0,
        "lon0": 0.0,
        "dlat": -0.5,
        "dlon": 0.5,
        "shape": [3, 5],
        "order": "row-major",
    }
    assert len(body["values"]) == 15
    assert body["cache"]["hit"] is False


def test_a_published_monthly_map_is_served_without_the_origin(tmp_path: Path) -> None:
    times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]")
    lat = np.array([1.0, 0.0, -1.0])
    lon = np.array([-1.0, 0.0, 1.0, 2.0])
    ds = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.full((2, 3, 4), index + 1.0, dtype=np.float32),
            )
            for index, name in enumerate(MONTHLY_VARS)
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    publish_monthly(ds, tmp_path, version="2020-02", require_canonical_grid=False)
    service = Service()

    response = _client(tmp_path, _state(tmp_path, service)).get(
        GRID,
        params={
            "bbox": "-1,-1,1,2",
            "var": "t2m",
            "time": "2020-02-15T12:00:00Z",
            "agg": "monthly",
            "units": "si",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "era5-final"
    assert body["agg"] == "monthly"
    assert body["time"] == "2020-02-01T00:00:00Z"
    assert body["cache"] == {"hit": True, "origin_latency_ms": 0}
    assert body["grid"]["shape"] == [3, 4]
    assert body["values"] == [1.0] * 12
    assert service.requests == []


@pytest.mark.parametrize(
    ("path", "params", "status", "error"),
    [
        (POINT, _point_params(**{"from": "1939-12-31T00:00:00Z"}), 404, "out_of_coverage"),
        (POINT, _point_params(to="1990-09-30T00:00:00Z"), 413, "too_many_steps"),
        (POINT, _point_params(agg="hourly"), 400, "bad_aggregation"),
        (
            GRID,
            {"bbox": "50,0,51,2", "var": "t2m", "time": STEPS[0].isoformat(), "agg": "monthly"},
            400,
            "bad_aggregation",
        ),
        (
            GRID,
            {"bbox": "-90,-180,90,180", "var": "t2m", "time": STEPS[0].isoformat()},
            413,
            "too_many_points",
        ),
    ],
)
def test_invalid_requests_are_refused_before_the_origin(
    tmp_path: Path,
    path: str,
    params: Mapping[str, object],
    status: int,
    error: str,
) -> None:
    service = Service()

    response = _client(tmp_path, _state(tmp_path, service)).get(path, params=params)

    assert response.status_code == status
    assert response.json()["error"] == error
    assert service.requests == []


class BrokenService:
    def __call__(self, dataset: str, request: Mapping[str, Any]) -> str:
        raise RuntimeError("CDS queue is unavailable")


def test_an_origin_failure_is_503_with_retry_after(tmp_path: Path) -> None:
    broken = CdsOrigin(grid=DAY_GRID, retriever=BrokenService(), clock=lambda: NOW)
    state = History(tmp_path / "cache", tmp_path / "index.sqlite", broken, broken)

    response = _client(tmp_path, state).get(POINT, params=_point_params())

    assert response.status_code == 503
    assert response.json()["error"] == "origin_unavailable"
    assert response.headers["Retry-After"] == "300"


def test_a_future_gap_is_404_and_not_503(tmp_path: Path) -> None:
    now = FIRST - timedelta(days=1)

    response = _client(tmp_path, _state(tmp_path, Service(), now=now)).get(
        POINT, params=_point_params()
    )

    assert response.status_code == 404
    assert response.json()["error"] == "out_of_coverage"


def test_the_fifth_cold_request_is_429_without_touching_the_origin(tmp_path: Path) -> None:
    service = Service()
    state = _state(tmp_path, service)
    client = _client(tmp_path, state)

    with ExitStack() as occupied:
        for _ in range(COLD_LIMIT):
            occupied.enter_context(state.gate.hold(True))
        response = client.get(POINT, params=_point_params())

    assert response.status_code == 429
    assert response.json()["error"] == "too_many_cold"
    assert response.headers["Retry-After"] == "30"
    assert service.requests == []

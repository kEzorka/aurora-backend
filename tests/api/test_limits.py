"""Лимиты объёма и коды отказов — приёмка BACKLOG 5.3.

Приёмка: «`413` с исполнимым `suggested_stride`; перепутанный `bbox` даёт
`400`». Вторая половина закрыта в 5.2 (`test_forecast_grid.py`), здесь —
первая: подсказка проверяется не глазами, а повтором запроса с тем самым
числом, которое сервис назвал.

Потолок виден только на настоящей сетке 0.25°: на маленькой фикстуре весь
глобус — это 48 точек, и любой потолок в неё влезает.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api.app import create_app
from contracts import canon
from storage.read import MAX_POINTS, MAX_STEPS
from storage.write import write_layer

GRID = "/v1/forecast/grid"
POINT = "/v1/forecast/point"
NOON = "2026-08-01T00:00:00Z"
#: Весь глобус: 721 × 1440 = 1 038 240 точек, двадцатикратный потолок.
WHOLE_WORLD = "-90,-180,90,180"


def _store(
    root: Path, steps: int, shape: tuple[int, int], lat: np.ndarray, lon: np.ndarray
) -> Path:
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), np.full((steps, *shape), 288.15, dtype=np.float32))},
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)],
                dtype="datetime64[ns]",
            ),
            "lat": lat,
            "lon": lon,
        },
        attrs={"init_time": NOON},
    )
    ds["2t"].attrs["units"] = canon.UNITS["2t"]
    run = root / "runs" / "r"
    write_layer(ds, run / "coarse", canon.Layer("coarse", ("2t",), (), canon.STEP_HOURS, steps))
    (run / "manifest.json").write_text('{"published": true}', encoding="utf-8")
    (root / "forecast").mkdir()
    (root / "forecast" / "current").symlink_to("../runs/r")
    return root


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Один срок на настоящей сетке — 4.15 МБ, и хватает на весь модуль."""
    root = _store(tmp_path_factory.mktemp("world"), 1, canon.GRID_SHAPE, canon.LAT, canon.LON)
    return TestClient(create_app(root))


def test_the_whole_world_is_four_hundred_thirteen(world: TestClient) -> None:
    """Потолок — защита не от злоумышленника, а от опечатки во фронтенде:
    без него первый же `bbox` на весь глобус собирает миллион чисел."""
    response = world.get(GRID, params={"bbox": WHOLE_WORLD, "var": "t2m", "time": NOON})

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "too_many_points"
    assert body["requested"] == 721 * 1440
    assert body["limit"] == MAX_POINTS == 50_000
    assert body["suggested_stride"] == 5
    assert "stride >= 5" in body["hint"]


def test_the_suggested_stride_actually_works(world: TestClient) -> None:
    """Главное в приёмке 5.3: подсказка исполнима. Число из отказа подставляется
    в тот же запрос и обязано дать `200`, а не второй `413`."""
    refused = world.get(GRID, params={"bbox": WHOLE_WORLD, "var": "t2m", "time": NOON}).json()
    stride = refused["suggested_stride"]

    response = world.get(
        GRID, params={"bbox": WHOLE_WORLD, "var": "t2m", "time": NOON, "stride": stride}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["values"]) == body["grid"]["shape"][0] * body["grid"]["shape"][1]
    assert len(body["values"]) <= MAX_POINTS


def test_the_suggested_stride_is_the_smallest_that_works(world: TestClient) -> None:
    """Подсказка не «какое-нибудь большое число»: на шаг мельче запрос обязан
    остаться отказом, иначе сервис отнимает у пользователя разрешение зря."""
    response = world.get(
        GRID, params={"bbox": WHOLE_WORLD, "var": "t2m", "time": NOON, "stride": 4}
    )

    assert response.status_code == 413
    # Подсказка не зависит от того, с каким `stride` пришёл отказ: она считается
    # от неразреженного окна, и пользователь подставляет её вместо своего числа.
    assert response.json()["suggested_stride"] == 5


def test_a_window_under_the_ceiling_is_untouched(world: TestClient) -> None:
    """Потолок не должен прореживать молча: окно, которое влезает, отдаётся
    как есть, и `stride` в ответе остаётся тем, что просили."""
    response = world.get(GRID, params={"bbox": "30,10,70,35", "var": "t2m", "time": NOON})

    assert response.status_code == 200
    assert response.json()["query"]["stride"] == 1


def test_a_point_series_past_the_step_ceiling_is_four_hundred_thirteen(tmp_path: Path) -> None:
    """Второй потолок из §3 — шаги: 501 срок в одном ответе это `413`,
    а не молчаливое обрезание ряда."""
    root = _store(
        tmp_path,
        MAX_STEPS + 1,
        (3, 4),
        np.linspace(90.0, -90.0, 3),
        np.linspace(-180.0, 90.0, 4),
    )
    response = TestClient(create_app(root)).get(
        POINT, params={"lat": 0.0, "lon": 0.0, "vars": "t2m"}
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "too_many_steps"
    assert (body["requested"], body["limit"]) == (MAX_STEPS + 1, MAX_STEPS)

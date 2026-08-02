"""`GET /v1/forecast/point` — приёмка BACKLOG 5.1.

Проверяется ответ по контракту (docs/API_CONTRACT.md §2): состав полей,
человеческие единицы по умолчанию, узел сетки и отказы вместо тихих подмен.
Время ответа — отдельным медленным тестом на настоящей сетке.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from contracts import canon

POINT = "/v1/forecast/point"
MOSCOW = {"lat": 55.75, "lon": 37.62}


def test_response_carries_everything_the_contract_promises(client: TestClient) -> None:
    """Приёмка 5.1: ответ по контракту, с `source` и `init_time`."""
    response = client.get(POINT, params=MOSCOW)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"query", "source", "init_time", "step_hours", "units", "times", "series"}
    assert body["source"] == "aurora-forecast"
    assert body["init_time"] == "2026-08-01T00:00:00Z"
    assert body["step_hours"] == canon.STEP_HOURS
    assert body["times"] == [
        "2026-08-01T00:00:00Z",
        "2026-08-01T06:00:00Z",
        "2026-08-01T12:00:00Z",
    ]
    assert set(body["series"]) == {"t2m", "wind_speed", "msl"}


def test_the_user_sees_which_grid_node_answered(client: TestClient) -> None:
    """Узел сетки 0.25° — это ≈25 км, и пользователь обязан видеть, что попал
    в него, а не в свой двор (docs/API_CONTRACT.md §2)."""
    query = client.get(POINT, params=MOSCOW).json()["query"]

    assert query == {"lat": 55.75, "lon": 37.62, "nearest_grid": {"lat": 54.0, "lon": 45.0}}


def test_human_units_are_the_default(client: TestClient) -> None:
    """Ответ по умолчанию — °C, гПа и м/с, и единицы в нём названы: даже
    очевидные, особенно очевидные (docs/API_CONTRACT.md §5.4)."""
    body = client.get(POINT, params=MOSCOW).json()

    assert body["units"] == {"t2m": "degC", "wind_speed": "m/s", "msl": "hPa"}
    assert body["series"]["t2m"] == pytest.approx([15.0, 15.0, 15.0], abs=1e-4)
    assert body["series"]["msl"] == pytest.approx([1013.25] * 3, abs=1e-4)


def test_si_returns_what_lies_on_disk(client: TestClient) -> None:
    body = client.get(POINT, params={**MOSCOW, "units": "si"}).json()

    assert body["units"] == {"t2m": "K", "wind_speed": "m s-1", "msl": "Pa"}
    assert body["series"]["t2m"] == pytest.approx([288.15] * 3, abs=1e-4)
    assert body["series"]["msl"] == pytest.approx([101_325.0] * 3, abs=0.5)


def test_wind_is_a_speed_and_not_two_components(client: TestClient) -> None:
    """Скорости ветра в хранилище нет: она считается из `10u` и `10v` здесь,
    и ключ ответа — `wind_speed` (docs/API_CONTRACT.md §2)."""
    body = client.get(POINT, params={**MOSCOW, "vars": "wind"}).json()

    assert list(body["series"]) == ["wind_speed"]
    assert body["series"]["wind_speed"] == pytest.approx([5.0] * 3, abs=1e-4)


def test_hourly_step_serves_the_hourly_layer(client: TestClient) -> None:
    """Часовой шаг — настоящий выход модели, а не интерполяция шестичасового
    (docs/API_CONTRACT.md §5.2): и сроки, и значения приходят из другого слоя."""
    body = client.get(POINT, params={**MOSCOW, "vars": "t2m", "step_hours": 1}).json()

    assert body["step_hours"] == 1
    assert body["times"][:2] == ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"]
    assert body["series"]["t2m"] == pytest.approx([16.0] * 4, abs=1e-4)


def test_hourly_step_beyond_the_horizon_is_refused(client: TestClient) -> None:
    """Дальше 72 ч часового слоя нет. Молчаливое округление до шести часов
    дало бы пользователю не тот ряд, который он просил."""
    response = client.get(POINT, params={**MOSCOW, "step_hours": 1, "to": "2026-08-05T00:00:00Z"})

    assert response.status_code == 400
    assert response.json()["error"] == "hourly_horizon"


def test_hourly_step_for_a_variable_outside_the_eight_is_refused(client: TestClient) -> None:
    response = client.get(POINT, params={**MOSCOW, "vars": "sp", "step_hours": 1})

    assert response.status_code == 400
    assert "step_hours=1" in response.json()["detail"]


def test_an_unknown_variable_is_four_hundred(client: TestClient) -> None:
    response = client.get(POINT, params={**MOSCOW, "vars": "temperature"})

    assert response.status_code == 400
    assert "temperature" in response.json()["detail"]


def test_a_reversed_range_is_four_hundred(client: TestClient) -> None:
    response = client.get(
        POINT, params={**MOSCOW, "from": "2026-08-02T00:00:00Z", "to": "2026-08-01T00:00:00Z"}
    )

    assert response.status_code == 400


def test_a_date_outside_the_forecast_is_four_hundred_four(client: TestClient) -> None:
    """Будущее дальше горизонта — не пустой ряд, а `404`
    (docs/API_CONTRACT.md §4)."""
    response = client.get(POINT, params={**MOSCOW, "from": "2026-09-01T00:00:00Z"})

    assert response.status_code == 404
    assert response.json()["error"] == "out_of_coverage"


def test_a_point_off_the_globe_is_four_hundred(client: TestClient) -> None:
    """Коды ответов контракт перечисляет (§4), и `422` в перечне нет: своя
    проверка вместо `Query(ge=..., le=...)`."""
    response = client.get(POINT, params={"lat": 91.0, "lon": 0.0})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_point"


def test_an_empty_store_answers_five_hundred_three(tmp_path: Path) -> None:
    """Прогона нет — виноват сервис, а не запрос пользователя."""
    response = TestClient(create_app(tmp_path)).get(POINT, params=MOSCOW)

    assert response.status_code == 503
    assert response.json()["error"] == "no_forecast"


def test_an_unpublished_run_is_invisible(published: Path) -> None:
    """Указатель переставлен, но `published` не поднят — читатель этот прогон
    видеть не должен (docs/STORAGE.md §5)."""
    from storage.manifest import read_manifest, write_manifest

    run = (published / "forecast" / "current").resolve()
    write_manifest(
        run / "manifest.json", {**read_manifest(run / "manifest.json"), "published": False}
    )

    assert TestClient(create_app(published)).get(POINT, params=MOSCOW).status_code == 503


def test_the_answer_is_cacheable_until_the_next_run(client: TestClient) -> None:
    """Прогноз меняется четыре раза в сутки — ответ обязан нести TTL
    (docs/API_CONTRACT.md §5.5)."""
    header = client.get(POINT, params=MOSCOW).headers["Cache-Control"]

    assert "max-age=" in header


@pytest.mark.slow
def test_a_point_answers_in_under_200_ms(tmp_path: Path) -> None:
    """Приёмка 5.1: `< 200 мс`. Настоящая сетка и все сорок сроков: цена
    раскладки видна только на 721 × 1440, и раскладка карт эту приёмку не
    проходит — 300 мс против 4 (docs/STORAGE.md §3)."""
    import numpy as np
    import xarray as xr

    from storage.write import write_layer

    ny, nx = canon.GRID_SHAPE
    steps = canon.FORECAST_STEPS
    names = ("2t", "10u", "10v", "msl")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                rng.normal(288.0, 10.0, (steps, ny, nx)).astype(np.float32),
            )
            for name in names
        },
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)],
                dtype="datetime64[ns]",
            ),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    for name in names:
        ds[name].attrs["units"] = canon.UNITS[name]
    run = tmp_path / "runs" / "r"
    write_layer(ds, run / "coarse", canon.Layer("coarse", names, (), canon.STEP_HOURS, steps))
    # То же, что делает публикация: восемь ходовых переменных рядами.
    write_layer(ds, run / "points", canon.Layer("points", names, (), canon.STEP_HOURS, steps))
    (run / "manifest.json").write_text('{"published": true}', encoding="utf-8")
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "current").symlink_to("../runs/r")

    client = TestClient(create_app(tmp_path))
    client.get(POINT, params=MOSCOW)  # прогрев кэша страниц
    started = time.perf_counter()
    response = client.get(POINT, params=MOSCOW)
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 0.2, f"точка отвечала {elapsed * 1000:.0f} мс"

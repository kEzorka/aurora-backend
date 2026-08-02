"""`GET /v1/forecast/grid` — приёмка BACKLOG 5.2.

Проверяется компактный формат (docs/API_CONTRACT.md §1): геометрия описана
началом и шагом, значения лежат одним плоским рядом, и 16 000 точек весят
килобайты, а не сотни килобайт. Отдельно — что округления (срок к шести
часам, окно к узлам сетки) видны в ответе, а не происходят молча.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api.app import create_app
from contracts import canon
from storage.write import write_layer

GRID = "/v1/forecast/grid"
#: south,west,north,east по узлам фикстуры: широты 54, 18, -18; долготы -90…45.
BOX = "-18,-90,54,45"
NOON = "2026-08-01T06:00:00Z"


def test_response_carries_everything_the_contract_promises(client: TestClient) -> None:
    response = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": NOON})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"query", "source", "init_time", "time", "units", "grid", "values"}
    assert body["source"] == "aurora-forecast"
    assert body["init_time"] == "2026-08-01T00:00:00Z"
    assert body["units"] == {"t2m": "degC"}
    assert body["grid"]["order"] == "row-major"


def test_the_geometry_is_a_start_and_a_step_not_a_list_of_coordinates(client: TestClient) -> None:
    """Компактный формат (§1): клиент восстанавливает координату как
    `lat0 + dlat * i`. Начало — первый **выбранный** узел, а не угол `bbox`."""
    grid = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": NOON}).json()["grid"]

    assert grid["shape"] == [3, 4]
    assert (grid["lat0"], grid["dlat"]) == (54.0, -36.0)
    assert (grid["lon0"], grid["dlon"]) == (-90.0, 45.0)
    # Последний узел окна восстанавливается из тех же четырёх чисел.
    assert grid["lat0"] + grid["dlat"] * (grid["shape"][0] - 1) == -18.0
    assert grid["lon0"] + grid["dlon"] * (grid["shape"][1] - 1) == 45.0


def test_values_are_one_flat_row_major_run(client: TestClient) -> None:
    """Приёмка 5.2: длина `values` равна произведению `shape`."""
    body = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": NOON}).json()

    assert len(body["values"]) == body["grid"]["shape"][0] * body["grid"]["shape"][1]
    assert body["values"] == [15.0] * 12


def test_stride_thins_the_window_and_says_so_in_the_step(client: TestClient) -> None:
    """Прореженная карта — это другая сетка, и `dlat`/`dlon` обязаны это
    показать: клиент, восстановивший координаты по 0.25° вместо 0.75°,
    получит сдвинутую карту и не узнает об этом."""
    grid = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": NOON, "stride": 2}).json()[
        "grid"
    ]

    assert grid["shape"] == [2, 2]
    assert (grid["lat0"], grid["dlat"]) == (54.0, -72.0)
    assert (grid["lon0"], grid["dlon"]) == (-90.0, 90.0)


def test_the_served_time_is_rounded_to_the_step_and_reported(client: TestClient) -> None:
    """Контракт округляет `time` к ближайшему шестичасовому сроку
    (docs/API_CONTRACT.md §2) — и отдаёт тот срок, который отдал."""
    body = client.get(
        GRID, params={"bbox": BOX, "var": "t2m", "time": "2026-08-01T04:00:00Z"}
    ).json()

    assert body["time"] == "2026-08-01T06:00:00Z"
    assert body["query"]["time"] == "2026-08-01T04:00:00Z"


def test_the_edge_of_the_rounding_is_inside_the_coverage(client: TestClient) -> None:
    """Граница округления — полшага за последним сроком. Слой кончается в 12:00,
    значит 15:00 ещё округляется к нему, а 15:01 уже вне покрытия. Место, на
    котором стоят лимиты 5.3, поэтому закреплено обеими сторонами."""
    inside = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": "2026-08-01T15:00:00Z"})
    outside = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": "2026-08-01T15:01:00Z"})

    assert inside.status_code == 200
    assert inside.json()["time"] == "2026-08-01T12:00:00Z"
    assert outside.status_code == 404


def test_wind_is_a_speed_here_too(client: TestClient) -> None:
    """`wind` — одна переменная ответа из двух полей на диске: ограничение
    «одна переменная на запрос» про `values`, а не про чтение."""
    body = client.get(GRID, params={"bbox": BOX, "var": "wind", "time": NOON}).json()

    assert body["units"] == {"wind_speed": "m/s"}
    assert body["values"] == [5.0] * 12


def test_si_is_rounded_by_the_unit_not_by_the_field(client: TestClient) -> None:
    """0.1 °C осмысленна, 0.1 Па — нет: знаки берутся от единицы, поэтому одно
    и то же поле в СИ округляется иначе."""
    body = client.get(GRID, params={"bbox": BOX, "var": "msl", "time": NOON, "units": "si"}).json()

    assert body["units"] == {"msl": "Pa"}
    assert body["values"] == [101_325.0] * 12


def test_a_reversed_bbox_is_four_hundred(client: TestClient) -> None:
    """`55.7,37.5,55.8,37.7` и `37.5,55.7,37.7,55.8` одинаково правдоподобны,
    но второй лежит в океане — это отказ, а не пустая карта."""
    response = client.get(GRID, params={"bbox": "54,45,-18,-90", "var": "t2m", "time": NOON})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_bbox"


def test_a_bbox_that_is_not_four_numbers_is_four_hundred(client: TestClient) -> None:
    response = client.get(GRID, params={"bbox": "55.7,37.5,55.8", "var": "t2m", "time": NOON})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_bbox"


def test_two_variables_on_a_grid_request_are_four_hundred(client: TestClient) -> None:
    response = client.get(GRID, params={"bbox": BOX, "var": "t2m,msl", "time": NOON})

    assert response.status_code == 400
    assert response.json()["error"] == "too_many_vars"


def test_a_time_outside_the_forecast_is_four_hundred_four(client: TestClient) -> None:
    """Без проверки покрытия «ближайший срок» молча отдал бы конец горизонта
    на запрос про сентябрь."""
    response = client.get(GRID, params={"bbox": BOX, "var": "t2m", "time": "2026-09-01T00:00:00Z"})

    assert response.status_code == 404
    assert response.json()["error"] == "out_of_coverage"


def test_a_bbox_between_grid_nodes_is_four_hundred_four(client: TestClient) -> None:
    """Окно уже шага сетки — это не пустой список значений и не подмена
    ближайшим узлом, а честное «данных нет»."""
    response = client.get(GRID, params={"bbox": "0.1,0.1,0.2,0.2", "var": "t2m", "time": NOON})

    assert response.status_code == 404


def test_an_empty_store_answers_five_hundred_three(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get(
        GRID, params={"bbox": BOX, "var": "t2m", "time": NOON}
    )

    assert response.status_code == 503
    assert response.json()["error"] == "no_forecast"


def test_sixteen_thousand_points_fit_in_ninety_kilobytes(tmp_path: Path) -> None:
    """Приёмка 5.2. Цену формата видно только на настоящем шаге 0.25°: наивный
    ответ парами координат стоит 45 байт на точку (docs/API_CONTRACT.md §1),
    а неокруглённый `float32` — 18 байт на одно значение.
    """
    ny, nx = canon.GRID_SHAPE
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), rng.normal(288.0, 10.0, (2, ny, nx)).astype(np.float32))},
        coords={
            "time": np.array(["2026-08-01T00", "2026-08-01T06"], dtype="datetime64[ns]"),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    ds["2t"].attrs["units"] = canon.UNITS["2t"]
    run = tmp_path / "runs" / "r"
    write_layer(ds, run / "coarse", canon.Layer("coarse", ("2t",), (), canon.STEP_HOURS, 2))
    (run / "manifest.json").write_text('{"published": true}', encoding="utf-8")
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "current").symlink_to("../runs/r")

    # 40° по широте и 25° по долготе на шаге 0.25° — 161 × 101 = 16 261 точка.
    response = TestClient(create_app(tmp_path)).get(
        GRID, params={"bbox": "30,10,70,35", "var": "t2m", "time": "2026-08-01T00:00:00Z"}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["values"]) == 16_261
    assert len(response.content) < 92_000, f"{len(response.content)} байт на 16 261 точку"


@pytest.mark.slow
def test_a_grid_answers_in_under_a_second(tmp_path: Path) -> None:
    """Карта поднимает чанк целиком — 4.15 МБ на срок в раскладке A. Приёмка
    здесь не 200 мс точки, но и не секунды: раскладка обязана оставаться той
    (docs/STORAGE.md §3)."""
    import time as clock

    ny, nx = canon.GRID_SHAPE
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), rng.normal(288.0, 10.0, (4, ny, nx)).astype(np.float32))},
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(4)],
                dtype="datetime64[ns]",
            ),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    ds["2t"].attrs["units"] = canon.UNITS["2t"]
    run = tmp_path / "runs" / "r"
    write_layer(ds, run / "coarse", canon.Layer("coarse", ("2t",), (), canon.STEP_HOURS, 4))
    (run / "manifest.json").write_text('{"published": true}', encoding="utf-8")
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "current").symlink_to("../runs/r")

    client = TestClient(create_app(tmp_path))
    params = {"bbox": "30,10,70,35", "var": "t2m", "time": "2026-08-01T12:00:00Z"}
    client.get(GRID, params=params)  # прогрев кэша страниц
    started = clock.perf_counter()
    response = client.get(GRID, params=params)
    elapsed = clock.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 1.0, f"карта отвечала {elapsed * 1000:.0f} мс"

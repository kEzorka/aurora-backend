"""Метаданные полосы 1 — приёмка BACKLOG 5.4.

Приёмка сформулирована отрицанием: «фронтенд может построить интерфейс,
не зная деталей хранилища». Значит проверяется не форма ответа, а что из
него **не** торчит: имена слоёв, канонические имена переменных, раскладки.
И обратное: всё, что торчит, должно работать — имя из покрытия обязано
приниматься в `vars`, лимиты обязаны совпадать с теми, по которым сервис
отказывает.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from contracts import canon
from storage.read import MAX_POINTS, MAX_STEPS

COVERAGE = "/v1/meta/coverage"
HEALTH = "/v1/health"

#: Слова хранилища. В теле ответа их быть не должно ни одного.
STORAGE_WORDS = ("coarse", "hourly", "points", "zarr", "chunk", "runs/")


def _strings(node: object) -> list[str]:
    """Все строковые значения ответа. Проверяются именно значения: `max_points`
    — ключ лимита, и слово «points» в нём про точки, а не про раскладку."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [text for value in node.values() for text in _strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _strings(value)]
    return []


def test_coverage_says_nothing_about_the_storage(client: TestClient) -> None:
    """Главное в приёмке 5.4. Узнав слово `coarse`, фронтенд начинает от него
    зависеть, и переименование слоя ломает интерфейс на ровном месте."""
    values = _strings(client.get(COVERAGE).json())

    for word in STORAGE_WORDS:
        assert not [text for text in values if word in text]


def test_a_layer_is_named_by_its_step_and_not_by_its_directory(client: TestClient) -> None:
    """Слой описан шагом: фронтенд выбирает `step_hours=1`, а не каталог."""
    layers = client.get(COVERAGE).json()["layers"]

    assert [layer["step_hours"] for layer in layers] == [
        canon.FINE_STEP_HOURS,
        canon.STEP_HOURS,
    ]


def test_the_bounds_come_from_the_disk_and_not_from_the_catalogue(client: TestClient) -> None:
    """Канон говорит, сколько шагов прогон **должен** был записать; покрытие —
    сколько записано. Оборванный прогон читатель обязан увидеть."""
    layers = {layer["step_hours"]: layer for layer in client.get(COVERAGE).json()["layers"]}

    hourly = layers[canon.FINE_STEP_HOURS]
    assert (hourly["from"], hourly["to"]) == ("2026-08-01T00:00:00Z", "2026-08-01T03:00:00Z")
    assert hourly["steps"] == 4
    coarse = layers[canon.STEP_HOURS]
    assert (coarse["from"], coarse["to"]) == ("2026-08-01T00:00:00Z", "2026-08-01T12:00:00Z")
    assert coarse["steps"] == 3


def test_every_offered_variable_is_actually_askable(client: TestClient) -> None:
    """Покрытие — это обещание. Имя из него подставляется в `vars` и обязано
    дать `200`, иначе фронтенд построит интерфейс с нерабочими кнопками."""
    layer = next(
        layer
        for layer in client.get(COVERAGE).json()["layers"]
        if layer["step_hours"] == canon.STEP_HOURS
    )
    names = [var["name"] for var in layer["vars"]]

    for name in names:
        response = client.get("/v1/forecast/point", params={"lat": 55.0, "lon": 37.0, "vars": name})
        assert response.status_code == 200, name

    # Одно имя на величину: `2t` сервис принял бы, но рядом с `t2m` это две
    # одинаковые температуры в списке переменных.
    assert "t2m" in names
    assert "2t" not in names
    # Скорости ветра в хранилище нет; она предлагается, потому что лежат обе
    # составляющие, и без неё виджет считал бы гипотенузу сам.
    assert "wind" in names


def test_units_come_with_the_names(client: TestClient) -> None:
    """«Единицы всегда в ответе, даже если очевидны» (§5.4): без них ось
    подписывать нечем, а °C и K различаются на 273 градуса."""
    layer = client.get(COVERAGE).json()["layers"][0]
    units = {var["name"]: var["units"] for var in layer["vars"]}

    assert units["t2m"] == {"si": "K", "human": "degC"}
    assert units["msl"] == {"si": "Pa", "human": "hPa"}


def test_the_limits_are_the_ones_the_service_enforces(client: TestClient) -> None:
    """Лимиты в покрытии затем, чтобы `stride` считался до запроса, а не
    узнавался из `413` (§3). Разойтись с настоящими им нельзя."""
    limits = client.get(COVERAGE).json()["limits"]

    assert limits["max_points"] == MAX_POINTS
    assert limits["max_steps"] == MAX_STEPS
    assert limits["vars_per_grid"] == 1


def test_the_age_is_a_number_and_not_a_verdict(client: TestClient) -> None:
    """Порога устаревания контракт не задаёт, и выдумывать его — решать за
    того, кто решения не принимал. Отдаётся возраст и когда ждать следующий."""
    body = client.get(COVERAGE).json()

    assert body["init_time"] == "2026-08-01T00:00:00Z"
    assert body["expected_next"] == "2026-08-01T06:00:00Z"
    assert isinstance(body["age_hours"], float)


def test_health_reports_the_filesystem_and_not_the_budget(client: TestClient) -> None:
    """40 ГБ ядра и 195 ГБ кэша из docs/STORAGE.md §3 — это план, а не разделы:
    пока ротация их не выдерживает, «кэш заполнен на 42 %» было бы выдумкой."""
    body = client.get(HEALTH).json()

    assert body["status"] == "ok"
    assert body["storage"]["readable"] is True
    assert body["storage"]["free_bytes"] > 0
    assert body["storage"]["free_bytes"] <= body["storage"]["total_bytes"]
    assert body["forecast"]["init_time"] == "2026-08-01T00:00:00Z"


def test_health_without_a_run_is_five_hundred_three_with_retry_after(tmp_path: Path) -> None:
    """Пустое хранилище — не `404`: дата пользователя ни при чём, просто
    сервис ещё ничего не посчитал. `503` без `Retry-After` контракт не
    допускает (§4): клиент вернётся либо через секунду, либо никогда."""
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)

    health = client.get(HEALTH)
    coverage = client.get(COVERAGE)

    assert health.status_code == 503
    assert health.json()["status"] == "no_forecast"
    assert int(health.headers["Retry-After"]) > 0
    # Диск виден и без прогона: «хранилище доступно» и «прогноз есть» — разные
    # вопросы, и ответ на первый не должен пропадать вместе со вторым.
    assert health.json()["storage"]["total_bytes"] > 0
    assert coverage.status_code == 503
    assert coverage.json()["error"] == "no_forecast"
    assert int(coverage.headers["Retry-After"]) > 0


def test_coverage_is_cacheable_until_the_next_run(client: TestClient) -> None:
    """Покрытие меняется ровно тогда, когда меняется прогон (§5.5)."""
    response = client.get(COVERAGE)

    assert "max-age=" in response.headers["Cache-Control"]
    # Тело — валидный JSON без `NaN`: `Infinity` и `NaN` json.loads не примет.
    json.loads(response.text)

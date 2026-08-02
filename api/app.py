"""Полоса 1: JSON `/v1` (docs/API_CONTRACT.md §0).

Приложение только читает: хранилище открывается по указателю
`forecast/current`, и всё, что оно знает про прогон, взято из самого прогона.
Ничего не считается на лету, кроме единиц и производных величин — модель,
конвейер и адаптеры сюда не импортируются (`tests/test_boundaries.py`).
"""

import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import fields
from contracts import canon
from storage import read

#: Корень хранилища. Переменная окружения, а не аргумент запуска: сервис
#: поднимается через `uvicorn api.app:app`, куда аргументы не передать.
ROOT_ENV: Final = "AURORA_ROOT"
DEFAULT_ROOT: Final = "/data/aurora"

#: `source` полосы прогноза (contracts/canon.py §SOURCES). Ответ без него
#: не даёт отличить прогноз восьмичасовой давности от реанализа
#: (docs/API_CONTRACT.md §5.1).
FORECAST_SOURCE: Final = "aurora-forecast"

#: Минимальный TTL ответа: прогон меняется четыре раза в сутки, но отдавать
#: `no-cache` из-за того, что следующий срок вот-вот наступит, незачем.
MIN_TTL_SEC: Final = 60


class ApiError(Exception):
    """Ошибка с кодом и телом по контракту (docs/API_CONTRACT.md §4).

    Тело плоское: `{"error": ..., ...}`. `HTTPException` завернул бы его
    в `detail`, а подсказка в `413` обязана быть исполнимой как есть.
    """

    def __init__(self, status: int, error: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status = status
        self.body: dict[str, Any] = {"error": error, "detail": detail, **extra}


def create_app(root: str | Path | None = None) -> FastAPI:
    """Собрать приложение над конкретным корнем хранилища."""
    store = Path(root if root is not None else os.environ.get(ROOT_ENV, DEFAULT_ROOT))
    app = FastAPI(title="Aurora backend", version="1", docs_url="/v1/docs")

    @app.exception_handler(ApiError)
    async def _handle(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content=error.body)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, error: RequestValidationError) -> JSONResponse:
        """`lat=abc` до тела ручки не доходит: FastAPI не смог привести тип и
        отвечает `422` своим телом. Кодов в контракте перечислено пять (§4), и
        `422` среди них нет, а тело обязано быть плоским `{"error", "detail"}` —
        значит перевод сюда, а не проверка внутри каждой ручки."""
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:]) or str(first["loc"][0])
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "detail": f"{where}: {first['msg']}"},
        )

    @app.get("/v1/forecast/point")
    def forecast_point(
        lat: float,
        lon: float,
        vars: str = fields.DEFAULT_VARS,
        start: str | None = Query(None, alias="from"),
        to: str | None = None,
        units: str = "human",
        step_hours: int = canon.STEP_HOURS,
    ) -> JSONResponse:
        """Прогноз в точке на 10 суток (docs/API_CONTRACT.md §2)."""
        # Границы проверяются здесь, а не через `Query(ge=..., le=...)`:
        # валидатор FastAPI отдаёт `422` со своим телом, а коды ответов
        # контракт перечисляет, и `422` в этом перечне нет (§4).
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ApiError(400, "bad_point", f"lat/lon вне глобуса: {lat}, {lon}")
        if units not in ("si", "human"):
            raise ApiError(400, "bad_units", f"units: got {units!r}, expected 'si' or 'human'")
        try:
            wanted = fields.resolve(vars)
            names = fields.canonical_names(wanted)
            layer = read.choose_layer(step_hours, names)
        except (fields.UnknownFieldError, read.UnsupportedError) as error:
            raise ApiError(400, "bad_request", str(error)) from error
        if start is not None and to is not None and _moment(start) > _moment(to):
            raise ApiError(400, "bad_range", f"from > to: {start!r} > {to!r}")

        layer_dir = _layer_dir(store, layer, names)
        _check_horizon(layer_dir, step_hours, to)
        try:
            point = read.point_series(layer_dir, names, lat, lon, start=start, end=to)
        except read.TooManyStepsError as error:
            raise ApiError(
                413,
                "too_many_steps",
                str(error),
                requested=error.requested,
                limit=error.limit,
            ) from error
        except read.OutOfCoverageError as error:
            raise ApiError(404, "out_of_coverage", str(error)) from error
        except read.UnsupportedError as error:
            raise ApiError(400, "bad_request", str(error)) from error

        body = {
            "query": {
                "lat": lat,
                "lon": lon,
                # Всегда: пользователь должен видеть, что попал в узел сетки
                # 0.25° (≈25 км), а не в свой двор (docs/API_CONTRACT.md §2).
                "nearest_grid": {"lat": point.lat, "lon": point.lon},
            },
            "source": FORECAST_SOURCE,
            "init_time": point.init_time,
            "step_hours": step_hours,
            "units": {field.name: field.unit(units) for field in wanted},
            "times": list(point.times),
            "series": {field.name: field.values(point.values, units) for field in wanted},
        }
        return JSONResponse(
            content=body, headers={"Cache-Control": _cache_control(point.init_time)}
        )

    @app.get("/v1/forecast/grid")
    def forecast_grid(
        bbox: str,
        var: str,
        time: str,
        stride: int = 1,
        units: str = "human",
    ) -> JSONResponse:
        """Карта переменной на срок в компактном формате (docs/API_CONTRACT.md §1, §2)."""
        if units not in ("si", "human"):
            raise ApiError(400, "bad_units", f"units: got {units!r}, expected 'si' or 'human'")
        box = _bbox(bbox)
        try:
            wanted = fields.resolve(var)
        except fields.UnknownFieldError as error:
            raise ApiError(400, "bad_request", str(error)) from error
        # Карта — это одна переменная за запрос (docs/API_CONTRACT.md §3):
        # `values` плоский, и второй переменной в нём просто некуда лечь.
        if len(wanted) != 1:
            raise ApiError(
                400,
                "too_many_vars",
                f"var: одна переменная на запрос сетки, получено {len(wanted)}",
            )
        field = wanted[0]
        # Слой всегда шестичасовой: часовой лежит рядами, и карту из него никто
        # не читает — `time` округляется к ближайшему сроку (docs/STORAGE.md §3).
        try:
            layer = read.choose_layer(canon.STEP_HOURS, field.inputs)
        except read.UnsupportedError as error:
            raise ApiError(400, "bad_request", str(error)) from error

        layer_dir = _must_exist(_run(store) / layer, layer)
        try:
            grid = read.grid_window(layer_dir, field.inputs, box, time, stride=stride)
        except read.TooManyPointsError as error:
            raise ApiError(
                413,
                "too_many_points",
                str(error),
                requested=error.requested,
                limit=error.limit,
                hint=f"используйте stride >= {error.suggested_stride} или уменьшите bbox",
                suggested_stride=error.suggested_stride,
            ) from error
        except read.OutOfCoverageError as error:
            raise ApiError(404, "out_of_coverage", str(error)) from error
        except read.UnsupportedError as error:
            raise ApiError(400, "bad_request", str(error)) from error

        body = {
            "query": {"bbox": list(box), "var": var, "time": time, "stride": stride},
            "source": FORECAST_SOURCE,
            "init_time": grid.init_time,
            # Отданный срок, а не запрошенный: округление к ближайшему шагу
            # пользователь обязан видеть — как и узел сетки в полосе точки.
            "time": grid.time,
            "units": {field.name: field.unit(units)},
            "grid": {
                "lat0": grid.lat0,
                "lon0": grid.lon0,
                "dlat": grid.dlat,
                "dlon": grid.dlon,
                "shape": list(grid.shape),
                "order": "row-major",
            },
            "values": field.compact(grid.values, units),
        }
        return JSONResponse(content=body, headers={"Cache-Control": _cache_control(grid.init_time)})

    return app


def _bbox(text: str) -> tuple[float, float, float, float]:
    """`south,west,north,east` из запроса в четыре числа.

    Порядок именно такой (docs/API_CONTRACT.md §2), и перепутанный `bbox` —
    это `400`, а не пустая карта: `55.7,37.5,55.8,37.7` и `37.5,55.7,37.7,55.8`
    выглядят одинаково правдоподобно, но второй лежит в океане.
    """
    parts = [chunk.strip() for chunk in text.split(",")]
    if len(parts) != 4:
        raise ApiError(400, "bad_bbox", f"bbox: ожидается south,west,north,east, получено {text!r}")
    try:
        south, west, north, east = (float(part) for part in parts)
    except ValueError as error:
        raise ApiError(400, "bad_bbox", f"bbox: не число в {text!r}") from error
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ApiError(400, "bad_bbox", f"bbox: широта вне глобуса: {south}, {north}")
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ApiError(400, "bad_bbox", f"bbox: долгота вне глобуса: {west}, {east}")
    if south >= north or west >= east:
        raise ApiError(400, "bad_bbox", f"bbox: south < north и west < east, получено {text!r}")
    return south, west, north, east


def _run(store: Path) -> Path:
    """Опубликованный прогон.

    Прогона нет — это `503`, а не `404`: дата пользователя ни при чём, просто
    сервис ещё ничего не посчитал (docs/API_CONTRACT.md §4).
    """
    run = read.published_run(store)
    if run is None:
        raise ApiError(503, "no_forecast", "опубликованного прогона нет")
    return run


def _must_exist(path: Path, layer: str) -> Path:
    if not path.is_dir():
        raise ApiError(503, "no_layer", f"в прогоне нет слоя {layer}")
    return path


def _layer_dir(store: Path, layer: str, names: Sequence[str]) -> Path:
    """Каталог, из которого читается ряд в точке.

    Какая раскладка отвечает — решает хранилище (`read.point_layer`): здесь
    известно, что спросили, но не то, что для этого лежит на диске. Полоса
    сетки, наоборот, идёт в слой напрямую: карту отдаёт раскладка карт.
    """
    return _must_exist(read.point_layer(_run(store), layer, names), layer)


def _check_horizon(layer_dir: Path, step_hours: int, to: str | None) -> None:
    """Часовой шаг дальше горизонта часового слоя — отказ, а не молчаливое
    округление до шести часов (docs/API_CONTRACT.md §2)."""
    if step_hours != canon.FINE_STEP_HOURS or to is None:
        return
    _, last = read.layer_span(layer_dir)
    if _moment(to) > _moment(last):
        raise ApiError(
            400,
            "hourly_horizon",
            f"step_hours=1 работает на первые {canon.FINE_HORIZON_HOURS} ч "
            f"(до {last}), запрошено до {to}",
        )


def _moment(text: str) -> datetime:
    """ISO-строка запроса во время. Строки сравнивать нельзя: `2026-08-04`
    лексикографически больше, чем `2026-08-01T00:00:00`, и запрос внутри
    горизонта получил бы отказ."""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ApiError(400, "bad_time", f"time: got {text!r}, expected ISO 8601") from error
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _cache_control(init_time: str) -> str:
    """TTL до следующего ожидаемого прогона (docs/API_CONTRACT.md §5.5)."""
    started = datetime.strptime(init_time, read.TIME_FORMAT).replace(tzinfo=UTC)
    left = (started + timedelta(hours=canon.STEP_HOURS)) - datetime.now(UTC)
    return f"public, max-age={max(MIN_TTL_SEC, int(left.total_seconds()))}"


app = create_app()

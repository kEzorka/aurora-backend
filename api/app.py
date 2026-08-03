"""Полоса 1: JSON `/v1` (docs/API_CONTRACT.md §0).

Приложение только читает: хранилище открывается по указателю
`forecast/current`, и всё, что оно знает про прогон, взято из самого прогона.
Ничего не считается на лету, кроме единиц и производных величин — модель,
конвейер и адаптеры сюда не импортируются (`tests/test_boundaries.py`).
"""

import hashlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import exports, fields
from api import history as history_api
from cache import history as history_cache
from cache import index as cache_index
from cache import origins as cache_origins
from cache import proxy as cache_proxy
from contracts import canon
from storage import monthly as monthly_store
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

#: `Retry-After` при `503` — обязателен (docs/API_CONTRACT.md §4). Не шесть
#: часов: прогон публикуется по расписанию, но заканчивается в непредсказуемую
#: минуту, и клиент, отосланный на шесть часов, узнает о готовом прогнозе
#: последним. Пять минут — компромисс между этим и опросом в холостую.
RETRY_AFTER_SEC: Final = 300

#: Длина ETag в шестнадцатеричных знаках. Половина sha256: столкновение двух
#: разных ответов на 128 битах — не тот риск, ради которого стоит гонять по
#: сети вдвое более длинный заголовок в каждом запросе и ответе.
ETAG_HEX: Final = 32

#: Статический demo-клиент поставляется вместе с API: отдельный Node/build
#: контур для трёх файлов был бы ещё одной точкой отказа при развёртывании.
FRONTEND_DIR: Final = Path(__file__).resolve().parents[1] / "frontend"


class ExportRequest(BaseModel):
    """Forecast subset for lane 3; variable names are canonical storage names."""

    vars: list[str]
    bbox: tuple[float, float, float, float]
    start: str = Field(alias="from")
    to: str
    stride: int = 1
    format: str = "zarr"


class ApiError(Exception):
    """Ошибка с кодом и телом по контракту (docs/API_CONTRACT.md §4).

    Тело плоское: `{"error": ..., ...}`. `HTTPException` завернул бы его
    в `detail`, а подсказка в `413` обязана быть исполнимой как есть.
    """

    def __init__(self, status: int, error: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status = status
        self.body: dict[str, Any] = {"error": error, "detail": detail, **extra}


def create_app(
    root: str | Path | None = None, *, history: history_api.History | None = None
) -> FastAPI:
    """Собрать приложение над конкретным корнем хранилища."""
    store = Path(root if root is not None else os.environ.get(ROOT_ENV, DEFAULT_ROOT))
    export_manager = exports.ExportManager(store)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        export_manager.start()
        try:
            yield
        finally:
            export_manager.stop()

    history_state = history or history_api.from_env(
        store / "cache", index_path=store / cache_index.INDEX_NAME
    )
    app = FastAPI(title="Aurora backend", version="1", docs_url="/v1/docs", lifespan=lifespan)
    app.state.exports = export_manager
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    @app.get("/demo", include_in_schema=False)
    def demo() -> FileResponse:
        """Минимальный клиент для проверки численного и сеточного API."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.exception_handler(ApiError)
    async def _handle(request: Request, error: ApiError) -> JSONResponse:
        # `503` без `Retry-After` контракт не допускает (§4): клиент, которому
        # не сказали когда, вернётся либо через секунду, либо никогда.
        headers = None
        if error.status == 503:
            headers = {"Retry-After": str(RETRY_AFTER_SEC)}
        elif error.status == 429:
            headers = {"Retry-After": str(history_api.COLD_RETRY_SEC)}
        return JSONResponse(status_code=error.status, content=error.body, headers=headers)

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
        request: Request,
        lat: float,
        lon: float,
        vars: str = fields.DEFAULT_VARS,
        start: str | None = Query(None, alias="from"),
        to: str | None = None,
        units: str = "human",
        step_hours: int = canon.STEP_HOURS,
    ) -> Response:
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
        return _answer(request, body, point.init_time)

    @app.get("/v1/forecast/grid")
    def forecast_grid(
        request: Request,
        bbox: str,
        var: str,
        time: str,
        stride: int = 1,
        units: str = "human",
    ) -> Response:
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

        body: dict[str, Any] = {
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
        return _answer(request, body, grid.init_time)

    @app.get("/v1/history/point")
    def history_point(
        request: Request,
        lat: float,
        lon: float,
        vars: str = fields.DEFAULT_VARS,
        start: str = Query(alias="from"),
        to: str = Query(),
        units: str = "human",
        agg: str = history_api.RAW,
    ) -> Response:
        """Ряд ERA5 в точке с 1940 года (docs/API_CONTRACT.md §2)."""
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ApiError(400, "bad_point", f"lat/lon вне глобуса: {lat}, {lon}")
        _check_units(units)
        aggregation = _aggregation(agg)
        wanted, names = _history_fields(vars)
        first, last = _history_period(start, to, aggregation)

        try:
            cold = any(
                cache_proxy.cold(
                    history_state.series,
                    cache_origins.at_point(name, lat, lon),
                    first,
                    last,
                    root=history_state.root,
                )
                for name in names
            )
            with history_state.gate.hold(cold), closing(history_state.connect()) as conn:
                result = history_cache.point_series(
                    conn,
                    history_state.series,
                    names,
                    lat,
                    lon,
                    first,
                    last,
                    root=history_state.root,
                )
                labels, values = history_api.series(
                    wanted, result.values, result.times, aggregation, units
                )
                cache_index.log_query(
                    conn,
                    endpoint="/v1/history/point",
                    area=f"{result.lat:g},{result.lon:g}",
                    variable=",".join(field.name for field in wanted),
                    start=start,
                    stop=to,
                    cache_hit=result.hit,
                    latency_ms=result.origin_latency_ms,
                )
        except history_api.TooManyColdError as error:
            raise ApiError(429, "too_many_cold", str(error)) from error
        except cache_proxy.TooManyChunksError as error:
            raise ApiError(413, "range_too_large", str(error)) from error
        except cache_proxy.NotYetError as error:
            raise ApiError(404, "out_of_coverage", str(error)) from error
        except cache_proxy.OriginError as error:
            raise ApiError(503, "origin_unavailable", str(error)) from error

        body: dict[str, Any] = {
            "query": {
                "lat": lat,
                "lon": lon,
                "nearest_grid": {"lat": result.lat, "lon": result.lon},
            },
            "source": result.source,
            "agg": aggregation,
            "cache": {"hit": result.hit, "origin_latency_ms": result.origin_latency_ms},
            "units": {field.name: field.unit(units) for field in wanted},
            "times": list(labels),
            "series": values,
        }
        if result.preliminary:
            body["preliminary"] = True
        return _history_answer(request, body, history_api.ttl(result.preliminary))

    @app.get("/v1/history/grid")
    def history_grid(
        request: Request,
        bbox: str,
        var: str,
        time: str,
        stride: int = 1,
        units: str = "human",
        agg: str = history_api.RAW,
    ) -> Response:
        """Карта ERA5 за прошлый срок или сутки (docs/API_CONTRACT.md §2)."""
        _check_units(units)
        monthly_run = monthly_store.current(store)
        aggregation = _aggregation(agg, grid=True, monthly_grid=monthly_run is not None)
        box = _bbox(bbox)
        wanted, names = _history_fields(var)
        if len(wanted) != 1:
            raise ApiError(
                400,
                "too_many_vars",
                f"var: одна переменная на запрос сетки, получено {len(wanted)}",
            )
        if stride < 1:
            raise ApiError(400, "bad_stride", f"stride: получено {stride}, ожидалось >= 1")
        moment = _moment(time)
        if moment < _history_start():
            raise ApiError(404, "out_of_coverage", f"{time}: история начинается в 1940 году")
        first, last = (
            (moment, moment)
            if aggregation == history_api.MONTHLY
            else history_api.span(moment, aggregation)
        )
        full = history_api.grid_shape(box)
        if 0 in full:
            raise ApiError(404, "out_of_coverage", f"bbox {bbox}: узлов сетки нет")
        points = -(-full[0] // stride) * -(-full[1] // stride)
        if points > read.MAX_POINTS:
            suggested = read.stride_under(full, read.MAX_POINTS)
            raise ApiError(
                413,
                "too_many_points",
                f"точек {points}, потолок {read.MAX_POINTS}",
                requested=points,
                limit=read.MAX_POINTS,
                hint=f"используйте stride >= {suggested} или уменьшите bbox",
                suggested_stride=suggested,
            )

        if aggregation == history_api.MONTHLY:
            try:
                monthly = monthly_store.grid_window(
                    store, names, box, moment, stride=stride, max_points=read.MAX_POINTS
                )
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
            except LookupError as error:
                raise ApiError(404, "out_of_coverage", str(error)) from error
            field = wanted[0]
            monthly_body: dict[str, Any] = {
                "query": {"bbox": list(box), "var": var, "time": time, "stride": stride},
                "source": "era5-final",
                "agg": aggregation,
                # Материализованный pinned-слой не обращается к origin. Для
                # клиента это тот же гарантированный локальный hit.
                "cache": {"hit": True, "origin_latency_ms": 0},
                "time": monthly.time,
                "units": {field.name: field.unit(units)},
                "grid": {
                    "lat0": monthly.lat0,
                    "lon0": monthly.lon0,
                    "dlat": monthly.dlat,
                    "dlon": monthly.dlon,
                    "shape": list(monthly.shape),
                    "order": "row-major",
                },
                "values": field.compact(monthly.values, units),
            }
            return _history_answer(request, monthly_body, history_api.SETTLED_TTL_SEC)

        try:
            cold = any(
                cache_proxy.cold(history_state.maps, name, first, last, root=history_state.root)
                for name in names
            )
            with history_state.gate.hold(cold), closing(history_state.connect()) as conn:
                result = history_cache.grid_window(
                    conn,
                    history_state.maps,
                    names,
                    box,
                    first,
                    last,
                    root=history_state.root,
                    stride=stride,
                )
                cache_index.log_query(
                    conn,
                    endpoint="/v1/history/grid",
                    area=bbox,
                    variable=wanted[0].name,
                    start=result.first,
                    stop=result.last,
                    cache_hit=result.hit,
                    latency_ms=result.origin_latency_ms,
                )
        except history_api.TooManyColdError as error:
            raise ApiError(429, "too_many_cold", str(error)) from error
        except cache_proxy.TooManyChunksError as error:
            raise ApiError(413, "range_too_large", str(error)) from error
        except cache_proxy.NotYetError as error:
            raise ApiError(404, "out_of_coverage", str(error)) from error
        except cache_proxy.OriginError as error:
            raise ApiError(503, "origin_unavailable", str(error)) from error
        except LookupError as error:
            raise ApiError(404, "out_of_coverage", str(error)) from error

        field = wanted[0]
        body: dict[str, Any] = {
            "query": {"bbox": list(box), "var": var, "time": time, "stride": stride},
            "source": result.source,
            "agg": aggregation,
            "cache": {"hit": result.hit, "origin_latency_ms": result.origin_latency_ms},
            "time": result.first,
            "units": {field.name: field.unit(units)},
            "grid": {
                "lat0": result.lat0,
                "lon0": result.lon0,
                "dlat": result.dlat,
                "dlon": result.dlon,
                "shape": list(result.shape),
                "order": "row-major",
            },
            "values": field.compact(result.values, units),
        }
        if result.preliminary:
            body["preliminary"] = True
        return _history_answer(request, body, history_api.ttl(result.preliminary))

    @app.get("/v1/meta/coverage")
    def meta_coverage(request: Request) -> Response:
        """Что вообще есть (docs/API_CONTRACT.md §2).

        Фронтенд обязан звать это при старте, чтобы не показывать недоступные
        периоды, — и обязан суметь это без знания хранилища. Поэтому наружу
        не выходят ни имена слоёв (`coarse`, `hourly`, `points` — слова диска),
        ни канонические имена переменных: слой описан своим шагом, переменная —
        тем именем, которое можно подставить в `vars`. Лимиты в теле по той же
        причине: `stride` фронтенд обязан уметь посчитать заранее, а не узнать
        из `413` (§3).
        """
        spans = read.coverage(_run(store))
        if not spans:
            raise ApiError(503, "no_layer", "в прогоне нет ни одного слоя")
        return _answer(
            request,
            {
                "source": FORECAST_SOURCE,
                **_freshness(spans[0].init_time),
                "layers": [
                    {
                        "step_hours": span.step_hours,
                        "from": span.first,
                        "to": span.last,
                        "steps": span.steps,
                        "vars": [_described(name) for name in fields.offered(span.names)],
                    }
                    for span in spans
                ],
                "limits": {
                    "max_points": read.MAX_POINTS,
                    "max_steps": read.MAX_STEPS,
                    "vars_per_grid": 1,
                },
            },
            spans[0].init_time,
        )

    @app.post("/v1/export", status_code=202)
    def create_export(body: ExportRequest) -> JSONResponse:
        """Поставить устойчивую выгрузку прогноза в очередь (полоса 3)."""
        run = _run(store)
        layer = _must_exist(run / "coarse", "coarse")
        spans = read.coverage(run)
        if not spans:
            raise ApiError(503, "no_layer", "в прогоне нет читаемого слоя")
        spec = exports.Spec(
            tuple(body.vars), body.bbox, body.start, body.to, body.stride, body.format
        )
        try:
            job = export_manager.submit(
                layer, spec, source=FORECAST_SOURCE, init_time=spans[0].init_time
            )
        except ValueError as error:
            raise ApiError(400, "bad_export", str(error)) from error
        # TestClient без context manager не вызывает lifespan; lazy start также
        # делает очередь рабочей в таком ASGI-хосте, не меняя durability.
        export_manager.start()
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.job_id,
                "status": job.status,
                "poll": f"/v1/export/{job.job_id}",
            },
        )

    @app.get("/v1/export/{job_id}")
    def export_status(job_id: str) -> JSONResponse:
        try:
            job = export_manager.lookup(job_id)
        except exports.ExpiredError as error:
            raise ApiError(410, "export_expired", f"export {job_id} expired") from error
        except LookupError as error:
            raise ApiError(404, "export_not_found", f"export {job_id} not found") from error
        return JSONResponse(content=dict(exports.public(job)))

    @app.get("/v1/export/{job_id}/download", response_class=FileResponse)
    def download_export(job_id: str) -> FileResponse:
        try:
            artifact = export_manager.artifact(job_id)
        except exports.ExpiredError as error:
            raise ApiError(410, "export_expired", f"export {job_id} expired") from error
        except exports.NotReadyError as error:
            raise ApiError(404, "export_not_ready", f"export {job_id} is not ready") from error
        except LookupError as error:
            raise ApiError(404, "export_not_found", f"export {job_id} not found") from error
        return FileResponse(artifact, media_type="application/zip", filename=artifact.name)

    @app.get("/v1/health")
    def health() -> JSONResponse:
        """Доступность хранилища, свежесть прогноза, место на диске (§2).

        Свежесть отдаётся числом, а не приговором «устарел»: порога устаревания
        контракт не задаёт, а выдуманный порог в ответе — это решение за того,
        кто его не принимал. `age_hours` и `expected_next` дают решить самому.

        Живости воркера здесь нет: воркера ещё нет, и описывать его heartbeat
        со стороны читателя значит закрепить формат, который писать будет не
        этот код (BACKLOG 4.x).
        """
        readable = store.is_dir()
        free, total = read.disk_usage(store) if readable else (0, 0)
        body: dict[str, Any] = {
            "status": "ok",
            "storage": {"readable": readable, "free_bytes": free, "total_bytes": total},
        }
        run = read.published_run(store) if readable else None
        spans = read.coverage(run) if run is not None else ()
        if not spans:
            body["status"] = "no_forecast"
            return JSONResponse(
                status_code=503, content=body, headers={"Retry-After": str(RETRY_AFTER_SEC)}
            )
        body["forecast"] = {
            "source": FORECAST_SOURCE,
            **_freshness(spans[0].init_time),
            "layers": [{"step_hours": span.step_hours, "steps": span.steps} for span in spans],
        }
        return JSONResponse(content=body)

    return app


def _check_units(units: str) -> None:
    if units not in ("si", "human"):
        raise ApiError(400, "bad_units", f"units: got {units!r}, expected 'si' or 'human'")


def _aggregation(agg: str, *, grid: bool = False, monthly_grid: bool = False) -> str:
    try:
        return history_api.check_aggregation(agg, grid=grid, monthly_grid=monthly_grid)
    except history_api.UnknownAggregationError as error:
        raise ApiError(400, "bad_aggregation", str(error)) from error


def _history_fields(names: str) -> tuple[tuple[fields.Field, ...], tuple[str, ...]]:
    """Публичные поля истории и их канонические входы.

    CDS timeseries не публикует уровни давления, а JSON-контракт не принимает
    `level`. Отказ стоит до холодного запроса: очередь CDS не должна объяснять
    клиенту ограничение нашей полосы доступа.
    """
    try:
        wanted = fields.resolve(names)
    except fields.UnknownFieldError as error:
        raise ApiError(400, "bad_request", str(error)) from error
    canonical = fields.canonical_names(wanted)
    surface = set(canon.SURFACE_INGESTED_VARS)
    unsupported = [name for name in canonical if name not in surface]
    if unsupported:
        raise ApiError(
            400,
            "bad_request",
            f"vars: история на уровнях не поддерживается: {', '.join(unsupported)}",
        )
    return wanted, canonical


def _history_start() -> datetime:
    return datetime(canon.HISTORY_START_YEAR, 1, 1, tzinfo=UTC)


def _history_period(start: str, stop: str, agg: str) -> tuple[datetime, datetime]:
    first, last = _moment(start), _moment(stop)
    if last < first:
        raise ApiError(400, "bad_range", f"from > to: {start!r} > {stop!r}")
    if first < _history_start():
        raise ApiError(404, "out_of_coverage", f"{start}: история начинается в 1940 году")
    if last > _plus_years(first, history_api.MAX_YEARS_POINT):
        raise ApiError(
            413,
            "range_too_large",
            f"период истории больше {history_api.MAX_YEARS_POINT} лет",
        )
    steps = history_api.step_count(first, last, agg)
    if steps == 0:
        raise ApiError(404, "out_of_coverage", f"{start}..{stop}: нет сроков {agg}")
    if steps > read.MAX_STEPS:
        raise ApiError(
            413,
            "too_many_steps",
            f"шагов {steps}, потолок {read.MAX_STEPS}",
            requested=steps,
            limit=read.MAX_STEPS,
        )
    return first, last


def _plus_years(moment: datetime, years: int) -> datetime:
    """Календарная граница периода; 29 февраля превращается в 28 февраля."""
    try:
        return moment.replace(year=moment.year + years)
    except ValueError:
        return moment.replace(year=moment.year + years, day=28)


def _described(name: str) -> dict[str, Any]:
    """Переменная покрытия: имя для `vars` и единицы в обеих системах.

    Единицы кладутся рядом с именем, потому что без них фронтенд подпишет ось
    наугад: «единицы всегда в ответе, даже если очевидны» (§5.4).
    """
    field = fields.FIELDS[name]
    return {"name": name, "units": {"si": field.unit_si, "human": field.unit_human}}


def _freshness(init_time: str) -> dict[str, Any]:
    """Возраст прогноза и когда ждать следующий.

    Возраст — то самое различие «прогноз, посчитанный 8 часов назад» против
    реанализа (§5.1), а `expected_next` избавляет фронтенд от знания, что
    прогон считается четыре раза в сутки.
    """
    started = datetime.strptime(init_time, read.TIME_FORMAT).replace(tzinfo=UTC)
    age = (datetime.now(UTC) - started).total_seconds() / 3600.0
    following = started + timedelta(hours=canon.STEP_HOURS)
    return {
        "init_time": init_time,
        "age_hours": round(age, 1),
        "expected_next": following.strftime(read.TIME_FORMAT),
    }


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
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _answer(request: Request, body: dict[str, Any], init_time: str) -> Response:
    """Ответ с `Cache-Control`, `ETag` и `304` на повтор (BACKLOG 5.7).

    TTL и ETag отвечают на разные вопросы, и одного мало. `max-age` говорит,
    сколько ответ можно не перепроверять; но прогон живёт шесть часов, а карта
    на 1440×721 весит мегабайты, и клиент, у которого TTL истёк за минуту до
    нового прогона, тянет их заново ради тех же байтов. ETag превращает эту
    перекачку в `304` длиной в заголовки.

    Сравнивается ответ целиком, а не `init_time`: тело зависит ещё и от
    запроса — единицы, шаг, набор переменных, — и ETag от одного `init_time`
    выдал бы клиенту `304` на запрос, которого тот раньше не делал. Хеш от
    готовых байтов такого не умеет по построению: другое тело — другой ETag.
    """
    rendered = JSONResponse(content=body)
    tag = _etag(rendered.body)
    if _unchanged(request.headers.get("if-none-match"), tag):
        # `304` идёт без тела, но с теми же заголовками кэша: клиент продлевает
        # по ним жизнь своей копии, и без `Cache-Control` он вернётся с тем же
        # вопросом через секунду (RFC 9110 §15.4.5).
        return Response(status_code=304, headers=_cache_headers(init_time, tag))
    rendered.headers.update(_cache_headers(init_time, tag))
    return rendered


def _history_answer(request: Request, body: dict[str, Any], ttl: int) -> Response:
    """История с фиксированным TTL: час для ERA5T, сутки для финального ERA5."""
    rendered = JSONResponse(content=body)
    tag = _etag(rendered.body)
    headers = {"Cache-Control": f"public, max-age={ttl}", "ETag": tag}
    if _unchanged(request.headers.get("if-none-match"), tag):
        return Response(status_code=304, headers=headers)
    rendered.headers.update(headers)
    return rendered


def _cache_headers(init_time: str, tag: str) -> dict[str, str]:
    return {"Cache-Control": _cache_control(init_time), "ETag": tag}


def _etag(body: bytes | memoryview) -> str:
    """Сильный ETag от тела ответа.

    Именно сильный: слабый (`W/`) разрешает считать ответы равными по смыслу
    при разных байтах, а здесь равенство и есть побайтовое. Кавычки —
    обязательная часть формата, а не украшение: без них заголовок невалиден.
    """
    return '"' + hashlib.sha256(body).hexdigest()[:ETAG_HEX] + '"'


def _unchanged(offered: str | None, tag: str) -> bool:
    """Совпал ли ETag клиента с нашим.

    Список, а не одно значение: клиент вправе перечислить несколько (`"a",
    "b"`) и прислать `*`. `W/` снимается — слабое сравнение для `304` контракт
    HTTP как раз и предписывает (RFC 9110 §13.1.2).
    """
    if not offered:
        return False
    known = {part.strip().removeprefix("W/") for part in offered.split(",")}
    return "*" in known or tag in known


def _cache_control(init_time: str) -> str:
    """TTL до следующего ожидаемого прогона (docs/API_CONTRACT.md §5.5)."""
    started = datetime.strptime(init_time, read.TIME_FORMAT).replace(tzinfo=UTC)
    left = (started + timedelta(hours=canon.STEP_HOURS)) - datetime.now(UTC)
    return f"public, max-age={max(MIN_TTL_SEC, int(left.total_seconds()))}"


app = create_app()

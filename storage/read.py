"""Чтение опубликованного прогона: какой слой отвечает и что в нём в точке.

Модуль знает про диск и про слои и не знает про HTTP. Имена здесь
канонические (`2t`, `10u`), единицы — СИ; `t2m`, `wind_speed` и градусы
Цельсия появляются в `api/` (contracts/canon.py §UNITS).

Отказы — исключения с причиной, а не `None`: «часовой шаг дальше 72 часов»
и «такой переменной нет» это разные ответы пользователю, и различать их
обязан тот, кто знает, почему отказано.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import xarray as xr

from contracts import canon
from storage.manifest import MANIFEST_NAME, read_manifest
from storage.publish import current_run

#: Формат времени в ответе: ISO 8601 с `Z` (docs/API_CONTRACT.md, шапка).
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

#: Шаг запроса → слой, который его обслуживает. Часовой слой это не мелкий шаг
#: шестичасового, а отдельный набор из восьми переменных (docs/STORAGE.md §2).
LAYER_BY_STEP: Final[Mapping[int, str]] = {
    canon.STEP_HOURS: "coarse",
    canon.FINE_STEP_HOURS: "hourly",
}

#: Потолок шагов в одном ответе (docs/API_CONTRACT.md §3).
MAX_STEPS: Final = 500

#: Копия восьми шестичасовых переменных в раскладке рядов (docs/STORAGE.md §3).
POINTS_LAYER: Final = "points"


class UnsupportedError(ValueError):
    """Запрос сформулирован так, что хранилище его обслужить не может."""


class OutOfCoverageError(LookupError):
    """Запрошенное время лежит вне того, что есть на диске."""


class TooManyStepsError(ValueError):
    """Шагов в ответе больше потолка."""

    def __init__(self, requested: int, limit: int) -> None:
        super().__init__(f"шагов {requested}, потолок {limit}")
        self.requested = requested
        self.limit = limit


class Point(NamedTuple):
    """Ряд в точке: узел сетки, сроки и значения по каноническим именам."""

    lat: float
    lon: float
    init_time: str
    times: tuple[str, ...]
    values: Mapping[str, tuple[float | None, ...]]


def published_run(root: str | Path) -> Path | None:
    """Прогон, который читателю обещан, или `None`.

    Указателя мало: `forecast/current` переставляется последним, но каталог,
    на который он смотрит, обязан нести поднятый `published` — иначе это
    прогон, оборванный посреди публикации (docs/STORAGE.md §5).
    """
    run = current_run(root)
    if run is None:
        return None
    manifest = run / MANIFEST_NAME
    if not manifest.is_file():
        return None
    return run if read_manifest(manifest).get("published") else None


def choose_layer(step_hours: int, names: Sequence[str]) -> str:
    """Слой, который обслужит запрос, или отказ с причиной.

    Часовой шаг молча не округляется до шестичасового: пользователь получил бы
    не тот ряд, который просил, и не узнал бы об этом (docs/API_CONTRACT.md §2).
    """
    if step_hours not in LAYER_BY_STEP:
        allowed = ", ".join(str(value) for value in sorted(LAYER_BY_STEP))
        raise UnsupportedError(f"step_hours: got {step_hours}, expected one of {allowed}")
    layer = LAYER_BY_STEP[step_hours]
    known = set(canon.LAYERS[layer].surface_vars) | set(canon.LAYERS[layer].atmos_vars)
    unknown = [name for name in names if name not in known]
    if unknown and step_hours == canon.FINE_STEP_HOURS:
        raise UnsupportedError(
            f"step_hours=1: часовой слой это восемь переменных "
            f"({', '.join(canon.HOURLY_VARS)}), а не {', '.join(unknown)}"
        )
    if unknown:
        raise UnsupportedError(f"vars: неизвестные переменные: {', '.join(unknown)}")
    return layer


def point_layer(run: str | Path, layer: str, names: Sequence[str]) -> Path:
    """Каталог, из которого читать ряд в точке.

    Шестичасовые поля лежат дважды: `coarse` картами и `points` рядами
    (docs/STORAGE.md §3). Ряд из карт поднимает по карте на срок — 664 МБ ради
    160 чисел, 80–300 мс вместо 4, — поэтому точечная полоса идёт в `points`.

    Отступ на `coarse` нужен для прогонов, разложенных до появления слоя, и
    для полей вне восьми (`sp`, поля на уровнях): их в `points` нет, и молча
    ответить «переменной нет» вместо медленного ответа было бы враньём.
    """
    run = Path(run)
    if layer != LAYER_BY_STEP[canon.STEP_HOURS]:
        return run / layer
    covered = set(canon.LAYERS[POINTS_LAYER].surface_vars)
    if set(names) <= covered and (run / POINTS_LAYER).is_dir():
        return run / POINTS_LAYER
    return run / layer


def point_series(
    layer_dir: str | Path,
    names: Sequence[str],
    lat: float,
    lon: float,
    *,
    start: str | None = None,
    end: str | None = None,
    max_steps: int = MAX_STEPS,
) -> Point:
    """Ряд в точке по списку канонических имён.

    Набор открывается без dask: точка — это ортогональная выборка, Zarr
    поднимает под неё ровно те чанки, которых она коснулась, а dask-граф на
    сорок шагов здесь только накладные расходы.
    """
    with xr.open_zarr(layer_dir, chunks=None) as ds:
        missing = [name for name in names if name not in ds.data_vars]
        if missing:
            raise UnsupportedError(f"vars: в слое нет: {', '.join(missing)}")
        # Поле на уровнях давления — это не ряд, а тринадцать рядов. Уровня в
        # запросе точки контракт не предусматривает (docs/API_CONTRACT.md §2),
        # и отдать вместо ряда матрицу значит сломать форму ответа.
        on_levels = [name for name in names if "level" in ds[name].dims]
        if on_levels:
            raise UnsupportedError(f"vars: поля на уровнях давления: {', '.join(on_levels)}")
        window = ds.sel(time=slice(_stamp(start), _stamp(end)))
        steps = int(window.sizes["time"])
        if steps == 0:
            raise OutOfCoverageError(
                f"{start or '-'}..{end or '-'}: вне покрытия "
                f"{_iso(ds['time'].values[0])}..{_iso(ds['time'].values[-1])}"
            )
        if steps > max_steps:
            raise TooManyStepsError(steps, max_steps)
        spot = window[list(names)].sel(lat=lat, lon=lon, method="nearest").load()
        return Point(
            lat=float(spot["lat"].values),
            lon=float(spot["lon"].values),
            init_time=str(ds.attrs.get("init_time", _iso(ds["time"].values[0]))),
            times=tuple(_iso(moment) for moment in spot["time"].values),
            values={name: _jsonable(spot[name].values) for name in names},
        )


def layer_span(layer_dir: str | Path) -> tuple[str, str]:
    """Первый и последний срок слоя — для покрытия и для проверок горизонта."""
    with xr.open_zarr(layer_dir, chunks=None) as ds:
        return _iso(ds["time"].values[0]), _iso(ds["time"].values[-1])


def _stamp(moment: str | None) -> np.datetime64 | None:
    """ISO-строка запроса в `datetime64`. `Z` numpy не понимает, а срез по
    строке молча берёт лексикографический порядок, а не время."""
    if moment is None:
        return None
    try:
        return np.datetime64(moment.rstrip("Z"), "ns")
    except ValueError as error:
        raise UnsupportedError(f"time: got {moment!r}, expected ISO 8601") from error


def _iso(moment: np.datetime64) -> str:
    return str(moment.astype("datetime64[s]")) + "Z"


def _jsonable(values: np.ndarray) -> tuple[float | None, ...]:
    """NaN — это `null`, а не `NaN`: `NaN` невалиден в JSON
    (docs/API_CONTRACT.md §1)."""
    return tuple(None if np.isnan(value) else float(value) for value in np.asarray(values))

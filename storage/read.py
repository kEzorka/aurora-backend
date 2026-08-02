"""Чтение опубликованного прогона: какой слой отвечает и что в нём в точке.

Модуль знает про диск и про слои и не знает про HTTP. Имена здесь
канонические (`2t`, `10u`), единицы — СИ; `t2m`, `wind_speed` и градусы
Цельсия появляются в `api/` (contracts/canon.py §UNITS).

Отказы — исключения с причиной, а не `None`: «часовой шаг дальше 72 часов»
и «такой переменной нет» это разные ответы пользователю, и различать их
обязан тот, кто знает, почему отказано.
"""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import xarray as xr

from contracts import canon
from storage.manifest import MANIFEST_NAME, SKIP_NAME, read_manifest
from storage.publish import RUNS_DIR, current_run

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

#: Потолок точек в одном ответе (docs/API_CONTRACT.md §3). Не защита от
#: злоумышленника, а защита от опечатки: `bbox=-90,-180,90,180` без потолка
#: собирает 1 038 240 чисел на срок и кладёт сервис на ровном месте.
MAX_POINTS: Final = 50_000

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


class TooManyPointsError(ValueError):
    """Точек в окне больше потолка.

    Несёт `suggested_stride` — прореживание, при котором **то же самое окно**
    в потолок укладывается. Подсказка обязана быть исполнимой
    (docs/API_CONTRACT.md §3): «уменьшите запрос» пользователь и сам понял,
    а какое именно число подставить — нет, и подбирать его перезапросами он
    будет теми же тяжёлыми запросами, от которых потолок и защищает.
    """

    def __init__(self, requested: int, limit: int, suggested_stride: int) -> None:
        super().__init__(f"точек {requested}, потолок {limit}")
        self.requested = requested
        self.limit = limit
        self.suggested_stride = suggested_stride


class Point(NamedTuple):
    """Ряд в точке: узел сетки, сроки и значения по каноническим именам."""

    lat: float
    lon: float
    init_time: str
    times: tuple[str, ...]
    values: Mapping[str, tuple[float | None, ...]]


class Grid(NamedTuple):
    """Окно карты на один срок: геометрия и значения строка за строкой.

    Координаты не перечисляются, а описываются началом и шагом: 16 000 точек
    с парой `lat`/`lon` у каждой стоят 45 байт на точку вместо 5
    (docs/API_CONTRACT.md §1). Поэтому `lat0`/`lon0` — первый **выбранный**
    узел, а не угол `bbox`, а `dlat`/`dlon` учитывают прореживание: клиент
    восстанавливает координату как `lat0 + dlat * i`, и ошибка здесь тихо
    сдвинет всю карту.
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    shape: tuple[int, int]
    init_time: str
    time: str
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


def grid_window(
    layer_dir: str | Path,
    names: Sequence[str],
    bbox: tuple[float, float, float, float],
    moment: str,
    *,
    stride: int = 1,
    max_points: int = MAX_POINTS,
) -> Grid:
    """Окно карты на один срок (docs/API_CONTRACT.md §2).

    `bbox` — это `(south, west, north, east)` в том порядке, в каком его
    принимает контракт. Ось широты в каноне убывает (90 → -90), поэтому срез
    по ней идёт от севера к югу: `slice(south, north)` вернул бы пустоту, а не
    ошибку, и пользователь получил бы `404` на честный запрос.

    Срок выбирается ближайшим — контракт это обещает, — но только внутри
    покрытия: без проверки `sel(method="nearest")` молча отдал бы конец
    горизонта на запрос про следующий месяц.

    Потолок точек проверяется **до** `load()`: смысл потолка в том, чтобы
    не поднимать с диска то, что всё равно не отдашь.
    """
    south, west, north, east = bbox
    if stride < 1:
        raise UnsupportedError(f"stride: got {stride}, expected >= 1")
    if south >= north or west >= east:
        raise UnsupportedError(
            f"bbox: ожидается south,west,north,east, получено {south},{west},{north},{east}"
        )
    with xr.open_zarr(layer_dir, chunks=None) as ds:
        missing = [name for name in names if name not in ds.data_vars]
        if missing:
            raise UnsupportedError(f"var: в слое нет: {', '.join(missing)}")
        on_levels = [name for name in names if "level" in ds[name].dims]
        if on_levels:
            raise UnsupportedError(f"var: поля на уровнях давления: {', '.join(on_levels)}")

        stamp = _stamp(moment)
        times = ds["time"].values
        # Полшага в обе стороны: «ближайший срок» для 04:00 это 06:00, а для
        # следующего месяца — не конец горизонта, а отказ.
        half = (times[1] - times[0]) // 2 if times.size > 1 else np.timedelta64(3, "h")
        if stamp is None or stamp < times[0] - half or stamp > times[-1] + half:
            raise OutOfCoverageError(f"{moment}: вне покрытия {_iso(times[0])}..{_iso(times[-1])}")
        window = ds[list(names)].sel(lat=slice(north, south), lon=slice(west, east))
        full = (int(window.sizes["lat"]), int(window.sizes["lon"]))
        if full[0] == 0 or full[1] == 0:
            raise OutOfCoverageError(f"bbox {south},{west},{north},{east}: узлов сетки нет")
        if stride > 1:
            window = window.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        ny, nx = int(window.sizes["lat"]), int(window.sizes["lon"])
        if ny * nx > max_points:
            # Подсказка считается от неразреженного окна: пользователь подставит
            # её вместо своего `stride`, а не поверх него.
            raise TooManyPointsError(ny * nx, max_points, stride_under(full, max_points))

        at = window.sel(time=stamp, method="nearest").transpose("lat", "lon").load()
        # Шаг берётся с полной оси, а не с выборки: у окна в одну строку
        # разности нет, а шаг у неё всё равно есть.
        step_lat = float(ds["lat"].values[1] - ds["lat"].values[0]) * stride
        step_lon = float(ds["lon"].values[1] - ds["lon"].values[0]) * stride
        return Grid(
            lat0=_coord(at["lat"].values[0]),
            lon0=_coord(at["lon"].values[0]),
            dlat=_coord(step_lat),
            dlon=_coord(step_lon),
            shape=(ny, nx),
            init_time=str(ds.attrs.get("init_time", _iso(times[0]))),
            time=_iso(at["time"].values),
            values={name: _jsonable(at[name].values.ravel()) for name in names},
        )


class Span(NamedTuple):
    """Что слой на самом деле покрывает.

    Числа берутся с диска, а не из `canon.Layer`: канон говорит, сколько шагов
    прогон *должен* был записать, а покрытие — это то, что записано. Разойтись
    они могут (оборванный прогон, урезанный горизонт), и увидеть расхождение
    обязан читатель, а не только тот, кто смотрит в каталог.
    """

    step_hours: int
    init_time: str
    first: str
    last: str
    steps: int
    names: tuple[str, ...]


def coverage(run: str | Path) -> tuple[Span, ...]:
    """Покрытие опубликованного прогона по шагам.

    Ключ — шаг в часах, а не имя слоя: `coarse`, `hourly` и `points` это
    слова хранилища, и фронтенд, узнавший их, начинает от них зависеть
    (docs/API_CONTRACT.md §2, `/v1/meta/coverage`).

    Перечисляются только поля с осями `(time, lat, lon)`. Поля на уровнях
    давления в слое лежат — их 65 из 91, — но ни точка, ни сетка не берут
    `level` (§2), и обе отвечают на них отказом. Имя в покрытии — это
    обещание, что его можно подставить в `vars`, а `t` дал бы кнопку,
    которая всегда возвращает `400`. Уровни — полоса 2, `/zarr/`.
    """
    run = Path(run)
    spans: list[Span] = []
    for step_hours, layer in sorted(LAYER_BY_STEP.items()):
        layer_dir = run / layer
        if not layer_dir.is_dir():
            continue
        with xr.open_zarr(layer_dir, chunks=None) as ds:
            times = ds["time"].values
            spans.append(
                Span(
                    step_hours=step_hours,
                    init_time=str(ds.attrs.get("init_time", _iso(times[0]))),
                    first=_iso(times[0]),
                    last=_iso(times[-1]),
                    steps=int(times.size),
                    names=tuple(str(name) for name in ds.data_vars if "level" not in ds[name].dims),
                )
            )
    return tuple(spans)


class RunState(NamedTuple):
    """Строка журнала прогонов: что стало с одним циклом.

    `state` — одно из `PUBLISHED` / `UNFINISHED` / `SKIPPED`. Три состояния, а
    не два, потому что «прогона нет» бывает по двум разным причинам, и лечатся
    они по-разному: пропуск — это решение, принятое расписанием и записанное
    на диск, а незавершённый прогон — публикация, которую оборвали на середине.

    `reason` заполнен только у пропуска: у остальных его неоткуда взять, и
    выдуманное «ok» там читалось бы как проверенный факт.
    """

    run_id: str
    state: str
    init_time: str
    reason: str | None


PUBLISHED: Final = "published"
UNFINISHED: Final = "unfinished"
SKIPPED: Final = "skipped"


def run_log(root: str | Path) -> tuple[RunState, ...]:
    """Все прогоны на диске по возрастанию идентификатора.

    Существует затем, чтобы пропуск было где увидеть. `coverage` и
    `published_run` смотрят только на текущий прогон и по построению не могут
    показать дырку: прогон, которого нет, из указателя `forecast/current` не
    виден никак (docs/PIPELINE.md §3.5).
    """
    runs = Path(root) / RUNS_DIR
    if not runs.is_dir():
        return ()
    log: list[RunState] = []
    for run in sorted(runs.iterdir()):
        if not run.is_dir():
            continue
        skip = run / SKIP_NAME
        if skip.is_file():
            record = read_manifest(skip)
            log.append(
                RunState(
                    run_id=run.name,
                    state=SKIPPED,
                    init_time=str(record.get("init_time", "")),
                    reason=str(record.get("reason", "")),
                )
            )
            continue
        manifest = run / MANIFEST_NAME
        if not manifest.is_file():
            continue
        record = read_manifest(manifest)
        log.append(
            RunState(
                run_id=run.name,
                state=PUBLISHED if record.get("published") else UNFINISHED,
                init_time=_init_time(record),
                reason=None,
            )
        )
    return tuple(log)


def _init_time(manifest: Mapping[str, Any]) -> str:
    """Срок прогона по его входам.

    В манифесте своего `init_time` нет: он есть у каждого входа, и у входа с
    переносом вперёд (`ci` из ERA5T) он чужой. Поэтому берётся самый поздний —
    тот, на который прогон и посчитан (docs/DATA_CONTRACT.md §3).
    """
    inputs = manifest.get("inputs") or []
    times = [str(entry.get("valid_time", "")) for entry in inputs]
    return max(times) if times else ""


def disk_usage(root: str | Path) -> tuple[int, int]:
    """Свободно и всего байт на файловой системе хранилища.

    Отдаётся файловая система, а не бюджеты из docs/STORAGE.md §3: 40 ГБ ядра
    и 195 ГБ кэша — это план, а не разделы, и пока ротация (2.6) их не
    выдерживает, «кэш заполнен на 42 %» было бы выдумкой.
    """
    usage = shutil.disk_usage(root)
    return usage.free, usage.total


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


def stride_under(shape: tuple[int, int], limit: int) -> int:
    """Наименьшее прореживание, при котором окно `shape` влезает в потолок.

    Считается перебором, а не формулой `sqrt(точек / потолок)`: прореженный
    размер — это округление вверх, и на узком окне (одна строка, много
    столбцов) формула даёт число, которое всё ещё не влезает. Подсказка,
    которая не работает, хуже отсутствующей.
    """
    ny, nx = shape
    stride = 1
    while -(-ny // stride) * -(-nx // stride) > limit:
        stride += 1
    return stride


def _coord(value: float | np.floating) -> float:
    """Координата в ответ. Округление до шести знаков — против шума `float32`:
    `55.75000762939453` в геометрии сетки выглядит как другая сетка."""
    return round(float(value), 6)


def _iso(moment: np.datetime64) -> str:
    return str(moment.astype("datetime64[s]")) + "Z"


def _jsonable(values: np.ndarray) -> tuple[float | None, ...]:
    """NaN — это `null`, а не `NaN`: `NaN` невалиден в JSON
    (docs/API_CONTRACT.md §1)."""
    return tuple(None if np.isnan(value) else float(value) for value in np.asarray(values))

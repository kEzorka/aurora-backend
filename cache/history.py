"""История поверх кэша: ряд в точке и карта за прошлое (BACKLOG 5.5).

Единственный, кому нужны байты чанков (`cache.chunks`), и единственное место,
где чанки снова становятся данными. `cache.proxy` кладёт и отдаёт файлы, не
заглядывая внутрь; API читает не файлы, а вот эти два ответа.

Что здесь есть и чего нет. Есть: склейка чанков во время, обрезка по
запрошенному периоду, окно карты и усреднение по нему, честный `cache.hit`.
Нет: имён `t2m` и `wind`, градусов Цельсия, агрегатов `daily`/`monthly` и
потолков контракта — это полоса 1 (`api/history.py`). Граница проходит там,
где кончается «данные из источника» и начинается «как их назвали снаружи»:
кэшу переименование единиц не нужно, а API — сроки чанков.

Версия источника берётся из ключей, а не из атрибутов чанка: ключ известен до
похода наружу, и по нему же чанк лежит на диске. Слабейшее звено решает —
один предварительный час в ряду за десять лет делает предварительным весь
ответ (docs/API_CONTRACT.md §5.3): «значения могут измениться задним числом»
верно для всего ряда, если верно хоть для одного его срока.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import xarray as xr

from adapters.era5_arco import FINAL, PRELIMINARY
from cache import chunks, proxy
from cache.origins import at_point

#: Формат срока в ответе. Тот же, что у полосы прогноза (`storage.read`), и
#: записан здесь второй раз намеренно: `cache` от `storage` не зависит, а
#: заводить общий модуль ради одной строки — это связь дороже строки.
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


class Series(NamedTuple):
    """Ряд в точке: сроки и значения под каноническими именами."""

    lat: float
    lon: float
    times: tuple[str, ...]
    values: dict[str, list[float | None]]
    source: str
    hit: bool
    origin_latency_ms: int

    @property
    def preliminary(self) -> bool:
        """Данные могут измениться задним числом (docs/API_CONTRACT.md §5.3)."""
        return self.source == PRELIMINARY


class Window(NamedTuple):
    """Окно карты, усреднённое по периоду.

    Геометрия описана началом и шагом — как в `storage.read.Grid`, и по той же
    причине: пара координат у каждой из 16 000 точек стоит вдевятеро дороже
    самого значения (docs/API_CONTRACT.md §1).
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    shape: tuple[int, int]
    first: str
    last: str
    steps: int
    values: dict[str, list[float | None]]
    source: str
    hit: bool
    origin_latency_ms: int

    @property
    def preliminary(self) -> bool:
        return self.source == PRELIMINARY


def point_series(
    conn: sqlite3.Connection,
    origin: proxy.Origin,
    names: Sequence[str],
    lat: float,
    lon: float,
    start: datetime,
    stop: datetime,
    *,
    root: str | Path,
) -> Series:
    """Ряды нескольких полей в одной точке за период.

    Точка уезжает в ключ внутри имени переменной (`cache.origins.at_point`):
    у `cache.index.Key` поля под неё нет, и заводить его ради одного источника
    значило бы объяснять картам, что такое координата.

    Поля склеиваются по общей оси сроков через `xr.merge`, а не подставляются
    друг под друга по длине. Разъехаться они могут по-настоящему: растущий
    чанк, взятый для `10u` утром, а для `10v` в полдень, отличается длиной на
    несколько часов — и `wind` из них, сложенный по индексу, соединил бы
    составляющие разных сроков.
    """
    if not names:
        raise ValueError("нужно хотя бы одно поле")
    per_field: list[xr.Dataset] = []
    hits, latency = True, 0
    for name in names:
        ds, served = _load(conn, origin, at_point(name, lat, lon), start, stop, root=root)
        per_field.append(ds)
        hits, latency = hits and served.hit, latency + served.origin_latency_ms
    merged = xr.merge(per_field, join="outer", combine_attrs="drop_conflicts").sortby("time")

    return Series(
        lat=float(per_field[0]["lat"]),
        lon=float(per_field[0]["lon"]),
        times=_stamps(merged["time"].values),
        values={name: _jsonable(merged[name].values) for name in names},
        source=_source(origin, start, stop),
        hit=hits,
        origin_latency_ms=latency,
    )


def grid_window(
    conn: sqlite3.Connection,
    origin: proxy.Origin,
    names: Sequence[str],
    bbox: tuple[float, float, float, float],
    start: datetime,
    stop: datetime,
    *,
    root: str | Path,
    stride: int = 1,
) -> Window:
    """Окно карты, усреднённое по периоду.

    Среднее считается здесь, а не наверху, потому что здесь оно дешевле всего:
    суточное окно — это 24 карты по 4 МБ, и поднимать их в полосу 1, чтобы
    сложить там, значит держать в памяти сотню мегабайт ради одной карты на
    выходе.

    Ось широты в каноне убывает (90 → -90), поэтому срез по ней идёт от севера
    к югу: `slice(south, north)` вернул бы пустоту, а не ошибку — и `404` на
    честный запрос.
    """
    if not names:
        raise ValueError("нужно хотя бы одно поле")
    if stride < 1:
        raise ValueError(f"stride: получено {stride}, ожидалось >= 1")
    south, west, north, east = bbox

    windows: list[xr.Dataset] = []
    hits, latency = True, 0
    span: tuple[str, ...] = ()
    for name in names:
        ds, served = _load(conn, origin, name, start, stop, root=root)
        stamps = _stamps(ds["time"].values)
        cut = ds.sel(lat=slice(north, south), lon=slice(west, east))
        cut = cut.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        # Среднее по времени, а не по всем осям: `mean("time")` оставляет карту
        # картой. Пропуски выбрасываются (`skipna`), иначе один битый срок в
        # сутках делает пустой всю суточную карту.
        windows.append(cut[[name]].mean("time", skipna=True))
        hits, latency, span = hits and served.hit, latency + served.origin_latency_ms, stamps

    merged = xr.merge(windows, combine_attrs="drop_conflicts")
    lats = np.asarray(merged["lat"].values, dtype=float)
    lons = np.asarray(merged["lon"].values, dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise LookupError(f"окно {bbox} не пересекается с сеткой")
    return Window(
        lat0=_coord(lats[0]),
        lon0=_coord(lons[0]),
        # Шаг с учётом прореживания: клиент восстанавливает координату как
        # `lat0 + dlat * i`, и шаг неразреженной сетки сдвинул бы всю карту.
        dlat=_coord(lats[1] - lats[0]) if lats.size > 1 else 0.0,
        dlon=_coord(lons[1] - lons[0]) if lons.size > 1 else 0.0,
        shape=(int(lats.size), int(lons.size)),
        first=span[0],
        last=span[-1],
        steps=len(span),
        values={name: _jsonable(merged[name].values.ravel()) for name in names},
        source=_source(origin, start, stop),
        hit=hits,
        origin_latency_ms=latency,
    )


def _load(
    conn: sqlite3.Connection,
    origin: proxy.Origin,
    variable: str,
    start: datetime,
    stop: datetime,
    *,
    root: str | Path,
) -> tuple[xr.Dataset, proxy.Served]:
    """Чанки периода → один Dataset, обрезанный по запросу.

    Обрезка обязательна: наружу просили период, а с диска приехали чанки,
    выровненные по границам источника (`cache.proxy`), — у ARCO лишнего нет,
    а у CDS чанк длиной в год, и без обрезки запрос на неделю вернул бы год.
    """
    served = proxy.serve(conn, origin, variable, start, stop, root=root)
    parts = [chunks.decode(path.read_bytes()) for path in served.paths]
    joined = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
    return joined.sel(time=slice(_stamp(start), _stamp(stop))), served


def _source(origin: proxy.Origin, start: datetime, stop: datetime) -> str:
    """Версия ответа целиком: предварительная, если хоть один чанк такой.

    Считается по ключам, а не по атрибутам приехавших данных: ключ известен
    до похода наружу и не зависит от того, что именно источник положил внутрь.
    """
    versions = {origin.source_version(chunk) for chunk in proxy.align(origin.grid, start, stop)}
    # `startswith`, а не равенство: у растущего чанка к метке приписана дата
    # (`cache.origins.CdsOrigin.source_version`), и он тем более предварительный.
    return PRELIMINARY if any(name.startswith(PRELIMINARY) for name in versions) else FINAL


def _stamp(moment: datetime) -> np.datetime64:
    """Момент в то, чем размечена ось: UTC без пояса."""
    return np.datetime64(moment.astimezone(UTC).replace(tzinfo=None), "ns")


def _stamps(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        datetime.fromisoformat(str(np.datetime_as_string(value, unit="s"))).strftime(TIME_FORMAT)
        for value in np.atleast_1d(values)
    )


def _jsonable(values: np.ndarray) -> list[float | None]:
    """`NaN` невалиден в JSON, пропуск отдаётся как `null` (§1)."""
    return [None if np.isnan(value) else float(value) for value in np.asarray(values, dtype=float)]


def _coord(value: float | np.floating) -> float:
    """Округление до шести знаков — против шума `float32` в геометрии сетки."""
    return round(float(value), 6)

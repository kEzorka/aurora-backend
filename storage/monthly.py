"""Материализованные месячные средние ERA5 (BACKLOG 2.5).

Слой маленький относительно почасового архива, но отвечает на два разных
вопроса: карта за месяц и длинный ряд в точке. Поэтому данные записываются в
обеих раскладках, как обещает `docs/STORAGE.md` §3, и публикуются одним
указателем только после готовности обеих копий.

Среднее неполного месяца публиковать нельзя. Последний ERA5T-месяц почти
всегда неполон из-за пятидневной задержки, а число, подписанное названием
месяца, не сообщает, что в нём нет последних суток. `aggregate` принимает
только непрерывные полные календарные месяцы и падает до записи.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import xarray as xr

from contracts import canon
from storage.layout import GRID_POINTS, LAYOUT_A, LAYOUT_B
from storage.read import MAX_POINTS, TooManyPointsError, stride_under
from storage.write import write_layer

#: Четыре поля из бюджета `history/monthly` (docs/STORAGE.md §2). Три
#: публичные величины по умолчанию требуют четыре канонических ряда: скорость
#: ветра собирается из двух составляющих.
MONTHLY_VARS: Final = ("2t", "10u", "10v", "msl")
MONTHLY_LAYOUTS: Final = 2
MONTHLY_DTYPES: Final = {"2t": "float16", "10u": "float16", "10v": "float16", "msl": "float32"}
MONTHLY_BYTES: Final = {"2t": 2, "10u": 2, "10v": 2, "msl": 4}

HISTORY_DIR: Final = "history"
RUNS_DIR: Final = "monthly-runs"
CURRENT_LINK: Final = "monthly"
MAPS_LAYER: Final = "maps"
SERIES_LAYER: Final = "series"
MANIFEST_NAME: Final = "manifest.json"
VALIDATION_NAME: Final = "validation.json"


class Point(NamedTuple):
    lat: float
    lon: float
    times: tuple[str, ...]
    values: dict[str, list[float | None]]


class Grid(NamedTuple):
    lat0: float
    lon0: float
    dlat: float
    dlon: float
    shape: tuple[int, int]
    time: str
    values: dict[str, list[float | None]]


def aggregate(ds: xr.Dataset, names: Sequence[str] = MONTHLY_VARS) -> xr.Dataset:
    """Полные почасовые месяцы → одна средняя карта на начало месяца."""
    selected = _validate_hourly(ds, names)
    # Сумма сотен значений давления в float32 теряет единицы паскалей даже на
    # константном поле. Редукция идёт в float64, готовый результат ниже снова
    # ужимается до типа хранения.
    result = selected.resample(time="MS").mean(
        "time", skipna=True, keep_attrs=True, dtype=np.float64
    )
    # Среднее считается в float32 и только готовый слой сжимается. Давление в
    # Па оставлено float32: максимум float16 равен 65 504, поэтому обычные
    # 101 325 Па превратились бы в `inf`. Остальные поля безопасны в float16.
    result = result.astype({name: MONTHLY_DTYPES[name] for name in names})
    result.attrs = {
        **ds.attrs,
        "source": "era5-final",
        "kind": "reanalysis-monthly",
        "aggregation": "monthly-mean",
    }
    for name in names:
        result[name].attrs = {
            "units": canon.UNITS[name],
            "_FillValue": np.float32(np.nan)
            if MONTHLY_DTYPES[name] == "float32"
            else np.float16(np.nan),
        }
    return result


def publish(
    ds: xr.Dataset,
    root: str | Path,
    *,
    version: str | None = None,
    require_canonical_grid: bool = True,
) -> Path:
    """Атомарно опубликовать обе раскладки и вернуть каталог версии.

    Старую версию не удаляет: ротация — отдельное разрушительное решение.
    Указатель переставляется одним `replace`, поэтому читатель видит либо обе
    старые раскладки, либо обе новые, но не смесь.
    """
    checked = _validate_monthly(ds, require_canonical_grid=require_canonical_grid)
    root = Path(root)
    run_id = version or str(np.datetime_as_string(checked["time"].values[-1], unit="D"))[:7]
    final = root / HISTORY_DIR / RUNS_DIR / run_id
    if final.exists():
        raise FileExistsError(f"{final}: месячный слой {run_id} уже существует")

    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f"monthly-{run_id}-", dir=scratch))
    layer = canon.Layer("monthly", tuple(MONTHLY_VARS), (), 0, int(checked.sizes["time"]))
    write_layer(checked, staged / MAPS_LAYER, layer, layout=LAYOUT_A)
    write_layer(checked, staged / SERIES_LAYER, layer, layout=LAYOUT_B)
    validation = {
        "ok": True,
        "checks": ["fields", "calendar_months", "coordinates", "dual_layout"],
    }
    _write_json(staged / VALIDATION_NAME, validation)
    manifest = {
        "artifact": f"history/{RUNS_DIR}/{run_id}",
        "source": "era5-final",
        "from": _iso(checked["time"].values[0]),
        "to": _iso(checked["time"].values[-1]),
        "steps": int(checked.sizes["time"]),
        "vars": list(MONTHLY_VARS),
        "logical_bytes": logical_bytes(int(checked.sizes["time"])),
        "validation": VALIDATION_NAME,
        "published": True,
    }
    _write_json(staged / MANIFEST_NAME, manifest)

    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)
    _point(root / HISTORY_DIR / CURRENT_LINK, final)
    return final


def current(root: str | Path) -> Path | None:
    """Опубликованная версия с обеими раскладками и манифестом."""
    link = Path(root) / HISTORY_DIR / CURRENT_LINK
    if not link.is_symlink():
        return None
    run = link.resolve()
    manifest = run / MANIFEST_NAME
    if (
        not (run / MAPS_LAYER).is_dir()
        or not (run / SERIES_LAYER).is_dir()
        or not manifest.is_file()
    ):
        return None
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return run if record.get("published") is True else None


def point_series(
    root: str | Path,
    names: Sequence[str],
    lat: float,
    lon: float,
    start: datetime,
    stop: datetime,
) -> Point:
    """Месячные средние в точке из раскладки рядов."""
    run = _available(root)
    with xr.open_zarr(run / SERIES_LAYER, chunks=None) as ds:
        _known(ds, names)
        window = ds[list(names)].sel(time=slice(_stamp(start), _stamp(stop)))
        if int(window.sizes["time"]) == 0:
            raise LookupError(f"{start.isoformat()}..{stop.isoformat()}: месячных средних нет")
        spot = window.sel(lat=lat, lon=lon, method="nearest").load()
        return Point(
            lat=float(spot["lat"]),
            lon=float(spot["lon"]),
            times=tuple(_iso(value) for value in spot["time"].values),
            values={name: _jsonable(spot[name].values) for name in names},
        )


def grid_window(
    root: str | Path,
    names: Sequence[str],
    bbox: tuple[float, float, float, float],
    moment: datetime,
    *,
    stride: int = 1,
    max_points: int = MAX_POINTS,
) -> Grid:
    """Одна месячная карта из раскладки карт, с лимитом до чтения."""
    if stride < 1:
        raise ValueError(f"stride: получено {stride}, ожидалось >= 1")
    south, west, north, east = bbox
    run = _available(root)
    with xr.open_zarr(run / MAPS_LAYER, chunks=None) as ds:
        _known(ds, names)
        window = ds[list(names)].sel(lat=slice(north, south), lon=slice(west, east))
        full = (int(window.sizes["lat"]), int(window.sizes["lon"]))
        if 0 in full:
            raise LookupError(f"bbox {bbox}: узлов сетки нет")
        if stride > 1:
            window = window.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        shape = (int(window.sizes["lat"]), int(window.sizes["lon"]))
        if shape[0] * shape[1] > max_points:
            raise TooManyPointsError(
                shape[0] * shape[1], max_points, stride_under(full, max_points)
            )
        month = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        stamp = _stamp(month)
        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        if stamp < times[0] or stamp > times[-1] or stamp not in times:
            raise LookupError(f"{month:%Y-%m}: месячных средних нет")
        at = window.sel(time=stamp).transpose("lat", "lon").load()
        lats = np.asarray(at["lat"].values, dtype=float)
        lons = np.asarray(at["lon"].values, dtype=float)
        return Grid(
            lat0=_coord(lats[0]),
            lon0=_coord(lons[0]),
            dlat=_coord(lats[1] - lats[0]) if lats.size > 1 else 0.0,
            dlon=_coord(lons[1] - lons[0]) if lons.size > 1 else 0.0,
            shape=shape,
            time=_iso(at["time"].values),
            values={name: _jsonable(at[name].values.ravel()) for name in names},
        )


def logical_bytes(months: int) -> int:
    """Несжатый объём обеих раскладок; физическая цель после zstd — 17 ГБ."""
    return months * GRID_POINTS * sum(MONTHLY_BYTES.values()) * MONTHLY_LAYOUTS


def _validate_hourly(ds: xr.Dataset, names: Sequence[str]) -> xr.Dataset:
    _known(ds, names)
    exact = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    times = exact.astype("datetime64[h]")
    if not np.array_equal(exact, times.astype("datetime64[ns]")):
        raise ValueError("time: почасовые сроки должны лежать ровно на границе часа")
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) == np.timedelta64(1, "h"))):
        raise ValueError("time: ожидалась непрерывная почасовая ось")
    first = datetime.fromisoformat(str(times[0]))
    last = datetime.fromisoformat(str(times[-1]))
    if first.day != 1 or first.hour != 0:
        raise ValueError(f"time: первый месяц неполон, начинается {first.isoformat()}")
    following = (
        datetime(last.year + 1, 1, 1)
        if last.month == 12
        else datetime(last.year, last.month + 1, 1)
    )
    if last + timedelta(hours=1) != following:
        raise ValueError(f"time: последний месяц неполон, заканчивается {last.isoformat()}")
    for name in names:
        if ds[name].dims != ("time", "lat", "lon"):
            raise ValueError(f"{name}: оси {ds[name].dims}, ожидались ('time', 'lat', 'lon')")
    return ds[list(names)]


def _validate_monthly(ds: xr.Dataset, *, require_canonical_grid: bool) -> xr.Dataset:
    _known(ds, MONTHLY_VARS)
    times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    if times.size == 0:
        raise ValueError("time: месячный слой пуст")
    labels = [
        datetime.fromisoformat(str(np.datetime_as_string(value, unit="s"))) for value in times
    ]
    if any(label.day != 1 or label.hour or label.minute or label.second for label in labels):
        raise ValueError("time: месяцы должны быть помечены первым числом в 00:00")
    expected = [(left.year + (left.month == 12), left.month % 12 + 1) for left in labels[:-1]]
    if any(
        (right.year, right.month) != pair for right, pair in zip(labels[1:], expected, strict=True)
    ):
        raise ValueError("time: месяцы идут не подряд")
    for name in MONTHLY_VARS:
        if ds[name].dims != ("time", "lat", "lon"):
            raise ValueError(f"{name}: оси {ds[name].dims}, ожидались ('time', 'lat', 'lon')")
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    if lat.size > 1 and not np.all(np.diff(lat) < 0):
        raise ValueError("lat: ожидалась строго убывающая ось")
    if lon.size > 1 and not np.all(np.diff(lon) > 0):
        raise ValueError("lon: ожидалась строго возрастающая ось")
    if require_canonical_grid:
        for name, wanted in (("lat", canon.LAT), ("lon", canon.LON)):
            got = np.asarray(ds[name].values, dtype=float)
            if not np.array_equal(got, wanted):
                raise ValueError(f"{name}: ожидалась каноническая сетка")
    checked = ds[list(MONTHLY_VARS)].astype(MONTHLY_DTYPES)
    for name in MONTHLY_VARS:
        fill = np.float32(np.nan) if MONTHLY_DTYPES[name] == "float32" else np.float16(np.nan)
        checked[name].attrs = {"units": canon.UNITS[name], "_FillValue": fill}
    return checked


def _known(ds: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise ValueError(f"vars: месячных средних нет: {', '.join(missing)}")


def _available(root: str | Path) -> Path:
    run = current(root)
    if run is None:
        raise LookupError("месячный слой не опубликован")
    return run


def _point(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"{link}: указатель занят каталогом, а не ссылкой")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(link.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"{temporary}: временный указатель уже занят")
    os.symlink(os.path.relpath(target, link.parent), temporary)
    os.replace(temporary, link)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stamp(moment: datetime) -> np.datetime64:
    utc = moment.astimezone(UTC) if moment.tzinfo is not None else moment
    return np.datetime64(utc.replace(tzinfo=None), "ns")


def _iso(value: np.datetime64) -> str:
    return f"{np.datetime_as_string(value, unit='s')}Z"


def _jsonable(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in np.asarray(values, dtype=float)]


def _coord(value: float | np.floating) -> float:
    return round(float(value), 6)

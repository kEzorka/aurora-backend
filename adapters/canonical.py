"""Приведение сообщения источника к канонической форме.

Здесь всё, что одинаково для любого источника: оси, время, уровни, единицы,
атрибуты провенанса (docs/DATA_CONTRACT.md §1 и §2). Что различается — имена
переменных, множители единиц и правило накопления — живёт в модуле источника
и передаётся сюда параметрами.

Про две ловушки docs/DOMAIN.md §6, которые ловятся именно тут:

* **Время.** В GRIB `time` — это начало прогона, а `valid_time` — момент, к
  которому поле относится. У всех фикстур `time` одинаковое, а `valid_time`
  разное; взять `time` за канон значит записать шаг +6 ч и шаг +12 ч под одной
  и той же меткой и не заметить этого никогда.
* **Долгота.** GFS отдаёт `0..359.75`. Перекладка обязана двигать данные вместе
  с координатой: если переписать только ось, Европа окажется на месте Тихого
  океана, и ни одна проверка на диапазон этого не увидит.

Память. Адаптер работает с приёмом, а это ровно два шага по 69 полей
(docs/PIPELINE.md §4), то есть ~570 МБ, а не прогон на 11.5 ГБ. Поэтому здесь
обычный numpy, без dask: ленивость тут не нужна и только запутала бы код.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

from adapters.errors import AdapterError
from contracts import canon

#: Имена осей у cfgrib. `isobaricInhPa` — уровни давления в гПа, они же канон.
COORD_RENAMES: Mapping[str, str] = {
    "latitude": "lat",
    "longitude": "lon",
    "isobaricInhPa": "level",
}

#: Координаты сообщения, которых в каноне нет. `time` и `step` уходят потому,
#: что их содержимое уже учтено в `valid_time`; остальные — это описание типа
#: уровня («2 метра над землёй», «поверхность»), а не измерение данных.
DROPPED_COORDS: tuple[str, ...] = (
    "time",
    "step",
    "valid_time",
    "surface",
    "heightAboveGround",
    "meanSea",
    "atmosphere",
    "entireAtmosphere",
    "depthBelowLandLayer",
    "number",
)

#: Диапазон накопления сообщения. Нужен дальше, в `adapters.accumulation`,
#: и в канон переносится под нейтральными именами: слово GRIB в атрибутах
#: хранилища означало бы, что формат источника протёк наружу.
STEP_ATTRS: Mapping[str, str] = {
    "GRIB_startStep": "start_step",
    "GRIB_endStep": "end_step",
    "GRIB_stepRange": "step_range",
}


@dataclass(frozen=True)
class Provenance:
    """Атрибуты из docs/DATA_CONTRACT.md §2, обязательные для всего, что вышло
    из адаптера. `init_time` не задаётся руками: он читается из сообщения."""

    source: str
    source_url: str
    retrieved_at: str
    adapter_version: str
    kind: str = "analysis"

    def __post_init__(self) -> None:
        if self.source not in canon.SOURCES:
            raise AdapterError("source", self.source, list(canon.SOURCES))


def to_canonical(
    ds: xr.Dataset,
    *,
    renames: Mapping[str, str],
    scales: Mapping[str, float],
    provenance: Provenance,
) -> xr.Dataset:
    """Сообщение источника → канонический Dataset.

    `renames` переводит имена cfgrib в имена Aurora, `scales` — множители
    единиц источника к СИ. Переменная, которой нет в `renames`, — отказ, а не
    поле, прошедшее насквозь под чужим именем.
    """
    init_time = _init_time(ds)
    ds = _rename_variables(ds, renames)
    ds = _rename_coords(ds)
    ds = _canonical_longitude(ds)
    ds = _canonical_latitude(ds)
    _check_grid(ds)
    ds = _canonical_time(ds)
    ds = _canonical_level(ds)
    ds = _canonical_values(ds, scales)
    return _with_provenance(ds, provenance, init_time)


def _init_time(ds: xr.Dataset) -> str:
    """Начало прогона — это GRIB `time`; для анализа оно совпадает с `valid_time`."""
    if "time" not in ds.coords:
        raise AdapterError("time", "missing", "coordinate present")
    stamp = np.asarray(ds["time"].values, dtype="datetime64[s]").reshape(-1)[0]
    return f"{stamp!s}Z"


def _rename_variables(ds: xr.Dataset, renames: Mapping[str, str]) -> xr.Dataset:
    unknown = sorted({str(name) for name in ds.data_vars} - set(renames))
    if unknown:
        raise AdapterError("data_vars", unknown, sorted(renames))
    return ds.rename({name: renames[str(name)] for name in ds.data_vars})


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename({old: new for old, new in COORD_RENAMES.items() if old in ds.coords})


def _canonical_longitude(ds: xr.Dataset) -> xr.Dataset:
    """`0..359.75` → `-180..179.75`, данные едут вместе с осью.

    Перенумеровать координату и отсортировать по ней — единственный способ,
    при котором значение остаётся на своей точке земного шара: xarray двигает
    данные вслед за координатой, а `np.roll` по массиву пришлось бы
    согласовывать с осью вручную.
    """
    if "lon" not in ds.coords:
        raise AdapterError("lon", "missing", "coordinate present")
    lon = np.round(np.asarray(ds["lon"].values, dtype=float), 2)
    ds = ds.assign_coords(lon=np.where(lon >= 180.0, lon - 360.0, lon))
    if not np.all(np.diff(ds["lon"].values) > 0):
        ds = ds.sortby("lon")
    return ds


def _canonical_latitude(ds: xr.Dataset) -> xr.Dataset:
    if "lat" not in ds.coords:
        raise AdapterError("lat", "missing", "coordinate present")
    ds = ds.assign_coords(lat=np.round(np.asarray(ds["lat"].values, dtype=float), 2))
    lat = np.asarray(ds["lat"].values, dtype=float)
    if lat.size > 1 and lat[0] < lat[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    return ds


def _check_grid(ds: xr.Dataset) -> None:
    """Сетка сверяется поэлементно, а не по краям и размеру.

    Совпадение первого и последнего узла с шагом 0.25 ничего не доказывает:
    так выглядит и сетка, сдвинутая на полшага в середине, и сетка другого
    источника, округлённая иначе.
    """
    for name, expected in (("lat", canon.LAT), ("lon", canon.LON)):
        got = np.asarray(ds[name].values, dtype=float)
        if got.shape != expected.shape or not np.array_equal(got, expected):
            raise AdapterError(
                name,
                f"{got.size} nodes {got[0]}..{got[-1]}" if got.size else "empty",
                f"{expected.size} nodes {expected[0]}..{expected[-1]}",
            )


def _canonical_time(ds: xr.Dataset) -> xr.Dataset:
    """Канонический `time` — это `valid_time` сообщения (ловушка 1)."""
    if "valid_time" not in ds.coords:
        raise AdapterError("valid_time", "missing", "coordinate present")
    valid = np.asarray(ds["valid_time"].values, dtype="datetime64[ns]")
    if "valid_time" in ds.dims:
        ds = ds.drop_vars([c for c in DROPPED_COORDS if c != "valid_time" and c in ds.coords])
        return ds.rename({"valid_time": "time"})
    ds = ds.drop_vars([c for c in DROPPED_COORDS if c in ds.coords])
    return ds.expand_dims(time=valid.reshape(1))


def _canonical_level(ds: xr.Dataset) -> xr.Dataset:
    """Уровень одного сообщения — ось длины 1, а не скаляр.

    Скалярная координата не даёт склеить тринадцать сообщений в поле на
    уровнях: `concat` по несуществующей оси создал бы её сам, но порядок
    уровней при этом определялся бы порядком файлов, а он часть контракта.
    """
    if "level" not in ds.coords:
        return ds
    ds = ds.assign_coords(level=np.asarray(ds["level"].values, dtype="int32"))
    if "level" not in ds.dims:
        ds = ds.expand_dims("level")
    return ds


def _canonical_values(ds: xr.Dataset, scales: Mapping[str, float]) -> xr.Dataset:
    out = ds.transpose("time", "level", "lat", "lon", missing_dims="ignore")
    for name in list(out.data_vars):
        key = str(name)
        if key not in canon.UNITS:
            raise AdapterError("data_vars", key, sorted(canon.UNITS))
        kept = {
            new: ds[name].attrs[old] for old, new in STEP_ATTRS.items() if old in ds[name].attrs
        }
        scaled = out[name].astype(np.float32) * np.float32(scales.get(key, 1.0))
        scaled.attrs = {"units": canon.UNITS[key], "_FillValue": np.float32(np.nan), **kept}
        out[name] = scaled
    return out


def _with_provenance(ds: xr.Dataset, provenance: Provenance, init_time: str) -> xr.Dataset:
    ds.attrs = {
        "source": provenance.source,
        "source_url": provenance.source_url,
        "retrieved_at": provenance.retrieved_at,
        "init_time": init_time,
        "kind": provenance.kind,
        "grid": canon.GRID_NAME,
        "adapter_version": provenance.adapter_version,
    }
    return ds

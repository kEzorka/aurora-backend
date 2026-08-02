"""План загрузки среза: какое поле из какого потока и на какой срок.

Aurora 1.5 просит 18 приземных входов (ADDENDUM-01 §1), и одного потока на них
не хватает: `lcc`/`mcc`/`hcc` есть только в `aifs-single`, а `ci` Open Data не
публикует вовсе и берётся из ERA5T (`adapters.ecmwf.STREAMS`, `FROM_ERA5T`).
Значит срез собирается из трёх загрузок, и две из них отличаются от первой не
только адресом:

* поток другой, и в манифесте это обязано быть видно (BACKLOG 1.10) — иначе
  через полгода никто не восстановит, почему облачность в прогоне за прошлый
  вторник другая, чем в прогоне за среду;
* у ERA5T другой **срок**. `ci` на 2026-08-01 приезжает за 2026-07-27 и
  переносится вперёд. Лёд за пять суток меняется мало, но «мало» — не «никак»,
  и прятать пятидневную давность внутри поля, которое выглядит ровно как
  остальные семнадцать, нельзя.

Поэтому срок здесь — часть запроса, а не общий фон, и `assemble` записывает
настоящий срок каждого поля в атрибуты переменной. Атрибуты Dataset
провенанс не удержат: они одни на весь срез (`adapters.canonical.Provenance`),
а срез склеен из трёх источников, и одно `source` на всех было бы неправдой.

Качать этот модуль ничего не умеет: HTTP, ретраи и расписание — BACKLOG 1.7.
Здесь только то, что решается **до** загрузки: сколько запросов, какие в них
поля и какой у каждого срок.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Final, NamedTuple

import numpy as np
import xarray as xr

from adapters import ecmwf
from adapters.errors import AdapterError
from contracts import canon

#: ISO 8601 в UTC — тот же формат, что в манифесте и в API.
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

#: Набор CDS, в котором лежит `ci`. ERA5T — это предварительная версия ERA5:
#: те же поля, но пересчитываются задним числом в течение трёх месяцев.
ERA5T_STREAM: Final = "reanalysis-era5-single-levels"

#: Имя `ci` в CDS. Оно длинное и не совпадает ни с GRIB `shortName`, ни с
#: каноном, и знать его должен тот, кто формирует запрос, а не тот, кто читает
#: скачанное: `sithick` рядом в том же наборе — это толщина льда, другое поле.
ERA5T_NAMES: Final = {"ci": "sea_ice_cover"}

#: Задержка ERA5T. Пять суток — обещание CDS, а не измеренная величина, и
#: загрузчик (BACKLOG 1.7) вправе отступить дальше, если срока ещё нет.
#: Поэтому это параметр `plan`, а не константа внутри неё.
ERA5T_LAG_DAYS: Final = 5

#: Поля, которые разрешено переносить вперёд с чужого срока. Список ровно один
#: и совпадает с `FROM_ERA5T` не случайно: переносить можно то, что за эти сутки
#: почти не меняется, а туда попало то, чего на нужный срок просто нет.
PERSISTED_VARS: Final = ecmwf.FROM_ERA5T


class Request(NamedTuple):
    """Одна загрузка: откуда, из какого потока, на какой срок и что именно.

    `source` — из словаря провенанса (`canon.SOURCES`), `stream` — адрес у
    поставщика. Это разные вещи: `ifs/0p25/oper` и `aifs-single/0p25/oper` дают
    один и тот же `ifs-analysis`, и без `stream` в манифесте они неразличимы.
    """

    source: str
    stream: str
    valid_time: str
    names: tuple[str, ...]


def plan(
    valid_time: str,
    names: Sequence[str] = canon.SURFACE_INGESTED_VARS,
    *,
    era5t_lag_days: int = ERA5T_LAG_DAYS,
) -> tuple[Request, ...]:
    """Разложить список полей по загрузкам.

    Порядок запросов — порядок полей в `names`, то есть контрактный порядок
    канона, а не алфавит потоков: план попадает в манифест, и переставлять его
    от запуска к запуску значит делать два одинаковых прогона разными на вид.
    """
    wanted = _unique(names)
    unknown = [name for name in wanted if name not in canon.UNITS]
    if unknown:
        raise AdapterError("names", unknown, sorted(canon.UNITS))

    groups: dict[tuple[str, str, str], list[str]] = {}
    for name in wanted:
        if name in ecmwf.FROM_ERA5T:
            key = ("era5t", ERA5T_STREAM, _shift(valid_time, -era5t_lag_days))
        else:
            key = (ecmwf.SOURCE, ecmwf.STREAMS.get(name, ecmwf.STREAM_DEFAULT), valid_time)
        groups.setdefault(key, []).append(name)
    return tuple(Request(*key, tuple(group)) for key, group in groups.items())


def assemble(
    parts: Sequence[tuple[Request, xr.Dataset]],
    *,
    valid_time: str,
    names: Sequence[str] = canon.SURFACE_INGESTED_VARS,
) -> xr.Dataset:
    """Склеить загрузки в один срез на срок `valid_time`.

    Поле с чужого срока переносится вперёд, если это разрешено (`PERSISTED_VARS`),
    и только тогда: `xr.merge` двух срезов с разным `time` даёт ось на два срока
    и молчит, а дальше сборка батча получает шаг, которого не просила.
    """
    if not parts:
        raise AdapterError("parts", "empty", "at least one request")
    target = _stamp(valid_time)
    ready = [_at_the_target_time(request, ds, target) for request, ds in parts]

    # `combine_attrs` в `xr.merge` управляет и атрибутами переменных: `drop`
    # снёс бы вместе с провенансом ещё и `units`, а поле без единиц не примет
    # валидатор (`validators.semantics`). `override` берёт атрибуты первого
    # источника, а у каждой переменной он ровно один.
    merged = xr.merge(ready, combine_attrs="override")
    merged.attrs = {
        "init_time": valid_time,
        "kind": str(ready[0].attrs.get("kind", "analysis")),
        "grid": canon.GRID_NAME,
        "adapter_version": str(ready[0].attrs.get("adapter_version", ecmwf.ADAPTER_VERSION)),
        # Не `source`: срез склеен из трёх источников, и одно имя на всех было
        # бы неправдой. Кто именно дал поле — в атрибутах самого поля.
        "sources": sorted({str(ds.attrs.get("source", "")) for ds in ready}),
    }

    missing = [name for name in _unique(names) if name not in merged.data_vars]
    if missing:
        raise AdapterError("data_vars", f"missing {missing}", sorted(_unique(names)))
    return merged


def _at_the_target_time(request: Request, ds: xr.Dataset, target: np.datetime64) -> xr.Dataset:
    stamped = ds.copy()
    for name in list(stamped.data_vars):
        stamped[name] = stamped[name].assign_attrs(
            source=request.source, stream=request.stream, valid_time=request.valid_time
        )
    times = np.asarray(stamped["time"].values, dtype="datetime64[s]").reshape(-1)
    if times.size != 1:
        raise AdapterError("time", f"{times.size} steps", "one step per part")
    if times[0] == target:
        return stamped
    carried = [name for name in stamped.data_vars if str(name) not in PERSISTED_VARS]
    if carried:
        raise AdapterError("time", f"{carried} at {times[0]!s}", str(target))
    return stamped.assign_coords(time=np.asarray([target], dtype="datetime64[ns]"))


def _unique(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(name) for name in names))


def _stamp(valid_time: str) -> np.datetime64:
    return np.datetime64(_parse(valid_time).replace(tzinfo=None), "s")


def _shift(valid_time: str, days: int) -> str:
    return (_parse(valid_time) + timedelta(days=days)).strftime(TIME_FORMAT)


def _parse(valid_time: str) -> datetime:
    try:
        return datetime.strptime(valid_time, TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as bad:
        raise AdapterError("valid_time", valid_time, TIME_FORMAT) from bad

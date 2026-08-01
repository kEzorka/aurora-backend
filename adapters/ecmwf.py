"""ECMWF Open Data, IFS 0.25° — основной вход (docs/PIPELINE.md §1).

Чекпоинты Aurora дообучены на IFS HRES T0, и авторы модели прямо просят
подавать им именно это; GFS остаётся отладочным путём.

Aurora 1.5 просит 18 приземных полей вместо четырёх (ADDENDUM-01 §1), и
Open Data закрывает 15 из них: `lcc`/`mcc`/`hcc` приходится брать из потока
`aifs-single` (см. `STREAMS`), `ci` — из ERA5T (см. `FROM_ERA5T`).

Источник уже отдаёт долготу `-180..179.75` и всё в СИ, поэтому здесь пусто
там, где у GFS правила. Пустота намеренная: адаптер ECMWF — это тот же
`to_canonical` с другой таблицей имён, а не другой код. Единственное отличие
по существу — накопление осадков идёт **от начала прогона** (`stepRange`
`0-6`, `0-12`), и шаг берётся разностью; решает это `adapters.accumulation`
по заголовку сообщения, а не этот модуль.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import xarray as xr

from adapters import grib
from adapters.canonical import Provenance, to_canonical

SOURCE: Final = "ifs-analysis"
ADAPTER_VERSION: Final = "1.0.0"

#: Ключи — `cfVarName` из eccodes (то имя, которое даёт cfgrib), а не
#: `shortName` из документации ECMWF: у половины полей они совпадают, а у
#: `2t`/`2d`/`10u`/`100u` — нет, и перепутать их значит потерять поле молча.
RENAMES: Final = MappingProxyType(
    {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "msl": "msl",
        "d2m": "2d",
        "tcwv": "tcwv",
        "tcc": "tcc",
        "u100": "100u",
        "v100": "100v",
        "sp": "sp",
        "lcc": "lcc",
        "mcc": "mcc",
        "hcc": "hcc",
        "skt": "skt",
        # В Open Data почва лежит четырьмя слоями под одним именем `sot`/`vsw`
        # (`typeOfLevel=soilLayer`, level 1..4). Aurora берёт только верхний,
        # поэтому качать нужно ровно level 1: таблица имён по `cfVarName`
        # различить слои не может, и сообщение со второго слоя приехало бы
        # сюда под именем `stl1`. Отбор по уровню — на слое загрузки.
        "sot": "stl1",
        "vsw": "swvl1",
        "sd": "sd",
        "t": "t",
        "u": "u",
        "v": "v",
        "q": "q",
        "z": "z",
        "tp": "tp",
        "lsm": "lsm",
        "slt": "slt",
    }
)

#: Поток Open Data, из которого берётся поле. Умолчание — `ifs/0p25/oper`;
#: перечислены только исключения.
#:
#: Проверено по настоящим `.index` за 20260801 00z: в потоках `ifs/oper` и
#: `ifs/enfo` низкой, средней и высокой облачности нет вообще. Они есть в
#: потоке `aifs-single`, у которого шаг 0 ч — это начальные условия, то есть
#: тот же анализ ECMWF, интерполированный на 0.25°. Это не прогноз чужой
#: модели: на шаге 0 AIFS ещё ничего не посчитал.
STREAM_DEFAULT: Final = "ifs/0p25/oper"
STREAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lcc": "aifs-single/0p25/oper",
        "mcc": "aifs-single/0p25/oper",
        "hcc": "aifs-single/0p25/oper",
    }
)

#: Сплочённость морского льда Open Data не публикует ни в одном потоке:
#: `sithick` — это толщина, другое поле, и подменять одно другим нельзя.
#: Берётся из ERA5T с задержкой около пяти суток; лёд за пять суток меняется
#: мало, и персистенция здесь честнее нуля (решение от 2026-08-02).
FROM_ERA5T: Final = ("ci",)

#: Всё уже в СИ. Словарь оставлен пустым, а не выкинут: он часть подписи
#: `to_canonical`, и его отсутствие означало бы, что источники различаются
#: не таблицами, а кодом.
SCALES: Final[Mapping[str, float]] = MappingProxyType({})


def read_message(
    path: Path | str,
    *,
    source_url: str,
    retrieved_at: str,
    kind: str = "analysis",
) -> xr.Dataset:
    """Одно сообщение GRIB → канонический Dataset."""
    provenance = Provenance(
        source=SOURCE,
        source_url=source_url,
        retrieved_at=retrieved_at,
        adapter_version=ADAPTER_VERSION,
        kind=kind,
    )
    with grib.open_message(path) as ds:
        return to_canonical(ds, renames=RENAMES, scales=SCALES, provenance=provenance)

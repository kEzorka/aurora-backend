"""Карта ERA5T из CDS — единственный вход Aurora, которого нет нигде больше
(BACKLOG 1.10).

`ci` (сплочённость морского льда) не публикует ни один поток ECMWF Open Data:
`sithick` рядом в тех же файлах — это толщина, другое поле
(`adapters.ecmwf.FROM_ERA5T`). Значит семнадцать приземных входов приезжают
загрузчиком Open Data (`adapters.opendata`), а восемнадцатый — отсюда.

Почему не ARCO, где `sea_ice_cover` уже читается (`adapters.era5_arco`):
архив в GCS отстаёт от календаря на месяцы, а ERA5T — на пять суток. Разница
не в удобстве. Перенос `ci` вперёд с чужого срока разрешён ровно потому, что
за пять суток лёд меняется мало (`adapters.plan.PERSISTED_VARS`); лёд
трёхмесячной давности — это уже другое поле, и взять его из ARCO значило бы
оставить все проверки зелёными, отменив при этом их предпосылку.

Почему не `adapters.era5_cds`: тот модуль отдаёт **ряд в точке** и разбирает
CSV. Здесь карта на 721×1440 и GRIB, то есть другой формат ответа, другой
разбор и другой путь в канон — общего между ними ровно один клиент CDS.
Складывать их в один модуль пришлось бы ценой обещания в его заголовке.

Почему GRIB, а не netCDF: разбор GRIB в проекте уже есть (`adapters.grib`), и
вместе с ним — проверка на уровне сообщений (BACKLOG 1.8). netCDF потребовал
бы второго читателя ради одного поля.

Качает этот модуль по-настоящему, но в тестах не качает ничего: `retriever` и
`reader` подставляются. Первый нужен потому, что за CDS стоит очередь на
минуты, второй — потому что cfgrib тянет бинарный eccodes, которого в
окружении тестов нет (docs/SETUP.md §4). Против живого CDS модуль не
проверялся: ни тесты, ни CI в сеть не ходят.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import xarray as xr

from adapters import grib
from adapters.canonical import Provenance, to_canonical
from adapters.era5_arco import source_version
from adapters.errors import AdapterError, NotYetInSourceError
from adapters.plan import ERA5T_LAG_DAYS, ERA5T_NAMES, ERA5T_STREAM

#: Набор CDS. Он же `stream` в плане и в манифесте: у CDS адрес набора и есть
#: имя потока, и вторым именем то же самое здесь называть незачем.
CDS_DATASET: Final = ERA5T_STREAM

ADAPTER_VERSION: Final = "1.0.0"

#: Имя переменной у cfgrib → имя канона. Третья таблица имён после
#: `ERA5T_NAMES` (как поле зовут в заявке) и канона (как оно зовётся у нас) —
#: и деться от неё некуда: в заявку уходит `sea_ice_cover`, а в скачанном
#: GRIB то же поле называется `siconc`. Связаны они тестом, а не надеждой.
GRIB_NAMES: Final[Mapping[str, str]] = MappingProxyType({"siconc": "ci"})

#: Множителей нет: ERA5 отдаёт СИ, а `ci` — доля от 0 до 1, как в каноне.
SCALES: Final[Mapping[str, float]] = MappingProxyType({})

#: Скачать набор `dataset` по заявке `request` в файл `target`.
Retriever = Callable[[str, Mapping[str, Any], Path], None]

#: Прочитать скачанный GRIB. По умолчанию — общий разбор (`adapters.grib`).
Reader = Callable[[Path], xr.Dataset]


def build_request(variable: str, moment: datetime) -> dict[str, Any]:
    """Заявка в CDS на одно поле за один час.

    Поле, которого нет в `ERA5T_NAMES`, отвергается здесь, а не в CDS: заявка
    стоит очереди на минуты, и узнавать «такой переменной нет» после неё —
    это минуты на ошибку в имени. `ERA5T_NAMES` берётся из плана, потому что
    там же решается, какие поля вообще идут этим путём.

    Час обязан быть целым: у ERA5 почасовая сетка, а `12:30` CDS разберёт как
    заявку без единого срока и вернёт пустой ответ.
    """
    name = ERA5T_NAMES.get(variable)
    if name is None:
        raise AdapterError("variable", variable, sorted(ERA5T_NAMES))
    at = _utc(moment)
    if (at.minute, at.second, at.microsecond) != (0, 0, 0):
        raise AdapterError("moment", at.isoformat(), "whole hour")
    return {
        "product_type": ["reanalysis"],
        "variable": [name],
        "year": [f"{at.year:04d}"],
        "month": [f"{at.month:02d}"],
        "day": [f"{at.day:02d}"],
        "time": [f"{at.hour:02d}:00"],
        "data_format": "grib",
        # Иначе ответ приезжает zip-архивом с одним файлом внутри, и распаковку
        # пришлось бы держать здесь ради упаковки, которую можно не просить.
        "download_format": "unarchived",
    }


def read_map(
    variable: str,
    moment: datetime,
    *,
    retriever: Retriever | None = None,
    reader: Reader | None = None,
    target: Path | None = None,
    lag_days: int = ERA5T_LAG_DAYS,
    now: datetime | None = None,
    retrieved_at: str | None = None,
) -> xr.Dataset:
    """Карта одного поля за один срок → канонический Dataset.

    Срок, до которого ERA5T ещё не дошёл, отвергается **до** заявки:
    `NotYetInSourceError` — это слепая зона реанализа, а не сбой, и наверх она
    идёт отдельным типом (docs/CACHE.md §3.3). Задержка тут параметр, а не
    константа: пять суток — обещание CDS, и загрузчик (BACKLOG 1.7) вправе
    отступать дальше, пока срок не найдётся.

    Версия данных (`era5t` против финального ERA5) считается тем же правилом,
    что у ARCO: граница одна на весь проект, и второе её место разъехалось бы
    с первым молча.
    """
    at = _utc(moment)
    request = build_request(variable, at)
    horizon = (now or datetime.now(UTC)) - timedelta(days=lag_days)
    if at > horizon:
        raise NotYetInSourceError(
            f"{at.isoformat()} новее ERA5T (задержка {lag_days} сут, по {horizon.isoformat()})"
        )

    fetch = retriever if retriever is not None else retrieve
    read = reader if reader is not None else grib.open_message
    path = target or Path(f"era5t-{variable}-{at:%Y%m%dT%H%M}.grib")
    fetch(CDS_DATASET, request, path)

    # Чужое поле в скачанном файле отвергает `to_canonical`: имя, которого нет
    # в `renames`, — отказ, а не поле, прошедшее насквозь. Второй такой проверки
    # здесь нет намеренно: две расходятся, а расходится тихо именно вторая.
    return to_canonical(
        read(path),
        renames=dict(GRIB_NAMES),
        scales=SCALES,
        provenance=Provenance(
            source=source_version(at, now=now),
            source_url=f"cds:{CDS_DATASET}#{ERA5T_NAMES[variable]}@{at.isoformat()}",
            retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
            adapter_version=ADAPTER_VERSION,
            kind="analysis",
        ),
    )


def retrieve(dataset: str, request: Mapping[str, Any], target: Path) -> None:
    """Боевой ретривер: заявка, очередь CDS, файл на диске.

    `cdsapi` импортируется здесь, а не наверху модуля: пакета нет в окружении
    тестов, и импорт наверху сделал бы неимпортируемым весь модуль — вместе со
    сборкой заявки, которая к сети отношения не имеет.
    """
    import cdsapi

    cdsapi.Client().retrieve(dataset, dict(request), str(target))


def _utc(moment: datetime) -> datetime:
    """Срок в UTC. Наивное время — отказ, а не «наверное, UTC»: ошибка в часовом
    поясе даёт заявку на соседний срок, и выглядит она как настоящий лёд."""
    if moment.tzinfo is None:
        raise AdapterError("moment", moment.isoformat(), "timezone-aware UTC")
    return moment.astimezone(UTC)

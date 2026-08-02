"""ERA5 из публичного Zarr в GCS — карты за прошлое (BACKLOG 1.5).

Архив `gcp-public-data-arco-era5` — это тот же ERA5, разложенный по чанкам
размером в один шаг времени (docs/CACHE.md §1). Отсюда всё поведение этого
модуля: срез за один срок читается двумя-тремя обращениями к бакету, а
скачивания архива не происходит вовсе — при 80 годах почасовых данных его
некуда скачивать.

Обратная сторона той же нарезки: длинный ряд в одной точке отсюда брать
нельзя. Час за 80 лет — это сотни тысяч обращений на один ответ; ряды идут
через CDS (1.6), и предел на число чанков стоит в `cache.proxy.align`, до
первого похода наружу.

Три отличия ARCO от GRIB, из-за которых тут не `adapters.grib`:

1. **Имена переменных полные и человеческие** — `2m_temperature`, а не `t2m`.
2. **Времени два не бывает.** В GRIB есть `time` (начало прогона) и
   `valid_time` (момент поля); у реанализа начало прогона смысла не имеет, и
   ось одна. `valid_time` для `adapters.canonical` тут приходится подставить —
   см. `_slice`.
3. **Уровней 37, а не 13.** Канон берёт свои тринадцать (`contracts.canon`),
   и отбирает их до чтения — лишних уровней в памяти не появляется.

Долгота у ARCO `0..359.75`, широта убывает — перекладка та же, что у GFS, и
делает её `adapters.canonical`.

Имена в `RENAMES` взяты из документации архива, а не проверены против бакета:
тесты сюда не ходят (docs/PROGRESS.md, «Тесты не ходят в сеть»), а фикстуру
на 80 лет не нарежешь. Проверено на фикстуре то, что модуль читает ровно
запрошенный срок и приводит его к канону; что `2m_temperature` в бакете зовут
именно так, доказывает пока только документация Google — это известный предел
здешних тестов, а не забытая проверка.

`gcsfs` этот модуль не импортирует: адрес `gs://` разбирает fsspec внутри
`xarray.open_zarr`, а пакета `gcsfs` в окружении тестов нет (docs/SETUP.md).
Локальный каталог с тем же устройством открывается тем же `open_archive` —
на этом и стоят тесты.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import xarray as xr

from adapters.canonical import Provenance, to_canonical
from adapters.errors import AdapterError, NotYetInSourceError
from contracts import canon

#: Публичный архив ERA5 в GCS: 0.25°, час, 37 уровней, чанк — один срок.
ARCO_URL: Final = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ADAPTER_VERSION: Final = "1.0.0"

#: Предварительный реанализ и финальный. Не украшение имени: docs/DATA_CONTRACT.md
#: §2 требует различать их в ответе, а docs/CACHE.md §3 — инвалидировать кэш при
#: замене одного другим, и находит он чанки по этому самому слову.
PRELIMINARY: Final = "era5t"
FINAL: Final = "era5-final"

#: С какого возраста срок считается финальным. Три месяца — из
#: docs/API_CONTRACT.md §6: «для дат моложе трёх месяцев указывать `era5t`».
#: Точной даты переключения архив не сообщает, и граница тут — допущение;
#: цена ошибки в нём известная и небольшая: `source_version` входит в ключ
#: кэша, поэтому сдвиг границы стоит лишнего промаха, но не неверных данных.
FINAL_AFTER: Final = timedelta(days=90)

#: Имя переменной в ARCO → имя канона. Полные имена ERA5, как в бакете.
RENAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "2m_temperature": "2t",
        "10m_u_component_of_wind": "10u",
        "10m_v_component_of_wind": "10v",
        "mean_sea_level_pressure": "msl",
        "2m_dewpoint_temperature": "2d",
        "total_column_water_vapour": "tcwv",
        "total_cloud_cover": "tcc",
        "100m_u_component_of_wind": "100u",
        "100m_v_component_of_wind": "100v",
        "surface_pressure": "sp",
        "low_cloud_cover": "lcc",
        "medium_cloud_cover": "mcc",
        "high_cloud_cover": "hcc",
        "skin_temperature": "skt",
        "soil_temperature_level_1": "stl1",
        "volumetric_soil_water_layer_1": "swvl1",
        "sea_ice_cover": "ci",
        "snow_depth": "sd",
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "specific_humidity": "q",
        "geopotential": "z",
    }
)

#: Имя канона → имя в ARCO. Строится из `RENAMES`, а не пишется вторым списком:
#: две таблицы разъезжаются молча, и разъехавшаяся половина — это поле, которое
#: перестало читаться.
SOURCE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {canonical: arco for arco, canonical in RENAMES.items()}
)

#: Множители к единицам канона. Пусто намеренно: ERA5 отдаёт СИ, включая
#: геопотенциал в м²/с² и водный эквивалент снега в метрах. Константа
#: оставлена ради симметрии с `adapters.gfs`, где множители есть.
SCALES: Final[Mapping[str, float]] = MappingProxyType({})


def open_archive(source: Any = ARCO_URL) -> xr.Dataset:
    """Открыть архив, не читая данных.

    `chunks=None` — не «без ленивости», а «не перекладывать чанки в dask»:
    нарезку архива уже сделал тот, кто его писал, и второй слой поверх неё
    только склеил бы соседние сроки в один запрос.

    `source` — адрес `gs://` или что угодно, что понимает `xarray.open_zarr`:
    локальный каталог, store. Тесты открывают каталог, боевой код — бакет; путь
    кода при этом один, и проверяется именно он.
    """
    archive: xr.Dataset = xr.open_zarr(source, chunks=None, consolidated=False)
    return archive


def covers(archive: xr.Dataset) -> tuple[datetime, datetime]:
    """Первый и последний срок архива — то, чем кончается его слепая зона.

    Читается из самого архива, а не считается как «сегодня минус пять суток»:
    отставание реанализа плавает, и константа тут означала бы отказ в данных,
    которые уже лежат в бакете.
    """
    stamps = np.asarray(archive["time"].values, dtype="datetime64[s]")
    if stamps.size == 0:
        raise AdapterError("time", "empty", "non-empty axis")
    first, last = stamps[0], stamps[-1]
    return _as_utc(first), _as_utc(last)


def source_version(moment: datetime, *, now: datetime | None = None) -> str:
    """`era5t` или `era5-final` — по возрасту срока (docs/API_CONTRACT.md §6).

    Спрашивается **до** чтения: версия входит в ключ кэша, а ключ нужен, чтобы
    решить, идти ли в бакет вообще (`cache.proxy.Origin`).
    """
    age = (now or datetime.now(UTC)) - moment
    return FINAL if age >= FINAL_AFTER else PRELIMINARY


def read_slice(
    archive: xr.Dataset,
    variable: str,
    moment: datetime,
    *,
    source_url: str = ARCO_URL,
    retrieved_at: str | None = None,
    now: datetime | None = None,
) -> xr.Dataset:
    """Один срок одной переменной из архива → канонический Dataset.

    Читается ровно запрошенный срок: переменная и уровни отбираются до
    `.load()`, поэтому в бакет уходят только чанки этого срока. В этом и весь
    смысл ARCO — «срез читается напрямую из бакета без скачивания архива».

    `NotYetInSourceError`, когда срок новее архива: это слепая зона реанализа,
    а не сбой, и наверх она идёт отдельным типом (docs/CACHE.md §3.3).
    """
    name = SOURCE_NAMES.get(variable)
    if name is None:
        raise AdapterError("variable", variable, sorted(SOURCE_NAMES))
    if name not in archive.data_vars:
        raise AdapterError("data_vars", name, sorted(str(v) for v in archive.data_vars))

    first, last = covers(archive)
    if moment > last:
        raise NotYetInSourceError(f"{moment.isoformat()} новее архива ERA5 (по {last.isoformat()})")
    if moment < first:
        raise AdapterError("time", moment.isoformat(), f"с {first.isoformat()}")

    provenance = Provenance(
        source=source_version(moment, now=now),
        source_url=f"{source_url}#{name}@{moment.isoformat()}",
        retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
        adapter_version=ADAPTER_VERSION,
        kind="analysis",
    )
    return to_canonical(
        _slice(archive, name, moment),
        renames={name: variable},
        scales=SCALES,
        provenance=provenance,
    )


def _slice(archive: xr.Dataset, name: str, moment: datetime) -> xr.Dataset:
    """Срез одной переменной за один срок, уже в памяти.

    Порядок здесь и есть предмет приёмки: сперва отбор переменной, уровней и
    срока — и только потом `.load()`. Переставь `.load()` наверх, и модуль
    начнёт качать архив, оставаясь при этом рабочим и проходя все проверки на
    содержимое.
    """
    stamp = np.datetime64(moment.astimezone(UTC).replace(tzinfo=None), "ns")
    sliced = archive[[name]].sel(time=stamp)
    if "level" in sliced.dims:
        # Тринадцать уровней канона из тридцати семи. Отбор до `.load()`: в
        # памяти лишних уровней не окажется никогда, а доедут ли их байты из
        # бакета — решает нарезка архива по этой оси, и она не наша.
        sliced = sliced.sel(level=list(canon.PRESSURE_LEVELS))
    sliced = sliced.load()
    # `valid_time` у реанализа нет: срок и есть момент поля. Подставляем его
    # сам в себя, потому что `adapters.canonical` требует обе оси — там они
    # различаются (в GRIB `time` это начало прогона), и различие важное,
    # ловушка 1 из docs/DOMAIN.md §6. Терять общий путь приведения к канону
    # ради этого не стоит: перекладка долготы, поэлементная сверка сетки и
    # единицы нужны здесь ровно те же.
    return sliced.assign_coords(valid_time=sliced["time"])


def _as_utc(stamp: np.datetime64) -> datetime:
    return datetime.fromisoformat(str(stamp)).replace(tzinfo=UTC)

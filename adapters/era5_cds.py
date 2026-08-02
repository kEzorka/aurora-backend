"""ERA5 из CDS — длинные ряды в одной точке (BACKLOG 1.6).

Второй адаптер к тем же данным, что и `adapters.era5_arco`, и существует он
из-за нарезки. ARCO нарезан по одному сроку на чанк: карта за час читается
двумя обращениями к бакету, а ряд за десять лет стоил бы 87 672 обращений на
один ответ. CDS отдаёт ровно обратное — «ERA5 hourly time-series on single
levels» возвращает весь ряд в точке одним файлом, зато карты не отдаёт вовсе.
Поэтому: карты — из ARCO, ряды — отсюда (docs/ARCHITECTURE.md, источник O4).

Три следствия, которые видно в коде:

1. **Канон тут не тот.** `adapters.canonical.to_canonical` сверяет сетку
   поэлементно и требует все 721×1440 узлов — ряд в точке через него не
   пройдёт по устройству. Атрибуты провенанса при этом обязаны быть теми же
   (docs/DATA_CONTRACT.md §2), поэтому они собираются здесь вручную и
   проверяются тестом против списка из `adapters.canonical._with_provenance`.
2. **Только приземные поля.** У датасета нет уровней давления, а `t`, `u`,
   `v`, `q`, `z` — законные имена канона. Проверять по таблице имён нельзя:
   она их знает, и запрос ушёл бы в CDS, чтобы вернуться отказом через
   очередь длиной в минуты. Гейт стоит по `canon.SURFACE_INGESTED_VARS`.
3. **`cdsapi` импортируется лениво.** Пакета нет в тестовом окружении
   (`requirements/service.txt` против `test-minimal.txt`), а модуль обязан
   импортироваться где угодно: тесты подставляют свой `Retriever` и в сеть не
   ходят.

Имена столбцов CSV взяты из документации CDS, а не проверены против сервиса —
как и `RENAMES` в `adapters.era5_arco`, и по той же причине: тесты в сеть не
ходят. Поэтому разбор устроен так, чтобы неизвестная шапка была громким
отказом, а не тихо пустым рядом: столбец со значениями ищется как
единственный, не входящий в `META_COLUMNS`, и если такого нет или их два —
`AdapterError` с полученной шапкой. Пустой ряд и «мы потеряли столбец»
выглядят снаружи одинаково, и различать их надо здесь.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import xarray as xr

from adapters.era5_arco import FINAL, PRELIMINARY, SOURCE_NAMES, source_version
from adapters.errors import AdapterError
from contracts import canon

#: Датасет CDS и адрес API. Имя датасета — часть запроса, а не украшение:
#: `reanalysis-era5-single-levels` (без `-timeseries`) отдаёт карты и ряд в
#: точке не умеет.
CDS_DATASET: Final = "reanalysis-era5-single-levels-timeseries"
CDS_URL: Final = "https://cds.climate.copernicus.eu/api"

ADAPTER_VERSION: Final = "1.0.0"

#: Столбцы CSV, которые описывают строку, а не измеряют величину. Всё
#: остальное в шапке — значение; см. `parse_csv`.
META_COLUMNS: Final[frozenset[str]] = frozenset(
    {"valid_time", "time", "date", "datetime", "latitude", "longitude", "number", "expver"}
)

#: Столбец со сроком. Параметр, а не константа в разборе: если сервис зовёт
#: его иначе, чинится одно место.
TIME_COLUMN: Final = "valid_time"

#: Ретривер: (датасет, запрос) → текст CSV. Тот же приём, что `Transport` в
#: `adapters.fetch`: боевой ходит в CDS, тестовый отдаёт заготовленный текст,
#: и путь разбора при этом один и тот же.
Retriever = Callable[[str, Mapping[str, Any]], str]


def build_request(
    variable: str,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Запрос к CDS за одним полем в одной точке.

    Точка округляется до узла канонической сетки (`snap`) до запроса, а не
    после: CDS всё равно возьмёт ближайший узел, но тогда узел выбирал бы он,
    а в ответе стояли бы координаты, о которых мы не договаривались. Заодно
    так совпадают ключ кэша и содержимое ответа.
    """
    lat, lon = snap(lat, lon)
    if end < start:
        raise AdapterError("date", f"{start.isoformat()}..{end.isoformat()}", "start <= end")
    return {
        "variable": [_source_name(variable)],
        "location": {"latitude": lat, "longitude": lon},
        "date": [f"{start.date().isoformat()}/{end.date().isoformat()}"],
        "data_format": "csv",
    }


def read_series(
    variable: str,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    *,
    retriever: Retriever | None = None,
    now: datetime | None = None,
    retrieved_at: str | None = None,
) -> xr.Dataset:
    """Ряд в точке → канонический Dataset с осью `time` и скалярными `lat`/`lon`.

    Версию источника решает **самый свежий** срок ряда: если хоть один час
    моложе трёх месяцев, весь ряд помечается `era5t`. Иначе строка в кэше
    обещала бы финальный ERA5 там, где лежит предварительный.

    Обратная сторона известна и записана здесь, потому что чинить её придётся
    не тут: инвалидация ищет чанки точным сравнением (`cache.index.entries`,
    `where source_version = 'era5t'`), поэтому ряд за десять лет с одним
    предварительным часом на конце будет выброшен целиком при каждой замене
    ERA5T финальным. Значит, кэш рядов обязан резать их по границе трёх
    месяцев (`adapters.era5_arco.FINAL_AFTER`) — на уровне `cache.origins`,
    когда до этого дойдёт, а не подгонкой метки здесь.
    """
    fetch = retriever if retriever is not None else retrieve
    request = build_request(variable, lat, lon, start, end)
    times, values = parse_csv(fetch(CDS_DATASET, request), time_column=TIME_COLUMN)
    lat, lon = snap(lat, lon)
    newest = _as_datetime(times[-1])
    return _to_series(
        variable,
        times,
        values,
        lat=lat,
        lon=lon,
        source=source_version(newest, now=now),
        retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
    )


def parse_csv(text: str, *, time_column: str = TIME_COLUMN) -> tuple[np.ndarray, np.ndarray]:
    """CSV из CDS → (сроки, значения). Разбор стоит отдельно от сети нарочно.

    Столбец со значениями — единственный, которого нет в `META_COLUMNS`. Не
    «столбец с ожидаемым именем» потому, что имя это короткое имя ERA5 (`t2m`
    против канонического `2t`), таблицу коротких имён пришлось бы держать
    третьей — а расходятся такие таблицы молча.

    Строгое возрастание сроков проверяется, а не чинится сортировкой:
    повторившийся срок — это склеенные два ответа, и молча выбирать из них
    один нельзя.
    """
    rows = csv.reader(io.StringIO(text))
    header = next(rows, None)
    if header is None:
        raise AdapterError("csv", "empty", "header row")
    header = [name.strip() for name in header]
    if time_column not in header:
        raise AdapterError("columns", header, f"{time_column} present")
    measured = [name for name in header if name not in META_COLUMNS]
    if len(measured) != 1:
        raise AdapterError("columns", header, f"one column outside {sorted(META_COLUMNS)}")
    at_time, at_value = header.index(time_column), header.index(measured[0])

    stamps: list[np.datetime64] = []
    numbers: list[float] = []
    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise AdapterError("row", row, f"{len(header)} columns")
        stamps.append(_as_stamp(row[at_time].strip()))
        numbers.append(_as_value(row[at_value].strip()))
    if not stamps:
        raise AdapterError("csv", "no rows", "at least one row")

    times = np.asarray(stamps, dtype="datetime64[ns]")
    if times.size > 1 and not np.all(np.diff(times) > np.timedelta64(0, "ns")):
        raise AdapterError("time", "not increasing", "strictly increasing")
    return times, np.asarray(numbers, dtype=np.float32)


def retrieve(dataset: str, request: Mapping[str, Any], *, target: Path | None = None) -> str:
    """Боевой ретривер: CDS, очередь, скачанный файл → текст CSV.

    `cdsapi` импортируется здесь, а не наверху модуля: в тестовом окружении
    пакета нет, и импорт наверху сделал бы весь модуль неимпортируемым — а
    вместе с ним и разбор CSV, который к сети отношения не имеет.

    Ответ приходит либо CSV, либо zip с одним CSV внутри — зависит от датасета
    и от того, сколько полей запрошено. Разбирать надо оба: клиент CDS про
    формат ответа не говорит, а `data_format: csv` в запросе относится к
    содержимому, а не к упаковке.

    В сеть отсюда не ходят ни тесты, ни CI (docs/PROGRESS.md, «Тесты не ходят
    в сеть»): проверен разбор, а не поход, — и это известный предел.
    """
    import cdsapi

    path = target or Path(f"cds-{datetime.now(UTC):%Y%m%dT%H%M%S%f}.csv")
    cdsapi.Client().retrieve(dataset, dict(request), str(path))
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            inside = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(inside) != 1:
                raise AdapterError("zip", archive.namelist(), "exactly one .csv")
            return archive.read(inside[0]).decode("utf-8")
    return path.read_text(encoding="utf-8")


def snap(lat: float, lon: float) -> tuple[float, float]:
    """Ближайший узел канонической сетки. Точка вне шара — отказ.

    Долгота приводится к `-180..179.75` до поиска узла: 359.5 и -0.5 — это
    одно место, и оба обязаны попасть в один ключ кэша.
    """
    if not -90.0 <= lat <= 90.0:
        raise AdapterError("lat", lat, "-90..90")
    if not math.isfinite(lon):
        raise AdapterError("lon", lon, "finite")
    lon = (lon + 180.0) % 360.0 - 180.0
    return _nearest(canon.LAT, lat), _nearest(canon.LON, lon)


def _to_series(
    variable: str,
    times: np.ndarray,
    values: np.ndarray,
    *,
    lat: float,
    lon: float,
    source: str,
    retrieved_at: str,
) -> xr.Dataset:
    """Ряд в каноне: ось `time`, скалярные `lat`/`lon`, единицы и провенанс.

    Скалярные координаты, а не оси длины 1: ряд в точке — это ряд в точке, и
    ось из одного узла позвала бы к склейке рядов в карту, чего из CDS делать
    нельзя (запрос на точку, а карта — это 1 038 240 запросов).
    """
    series = xr.DataArray(values.astype(np.float32), dims=("time",), coords={"time": times})
    series.attrs = {"units": canon.UNITS[variable], "_FillValue": np.float32(np.nan)}
    ds = xr.Dataset({variable: series})
    ds = ds.assign_coords(lat=np.float64(lat), lon=np.float64(lon))
    ds.attrs = {
        "source": source,
        "source_url": f"{CDS_URL}/{CDS_DATASET}#{variable}@{lat},{lon}",
        "retrieved_at": retrieved_at,
        "init_time": f"{np.datetime_as_string(times[0], unit='s')}Z",
        "kind": "analysis",
        "grid": canon.GRID_NAME,
        "adapter_version": ADAPTER_VERSION,
    }
    return ds


def _source_name(variable: str) -> str:
    """Имя канона → полное имя ERA5, с гейтом по приземным полям.

    Гейт стоит по `canon.SURFACE_INGESTED_VARS`, а не по таблице имён: `t` и
    `q` в таблице есть, но у датасета уровней давления нет, и запрос вернулся
    бы отказом CDS через очередь — или, хуже, приземным полем под именем
    уровневого.
    """
    if variable not in canon.SURFACE_INGESTED_VARS:
        raise AdapterError("variable", variable, list(canon.SURFACE_INGESTED_VARS))
    # Таблица одна на оба адаптера ERA5 (`SOURCE_NAMES`): имена полей у CDS и
    # ARCO те же полные имена, и вторая копия разъехалась бы молча.
    name = SOURCE_NAMES.get(variable)
    if name is None:
        raise AdapterError("variable", variable, sorted(SOURCE_NAMES))
    return name


def _nearest(nodes: np.ndarray, value: float) -> float:
    index = int(np.abs(np.asarray(nodes, dtype=float) - value).argmin())
    return float(nodes[index])


def _as_stamp(text: str) -> np.datetime64:
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as bad:
        raise AdapterError("time", text, "ISO 8601") from bad
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(moment, "ns")


def _as_value(text: str) -> float:
    """Пустая клетка — пропуск (`nan`), а не ноль.

    Ноль на месте пропуска — это -273.15 °C в ряду температуры, и заметен он
    только глазами; `nan` дальше ловят валидаторы.
    """
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError as bad:
        raise AdapterError("value", text, "float") from bad


def _as_datetime(stamp: np.datetime64) -> datetime:
    return datetime.fromisoformat(str(np.datetime_as_string(stamp, unit="s"))).replace(tzinfo=UTC)


__all__ = [
    "ADAPTER_VERSION",
    "CDS_DATASET",
    "CDS_URL",
    "FINAL",
    "PRELIMINARY",
    "Retriever",
    "build_request",
    "parse_csv",
    "read_series",
    "retrieve",
    "snap",
]

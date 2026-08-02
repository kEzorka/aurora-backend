"""Запрос плана → адрес, индекс и файл на диске (BACKLOG 1.7, 1.10).

Здесь сходятся три готовые части: план (`adapters.plan`) говорит, какие поля
и на какой срок; индекс (`adapters.index`) — где они лежат в файле; загрузка
(`adapters.fetch`) — как их взять. До этого модуля соединять их было негде, и
«18 входов собраны» держалось на подставных срезах: живой Open Data не
опрашивался ни разу, потому что адреса не знал никто.

Что знает этот модуль и не знает никто другой:

* **раскладка адресов Open Data.** Файл прогона и его индекс лежат рядом и
  различаются расширением, а имя собирается из даты, часа, потока и шага:
  `.../20260801/00z/ifs/0p25/oper/20260801000000-0h-oper-fc.grib2`;
* **имена полей в индексе.** Это `shortName` ECMWF, а не `cfVarName`, который
  даёт cfgrib, и не канон. Совпадают они у большинства полей — и не совпадают
  ровно там, где ошибка дорога: почва в Open Data лежит четырьмя слоями под
  одним `sot`/`vsw`, и без `levelist=1` приедет слой, который Aurora не
  просила, под именем верхнего.

Шаг 0 — это анализ. У `ifs/0p25/oper` он есть, у `aifs-single` на нулевом шаге
модель ещё ничего не посчитала, то есть там лежат те же начальные условия
ECMWF (`adapters.ecmwf.STREAMS`). Поэтому за срез анализа берётся `0h` в обоих
потоках, а не прогноз на шесть часов вперёд с прошлого прогона.

ERA5T сюда не ходит: это CDS с очередью и заявками, другой протокол целиком
(BACKLOG 1.6). Запрос на него отвергается, а не молча даёт пустой файл.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from adapters import ecmwf
from adapters.errors import AdapterError
from adapters.fetch import DEFAULT_DELAYS, Transport, fetch_document, fetch_to_file
from adapters.index import DEFAULT_GAP, parse_ecmwf, ranges, select
from adapters.plan import TIME_FORMAT, Request
from contracts import canon

#: Корень Open Data. Тот же адрес, что у `tests.fixtures.fetch`, — и это
#: единственное место, где он теперь записан для боевой загрузки.
ROOT: Final = "https://data.ecmwf.int/forecasts"

#: Имя файла внутри каталога потока. `oper-fc` — тип данных, а не поток:
#: `aifs-single` называет свои файлы так же.
FILE_FORMAT: Final = "{date}{hour}0000-{step}h-oper-fc"

#: Расширения соседей: данные и индекс к ним.
DATA_SUFFIX: Final = ".grib2"
INDEX_SUFFIX: Final = ".index"

#: Шаг, на котором лежит анализ. Не параметр по умолчанию ради красоты:
#: `plan` даёт срок, а не пару «прогон + шаг», и любой другой шаг означал бы
#: прогноз с прошлого прогона, выданный за анализ.
ANALYSIS_STEP: Final = 0

#: Поля, у которых имя в индексе не совпадает с каноном. Остальные семнадцать
#: приземных совпадают: канон и есть `shortName` ECMWF (`adapters.ecmwf`).
#:
#: Почва — единственный случай, где нужен ещё и уровень: `sot` и `vsw` лежат
#: четырьмя слоями под одним именем, и Aurora берёт верхний.
INDEX_NAMES: Final[dict[str, tuple[str, str]]] = {
    "stl1": ("sot", "1"),
    "swvl1": ("vsw", "1"),
    # Статический геопотенциал в индексе зовётся `z` — так же, как поле на
    # уровнях давления. Различает их уровень: у приземного его нет.
    "z_surf": ("z", ""),
}


def data_url(request: Request, *, step: int = ANALYSIS_STEP) -> str:
    """Адрес файла GRIB для запроса."""
    return _base(request, step) + DATA_SUFFIX


def index_url(request: Request, *, step: int = ANALYSIS_STEP) -> str:
    """Адрес индекса к тому же файлу."""
    return _base(request, step) + INDEX_SUFFIX


def index_keys(
    names: Iterable[str], *, levels: Sequence[int] = canon.PRESSURE_LEVELS
) -> tuple[tuple[str, str], ...]:
    """Канонические имена → пары «параметр, уровень» для отбора в индексе.

    Поле на уровнях давления разворачивается в тринадцать пар: в индексе это
    тринадцать отдельных сообщений, и просить `t` без уровня значит не найти
    ни одного.
    """
    keys: list[tuple[str, str]] = []
    for name in names:
        if name in canon.ATMOS_VARS:
            keys.extend((name, str(level)) for level in levels)
        else:
            keys.append(INDEX_NAMES.get(name, (name, "")))
    return tuple(keys)


def download(
    request: Request,
    path: str | Path,
    *,
    transport: Transport,
    step: int = ANALYSIS_STEP,
    levels: Sequence[int] = canon.PRESSURE_LEVELS,
    gap: int = DEFAULT_GAP,
    delays: Sequence[float] = DEFAULT_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Скачать поля запроса в один файл GRIB.

    Порядок: индекс целиком, отбор нужных сообщений, склейка их смещений в
    диапазоны, запрос диапазонов. Файл прогона при этом не качается — из 200+
    МБ приезжают те сообщения, что просили, и разница здесь двузначная.

    Отсутствие поля в индексе — отказ (`adapters.index.select`), а не файл на
    одно поле меньше: пропуск заметили бы на сборке батча, а причину искали бы
    в другом месте.
    """
    if request.source != ecmwf.SOURCE:
        raise AdapterError("source", request.source, f"{ecmwf.SOURCE} (ERA5T — это BACKLOG 1.6)")
    index = fetch_document(
        index_url(request, step=step), transport=transport, delays=delays, sleep=sleep
    )
    chosen = select(parse_ecmwf(index.decode()), index_keys(request.names, levels=levels))
    return fetch_to_file(
        data_url(request, step=step),
        ranges(chosen, gap=gap),
        path,
        transport=transport,
        delays=delays,
        sleep=sleep,
    )


def _base(request: Request, step: int) -> str:
    if step < 0:
        raise AdapterError("step", step, "zero or more hours")
    moment = _parse(request.valid_time)
    date, hour = moment.strftime("%Y%m%d"), moment.strftime("%H")
    name = FILE_FORMAT.format(date=date, hour=hour, step=step)
    return f"{ROOT}/{date}/{hour}z/{request.stream}/{name}"


def _parse(valid_time: str) -> datetime:
    try:
        return datetime.strptime(valid_time, TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as bad:
        raise AdapterError("valid_time", valid_time, TIME_FORMAT) from bad

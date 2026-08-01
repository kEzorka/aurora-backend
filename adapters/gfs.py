"""GFS 0.25° — резервный вход (docs/PIPELINE.md §1).

Три вещи, которыми GFS отличается от ECMWF, и все три обязаны умереть внутри
этого модуля:

1. **Долгота `0..359.75`** — ловушка 2, перекладку делает `adapters.canonical`.
2. **Единицы.** Осадки в `kg m-2`, то есть в миллиметрах, — множитель 1e-3 к
   метрам канона. Геопотенциал приходит как *высота* (`HGT`, геопотенциальные
   метры), а не как геопотенциал: множитель g = 9.80665 м/с².
3. **Накопление за интервал**, а не от начала прогона — `adapters.accumulation`,
   по заголовку сообщения.

Таблица покрывает приземный набор резервного чекпоинта `aurora-0.25-finetuned`
(четыре поля), а не 18 входов Aurora 1.5: 1.5 гоняется на входе ECMWF, а GFS
остаётся отладочным. Дописывать сюда 14 строк по документации NOMADS без
фикстур — значит завести таблицу, которую нечем проверить.

Против настоящих файлов проверены `t2m` и `tp`: только они есть в фикстурах
(`tests/fixtures/README.md`). Остальные строки таблицы имён взяты из
документации NOMADS и проверяются лишь на полноту относительно канона —
это известный предел здешних тестов, а не забытая проверка.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Final

import xarray as xr

from adapters import grib
from adapters.canonical import Provenance, to_canonical

SOURCE: Final = "gfs-analysis"
ADAPTER_VERSION: Final = "1.0.0"

#: Имя переменной у cfgrib → имя канона.
RENAMES: Final = MappingProxyType(
    {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "prmsl": "msl",
        "t": "t",
        "u": "u",
        "v": "v",
        "q": "q",
        "gh": "z",
        "tp": "tp",
        "lsm": "lsm",
        "orog": "z_surf",
        "slt": "slt",
    }
)

#: Множитель к единицам канона. Отсутствие имени означает 1.0, а не «неизвестно»:
#: остальные поля GFS уже в СИ.
SCALES: Final = MappingProxyType(
    {
        "z": 9.806_65,  # геопотенциальные метры -> m2 s-2
        "z_surf": 9.806_65,
        "tp": 1e-3,  # kg m-2 (= мм) -> m
    }
)


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

"""ECMWF Open Data, IFS 0.25° — основной вход (docs/PIPELINE.md §1).

Чекпоинт `aurora-0.25-finetuned` дообучен на IFS HRES T0, и авторы модели
прямо просят подавать ему именно это; GFS остаётся отладочным путём.

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

RENAMES: Final = MappingProxyType(
    {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "msl": "msl",
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

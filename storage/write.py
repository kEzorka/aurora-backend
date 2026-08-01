"""Запись слоя в Zarr v3.

Пишется ровно то, что слою принадлежит, и ровно теми чанками, что объявлены
в раскладке. Поверх существующего слоя запись не идёт: перезапись на месте
даёт читателю полусмешанный срез — часть переменных старая, часть новая
(docs/STORAGE.md §5). Новый прогон — новый ключ; переключение указателя и
флаг `published` — задача атомарной публикации (2.3).
"""

from pathlib import Path

import xarray as xr

from contracts import canon
from storage.layout import LAYOUT_A, Chunking, encoding_for, select_layer


def write_layer(
    ds: xr.Dataset,
    path: str | Path,
    layer: canon.Layer,
    *,
    layout: Chunking = LAYOUT_A,
) -> Path:
    """Записать слой и вернуть путь, по которому он лёг.

    Валидация здесь не вызывается намеренно: проверяется записанное, а не то,
    что собирались записать (docs/PIPELINE.md §3), — иначе проверка не увидит
    ни кодека, ни округления, ни того, что на диск легло не всё.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target}: слой уже записан, пишите в новый ключ")

    selected = select_layer(ds, layer)
    selected.to_zarr(
        target,
        mode="w-",
        zarr_format=3,
        consolidated=True,
        encoding=encoding_for(selected, layout),
    )
    return target

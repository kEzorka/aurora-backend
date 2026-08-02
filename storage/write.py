"""Запись слоя в Zarr v3.

Пишется ровно то, что слою принадлежит, и ровно теми чанками, что объявлены
в раскладке. Поверх существующего слоя запись не идёт: перезапись на месте
даёт читателю полусмешанный срез — часть переменных старая, часть новая
(docs/STORAGE.md §5). Новый прогон — новый ключ; переключение указателя и
флаг `published` — задача атомарной публикации (2.3).
"""

from pathlib import Path
from typing import cast

import xarray as xr

from contracts import canon
from storage.layout import Chunking, encoding_for, layout_for, select_layer


def write_layer(
    ds: xr.Dataset,
    path: str | Path,
    layer: canon.Layer,
    *,
    layout: Chunking | None = None,
) -> Path:
    """Записать слой и вернуть путь, по которому он лёг.

    Раскладка по умолчанию — та, что закреплена за именем слоя
    (`layout.layout_for`): забытый аргумент дал бы слой, который на свой
    запрос отвечает в сто раз дольше и ничем себя не выдаёт.

    Валидация здесь не вызывается намеренно: проверяется записанное, а не то,
    что собирались записать (docs/PIPELINE.md §3), — иначе проверка не увидит
    ни кодека, ни округления, ни того, что на диск легло не всё.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target}: слой уже записан, пишите в новый ключ")

    selected = select_layer(ds, layer)
    encoding = encoding_for(selected, layout if layout is not None else layout_for(layer.name))
    _align_chunks(selected, encoding).to_zarr(
        target,
        mode="w-",
        zarr_format=3,
        consolidated=True,
        encoding=encoding,
    )
    return target


def _align_chunks(ds: xr.Dataset, encoding: dict[str, dict[str, object]]) -> xr.Dataset:
    """Переразбить ленивый набор по границам шардов.

    Набор, открытый из Zarr, приходит с чужими чанками, и шард, накрывающий
    два dask-чанка, xarray писать отказывается: две задачи пишут один файл
    шарда параллельно и затирают друг друга. Набор в памяти не трогаем —
    у него чанков нет вовсе.
    """
    if not any(variable.chunks for variable in ds.data_vars.values()):
        return ds
    aligned = ds.copy()
    for name, variable in ds.data_vars.items():
        if variable.chunks is None:
            continue
        shards = cast(tuple[int, ...], encoding[str(name)]["shards"])
        aligned[name] = variable.chunk(dict(zip(variable.dims, shards, strict=True)))
    return aligned

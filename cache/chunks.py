"""Чем чанк лежит на диске между origin и тем, кто его просил.

Формат — забота кэша, а не источника (docs/CACHE.md §3.1): `cache.proxy`
кладёт байты, не заглядывая внутрь, а разбирает их тот, кто просил. Поэтому
кодек живёт здесь, а не в `adapters/`: API читать `adapters` не имеет права
(`tests/test_boundaries.py`), а история `/v1/history/*` (5.5) — единственный,
кому эти байты вообще нужны.

Взят `.npz`, а не netCDF: писателя netCDF в окружении нет ни одного —
`netcdf4` и `h5netcdf` не стоят ни в `requirements/service.txt`, ни в
`test-minimal.txt`, — а тащить его ради кэша значит завести зависимость,
которой нет у боевого кода. Zarr тоже не годится: он каталог, а чанк кэша —
файл, который кладут переименованием.

Что теряется при обходе: numpy-скаляры в атрибутах возвращаются питоновскими
числами (`_FillValue` был `float32`, стал `float`). Значения полей, их тип,
оси и провенанс не меняются — а на них и стоит всё остальное.
"""

from __future__ import annotations

import io
import json
from typing import Any, Final, cast

import numpy as np
import xarray as xr

#: Ключ описания внутри архива. Двойные подчёркивания — чтобы он не совпал с
#: именем поля: имена полей и осей в одном Dataset живут в общем пространстве,
#: и `time` от источника затёрло бы описание молча.
META: Final = "__meta__"


def encode(ds: xr.Dataset) -> bytes:
    """Канонический Dataset → байты чанка.

    Массивы кладутся как есть, остальное — в JSON: дописать в описание поле
    дешевле, чем менять раскладку архива.
    """
    meta: dict[str, Any] = {
        "attrs": _plain(ds.attrs),
        "coords": {str(name): _entry(ds[name]) for name in ds.coords},
        "vars": {str(name): _entry(ds[name]) for name in ds.data_vars},
    }
    payload = {META: _packed(meta)}
    payload.update({str(name): np.asarray(ds[name].values) for name in (*ds.coords, *ds.data_vars)})
    buffer = io.BytesIO()
    # `cast` — из-за стабов numpy: в них `savez_compressed` объявлен как
    # `(file, *args, allow_pickle, **kwds)`, и `**payload` mypy подставляет в
    # `allow_pickle`. Имя массива внутри архива задаётся только именованным
    # аргументом, другого способа положить их под именами нет.
    save = cast(Any, np.savez_compressed)
    save(buffer, **payload)
    return buffer.getvalue()


def decode(data: bytes) -> xr.Dataset:
    """Байты чанка → тот же Dataset.

    `allow_pickle=False` не перестраховка: кэш — это каталог с файлами, который
    чистят руками, переносят между машинами и монтируют с чужого диска, а
    pickle в нём означал бы выполнение чужого кода при чтении.
    """
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        meta = json.loads(bytes(archive[META]).decode())
        coords = {
            name: (entry["dims"], archive[name], entry["attrs"])
            for name, entry in meta["coords"].items()
        }
        variables = {
            name: (entry["dims"], archive[name], entry["attrs"])
            for name, entry in meta["vars"].items()
        }
    return xr.Dataset(variables, coords=coords, attrs=meta["attrs"])


def _entry(array: xr.DataArray) -> dict[str, Any]:
    return {"dims": [str(dim) for dim in array.dims], "attrs": _plain(array.attrs)}


def _plain(attrs: Any) -> dict[str, Any]:
    """Атрибуты в то, что переживёт JSON. Скаляры numpy — в питоновские."""
    return {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in attrs.items()
    }


def _packed(meta: dict[str, Any]) -> np.ndarray:
    """JSON байтами внутри `.npz`: массив — единственное, что туда кладётся."""
    return np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

"""Кусок ARCO ERA5 на диске: имена, оси и нарезка по сроку — как в бакете.

Живёт отдельно от тестов адаптера, потому что нужен ещё и кэшу: origin поверх
ARCO проверяется на том же архиве, каким проверялся сам адаптер, — иначе
«работает у адаптера, но не через кэш» никто не поймает.

Сетка настоящего размера, 721×1440: `adapters.canonical` сверяет её поэлементно,
и на уменьшенной проверялся бы не тот код, который поедет в бакет. Отсюда
осторожность с объёмом — сроков три, а не восемьдесят лет.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from contracts import canon

#: Сроки фикстуры: три подряд идущих часа.
STEPS: tuple[datetime, ...] = tuple(datetime(2020, 6, 1, hour, tzinfo=UTC) for hour in range(3))

#: Уровни фикстуры: канонические тринадцать плюс четыре чужих. Чужие нужны,
#: чтобы отбор уровней проверялся на архиве, где есть что не брать.
LEVELS: tuple[int, ...] = tuple(sorted({*canon.PRESSURE_LEVELS, 10, 775, 800, 875}))


def build(
    path: Path,
    name: str,
    times: tuple[datetime, ...] = STEPS,
    levels: tuple[int, ...] = (),
) -> Path:
    """Написать архив. Вернуть путь к нему.

    Значение в точке равно её номеру по долготе плюс тысяча за каждый срок:
    перекладка долготы двигает данные вместе с осью, и проверить это можно
    только по значениям, а не по подписям осей (ловушка 2 из docs/DOMAIN.md §6).
    """
    lat = np.round(np.arange(90.0, -90.0 - canon.GRID_STEP / 2, -canon.GRID_STEP), 2)
    lon = np.round(np.arange(0.0, 360.0 - canon.GRID_STEP / 2, canon.GRID_STEP), 2)
    ramp = np.tile(np.arange(lon.size, dtype=np.float32), (lat.size, 1))
    stamps = np.array([t.replace(tzinfo=None) for t in times], dtype="datetime64[ns]")

    if levels:
        values = np.stack(
            [
                np.stack([ramp + step * 1000.0 + level for level in levels])
                for step in range(len(times))
            ]
        )
        dims: tuple[str, ...] = ("time", "level", "latitude", "longitude")
        coords: dict[str, np.ndarray] = {
            "time": stamps,
            "level": np.array(levels, dtype="int32"),
            "latitude": lat,
            "longitude": lon,
        }
        # Нарезка по сроку — то самое свойство ARCO, ради которого карта за
        # прошлое читается парой запросов (docs/CACHE.md §1).
        chunks: tuple[int, ...] = (1, len(levels), lat.size, lon.size)
    else:
        values = np.stack([ramp + step * 1000.0 for step in range(len(times))])
        dims = ("time", "latitude", "longitude")
        coords = {"time": stamps, "latitude": lat, "longitude": lon}
        chunks = (1, lat.size, lon.size)

    ds = xr.Dataset({name: (dims, values.astype(np.float32))}, coords=coords)
    ds.to_zarr(
        path, mode="w", zarr_format=3, consolidated=False, encoding={name: {"chunks": chunks}}
    )
    return path


def spoil(root: Path, name: str, keep: int) -> None:
    """Забить мусором все чанки переменной, кроме принадлежащих сроку `keep`.

    Мусор, а не удаление: на месте недостающего чанка Zarr молча отдаёт
    `fill_value`, и проверка «читался только нужный срок» прошла бы, даже если
    бы читался весь архив.

    Имя чанка в Zarr v3 — `c/<срок>/<...>`; первый индекс и есть номер срока.
    """
    base = root / name / "c"
    for chunk in sorted(base.rglob("*")):
        if chunk.is_file() and int(chunk.relative_to(base).parts[0]) != keep:
            chunk.write_bytes(b"\x00" * 32)

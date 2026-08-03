"""Чтение GRIB. Единственное место в проекте, которое знает про cfgrib.

`indexpath=""` — не писать `.idx` рядом с исходным файлом: сырой слой
неизменяем (docs/PIPELINE.md §3), и чтение не имеет права его трогать.

`read_keys` — единственный способ увидеть границы интервала накопления:
в координату `step` cfgrib кладёт только `endStep`, а весь смысл различия
между источниками сидит в `startStep`.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

MESSAGE_KEYS: tuple[str, ...] = ("stepRange", "startStep", "endStep")


def open_message(path: Path | str) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "read_keys": list(MESSAGE_KEYS)},
    )


def open_messages(path: Path | str) -> tuple[xr.Dataset, ...]:
    """Открыть неоднородный GRIB, сгруппировав совместимые сообщения.

    Один оперативный range-download содержит приземные поля и тринадцать
    уровней сразу. ``xr.open_dataset`` выбирает только одну совместимую
    группу; ``cfgrib.open_datasets`` возвращает их все, после чего адаптер
    объединяет уже канонические группы.
    """
    # cfgrib/eccodes is optional on the service/test laptop and is imported
    # only at the operational GRIB boundary (docs/SETUP.md §4).
    import cfgrib

    return tuple(
        cfgrib.open_datasets(
            path,
            backend_kwargs={"indexpath": "", "read_keys": list(MESSAGE_KEYS)},
        )
    )

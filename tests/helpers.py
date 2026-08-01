"""Синтетические канонические Dataset'ы — только для тестов самих валидаторов.

docs/TESTING.md запрещает синтетику там, где проверяется чтение источника:
у неё правильные оси, правильные единицы и нет накопленных величин, поэтому
ловушки она не воспроизводит. Здесь проверяется сам валидатор, и «всё
правильное» — ровно то, что нужно: тест портит одну вещь за раз и смотрит,
поймана ли она.

Про память. Настоящая сетка — 721 × 1440, одно поле float32 весит 4.15 МБ,
69 полей — 286 МБ на один тест. np.broadcast_to даёт read-only вид без
копирования, поэтому набор из 69 «полей» стоит несколько килобайт.
"""

from collections.abc import Iterable

import numpy as np
import xarray as xr

from contracts import canon

SURFACE_DEFAULTS = {"2t": 288.0, "10u": 3.0, "10v": -2.0, "msl": 101_325.0}
ATMOS_DEFAULTS = {"t": 250.0, "u": 10.0, "v": 5.0, "q": 0.004, "z": 50_000.0}


def varying_field(shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    """Поле с градиентом: константное поле валится проверкой «здравый смысл»."""
    total = int(np.prod(shape))
    ramp = np.linspace(0.0, scale, total, dtype=np.float32)
    return ramp.reshape(shape)


def canonical_dataset(
    times: int = 1,
    init_time: str = "2026-08-01T00:00:00",
    surface_vars: Iterable[str] | None = None,
    atmos_vars: Iterable[str] | None = None,
) -> xr.Dataset:
    """Dataset в канонической форме: оси, уровни, единицы, атрибуты провенанса."""
    surface = tuple(surface_vars) if surface_vars is not None else canon.SURFACE_VARS
    atmos = tuple(atmos_vars) if atmos_vars is not None else canon.ATMOS_VARS

    time = np.array(
        [
            np.datetime64(init_time) + np.timedelta64(canon.STEP_HOURS * i, "h")
            for i in range(times)
        ],
        dtype="datetime64[ns]",
    )
    ny, nx = canon.GRID_SHAPE
    nl = len(canon.PRESSURE_LEVELS)

    data: dict[str, xr.DataArray] = {}
    for name in surface:
        value = np.float32(SURFACE_DEFAULTS.get(name, 1.0))
        data[name] = xr.DataArray(
            np.broadcast_to(value, (times, ny, nx)), dims=("time", "lat", "lon")
        )
    for name in atmos:
        value = np.float32(ATMOS_DEFAULTS.get(name, 1.0))
        data[name] = xr.DataArray(
            np.broadcast_to(value, (times, nl, ny, nx)), dims=("time", "level", "lat", "lon")
        )

    ds = xr.Dataset(
        data,
        coords={
            "time": time,
            "level": np.array(canon.PRESSURE_LEVELS, dtype="int32"),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
        attrs={
            "source": "ifs-analysis",
            "source_url": "test://fixture",
            "retrieved_at": "2026-08-01T07:41:12Z",
            "init_time": f"{init_time}Z",
            "kind": "analysis",
            "grid": canon.GRID_NAME,
            "adapter_version": "0.0.0-test",
        },
    )
    for key, variable in ds.data_vars.items():
        variable.attrs["units"] = canon.UNITS[str(key)]
        variable.attrs["_FillValue"] = np.float32(np.nan)
    return ds


def plausible_dataset(times: int = 1, base_2t: float = 288.0) -> xr.Dataset:
    """Канонический Dataset, у которого 2t меняется по сетке.

    Нужен там, где проверяется «здравый смысл»: константное поле такую проверку
    не проходит и не должно проходить.
    """
    ds = canonical_dataset(times=times)
    ny, nx = canon.GRID_SHAPE
    field = base_2t + varying_field((times, ny, nx), scale=0.5)
    out = ds.assign({"2t": (("time", "lat", "lon"), field)})
    out["2t"].attrs.update(ds["2t"].attrs)
    return out

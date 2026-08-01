"""Валидатор не имеет права загружать срез целиком.

Один прогон — 40 шагов, 90 полей, 15 ГБ float32 (ADDENDUM-01 §3).
Одна температура на уровнях давления — 2.2 ГБ, а приведение к float64 внутри
проверки удваивает их. Проверка, которая делает `.values`, работает на
синтетике 1×721×1440 и падает по памяти ровно там, где нужна: на настоящем
артефакте. Поэтому здесь измеряется не результат, а расход памяти.
"""

import numpy as np
import pytest
import xarray as xr

from contracts import canon
from tests.helpers import SURFACE_DEFAULTS
from validators import validate

dask_array = pytest.importorskip("dask.array")

#: Номинальный размер набора: 8 шагов × 13 уровней × 721 × 1440 × float32 ≈ 2.7 ГБ
#: на переменную. Считается по чанкам, поэтому реально не занимает ничего.
STEPS = 8

#: Потолок расхода. Один чанк (1 шаг × 1 уровень) — 4.15 МБ; редукция считает
#: несколько чанков сразу, поэтому запас взят с большим коэффициентом. Важно
#: не точное число, а то, что оно не порядка гигабайта.
PEAK_LIMIT_MB = 300


def _lazy_canonical_dataset(steps: int) -> xr.Dataset:
    """Канонический срез из dask-чанков, который не помещается в память целиком."""
    ny, nx = canon.GRID_SHAPE
    nl = len(canon.PRESSURE_LEVELS)
    time = np.array(
        [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)],
        dtype="datetime64[ns]",
    )
    relief = np.linspace(-1.0, 1.0, ny, dtype=np.float32).reshape(ny, 1)

    def surface(value: float) -> xr.DataArray:
        profile = np.float32(value) * (1.0 + np.float32(0.01) * relief)
        block = dask_array.from_array(profile, chunks=(ny, 1))
        return xr.DataArray(
            dask_array.broadcast_to(block, (steps, ny, nx)).rechunk((1, ny, nx)),
            dims=("time", "lat", "lon"),
        )

    def atmos(value: float) -> xr.DataArray:
        profile = np.float32(value) * (1.0 + np.float32(0.01) * relief)
        block = dask_array.from_array(profile, chunks=(ny, 1))
        return xr.DataArray(
            dask_array.broadcast_to(block, (steps, nl, ny, nx)).rechunk((1, 1, ny, nx)),
            dims=("time", "level", "lat", "lon"),
        )

    data: dict[str, xr.DataArray] = {
        n: surface(SURFACE_DEFAULTS[n]) for n in canon.SURFACE_STORED_VARS
    }
    for name, value in (("t", 250.0), ("u", 10.0), ("v", 5.0), ("q", 0.004), ("z", 50_000.0)):
        data[name] = atmos(value)

    ds = xr.Dataset(
        data,
        coords={
            "time": time,
            "level": np.array(canon.PRESSURE_LEVELS, dtype="int32"),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    for key, variable in ds.data_vars.items():
        variable.attrs["units"] = canon.UNITS[str(key)]
    return ds


def test_nominal_size_is_large_enough_for_the_test_to_mean_anything() -> None:
    """Иначе тест ниже проходит потому, что грузить было нечего."""
    ds = _lazy_canonical_dataset(STEPS)
    assert ds.nbytes > 2 * 1024**3


def test_validation_of_a_multi_gigabyte_slice_stays_within_a_few_hundred_megabytes() -> None:
    import tracemalloc

    ds = _lazy_canonical_dataset(STEPS)
    tracemalloc.start()
    try:
        report = validate(ds)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert report.ok, [c.message for c in report.failures()]
    assert peak < PEAK_LIMIT_MB * 1024**2, f"пик {peak / 1024**2:.0f} МБ"

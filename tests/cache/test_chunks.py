"""Кодек чанка кэша.

Формат придуман тут же (`cache.chunks`), поэтому проверять его надо не «читает
ли», а «что теряется по дороге»: чанк лежит на диске между походом наружу и
ответом пользователю, и всё, что кодек не сохранил, потеряно навсегда — второго
похода в бакет за тем же чанком не будет, в этом и смысл кэша.

Сетка тут маленькая, в отличие от `tests/fixtures/arco.py`: кодек не знает про
канон и обязан работать с любым Dataset.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import xarray as xr

from cache.chunks import decode, encode


def _sample() -> xr.Dataset:
    """Срез со всем, что кодеку тяжело: время, целые уровни, атрибуты полей."""
    return xr.Dataset(
        {
            "t": (
                ("time", "level", "lat", "lon"),
                np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
                {"units": "K", "_FillValue": np.float32(np.nan)},
            )
        },
        coords={
            "time": np.array(["2020-06-01T00"], dtype="datetime64[ns]"),
            "level": np.array([850, 1000], dtype="int32"),
            "lat": np.array([1.0, 0.75, 0.5]),
            "lon": np.array([0.0, 0.25, 0.5, 0.75]),
        },
        attrs={"source": "era5-final", "grid": "0.25deg-global"},
    )


def test_the_values_and_axes_come_back_the_same() -> None:
    """Данные, оси и порядок осей — то, ради чего чанк и лежит на диске."""
    ds = _sample()

    got = decode(encode(ds))

    assert got["t"].dims == ds["t"].dims
    assert got["t"].dtype == np.float32
    assert np.array_equal(got["t"].values, ds["t"].values)
    for name in ("time", "level", "lat", "lon"):
        assert np.array_equal(got[name].values, ds[name].values)
        assert got[name].dtype == ds[name].dtype


def test_the_provenance_survives() -> None:
    """Без провенанса срез из кэша неотличим от среза неизвестно откуда, а
    ответ API обязан назвать источник (docs/DATA_CONTRACT.md §2)."""
    got = decode(encode(_sample()))

    assert got.attrs == {"source": "era5-final", "grid": "0.25deg-global"}
    assert got["t"].attrs["units"] == "K"


def test_the_fill_value_comes_back_a_plain_float() -> None:
    """Известная потеря обхода: numpy-скаляр в атрибутах становится
    питоновским числом. Записано тестом, а не только в докстроке: молчаливое
    изменение типа однажды всплывёт в сравнении атрибутов на публикации."""
    got = decode(encode(_sample()))

    assert np.isnan(got["t"].attrs["_FillValue"])
    assert not isinstance(got["t"].attrs["_FillValue"], np.generic)


def test_a_slice_with_no_levels_needs_no_special_case() -> None:
    """Приземное поле — это тот же Dataset без оси уровней. Кодек про канон не
    знает и знать не должен."""
    ds = _sample().isel(level=0, drop=True)

    got = decode(encode(ds))

    assert got["t"].dims == ("time", "lat", "lon")
    assert np.array_equal(got["t"].values, ds["t"].values)


def test_garbage_does_not_pass_for_a_chunk() -> None:
    """Кэш чистят руками и переносят между машинами; обрезанный файл обязан
    оказаться отказом, а не срезом из мусора."""
    data = encode(_sample())

    with pytest.raises(zipfile.BadZipFile):
        decode(data[: len(data) // 2])

"""Приведение к канону — на синтетике, где источник можно портить по одной вещи.

Настоящие файлы проверяют, что адаптер читает то, что реально приходит
(`test_gfs.py`, `test_ecmwf.py`). Здесь проверяется обратное: что он отвергает
то, чего быть не должно, и что перекладка долготы двигает данные, а не только
подписи осей. На настоящем файле такое не изобразишь — он всего один и он
правильный.
"""

import numpy as np
import pytest
import xarray as xr

from adapters.canonical import Provenance, to_canonical
from adapters.errors import AdapterError
from contracts import canon

RENAMES = {"t2m": "2t", "t": "t"}
PROVENANCE = Provenance(
    source="gfs-analysis",
    source_url="test://message",
    retrieved_at="2026-08-01T07:41:12Z",
    adapter_version="0.0.0-test",
)

INIT = np.datetime64("2026-07-31T00:00:00", "ns")
VALID = np.datetime64("2026-07-31T06:00:00", "ns")


def _message(
    *,
    lon: np.ndarray,
    lat: np.ndarray | None = None,
    values: np.ndarray | None = None,
    name: str = "t2m",
) -> xr.Dataset:
    """Сообщение в том виде, в каком его отдаёт cfgrib: оси `latitude`/`longitude`,
    скалярные `time`, `step` и `valid_time`, время прогона отдельно от времени поля."""
    latitude = canon.LAT if lat is None else lat
    field = np.zeros((latitude.size, lon.size), dtype=np.float32) if values is None else values
    return xr.Dataset(
        {name: (("latitude", "longitude"), field, {"GRIB_startStep": 0, "GRIB_endStep": 6})},
        coords={
            "latitude": latitude,
            "longitude": lon,
            "time": INIT,
            "step": np.timedelta64(6, "h"),
            "valid_time": VALID,
            "heightAboveGround": 2.0,
        },
    )


def _canonical(ds: xr.Dataset) -> xr.Dataset:
    return to_canonical(ds, renames=RENAMES, scales={}, provenance=PROVENANCE)


def test_longitude_shifts_the_data_and_not_only_the_axis() -> None:
    """Ловушка 2: если переписать ось и не двинуть данные, проверки диапазона
    пройдут, а Европа окажется на месте Тихого океана.

    Поле здесь — сама долгота, поэтому вопрос «поехало ли значение вместе с
    координатой» имеет буквальный ответ: после перекладки в узле -179.75
    обязано лежать 180.25, то есть значение, приехавшее с другого конца оси.
    """
    lon = np.round(np.arange(0.0, 360.0, canon.GRID_STEP), 2)
    marked = np.broadcast_to(lon.astype(np.float32), (canon.LAT.size, lon.size))
    out = _canonical(_message(lon=lon, values=marked))

    assert np.array_equal(out["lon"].values, canon.LON)
    assert float(out["2t"].sel(lon=-179.75).isel(time=0, lat=0)) == 180.25
    assert float(out["2t"].sel(lon=0.0).isel(time=0, lat=0)) == 0.0
    assert float(out["2t"].sel(lon=179.75).isel(time=0, lat=0)) == 179.75


def test_time_is_the_valid_time_of_the_message_not_the_start_of_the_run() -> None:
    """Ловушка 1. У сообщения `time` = 00 UTC, `valid_time` = 06 UTC; взять
    первое значит записать шаг +6 ч под меткой прогона."""
    out = _canonical(_message(lon=canon.LON))
    assert out["time"].values.tolist() == [VALID.astype("datetime64[ns]").tolist()]
    assert out.attrs["init_time"] == "2026-07-31T00:00:00Z"
    assert "step" not in out.coords
    assert "valid_time" not in out.coords


def test_latitude_flipped_by_the_source_is_flipped_back() -> None:
    ascending = canon.LAT[::-1].copy()
    out = _canonical(_message(lon=canon.LON, lat=ascending))
    assert np.array_equal(out["lat"].values, canon.LAT)


def test_dimension_order_matches_the_canon() -> None:
    out = _canonical(_message(lon=canon.LON))
    assert out["2t"].dims == ("time", "lat", "lon")


def test_units_and_fill_value_are_set_from_the_canon() -> None:
    out = _canonical(_message(lon=canon.LON))
    assert out["2t"].attrs["units"] == canon.UNITS["2t"]
    assert np.isnan(out["2t"].attrs["_FillValue"])
    assert out["2t"].dtype == np.float32


def test_step_range_survives_into_the_canonical_attributes() -> None:
    """Правило накопления выводится из этих атрибутов; потеряв их, адаптер
    заставил бы `accumulation` гадать по имени источника."""
    out = _canonical(_message(lon=canon.LON))
    assert (out["2t"].attrs["start_step"], out["2t"].attrs["end_step"]) == (0, 6)


def test_scales_convert_units_to_si() -> None:
    lon = canon.LON
    field = np.full((canon.LAT.size, lon.size), 2.0, dtype=np.float32)
    ds = _message(lon=lon, values=field, name="t")
    out = to_canonical(ds, renames=RENAMES, scales={"t": 0.5}, provenance=PROVENANCE)
    assert float(out["t"].isel(time=0, lat=0, lon=0)) == 1.0


def test_a_variable_missing_from_the_table_is_a_refusal() -> None:
    """Иначе поле прошло бы насквозь под именем источника и потерялось бы
    в проверке «все 69 полей на месте» — там смотрят на имена канона."""
    with pytest.raises(AdapterError) as raised:
        _canonical(_message(lon=canon.LON, name="gh"))
    assert raised.value.field == "data_vars"
    assert raised.value.got == ["gh"]


def test_a_foreign_grid_is_refused_with_the_numbers_named() -> None:
    coarse = np.round(np.arange(0.0, 360.0, 0.5), 2)
    lat = np.round(np.arange(90.0, -90.5, -0.5), 2)
    with pytest.raises(AdapterError) as raised:
        _canonical(_message(lon=coarse, lat=lat))
    assert raised.value.field == "lat"
    assert str(raised.value) == (
        "lat: получено '361 nodes 90.0..-90.0', ожидалось '721 nodes 90.0..-90.0'"
    )


def test_a_message_without_valid_time_is_refused() -> None:
    ds = _message(lon=canon.LON).drop_vars("valid_time")
    with pytest.raises(AdapterError) as raised:
        _canonical(ds)
    assert raised.value.field == "valid_time"


def test_source_outside_the_contract_is_refused() -> None:
    """`source` уходит в ответ API, и его значения перечислены в
    docs/DATA_CONTRACT.md §3."""
    with pytest.raises(AdapterError) as raised:
        Provenance(
            source="gfs",
            source_url="test://message",
            retrieved_at="2026-08-01T07:41:12Z",
            adapter_version="0.0.0-test",
        )
    assert raised.value.field == "source"

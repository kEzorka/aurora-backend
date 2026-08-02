"""План загрузки среза — приёмка BACKLOG 1.10.

Приёмка: «все 18 приземных входов Aurora 1.5 собраны; в манифесте видно, какое
поле из какого потока». Первая половина проверяется здесь, вторая — в
`tests/pipeline/test_inputs.py` и `tests/storage/test_manifest.py`.

Сети тут нет: HTTP и ретраи — BACKLOG 1.7. Проверяется решение, принимаемое до
загрузки, и склейка того, что загрузка вернёт.
"""

import numpy as np
import pytest
import xarray as xr

from adapters import ecmwf
from adapters.errors import AdapterError
from adapters.plan import ERA5T_STREAM, Request, assemble, plan
from contracts import canon

NOON = "2026-08-01T00:00:00Z"


def _slice(names: tuple[str, ...], valid_time: str, *, source: str = ecmwf.SOURCE) -> xr.Dataset:
    """Кусок среза так, как его вернёт адаптер: одно время, канонические имена."""
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.full((1, 1, 1), 1.0, dtype=np.float32),
                dims=("time", "lat", "lon"),
                attrs={"units": canon.UNITS[name]},
            )
            for name in names
        },
        coords={
            "time": np.array([valid_time.rstrip("Z")], dtype="datetime64[ns]"),
            "lat": [55.75],
            "lon": [37.5],
        },
        attrs={
            "source": source,
            "init_time": valid_time,
            "kind": "analysis",
            "adapter_version": ecmwf.ADAPTER_VERSION,
        },
    )


def test_all_eighteen_surface_inputs_are_planned() -> None:
    """Главное в приёмке. Пропущенное поле — это не отказ на этапе приёма, а
    батч, в котором Aurora 1.5 недосчитается входа уже на GPU."""
    planned = plan(NOON)

    assert sorted(name for request in planned for name in request.names) == sorted(
        canon.SURFACE_INGESTED_VARS
    )


def test_cloud_cover_goes_to_the_other_stream() -> None:
    """`lcc`/`mcc`/`hcc` нет ни в `ifs/oper`, ни в `ifs/enfo` — проверено по
    настоящим `.index` (`ecmwf.STREAMS`). Запрос к умолчанию вернул бы 404 на
    три поля из восемнадцати."""
    streams = {name: request.stream for request in plan(NOON) for name in request.names}

    assert streams["lcc"] == streams["mcc"] == streams["hcc"] == "aifs-single/0p25/oper"
    assert streams["2t"] == ecmwf.STREAM_DEFAULT


def test_sea_ice_is_asked_for_an_earlier_date() -> None:
    """ERA5T отстаёт примерно на пять суток. Просить `ci` на сегодня — просить
    то, чего в наборе ещё нет, и получить пустой ответ вместо льда."""
    ice = next(request for request in plan(NOON) if "ci" in request.names)

    assert ice.source == "era5t"
    assert ice.stream == ERA5T_STREAM
    assert ice.valid_time == "2026-07-27T00:00:00Z"
    # Задержка — обещание CDS, а не измеренная величина: загрузчик вправе
    # отступить дальше, если срока ещё нет, и план обязан это позволять.
    assert (
        next(
            request for request in plan(NOON, era5t_lag_days=7) if "ci" in request.names
        ).valid_time
        == "2026-07-25T00:00:00Z"
    )


def test_a_stream_is_asked_once_for_all_its_fields() -> None:
    """Пятнадцать полей одного потока — один запрос. Пятнадцать запросов к
    одному и тому же файлу — это пятнадцать скачиваний целого GRIB ради
    байтового диапазона (docs/PIPELINE.md §2)."""
    planned = plan(NOON)

    assert len(planned) == 3
    assert [len(request.names) for request in planned] == [14, 3, 1]


def test_the_order_of_the_plan_is_the_order_of_the_canon() -> None:
    """План уезжает в манифест. Переставить его от запуска к запуску — сделать
    два одинаковых прогона разными на вид."""
    assert plan(NOON) == plan(NOON)
    assert plan(NOON)[0].names[0] == canon.SURFACE_INGESTED_VARS[0]


def test_a_name_outside_the_canon_is_refused() -> None:
    with pytest.raises(AdapterError, match="sithick"):
        plan(NOON, ("2t", "sithick"))


def test_a_time_that_is_not_iso_is_refused() -> None:
    with pytest.raises(AdapterError, match="valid_time"):
        plan("2026-08-01 00:00")


def test_the_slice_is_glued_from_three_downloads() -> None:
    """Собранный срез — это все восемнадцать полей на одном сроке."""
    planned = plan(NOON)
    parts = [
        (request, _slice(request.names, request.valid_time, source=request.source))
        for request in planned
    ]

    slab = assemble(parts, valid_time=NOON)

    assert set(slab.data_vars) == set(canon.SURFACE_INGESTED_VARS)
    assert slab["time"].size == 1
    assert slab["time"].values[0] == np.datetime64("2026-08-01T00:00:00", "ns")
    # Склейка добавляет провенанс, а не заменяет им единицы: поле без `units`
    # не примет валидатор, и потерять их здесь значит уронить прогон на записи.
    assert slab["ci"].attrs["units"] == canon.UNITS["ci"]


def test_the_stale_field_says_when_it_is_really_from() -> None:
    """Перенос вперёд честен ровно до тех пор, пока он виден. `ci` в срезе
    лежит на сроке прогона, а его настоящий срок — в атрибутах поля."""
    planned = plan(NOON)
    parts = [
        (request, _slice(request.names, request.valid_time, source=request.source))
        for request in planned
    ]

    slab = assemble(parts, valid_time=NOON)

    assert slab["ci"].attrs["valid_time"] == "2026-07-27T00:00:00Z"
    assert slab["ci"].attrs["source"] == "era5t"
    assert slab["ci"].attrs["stream"] == ERA5T_STREAM
    # А у полей основного потока — срок прогона, и поток, из которого пришли.
    assert slab["2t"].attrs["valid_time"] == NOON
    assert slab["hcc"].attrs["stream"] == "aifs-single/0p25/oper"


def test_the_slice_has_no_single_source() -> None:
    """Срез склеен из двух источников, и написать на нём одно `source` значило
    бы соврать про лёд. Кто дал поле — в атрибутах поля."""
    planned = plan(NOON)
    parts = [
        (request, _slice(request.names, request.valid_time, source=request.source))
        for request in planned
    ]

    slab = assemble(parts, valid_time=NOON)

    assert "source" not in slab.attrs
    assert slab.attrs["sources"] == ["era5t", "ifs-analysis"]
    assert slab.attrs["init_time"] == NOON


def test_a_field_from_another_date_is_not_carried_forward_silently() -> None:
    """`xr.merge` двух срезов с разным `time` даёт ось на два срока и молчит.
    Дальше сборка батча получает шаг, которого не просила, — и это уже не
    видно нигде."""
    late = Request(ecmwf.SOURCE, ecmwf.STREAM_DEFAULT, "2026-07-31T18:00:00Z", ("2t",))
    parts = [(late, _slice(("2t",), late.valid_time))]

    with pytest.raises(AdapterError, match="time"):
        assemble(parts, valid_time=NOON, names=("2t",))


def test_a_missing_field_is_named() -> None:
    """Отказ называет, чего не хватает: срез без `ci` от полного отличается
    одним полем из восемнадцати, и глазами это не находится."""
    planned = plan(NOON)
    parts = [
        (request, _slice(request.names, request.valid_time, source=request.source))
        for request in planned
        if "ci" not in request.names
    ]

    with pytest.raises(AdapterError, match="ci"):
        assemble(parts, valid_time=NOON)


def test_nothing_downloaded_is_not_an_empty_slice() -> None:
    with pytest.raises(AdapterError, match="parts"):
        assemble([], valid_time=NOON)

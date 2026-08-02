"""Карта `ci` из ERA5T — восемнадцатый приземный вход (BACKLOG 1.10).

Сети и eccodes здесь нет: `retriever` и `reader` подставляются. Проверяется
то, что решается до заявки (какое поле, на какой срок, дошёл ли туда ERA5T) и
после скачивания (имя поля, оси, провенанс) — а не поход в CDS, который в
тестах не случается никогда (docs/PROGRESS.md, «Тесты не ходят в сеть»).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from adapters import ecmwf, era5_grid
from adapters.era5_arco import FINAL, PRELIMINARY
from adapters.errors import AdapterError, NotYetInSourceError
from adapters.plan import ERA5T_NAMES, ERA5T_STREAM
from contracts import canon

#: Срок карты и «сегодня» теста. Между ними двадцать суток: ERA5T с его пятью
#: туда дошёл, а трёх месяцев ещё не прошло — значит ждём метку `era5t`.
MOMENT = datetime(2026, 7, 27, tzinfo=UTC)
NOW = MOMENT + timedelta(days=20)


def _message(name: str = "siconc") -> xr.Dataset:
    """Скачанный GRIB так, как его отдаёт cfgrib: полная сетка ERA5, долгота
    `0..359.75`, широта по убыванию, скалярные `time`/`valid_time`/`surface`.

    Сетка настоящая, 721×1440: `adapters.canonical` сверяет её поэлементно, и
    на обрезанной проверялась бы не та ветка, которая работает в бою.
    """
    stamp = np.datetime64(MOMENT.replace(tzinfo=None), "ns")
    lon = np.round(np.arange(0.0, 360.0, canon.GRID_STEP), 2)
    values = np.tile(np.linspace(0.0, 1.0, lon.size, dtype=np.float32), (canon.LAT.size, 1))
    return xr.Dataset(
        {name: xr.DataArray(values, dims=("latitude", "longitude"))},
        coords={
            "latitude": canon.LAT,
            "longitude": lon,
            "time": stamp,
            "step": np.timedelta64(0, "ns"),
            "surface": np.float64(0.0),
            "valid_time": stamp,
        },
    )


class _Service:
    """Подставной CDS: помнит заявки, кладёт «файл», отдаёт сообщение."""

    def __init__(self, message: xr.Dataset | None = None) -> None:
        self.requests: list[tuple[str, Mapping[str, Any], Path]] = []
        self.read: list[Path] = []
        self._message = message if message is not None else _message()

    def retrieve(self, dataset: str, request: Mapping[str, Any], target: Path) -> None:
        self.requests.append((dataset, dict(request), target))
        target.write_bytes(b"GRIB")

    def reader(self, path: Path) -> xr.Dataset:
        # Читается ровно то, что скачано: подмена пути между заявкой и разбором
        # дала бы вчерашний файл под сегодняшним сроком.
        assert path.read_bytes() == b"GRIB"
        self.read.append(path)
        return self._message


def _read(service: _Service, tmp_path: Path, **kwargs: Any) -> xr.Dataset:
    return era5_grid.read_map(
        "ci",
        MOMENT,
        retriever=service.retrieve,
        reader=service.reader,
        target=tmp_path / "ice.grib",
        now=NOW,
        **kwargs,
    )


@pytest.mark.slow
def test_the_ice_map_lands_in_the_canon(tmp_path: Path) -> None:
    """Главное: восемнадцатое поле приходит в той же форме, что остальные
    семнадцать. Иначе сборка среза (`adapters.plan.assemble`) получит поле с
    чужой сеткой и склеит его молча."""
    service = _Service()

    ice = _read(service, tmp_path, retrieved_at="2026-08-16T00:00:00+00:00")

    assert list(ice.data_vars) == ["ci"]
    assert ice["ci"].dims == ("time", "lat", "lon")
    assert ice["ci"].dtype == np.float32
    assert ice["ci"].attrs["units"] == canon.UNITS["ci"]
    assert np.array_equal(ice["lon"].values, canon.LON)
    assert np.array_equal(ice["lat"].values, canon.LAT)
    assert ice["time"].values[0] == np.datetime64(MOMENT.replace(tzinfo=None), "ns")
    assert ice.attrs["source"] == PRELIMINARY
    assert ice.attrs["grid"] == canon.GRID_NAME
    assert ice.attrs["source_url"].endswith("sea_ice_cover@2026-07-27T00:00:00+00:00")
    assert ice.attrs["adapter_version"] == era5_grid.ADAPTER_VERSION
    assert service.read == [tmp_path / "ice.grib"]


@pytest.mark.slow
def test_the_prime_meridian_does_not_move_the_ice(tmp_path: Path) -> None:
    """Перекладка долготы обязана двигать данные вместе с осью. Лёд — поле, на
    котором ошибка не видна ни одной проверкой на диапазон: 0..1 и слева, и
    справа, и в Тихом океане вместо Гренландии."""
    service = _Service()

    ice = _read(service, tmp_path)

    # В сообщении значение росло с долготой от 0 до 1: узел 0° был нулём,
    # 180° — серединой. После перекладки они обязаны остаться на своих местах.
    assert float(ice["ci"].sel(lon=0.0)[0, 0]) == pytest.approx(0.0)
    assert float(ice["ci"].sel(lon=-180.0)[0, 0]) == pytest.approx(0.5, abs=1e-3)


def test_a_field_that_is_not_taken_from_era5t_never_reaches_the_service(tmp_path: Path) -> None:
    """`2t` есть в ERA5, но берётся из Open Data. Заявка на него стоила бы
    очереди CDS, чтобы узнать про ошибку в маршрутизации."""
    service = _Service()

    with pytest.raises(AdapterError, match="variable"):
        era5_grid.read_map("2t", MOMENT, retriever=service.retrieve, reader=service.reader, now=NOW)

    assert service.requests == []


def test_a_date_the_reanalysis_has_not_reached_is_a_blind_zone(tmp_path: Path) -> None:
    """Срок моложе задержки ERA5T — не сбой, а слепая зона: отдельный тип, из
    которого кэш делает отказ на шесть часов (`cache.origins`). Заявка при этом
    не уходит: очередь CDS ради пустого ответа — это минуты на ничего."""
    service = _Service()
    yesterday = NOW - timedelta(days=1)

    with pytest.raises(NotYetInSourceError, match="новее ERA5T"):
        era5_grid.read_map(
            "ci", yesterday, retriever=service.retrieve, reader=service.reader, now=NOW
        )

    assert service.requests == []


def test_the_loader_may_step_further_back(tmp_path: Path) -> None:
    """Пять суток — обещание CDS, а не измеренная величина. Загрузчик
    (BACKLOG 1.7) отступает дальше, и задержка обязана быть параметром."""
    service = _Service()
    edge = NOW - timedelta(days=6)

    with pytest.raises(NotYetInSourceError):
        era5_grid.read_map(
            "ci", edge, retriever=service.retrieve, reader=service.reader, now=NOW, lag_days=7
        )

    era5_grid.read_map(
        "ci",
        edge,
        retriever=service.retrieve,
        reader=service.reader,
        target=tmp_path / "ice.grib",
        now=NOW,
        lag_days=5,
    )
    assert len(service.requests) == 1


def test_the_request_asks_for_one_hour_the_way_cds_does() -> None:
    """Заявка — это то, что уедет в очередь. Ошибка в ней стоит не отказа, а
    ожидания и пустого файла."""
    request = era5_grid.build_request("ci", MOMENT)

    assert request["variable"] == ["sea_ice_cover"]
    assert request["product_type"] == ["reanalysis"]
    assert (request["year"], request["month"], request["day"]) == (["2026"], ["07"], ["27"])
    assert request["time"] == ["00:00"]
    assert request["data_format"] == "grib"


def test_half_an_hour_is_refused_and_not_rounded() -> None:
    """У ERA5 почасовая сетка. `12:30` CDS разберёт как заявку без единого
    срока и вернёт пустой ответ — а округлить самим значит подменить срок."""
    with pytest.raises(AdapterError, match="whole hour"):
        era5_grid.build_request("ci", MOMENT + timedelta(minutes=30))


def test_a_naive_moment_is_refused() -> None:
    """Ошибка в часовом поясе даёт заявку на соседний срок, и приезжает по ней
    настоящий лёд — просто не тот."""
    with pytest.raises(AdapterError, match="timezone-aware"):
        era5_grid.build_request("ci", MOMENT.replace(tzinfo=None))


@pytest.mark.slow
def test_the_mark_says_which_version_of_the_reanalysis_it_is(tmp_path: Path) -> None:
    """`era5t` входит в ключ кэша, а не только в ответ (docs/CACHE.md §5), и
    граница у карты из CDS та же, что у карты из бакета: одно правило на две
    двери."""
    service = _Service()

    old = era5_grid.read_map(
        "ci",
        MOMENT,
        retriever=service.retrieve,
        reader=service.reader,
        target=tmp_path / "ice.grib",
        now=MOMENT + timedelta(days=400),
    )

    assert old.attrs["source"] == FINAL
    assert _read(service, tmp_path).attrs["source"] == PRELIMINARY


@pytest.mark.slow
def test_a_stranger_in_the_downloaded_file_is_refused(tmp_path: Path) -> None:
    """`sithick` лежит в том же наборе CDS и на той же сетке — это толщина
    льда, другое поле. Пройди оно насквозь под именем `ci`, отличить его от
    сплочённости было бы нечем: обе безымянные доли и метры выглядят числами."""
    service = _Service(_message("sithick"))

    with pytest.raises(AdapterError, match="sithick"):
        _read(service, tmp_path)


def test_the_three_name_tables_say_the_same_thing() -> None:
    """Имён у `ci` три: канон, имя в заявке CDS и имя в скачанном GRIB.
    Разъехавшись, они дают либо отказ CDS про несуществующую переменную, либо
    отказ канона про незнакомое поле — и ни один из них не назовёт причину."""
    assert set(era5_grid.GRIB_NAMES.values()) == set(ERA5T_NAMES) == set(ecmwf.FROM_ERA5T)
    assert era5_grid.CDS_DATASET == ERA5T_STREAM

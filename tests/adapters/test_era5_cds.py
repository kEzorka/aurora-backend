"""Ряды ERA5 из CDS — приёмка BACKLOG 1.6: «ряд в точке за 10 лет получен и
уложен в канон».

В сеть тесты не ходят: `Retriever` подставляется, и проверяется всё, что
случается с ответом после скачивания. Это осознанный предел — что CDS зовёт
столбец со сроком именно `valid_time`, доказывает только документация. Зато
проверено то, что от документации не зависит: неизвестная шапка обязана быть
громким отказом, а не пустым рядом, десять лет часов обязаны доехать до
канона все до одного, а уровневая переменная — не доехать до сети вовсе.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
import xarray as xr

from adapters import era5_cds
from adapters.canonical import Provenance, to_canonical
from adapters.era5_arco import FINAL, PRELIMINARY
from adapters.errors import AdapterError
from contracts import canon

#: Десять лет часов. Границы взяты так, чтобы внутрь попали два високосных
#: года (2012 и 2016): длина ряда, посчитанная как «10 × 8760», разошлась бы
#: с настоящей на 48 часов, и разошлась бы молча.
START = datetime(2010, 1, 1, tzinfo=UTC)
END = datetime(2019, 12, 31, 23, tzinfo=UTC)

#: Москва, округлённая до узла канонической сетки заранее — чтобы тест
#: сравнивал с числом, а не повторял в себе логику `snap`.
MOSCOW = (55.75, 37.5)


def _csv(times: list[datetime], values: list[float], *, name: str = "t2m") -> str:
    """CSV в том виде, в каком его описывает CDS: срок, точка, короткое имя ERA5."""
    lines = [f"valid_time,latitude,longitude,{name}"]
    lines += [
        f"{moment:%Y-%m-%d %H:%M:%S},{MOSCOW[0]},{MOSCOW[1]},{value}"
        for moment, value in zip(times, values, strict=True)
    ]
    return "\n".join(lines) + "\n"


def _hours(start: datetime, end: datetime) -> list[datetime]:
    count = int((end - start) / timedelta(hours=1)) + 1
    return [start + timedelta(hours=step) for step in range(count)]


def _retriever(text: str) -> era5_cds.Retriever:
    def retrieve(dataset: str, request: Mapping[str, Any]) -> str:
        assert dataset == era5_cds.CDS_DATASET
        return text

    return retrieve


def test_a_decade_of_hours_lands_in_the_canon() -> None:
    """Приёмка 1.6 целиком: ряд за десять лет, ось `time`, скалярная точка,
    единицы канона.

    Считается длина ряда, а не «непусто»: потерянный столбец, обрезанный ответ
    и пропущенные високосные сутки одинаково выглядят как «данные пришли».
    """
    times = _hours(START, END)
    values = [float(step % 50) + 250.0 for step in range(len(times))]
    series = era5_cds.read_series(
        "2t",
        *MOSCOW,
        START,
        END,
        retriever=_retriever(_csv(times, values)),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(times) == 87_648  # 3652 суток, включая 2012 и 2016
    assert series["2t"].dims == ("time",)
    assert series["2t"].sizes["time"] == len(times)
    assert series["2t"].dtype == np.float32
    assert series["2t"].attrs["units"] == canon.UNITS["2t"]
    assert float(series["lat"]) == MOSCOW[0] and float(series["lon"]) == MOSCOW[1]
    assert series["time"].values[0] == np.datetime64(START.replace(tzinfo=None), "ns")
    assert series["time"].values[-1] == np.datetime64(END.replace(tzinfo=None), "ns")
    assert np.all(np.diff(series["time"].values) == np.timedelta64(1, "h"))


def test_a_level_variable_never_reaches_the_service() -> None:
    """У датасета нет уровней давления, а `t` — законное имя канона и лежит в
    таблице имён. Проверка по таблице пропустила бы запрос наружу, и отказ CDS
    вернулся бы через очередь длиной в минуты."""
    called: list[str] = []

    def retrieve(dataset: str, request: Mapping[str, Any]) -> str:
        called.append(dataset)
        return ""

    with pytest.raises(AdapterError) as refusal:
        era5_cds.read_series("t", *MOSCOW, START, END, retriever=retrieve)

    assert refusal.value.field == "variable"
    assert not called


def test_an_unknown_column_layout_is_a_refusal_and_not_an_empty_series() -> None:
    """Главный страх этого модуля: имена столбцов взяты из документации. Если
    сервис назовёт столбец иначе, разбор обязан упасть, а не отдать ряд без
    значений — снаружи пустой ряд неотличим от «в этой точке данных нет»."""
    header = "valid_time,latitude,longitude,t2m,d2m\n2010-01-01 00:00:00,55.75,37.5,250.0,240.0\n"

    with pytest.raises(AdapterError) as two_columns:
        era5_cds.parse_csv(header)
    with pytest.raises(AdapterError) as no_time:
        era5_cds.parse_csv("latitude,longitude,t2m\n55.75,37.5,250.0\n")

    assert two_columns.value.field == "columns"
    assert no_time.value.field == "columns"


def test_a_repeated_hour_is_refused_and_not_quietly_sorted() -> None:
    """Повторившийся срок — это склеенные два ответа. Отсортировать и взять
    один молча означало бы выбрать за пользователя, какой из них правда."""
    moment = datetime(2010, 1, 1, tzinfo=UTC)
    text = _csv([moment, moment], [250.0, 251.0])

    with pytest.raises(AdapterError) as refusal:
        era5_cds.parse_csv(text)

    assert refusal.value.field == "time"


def test_an_empty_cell_becomes_a_gap_and_not_a_zero() -> None:
    """Ноль на месте пропуска — это −273 °C в ряду температуры, и заметно это
    только глазами. `nan` дальше ловят валидаторы."""
    times = _hours(START, START + timedelta(hours=2))
    text = _csv(times, [250.0, 251.0, 252.0]).replace(",251.0", ",")

    _, values = era5_cds.parse_csv(text)

    assert np.isnan(values[1])
    assert values[0] == pytest.approx(250.0)


def test_the_point_snaps_to_a_node_of_the_canonical_grid() -> None:
    """Точку выбирает адаптер, а не CDS: иначе в ответе стояли бы координаты, о
    которых мы не договаривались, и ключ кэша разошёлся бы с содержимым.

    Долгота 359.6 и −0.4 — одно место; оба обязаны дать один узел."""
    assert era5_cds.snap(55.7558, 37.6173) == (55.75, 37.5)
    assert era5_cds.snap(0.0, 359.6) == era5_cds.snap(0.0, -0.4)
    assert era5_cds.snap(90.0, -180.0) == (90.0, -180.0)

    with pytest.raises(AdapterError):
        era5_cds.snap(91.0, 0.0)


def test_the_request_names_the_field_the_way_the_service_does() -> None:
    """Канон снаружи, полные имена ERA5 в запросе — таблица одна на оба
    адаптера ERA5 (`era5_arco.SOURCE_NAMES`)."""
    request = era5_cds.build_request("2t", 55.7558, 37.6173, START, END)

    assert request["variable"] == ["2m_temperature"]
    assert request["location"] == {"latitude": 55.75, "longitude": 37.5}
    assert request["date"] == ["2010-01-01/2019-12-31"]
    assert request["data_format"] == "csv"

    with pytest.raises(AdapterError):
        era5_cds.build_request("2t", *MOSCOW, END, START)


def test_the_preliminary_mark_covers_the_whole_series() -> None:
    """Версию решает самый свежий срок ряда: один предварительный час на конце
    делает предварительным весь ряд.

    Иначе строка в кэше обещала бы финальный ERA5 там, где лежит ERA5T. Цена
    записана в `read_series`: инвалидация ищет точным сравнением, поэтому
    кэшировать такие ряды придётся кусками по границе трёх месяцев."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    times = _hours(START, START + timedelta(hours=1))
    old = era5_cds.read_series(
        "2t", *MOSCOW, START, END, retriever=_retriever(_csv(times, [250.0, 251.0])), now=now
    )

    fresh_times = _hours(now - timedelta(days=2), now - timedelta(days=2, hours=-1))
    fresh = era5_cds.read_series(
        "2t",
        *MOSCOW,
        START,
        END,
        retriever=_retriever(_csv(fresh_times, [250.0, 251.0])),
        now=now,
    )

    assert old.attrs["source"] == FINAL
    assert fresh.attrs["source"] == PRELIMINARY


@pytest.mark.slow
def test_the_provenance_is_the_same_set_as_for_maps() -> None:
    """Ряд идёт мимо `adapters.canonical` — сетку он не проходит по устройству.
    Значит, атрибуты собираются вторым местом, и разъехаться они обязаны не
    молча: набор ключей сверяется с тем, что кладёт общий путь приведения."""
    lon = np.round(np.arange(0.0, 360.0, canon.GRID_STEP), 2)
    stamp = np.datetime64("2010-01-01T00:00:00", "ns")
    message = xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.zeros((canon.LAT.size, lon.size), np.float32))},
        coords={"latitude": canon.LAT, "longitude": lon, "time": stamp, "valid_time": stamp},
    )
    grid_map = to_canonical(
        message,
        renames={"t2m": "2t"},
        scales={},
        provenance=Provenance(
            source=FINAL,
            source_url="test://map",
            retrieved_at="2026-01-01T00:00:00Z",
            adapter_version="0.0.0-test",
        ),
    )

    times = _hours(START, START + timedelta(hours=1))
    series = era5_cds.read_series(
        "2t",
        *MOSCOW,
        START,
        END,
        retriever=_retriever(_csv(times, [250.0, 251.0])),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert set(series.attrs) == set(grid_map.attrs)
    assert series.attrs["grid"] == grid_map.attrs["grid"] == canon.GRID_NAME
    assert series.attrs["kind"] == "analysis"
    assert series.attrs["init_time"] == "2010-01-01T00:00:00Z"

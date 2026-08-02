"""Адаптер ERA5 (ARCO) — приёмка BACKLOG 1.5.

Критерий дословно: «срез читается напрямую из бакета без скачивания архива».
Доказывается он тут порчей: архив пишется с нарезкой по сроку, чанки чужих
сроков забиваются мусором — и если запрошенный срок всё равно читается
правильно, значит чужих чанков модуль не трогал (`tests/fixtures/arco.py`).

В сеть тесты не ходят: `open_archive` открывает локальный каталог тем же
вызовом, что и `gs://`. Что переменные в бакете названы так, как в `RENAMES`,
здесь не проверяется и проверено быть не может (см. модуль).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from adapters.era5_arco import (
    FINAL,
    FINAL_AFTER,
    PRELIMINARY,
    RENAMES,
    covers,
    open_archive,
    read_period,
    read_slice,
    source_version,
)
from adapters.errors import AdapterError, NotYetInSourceError
from contracts import canon
from tests.fixtures.arco import LEVELS, STEPS, build, spoil

NOW = datetime(2020, 12, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def surface(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("arco") / "sfc.zarr", "2m_temperature")


@pytest.fixture(scope="module")
def upper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("arco") / "pl.zarr", "temperature", STEPS[:1], LEVELS)


def _spoiled(surface: Path, root: Path) -> Path:
    """Копия архива, у которой целы чанки только второго срока."""
    shutil.copytree(surface, root)
    spoil(root, "2m_temperature", keep=1)
    return root


def test_a_slice_comes_out_canonical(surface: Path) -> None:
    """Срез из бакета отличается от среза из GRIB только дорогой сюда: дальше
    по конвейеру он обязан быть неотличим — те же оси, порядок, единицы."""
    got = read_slice(open_archive(surface), "2t", STEPS[1], now=NOW)

    assert list(got.data_vars) == ["2t"]
    assert got["2t"].dims == ("time", "lat", "lon")
    assert got["2t"].dtype == np.float32
    assert got["2t"].attrs["units"] == canon.UNITS["2t"]
    assert np.array_equal(got["lat"].values, canon.LAT)
    assert np.array_equal(got["lon"].values, canon.LON)
    assert got["time"].values[0] == np.datetime64(STEPS[1].replace(tzinfo=None), "ns")
    assert got.attrs["grid"] == canon.GRID_NAME


def test_a_period_stays_lazy_for_the_monthly_builder(surface: Path) -> None:
    archive = open_archive(surface, chunks={})

    got = read_period(archive, ("2t",), STEPS[0], STEPS[-1], now=NOW)

    assert got["2t"].dims == ("time", "lat", "lon")
    assert got["2t"].chunks is not None
    assert got.sizes["time"] == len(STEPS)
    assert got["time"].values[0] == np.datetime64("2020-06-01T00", "ns")
    assert got["2t"].isel(time=2, lat=0, lon=720).compute().item() == pytest.approx(2000.0)


def test_the_longitude_moves_with_its_data(surface: Path) -> None:
    """Гринвич обязан остаться Гринвичем. Значение в фикстуре равно номеру
    точки по долготе исходной оси `0..359.75`; после перекладки в `-180..179.75`
    оно должно найтись там же на земном шаре, а не на прежнем месте массива."""
    got = read_slice(open_archive(surface), "2t", STEPS[0], now=NOW)

    assert got["2t"].sel(lon=0.0).values[0][0] == pytest.approx(0.0)
    # 100° в. д. — 400-й узел исходной оси; в канонической он лежит правее нуля.
    assert got["2t"].sel(lon=100.0).values[0][0] == pytest.approx(400.0)
    # 100° з. д. — это 260° исходной оси, узел 1040. Перепиши перекладку как
    # переименование оси, и здесь окажется 400 вместо 1040.
    assert got["2t"].sel(lon=-100.0).values[0][0] == pytest.approx(1040.0)


def test_only_the_requested_step_leaves_the_archive(surface: Path, tmp_path: Path) -> None:
    """Сам критерий приёмки. Чанки чужих сроков — мусор; срез читается верно,
    значит их не читали. Порядок в `_slice` тут и проверяется: `.load()` до
    отбора срока превратил бы это в чтение всего архива."""
    root = _spoiled(surface, tmp_path / "sfc.zarr")

    got = read_slice(open_archive(root), "2t", STEPS[1], now=NOW)

    assert got["2t"].sel(lon=0.0).values[0][0] == pytest.approx(1000.0)


def test_the_spoiled_steps_are_really_unreadable(surface: Path, tmp_path: Path) -> None:
    """Страховка к предыдущему тесту: если мусор читается как данные, тот тест
    ничего не доказывает и проходит при любом поведении адаптера."""
    root = _spoiled(surface, tmp_path / "sfc.zarr")

    # Тип отказа выбирает кодек Zarr, а не мы: сейчас это `RuntimeError`
    # («Zstd decompression error»), при другом сжатии будет `ValueError`.
    # Проверяется здесь не он, а то, что мусор не проходит за данные.
    with pytest.raises((RuntimeError, ValueError)):
        read_slice(open_archive(root), "2t", STEPS[0], now=NOW)


def test_the_canon_takes_thirteen_levels_of_thirty_seven(upper: Path) -> None:
    """Уровни канона и в том порядке, в каком их ждёт модель. Чужие уровни
    архива не должны доехать ни одним значением: `level` — ось, и лишний узел
    в ней сдвигает всё, что дальше собирает батч."""
    got = read_slice(open_archive(upper), "t", STEPS[0], now=NOW)

    assert got["t"].dims == ("time", "level", "lat", "lon")
    assert tuple(int(v) for v in got["level"].values) == canon.PRESSURE_LEVELS


def test_a_date_newer_than_the_archive_is_a_gap_and_not_a_failure(surface: Path) -> None:
    """Слепая зона реанализа — нормальное состояние, а не сбой: ARCO отстаёт
    от сегодня (docs/PIPELINE.md §1). Отдельный тип нужен кэшу, чтобы отличить
    «данных ещё нет» от «источник упал», — первое запоминается, второе нет."""
    with pytest.raises(NotYetInSourceError):
        read_slice(open_archive(surface), "2t", STEPS[-1] + timedelta(hours=1), now=NOW)


def test_a_date_before_the_archive_is_an_adapter_error(surface: Path) -> None:
    """1939 год ждать бесполезно: ERA5 начинается с 1940 (`contracts.canon`).
    Запомнить такой отказ на шесть часов, как слепую зону, было бы неверно —
    он вечный, и ответ на него другой."""
    with pytest.raises(AdapterError):
        read_slice(open_archive(surface), "2t", STEPS[0] - timedelta(hours=1), now=NOW)


def test_an_unknown_variable_is_refused_by_name(surface: Path) -> None:
    """Поле, которого нет в таблице имён, — отказ, а не пустой ответ."""
    with pytest.raises(AdapterError) as refusal:
        read_slice(open_archive(surface), "blh", STEPS[0], now=NOW)

    assert refusal.value.field == "variable"


def test_the_names_cover_what_the_model_eats() -> None:
    """Таблица имён обязана покрывать вход Aurora 1.5 целиком, кроме того, что
    в ERA5 не лежит: `insolation` модель считает сама, `lcc`/`mcc`/`hcc` берутся
    отсюда же (1.10). Иначе нехватка поля вскроется на сборке батча — за
    несколько часов до неё уже потраченных на выкачивание."""
    covered = set(RENAMES.values())

    assert set(canon.SURFACE_INGESTED_VARS) <= covered
    assert set(canon.ATMOS_VARS) <= covered


def test_the_source_version_turns_final_at_three_months() -> None:
    """ERA5T переписывается финальным ERA5 задним числом (docs/STORAGE.md §5).
    Версия входит в ключ кэша, поэтому спрашивается до чтения и по возрасту
    срока, а не по тому, что приехало."""
    edge = NOW - FINAL_AFTER

    assert source_version(edge, now=NOW) == FINAL
    assert source_version(edge + timedelta(seconds=1), now=NOW) == PRELIMINARY


def test_the_preliminary_mark_reaches_the_attributes(surface: Path) -> None:
    """Пользователь обязан узнать, что значение может измениться задним числом
    (docs/API_CONTRACT.md §6). Узнаёт он это из атрибутов среза — больше
    неоткуда."""
    got = read_slice(open_archive(surface), "2t", STEPS[0], now=STEPS[0] + timedelta(days=1))

    assert got.attrs["source"] == PRELIMINARY


def test_the_archive_tells_its_own_edges(surface: Path) -> None:
    """Край архива читается из архива, а не считается как «сегодня минус пять
    суток»: отставание реанализа плавает, и константа означала бы отказ в
    данных, которые уже лежат в бакете."""
    first, last = covers(open_archive(surface))

    assert (first, last) == (STEPS[0], STEPS[-1])

"""Раскладка: бюджет прогона, форма чанков и шардов, состав слоя.

Проверяется форма хранилища, а не запись в него: запись — 2.1 и 2.2. Ошибка
в форме дороже ошибки в записи, потому что переписывать придётся всё уже
записанное (docs/BACKLOG.md, оговорка 3).
"""

import pytest

from contracts import canon
from storage.layout import (
    LAYOUT_A,
    LAYOUT_B,
    RUN_BUDGET_BYTES,
    Chunking,
    layer_bytes,
    layer_path,
    run_bytes,
    select_layer,
)
from tests.helpers import canonical_dataset, layer_dataset


def test_whole_run_fits_the_budget() -> None:
    """Прогон целиком — 6-часовой слой плюс часовой — не больше 18 ГБ."""
    assert run_bytes() <= RUN_BUDGET_BYTES


def test_previous_run_weighs_a_tenth_of_the_current_one() -> None:
    """docs/STORAGE.md §2: прошлый прогон — 1.3 ГБ. Держать его целиком нельзя:
    два шестичасовых слоя это 30 ГБ при закреплённом ядре в 40."""
    assert layer_bytes(canon.LAYERS["previous"]) == pytest.approx(1.33e9, rel=0.02)


def test_pinned_forecast_layers_fit_the_run_budget() -> None:
    """Закреплено прогнозом три слоя: текущий шестичасовой, текущий часовой и
    прошлый прогон. Вместе 18.7 ГБ — это меньше половины ядра в 40 ГБ, где ещё
    лежат анализ, месячные средние и последние 30 суток (docs/STORAGE.md §2)."""
    assert run_bytes() + layer_bytes(canon.LAYERS["previous"]) <= 19 * 10**9


def test_layer_sizes_match_the_documented_numbers() -> None:
    """docs/STORAGE.md §1: 15.0 ГБ + 2.4 ГБ = 17.4 ГБ."""
    assert layer_bytes(canon.LAYERS["coarse"]) == pytest.approx(15.0e9, rel=0.01)
    assert layer_bytes(canon.LAYERS["hourly"]) == pytest.approx(2.4e9, rel=0.01)
    assert run_bytes() == pytest.approx(17.4e9, rel=0.01)


def test_hourly_layer_is_exactly_eight_variables() -> None:
    """Девятая переменная в часовом слое — это +300 МБ на прогон и другая
    раскладка: слой лежит одним массивом по оси переменных (canon.HOURLY_VARS)."""
    hourly = canon.LAYERS["hourly"]
    assert len(hourly.surface_vars) == 8
    assert hourly.atmos_vars == ()
    assert hourly.fields == 8 * 72


def test_layer_paths_are_the_documented_ones() -> None:
    """Имена каталогов — интерфейс: по ним CLI определяет слой."""
    assert layer_path("/data", "coarse").as_posix() == "/data/forecast/current/coarse"
    assert layer_path("/data", "hourly").as_posix() == "/data/forecast/current/hourly"
    assert layer_path("/data", "analysis").as_posix() == "/data/analysis/recent"


def test_unknown_layer_has_no_path() -> None:
    with pytest.raises(ValueError, match="layer"):
        layer_path("/data", "hourlyish")


def test_select_keeps_exactly_the_layer_fields() -> None:
    ds = canonical_dataset(times=2)
    selected = select_layer(ds, canon.LAYERS["coarse"])
    assert set(selected.data_vars) == set(canon.SURFACE_STORED_VARS) | set(canon.ATMOS_VARS)


def test_select_drops_fields_the_layer_does_not_own() -> None:
    """У адаптера полей больше, чем у слоя: `tp` он отдаёт наравне с `tp_1h`.
    Лишнее не ошибка чтения — но записать его в часовой слой значит сломать
    и бюджет, и раскладку, поэтому оно отсекается на входе в запись."""
    ds = canonical_dataset(times=2, surface_vars=(*canon.HOURLY_VARS, "tp"), atmos_vars=())
    selected = select_layer(ds, canon.LAYERS["hourly"])
    assert set(selected.data_vars) == set(canon.HOURLY_VARS)


def test_select_refuses_a_layer_with_a_field_missing() -> None:
    """Тихо записать слой без переменной нельзя: читатель увидит не «нет поля»,
    а прогноз, в котором ветра не было."""
    ds = layer_dataset(canon.LAYERS["hourly"]).drop_vars("i10fg")
    with pytest.raises(ValueError, match="i10fg"):
        select_layer(ds, canon.LAYERS["hourly"])


def test_chunk_is_the_size_of_the_query_it_serves() -> None:
    """Раскладка A — одна карта на чанк (4.15 МБ), B — весь ряд в точке."""
    assert LAYOUT_A.chunk_bytes == pytest.approx(4.15e6, rel=0.01)
    assert LAYOUT_B.chunk == (1460, 1, 4, 4)


def test_shards_stay_in_the_file_size_band() -> None:
    """Шард — единица файла: мельче — миллионы файлов, крупнее — незачем."""
    for layout in (LAYOUT_A, LAYOUT_B):
        assert 20e6 <= layout.shard_bytes <= 200e6, layout.name


def test_shard_is_a_whole_number_of_chunks() -> None:
    """Zarr v3 требует, чтобы шард делился на чанки нацело."""
    for layout in (LAYOUT_A, LAYOUT_B):
        for shard, chunk in zip(layout.shard, layout.chunk, strict=True):
            assert shard % chunk == 0, layout.name


def test_point_series_layout_reads_one_chunk_per_year() -> None:
    """Смысл раскладки B: ряд в точке за год — один чанк, а не 1460 карт."""
    assert LAYOUT_B.chunk[0] == 1460
    year_of_maps = Chunking(name="a", chunk=(1, 1, 721, 1440), shard=(12, 1, 721, 1440))
    assert LAYOUT_B.chunk_bytes < year_of_maps.chunk_bytes / 40

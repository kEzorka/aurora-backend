"""Канон — это контракт, а не настройки. Числа здесь проверяются поэлементно."""

import numpy as np

from contracts import canon


def test_latitude_runs_from_north_to_south() -> None:
    assert canon.LAT.shape == (721,)
    assert canon.LAT[0] == 90.0
    assert canon.LAT[-1] == -90.0
    assert np.all(np.diff(canon.LAT) < 0)


def test_longitude_is_minus180_to_179_75() -> None:
    assert canon.LON.shape == (1440,)
    assert canon.LON[0] == -180.0
    assert canon.LON[-1] == 179.75
    assert np.all(np.diff(canon.LON) > 0)


def test_grid_holds_1038240_points() -> None:
    assert canon.GRID_SHAPE == (721, 1440)
    assert canon.GRID_SHAPE[0] * canon.GRID_SHAPE[1] == 1_038_240


def test_pressure_levels_match_domain_doc_elementwise() -> None:
    assert canon.PRESSURE_LEVELS == (
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    )


def test_sixty_nine_fields_per_step() -> None:
    assert len(canon.SURFACE_VARS) == 4
    assert len(canon.ATMOS_VARS) * len(canon.PRESSURE_LEVELS) == 65
    assert canon.FIELDS_PER_STEP == 69


def test_units_are_si() -> None:
    assert canon.UNITS["2t"] == "K"
    assert canon.UNITS["msl"] == "Pa"
    assert canon.UNITS["z"] == "m2 s-2"
    assert canon.UNITS["q"] == "kg kg-1"


def test_ten_days_is_forty_six_hour_steps() -> None:
    assert canon.STEP_HOURS == 6
    assert canon.FORECAST_STEPS == 40
    assert canon.STEP_HOURS * canon.FORECAST_STEPS == 240


def test_every_canonical_variable_has_units() -> None:
    for name in canon.SURFACE_VARS + canon.ATMOS_VARS + canon.STATIC_VARS:
        assert name in canon.UNITS, f"{name}: единицы не объявлены"


def test_provenance_sources_are_the_ones_the_api_may_return() -> None:
    assert set(canon.SOURCES) == {
        "aurora-forecast",
        "ifs-analysis",
        "gfs-analysis",
        "era5t",
        "era5-final",
        "climatology",
    }

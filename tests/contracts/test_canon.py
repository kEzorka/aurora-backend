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


def test_static_and_pressure_level_names_do_not_collide() -> None:
    """Геопотенциал поверхности и геопотенциал на уровнях — два разных поля.

    Общее имя `z` в одном Dataset означало бы, что одно из них молча затирает
    другое, а проверка полноты набора полей схлопывает их в одно имя.
    """
    assert set(canon.STATIC_VARS) & set(canon.ATMOS_VARS) == set()
    assert set(canon.STATIC_VARS) & set(canon.SURFACE_VARS) == set()
    assert canon.AURORA_STATIC_NAMES["z_surf"] == "z"
    assert set(canon.AURORA_STATIC_NAMES) == set(canon.STATIC_VARS)


def test_pressure_geopotential_range_does_not_cover_orography() -> None:
    """Иначе орография, записанная вместо уровня 50 гПа, пройдёт проверку."""
    assert canon.PHYSICAL_RANGES["z"][1] > canon.PHYSICAL_RANGES["z_surf"][1] * 3


def test_provenance_sources_are_the_ones_the_api_may_return() -> None:
    assert set(canon.SOURCES) == {
        "aurora-forecast",
        "ifs-analysis",
        "gfs-analysis",
        "era5t",
        "era5-final",
        "climatology",
    }

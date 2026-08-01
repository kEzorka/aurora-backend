"""Уровень «структура»: оси, размерности, полнота набора полей.

Отрицательные тесты здесь важнее положительных: валидатор, который никогда
не срабатывал, скорее всего не работает вовсе (docs/TESTING.md).
"""

import numpy as np

from contracts import canon
from tests.helpers import canonical_dataset, latitude_profile
from validators.result import Check
from validators.structure import check_structure


def failure(checks: list[Check], name: str) -> Check:
    matching = [c for c in checks if c.name == name]
    assert matching, f"проверка {name} не выполнялась вовсе"
    failed = [c for c in matching if not c.passed]
    assert failed, f"проверка {name} прошла, а должна была отвергнуть данные"
    return failed[0]


def test_canonical_dataset_passes_every_structural_check() -> None:
    checks = check_structure(canonical_dataset())
    assert [c.message for c in checks if not c.passed] == []


def test_flipped_latitude_is_rejected() -> None:
    ds = canonical_dataset()
    flipped = ds.assign_coords(lat=ds["lat"].values[::-1])
    assert failure(check_structure(flipped), "lat_descending")


def test_zero_to_360_longitude_is_rejected() -> None:
    """Вторая ловушка docs/DOMAIN.md §6: GFS отдаёт 0..360, молча и без ошибки."""
    ds = canonical_dataset()
    shifted = ds.assign_coords(lon=np.round(np.arange(0.0, 360.0, 0.25), 2))
    check = failure(check_structure(shifted), "lon_range")
    assert check.details["got"] == (0.0, 359.75)
    assert check.details["expected"] == (-180.0, 179.75)


def test_missing_fields_are_named() -> None:
    ds = canonical_dataset(atmos_vars=("t", "u"))
    check = failure(check_structure(ds), "fields_present")
    assert "q" in check.message
    assert "v" in check.message


def test_duplicate_times_are_rejected() -> None:
    ds = canonical_dataset(times=2)
    duplicated = ds.assign_coords(
        time=np.array(["2026-08-01T00", "2026-08-01T00"], dtype="datetime64[ns]")
    )
    assert failure(check_structure(duplicated), "time_unique")


def test_gap_in_the_time_axis_is_rejected() -> None:
    ds = canonical_dataset(times=3)
    times = ds["time"].values.copy()
    times[2] = times[1] + np.timedelta64(18, "h")
    check = failure(check_structure(ds.assign_coords(time=times)), "time_regular")
    assert "18" in check.message


def test_wrong_grid_shape_is_rejected() -> None:
    ds = canonical_dataset().isel(lat=slice(0, 100))
    check = failure(check_structure(ds), "grid_shape")
    assert check.details["expected"] == (721, 1440)
    assert check.details["got"] == (100, 1440)


def test_levels_out_of_order_are_rejected() -> None:
    ds = canonical_dataset()
    reversed_levels = ds.assign_coords(level=ds["level"].values[::-1])
    check = failure(check_structure(reversed_levels), "levels_match")
    assert check.details["expected"][0] == 50


def test_uneven_step_is_rejected_without_rounding() -> None:
    """Шаг 6 ч 1 мин — это сдвиг времени, а не шаг 6 ч, округлённый вниз."""
    ds = canonical_dataset(times=2)
    times = ds["time"].values.copy()
    times[1] = times[0] + np.timedelta64(6 * 60 + 1, "m")
    assert failure(check_structure(ds.assign_coords(time=times)), "time_regular")


def test_static_fields_are_only_required_when_asked_for() -> None:
    ds = canonical_dataset()
    assert [c for c in check_structure(ds) if not c.passed] == []
    check = failure(check_structure(ds, expect_static=True), "fields_present")
    assert "lsm" in check.message


def test_missing_orography_is_caught_although_pressure_z_is_present() -> None:
    """`z` на уровнях давления и `z` статический — разные поля.

    Если бы канон звал их одинаково, набор required схлопывал бы их в одно имя
    и срез без орографии проходил бы проверку с expect_static=True.
    """
    ds = canonical_dataset()
    ny, nx = canon.GRID_SHAPE
    with_partial_static = ds.assign(
        {
            "lsm": (("lat", "lon"), latitude_profile(0.5, (ny, nx), ny)),
            "slt": (("lat", "lon"), latitude_profile(3.0, (ny, nx), ny)),
        }
    )
    assert "z" in with_partial_static.data_vars
    check = failure(check_structure(with_partial_static, expect_static=True), "fields_present")
    assert "z_surf" in check.message

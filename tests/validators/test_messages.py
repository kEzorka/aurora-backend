"""Уровень «сообщения»: то, что cfgrib потерял по дороге (ADDENDUM-01 §5).

cfgrib молча выбрасывает сообщения, которые не смог уложить в один Dataset:
дубликат по (переменная, уровень, срок) не даёт ни исключения, ни warning —
поле просто одно вместо двух. Дальше это выглядит как полный набор с
устаревшими значениями, и ни один валидатор набора такого не видит: там всё
на месте и всё в диапазоне.

Файл с дубликатом собирается конкатенацией фикстуры с самой собой. Это не
синтетика: байты настоящие, а склейка сообщений — ровно то, что делает
загрузчик, когда качает поля по одному и пишет в один файл.
"""

from pathlib import Path

import pytest

pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")

import xarray as xr

from adapters import grib
from validators import validate
from validators.messages import check_messages, count_fields, count_messages

GRIB = Path(__file__).resolve().parents[1] / "fixtures" / "grib"


def _failures(path: Path) -> dict[str, str]:
    try:
        ds: xr.Dataset | None = grib.open_message(path)
    except Exception:
        ds = None
    return {c.name: c.message for c in check_messages(path, ds) if not c.passed}


def test_intact_message_passes() -> None:
    with grib.open_message(GRIB / "ecmwf_2t_6h.grib2") as ds:
        checks = check_messages(GRIB / "ecmwf_2t_6h.grib2", ds)
    assert [c.name for c in checks if not c.passed] == []
    assert all(c.level == "grib" for c in checks)


def test_message_count_is_read_without_cfgrib() -> None:
    assert count_messages(GRIB / "ecmwf_2t_6h.grib2") == 1
    assert count_messages(GRIB / "gfs_t2m.grib2") == 1


def test_one_field_per_message_for_a_single_level_message() -> None:
    with grib.open_message(GRIB / "ecmwf_2t_6h.grib2") as ds:
        assert count_fields(ds) == 1


def test_level_dimension_counts_as_many_fields_as_levels() -> None:
    """Поле на 13 уровнях — это 13 сообщений, а не одно."""
    with grib.open_message(GRIB / "ecmwf_t850_6h.grib2") as ds:
        # В фикстуре уровень скалярный (одно сообщение); три уровня выглядели
        # бы вот так — и стоили бы трёх сообщений.
        stacked = ds.drop_vars("isobaricInhPa").expand_dims({"isobaricInhPa": 3}, axis=0)
    assert count_fields(stacked) == 3


def test_silently_dropped_duplicate_is_caught(tmp_path: Path) -> None:
    """Два сообщения на входе, одно поле на выходе — и cfgrib молчит."""
    doubled = tmp_path / "doubled.grib2"
    doubled.write_bytes((GRIB / "ecmwf_2t_6h.grib2").read_bytes() * 2)

    assert count_messages(doubled) == 2
    with grib.open_message(doubled) as ds:
        assert count_fields(ds) == 1  # вот она, потеря
        failed = {c.name: c.message for c in check_messages(doubled, ds) if not c.passed}

    assert "messages_parsed" in failed
    assert "2" in failed["messages_parsed"] and "1" in failed["messages_parsed"]


def test_truncated_file_fails_with_a_readable_verdict(tmp_path: Path) -> None:
    """Обрезанный файл обязан дать вердикт, а не трассировку из eccodes."""
    intact = (GRIB / "gfs_t2m.grib2").read_bytes()
    truncated = tmp_path / "truncated.grib2"
    truncated.write_bytes(intact[: len(intact) // 2])

    failed = _failures(truncated)
    assert set(failed) == {"grib_readable"}
    assert "truncated.grib2" in failed["grib_readable"]


def test_validate_runs_the_grib_level_when_given_the_file(tmp_path: Path) -> None:
    """Потерянное сообщение обязано валить общий вердикт, а не только свой уровень."""
    doubled = tmp_path / "doubled.grib2"
    doubled.write_bytes((GRIB / "ecmwf_2t_6h.grib2").read_bytes() * 2)
    with grib.open_message(doubled) as ds:
        report = validate(ds, levels=["grib"], grib_path=doubled)
    assert report.ok is False
    assert {c.name for c in report.failures()} == {"messages_parsed"}


def test_empty_file_is_not_a_valid_grib(tmp_path: Path) -> None:
    """Ноль сообщений — это скачанная страница с ошибкой, а не пустой прогноз."""
    empty = tmp_path / "empty.grib2"
    empty.write_bytes(b"")
    assert "grib_readable" in _failures(empty)

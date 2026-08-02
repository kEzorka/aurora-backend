"""Агрегатор: четыре уровня, один вердикт, один validation.json."""

import json
from pathlib import Path

import pytest
import xarray as xr

from contracts import canon
from tests.helpers import canonical_dataset, layer_dataset
from validators import RejectedError, raise_if_rejected, validate, write_report


def _plausible() -> xr.Dataset:
    return canonical_dataset()


def test_plausible_slice_passes_all_four_levels() -> None:
    report = validate(_plausible())
    assert {c.level for c in report.checks} == {"structure", "semantics", "physics", "sanity"}
    assert report.ok, [c.message for c in report.failures()]


def test_a_single_broken_unit_makes_the_whole_report_fail() -> None:
    ds = _plausible()
    ds["2t"].attrs["units"] = "degC"
    report = validate(ds)
    assert report.ok is False
    assert any(c.details.get("field") == "2t" for c in report.failures())


def test_levels_can_be_restricted() -> None:
    report = validate(_plausible(), levels=["structure"])
    assert {c.level for c in report.checks} == {"structure"}


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="levels"):
        validate(_plausible(), levels=["vibes"])


def test_grib_level_without_a_file_is_refused() -> None:
    """Иначе отчёт выглядит полным, а самой ранней проверки в нём нет."""
    with pytest.raises(ValueError, match="grib_path"):
        validate(_plausible(), levels=["grib"])


def test_hourly_layer_passes_only_when_it_names_itself() -> None:
    """Часовой слой — 8 полей и шаг 1 ч. Без имени слоя он неотличим от
    шестичасового прогноза, у которого потеряли 82 поля."""
    ds = layer_dataset(canon.LAYERS["hourly"])
    assert validate(ds, layer="hourly").ok
    default = validate(ds)
    assert default.ok is False
    assert {c.name for c in default.failures()} == {"fields_present", "time_regular"}


def test_six_hourly_data_written_as_the_hourly_layer_is_rejected() -> None:
    """Слой объявлен часовым, а сроки идут через шесть часов — сроки перепутаны,
    и любая сумма по времени после этого врёт (docs/DATA_CONTRACT.md §4)."""
    six_hourly = canonical_dataset(
        times=3,
        surface_vars=canon.HOURLY_VARS,
        atmos_vars=(),
        step_hours=canon.STEP_HOURS,
    )
    report = validate(six_hourly, layer="hourly", levels=["structure"])
    assert report.ok is False
    assert [c.name for c in report.failures()] == ["time_regular"]


def test_unknown_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="layer"):
        validate(canonical_dataset(), layer="hourlyish")


def test_report_is_written_next_to_the_artifact(tmp_path: Path) -> None:
    report = validate(_plausible())
    path = write_report(report, tmp_path / "validation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert len(payload["checks"]) == len(report.checks)


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    """validation.json пишется через .tmp: половина отчёта хуже отсутствия."""
    write_report(validate(_plausible()), tmp_path / "validation.json")
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["validation.json"]


def test_rejection_raises_with_the_failures_attached() -> None:
    ds = _plausible()
    ds["2t"].attrs["units"] = "degC"
    with pytest.raises(RejectedError) as excinfo:
        raise_if_rejected(validate(ds))
    assert "2t" in str(excinfo.value)
    assert excinfo.value.report.failures()

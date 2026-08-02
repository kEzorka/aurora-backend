"""Единицы и три вида времени — ловушки 1 и 3 из docs/DOMAIN.md §6."""

import numpy as np
import pandas as pd

from tests.helpers import canonical_dataset
from validators.semantics import check_semantics


def test_canonical_dataset_passes_semantics() -> None:
    assert [c.name for c in check_semantics(canonical_dataset()) if not c.passed] == []


def test_celsius_units_are_rejected_naming_the_variable() -> None:
    """Третья ловушка docs/DOMAIN.md §6."""
    ds = canonical_dataset()
    ds["2t"].attrs["units"] = "degC"
    failure = next(c for c in check_semantics(ds) if not c.passed)
    assert failure.details["field"] == "2t"
    assert failure.details["got"] == "degC"
    assert failure.details["expected"] == "K"


def test_missing_units_attribute_is_rejected() -> None:
    ds = canonical_dataset()
    del ds["msl"].attrs["units"]
    failure = next(c for c in check_semantics(ds) if not c.passed)
    assert failure.details["field"] == "msl"


def test_timezone_aware_time_axis_is_rejected() -> None:
    ds = canonical_dataset()
    aware = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
    ds = ds.assign_coords(time=aware)
    checks = check_semantics(ds)
    assert any(c.name == "time_naive_utc" and not c.passed for c in checks)


def test_valid_time_must_equal_init_plus_lead() -> None:
    ds = canonical_dataset(times=2)
    ds = ds.assign_coords(lead_time=("time", np.array([0, 12], dtype="int32")))
    checks = check_semantics(ds)
    failure = next(c for c in checks if c.name == "valid_time_consistent" and not c.passed)
    assert "12" in failure.message

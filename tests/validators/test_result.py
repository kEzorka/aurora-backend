"""Отказ обязан называть поле, значение и ожидание — docs/DATA_CONTRACT.md §4."""

import json

import numpy as np
import pytest

from validators.result import Check, ValidationReport, fail, ok


def test_failure_message_names_field_value_and_expectation() -> None:
    check = fail("units", "semantics", field="2t", got="degC", expected="K")
    assert check.passed is False
    assert "2t" in check.message
    assert "degC" in check.message
    assert "K" in check.message
    assert check.details == {"field": "2t", "got": "degC", "expected": "K"}


def test_report_is_ok_only_when_every_check_passed() -> None:
    passing = ValidationReport((ok("a", "structure"), ok("b", "physics")))
    assert passing.ok is True
    assert passing.failures() == ()

    mixed = ValidationReport((ok("a", "structure"), fail("b", "physics", "2t", 400.0, "180..340")))
    assert mixed.ok is False
    assert [c.name for c in mixed.failures()] == ["b"]


def test_report_serialises_to_valid_json() -> None:
    report = ValidationReport((fail("range", "physics", "2t", 400.0, "180.0..340.0"),))
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["ok"] is False
    assert restored["checks"][0]["level"] == "physics"
    assert restored["checks"][0]["details"]["got"] == 400.0


def test_numpy_scalars_survive_serialisation() -> None:
    """Отказ, который не долетает до validation.json, бесполезен."""
    report = ValidationReport((fail("range", "physics", "2t", np.float32(400.5), (180.0, 340.0)),))
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["checks"][0]["details"]["got"] == pytest.approx(400.5)


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="level"):
        Check(name="x", level="vibes", passed=True)

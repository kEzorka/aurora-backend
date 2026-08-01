"""Валидаторы данных: четыре уровня из docs/DATA_CONTRACT.md §4.

Валидатор ничего не чинит и ничего не публикует. Он выносит вердикт, а решение
«публиковать или нет» принимает вызывающий — pipeline. Отвергнутый срез
не публикуется, предыдущий остаётся актуальным (docs/PIPELINE.md §3).
"""

import json
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from validators.physics import check_physics, check_sanity
from validators.result import Check, ValidationReport
from validators.semantics import check_semantics
from validators.structure import check_structure

ALL_LEVELS = ("structure", "semantics", "physics", "sanity")

__all__ = [
    "ALL_LEVELS",
    "Check",
    "RejectedError",
    "ValidationReport",
    "raise_if_rejected",
    "validate",
    "write_report",
]


class RejectedError(Exception):
    """Срез не прошёл валидацию и потому не публикуется."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__("; ".join(c.message for c in report.failures()))
        self.report = report


def validate(
    ds: xr.Dataset,
    *,
    levels: Sequence[str] | None = None,
    expect_static: bool = False,
) -> ValidationReport:
    wanted = tuple(levels) if levels is not None else ALL_LEVELS
    unknown = set(wanted) - set(ALL_LEVELS)
    if unknown:
        raise ValueError(f"levels: got {sorted(unknown)}, expected a subset of {ALL_LEVELS}")

    checks: list[Check] = []
    if "structure" in wanted:
        checks.extend(check_structure(ds, expect_static=expect_static))
    if "semantics" in wanted:
        checks.extend(check_semantics(ds))
    if "physics" in wanted:
        checks.extend(check_physics(ds))
    if "sanity" in wanted:
        checks.extend(check_sanity(ds))
    return ValidationReport(tuple(checks))


def write_report(report: ValidationReport, path: Path) -> Path:
    """Через .tmp и replace: половина отчёта хуже, чем его отсутствие."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def raise_if_rejected(report: ValidationReport) -> None:
    if not report.ok:
        raise RejectedError(report)

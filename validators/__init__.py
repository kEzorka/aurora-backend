"""Валидаторы данных: четыре уровня из docs/DATA_CONTRACT.md §4.

Валидатор ничего не чинит и ничего не публикует. Он выносит вердикт, а решение
«публиковать или нет» принимает вызывающий — pipeline. Отвергнутый срез
не публикуется, предыдущий остаётся актуальным (docs/PIPELINE.md §3).
"""

import json
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from validators.messages import check_messages
from validators.physics import check_physics, check_sanity
from validators.result import Check, ValidationReport
from validators.semantics import check_semantics
from validators.structure import check_structure

#: Уровни, которым хватает Dataset.
DATASET_LEVELS = ("structure", "semantics", "physics", "sanity")

#: «grib» первым: он единственный смотрит на файл, а не на разобранный
#: Dataset, и потому единственный видит, что cfgrib потерял (ADDENDUM-01 §5).
ALL_LEVELS = ("grib", *DATASET_LEVELS)

__all__ = [
    "ALL_LEVELS",
    "DATASET_LEVELS",
    "Check",
    "RejectedError",
    "ValidationReport",
    "check_messages",
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
    grib_path: Path | str | None = None,
) -> ValidationReport:
    """Отчёт по набору. `grib_path` включает уровень «grib» — он нужен там,
    где Dataset только что получен из GRIB, и ещё можно сверить число полей
    с числом сообщений в файле."""
    default = ALL_LEVELS if grib_path is not None else DATASET_LEVELS
    wanted = tuple(levels) if levels is not None else default
    unknown = set(wanted) - set(ALL_LEVELS)
    if unknown:
        raise ValueError(f"levels: got {sorted(unknown)}, expected a subset of {ALL_LEVELS}")
    if "grib" in wanted and grib_path is None:
        # Молча пропустить уровень нельзя: отчёт выглядел бы полным, а самая
        # ранняя проверка в нём просто отсутствовала бы.
        raise ValueError("levels: 'grib' requires grib_path")

    checks: list[Check] = []
    if "grib" in wanted and grib_path is not None:
        checks.extend(check_messages(grib_path, ds))
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

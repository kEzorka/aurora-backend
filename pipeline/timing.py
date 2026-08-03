"""Cycle budget report from a published run manifest (BACKLOG 6.2)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from pipeline.schedule import FALLBACK_AFTER, POLL_AFTER, PUBLISH_AFTER
from storage.manifest import TIMING_KEYS, read_manifest


class Budget(NamedTuple):
    stages: Mapping[str, int]
    total_seconds: int
    normal_window_seconds: int
    fallback_window_seconds: int
    remaining_seconds: int
    within_budget: bool


def summarize(manifest: Mapping[str, Any]) -> Budget:
    raw = manifest.get("timings_sec")
    if not isinstance(raw, Mapping) or set(raw) != set(TIMING_KEYS):
        raise ValueError(f"timings_sec: expected exactly {', '.join(TIMING_KEYS)}")
    stages: dict[str, int] = {}
    for key in TIMING_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"timings_sec.{key}: expected a non-negative number")
        stages[key] = round(float(value))
    total = sum(stages.values())
    normal = round((PUBLISH_AFTER - POLL_AFTER).total_seconds())
    fallback = round((PUBLISH_AFTER - FALLBACK_AFTER).total_seconds())
    return Budget(stages, total, normal, fallback, normal - total, total <= normal)


def markdown(budget: Budget) -> str:
    rows = ["| stage | seconds |", "|---|---:|"]
    rows.extend(f"| {key} | {budget.stages[key]} |" for key in TIMING_KEYS)
    rows.extend(
        (
            f"| **total** | **{budget.total_seconds}** |",
            "",
            f"Normal window (poll → publish): {budget.normal_window_seconds} s.  ",
            f"Fallback window (fallback → publish): {budget.fallback_window_seconds} s.  ",
            f"Remaining normal window: {budget.remaining_seconds} s.  ",
            f"Result: **{'within budget' if budget.within_budget else 'over budget'}**.",
        )
    )
    return "\n".join(rows)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = summarize(read_manifest(args.manifest))
    print(markdown(report))
    return 0 if report.within_budget else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

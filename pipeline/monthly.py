"""Построение pinned-слоя месячных средних ERA5 (BACKLOG 2.5)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapters import era5_arco
from storage import monthly


def build(
    root: str | Path,
    start: datetime,
    stop: datetime,
    *,
    source: Any = era5_arco.ARCO_URL,
    now: datetime | None = None,
) -> Path:
    """Прочитать финальный ERA5 лениво, агрегировать и опубликовать обе раскладки."""
    clock = now or datetime.now(UTC)
    if era5_arco.source_version(stop, now=clock) != era5_arco.FINAL:
        raise ValueError(
            f"{stop.isoformat()}: месяц ещё предварительный; "
            "monthly публикует только финальный ERA5"
        )
    archive = era5_arco.open_archive(source, chunks={})
    hourly = era5_arco.read_period(
        archive, monthly.MONTHLY_VARS, start, stop, source_url=str(source), now=clock
    )
    return monthly.publish(monthly.aggregate(hourly), root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="построить history/monthly из ARCO ERA5")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--from", dest="start", required=True, type=_moment)
    parser.add_argument("--to", dest="stop", required=True, type=_moment)
    parser.add_argument("--source", default=era5_arco.ARCO_URL)
    args = parser.parse_args(argv)
    path = build(args.root, args.start, args.stop, source=args.source)
    print(path)
    return 0


def _moment(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


if __name__ == "__main__":
    raise SystemExit(main())

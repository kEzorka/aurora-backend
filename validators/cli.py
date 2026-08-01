"""`make validate` — прогнать валидаторы по записанному срезу.

Код возврата важен: 1 означает «срез отвергнут», и конвейер обязан на этом
остановиться, а не публиковать (docs/PIPELINE.md §3).
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

from validators import validate, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validators.cli")
    parser.add_argument("path", nargs="?", type=Path, help="путь к Zarr-срезу")
    parser.add_argument("--latest", action="store_true", help="взять последний срез из artifacts/")
    parser.add_argument("--out", type=Path, default=None, help="куда положить validation.json")
    args = parser.parse_args(argv)

    target: Path | None = args.path
    if args.latest:
        candidates = sorted(Path("artifacts").glob("*/forecast.zarr"))
        if not candidates:
            print("artifacts/: срезов нет", file=sys.stderr)
            return 2
        target = candidates[-1]
    if target is None:
        parser.error("укажите путь или --latest")

    # chunks={} — читать чанками с диска, как они записаны. Без этого редукции
    # валидатора грузят поле целиком: 2.2 ГБ одной температуры на 40 шагов.
    report = validate(xr.open_zarr(target, chunks={}))
    destination = args.out or Path(target).parent / "validation.json"
    write_report(report, destination)

    for check in report.failures():
        print(f"FAIL [{check.level}] {check.name}: {check.message}", file=sys.stderr)
    print(f"{'ok' if report.ok else 'rejected'}: {len(report.checks)} checks -> {destination}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

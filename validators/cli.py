"""`make validate` — прогнать валидаторы по записанному срезу.

Код возврата важен: 1 означает «срез отвергнут», и конвейер обязан на этом
остановиться, а не публиковать (docs/PIPELINE.md §3).
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

from contracts import canon
from validators import validate, write_report


def layer_from_path(path: Path) -> str:
    """Слой по имени каталога: `.../forecast/current/hourly` — часовой.

    Угадывать нечего: имена каталогов заданы docs/STORAGE.md §2. Всё, что не
    названо слоем, считается `coarse` — основным слоем прогноза.
    """
    for part in reversed(path.parts):
        if part in canon.LAYERS:
            return part
    return "coarse"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validators.cli")
    parser.add_argument("path", nargs="?", type=Path, help="путь к Zarr-срезу")
    parser.add_argument("--latest", action="store_true", help="взять последний срез из artifacts/")
    parser.add_argument("--out", type=Path, default=None, help="куда положить validation.json")
    parser.add_argument(
        "--layer",
        choices=sorted(canon.LAYERS),
        default=None,
        help="слой хранилища; по умолчанию определяется по пути",
    )
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
    layer = args.layer or layer_from_path(Path(target))
    report = validate(xr.open_zarr(target, chunks={}), layer=layer)
    destination = args.out or Path(target).parent / "validation.json"
    write_report(report, destination)

    for check in report.failures():
        print(f"FAIL [{check.level}] {check.name}: {check.message}", file=sys.stderr)
    print(f"{'ok' if report.ok else 'rejected'}: {len(report.checks)} checks -> {destination}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

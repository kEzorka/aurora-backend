"""Скачать фикстуры GRIB заново.

    python -m tests.fixtures.fetch [--date 20260731]

Ходит в сеть, поэтому живёт отдельно от тестов и не вызывается из них
(`docs/TESTING.md`: тест, которому нужна сеть, — это мониторинг, а не тест).

Файлы кладутся байт в байт такими, какими их отдал источник. Из ECMWF берётся
одно сообщение из общего файла на 200+ МБ — по смещению из соседнего `.index`,
обычным `Range`-запросом; из GFS — через фильтр NOMADS, который собирает GRIB
из выбранных сообщений на своей стороне.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

GRIB_DIR = Path(__file__).resolve().parent / "grib"

ECMWF_BASE = (
    "https://data.ecmwf.int/forecasts/{date}/00z/ifs/0p25/oper/{date}000000-{step}h-oper-fc"
)
NOMADS = (
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
    "?dir=%2Fgfs.{date}%2F00%2Fatmos&file=gfs.t00z.pgrb2.0p25.f{lead:03d}&{selector}"
)

#: (файл, шаг в часах, отбор сообщения в `.index`)
ECMWF_WANTED: tuple[tuple[str, int, dict[str, str]], ...] = (
    ("ecmwf_2t_6h.grib2", 6, {"param": "2t"}),
    ("ecmwf_t850_6h.grib2", 6, {"param": "t", "levelist": "850"}),
    ("ecmwf_tp_6h.grib2", 6, {"param": "tp"}),
    ("ecmwf_tp_12h.grib2", 12, {"param": "tp"}),
)

#: (файл, лид в часах, параметры фильтра NOMADS)
GFS_WANTED: tuple[tuple[str, int, str], ...] = (
    ("gfs_t2m.grib2", 6, "var_TMP=on&lev_2_m_above_ground=on"),
    ("gfs_apcp_f006.grib2", 6, "var_APCP=on&lev_surface=on"),
    ("gfs_apcp_f012.grib2", 12, "var_APCP=on&lev_surface=on"),
)


def _get(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return bytes(response.read())


def _ecmwf_message(date: str, step: int, match: dict[str, str]) -> bytes:
    """Одно сообщение из файла прогона, без скачивания файла целиком."""
    base = ECMWF_BASE.format(date=date, step=step)
    index = [json.loads(line) for line in _get(f"{base}.index").decode().splitlines() if line]
    rows = [r for r in index if all(r.get(k) == v for k, v in match.items())]
    if len(rows) != 1:
        raise SystemExit(f"{match} в {base}.index: найдено {len(rows)} сообщений, нужно ровно одно")
    offset, length = rows[0]["_offset"], rows[0]["_length"]
    # Range включает оба конца, поэтому −1: иначе к сообщению приклеится первый
    # байт следующего, и cfgrib прочитает его как обрезанный файл.
    return _get(f"{base}.grib2", {"Range": f"bytes={offset}-{offset + length - 1}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tests.fixtures.fetch")
    parser.add_argument("--date", default="20260731", help="дата прогона, YYYYMMDD")
    args = parser.parse_args(argv)

    GRIB_DIR.mkdir(parents=True, exist_ok=True)
    for name, step, match in ECMWF_WANTED:
        (GRIB_DIR / name).write_bytes(_ecmwf_message(args.date, step, match))
        print(f"{name}: {(GRIB_DIR / name).stat().st_size} байт")
    for name, lead, selector in GFS_WANTED:
        url = NOMADS.format(date=args.date, lead=lead, selector=selector)
        (GRIB_DIR / name).write_bytes(_get(url))
        print(f"{name}: {(GRIB_DIR / name).stat().st_size} байт")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

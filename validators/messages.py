"""Уровень «сообщения»: сколько полей было в файле до cfgrib (ADDENDUM-01 §5).

Все остальные уровни смотрят на Dataset, то есть на то, что cfgrib **сумел**
разобрать. А cfgrib молча выбрасывает сообщения, которые не уложились в один
Dataset: два сообщения на одно (переменная, уровень, срок) дают одно поле, без
исключения и без warning. Дальше набор выглядит полным — все поля на месте,
все значения в диапазоне, — просто часть из них не та, что скачали.

Поэтому сообщения считаются по файлу, напрямую через eccodes, и сравниваются
с числом полей в Dataset. Это единственная проверка в проекте, которая
принимает путь, а не Dataset: после `open_dataset` спрашивать уже поздно.
"""

from __future__ import annotations

from pathlib import Path

import eccodes
import xarray as xr

from validators.result import Check, fail, ok

LEVEL = "grib"

#: Оси сетки. Всё, что не они, — это отдельные сообщения GRIB: уровень, срок,
#: номер ансамбля. Имена и до канонизации (`latitude`), и после (`lat`).
GRID_DIMS = frozenset({"latitude", "longitude", "lat", "lon"})


def count_messages(path: Path | str) -> int:
    """Число сообщений в файле. Читает только заголовки, не значения."""
    with Path(path).open("rb") as handle:
        return int(eccodes.codes_count_in_file(handle))


def count_fields(ds: xr.Dataset) -> int:
    """Сколько сообщений должно было уйти на этот Dataset.

    Поле на 13 уровнях — это 13 сообщений, а не одно: у GRIB нет размерностей,
    каждый двумерный срез приезжает отдельным сообщением.
    """
    total = 0
    for variable in ds.data_vars.values():
        slices = 1
        for dim, size in zip(variable.dims, variable.shape, strict=True):
            if str(dim) not in GRID_DIMS:
                slices *= int(size)
        total += slices
    return total


def check_messages(path: Path | str, ds: xr.Dataset | None) -> list[Check]:
    """Проверки уровня «сообщения». `ds` — то, что вернул cfgrib, или None.

    None означает, что cfgrib до файла не дошёл: тогда вердикт один — файл
    нечитаем, и сравнивать не с чем.
    """
    path = Path(path)
    try:
        messages = count_messages(path)
    except Exception as error:
        return [fail("grib_readable", LEVEL, str(path), f"{type(error).__name__}", "readable GRIB")]

    if messages == 0:
        # Ноль сообщений — это чаще всего скачанная HTML-страница с ошибкой:
        # eccodes на ней не спотыкается, просто не находит ни одного `GRIB`.
        return [fail("grib_readable", LEVEL, str(path), "0 messages", "at least one message")]

    checks = [ok("grib_readable", LEVEL, f"{messages} messages")]
    if ds is None:
        return [
            *checks,
            fail(
                "messages_parsed", LEVEL, str(path), "cfgrib failed to open", f"{messages} fields"
            ),
        ]

    fields = count_fields(ds)
    if fields != messages:
        # Меньше — потеря (дубликаты, несовместимые заголовки). Больше —
        # cfgrib что-то размножил broadcast'ом, и это тоже не то, что скачали.
        return [
            *checks,
            fail("messages_parsed", LEVEL, str(path), f"{fields} fields", f"{messages} fields"),
        ]
    return [*checks, ok("messages_parsed", LEVEL, f"{fields} fields from {messages} messages")]

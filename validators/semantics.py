"""Уровень «семантика»: единицы, времена, уровни. Ловушки 1 и 3 из docs/DOMAIN.md §6.

Данные могут быть правильной формы и всё равно значить не то: °C вместо K
проходит любую проверку размерностей, а сдвиг времени на один шаг превращает
прогноз в отчёт о прошлом. Здесь проверяются подписи, а не числа.
"""

import numpy as np
import xarray as xr

from contracts import canon
from validators.result import Check, fail, ok

LEVEL = "semantics"


def check_semantics(ds: xr.Dataset) -> list[Check]:
    checks: list[Check] = []
    checks.extend(_units(ds))
    if "time" in ds.coords:
        checks.append(_time_naive_utc(ds))
        checks.append(_valid_time_consistent(ds))
    return checks


def _units(ds: xr.Dataset) -> list[Check]:
    checks: list[Check] = []
    for raw_name in ds.data_vars:
        name = str(raw_name)
        expected = canon.UNITS.get(name)
        if expected is None:
            checks.append(fail("units_known", LEVEL, name, "unknown variable", sorted(canon.UNITS)))
            continue
        got = ds[name].attrs.get("units")
        if got != expected:
            checks.append(fail("units", LEVEL, name, got, expected))
        else:
            checks.append(ok("units", LEVEL, f"{name} in {expected}"))
    return checks


def _time_naive_utc(ds: xr.Dataset) -> Check:
    dtype = ds["time"].dtype
    # Проверка tz идёт первой: np.issubdtype на pandas-типе datetime64[ns, UTC]
    # кидает TypeError вместо ответа, и валидатор падает вместо отказа.
    if getattr(dtype, "tz", None) is not None:
        return fail("time_naive_utc", LEVEL, "time.dtype", str(dtype), "datetime64[ns] without tz")
    if not (isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64)):
        return fail("time_naive_utc", LEVEL, "time.dtype", str(dtype), "datetime64[ns]")
    return ok("time_naive_utc", LEVEL)


def _valid_time_consistent(ds: xr.Dataset) -> Check:
    """valid_time == init_time + lead_time, docs/DOMAIN.md §2."""
    if "lead_time" not in ds.coords or "init_time" not in ds.attrs:
        return ok("valid_time_consistent", LEVEL, "no lead_time coordinate; nothing to cross-check")

    init = np.datetime64(str(ds.attrs["init_time"]).rstrip("Z"), "ns")
    leads = np.asarray(ds["lead_time"].values).astype("int64")
    expected = init + leads.astype("timedelta64[h]")
    got = np.asarray(ds["time"].values).astype("datetime64[ns]")
    if not np.array_equal(got, expected):
        bad = int(np.argmax(got != expected))
        return fail(
            "valid_time_consistent",
            LEVEL,
            f"time[{bad}] (lead_time={leads[bad]} h)",
            str(got[bad]),
            str(expected[bad]),
        )
    return ok("valid_time_consistent", LEVEL)

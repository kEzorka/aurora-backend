"""What a field has to look like before anything downstream may touch it.

This exists because there is about to be more than one source. Today everything
comes from one ERA5 archive and the conventions are whatever that archive
happens to use; the moment an operational feed lands next to it, "whatever the
file says" stops being a definition. So the grid, the names, the units and the
plausible ranges live here as data, and every adapter is judged against them.

The checks are named after the four ways a weather pipeline goes wrong quietly —
none of them raise on their own, all of them produce a plausible-looking wrong
answer:

* **axes** — latitude ascending instead of descending, longitude on the wrong
  branch. Aurora wants lat 90→−90 and lon 0→360; a flipped field is a smooth,
  believable, mirrored forecast.
* **units** — the one that will actually bite when GFS arrives. ERA5 `z` is
  geopotential in m²/s²; GFS `HGT` is geopotential *height* in gpm. They differ
  by 9.80665 and both look like reasonable numbers.
* **time** — a gap, a duplicate, or a step that is not what the caller assumed.
* **accumulation** — an accumulated field read as an instantaneous one. We do not
  carry any yet, which is exactly why it is written down before we do.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

# The 0.25-degree grid, in the orientation Aurora requires. Longitude stays on
# 0..360 everywhere inside the backend; the -180..180 that people speak in is a
# presentation detail and is converted at the API edge only.
LAT_SIZE, LON_SIZE = 721, 1440
RESOLUTION = 0.25
LAT_FIRST, LAT_LAST = 90.0, -90.0
LON_FIRST, LON_LAST = 0.0, 359.75

# Aurora's output is one row shorter than its input: the analysis grid runs
# 90..-90 in 721 steps, and every forecast this backend has ever written runs
# 90..-89.75 in 720. The model does not produce a value at the south pole, so
# the two are genuinely different grids and padding one into the other would
# invent a number nobody computed.
#
# Both are therefore legal, and which one a store is on is reported per layer by
# /v1/meta rather than asserted once for the whole backend. What is *not* legal
# is any other size, or either of these with the wrong end point.
GRIDS = {
    721: ("analysis", 90.0, -90.0),
    720: ("forecast", 90.0, -89.75),
}

STEP_HOURS = 6

# name -> (units, plausible physical range). The range is a sanity bound, not a
# climatology: it should reject a unit mix-up and pass any real weather.
SURFACE: dict[str, tuple[str, tuple[float, float]]] = {
    "2t": ("K", (150.0, 350.0)),
    "10u": ("m s**-1", (-150.0, 150.0)),
    "10v": ("m s**-1", (-150.0, 150.0)),
    "msl": ("Pa", (85_000.0, 110_000.0)),
}

ATMOS: dict[str, tuple[str, tuple[float, float]]] = {
    # The lower bound is not sea level. On the 1000 hPa surface over high ground
    # the field is extrapolated below the terrain and goes negative, and how far
    # is a property of the producing centre: ERA5 stays inside -6 000, NCEP's
    # GFS analysis for 2026-07-31T06Z reached -7 760 m**2 s**-2, which is -791
    # geopotential metres. The old bound rejected a correct GFS field.
    #
    # -12 000 is about -1 220 gpm and still tight enough for the job this range
    # exists to do. Tighter would in fact be *worse* at it: the gpm-versus-m2/s2
    # hint below fires only when the value scaled by G would land back inside
    # the range, and at -6 000 a genuinely mis-scaled field failed that test and
    # got the bare "out of range" message instead of being named.
    "z": ("m**2 s**-2", (-12_000.0, 250_000.0)),
    "q": ("kg kg**-1", (0.0, 0.06)),
    "t": ("K", (150.0, 350.0)),
    "u": ("m s**-1", (-200.0, 200.0)),
    "v": ("m s**-1", (-200.0, 200.0)),
}

# The number that turns geopotential into geopotential height and back. Named,
# because a bare 9.80665 in a diff tells the reviewer nothing.
G = 9.80665

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


class ContractError(ValueError):
    """A field that must not be written or served."""


def grid_name(lat: np.ndarray) -> str | None:
    """"analysis", "forecast", or nothing recognisable."""
    spec = GRIDS.get(len(lat))
    return spec[0] if spec else None


def check_axes(lat: np.ndarray, lon: np.ndarray) -> list[str]:
    problems = []
    spec = GRIDS.get(len(lat))
    if spec is None or len(lon) != LON_SIZE:
        return [f"grid is {len(lat)}x{len(lon)}; contract knows "
                f"{sorted(GRIDS)} x {LON_SIZE}"]
    _, lat_first, lat_last = spec

    if len(lat) > 1 and lat[0] < lat[-1]:
        problems.append("latitude ascends; Aurora wants 90 -> -90")
    if len(lon) > 1 and lon[0] > lon[-1]:
        problems.append("longitude descends; contract says 0 -> 360 ascending")
    if len(lon) and lon.min() < 0:
        problems.append(f"longitude starts at {lon.min()}; contract says 0..360, "
                        "not -180..180 — convert at the API edge, not in the store")
    for name, axis, first, last in (("latitude", lat, lat_first, lat_last),
                                    ("longitude", lon, LON_FIRST, LON_LAST)):
        if len(axis) and not (np.isclose(axis[0], first) and np.isclose(axis[-1], last)):
            problems.append(f"{name} spans {axis[0]}..{axis[-1]}, contract says {first}..{last}")
    return problems


def check_time(times: np.ndarray, step_hours: int = STEP_HOURS) -> list[str]:
    problems = []
    if len(times) < 2:
        return problems
    t = np.asarray(times, dtype="datetime64[m]")
    d = np.diff(t).astype("timedelta64[m]").astype(int)
    if (d <= 0).any():
        problems.append("time is not strictly increasing (duplicate or out-of-order moment)")
    want = step_hours * 60
    if (d != want).any():
        odd = sorted({int(v) for v in d if v != want})[:5]
        problems.append(f"time steps of {odd} minutes, expected {want}")
    return problems


def check_values(name: str, arr: np.ndarray) -> list[str]:
    """Range check, plus the two unit mix-ups that produce believable numbers."""
    spec = SURFACE.get(name) or ATMOS.get(name)
    if spec is None:
        return [f"{name} is not in the contract"]
    _, (lo, hi) = spec

    finite = np.isfinite(arr)
    problems = []
    if not finite.all():
        problems.append(f"{name}: {int((~finite).sum())} non-finite values")
    if not finite.any():
        return problems

    amin, amax = float(np.min(arr[finite])), float(np.max(arr[finite]))
    if amin < lo or amax > hi:
        problems.append(f"{name}: range {amin:.3g}..{amax:.3g} outside contract {lo}..{hi}")
        if name == "z" and lo <= amin * G and amax * G <= hi:
            problems.append("z looks like geopotential HEIGHT in gpm, not geopotential "
                            f"in m**2 s**-2 — multiply by {G}")
        if name in ("2t", "t") and -100.0 < amin and amax < 100.0:
            problems.append(f"{name} looks like Celsius, not kelvin — add 273.15")
        if name == "msl" and 500.0 < amin and amax < 1200.0:
            problems.append("msl looks like hPa, not Pa — multiply by 100")
    return problems


def validate(lat, lon, times, fields: dict[str, np.ndarray],
             step_hours: int = STEP_HOURS) -> dict:
    """Everything at once. Returns a report; raising is the caller's decision.

    A report rather than an exception because the producer wants to write it
    next to the data it just made, and a validator that can only crash cannot
    be used that way.
    """
    problems = check_axes(np.asarray(lat), np.asarray(lon))
    problems += check_time(np.asarray(times), step_hours)
    for name, arr in fields.items():
        problems += check_values(name, np.asarray(arr))
    return {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fields": sorted(fields),
        "moments": int(len(times)),
        "ok": not problems,
        "problems": problems,
    }


def require(report: dict) -> None:
    if not report["ok"]:
        raise ContractError("; ".join(report["problems"]))

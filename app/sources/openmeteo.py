"""Deep-history point series, from a service that is shaped like a point.

Reads like a `SeriesStore` on purpose. `read_api._plan` asks a source for its
time axis, clips it against what a better source already claimed, and then calls
`point()`; nothing in that machinery cares whether the answer came off this disk
or out of Germany. The segment bookkeeping — which source answered which hours,
at which grid node — then covers the proxy for free, and that matters more here
than locally, because the node this service returns is not always ours.

Three conversions, all of them the kind of thing `contracts.py` exists for:

* **°C to K.** The archive is kelvin. Adding 273.15 is trivial; noticing that it
  is needed is not, and a series that steps 273 K at the seam between 1995 and
  last week is the failure this catches.
* **hPa to Pa.** Same species. `msl` around 1013 instead of 101325.
* **speed and direction to u and v.** There are no wind components in this API.
  Meteorological direction is where the wind comes *from*, so the component it
  blows *towards* is negative: `u = -speed·sin(θ)`, `v = -speed·cos(θ)`. Getting
  the sign wrong reverses every wind in the answer and breaks nothing loudly.

And one thing that is not a conversion but a choice of product. The archive
endpoint blends ERA5 with ERA5-Land and prefers Land, which is a finer grid over
a different orography — probed at 55.75 N, 37.5 E:

    default ........ returned 55.711773 N, -2.5 °C
    models=era5 .... returned 55.750000 N, -1.3 °C

The first is not our grid and not our dataset. `models=era5` is pinned here and
must stay pinned; the returned coordinates are checked against what was asked
for on every call, so if the service ever silently re-blends, the segment says
so rather than the number quietly changing.

Coverage: 1940-01-01 onward, which is ERA5's own start. The recent end is soft —
probed at 2026-07-31, the last eight days came back all-null while the API still
accepted the dates — so the tail is trimmed by `LAG_DAYS` rather than trusted.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import requests

from .. import config, contracts

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

# ERA5 itself starts here; the API refuses 1939-12-31 with "out of allowed range
# from 1940-01-01", so this is their bound as well as the dataset's.
START = dt.datetime(1940, 1, 1)

# The API accepts dates up to today and answers them with nulls. Measured on
# 2026-07-31: today-8 was full, today-5 and today-2 were 0/24 non-null. Trimming
# is what turns "a row of nulls" into "not our range", which is a 404 a caller
# can act on rather than a series with holes in it.
LAG_DAYS = 10

# Requested per variable, then converted. The right-hand side is what this
# service calls the thing; the left is the contract's name.
FIELDS = {
    "2t": ("temperature_2m",),
    "msl": ("pressure_msl",),
    "10u": ("wind_speed_10m", "wind_direction_10m"),
    "10v": ("wind_speed_10m", "wind_direction_10m"),
}

VARIABLES = tuple(FIELDS)

# What the service says it returns, checked rather than assumed. If a unit here
# stops matching the response, the conversion below is wrong and the run stops.
EXPECT_UNITS = {
    "temperature_2m": "°C",
    "pressure_msl": "hPa",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
}


class OpenMeteoSeries:
    """A `SeriesStore` over somebody else's ERA5, for the years we do not hold.

    The time axis is synthetic — this service has no discoverable one — built at
    `config.STEP_HOURS` to match the local stores. Hourly is available and is
    deliberately not used: a series whose step changes at the seam violates the
    same time contract `contracts.check_time` enforces on everything written
    here, and "more data" is not worth a silent discontinuity in the axis.
    """

    name = "proxied-history"

    def __init__(self, until: dt.datetime | None = None,
                 step_hours: int = config.STEP_HOURS,
                 session: requests.Session | None = None,
                 timeout: float = 120.0):
        self.step_hours = int(step_hours)
        self.timeout = timeout
        self._session = session or requests.Session()

        # `until` is where the local archive begins, and it is *exclusive*:
        # everything from there on is on this disk and answered in 11 ms, so the
        # proxy must not claim it. It sorts earliest-first in the plan, so an
        # unclipped proxy would swallow the range and serve every query over the
        # network — and a proxy clipped inclusively would still steal exactly one
        # moment, which is the harder bug to see.
        hard = dt.datetime.utcnow() - dt.timedelta(days=LAG_DAYS)
        cut = until - dt.timedelta(hours=self.step_hours) if until else None
        self.end = min(cut, hard) if cut else hard
        self.end = self.end.replace(minute=0, second=0, microsecond=0)
        self.end -= dt.timedelta(hours=self.end.hour % self.step_hours)

        self.variables = VARIABLES
        self._times: np.ndarray | None = None

    # ------------------------------------------------------------------- axis

    @property
    def times(self) -> np.ndarray:
        """Built once, on first use. 1940 to now at six hours is ~126 000
        moments — a megabyte, against four seconds of opening a remote store."""
        if self._times is None:
            if self.end <= START:
                self._times = np.empty(0, dtype="datetime64[ns]")
            else:
                self._times = np.arange(
                    np.datetime64(START, "ns"),
                    np.datetime64(self.end, "ns") + np.timedelta64(1, "ns"),
                    np.timedelta64(self.step_hours, "h"),
                ).astype("datetime64[ns]")
        return self._times

    def time_slice(self, start: dt.datetime | None, end: dt.datetime | None) -> slice:
        t = self.times
        i0 = 0 if start is None else int(np.searchsorted(t, np.datetime64(start), "left"))
        i1 = len(t) if end is None else int(np.searchsorted(t, np.datetime64(end), "right"))
        return slice(i0, max(i0, i1))

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        t = self.times
        if len(t) == 0:
            return None
        return (pd.Timestamp(t[0]).to_pydatetime(), pd.Timestamp(t[-1]).to_pydatetime())

    def units(self, name: str) -> str | None:
        """The contract's unit, because that is what this returns — the service's
        own units are converted away in `_convert`, not passed through."""
        spec = contracts.SURFACE.get(name)
        return spec[0] if spec else None

    def nearest_point(self, lat: float, lon: float) -> tuple[int, int, float, float]:
        """Where the 0.25° grid puts this, computed rather than fetched.

        Used to decide what to ask for, and then checked against what came back.
        """
        r = contracts.RESOLUTION
        return (-1, -1, round(round(lat / r) * r, 2), round(round((lon % 360.0) / r) * r, 2))

    # ------------------------------------------------------------------ reads

    def point(self, lat: float, lon: float, variables=None,
              start=None, end=None) -> dict:
        """One node over a range, in one request, in the contract's units."""
        names = list(variables or self.variables)
        unknown = [n for n in names if n not in FIELDS]
        if unknown:
            raise ValueError(f"{unknown} not available from {self.name}; "
                             f"have {sorted(self.variables)}")

        sel = self.time_slice(start, end)
        stamps = self.times[sel]
        if len(stamps) == 0:
            raise ValueError("no moments in range")
        lo = pd.Timestamp(stamps[0]).to_pydatetime()
        hi = pd.Timestamp(stamps[-1]).to_pydatetime()

        hourly = sorted({f for n in names for f in FIELDS[n]})
        raw = self._fetch(lat, lon, hourly, lo, hi)

        node_lat, node_lon = float(raw["latitude"]), float(raw["longitude"])
        got = np.asarray(raw["hourly"]["time"], dtype="datetime64[ns]")
        cols = {f: _floats(raw["hourly"][f]) for f in hourly}
        values = {n: self._convert(n, cols) for n in names}

        # Subsample the hourly answer onto our own axis by matching timestamps,
        # not by striding. A stride assumes the service returned exactly the
        # hours asked for with none missing, and a single gap would then shift
        # every later value in the series onto the wrong time.
        take = np.searchsorted(got, stamps)
        ok = (take < len(got)) & (got[np.minimum(take, len(got) - 1)] == stamps)
        if not ok.all():
            missing = int((~ok).sum())
            raise ValueError(f"{self.name} returned no value for {missing} of "
                             f"{len(stamps)} requested moments")
        take = take[ok]

        out = {n: np.asarray(v)[take] for n, v in values.items()}
        # The same range check every locally written field passes, applied to a
        # source we do not control. It is the only thing standing between a
        # silently changed upstream convention and a series that reads 20 K.
        problems = [p for n, v in out.items() for p in contracts.check_values(n, v)]
        if problems:
            raise ValueError(f"{self.name} failed the value contract: {problems}")

        return {
            "point": {"lat": node_lat, "lon": _signed(node_lon),
                      "requested": {"lat": lat, "lon": lon}},
            "times": [pd.Timestamp(t).to_pydatetime() for t in stamps],
            "values": out,
        }

    def _fetch(self, lat: float, lon: float, hourly, lo: dt.datetime, hi: dt.datetime) -> dict:
        _, _, want_lat, want_lon = self.nearest_point(lat, lon)
        params = {
            "latitude": want_lat,
            "longitude": _signed(want_lon),
            "start_date": lo.date().isoformat(),
            "end_date": hi.date().isoformat(),
            "hourly": ",".join(hourly),
            # Both are load-bearing. Without `models` this endpoint answers from
            # ERA5-Land on a 0.1 degree grid, which is a different dataset on a
            # different orography; without the wind unit it answers in km/h.
            "models": "era5",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        r = self._session.get(ENDPOINT, params=params, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise ValueError(f"{self.name}: {body.get('reason', 'request refused')}")

        bad = {f: body["hourly_units"].get(f) for f in hourly
               if body["hourly_units"].get(f) != EXPECT_UNITS[f]}
        if bad:
            raise ValueError(f"{self.name} changed units: {bad}; expected "
                             f"{ {f: EXPECT_UNITS[f] for f in bad} }. The conversions "
                             "below are no longer valid — do not serve this.")

        off = abs(float(body["latitude"]) - want_lat) + abs(
            _signed(float(body["longitude"])) - _signed(want_lon))
        if off > contracts.RESOLUTION / 2:
            raise ValueError(
                f"{self.name} answered for {body['latitude']},{body['longitude']} "
                f"when asked for {want_lat},{_signed(want_lon)}. That is off the "
                "0.25 degree grid, which means a different product — most likely "
                "ERA5-Land. Refusing rather than mixing two datasets in one series.")
        return body

    @staticmethod
    def _convert(name: str, cols: dict[str, np.ndarray]) -> np.ndarray:
        """Into the contract's units. Every branch here is a documented trap."""
        if name == "2t":
            return cols["temperature_2m"] + 273.15
        if name == "msl":
            return cols["pressure_msl"] * 100.0
        speed = cols["wind_speed_10m"]
        theta = np.radians(cols["wind_direction_10m"])
        # Negative because the reported direction is where the wind comes from.
        return -speed * (np.sin(theta) if name == "10u" else np.cos(theta))

    # ------------------------------------------------------------------- meta

    def describe(self) -> dict:
        c = self.coverage
        return {
            "name": self.name,
            "via": "open-meteo",
            "available": c is not None,
            "kind": "proxy",
            "upstream": ENDPOINT,
            "dataset": "ERA5 (models=era5; the default ERA5-Land blend is refused)",
            "grid": {"resolution_deg": contracts.RESOLUTION,
                     "kind": "analysis",
                     "note": "node checked per request against what was asked for"},
            "variables": sorted(self.variables),
            "from": c[0].isoformat() + "Z" if c else None,
            "to": c[1].isoformat() + "Z" if c else None,
            "step_hours": self.step_hours,
            "layout": "point-shaped upstream, answers /v1/point",
            "converted": ["degC->K", "hPa->Pa", "speed+direction->10u,10v"],
        }


def _floats(raw) -> np.ndarray:
    """Nulls become NaN. The API uses `null` for a moment it has no value for,
    and `float(None)` would raise halfway through a thirty-year series."""
    return np.asarray([np.nan if v is None else v for v in raw], dtype="float64")


def _signed(lon: float) -> float:
    lon = float(lon) % 360.0
    return lon - 360.0 if lon > 180.0 else lon

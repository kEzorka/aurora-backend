"""The read-only surface: a point, a map, and what the backend actually holds.

The job API next door exists to *make* forecasts, and a caller who wants weather
should never touch it — running a model on the request path is what this design
exists to stop. These endpoints only read what a producer has already written.

**One query surface over past and future.** A caller asks for a range of valid
times and gets it; whether an hour of that range came out of the archive or out
of this morning's rollout is reported, not asked about. Internally that is four
stores, and each query goes to the one shaped for it — the entire reason for
keeping more than one:

* `/v1/point` reads the time-major stores. Measured at 0.011 s for four
  variables over the archive, against 2.036 s from the map-major one
  (`series_probe.json`).
* `/v1/map` reads the map-major stores, where one moment is one chunk — 0.005 s,
  against 0.23–2.5 s time-major.

Every response carries `segments`: which store answered which part, and over
what range. That is not decoration. A series silently stitched across an
analysis and a forecast has a step in it that was never in the weather, and the
only defence is saying where the seam is.

Where they overlap, analysis wins. The archive's value for a moment is a better
answer than a forecast of the same moment, and the preference has to be fixed
somewhere rather than depending on which store was opened first.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from . import config, contracts
from .forecast_store import ForecastRun
from .map_store import MapStore
from .series_store import SeriesStore
from .sources import ArcoMaps, OpenMeteoSeries

router = APIRouter(prefix="/v1", tags=["read"])

# A ceiling on what one request may return, so a careless bbox cannot turn into
# a gigabyte of JSON. Both are generous for anything a person asks by hand.
MAX_SERIES_POINTS = 200_000

# A global map is 1 038 240 cells. As JSON that is ~9 MB of decimal text and a
# second of serialising, so JSON is capped at a window and the whole world is
# served as raw float32 instead — 4 MB, and the client gets an array rather than
# a list of lists it has to convert.
MAX_JSON_CELLS = 200_000

_series: SeriesStore | None = None
_series_error: str | None = None
_maps: MapStore | None = None
_forecast: ForecastRun | None = None
_proxy_series: OpenMeteoSeries | None = None
_proxy_maps: ArcoMaps | None = None


def series() -> SeriesStore:
    """Opened once, on first use rather than at import: a machine with no
    history store should still be able to serve /health and the job API."""
    global _series, _series_error
    if _series is None:
        if _series_error:
            raise HTTPException(503, _series_error)
        path = Path(config.HISTORY_STORE)
        if not path.exists():
            _series_error = f"history store not built ({path}); run scripts/build_series.py"
            raise HTTPException(503, _series_error)
        _series = SeriesStore(path)
    return _series


def maps() -> MapStore:
    """The archive, read as what it already is: one chunk per map."""
    global _maps
    if _maps is None:
        path = Path(config.DATA_ROOT)
        if not (path / "zarr.json").exists() and not (path / ".zgroup").exists():
            raise HTTPException(503, f"map store not available ({path})")
        _maps = MapStore(path)
    return _maps


def forecast() -> ForecastRun:
    """The published run. Cheap to hold: it re-resolves `latest` per access, so
    a run published three minutes ago is served without a restart."""
    global _forecast
    if _forecast is None:
        _forecast = ForecastRun(Path(config.FORECAST_ROOT))
    return _forecast


def _local_start(store) -> dt.datetime | None:
    """Where the local layer begins, which is where the proxy has to stop.

    The proxy sorts first in the plan because it is the earliest source, and an
    unclipped one would therefore claim the entire range and answer every query
    over the network at 1.3 s instead of off this disk at 5 ms. Whatever we hold,
    we serve.
    """
    try:
        c = store().coverage
    except HTTPException:
        return None
    return c[0] if c else None


def proxy_series() -> OpenMeteoSeries | None:
    """Deep-history points, from a point-shaped upstream. See app/sources/."""
    global _proxy_series
    if not config.PROXY_ENABLED:
        return None
    if _proxy_series is None:
        _proxy_series = OpenMeteoSeries(until=_local_start(series))
    return _proxy_series


def proxy_maps() -> ArcoMaps | None:
    """Deep-history maps, from a map-shaped upstream. See app/sources/."""
    global _proxy_maps
    if not config.PROXY_ENABLED:
        return None
    if _proxy_maps is None:
        _proxy_maps = ArcoMaps(until=_local_start(maps))
    return _proxy_maps


# ---------------------------------------------------------------------- planning

# Preference order, and it is a claim about correctness rather than about speed:
# for a moment both hold, the archive is what happened and the forecast is what
# was expected to. Sources are also in time order, earliest first, which is what
# lets `_plan` clip an overlap by taking the later source's tail.
def _series_sources() -> list[tuple[str, SeriesStore, str]]:
    out: list[tuple[str, SeriesStore, str]] = []
    p = proxy_series()
    if p is not None and p.coverage:
        out.append((p.name, p, "open-meteo/era5"))
    try:
        out.append(("local-history", series(), config.HISTORY_STORE.name))
    except HTTPException:
        pass
    run = forecast()
    if run.available:
        out.append(("forecast", run.series, run.run_name))
    return out


def _map_sources() -> list[tuple[str, MapStore, str]]:
    out: list[tuple[str, MapStore, str]] = []
    p = proxy_maps()
    if p is not None and p.coverage:
        out.append((p.name, p, "arco-era5"))
    try:
        out.append(("local-maps", maps(), Path(config.DATA_ROOT).name))
    except HTTPException:
        pass
    run = forecast()
    if run.available:
        out.append(("forecast", run.maps, run.run_name))
    return out


def _plan(sources, start: dt.datetime | None, end: dt.datetime | None):
    """Which store answers which part of [start, end], no moment twice.

    Sources arrive earliest-first and best-first, which here are the same order,
    so a later source contributes only what is strictly after everything already
    claimed. The result is a list of (name, store, ref, slice) with the slices
    contiguous in each store — one read each, not one read per moment.
    """
    plan = []
    taken_until: np.datetime64 | None = None
    for name, store, ref in sources:
        sel = store.time_slice(start, end)
        if sel.stop <= sel.start:
            continue
        if taken_until is not None:
            fresh = store.times[sel] > taken_until
            if not fresh.any():
                continue
            sel = slice(sel.start + int(np.argmax(fresh)), sel.stop)
        plan.append((name, store, ref, sel))
        taken_until = store.times[sel][-1]
    return plan


def _read(source: str, fn, *args):
    """Run one store's read and turn its failures into an answer, not a stack.

    A local store fails when the disk does. A proxy fails when somebody else's
    service is down, rate-limits us, changes a unit, or answers for a coordinate
    we did not ask for — all of which are ordinary and none of which the caller
    can do anything about by seeing our file paths and line numbers. The upstream
    is named because that *is* actionable: it says whose outage this is.
    """
    try:
        return fn(*args)
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(404, f"{source}: {exc}") from None
    except ValueError as exc:
        # The contract checks live in here. A source that starts answering in
        # Celsius is refused rather than served, and 502 is the honest code:
        # upstream gave us something we will not pass on.
        raise HTTPException(502, f"{source}: {exc}") from None
    except OSError as exc:
        raise HTTPException(502, f"{source} unreachable: {type(exc).__name__}") from None


def _coverage_text(sources) -> str:
    parts = []
    for name, store, _ in sources:
        c = store.coverage
        parts.append(f"{name} {c[0]:%Y-%m-%d %H:%M}..{c[1]:%Y-%m-%d %H:%M}" if c
                     else f"{name} empty")
    return "; ".join(parts) or "nothing"


def _parse_vars(raw: str | None, available) -> list[str] | None:
    if not raw:
        return None
    names = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in names if v not in available]
    if unknown:
        raise HTTPException(400, f"unknown variables {unknown}; have {sorted(available)}")
    return names


def _naive(t: dt.datetime | None) -> dt.datetime | None:
    """Timestamps are UTC throughout. An offset is honoured, then dropped, so
    everything downstream compares like with like."""
    if t is None:
        return None
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None) if t.tzinfo else t


# ------------------------------------------------------------------- endpoints


@router.get("/point")
def point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-360, le=360, description="-180..180 or 0..360, either works"),
    start: dt.datetime | None = Query(None, alias="from"),
    end: dt.datetime | None = Query(None, alias="to"),
    variables: str | None = Query(None, alias="vars", description="comma separated"),
) -> dict:
    """One place over a range of valid times, past and future in one answer."""
    sources = _series_sources()
    if not sources:
        raise HTTPException(503, "no history and no published forecast")

    plan = _plan(sources, _naive(start), _naive(end))
    if not plan:
        raise HTTPException(404, f"no moments in range; have {_coverage_text(sources)}")

    names = _parse_vars(variables, set().union(*(set(s.variables) for _, s, _ in sources)))
    for name, store, _, _ in plan:
        missing = [v for v in (names or store.variables) if v not in store.variables]
        if missing:
            raise HTTPException(400, f"{missing} not in {name} (has {sorted(store.variables)}); "
                                     "narrow the time range or the variable list")
    wanted = list(names or plan[0][1].variables)

    count = sum(sel.stop - sel.start for _, _, _, sel in plan) * len(wanted)
    if count > MAX_SERIES_POINTS:
        raise HTTPException(413, f"{count} values requested, limit {MAX_SERIES_POINTS}; "
                                 "narrow the range or the variable list")

    times: list[dt.datetime] = []
    values: dict[str, list] = {v: [] for v in wanted}
    segments = []
    for name, store, ref, sel in plan:
        lo = store.times[sel][0].astype("datetime64[s]").item()
        hi = store.times[sel][-1].astype("datetime64[s]").item()
        r = _read(name, store.point, lat, lon, wanted, lo, hi)
        times += r["times"]
        for v in wanted:
            values[v] += _clean(r["values"][v])
        # The node goes in the segment, not only at the top. History is a 721-row
        # grid ending at -90.0 and the forecast is 720 ending at -89.75, so near
        # the south pole the two halves of one series are two different places.
        # Reporting the first store's node for the whole thing is exactly the
        # silent wrong answer `nearest_point` was written to avoid.
        segments.append({
            "from": r["times"][0].isoformat() + "Z",
            "to": r["times"][-1].isoformat() + "Z",
            "moments": len(r["times"]),
            "source": name,
            "store": ref,
            "lat": r["point"]["lat"],
            "lon": r["point"]["lon"],
        })

    nodes = {(s["lat"], s["lon"]) for s in segments}
    point = {"requested": {"lat": lat, "lon": lon}}
    if len(nodes) == 1:
        point["lat"], point["lon"] = segments[0]["lat"], segments[0]["lon"]
    else:
        point["lat"] = point["lon"] = None
        point["note"] = ("segments resolve to different grid nodes; the stores are "
                         "on different grids here — see each segment's lat/lon")

    units = {v: _units_across(plan, v) for v in wanted}
    return {
        "point": point,
        "times": [t.isoformat() + "Z" for t in times],
        "variables": {v: {"units": units[v], "values": values[v]} for v in wanted},
        "segments": segments,
    }


@router.get("/map")
def map_(
    time: dt.datetime = Query(..., description="a 6-hourly valid time, UTC"),
    var: str = Query(..., description="one variable"),
    bbox: str | None = Query(None, description="west,south,east,north in degrees"),
    level: int | None = Query(None, description="pressure level in hPa, for z/q/t/u/v"),
    fmt: str = Query("json", alias="format", pattern="^(json|npy)$"),
):
    """One moment, the whole world or a window of it.

    Read from whichever map-major store holds that moment — the archive for a
    past one, the published run for a future one. The time-major copies hold the
    same surface fields and would answer this too, at 1.5 s for a global map
    against 5 ms here, which is the whole reason both exist.
    """
    sources = _map_sources()
    if not sources:
        raise HTTPException(503, "no archive and no published forecast")

    when = _naive(time)
    picked = None
    for name, store, ref in sources:
        try:
            store.index_of(when)
        except KeyError:
            continue
        picked = (name, store, ref)
        break
    if picked is None:
        raise HTTPException(404, f"{when:%Y-%m-%d %H:%M} in no store; have "
                                 f"{_coverage_text(sources)}, {config.STEP_HOURS}-hourly")
    source, m, ref = picked

    if var not in m.variables:
        raise HTTPException(400, f"unknown variable {var!r} for {source}; "
                                 f"have {sorted(m.variables)}")
    # Before the size check, so a caller who got the arguments wrong is told
    # that rather than being told the answer would have been too large.
    try:
        m.check_level(var, level)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    lat_sel, lon_sel, wrapped = _window(m, bbox)
    cells = _count(m, lat_sel, lon_sel, wrapped)
    # Checked before the read, not after: a request that is going to be refused
    # should not first decode the map to find that out.
    if fmt == "json" and cells > MAX_JSON_CELLS:
        raise HTTPException(413, f"{cells} cells as JSON, limit {MAX_JSON_CELLS}; "
                                 "pass a bbox or format=npy")

    full = _read(source, m.map_at, var, when, level)
    if wrapped:
        east, west_ = wrapped
        values = np.concatenate([full[lat_sel, east], full[lat_sel, west_]], axis=-1)
        lons = np.concatenate([m.lon[east], m.lon[west_]])
    else:
        values = full[lat_sel, lon_sel]
        lons = m.lon[lon_sel]
    lat, lon = m.lat[lat_sel].tolist(), [_signed(v) for v in lons]

    if fmt == "npy":
        return _npy_response(values, when, var, m.units(var), lat, lon, source,
                             contracts.grid_name(m.lat))
    return {
        "time": when.isoformat() + "Z",
        "variable": {"name": var, "units": m.units(var), "level": level},
        "lat": lat, "lon": lon,
        "values": _clean_rows(values),
        # The grid goes in the segment for the same reason it goes in a point
        # segment: the same bbox answered from the archive and from a forecast
        # returns different rows near the south pole, and a client reading only
        # this response has no other way to tell which grid it got.
        "segments": [{"from": when.isoformat() + "Z", "to": when.isoformat() + "Z",
                      "moments": 1, "source": source, "store": ref,
                      "grid": contracts.grid_name(m.lat)}],
    }


def _window(m, bbox: str | None):
    """(lat slice, lon slice, wrap) in index space. `wrap` is a pair of slices
    when the box crosses the prime meridian on a 0..360 grid — the case that
    silently returns nothing if you just compare indices."""
    if not bbox:
        return slice(None), slice(None), None
    west, south, east, north = _parse_bbox(bbox)
    ilat0, ilon0, _, _ = m.nearest_point(north, west)
    ilat1, ilon1, _, _ = m.nearest_point(south, east)
    lat_sel = slice(min(ilat0, ilat1), max(ilat0, ilat1) + 1)
    if ilon1 >= ilon0:
        return lat_sel, slice(ilon0, ilon1 + 1), None
    return lat_sel, None, (slice(ilon0, len(m.lon)), slice(0, ilon1 + 1))


def _count(m, lat_sel: slice, lon_sel, wrapped) -> int:
    nlat = len(range(*lat_sel.indices(len(m.lat))))
    if wrapped:
        east, west = wrapped
        nlon = len(range(*east.indices(len(m.lon)))) + len(range(*west.indices(len(m.lon))))
    else:
        nlon = len(range(*lon_sel.indices(len(m.lon))))
    return nlat * nlon


def _npy_response(values, when, var, units, lat, lon, source, grid=None):
    """float32 in .npy, with the axes in headers so one request still answers.

    numpy.load reads this directly; the alternative was a list of 1 038 240
    decimal strings.
    """
    buf = io.BytesIO()
    np.save(buf, np.asarray(values, dtype="float32"), allow_pickle=False)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "x-time": when.isoformat() + "Z",
            "x-variable": var,
            "x-units": units or "",
            "x-shape": ",".join(str(n) for n in np.shape(values)),
            "x-lat-range": f"{lat[0]},{lat[-1]}",
            "x-lon-range": f"{lon[0]},{lon[-1]}",
            "x-source": source,
            "x-grid": grid or "",
        },
    )


@router.get("/meta")
def meta() -> dict:
    """What is here, what is not, and where the not-here would come from.

    Worth a request of its own: without it a client discovers coverage by
    collecting 404s, and every miss into the range we do not hold locally is a
    round trip to somebody else's bucket.
    """
    out: dict = {
        "grid": {"resolution_deg": contracts.RESOLUTION,
                 "latitude": "descending, 90 first",
                 "longitude": "0 to 359.75 internally",
                 "api_longitude": "-180 to 180",
                 "note": "the exact shape differs per layer; see each layer's grid"},
        "step_hours": config.STEP_HOURS,
        "layers": [],
    }
    try:
        s = series()
        first, last = s.coverage or (None, None)
        out["layers"].append({
            "name": "local-history",
            "available": True,
            "grid": _grid(s),
            "variables": sorted(s.variables),
            "from": first.isoformat() + "Z" if first else None,
            "to": last.isoformat() + "Z" if last else None,
            "moments": len(s.times),
            "layout": "time-major, answers /v1/point",
        })
    except HTTPException as exc:
        out["layers"].append({"name": "local-history", "available": False,
                              "reason": exc.detail})
    try:
        m = maps()
        c = m.coverage
        out["layers"].append({
            "name": "local-maps",
            "available": True,
            "grid": _grid(m),
            "variables": sorted(m.variables),
            "levels": list(m.levels),
            "from": c[0].isoformat() + "Z" if c else None,
            "to": c[1].isoformat() + "Z" if c else None,
            "moments": len(m.times),
            "layout": "map-major, answers /v1/map",
        })
    except HTTPException as exc:
        out["layers"].append({"name": "local-maps", "available": False,
                              "reason": exc.detail})
    out["layers"].append(forecast().describe())
    if not config.PROXY_ENABLED:
        out["layers"].append({
            "name": "proxied-history", "available": False,
            "reason": "proxying disabled (AURORA_PROXY=0); nothing older than "
                      "local-history has an answer"})
    else:
        # Two entries under one name on purpose. They are the same years of the
        # same reanalysis reached two ways, and which one answers depends on the
        # shape of the query rather than on the date in it — a client that sees
        # only "proxied-history: 1940 onward" would reasonably expect a thirty
        # year point series to cost what a thirty year map series costs.
        out["layers"] += [proxy_maps().describe(), proxy_series().describe()]
    return out


def _grid(store) -> dict:
    """Per layer, because the layers are not on the same grid.

    Aurora's output is one row shorter than its input — 720 rows ending at
    -89.75 against the analysis grid's 721 ending at -90 — so a client that
    hard-codes 721 and indexes a forecast map is off by nothing at the equator
    and off by a row at the pole. Saying it per layer is cheaper than the bug.
    """
    return {
        "shape": [len(store.lat), len(store.lon)],
        "kind": contracts.grid_name(store.lat),
        "latitude": f"{store.lat[0]} to {store.lat[-1]}",
        "longitude": f"{store.lon[0]} to {store.lon[-1]}",
    }


def _units_across(plan, name: str) -> str | None:
    """The unit every segment agrees on, or a refusal to guess.

    Two stores disagreeing about a unit is the failure the contract exists to
    catch — Celsius against kelvin, hPa against Pa — and the series has already
    been concatenated by the time anyone would notice. Saying so beats picking
    the first one.
    """
    seen = {store.units(name) for _, store, _, _ in plan if store.units(name)}
    if len(seen) > 1:
        raise HTTPException(500, f"segments disagree on units for {name}: {sorted(seen)}")
    return seen.pop() if seen else None


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(v) for v in raw.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be west,south,east,north in degrees") from None
    if south > north:
        raise HTTPException(400, f"bbox south {south} is north of north {north}")
    return west, south, east, north


def _signed(lon: float) -> float:
    lon = float(lon) % 360.0
    return lon - 360.0 if lon > 180.0 else lon


def _clean(arr) -> list:
    """JSON has no NaN. Missing stays missing rather than becoming a number.

    Vectorised, and the reason is measured: `concurrency.json` has the JSON map
    path flat at ~11 req/s per worker no matter how many clients ask, because
    a Python loop over every cell holds the GIL for the whole response. numpy
    does the rounding and the finite test in C; the object round-trip only
    happens when something is actually missing, which for a model output is
    never.
    """
    a = np.round(np.asarray(arr, dtype="float64").ravel(), 4)
    finite = np.isfinite(a)
    if finite.all():
        return a.tolist()
    obj = a.astype(object)
    obj[~finite] = None
    return obj.tolist()


def _clean_rows(values) -> list:
    """The same, for a 2-D map, in one pass rather than one pass per row."""
    a = np.round(np.asarray(values, dtype="float64"), 4)
    finite = np.isfinite(a)
    if finite.all():
        return a.tolist()
    obj = a.astype(object)
    obj[~finite] = None
    return obj.tolist()

"""Score a forecast against the ERA5 truth that is already in the archive.

    python -m scripts.score_forecast outputs/forecast_20260501T0000_024h.zarr

Shapes and dtypes passing is not evidence the forecast is right — a wrong
checkpoint or a scrambled level axis produces a perfectly well-formed NetCDF.
The check that actually bites is the persistence baseline: holding the init
field constant is what a forecast has to beat. Aurora should beat it by a wide
margin at every lead, and the gap should widen as lead time grows. If it
doesn't, something upstream is silently wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np

from app import postprocess
from app.era5_store import ERA5Store

LEVEL_FOR_T = 500


def weighted_rmse(a: np.ndarray, b: np.ndarray, lat: np.ndarray) -> float:
    """Latitude-weighted RMSE: grid cells shrink towards the poles."""
    w = np.cos(np.deg2rad(lat))[:, None]
    w = np.broadcast_to(w, a.shape)
    return float(np.sqrt((w * (a - b) ** 2).sum() / w.sum()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("forecast", help="zarr store or NetCDF from run_forecast / the API")
    args = p.parse_args()

    ds = postprocess.open_forecast(args.forecast)
    init_time = dt.datetime.fromisoformat(ds.attrs["init_time"])
    lat = ds["latitude"].values
    levels = list(ds["pressure_level"].values)
    li = levels.index(LEVEL_FOR_T)

    store = ERA5Store()

    # Aurora returns 720 latitudes, not the 721 it was given: the grid has to
    # be divisible by the patch size, so the south pole row is dropped. Truth
    # and persistence have to be cropped the same way before anything is
    # compared.
    nlat = len(lat)
    if not np.allclose(store.lat[:nlat], lat):
        raise SystemExit("forecast latitudes are not a prefix of the archive grid")
    crop = lambda a: a[..., :nlat, :]

    init_surf = {k: crop(v) for k, v in store.surface_slice(init_time).items()}
    init_atmos = {k: crop(v) for k, v in store.atmos_slice(init_time).items()}

    fields = [("2t", "K"), ("msl", "Pa"), (f"t{LEVEL_FOR_T}", "K")]
    print(f"init {init_time:%Y-%m-%d %H:%M}   model {ds.attrs.get('model')}")
    print(f"{'lead':>6}  {'field':>6}  {'aurora':>10}  {'persist':>10}  {'skill':>7}")

    failures: list[str] = []

    for lead in [int(v) for v in ds["lead_time"].values]:
        valid = init_time + dt.timedelta(hours=lead)
        if not store.has(valid):
            print(f"{lead:>5}h  no truth in archive for {valid:%Y-%m-%d %H:%M}, stopping")
            break

        truth_surf = {k: crop(v) for k, v in store.surface_slice(valid).items()}
        truth_atmos = {k: crop(v) for k, v in store.atmos_slice(valid).items()}

        pairs = {
            "2t": (ds["2t"].sel(lead_time=lead).values, truth_surf["2t"], init_surf["2t"]),
            "msl": (ds["msl"].sel(lead_time=lead).values, truth_surf["msl"], init_surf["msl"]),
            f"t{LEVEL_FOR_T}": (
                ds["t"].sel(lead_time=lead).values[li],
                truth_atmos["t"][li],
                init_atmos["t"][li],
            ),
        }

        for name, _ in fields:
            pred, truth, persist = pairs[name]
            e_model = weighted_rmse(pred, truth, lat)
            e_persist = weighted_rmse(persist, truth, lat)
            # Fraction of the persistence error removed. 0 means no better
            # than doing nothing; 1 would be a perfect forecast.
            skill = 1.0 - e_model / e_persist
            print(f"{lead:>5}h  {name:>6}  {e_model:>10.4g}  {e_persist:>10.4g}  {skill:>7.3f}")
            if skill <= 0.0:
                failures.append(f"{name} at +{lead}h is no better than persistence")

    ds.close()
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("PASS — beats persistence at every lead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

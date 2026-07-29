"""Put two forecasts of the same init side by side.

    python -m scripts.compare_precision outputs_long_fp32/... outputs_long_fp16/...

Written for the fp32-vs-fp16 question, but it compares any two runs of the same
init. Two numbers matter and they answer different things:

* RMSE against ERA5 says whether the cheaper run is *worse*. It is the number
  a user of the forecast cares about.
* The direct divergence between the two runs says whether they are the same
  computation. Two runs can drift apart badly and still score alike, because
  at long leads both are mostly wrong in the same direction as the truth.

The ratio of the second to the first is the honest summary: divergence far
below the error means the precision change is lost in the noise floor of the
model itself.
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from app import postprocess
from app.era5_store import ERA5Store

LEVEL_FOR_T = 500


def weighted_rmse(a: np.ndarray, b: np.ndarray, lat: np.ndarray) -> float:
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    return float(np.sqrt((w * (a - b) ** 2).sum() / w.sum()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference", help="the run treated as ground truth for divergence")
    p.add_argument("candidate")
    p.add_argument("--every", type=int, default=4, help="print every Nth lead")
    args = p.parse_args()

    a = postprocess.open_forecast(args.reference)
    b = postprocess.open_forecast(args.candidate)

    init = dt.datetime.fromisoformat(a.attrs["init_time"])
    if b.attrs["init_time"] != a.attrs["init_time"]:
        raise SystemExit("the two forecasts start from different init times")

    lat = a["latitude"].values
    nlat = len(lat)
    li = list(a["pressure_level"].values).index(LEVEL_FOR_T)
    store = ERA5Store()

    print(f"init {init:%Y-%m-%d %H:%M}")
    print(f"  reference {args.reference}")
    print(f"  candidate {args.candidate}\n")
    print(f"{'lead':>5}  {'field':>5}  {'rmse ref':>9}  {'rmse cand':>9}  "
          f"{'delta':>8}  {'divergence':>10}  {'div/rmse':>8}")

    leads = [int(v) for v in a["lead_time"].values]
    worst = 0.0

    for lead in leads:
        valid = init + dt.timedelta(hours=lead)
        if not store.has(valid):
            break
        if lead % (args.every * 6) and lead != leads[-1]:
            continue

        ts = {k: v[..., :nlat, :] for k, v in store.surface_slice(valid).items()}
        ta = {k: v[..., :nlat, :] for k, v in store.atmos_slice(valid).items()}

        cases = {
            "2t": (a["2t"].sel(lead_time=lead).values, b["2t"].sel(lead_time=lead).values, ts["2t"]),
            "msl": (a["msl"].sel(lead_time=lead).values, b["msl"].sel(lead_time=lead).values, ts["msl"]),
            f"t{LEVEL_FOR_T}": (
                a["t"].sel(lead_time=lead).values[li],
                b["t"].sel(lead_time=lead).values[li],
                ta["t"][li],
            ),
        }

        for name, (ref, cand, truth) in cases.items():
            e_ref = weighted_rmse(ref, truth, lat)
            e_cand = weighted_rmse(cand, truth, lat)
            # Divergence is measured the same weighted way, against each other
            # rather than against the truth.
            div = weighted_rmse(ref, cand, lat)
            ratio = div / e_ref
            worst = max(worst, ratio)
            print(f"{lead:>4}h  {name:>5}  {e_ref:>9.4g}  {e_cand:>9.4g}  "
                  f"{100 * (e_cand - e_ref) / e_ref:>+7.2f}%  {div:>10.4g}  {ratio:>7.1%}")

    print(f"\nworst divergence over any printed field: {worst:.1%} of that field's own error")
    a.close()
    b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

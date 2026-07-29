"""Turn two archive timestamps into the aurora.Batch the model consumes."""

from __future__ import annotations

import datetime as dt

import numpy as np
import torch
from aurora import Batch, Metadata

from . import config
from .era5_store import ERA5Store


def build_batch(store: ERA5Store, init_time: dt.datetime) -> Batch:
    """Batch initialised at `init_time`, carrying history (init_time - 6h, init_time).

    Aurora has no equations to differentiate, so the pair of snapshots is what
    lets it infer the tendency of every field.
    """
    prev_time = init_time - dt.timedelta(hours=config.STEP_HOURS)
    for t in (prev_time, init_time):
        if not store.has(t):
            store._require(t)  # raises with a readable range message

    surf_prev, surf_cur = store.surface_slice(prev_time), store.surface_slice(init_time)
    atmos_prev, atmos_cur = store.atmos_slice(prev_time), store.atmos_slice(init_time)
    static = store.static_fields()

    def pair(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        # (2, ...) history axis, then a leading batch axis of 1.
        return torch.from_numpy(np.stack([a, b])).unsqueeze(0).contiguous()

    return Batch(
        surf_vars={k: pair(surf_prev[k], surf_cur[k]) for k in surf_cur},
        static_vars={k: torch.from_numpy(v) for k, v in static.items()},
        atmos_vars={k: pair(atmos_prev[k], atmos_cur[k]) for k in atmos_cur},
        metadata=Metadata(
            lat=torch.from_numpy(store.lat),
            lon=torch.from_numpy(store.lon),
            time=(init_time,),
            atmos_levels=store.levels,
        ),
    )


def describe(batch: Batch) -> str:
    lines = ["surf_vars:"]
    lines += [f"  {k:5s} {tuple(v.shape)} {v.dtype}" for k, v in batch.surf_vars.items()]
    lines.append("static_vars:")
    lines += [f"  {k:5s} {tuple(v.shape)} {v.dtype}" for k, v in batch.static_vars.items()]
    lines.append("atmos_vars:")
    lines += [f"  {k:5s} {tuple(v.shape)} {v.dtype}" for k, v in batch.atmos_vars.items()]
    m = batch.metadata
    lines.append(f"metadata: time={m.time} levels={m.atmos_levels}")
    lines.append(f"          lat {float(m.lat[0]):.2f}..{float(m.lat[-1]):.2f} (n={len(m.lat)})")
    lines.append(f"          lon {float(m.lon[0]):.2f}..{float(m.lon[-1]):.2f} (n={len(m.lon)})")
    return "\n".join(lines)

"""Runtime configuration. Everything overridable through environment variables."""

import os
from pathlib import Path

# The ERA5 archive. A zarr store by default; point either variable at the raw
# CDS layout (<root>/<YYYY-MM>/{surface,pressure}.nc + static.nc) and the store
# reads that instead — the format is detected, not configured.
DATA_ROOT = Path(os.environ.get("AURORA_DATA_ROOT", Path.home() / "data" / "global.zarr"))

# Time-invariant fields (lsm, z, slt).
STATIC_FILE = Path(
    os.environ.get("AURORA_STATIC_FILE", Path.home() / "data" / "static.zarr")
)

OUTPUT_DIR = Path(
    os.environ.get("AURORA_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs")
)

DEVICE = os.environ.get("AURORA_DEVICE", "cuda:0")

# AuroraPretrained is the checkpoint documented for ERA5 reanalysis input.
# The plain `Aurora` class is fine-tuned for IFS HRES T0 and would silently
# degrade on our data.
MODEL_NAME = os.environ.get("AURORA_MODEL", "AuroraPretrained")

# Off by default and it should stay that way for serving: under
# torch.inference_mode() there is no graph to trade compute against, so
# torch.utils.checkpoint just calls the function and saves nothing. The knob
# exists for a future fine-tuning path. To cut inference memory, use
# AURORA_AUTOCAST=fp16 instead.
ACTIVATION_CHECKPOINTING = os.environ.get("AURORA_ACT_CKPT", "0") == "1"

# V100 is compute capability 7.0: fp16 tensor cores yes, bf16 no. fp16 is the
# default because this backend exists to look at Aurora quickly, not to compare
# precisions — it is the faster path on this hardware and the one every analysis
# run wants. Set AURORA_AUTOCAST=off for an fp32 reference when a result is being
# checked against physics rather than explored.
#
# It lives here rather than in `inference` because the registry needs it too: it
# is part of what identifies a forecast, and a lookup must not hand an fp16 store
# to a process that asked for fp32.
AUTOCAST = os.environ.get("AURORA_AUTOCAST", "fp16")

# Aurora steps 6 h at a time.
STEP_HOURS = 6

# Forecast output format: "zarr" (a store, written one rollout step at a time)
# or "netcdf" (a single file, assembled in host memory first).
OUTPUT_FORMAT = os.environ.get("AURORA_OUTPUT_FORMAT", "zarr")

# The job registry. A file rather than a dict in memory, so that a restart
# does not orphan finished forecasts and four worker processes can see one
# another's jobs. Lives next to the outputs it accounts for.
STATE_DB = Path(os.environ.get("AURORA_STATE_DB", OUTPUT_DIR / "registry.db"))

# How much disk the forecasts may hold, and how far eviction cuts back when
# they exceed it. An absolute figure, not a share of the filesystem: the box
# is shared, and "clean at 80% full" would make our behaviour depend on how
# much somebody else downloaded. 300 GB is about 40 ten-day runs at the
# current 7.3 GB each. The 50 GB gap keeps eviction rare — with one mark the
# store would sit on the threshold and delete something on every new job.
DISK_CAP_BYTES = int(float(os.environ.get("AURORA_DISK_CAP_GB", "300")) * 1024**3)
DISK_LOW_BYTES = int(float(os.environ.get("AURORA_DISK_LOW_GB", "250")) * 1024**3)

# Wall-clock cost of one rollout step, measured on this box: forward 2.66 +
# tensors to host 0.24 + zarr write 0.62. The forward figure alone used to stand
# in for the whole step, which made every ETA a third too optimistic.
STEP_WALL_S = float(os.environ.get("AURORA_STEP_WALL_S", "3.52"))

# How long POST /forecast is willing to wait for its own result before giving
# up and handing back a job id. Three steps plus a little slack, so a short
# request and every cache hit come back inline; a ten-day rollout never fits
# and is not meant to.
INLINE_WAIT_S = float(os.environ.get("AURORA_INLINE_WAIT_S", str(round(3 * STEP_WALL_S + 1.5, 1))))

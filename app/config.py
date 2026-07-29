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
# current 7.3 GB each. The 80 GB gap keeps eviction rare — with one mark the
# store would sit on the threshold and delete something on every new job.
DISK_CAP_BYTES = int(float(os.environ.get("AURORA_DISK_CAP_GB", "300")) * 1024**3)
DISK_LOW_BYTES = int(float(os.environ.get("AURORA_DISK_LOW_GB", "220")) * 1024**3)

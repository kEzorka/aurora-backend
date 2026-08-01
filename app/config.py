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
# It lives here rather than in `inference` because it is part of what identifies
# a forecast: the manifest records it, so a store built in fp16 is never mistaken
# for the fp32 reference it should be compared against.
AUTOCAST = os.environ.get("AURORA_AUTOCAST", "fp16")

# Aurora steps 6 h at a time.
STEP_HOURS = 6

# The time-major copy of the surface fields — the layout point queries are read
# from. It holds the same data as DATA_ROOT and exists only because the chunk
# shape is the opposite one: bench/results/series_probe.json measures a
# four-variable point series at 0.011 s here against 2.036 s there. It is a
# duplicate and it costs full price, ~3.3 GB against the archive's 3.1 GB for the
# same four fields, so it carries surface variables only.
#
# This is *not* the preloaded window `PRELOAD_DAYS` describes. That one holds all
# 69 variables, is chunked `[HISTORY_TIME_CHUNK, 32, 32]`, and lives at its own
# path — a separate store rather than this one extended, because the two differ
# in every dimension that matters: variable set, chunk shape, and who writes
# them. This store is built alongside the archive; that one is filled from ARCO
# and kept on the fresh edge by the six-hourly cycle.
HISTORY_STORE = Path(
    os.environ.get("AURORA_HISTORY_STORE", Path.home() / "data" / "history_surface.zarr")
)

# Where the scheduled producer publishes. Not OUTPUT_DIR: that holds the full
# state of forecasts somebody asked for by hand, keyed by job, and this holds one
# surface-only run in two serving layouts with a `latest` symlink over it. Mixing
# them would put eviction of the served forecast under the same rule as eviction
# of an experiment nobody has opened.
FORECAST_ROOT = Path(
    os.environ.get("AURORA_FORECAST_ROOT", Path.home() / "data" / "forecast")
)

# How far the scheduled run goes. Ten days is 240 h, but "ten days from now" is
# not "ten days from the init time": the operational cycle we take input from is
# already a few hours old when it lands, so a 240 h forecast stops answering
# before the next one arrives. 44 steps is 264 h and covers the gap.
FORECAST_STEPS = int(os.environ.get("AURORA_FORECAST_STEPS", "44"))

# Published runs kept on disk. Measured: a 44-step surface-only run is 954 MB
# across both layouts, so four of them is 3.8 GB — a day of history at the
# six-hourly cadence for about one percent of the disk budget. The number is
# low because it can be: raising it costs almost nothing.
FORECAST_KEEP = int(os.environ.get("AURORA_FORECAST_KEEP", "4"))

# Anything older than the local archive is fetched from the data centres that
# hold it, and what comes back lands here. See app/sources/ for why there are two
# upstreams rather than one: the query shape, not the data's age, picks the
# source.
CACHE_ROOT = Path(os.environ.get("AURORA_CACHE_ROOT", Path.home() / "data" / "cache"))

# The ceiling the ARCO map cache keeps on itself, enforced on every write rather
# than by a sweep somebody has to remember to run. 80 GB is about 20 000 global
# maps at 4.15 MB each; a cold map costs 1.30 s from the bucket and 4 ms from
# here. It is a hard number and not a share of the filesystem because this disk
# is shared with other people's jobs.
ARCO_CACHE_BYTES = int(float(os.environ.get("AURORA_ARCO_CACHE_GB", "80")) * 1024**3)

# Whether /v1/point and /v1/map may leave this machine at all. On by default —
# without it the backend simply has no answer before the archive starts — but a
# single switch to turn off is worth having when an upstream is down or when a
# deployment must not talk to the outside world.
PROXY_ENABLED = os.environ.get("AURORA_PROXY", "1") == "1"

# Forecast output format: "zarr" (a store, written one rollout step at a time)
# or "netcdf" (a single file, assembled in host memory first).
OUTPUT_FORMAT = os.environ.get("AURORA_OUTPUT_FORMAT", "zarr")

# How much disk this backend may hold in total. An absolute figure, not a share
# of the filesystem: the box is shared, and "clean at 80% full" would make our
# behaviour depend on how much somebody else downloaded.
#
# Decimal GB, not GiB, and the distinction is not pedantry: every budget figure
# in the design — 180 GB of history, 18 GB of forecast, 100 GB of cache — comes
# from `nbytes` arithmetic in decimal, so a `1024**3` here would quietly hand the
# eviction rule 22 GB that does not exist.
#
# The split is set by `PRELOAD_DAYS` below. What is left over after the
# forecast and the preloaded window is the LRU cache for proxied history.
#
# 300 GB is a budget, not headroom: the filesystem is 2.0 TB, 87% full, with
# 249 GB free at the time of writing. Reaching the budget means reclaiming from
# what this project already holds (four 7.3 GB experiment output trees and a
# 15 GB bench directory), not taking it from other tenants.
DISK_CAP_BYTES = int(float(os.environ.get("AURORA_DISK_CAP_GB", "300")) * 1_000_000_000)

# Days of real history kept locally, all 69 variables, one layout (time-major).
# 220 days is 880 moments and 252.2 GB raw. It costs **172 GB** on disk, not the
# 180 GB this comment claimed before anyone measured it: the 1.4x divisor was
# borrowed from `bench_compress.py`, which measured ERA5's int16 *packing* rather
# than what lz4 does to raw fp32, and the real figure from
# `bench/results/store_probe.json` is 1.49. With the 18 GB the forecast costs
# that leaves about 110 GB of cache inside the 300 GB budget.
#
# 1.49 is the map-major number and this store is time-major, which is 1.6% worse
# on the like-for-like subset in the same file — 1.708 against 1.681 — so the
# figure the budget actually uses is 1.466. The correction is 3 GB and states
# itself here rather than being rounded away, because using a ratio measured on
# one layout to budget another is the exact mistake the 1.4x was.
#
# The ratio is set almost entirely by the wind fields. Measured per variable:
# msl 1.91, z 1.85, t 1.81, 2t 1.77, q 1.42, u 1.30, v 1.26, 10u 1.25, 10v 1.24.
# Wind is turbulent at the grid scale and there is nothing there to compress.
# That is **28 of the 69 fields** — `u` and `v` on 13 levels each, plus `10u` and
# `10v` — so 41% of the store compresses at 1.28 and the other 41 fields at 1.68.
# No codec choice will move the budget: the incompressible part is the low bytes
# of a turbulent field's mantissa. Height matters for q (1.32 at 500 hPa against
# 1.81 at 50 hPa, because there is barely any moisture up there) and for z.
#
# Historical *maps* are not preloaded: they are rare and a proxied one costs
# 1.30 s cold and 3 ms warm.
#
# What filling it costs, measured rather than assumed
# (`bench/results/preload_probe.json`): 34.2 s for one moment's 69 fields, so
# 880 moments is 8.4 h serially. The cost is bandwidth and not requests — ARCO
# chunks the pressure-level arrays `[1, 37, 721, 1440]`, so the 13 levels Aurora
# uses arrive inside all 37, and one moment is 784 MB on the wire to keep 287.
# That is 2.7x of overfetch nothing can be done about from this side. At eight
# workers the fetch runs at 174 MB/s, so the whole window fills in about an hour.
PRELOAD_DAYS = int(os.environ.get("AURORA_PRELOAD_DAYS", "220"))

# Moments inside one time-major chunk of the history store — `[88, 32, 32]`.
# Baked into the store: changing it means refilling, which is why it was measured
# before anything was written rather than picked and regretted.
#
# The argument against a large extent is amplification: a query for one day out
# of a store chunked 88 moments deep decompresses all 88, a 22x overread. The
# measurement says that costs nothing, because these reads are request-bound and
# not byte-bound. A 41x41 region over one day takes 23.8 ms at N=16 and 27.5 ms
# at N=88 — the 22x of extra bytes is 0.45 ms of decode at 3.2 GB/s, lost inside
# the per-file overhead of opening the chunks at all. Everything else improves
# monotonically with N: an 11-day region goes 107.6 ms at N=8 to 24.7 ms at
# N=88, the write side goes 38 MB/s to 263 MB/s, and the file count for the same
# data goes 56 931 to 5181.
#
# 88 also divides 880 exactly, so the window is ten blocks. Eviction deletes a
# whole block, so the window slides 22 days at a time and holds 220–242 days
# rather than exactly 220.
#
# The six-hourly append goes **into the newest block in place**. The obvious
# objection is that it can't: writing one moment into an `[88, 32, 32]` chunk
# makes zarr decompress the chunk, merge, and compress it back, which is 25 GB of
# read-modify-write every cycle. Measured (`bench/results/edge_probe.json`) it is
# 98 s at 69 variables, against a cycle that is 21 600 s long and whose rollout
# alone is 127 s. The reason is the one that keeps turning up here: the append
# touches the same ~5200 chunk files at any extent, so N changes only the bytes,
# and bytes are cheap — N=8 moves 2.3 GB in 84 s and N=88 moves 25.2 GB in 98 s.
# That figure is page-cache-warm on the read side; a block last touched six hours
# ago has to come off the disk, which adds tens of seconds and changes nothing.
#
# So there is no separate store for the fresh edge, no sealing pass, and one
# chunk shape everywhere. The alternative — accumulate at small N and rechunk at
# 88 — costs a 518 s rebuild every 22 days, makes the reader union two layouts,
# and serves the hottest queries from the shape that reads worst: an 11-day
# region off the edge is 56.5 ms at N=8 against 19.2 ms at N=88, while "yesterday"
# is 18.6 against 19.8. The small-N edge is slower at exactly what an edge is for.
#
# What in-place costs that a small-N edge would not is write *volume*, and the
# decision was taken on latency alone, so record it: the append rewrites the
# newest block's ~17 GB of compressed chunks every cycle, which is 68 GB/day
# against about 6 GB/day at N=8. The disk is shared, so if I/O contention ever
# becomes the complaint, this is the knob — not the chunk shape.
#
# A block is 88 x 286.6 MB = 25 GB in RAM, and this box has 243 GB available, so
# the initial fill holds a whole block in memory and writes it time-major in one
# pass. There is no map-major staging store and no rechunk step anywhere.
HISTORY_TIME_CHUNK = int(os.environ.get("AURORA_HISTORY_TIME_CHUNK", "88"))

# Where the preloaded window lives. Its own path and not `HISTORY_STORE`, which
# is the surface-only time-major copy built alongside the archive: different
# variable set, different chunk shape, different writer.
#
# Past only. The ten forecast days are a separate store under `FORECAST_ROOT`
# with its own atomic publish, and a query spanning "last week through next week"
# is stitched in `read_api` rather than by giving one array a time axis that runs
# into the future. Folding them would mean rewriting chunks that hold real
# analysis every six hours, on a merge that must never drop a truth moment — a
# transaction problem traded for a routing one, and routing is the cheaper side.
PRELOAD_ROOT = Path(
    os.environ.get("AURORA_PRELOAD_ROOT", Path.home() / "data" / "preload.zarr")
)

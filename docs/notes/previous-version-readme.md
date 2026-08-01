# Aurora forecast backend

Serves Aurora forecasts from the ERA5 archive on this box. zarr in, zarr out.

```
app/
  config.py         paths, device, model choice, output format — env-overridable
  era5_store.py     timestamp -> slice, over a zarr store or the raw NetCDF
  batch_builder.py  two timestamps -> aurora.Batch
  inference.py      checkpoint lifecycle + rollout
  postprocess.py    rollout steps -> a forecast store, written a step at a time
  jobs.py           single-worker queue (one model, one GPU)
  api.py            FastAPI
scripts/
  to_zarr.py        NetCDF archive -> one zarr store (run once)
  verify_batch.py   asserts every assumption Aurora makes silently
  run_forecast.py   end-to-end forecast, no HTTP
  score_forecast.py latitude-weighted RMSE vs ERA5 and vs persistence
outputs/            forecast_YYYYMMDDTHHMM_NNNh.zarr  (init time + final lead)
```

## Setup

```bash
python3.12 -m venv .venv312
./.venv312/bin/pip install torch==2.13.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126
./.venv312/bin/pip install -r requirements.txt

./.venv312/bin/python -m scripts.to_zarr --workers 12    # 18 min, run once
```

## Data layout

The backend reads a zarr store by default (`~/data/global.zarr` plus
`~/data/static.zarr`) and falls back to the raw CDS download
(`~/data/global/<YYYY-MM>/{surface,pressure}.nc`) if you point
`AURORA_DATA_ROOT` at it. The format is detected from the path, not
configured, so both are exercised by the same code.

Chunking is the reason zarr is worth converting to: one chunk per
(timestamp, level) field, latitude and longitude never split. A forecast reads
two timestamps, so it touches ~140 chunks of 4 MB instead of seeking twice
into two 13 GB files that also happen to straddle a month boundary. Building
the input batch for `2026-05-01T00`, same box, same code:

| Archive | Batch build |
| --- | --- |
| `global.zarr` | 1.0 s |
| raw NetCDF months | 63 s |

The NetCDF files are chunked along latitude for streaming a whole field, so
pulling one level out of one timestamp decompresses far more than it keeps.
Converting costs 18 minutes once and pays for itself on the fourth forecast.

### Why the install order matters

Two separate traps, both of which cost time here:

1. **The cu126 index is not optional.** The V100 is compute capability 7.0 and
   the default `torch==2.13.0+cu130` wheels ship kernels for 7.5 and up only.
   Every CUDA call fails with
   `no kernel image is available for execution on the device`.
2. **torch 2.13 picks its CUDA variant from the installed `nvidia-*`
   packages, not from its own version string.** Installing the default torch
   first and then force-reinstalling the cu126 build leaves the cu13
   `nvidia-cublas` / `nvidia-cuda-runtime` wheels behind, and torch quietly
   loads cu130 again — `pip list` says `2.13.0+cu126` while
   `torch.__version__` says `2.13.0+cu130`. It surfaces as a confusing
   torchvision error:
   `PyTorch and torchvision were compiled with different CUDA major versions`.
   The cure is a clean venv with torch installed first, not another
   `--force-reinstall`.

## Run

```bash
./.venv312/bin/python -m scripts.verify_batch 2026-05-01T00
./.venv312/bin/python -m scripts.run_forecast 2026-05-01T00 --steps 4
./.venv312/bin/python -m scripts.score_forecast outputs/forecast_20260501T0000_024h.zarr
./.venv312/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
./scripts/smoke_api.sh          # starts a server, submits, polls, downloads
```

`verify_batch` proves the input is well formed; `score_forecast` proves the
output is worth anything. It scores the forecast against the ERA5 truth that
is already in the archive and against a persistence baseline (the init field
held constant). Aurora has to beat persistence by a wide margin at every lead,
with the gap widening as lead time grows. A wrong checkpoint or a scrambled
level axis still writes a perfectly well-shaped NetCDF — this is the check
that catches it.

```bash
curl localhost:8000/health
curl -X POST localhost:8000/forecast \
     -H 'content-type: application/json' \
     -d '{"init_time": "2026-05-01T00:00", "steps": 4}'
curl localhost:8000/forecast/<job_id>
curl -s localhost:8000/forecast/<job_id>/download | tar -x -C .
```

A zarr forecast is a directory, so `/download` streams it as a tar rather than
pretending it is a file — nothing is staged on disk first. With
`AURORA_OUTPUT_FORMAT=netcdf` the same endpoint serves a plain `.nc`.

Read the result with `app.postprocess.open_forecast`, which knows to keep
`lead_time` as an integer hour offset under both backends.

Forecasts run in a background thread, so `POST /forecast` returns a `job_id`
immediately and `GET /forecast/{job_id}` reports `queued` / `running` /
`done` / `failed` plus how many steps have completed.

## Things that are easy to get wrong here

- **Checkpoint choice.** `AuroraPretrained` is the ERA5 model. The plain
  `Aurora` class is fine-tuned for IFS HRES T0 and produces plausible-looking
  wrong forecasts on reanalysis input.
- **Pressure level order.** The CDS files store levels descending
  (1000 -> 50); Aurora's examples use ascending. The zarr store is written
  ascending once and for all, and the NetCDF path sorts on read.
  `verify_batch` confirms it physically: mean temperature at 50 hPa must be
  colder than at 1000 hPa.
- **Month boundaries.** An init at `2026-05-01T00` needs `2026-04-30T18`,
  which in the NetCDF layout lives in the previous month's file. The store
  indexes across all months, so this works either way; it is the default
  argument of `verify_batch`.
- **Static fields.** The source `static.nc` carries a length-1 time axis that
  has to be squeezed to `(721, 1440)` — the converter drops it. Its `z` is
  orography and is a different field from the `z` in the pressure data, which
  is per-level geopotential.
- **Converting the archive needs processes, not threads.** The obvious
  `open_dataset(chunks=...)` + `to_zarr` runs at one core and 2.4 MB/s here,
  because HDF5 reads hold the GIL and dask's threads queue up behind it.
  `to_zarr.py` writes the metadata once and fills the data from a process
  pool: 18 minutes instead of eight hours. It also means the store's time
  chunk has to stay at 1 — two processes writing into one chunk is a
  read-modify-write race, and the result would be silently wrong rather than
  an error.
- **Output grid is 720 latitudes, not 721.** The grid has to be divisible by
  the patch size, so Aurora drops the south pole row: input is 90 .. -90,
  output is 90 .. -89.75. Anything comparing a forecast against ERA5 has to
  crop the truth to the forecast's own grid first, which is what
  `score_forecast` does.
- **`lead_time` units.** The attribute must not contain the word `since`, or
  xarray treats the axis as CF time. `hours` alone is not safe either — xarray
  will read it as a *timedelta* — so `open_forecast` passes both
  `decode_times=False` and `decode_timedelta=False`.
- **Output names carry the lead, not just the init time.** Keying only on init
  time means a 1-step job overwrites the store a finished 4-step job is still
  serving from `/forecast/{id}/download`. With the lead in the name a collision
  implies identical contents. Nothing prunes `outputs/` — a 24 h forecast is
  ~700 MB, so add a cleanup job before running this continuously.
- **Memory.** Predictions are moved to the CPU as they come out of `rollout`,
  and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in
  `app/__init__.py`. Without it a 4-step rollout dies at step 4 with 9.85 GiB
  stranded as reserved-but-unallocated — the forward pass itself fits fine.
  On the host side the zarr writer appends each step and releases it, so a
  40-step job costs one step (~290 MB) rather than the whole forecast.

## Measured on this box

One V100, `AuroraPretrained`, fp32, init `2026-05-01T00`:

| | |
| --- | --- |
| Checkpoint load | 23 s |
| Per rollout step | 8.2 s |
| Peak VRAM | 23.9 GiB of 31.7 |

Latitude-weighted RMSE against ERA5, with the persistence baseline alongside
(`skill` is the fraction of the persistence error removed):

| lead | 2t (K) | persist | msl (Pa) | persist | t500 (K) | persist |
| --- | --- | --- | --- | --- | --- | --- |
| +6h | 0.51 | 2.89 | 21.4 | 249 | 0.19 | 1.31 |
| +12h | 0.55 | 4.18 | 37.4 | 363 | 0.32 | 1.97 |
| +18h | 0.65 | 2.93 | 42.7 | 470 | 0.36 | 2.42 |
| +24h | 0.71 | 1.98 | 51.7 | 539 | 0.44 | 2.81 |

These numbers are identical to four significant figures whether the batch was
read from the zarr store or from the raw NetCDF months, which is the real
check on the converter: a permuted level axis or an off-by-one time index
would still beat persistence, but it would not reproduce the other path
digit for digit.

### fp16 costs nothing measurable

`AURORA_AUTOCAST=fp16` was tested where it should hurt most: 40 rollout steps,
+240 h, two independent inits, fresh processes and separate output stores so
nothing was reused between the runs.

| | fp32 | fp16 |
| --- | --- | --- |
| 40 steps | 344 s | 132 s |
| Per step | 8.2 s | 2.65 s |
| Peak VRAM | 23.9 GiB | 19.0 GiB |

RMSE against ERA5 at +240 h, and the same two runs measured directly against
each other (`scripts/compare_precision.py`):

| init | field | fp32 | fp16 | delta | fp32 vs fp16 | as % of error |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-01 | 2t | 2.808 | 2.804 | -0.13% | 0.20 K | 7.1% |
| 2026-05-01 | msl | 687.6 | 686.9 | -0.10% | 36.1 Pa | 5.2% |
| 2026-05-01 | t500 | 3.585 | 3.595 | +0.28% | 0.22 K | 6.0% |
| 2026-06-01 | 2t | 2.862 | 2.849 | -0.45% | 0.24 K | 8.3% |
| 2026-06-01 | msl | 736.1 | 734.7 | -0.19% | 42.8 Pa | 5.8% |
| 2026-06-01 | t500 | 3.376 | 3.366 | -0.30% | 0.25 K | 7.4% |

Two things to read off it. The RMSE deltas stay inside ±0.5% and change sign
between fields and inits — fp16 comes out *ahead* on five of the six — so they
are the noise floor, not degradation. The last column is the honest one: the
two runs do drift apart, from ~1% of the field's own error at +24 h to 6-8% at
+240 h, which is real accumulation. It is still an order of magnitude below
how far either run has drifted from reality.

All 69 output channels are finite in both runs at the worst lead. This is
worth checking rather than assuming: `z` reaches 2.04e5 and fp16 tops out at
65504, so the field would overflow if the model touched it unnormalised.
It does not — the range is identical in the two runs to three figures.

The default is still `off`. Nothing above argues for it; the case for keeping
the slower default is only that fp32 is the reference these numbers are
measured against.

## Scaling out

Three of the four V100s are idle. The unit of parallelism is a process, not a
thread: run one uvicorn worker per GPU with `AURORA_DEVICE=cuda:N` behind a
load balancer. Independent init times are embarrassingly parallel; a single
rollout is inherently sequential.

`AURORA_DEVICE=cuda:2` is verified end-to-end, not just documented — the
engine calls `torch.cuda.set_device` as well as moving the model, so nothing
leaks back onto `cuda:0` through a plain `"cuda"` allocation.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `AURORA_DATA_ROOT` | `~/data/global.zarr` | Archive — zarr store or NetCDF month root |
| `AURORA_STATIC_FILE` | `~/data/static.zarr` | Time-invariant fields |
| `AURORA_OUTPUT_DIR` | `./outputs` | Where forecasts land |
| `AURORA_OUTPUT_FORMAT` | `zarr` | `netcdf` for a single file instead |
| `AURORA_DEVICE` | `cuda:0` | Which GPU |
| `AURORA_MODEL` | `AuroraPretrained` | Checkpoint class |
| `AURORA_ACT_CKPT` | `0` | Activation checkpointing — a no-op under `inference_mode`, kept for a future fine-tuning path |
| `AURORA_AUTOCAST` | `fp16` | `off` for an fp32 reference run (V100 has no bf16). Recorded per job: an fp16 store is never handed back to a request made in fp32, and the two write to different paths |

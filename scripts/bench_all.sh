#!/usr/bin/env bash
# Every measurement in the report, in one pass, in the order that keeps them
# honest. Run it from the repo root on the machine with the GPUs:
#
#     setsid nohup ./scripts/bench_all.sh > /tmp/bench_all.log 2>&1 < /dev/null &
#
# Sequential on purpose. Each mode wants the box to itself: `service` measures
# what a caller waits, `concurrency` measures what four callers do to each
# other, and either one running under the other's load measures neither.
set -u

PY=./.venv/bin/python

run() {
    echo "=== $* ($(date +%H:%M:%S))"
    "$PY" -u -m scripts.bench_suite "$@" || echo "!!! failed: $*"
}

# The prioritised question: what a resident worker costs its caller, at one
# worker and at four. Two runs because `service` appends — the 4-worker numbers
# are only meaningful next to the 1-worker baseline.
run service --workers 1 --steps 4 --rounds 3
run service --workers 4 --steps 4 --rounds 3

# n=8 oversubscribes the four cards deliberately. Two fp16 forecasts on one
# V100 do not run slowly, they run out of memory, and that failure is the
# reason the compose file is one container per GPU.
run concurrency --steps 4 --variant fp16

run read

# The 240 h pairs, fp32 as the reference and fp16 as the candidate. CPU only.
OUT=./outputs
run accuracy \
    --pair "$OUT"_long_fp32/forecast_20260501T0000_240h.zarr \
           "$OUT"_long_fp16/forecast_20260501T0000_240h.zarr may \
    --pair "$OUT"_jun_fp32/forecast_20260601T0000_240h.zarr \
           "$OUT"_jun_fp16/forecast_20260601T0000_240h.zarr jun

echo "=== done ($(date +%H:%M:%S))"
ls -la bench/results/

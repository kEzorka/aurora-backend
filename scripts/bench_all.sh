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

PY=./.venv312/bin/python

run() {
    echo "=== $* ($(date +%H:%M:%S))"
    "$PY" -u -m scripts.bench_suite "$@" || echo "!!! failed: $*"
}

# The box is shared. A card another user has 20 GiB on cannot hold an fp16
# forecast's 19 GiB, so a multi-GPU run started under those conditions does not
# measure scaling — it measures somebody else's campaign, and it OOMs. Count
# the cards that are actually free and skip rather than produce that chart.
free_gpus() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits |
        awk -F, '($2 - $1) > 22000 {n++} END {print n + 0}'
}

# The prioritised question: what a resident worker costs its caller, at one
# worker and at four. Two runs because `service` appends — the 4-worker numbers
# are only meaningful next to the 1-worker baseline.
run service --workers 1 --gpus 1 --steps 4 --rounds 3

N=$(free_gpus)
if [ "$N" -ge 4 ]; then
    run service --workers 4 --steps 4 --rounds 3
    # n=8 oversubscribes the four cards deliberately. Two fp16 forecasts on one
    # V100 do not run slowly, they run out of memory, and that failure is the
    # reason the compose file is one container per GPU.
    run concurrency --steps 4 --variant fp16
else
    echo "=== skipped: service --workers 4 and concurrency need 4 free GPUs, $N free"
fi

# The hard limit, on one card: n=1 is the baseline, n=2 is the failure. Kept
# outside the branch above because it needs one free card, not four, and the
# answer it gives — one worker per GPU — is what makes the scaling question
# mostly moot.
if [ "$N" -ge 1 ]; then
    run concurrency --gpus 1 --levels 1 2 --steps 4 --variant fp16 \
        --out bench/results/oversubscribe.json
fi

# Twenty steps, one run per variant. Four steps are not enough for the compiled
# graph to stop recompiling, so a short sweep reports compile as slower than
# eager — that is warmup, not a steady state. Separate --out per variant because
# `case --out` overwrites.
INIT=2026-04-05T00
run case --init "$INIT" --steps 20 --variant fp16 \
    --out bench/results/longrun_fp16.json
run case --init "$INIT" --steps 20 --variant fp16+compile \
    --out bench/results/longrun.json

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

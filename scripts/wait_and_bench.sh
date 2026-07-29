#!/usr/bin/env bash
# Wait for the box to be free, then run the two measurements that need all four
# cards. Start it detached and forget it:
#
#     nohup ./scripts/wait_and_bench.sh > /dev/null 2>&1 &
#
# The machine is shared. `service --workers 4` and the full `concurrency` sweep
# are the only two measurements the report is missing, and both need four cards
# that nobody else is holding: one fp16 forecast peaks at 19.0 GiB, so a card
# with somebody's 12 GiB campaign on it cannot take a worker at all. Rather than
# ask a human to watch nvidia-smi for hours, poll it.
#
# Two things this deliberately does not do: it never touches another user's
# processes, and it never starts on a machine that merely looks free for a
# moment. A campaign between two jobs frees its memory for a few seconds; three
# consecutive clean polls a minute apart is the cheapest filter that tells that
# apart from a finished campaign.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=./.venv312/bin/python
LOG=bench/wait_and_bench.log
OCC=bench/results/occupancy.csv
DONE=bench/results/wait_and_bench.status

NEED=4          # cards required
FREE_MIB=22000  # free memory that counts as "a worker fits here"
STABLE=3        # consecutive clean polls before believing it
POLL=60         # seconds between polls
MAX_H=48        # give up after this long rather than poll forever

say() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" >> "$LOG"; }

# Every poll is also a datapoint: how much of the day this box is actually
# available is a number the report currently guesses at.
sample() {
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
}

free_count() {
    sample | awk -F, -v m="$FREE_MIB" '($3 - $2) > m {n++} END {print n + 0}'
}

mkdir -p bench/results
[ -f "$OCC" ] || echo "ts,free_gpus,load1,gpu0_mib,gpu1_mib,gpu2_mib,gpu3_mib" > "$OCC"

say "waiting for $NEED cards with >${FREE_MIB} MiB free, poll ${POLL}s, giving up after ${MAX_H}h"

deadline=$(( $(date +%s) + MAX_H * 3600 ))
clean=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    n=$(free_count)
    mib=$(sample | awk -F, '{printf "%s,", $2}' | sed 's/,$//')
    load=$(awk '{print $1}' /proc/loadavg)
    echo "$(date +%Y-%m-%dT%H:%M:%S),$n,$load,$mib" >> "$OCC"

    if [ "$n" -ge "$NEED" ]; then
        clean=$((clean + 1))
        say "free=$n ($clean/$STABLE consecutive)"
        [ "$clean" -ge "$STABLE" ] && break
    else
        [ "$clean" -gt 0 ] && say "free=$n — streak broken, restarting the count"
        clean=0
    fi
    sleep "$POLL"
done

if [ "$clean" -lt "$STABLE" ]; then
    say "gave up after ${MAX_H}h without $NEED free cards"
    echo "timeout" > "$DONE"
    exit 1
fi

run() {
    say "=== $*"
    "$PY" -u -m scripts.bench_suite "$@" >> "$LOG" 2>&1 || say "!!! failed: $*"
}

# Order matters. `service` measures what one caller waits with four workers
# resident; `concurrency` measures four and eight callers fighting over four
# cards. Either one running under the other's load measures neither.
run service --workers 4 --gpus 4 --steps 4 --rounds 3
# One last look: if somebody claimed a card during the service run, the
# concurrency numbers would be theirs, not ours.
say "free before concurrency: $(free_count)"
run concurrency --gpus 4 --steps 4 --variant fp16

say "done"
echo "done" > "$DONE"

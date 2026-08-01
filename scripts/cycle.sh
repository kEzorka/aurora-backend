#!/usr/bin/env bash
# The three-hourly cycle, without cron.
#
# Two ways to run the producer on a schedule, and this is the one for
# containers. `deploy/crontab` is the one for a machine you own: cron is the
# right tool there, it is already running, and it logs and mails on its own.
# Inside a container it is the wrong tool — it wants to be pid 1, it wants root,
# it does not inherit the environment the image set, and a crashing job under it
# is invisible to `docker logs`. A loop in the foreground is pid 1, keeps the
# container's exit status meaningful, and prints to stdout where the runtime is
# already looking.
#
# Alignment matters more than the interval. Sleeping a flat three hours from
# whenever the container happened to start drifts into the worst case, where
# every tick lands just before a cycle appears. This sleeps to the next wall
# clock multiple of INTERVAL_H plus OFFSET_MIN, so the schedule is the same
# whether the container started at 04:59 or at 05:01.
#
# Failures do not stop the loop. An upstream outage, a truncated GRIB, a
# contract violation on a bad cycle — all of them are things that fix themselves
# three hours later, and a producer that exits on the first one stops the
# backend until somebody notices.
set -u

INTERVAL_H="${AURORA_CYCLE_HOURS:-3}"
OFFSET_MIN="${AURORA_CYCLE_OFFSET_MIN:-20}"
SOURCE="${AURORA_SOURCE:-gfs}"
PYTHON="${AURORA_PYTHON:-python3}"

cd "$(dirname "$0")/.." || exit 1

echo "cycle: every ${INTERVAL_H}h at +${OFFSET_MIN}min, source ${SOURCE}"

next_sleep() {
    # Seconds until the next aligned slot, computed in UTC.
    local now_s slot_s next
    now_s=$(date -u +%s)
    slot_s=$(( INTERVAL_H * 3600 ))
    next=$(( (now_s / slot_s + 1) * slot_s + OFFSET_MIN * 60 ))
    # If the offset puts the next slot in the past (we are inside the offset
    # window of the current slot), that slot is still ahead of us.
    if [ $(( (now_s / slot_s) * slot_s + OFFSET_MIN * 60 )) -gt "$now_s" ]; then
        next=$(( (now_s / slot_s) * slot_s + OFFSET_MIN * 60 ))
    fi
    echo $(( next - now_s ))
}

# Run once at startup rather than waiting up to three hours to produce anything.
# produce.py is idempotent, so on a restart this exits immediately.
while true; do
    echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) tick"
    if ! "$PYTHON" -m scripts.produce --source "$SOURCE"; then
        echo "--- produce failed (exit $?); the loop continues" >&2
    fi
    s=$(next_sleep)
    echo "--- sleeping ${s}s until $(date -u -d "@$(( $(date -u +%s) + s ))" \
         +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "next slot")"
    sleep "$s"
done

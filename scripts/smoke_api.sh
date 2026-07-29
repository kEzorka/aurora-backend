#!/usr/bin/env bash
# End-to-end exercise of the HTTP layer: start uvicorn, submit a forecast,
# poll until it finishes, download the result, shut down.
#
#   ./scripts/smoke_api.sh [init_time] [steps]

set -euo pipefail

INIT="${1:-2026-05-01T00:00}"
STEPS="${2:-2}"
PORT="${PORT:-8099}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
./.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port "$PORT" >/tmp/uvicorn.log 2>&1 &
UVICORN_PID=$!
trap 'kill $UVICORN_PID 2>/dev/null || true' EXIT

echo "waiting for server..."
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 1
done

echo "--- health ---"
curl -sf "http://127.0.0.1:$PORT/health"
echo

echo "--- submit ---"
JOB=$(curl -sf -X POST "http://127.0.0.1:$PORT/forecast" \
      -H 'content-type: application/json' \
      -d "{\"init_time\": \"$INIT\", \"steps\": $STEPS}" \
      | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
echo "job_id=$JOB"

echo "--- poll ---"
# First job also loads the checkpoint onto the GPU, so allow a generous budget.
for _ in $(seq 1 240); do
    BODY=$(curl -sf "http://127.0.0.1:$PORT/forecast/$JOB")
    STATUS=$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' <<<"$BODY")
    echo "  $STATUS $(sed -n 's/.*"progress":\([0-9]*\).*/step \1/p' <<<"$BODY")"
    case "$STATUS" in
        done)   echo "$BODY"; break ;;
        failed) echo "$BODY"; exit 1 ;;
    esac
    sleep 10
done

[ "$STATUS" = "done" ] || { echo "timed out in state $STATUS"; exit 1; }

echo "--- download ---"
# The endpoint serves a tar for zarr output and a plain file for NetCDF, so
# branch on what came back rather than on the configured format. Downloading a
# store and never opening it would pass on a truncated tar.
DEST=/tmp/smoke_download
rm -rf "$DEST"; mkdir -p "$DEST"
TYPE=$(curl -sf -o "$DEST/payload" -w '%{content_type}' \
       "http://127.0.0.1:$PORT/forecast/$JOB/download")
echo "content-type=$TYPE  $(du -h "$DEST/payload" | cut -f1)"

if [ "$TYPE" = "application/x-tar" ]; then
    tar -xf "$DEST/payload" -C "$DEST"
    rm "$DEST/payload"
    OPEN="$DEST/$(ls "$DEST")"
else
    OPEN="$DEST/payload"
fi

./.venv/bin/python -c "
import sys
from app import postprocess
ds = postprocess.open_forecast(sys.argv[1])
print('  opened', dict(ds.sizes))
print('  leads', [int(v) for v in ds['lead_time'].values])
print('  2t at the last lead: mean %.2f K' % float(ds['2t'].isel(lead_time=-1).mean()))
assert len(ds['lead_time']) == int(sys.argv[2]), 'wrong number of leads'
" "$OPEN" "$STEPS"

echo "API smoke test passed"

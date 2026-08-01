"""How many read requests a second, and where the ceiling actually is.

The brief asks for "больше запросов одновременно и время запроса меньше", and
every other measurement in this repo is single-threaded: `series_probe.json`
times one read at a time, `read.json` times one read at a time. Neither says
what happens when thirty clients ask at once, which is the only regime a serving
backend is ever in.

Four query classes, because they load different parts of the process:

* `/v1/meta`     — no store read at all. The framework's ceiling, and the
                   control: if this plateaus at the same number as the others,
                   the load generator is what was measured, not the server.
* `/v1/point`    — the time-major store. A large read collapsed to a tiny
                   response, so the cost is decode, not the wire.
* `/v1/map` npy  — the map-major store. One chunk decoded, 4 MB out. blosc
                   releases the GIL here, so threads should overlap.
* `/v1/map` json — the same read plus `_clean()`, a Python loop over every cell.
                   That holds the GIL, so this is where threads should stop
                   helping and processes should start.

The generator is processes, not threads, for the same reason: a threaded client
pulling 4 MB responses is GIL-bound and would plateau before the server does.

Usage — starts and stops its own server, so the numbers always belong to the
configuration named next to them:

    python bench/concurrency.py --workers 1,4,8 --clients 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "concurrency.json"

# One timestamp throughout. Not laziness: serving the newest run to many clients
# is the case this backend exists for, and it is warm by construction. A ladder
# that walked the archive would measure the page cache instead of the server.
CASES = {
    "meta": "/v1/meta",
    "point": "/v1/point?lat=55.75&lon=37.62&vars=2t,10u,10v,msl",
    "map_npy": "/v1/map?time={t}&var=2t&format=npy",
    "map_json_bbox": "/v1/map?time={t}&var=2t&bbox=-10,35,40,70",
}


def _worker(args) -> list[float]:
    """One client: a single kept-alive connection, requests until the deadline."""
    host, port, path, seconds = args
    conn = http.client.HTTPConnection(host, port, timeout=120)
    out: list[float] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        t0 = time.perf_counter()
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        out.append(time.perf_counter() - t0)
        if r.status != 200:
            conn.close()
            raise SystemExit(f"{path} -> {r.status} {body[:200]!r}")
    conn.close()
    return out


def run_case(host: str, port: int, path: str, clients: int, seconds: float) -> dict:
    with ProcessPoolExecutor(max_workers=clients) as pool:
        t0 = time.perf_counter()
        batches = list(pool.map(_worker, [(host, port, path, seconds)] * clients))
        wall = time.perf_counter() - t0

    lat = sorted(v for b in batches for v in b)
    n = len(lat)
    return {
        "clients": clients,
        "requests": n,
        "wall_s": round(wall, 2),
        "rps": round(n / wall, 1),
        "p50_ms": round(_pct(lat, 0.50) * 1000, 1),
        "p95_ms": round(_pct(lat, 0.95) * 1000, 1),
        "p99_ms": round(_pct(lat, 0.99) * 1000, 1),
        "mean_ms": round(statistics.fmean(lat) * 1000, 1),
    }


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    i = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[i]


def start_server(port: int, workers: int) -> subprocess.Popen:
    env = {**os.environ, "AURORA_DEVICE": "cpu"}
    cmd = [sys.executable, "-m", "uvicorn", "app.read_app:app",
           "--host", "127.0.0.1", "--port", str(port),
           "--workers", str(workers), "--log-level", "warning"]
    # New process group: uvicorn's --workers spawns children, and killing the
    # parent alone leaves them holding the port.
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                c.close()
                return proc
        except OSError:
            time.sleep(0.5)
    stop_server(proc)
    raise SystemExit(f"server with {workers} workers never came up on {port}")


def stop_server(proc: subprocess.Popen) -> None:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def latest_time(port: int) -> str:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", "/v1/meta")
    meta = json.loads(c.getresponse().read())
    c.close()
    maps = next(l for l in meta["layers"] if l["name"] == "local-maps")
    if not maps.get("available"):
        raise SystemExit("map store unavailable; nothing to benchmark")
    return maps["to"].rstrip("Z")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,4,8", help="uvicorn worker counts")
    ap.add_argument("--clients", default="1,2,4,8,16", help="concurrent clients")
    ap.add_argument("--seconds", type=float, default=6.0, help="per measurement")
    ap.add_argument("--port", type=int, default=8079)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    worker_counts = [int(v) for v in args.workers.split(",")]
    client_counts = [int(v) for v in args.clients.split(",")]

    results: list[dict] = []
    when = None
    for workers in worker_counts:
        proc = start_server(args.port, workers)
        try:
            when = when or latest_time(args.port)
            paths = {k: v.format(t=when) for k, v in CASES.items()}
            # Warm the page cache and the store objects once per server, so the
            # first rung of the ladder is not paying for everybody else's setup.
            for path in paths.values():
                _worker(("127.0.0.1", args.port, path, 1.0))
            for name, path in paths.items():
                for clients in client_counts:
                    row = run_case("127.0.0.1", args.port, path, clients, args.seconds)
                    row |= {"workers": workers, "case": name}
                    results.append(row)
                    print(f"w={workers:<2} {name:<14} c={clients:<3} "
                          f"{row['rps']:>8} rps  p50 {row['p50_ms']:>7} ms  "
                          f"p99 {row['p99_ms']:>8} ms", flush=True)
        finally:
            stop_server(proc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "time": when,
        "seconds_per_point": args.seconds,
        "cases": CASES,
        "results": results,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

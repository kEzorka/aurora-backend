"""Why long SSH commands to the box die with exit 255, measured rather than guessed.

    python -m scripts.ssh_probe            # ~16 min, writes bench/results/ssh.json

Every long-running command in this project has been driven over SSH, and the
ones that sit quiet -- a wait loop, a 40-step rollout printing nothing -- come
back "Connection closed by remote host", exit 255, while short ones never do.
Two explanations fit that pattern and they have opposite fixes:

  idle    something on the path (NAT, firewall, the provider's edge) drops a
          TCP flow with no packets on it. Fix is client-side keepalive.
  server  sshd or the box itself kills the session -- load, session limits,
          an idle timeout in sshd_config. Keepalive would not help.

The experiment separates them. One connection per (idle duration, keepalive)
cell, all of them opened at once and each sleeping silently for its duration.
If the no-keepalive column dies past some duration and the keepalive column
survives the same durations, it is the path, not the server.

Connections are staggered and kept to twelve: a burst of password logins is
indistinguishable from a brute-force attempt to anything watching the box.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "ssh.json"

# No default. The address of the box this was written against used to sit here
# as one, which put a live host and a valid username into every commit — the
# password was never committed, but the pair is what makes a scan into an
# attempt, and the box is shared. Nothing to fall back to now: set it, or the
# script tells you to.
HOST = os.environ.get("AURORA_SSH_HOST", "")
PASSWORD = os.environ.get("AURORA_SSH_PASSWORD", "")

if not HOST:
    raise SystemExit(
        "set AURORA_SSH_HOST=user@host (and AURORA_SSH_PASSWORD if the box "
        "wants a password rather than a key)"
    )

IDLE_S = (30, 60, 120, 240, 480, 900)
KEEPALIVE = (0, 15)  # ServerAliveInterval; 0 is OpenSSH's default, i.e. off

BASE = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-o", "BatchMode=no"]


def probe(idle: int, keepalive: int, delay: float) -> dict:
    """Hold one connection open, silent, for `idle` seconds."""
    time.sleep(delay)
    opts = list(BASE)
    if keepalive:
        opts += ["-o", f"ServerAliveInterval={keepalive}",
                 "-o", "ServerAliveCountMax=6"]
    # The remote side prints nothing until the end on purpose: the whole
    # question is what happens to a flow with no packets on it.
    cmd = (["sshpass", "-p", PASSWORD] if PASSWORD else []) + \
        ["ssh", *opts, HOST, f"sleep {idle}; echo ALIVE $(date +%s)"]

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    alive = p.stdout.startswith("ALIVE")
    print(f"  idle={idle:4d}s keepalive={keepalive or 'off':>3}  "
          f"{'ok' if alive else 'DIED'} after {wall:.0f}s (rc={p.returncode})",
          flush=True)
    return {
        "kind": "ssh", "idle_s": idle, "keepalive_s": keepalive,
        "survived": alive, "returncode": p.returncode, "wall_s": wall,
        # Kept whole: "Connection to ... closed by remote host" and "Timeout,
        # server ... not responding" are different failures with different causes.
        "stderr": p.stderr.strip()[-300:],
    }


def main() -> int:
    cells = [(idle, ka) for ka in KEEPALIVE for idle in IDLE_S]
    print(f"ssh probe: {len(cells)} connections, longest {max(IDLE_S)}s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(cells)) as pool:
        futures = [pool.submit(probe, idle, ka, i * 3.0)
                   for i, (idle, ka) in enumerate(cells)]
        # Written as each cell lands, not at the end. The run takes a quarter of
        # an hour and the short cells answer most of the question; losing them
        # because the longest one was still in flight would be the one avoidable
        # way to have to run this twice.
        for f in as_completed(futures):
            records.append(f.result())
            OUT.write_text(json.dumps(records, indent=2))

    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

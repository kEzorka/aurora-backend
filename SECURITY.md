# Security

## What this service is

An analysis tool. It exists to roll Aurora forward quickly on a box with four
V100s, so its own hardware can be looked at while it runs. It has **no
authentication of any kind**, and adding one was never in scope, so the
deployment assumption is not "hardened service" but "reachable only from the
machine it runs on, or from a network you already trust".

That assumption is load-bearing rather than a caveat. `POST /forecast` occupies
a GPU for up to 3.5 minutes and writes up to 11 GB, and it takes no credential
to call. Everything in the High finding below follows from that one fact.

## Threat model

The box this was written on is shared with other tenants, so the thing worth
defending is not the forecasts — every one of them is recomputable in minutes —
but the GPUs, the disk, and the 50 GB ERA5 archive that cost a CDS quota and
cannot be recomputed on demand. Ranked by what an attacker gets:

1. **Compute.** Four cards, no credential, one HTTP POST each.
2. **Disk.** The eviction cap is global, so filling it deletes forecasts that
   belong to other users of the service.
3. **The archive.** Mounted read-only in the container, and nothing on the
   serving path writes to it. This is the one asset already protected properly.

## Findings

Audited 2026-07-30 over the full history (42 commits, all branches) and the
serving path.

### High — the API is unauthenticated and published on all interfaces

`app/api.py:3` documents `--host 0.0.0.0`, `deploy/Dockerfile:62` runs it that
way, and `deploy/docker-compose.yml:41` publishes `8000:8000` through `8003`.
Docker's port publishing inserts its rules into the `DOCKER` iptables chain,
which is traversed **before** `INPUT` — so a host-level `ufw`/`iptables` rule
does not block a published port. On a machine with a public IP that means four
unauthenticated GPU endpoints reachable from the internet.

What an unauthenticated caller can do: hold all four cards indefinitely with
60-step jobs; write to the 300 GB cap and force eviction of everybody else's
forecasts; download any finished forecast; and pin runs so eviction cannot
reclaim them (`POST /forecast/{id}/pin`), which turns the disk cap into a
permanent loss of capacity.

Fix, in order of preference:

- Bind to loopback: `ports: ["127.0.0.1:8000:8000"]` in the compose file, and
  `--host 127.0.0.1` when running uvicorn directly. Reach it over an SSH
  tunnel.
- Or front it with a reverse proxy that authenticates, and keep the containers
  on loopback behind it.

Not a fix: a host firewall rule, for the reason above.

### Medium — Python tracebacks are returned to the caller

`app/jobs.py` stores `traceback.format_exc(limit=3)` in the `error` column, and
both `GET /forecast/{id}` and the SSE `failed` event hand that string back
verbatim (`app/api.py:89`, `app/api.py:190`). It carries absolute paths, the
venv layout, and library versions. Log it; return the job id and a short
message.

### Medium — absolute filesystem paths in normal responses

`_job_json` returns `output` (`app/api.py:88`), `GET /health` returns the
archive path (`app/api.py:65`). Both describe the host to anyone who asks. The
client only ever needs `/forecast/{id}/download`.

### Medium — the container runs as root

`deploy/Dockerfile` has no `USER`, and `/outputs` plus the HuggingFace cache are
bind-mounted writable from the host. Any code execution inside the container
writes to those host directories as uid 0. Add a non-root user and chown the
mounts.

### Medium — the synchronous POST blocks a threadpool thread

`app/api.py:110` calls `time.sleep(0.25)` in a loop inside a `def` (not `async
def`) endpoint, so the wait runs on FastAPI's threadpool and holds a thread for
up to `INLINE_WAIT_S` — 12.1 s at the current `STEP_WALL_S`. The default pool is
40 threads, so roughly 40 concurrent POSTs make every endpoint including
`/health` unresponsive, without costing the attacker a single GPU second. Make
the endpoint `async def` and `await asyncio.sleep(0.25)`.

### Low — `Registry.update` interpolates column names into SQL

`app/registry.py:207` builds `UPDATE jobs SET {cols}` with an f-string over the
caller's keyword names. Values are bound as parameters, and every current caller
passes literal keywords, so nothing is injectable today. It stops being true the
first time somebody splats a dict from a request into it. Check the keys against
the schema's column set before formatting.

### Low — two processes can queue the same forecast and delete each other's output

`ForecastService.submit` looks for a duplicate in `registry.all()`
(`app/jobs.py`), but the compose file runs four independent processes over one
`registry.db`, and the check is not inside a transaction that claims the row. Two
processes can both miss and both queue `(init_time, steps, precision)`. Since
`ForecastWriter.__init__` rmtree's whatever is already at the output path
(`app/postprocess.py:145`), the second run deletes the first's finished store
while a client is holding its download URL — which is the exact failure the
precision-in-the-filename change was made to prevent, reappearing across
processes instead of within one. Either claim the job with a unique index on the
non-terminal rows, or accept it and document that dedup is per-process.

### Low — host PIDs and GPU state in committed benchmark output

`bench/results/oversubscribe.json` records process ids and per-process GPU memory
from the box. Harmless in isolation, but it is host state in a public file.

## Checked and clean

Worth writing down, because these are the things a reader will assume are wrong:

- **No secrets in history.** No SSH password, no Copernicus CDS key, no
  HuggingFace or GitHub token, no private key, in any of the 42 commits on any
  branch. The only `sshpass` match is `scripts/ssh_probe.py:55`, which reads the
  password from `AURORA_SSH_PASSWORD` and embeds nothing.
- **`/download` cannot be walked out of.** The path comes from the registry row,
  never from the request, and `subprocess.Popen` gets an argument list with no
  shell (`app/api.py:225`).
- **`output_path` takes nothing from the caller as a string.** The filename is
  formatted from a pydantic-validated `datetime`, an `int` constrained to 1..60,
  and a precision string from the process config, so no request text reaches the
  filesystem.
- **No CORS middleware**, so a browser will not send cross-origin requests to
  it. Do not add `allow_origins=["*"]` before there is something to
  authenticate with.
- **The archive is mounted `:ro`** and nothing on the serving path writes to it.
- **No other tenants' usernames** appear anywhere in the history.

## One thing this repository does disclose

`AURORA_HOST` — a username and a live SSH host — is in **34 of the 42
commits**, at `scripts/ssh_probe.py:36` (as the default when
`AURORA_SSH_HOST` is unset), `deploy/ssh_config.example:4-5`, and
`bench/results/ssh.json:81,90`. `/home/kostya` also appears in several
`bench/results/*.json` paths.

The password is not there, but the pair (host, valid username) is what turns a
scan into a targeted attempt, and the box is shared with other tenants who did
not choose that exposure. Anyone forking or reading this should know it is
deliberate history, not an accident — and anyone running the probe script should
set `AURORA_SSH_HOST` rather than rely on that default.

## Reporting

This is a research backend with no users to notify. Open an issue.

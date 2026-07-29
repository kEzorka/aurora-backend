"""Run the test set and write JSON. Nothing here prints a conclusion.

    python -m scripts.bench_suite case --init 2026-05-01T00 --steps 4 --variant fp16
    python -m scripts.bench_suite sweep       --out bench/results/stages.json
    python -m scripts.bench_suite concurrency --out bench/results/concurrency.json
    python -m scripts.bench_suite read        --out bench/results/read.json
    python -m scripts.bench_suite accuracy    --out bench/results/accuracy.json

Every mode writes a list of records to one file and the charts are drawn from
those files, never from numbers pasted into a plotting script. A measurement
that cannot be re-run and re-plotted is an anecdote.

`sweep` and `concurrency` do not measure anything themselves — they launch
`case` in a subprocess, one process per case. That is deliberate and costs a
24 s checkpoint load every time: a second run inside a warm process would
inherit the first one's allocator state and compiled kernels, and would measure
the wrong thing. It also means concurrency is free to arrange, because separate
processes on separate GPUs is exactly how the service runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bench import testset

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def box_state() -> dict:
    """Who else is on the machine while this measurement is being taken.

    The box is shared. Another user's three training campaigns can be holding
    three of the four V100s and seven of the sixteen cores, and a timing taken
    then is not comparable with one taken on an idle box — but it is not junk
    either, as long as the conditions are recorded next to the number. This is
    what makes a surprising result diagnosable after the fact instead of
    unexplainable.
    """
    state: dict = {"load1": os.getloadavg()[0]}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        state["gpus"] = [
            {"index": int(a), "mem_mib": int(b), "util_pct": int(c)}
            for a, b, c in (line.split(", ") for line in out.strip().splitlines())
        ]
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        state["compute_apps"] = len(apps.strip().splitlines()) if apps.strip() else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return state


# --------------------------------------------------------------------------
# one case, measured in this process


def run_case(init: str, steps: int, variant: str, device: str, keep: bool) -> dict:
    # Imported here, not at module scope: `sweep` must not pay a CUDA context
    # to launch subprocesses that will each build their own.
    import torch
    from aurora import rollout as aurora_rollout

    from app import batch_builder, config, postprocess
    from app.era5_store import ERA5Store
    from app.inference import AuroraEngine

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    init_time = dt.datetime.fromisoformat(init)
    rec: dict = {
        "kind": "case",
        "init": init,
        "steps": steps,
        "variant": variant,
        "device": device,
        "box_before": box_state(),
    }

    t0 = time.time()
    store = ERA5Store()
    rec["index_s"] = time.time() - t0

    t0 = time.time()
    batch = batch_builder.build_batch(store, init_time)
    rec["read_s"] = time.time() - t0

    t0 = time.time()
    engine = AuroraEngine(
        device=device,
        autocast="fp16" if "fp16" in variant else "off",
        compile_model="compile" in variant,
    )
    rec["load_s"] = time.time() - t0

    writer = postprocess.ForecastWriter(init_time, steps)
    batch = batch.to(engine.device)

    fwd, cpu, wrt = [], [], []
    with torch.inference_mode(), engine._autocast():
        it = aurora_rollout(engine.model, batch, steps=steps)
        for _ in range(steps):
            sync()
            a = time.time()
            pred = next(it)
            sync()
            b = time.time()
            pred = pred.to("cpu")
            c = time.time()
            writer.add(pred)
            d = time.time()
            fwd.append(b - a)
            cpu.append(c - b)
            wrt.append(d - c)

    t0 = time.time()
    path = writer.finish()
    rec["finish_s"] = time.time() - t0

    rec["forward_s"] = fwd
    rec["to_cpu_s"] = cpu
    rec["write_s"] = wrt
    # The first step under torch.compile carries the whole graph trace, and the
    # second can still hit a recompile — dynamo guards on the dispatch key set,
    # which changes once autocast has run. A median over the remaining steps
    # survives both; a mean would carry the warmup into the steady state.
    rec["forward_first_s"] = fwd[0]
    rest = sorted(fwd[1:]) or fwd
    rec["forward_steady_s"] = rest[len(rest) // 2]
    rec["peak_gib"] = (
        torch.cuda.max_memory_allocated(device) / (1 << 30)
        if torch.cuda.is_available()
        else 0.0
    )
    rec["box_after"] = box_state()
    rec["output"] = str(path)
    rec["output_mb"] = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / (1 << 20) if Path(path).is_dir() else Path(path).stat().st_size / (1 << 20)

    if not keep:
        shutil.rmtree(path, ignore_errors=True) if Path(path).is_dir() else Path(path).unlink(missing_ok=True)
    return rec


# --------------------------------------------------------------------------
# orchestration


def launch(init: str, steps: int, variant: str, device: str, out: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(testset.variant_env(variant))
    # Every case writes into its own directory. Two concurrent cases of the
    # same init and length would otherwise collide on one store name and the
    # loser would append into the winner's file.
    env["AURORA_OUTPUT_DIR"] = str(ROOT / "bench" / "scratch" / out.stem)
    env["AURORA_DEVICE"] = device
    cmd = [
        sys.executable, "-m", "scripts.bench_suite", "case",
        "--init", init, "--steps", str(steps), "--variant", variant,
        "--device", device, "--out", str(out),
    ]
    return subprocess.Popen(cmd, cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def collect(procs: list[tuple[subprocess.Popen, Path]]) -> list[dict]:
    """Gather finished cases, keeping the failures as records rather than raising.

    A sweep is half an hour of GPU time. Losing all of it because case 19 hit
    an out-of-memory — which is a result, not an accident, once the box is
    oversubscribed — is the wrong trade. Failures come back with the last of
    their stderr attached and the charts skip them.
    """
    out = []
    for p, path in procs:
        _, err = p.communicate()
        if p.returncode != 0 or not path.exists():
            tail = err.decode(errors="replace").strip().splitlines()
            print(f"    FAILED: {tail[-1] if tail else 'no output'}", flush=True)
            out.append({"kind": "failed", "returncode": p.returncode,
                        "stderr_tail": "\n".join(tail[-12:])})
            continue
        out.append(json.loads(path.read_text()))
        path.unlink()
    return out


def sweep(args) -> list[dict]:
    tmp = RESULTS / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    records = []
    for init in testset.INITS:
        for steps in testset.SHORT_STEPS:
            for variant in testset.VARIANTS:
                name = f"{init.tag}-{steps}-{variant}".replace("+", "_")
                path = tmp / f"{name}.json"
                print(f"  {name}", flush=True)
                t0 = time.time()
                p = launch(init.time, steps, variant, args.device, path)
                rec = collect([(p, path)])[0]
                rec.update(tag=init.tag, init=init.time, steps=steps,
                           variant=variant, wall_s=time.time() - t0)
                records.append(rec)
    return records


def concurrency(args) -> list[dict]:
    """Per-request latency as the box fills up.

    The question this answers is whether the current design — one process per
    GPU, model resident — already gets the throughput that splitting the
    service into containers is supposed to buy. If per-request wall time at 4
    concurrent requests matches the time at 1, the GPUs are genuinely
    independent and there is nothing left for an architecture change to win.
    """
    tmp = RESULTS / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    ngpu = args.gpus
    records = []

    for n in (args.levels or testset.CONCURRENCY):
        procs = []
        started = time.time()
        for i in range(n):
            init = testset.INITS[i % len(testset.INITS)]
            path = tmp / f"conc{n}-{i}.json"
            # Requests land on GPUs round robin. Past `ngpu` this means two
            # processes share a device, which is the oversubscribed case.
            procs.append((launch(init.time, args.steps, args.variant,
                                 f"cuda:{i % ngpu}", path), path))
        got = collect(procs)
        wall = time.time() - started
        for r in got:
            r["concurrency"] = n
            r["batch_wall_s"] = wall
            if r["kind"] != "failed":
                r["kind"] = "concurrency"
        records.extend(got)
        print(f"  n={n}  batch wall {wall:.1f}s", flush=True)
    return records


def service(args) -> list[dict]:
    """What a user of the running service actually waits, cold and warm.

    Every other mode measures a fresh process and therefore pays the 24 s
    checkpoint load every time. The service does not: `ForecastService` loads
    the model on the first job and keeps it resident, so the second request and
    every one after it skips that entirely. The gap between the first and the
    second job is the whole reason a resident worker beats a process-per-request
    design — and it is the same gap a container would have to pay again on every
    cold start.

    `--workers N` starts N independent uvicorn processes, one per GPU, which is
    the deployment the README prescribes. Running the same submissions against
    1 worker and against 4 is the honest test of whether splitting the service
    up buys anything.
    """
    import urllib.error
    import urllib.request

    n = args.workers
    procs, ports = [], []
    for i in range(n):
        port = 8100 + i
        env = dict(os.environ)
        env.update(testset.variant_env(args.variant))
        env["AURORA_DEVICE"] = f"cuda:{i % args.gpus}"
        env["AURORA_OUTPUT_DIR"] = str(ROOT / "bench" / "scratch" / f"svc{i}")
        procs.append(subprocess.Popen(
            [str(ROOT / ".venv312" / "bin" / "uvicorn"), "app.api:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        ports.append(port)

    def call(port, path, data=None):
        url = f"http://127.0.0.1:{port}{path}"
        if data is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url, data=json.dumps(data).encode(),
                headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    # Wait for every worker to answer /health before timing anything, or the
    # first job carries uvicorn's own startup.
    deadline = time.time() + 180
    for port in ports:
        while time.time() < deadline:
            try:
                call(port, "/health")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        else:
            raise SystemExit(f"worker on {port} never came up")

    records = []
    try:
        for round_index in range(args.rounds):
            submitted = []
            t0 = time.time()
            for i, port in enumerate(ports):
                init = testset.INITS[(round_index * n + i) % len(testset.INITS)]
                job = call(port, "/forecast", {
                    "init_time": dt.datetime.fromisoformat(init.time).isoformat(),
                    "steps": args.steps,
                })
                submitted.append((port, job["job_id"], init, time.time()))

            pending = list(submitted)
            while pending:
                still = []
                for port, jid, init, started in pending:
                    st = call(port, f"/forecast/{jid}")
                    if st["status"] in ("done", "failed"):
                        records.append({
                            "kind": "service", "round": round_index,
                            "cold": round_index == 0, "workers": n,
                            "port": port, "init": init.time, "tag": init.tag,
                            "steps": args.steps, "variant": args.variant,
                            "status": st["status"],
                            "latency_s": time.time() - started,
                        })
                    else:
                        still.append((port, jid, init, started))
                pending = still
                if pending:
                    time.sleep(0.5)
            print(f"  workers={n} round={round_index} "
                  f"{'cold' if round_index == 0 else 'warm'} "
                  f"wall {time.time() - t0:.1f}s", flush=True)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
    return records


# --------------------------------------------------------------------------


def read_bench(args) -> list[dict]:
    import numpy as np
    import zarr

    from app import config

    MB = 1 << 20
    r = zarr.open(str(config.DATA_ROOT), mode="r")
    n = r["t2m"].shape[0]
    chunk = 721 * 1440 * 4 / MB

    def full_slice():
        out = [r[v][n - 1] for v in ("t2m", "u10", "v10", "msl")]
        out += [r[v][n - 1] for v in ("z", "q", "t", "u", "v")]
        return np.concatenate([np.asarray(a).ravel() for a in out])

    plans = {
        "one map": (lambda: r["t2m"][n - 1], chunk),
        "one day": (lambda: r["t2m"][n - 4:n], 4 * chunk),
        "one month": (lambda: r["t2m"][n - 120:n], 120 * chunk),
        "whole archive": (lambda: r["t2m"][:], n * chunk),
        "one point, whole archive": (lambda: r["t2m"][:, 360, 720], n * chunk),
        "one level, one month": (lambda: r["t"][n - 120:n, 7], 120 * chunk),
        "all levels, one moment": (lambda: r["t"][n - 1], 13 * chunk),
        "full input": (full_slice, 69 * chunk),
    }

    records = []
    for name, description in testset.READ_QUERIES:
        fn, touched = plans[name]
        t0 = time.time()
        out = fn()
        dtime = time.time() - t0
        returned = np.asarray(out).size * 4 / MB
        records.append({
            "kind": "read", "query": name, "description": description,
            "seconds": dtime, "touched_mb": touched, "returned_mb": returned,
            "waste": touched / max(returned, 1e-9),
            "mb_per_s": touched / max(dtime, 1e-9),
        })
        print(f"  {name:<26} {dtime:>6.2f}s", flush=True)
    return records


def accuracy(args) -> list[dict]:
    """RMSE against ERA5 and fp32-vs-fp16 divergence, at every lead.

    Both numbers on one axis on purpose. The divergence alone invites the
    reading that the cheaper run has lost that much accuracy; next to the error
    it is obvious that it has not.
    """
    import numpy as np

    from app import postprocess
    from app.era5_store import ERA5Store

    def wrmse(a, b, lat):
        w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
        return float(np.sqrt((w * (a - b) ** 2).sum() / w.sum()))

    store = ERA5Store()
    records = []

    for ref_path, cand_path, tag in args.pairs:
        a = postprocess.open_forecast(ref_path)
        b = postprocess.open_forecast(cand_path)
        init_time = dt.datetime.fromisoformat(a.attrs["init_time"])
        lat = a["latitude"].values
        nlat = len(lat)
        li = list(a["pressure_level"].values).index(500)

        for lead in [int(v) for v in a["lead_time"].values]:
            valid = init_time + dt.timedelta(hours=lead)
            if not store.has(valid):
                break
            ts = {k: v[..., :nlat, :] for k, v in store.surface_slice(valid).items()}
            ta = {k: v[..., :nlat, :] for k, v in store.atmos_slice(valid).items()}
            i0s = {k: v[..., :nlat, :] for k, v in store.surface_slice(init_time).items()}
            i0a = {k: v[..., :nlat, :] for k, v in store.atmos_slice(init_time).items()}

            cases = {
                "2t": (a["2t"].sel(lead_time=lead).values,
                       b["2t"].sel(lead_time=lead).values, ts["2t"], i0s["2t"]),
                "msl": (a["msl"].sel(lead_time=lead).values,
                        b["msl"].sel(lead_time=lead).values, ts["msl"], i0s["msl"]),
                "t500": (a["t"].sel(lead_time=lead).values[li],
                         b["t"].sel(lead_time=lead).values[li], ta["t"][li], i0a["t"][li]),
            }
            for field, (ref, cand, truth, persist) in cases.items():
                records.append({
                    "kind": "accuracy", "tag": tag, "init": a.attrs["init_time"],
                    "lead_h": lead, "field": field,
                    "rmse_ref": wrmse(ref, truth, lat),
                    "rmse_cand": wrmse(cand, truth, lat),
                    "rmse_persistence": wrmse(persist, truth, lat),
                    "divergence": wrmse(ref, cand, lat),
                })
            print(f"  {tag} +{lead}h", flush=True)
        a.close()
        b.close()
    return records


# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("case")
    c.add_argument("--init", required=True)
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--variant", required=True)
    c.add_argument("--device", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--keep", action="store_true")

    s = sub.add_parser("sweep")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", default=str(RESULTS / "stages.json"))

    k = sub.add_parser("concurrency")
    k.add_argument("--steps", type=int, default=4)
    k.add_argument("--variant", default="fp16")
    k.add_argument("--gpus", type=int, default=4)
    # Overridable because eight processes each loading a checkpoint is a lot of
    # host RAM on a machine somebody else is also using: on a busy box the
    # honest run is the small one, not none at all.
    k.add_argument("--levels", type=int, nargs="+", default=None)
    k.add_argument("--out", default=str(RESULTS / "concurrency.json"))

    v = sub.add_parser("service")
    v.add_argument("--workers", type=int, default=1)
    v.add_argument("--gpus", type=int, default=4)
    v.add_argument("--steps", type=int, default=4)
    v.add_argument("--rounds", type=int, default=3)
    v.add_argument("--variant", default="fp16")
    v.add_argument("--out", default=str(RESULTS / "service.json"))

    r = sub.add_parser("read")
    r.add_argument("--out", default=str(RESULTS / "read.json"))

    a = sub.add_parser("accuracy")
    a.add_argument("--pair", action="append", nargs=3, dest="pairs",
                   metavar=("REF", "CAND", "TAG"), required=True)
    a.add_argument("--out", default=str(RESULTS / "accuracy.json"))

    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.mode == "case":
        from app import config
        rec = run_case(args.init, args.steps, args.variant,
                       args.device or config.DEVICE, args.keep)
        text = json.dumps(rec, indent=2)
        Path(args.out).write_text(text) if args.out else print(text)
        return 0

    records = {"sweep": sweep, "concurrency": concurrency, "service": service,
               "read": read_bench, "accuracy": accuracy}[args.mode](args)
    # `service` is run once per worker count, so it appends rather than
    # replacing — otherwise the 4-worker run erases the 1-worker baseline it is
    # supposed to be compared against.
    out = Path(args.out)
    if args.mode == "service" and out.exists():
        records = json.loads(out.read_text()) + records
    out.write_text(json.dumps(records, indent=2))
    print(f"-> {args.out}  ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

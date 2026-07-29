"""Draw the charts from bench/results/*.json.

    python -m scripts.plot_bench

Reads only the JSON the suite wrote. No number is typed into this file — if a
chart disagrees with the text somewhere, the JSON is the arbiter and this
script is what has to change.

Charts are rendered on a fixed light card because they get embedded in a
theme-aware page: a plot that inverts with the theme either needs two renders
or ends up with grey-on-grey axes in one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
CHARTS = ROOT / "bench" / "charts"

INK = "#1b1d21"
MUTED = "#6b7077"
GRID = "#e2e0dc"
PAPER = "#fbfaf7"

# One colour per variant, held constant across every chart so the eye can
# carry a meaning from one to the next.
COLOR = {
    "fp32": "#3a5a8c",
    "fp16": "#c26b3d",
    "fp16+compile": "#5c8a6a",
    "persistence": "#a8a29a",
    "divergence": "#8c3a5a",
}
STAGE_COLOR = {
    "load": "#b8bec7",
    "read": "#8fa6c4",
    "forward": "#3a5a8c",
    "to_cpu": "#c26b3d",
    "write": "#d9b384",
}


def load(name: str):
    """Read a result file, dropping the cases that failed.

    A failed case is kept in the JSON on purpose — an out-of-memory at eight
    concurrent requests is a finding — but it has no timings to draw, so it is
    filtered out here rather than guarded against in every chart.
    """
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text())
    good = [r for r in records if r.get("kind") != "failed"]
    if len(good) != len(records):
        print(f"  ({len(records) - len(good)} failed case(s) in {name}.json, not plotted)")
    return good


def style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=18 if subtitle else 10,
                 fontweight="600")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9.5, va="bottom")


def save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(PAPER)
    fig.savefig(CHARTS / f"{name}.png", dpi=170, bbox_inches="tight",
                facecolor=PAPER)
    plt.close(fig)
    print(f"  {name}.png")


# --------------------------------------------------------------------------


def chart_stages(records) -> None:
    """Where one request's wall clock goes, by variant and by rollout length."""
    variants = ["fp32", "fp16", "fp16+compile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, steps in zip(axes, (1, 4)):
        bottoms = np.zeros(len(variants))
        for stage in ("load", "read", "forward", "to_cpu", "write"):
            vals = []
            for v in variants:
                rs = [r for r in records if r["variant"] == v and r["steps"] == steps]
                if stage == "load":
                    vals.append(np.mean([r["load_s"] for r in rs]))
                elif stage == "read":
                    vals.append(np.mean([r["read_s"] + r["index_s"] for r in rs]))
                else:
                    key = {"forward": "forward_s", "to_cpu": "to_cpu_s", "write": "write_s"}[stage]
                    vals.append(np.mean([sum(r[key]) for r in rs]))
            vals = np.array(vals)
            ax.bar(variants, vals, bottom=bottoms, color=STAGE_COLOR[stage],
                   label=stage, width=0.55, edgecolor=PAPER, linewidth=1.2)
            bottoms += vals

        for x, total in enumerate(bottoms):
            ax.text(x, total + max(bottoms) * 0.02, f"{total:.0f} s", ha="center",
                    color=INK, fontsize=10, fontweight="600")
        ax.set_ylim(0, max(bottoms) * 1.16)
        style(ax, f"{steps} step{'s' if steps > 1 else ''}  (+{6 * steps} h)",
              "mean over the four test inits, cold process")
        ax.set_ylabel("seconds", color=MUTED, fontsize=9.5)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, ncol=5,
                   loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.suptitle("A request, cold: the checkpoint load is the fixed cost, the forward is the variable one",
                 color=INK, fontsize=11.5, y=1.03, x=0.5)
    save(fig, "stages")


def chart_perstep(records) -> None:
    """Per-step forward cost, and what the compile warmup costs to get it."""
    variants = ["fp32", "fp16", "fp16+compile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    steady = [np.mean([r["forward_steady_s"] for r in records
                       if r["variant"] == v and r["steps"] == 4]) for v in variants]
    bars = ax.bar(variants, steady, color=[COLOR[v] for v in variants],
                  width=0.55)
    base = steady[0]
    for b, val in zip(bars, steady):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.15, f"{val:.2f} s",
                ha="center", color=INK, fontsize=10, fontweight="600")
        if val != base:
            ax.text(b.get_x() + b.get_width() / 2, val / 2,
                    f"{base / val:.2f}x", ha="center", color="white",
                    fontsize=12, fontweight="700")
    ax.set_ylim(0, max(steady) * 1.2)
    style(ax, "Steady-state forward, one rollout step",
          "median over steps 2..n, so no warmup is counted")
    ax.set_ylabel("seconds per step", color=MUTED, fontsize=9.5)

    ax = axes[1]
    for v in variants:
        rs = [r for r in records if r["variant"] == v and r["steps"] == 4]
        curves = np.array([r["forward_s"] for r in rs])
        mean = curves.mean(axis=0)
        ax.plot(range(1, len(mean) + 1), mean, "o-", color=COLOR[v], label=v,
                linewidth=2, markersize=5)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    style(ax, "The same, step by step",
          "log scale — the compile warmup is a different order of magnitude")
    ax.set_xlabel("rollout step", color=MUTED, fontsize=9.5)
    ax.set_ylabel("seconds", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "perstep")


def chart_memory(records) -> None:
    variants = ["fp32", "fp16", "fp16+compile"]
    vals = [np.mean([r["peak_gib"] for r in records if r["variant"] == v])
            for v in variants]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.barh(variants[::-1], vals[::-1], color=[COLOR[v] for v in variants[::-1]],
                   height=0.5)
    for b, val in zip(bars, vals[::-1]):
        ax.text(val + 0.3, b.get_y() + b.get_height() / 2, f"{val:.1f} GiB",
                va="center", color=INK, fontsize=10, fontweight="600")
    ax.axvline(31.7, color=COLOR["divergence"], linestyle="--", linewidth=1.2)
    ax.text(31.4, -0.45, "V100 32 GB", ha="right", color=COLOR["divergence"], fontsize=9)
    ax.set_xlim(0, 34)
    style(ax, "Peak VRAM", "headroom is what decides how many workers fit on one card")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    save(fig, "memory")


def chart_concurrency(records) -> None:
    """The measurement the microservice question actually turns on."""
    ns = sorted({r["concurrency"] for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    med, lo, hi = [], [], []
    for n in ns:
        rs = [r for r in records if r["concurrency"] == n]
        wall = [r["load_s"] + r["read_s"] + sum(r["forward_s"])
                + sum(r["to_cpu_s"]) + sum(r["write_s"]) for r in rs]
        med.append(np.median(wall))
        lo.append(min(wall))
        hi.append(max(wall))
    ax.fill_between(ns, lo, hi, color=COLOR["fp16"], alpha=0.15)
    ax.plot(ns, med, "o-", color=COLOR["fp16"], linewidth=2, markersize=6)
    ideal = med[0]
    ax.axhline(ideal, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ns[-1], ideal, "  latency of a request that has the box to itself",
            color=MUTED, fontsize=8.5, va="bottom", ha="right")
    for n, m in zip(ns, med):
        ax.text(n, m + (max(hi) - min(lo)) * 0.05, f"{m:.0f}s", ha="center",
                color=INK, fontsize=9.5, fontweight="600")
    ax.set_xticks(ns)
    ax.set_ylim(0, max(hi) * 1.2)
    style(ax, "Latency of one request while others run",
          "band is fastest to slowest request in the same batch; 4 GPUs")
    ax.set_xlabel("concurrent requests", color=MUTED, fontsize=9.5)
    ax.set_ylabel("seconds", color=MUTED, fontsize=9.5)

    ax = axes[1]
    thr = []
    for n in ns:
        rs = [r for r in records if r["concurrency"] == n]
        thr.append(n / (rs[0]["batch_wall_s"] / 60))
    ax.plot(ns, thr, "o-", color=COLOR["fp16+compile"], linewidth=2, markersize=6)
    ax.plot(ns, [thr[0] * n for n in ns], "--", color=MUTED, linewidth=1,
            label="perfect scaling")
    for n, t in zip(ns, thr):
        ax.text(n, t, f"  {t:.1f}", color=INK, fontsize=9.5, va="center")
    ax.set_xticks(ns)
    style(ax, "Throughput", "forecasts finished per minute, 4 steps each")
    ax.set_xlabel("concurrent requests", color=MUTED, fontsize=9.5)
    ax.set_ylabel("forecasts / minute", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    save(fig, "concurrency")


def chart_read(records) -> None:
    names = [r["query"] for r in records]
    secs = [r["seconds"] for r in records]
    waste = [r["waste"] for r in records]
    order = np.argsort(secs)
    names = [names[i] for i in order]
    secs = [secs[i] for i in order]
    waste = [waste[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    colors = [COLOR["divergence"] if w > 100 else COLOR["fp32"] for w in waste]
    bars = ax.barh(names, secs, color=colors, height=0.6)
    for b, s in zip(bars, secs):
        ax.text(s + max(secs) * 0.02, b.get_y() + b.get_height() / 2, f"{s:.2f} s",
                va="center", color=INK, fontsize=9.5)
    ax.set_xlim(0, max(secs) * 1.2)
    style(ax, "How long the archive takes to answer",
          "warm page cache; 50 GB store, 251 GB of RAM on this box")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlabel("seconds", color=MUTED, fontsize=9.5)

    ax = axes[1]
    bars = ax.barh(names, waste, color=colors, height=0.6)
    ax.set_xscale("log")
    for b, w in zip(bars, waste):
        ax.text(w * 1.3, b.get_y() + b.get_height() / 2,
                "1x" if w < 1.5 else f"{w:,.0f}x", va="center", color=INK, fontsize=9.5)
    ax.set_xlim(0.7, max(waste) * 12)
    ax.set_yticklabels([])
    style(ax, "Bytes read per byte returned",
          "chunking decides this, not disk speed")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    save(fig, "read")


def chart_accuracy(records) -> None:
    """The chart that settles the 6-8% question."""
    fields = ["2t", "msl", "t500"]
    units = {"2t": "K", "msl": "Pa", "t500": "K"}
    tags = sorted({r["tag"] for r in records})
    tag = tags[0]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, field in zip(axes, fields):
        rs = sorted([r for r in records if r["field"] == field and r["tag"] == tag],
                    key=lambda r: r["lead_h"])
        leads = [r["lead_h"] for r in rs]
        ax.plot(leads, [r["rmse_persistence"] for r in rs], color=COLOR["persistence"],
                linewidth=1.6, linestyle="--", label="persistence")
        ax.plot(leads, [r["rmse_ref"] for r in rs], color=COLOR["fp32"],
                linewidth=2, label="fp32 vs ERA5")
        ax.plot(leads, [r["rmse_cand"] for r in rs], color=COLOR["fp16"],
                linewidth=2, linestyle=(0, (4, 2)), label="fp16 vs ERA5")
        ax.plot(leads, [r["divergence"] for r in rs], color=COLOR["divergence"],
                linewidth=2, label="fp32 vs fp16")
        style(ax, f"{field}  ({units[field]})",
              "" if field != "2t" else "init 2026-05-01, 40 steps")
        ax.set_xlabel("lead time, hours", color=MUTED, fontsize=9.5)
    axes[0].set_ylabel("latitude-weighted RMSE", color=MUTED, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("The fp32 and fp16 curves lie on top of each other; the distance between them is the bottom line",
                 color=INK, fontsize=11.5, y=1.02)
    save(fig, "accuracy")


def chart_divergence_ratio(records) -> None:
    tags = sorted({r["tag"] for r in records})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    styles = ["-", "--"]
    for tag, ls in zip(tags, styles):
        for field, color in (("2t", COLOR["fp32"]), ("msl", COLOR["fp16"]),
                             ("t500", COLOR["fp16+compile"])):
            rs = sorted([r for r in records if r["field"] == field and r["tag"] == tag],
                        key=lambda r: r["lead_h"])
            ax.plot([r["lead_h"] for r in rs],
                    [100 * r["divergence"] / r["rmse_ref"] for r in rs],
                    ls, color=color, linewidth=1.8,
                    label=f"{field}, {tag}" if len(tags) > 1 else field)
    ax.axhline(100, color=COLOR["divergence"], linewidth=1.2)
    ax.text(6, 101, "the forecast's own error against reality", color=COLOR["divergence"],
            fontsize=9, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 200)
    style(ax, "How far fp16 is from fp32, as a share of how far both are from the truth",
          "this is the number that reads as 6-8% — and this is what it is a share of")
    ax.set_xlabel("lead time, hours", color=MUTED, fontsize=9.5)
    ax.set_ylabel("% of the field's own RMSE", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="upper left")
    save(fig, "divergence")


def chart_service(records) -> None:
    """What the running service costs its caller, cold and warm, 1 worker vs 4.

    The left panel is the argument against process-per-request and against a
    container that gets torn down between jobs: the first request pays the
    checkpoint load, every later one does not, and the difference is the whole
    of that gap. The right panel is whether adding workers adds throughput —
    if it scales, the current process model already has what a microservice
    split is supposed to provide.
    """
    ok = [r for r in records if r.get("status") == "done"]
    counts = sorted({r["workers"] for r in ok})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    width = 0.36
    x = np.arange(len(counts))
    for offset, (cold, color, label) in enumerate((
            (True, COLOR["fp32"], "first request (loads the checkpoint)"),
            (False, COLOR["fp16"], "every request after"))):
        vals = [np.median([r["latency_s"] for r in ok
                           if r["workers"] == n and r["cold"] is cold] or [0])
                for n in counts]
        pos = x + (offset - 0.5) * width
        ax.bar(pos, vals, width, color=color, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v, f"{v:.0f}s", ha="center", va="bottom", color=INK,
                    fontsize=9.5, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} worker{'s' if n > 1 else ''}" for n in counts])
    ax.xaxis.grid(False)
    style(ax, "What the caller waits",
          "4-step forecast, fp16; the model stays resident between requests")
    ax.set_ylabel("seconds", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")

    ax = axes[1]
    thr = [n * 60 / np.median([r["latency_s"] for r in ok
                               if r["workers"] == n and not r["cold"]])
           for n in counts]
    ax.plot(counts, thr, "o-", color=COLOR["fp16+compile"], linewidth=2, markersize=6)
    ax.plot(counts, [thr[0] * n / counts[0] for n in counts], "--", color=MUTED,
            linewidth=1, label="one GPU's rate, times the number of GPUs")
    for n, t in zip(counts, thr):
        ax.text(n, t, f"  {t:.1f}", color=INK, fontsize=9.5, va="center")
    ax.set_xticks(counts)
    if len(counts) == 1:
        # One point is not a scaling curve, and saying so on the chart is
        # better than letting the dashed ideal line look like a measurement.
        # The missing point needs three GPUs that another tenant is holding.
        ideal = thr[0] * 4 / counts[0]
        ax.plot([counts[0], 4], [thr[0], ideal], "--", color=MUTED, linewidth=1)
        ax.plot([4], [ideal], "o", color=PAPER, markeredgecolor=MUTED, markersize=6)
        ax.set_xlim(0.5, 4.6)
        ax.set_ylim(0, ideal * 1.25)
        ax.set_xticks([1, 4])
        ax.annotate("not measured: three of the four\nGPUs were held by another user",
                    xy=(4, ideal), xytext=(1.35, ideal * 0.78),
                    color=MUTED, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    style(ax, "Throughput of the service as it stands",
          "warm requests only, so no checkpoint load is counted")
    ax.set_xlabel("resident workers, one per GPU", color=MUTED, fontsize=9.5)
    ax.set_ylabel("forecasts / minute", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    save(fig, "service")


def main() -> int:
    print("charts:")
    stages = load("stages")
    if stages:
        chart_stages(stages)
        chart_perstep(stages)
        chart_memory(stages)
    svc = load("service")
    if svc:
        chart_service(svc)
    conc = load("concurrency")
    if conc:
        chart_concurrency(conc)
    rd = load("read")
    if rd:
        chart_read(rd)
    acc = load("accuracy")
    if acc:
        chart_accuracy(acc)
        chart_divergence_ratio(acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figures for the submission report, drawn from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
INK, MUTED, LINE, ACCENT, CPU, AMBER = "#0F1B26", "#56687A", "#D7DEE4", "#0B6E63", "#8A97A5", "#B7791F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

# ---- 1. pipeline -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.5), dpi=200)
ax.set_xlim(0, 110); ax.set_ylim(0, 24); ax.axis("off")
boxes = [("Transfer stream", "NIP / ISO 20022\ncopy of the\npayment feed", "#EAEEF1"),
         ("GPU window graph", "last 96 hours,\nrebuilt every\nmicro-batch (~1 ms)", "#EAEEF1"),
         ("4 CUDA detectors", "velocity, pass-through,\nscatter-gather,\ncycles", "#D6EEEA"),
         ("AI risk model", "gradient-boosted\ntrees on GPU,\nSHAP reasons", "#D6EEEA"),
         ("Rings & actions", "linked rings, tiers,\nGenAI STR\ndraft", "#FBE9DD")]
w, gap, y, h = 18.6, 3.6, 2, 20
for i, (title, sub, c) in enumerate(boxes):
    x = 1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=c, ec=LINE, lw=1))
    ax.text(x + w / 2, y + h - 4, title, ha="center", va="center", fontsize=9.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + (h - 6.5) / 2, sub, ha="center", va="center", fontsize=8.4, color=MUTED, linespacing=1.45)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.5, y + h / 2), (x + w + gap - 0.5, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.3))
fig.savefig(OUT / "pipeline.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 2. ablation -------------------------------------------------------------------------------
tr = json.loads((R / "train_results.json").read_text())["ladder"]
names = list(tr.keys())
labels = ["Transfer fields only", "+ account velocity", "+ pass-through", "+ scatter-gather", "+ temporal cycles (full)"]
vals = [tr[n]["f1"][0] * 100 for n in names]
errs = [tr[n]["f1"][1] * 100 for n in names]
fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
ys = list(range(len(vals)))[::-1]
ax.barh(ys, vals, xerr=errs, color=[CPU] * 4 + [ACCENT], height=0.6, error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
for y_, v in zip(ys, vals):
    ax.text(v + 1.2, y_, f"{v:.1f}%", va="center", fontsize=9, weight="bold")
for v in (64.77, 56.77):
    ax.axvline(v, color=INK, ls="--", lw=0.8, alpha=0.6)
ax.text(65.6, len(vals) - 0.45, "best published\n(IBM, 2024) 64.8", ha="left", fontsize=7.2, color=MUTED, va="center")
ax.text(56.0, len(vals) - 0.45, "graph neural\nnetwork 56.8", ha="right", fontsize=7.2, color=MUTED, va="center")
ax.set_ylim(-0.6, len(vals) + 0.1)
ax.set_yticks(ys, labels); ax.set_xlim(0, 82); ax.set_xlabel("Laundering-transaction F1, test split (%)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True)
fig.savefig(OUT / "ablation.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 3. throughput -----------------------------------------------------------------------------
b = json.loads((R / "bench.json").read_text())["rows"]
fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
xs = [r["batch"] for r in b]
ax.plot(xs, [r["gpu_txn_s"] for r in b], "-o", color=ACCENT, lw=2, ms=4, label="GPU (A100, CUDA kernels)")
ax.plot(xs, [r["cpu_txn_s"] for r in b], "-o", color=CPU, lw=2, ms=4, label="CPU (8 cores, same algorithm)")
ax.axhline(11.2e9 / 365 / 86400 * 10, color=AMBER, ls="--", lw=1)
ax.text(140, 11.2e9 / 365 / 86400 * 10 * 1.25, "10× Nigeria's average e-payment rate (2024)", color=AMBER, fontsize=7.5)
for r in (b[2], b[-1]):
    lab = f"{r['gpu_txn_s'] / 1e6:.2f}M/s" if r["gpu_txn_s"] >= 1e6 else f"{r['gpu_txn_s'] / 1e3:,.0f}k/s"
    ax.annotate(f"{lab}\n×{r['speedup']:.0f} vs CPU", (r["batch"], r["gpu_txn_s"]), textcoords="offset points",
                xytext=(-6, 8), ha="right", fontsize=8, color=ACCENT, weight="bold")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(xs, [f"{x:,}" for x in xs])
ax.set_xlabel("Transfers per micro-batch"); ax.set_ylabel("Transfers scored per second")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=LINE, lw=0.6, which="major"); ax.set_axisbelow(True)
fig.savefig(OUT / "throughput.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))

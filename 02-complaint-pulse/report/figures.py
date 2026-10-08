"""Figures for the ComplaintPulse submission, drawn from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
INK, MUTED, LINE, ACCENT, GREY, HOT = "#121827", "#58637A", "#D9DDE8", "#3D4FB8", "#9AA3B8", "#C2410C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

# ---- 1. pipeline ----
fig, ax = plt.subplots(figsize=(11, 2.6), dpi=200)
ax.set_xlim(0, 110); ax.set_ylim(0, 24); ax.axis("off")
boxes = [("All channels", "calls, chat, social,\nemail, reviews,\nbranch · 5 languages", "#ECEEF5"),
         ("Redact + N-ATLaS", "PII masked; issue,\nEnglish summary,\namount, urgency", "#E3E7FA"),
         ("Topics + K3", "multilingual topics;\nviral reposts\ncounted once", "#E3E7FA"),
         ("K1 burst cube", "33,516 series\nscored every\n15 minutes (GPU)", "#E3E7FA"),
         ("K2 root cause", "every attribute pair;\nlinked operational\nevent", "#E3E7FA"),
         ("Brief + message", "N-ATLaS brief for the\nowner; message in the\ncustomer's language", "#FBE9DD")]
w, gap, y, h = 15.6, 2.8, 2, 20
for i, (title, sub, c) in enumerate(boxes):
    x = 1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=c, ec=LINE, lw=1))
    ax.text(x + w / 2, y + h - 4, title, ha="center", va="center", fontsize=8.8, weight="bold", color=INK)
    ax.text(x + w / 2, y + (h - 6.5) / 2, sub, ha="center", va="center", fontsize=7.6, color=MUTED, linespacing=1.45)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.4, y + h / 2), (x + w + gap - 0.4, y + h / 2), arrowstyle="-|>",
                                     mutation_scale=11, color=MUTED, lw=1.2))
fig.savefig(OUT / "pipeline.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 2. detection timeline: ours vs daily-count rule ----
d = json.loads((R / "detect_heldout_gpu.json").read_text())
rows = d["incidents"]
short = ["OTP failure after app release", "Reversals stuck on one switch", "ATM cluster in Kano", "USSD timeouts on one telco",
         "Card declines after maintenance", "POS debits in Lagos", "Unauthorised debits (fraud)", "SMS charge error (students)"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for k, r in enumerate(rows):
    yk = len(rows) - 1 - k
    peak = (r["peak"] - r["start"]) / 3600
    nv = r["naive_hours_to_detect"]
    if nv is not None:
        ax.barh(yk - 0.18, nv, height=0.32, color=GREY)
    if r["detected"]:
        ax.barh(yk + 0.18, r["hours_to_detect"], height=0.32, color=ACCENT)
        ax.text(r["hours_to_detect"] + 0.4, yk + 0.18, f"{r['hours_to_detect']:.1f} h", va="center", fontsize=7.5, color=ACCENT, weight="bold")
    ax.plot([peak], [yk], marker="v", color=HOT, ms=7)
ax.set_yticks(range(len(rows)), short[::-1] if len(short) == len(rows) else [r["incident"][:30] for r in rows][::-1], fontsize=8)
ax.set_xlabel("Hours after the incident started (held-out stream)")
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=ACCENT, label="ComplaintPulse alert"), Patch(color=GREY, label="Daily-count rule alert"),
                   Line2D([], [], marker="v", ls="", color=HOT, label="Incident peak")],
          frameon=True, facecolor="white", edgecolor=LINE, fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True)
fig.savefig(OUT / "timeline.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 3. kernel speed-ups ----
b = json.loads((R / "bench_kernels.json").read_text())
fig, axs = plt.subplots(1, 3, figsize=(9, 2.7), dpi=200)
for ax, (key, xk, lab) in zip(axs, [("k1", "series", "K1 burst cube (series)"), ("k2", "pairs", "K2 root cause (attribute pairs)"),
                                    ("k3", "docs", "K3 MinHash (posts)")]):
    rr = b[key]
    xs = [r[xk] for r in rr]
    ax.plot(xs, [r["cpu_ms"] for r in rr], "-o", color=GREY, ms=3.5, label="CPU")
    ax.plot(xs, [r["gpu_ms"] for r in rr], "-o", color=ACCENT, ms=3.5, label="A100 GPU")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(lab, fontsize=9)
    last = rr[-1]
    ax.annotate(f"×{last['speedup']:,.0f}", (xs[-1], last["gpu_ms"]), textcoords="offset points", xytext=(-4, 10), ha="right",
                fontsize=9, color=ACCENT, weight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color=LINE, lw=0.5); ax.tick_params(labelsize=7)
axs[0].set_ylabel("milliseconds per run"); axs[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "kernels.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))

"""Generate the architecture and timing figures.

Three figures, each answering a different question:

  fig_split_principle  the partitioning *principle* -- conceptual: which work
                       stays on the RTOS node and which may go to the cloud
  fig_topology         the *implemented* topology -- concrete task names,
                       priorities and AWS resources
  fig_timeline         the timeline of one window round trip, which makes the
                       staleness of a correction visible; drawn from the E4b
                       measurements

Run:  python gen_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "#2b6cb0"
CLOUD = "#276749"
NEUTRAL = "#4a5568"
WARN = "#c53030"


def box(ax, x, y, w, h, text, color, fontsize=9, bold=False, alpha=0.10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        linewidth=1.6, edgecolor=color, facecolor=color, alpha=alpha, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        linewidth=1.6, edgecolor=color, facecolor="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1a202c", zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, xy_from, xy_to, color=NEUTRAL, text=None, style="-|>", offset=0.0):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=16,
        linewidth=1.6, color=color, zorder=5,
        connectionstyle="arc3,rad=%.2f" % offset))
    if text:
        mx = (xy_from[0] + xy_to[0]) / 2
        my = (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + 0.022, text, ha="center", va="bottom",
                fontsize=8, color=color, zorder=6)


# ------------------------------------------------------- split principle
def fig_split_principle():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.95, "Work-partitioning principle", ha="center",
            fontsize=13, fontweight="bold", color="#1a202c")

    box(ax, 0.28, 0.74, 0.44, 0.13,
        "Does this task have\na hard deadline?",
        NEUTRAL, fontsize=10, bold=True)

    arrow(ax, (0.42, 0.74), (0.24, 0.60), DEVICE)
    arrow(ax, (0.58, 0.74), (0.76, 0.60), CLOUD)
    ax.text(0.30, 0.665, "YES", fontsize=10, fontweight="bold", color=DEVICE)
    ax.text(0.68, 0.665, "NO", fontsize=10, fontweight="bold", color=CLOUD)

    box(ax, 0.04, 0.34, 0.42, 0.26,
        "STAYS ON THE DEVICE\n\n"
        "• 200 Hz inertial sampling\n"
        "• attitude filter\n"
        "• strapdown integration\n\n"
        "measured cost: 3.5% of the budget",
        DEVICE, fontsize=9)

    box(ax, 0.54, 0.34, 0.42, 0.26,
        "GOES TO THE CLOUD\n\n"
        "• bias estimation (ESKF)\n"
        "• 15 states, 15×15 covariance\n\n"
        "does not fit on-device at 21–27 states\n"
        "measured cost: 221 ms in Lambda",
        CLOUD, fontsize=9)

    box(ax, 0.04, 0.09, 0.92, 0.17,
        "The criterion is NOT complexity or required compute.\n"
        "A network gives no timing guarantee — so every task with a deadline stays local,\n"
        "and the cloud only provides work the system can continue without.",
        WARN, fontsize=9.5, alpha=0.06)

    fig.tight_layout()
    p = OUT / "fig_split_principle.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------- topology
def fig_topology():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, "Implemented topology", ha="center",
            fontsize=13, fontweight="bold", color="#1a202c")

    # --- device ---
    ax.add_patch(FancyBboxPatch((0.02, 0.16), 0.40, 0.72,
                                boxstyle="round,pad=0.014", linewidth=2.0,
                                edgecolor=DEVICE, facecolor=DEVICE,
                                alpha=0.05, zorder=0))
    ax.text(0.22, 0.845, "Raspberry Pi Pico W  ·  FreeRTOS V11.1.0",
            ha="center", fontsize=10, fontweight="bold", color=DEVICE)
    ax.text(0.22, 0.815, "RP2040, 2× Cortex-M0+ @125 MHz, no FPU",
            ha="center", fontsize=8, color=NEUTRAL)

    box(ax, 0.05, 0.60, 0.34, 0.19,
        "task  rt   ·  priority 5\n"
        "200 Hz  ·  HARD DEADLINE 5 ms\n\n"
        "vIMU sample → Mahony → strapdown\n"
        "measured cost: 179 µs mean",
        DEVICE, fontsize=8.5)

    box(ax, 0.05, 0.40, 0.34, 0.16,
        "task  net  ·  priority 1\n\n"
        "mbedTLS 3.6  ·  MQTT  ·  lwIP\n"
        "16 kB stack from the kernel heap",
        DEVICE, fontsize=8.5)

    box(ax, 0.05, 0.21, 0.34, 0.15,
        "ring buffer  ·  2 s window = 400 samples\n"
        "correction applied by timestamp",
        DEVICE, fontsize=8.5)

    # --- cloud ---
    ax.add_patch(FancyBboxPatch((0.56, 0.16), 0.42, 0.72,
                                boxstyle="round,pad=0.014", linewidth=2.0,
                                edgecolor=CLOUD, facecolor=CLOUD,
                                alpha=0.05, zorder=0))
    ax.text(0.77, 0.845, "AWS  ·  region eu-north-1",
            ha="center", fontsize=10, fontweight="bold", color=CLOUD)

    box(ax, 0.59, 0.62, 0.36, 0.16,
        "AWS IoT Core\n"
        "MQTT broker  ·  mutual TLS\n"
        "Thing: picoW_thing  ·  X.509 cert RSA-2048",
        CLOUD, fontsize=8.5)

    box(ax, 0.59, 0.44, 0.36, 0.13,
        "IoT Rule\n"
        "SELECT * FROM 'imu/+/window'",
        CLOUD, fontsize=8.5)

    box(ax, 0.59, 0.21, 0.36, 0.18,
        "AWS Lambda  ·  picoW_estimator\n"
        "Python 3.12  ·  1769 MB  ·  x86_64\n\n"
        "15-state ESKF\n"
        "measured cost: 221 ms",
        CLOUD, fontsize=8.5)

    # --- links ---
    arrow(ax, (0.39, 0.50), (0.59, 0.68), NEUTRAL,
          "MQTT/TLS  ·  7.5 kB", offset=-0.12)
    arrow(ax, (0.59, 0.30), (0.39, 0.30), WARN,
          "correction + valid_at_us", offset=-0.12)
    arrow(ax, (0.77, 0.62), (0.77, 0.575), CLOUD)
    arrow(ax, (0.77, 0.44), (0.77, 0.395), CLOUD)

    ax.text(0.22, 0.10, "keeps working without the network", ha="center",
            fontsize=9, style="italic", color=DEVICE)
    ax.text(0.77, 0.10, "stateless  ·  scalable", ha="center",
            fontsize=9, style="italic", color=CLOUD)

    fig.tight_layout()
    p = OUT / "fig_topology.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------- timeline
def fig_timeline():
    """Timeline of one window round trip.

    Drawn as two panels because a single to-scale axis makes the point badly:
    the window is 2000 ms and the round trip totals 703 ms, so on
    one axis they collapse into an illegible sliver. The top panel keeps the
    true proportions -- which is itself the finding -- and the bottom panel
    expands the tail so the breakdown can be read.
    """
    import matplotlib.patches as mpatches

    # Values measured directly in E4b (docs/e4b_results.txt). Network, broker
    # and TLS are given as a single term because the measurement cannot split
    # them, and halving it would assume a symmetry that the 6.6 kB / 1.5 kB
    # payloads do not have.
    win, comp, rest = 2000.0, 277.6, 424.7
    trip = comp + rest
    t_apply = win + trip

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.6),
                                   gridspec_kw={"height_ratios": [1, 1.15]})
    fig.suptitle("Timeline of one window  —  correction staleness",
                 fontsize=13, fontweight="bold", y=0.99)

    def seg(ax, x0, w, color, y=0.30, h=0.34, alpha=0.20):
        ax.add_patch(mpatches.Rectangle((x0, y), w, h, facecolor=color,
                                        edgecolor=color, alpha=alpha, lw=1.6))
        ax.add_patch(mpatches.Rectangle((x0, y), w, h, facecolor="none",
                                        edgecolor=color, lw=1.6))

    # ---------------- panel 1: true proportions ----------------
    ax1.set_xlim(-60, t_apply + 60); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.text(-40, 0.88, "True proportions", fontsize=10,
             fontweight="bold", color="#1a202c")

    seg(ax1, 0, win, DEVICE)
    seg(ax1, win, trip, WARN)
    ax1.text(win / 2, 0.47, "window collection", ha="center",
             fontsize=10, fontweight="bold")
    ax1.text(win / 2, 0.38, "2000 ms  ·  74% of the staleness", ha="center",
             fontsize=9, color=NEUTRAL)
    ax1.text(win + trip / 2, 0.72, "network + cloud\n703 ms  ·  26%", ha="center",
             fontsize=9, color=WARN, fontweight="bold")
    ax1.annotate("", xy=(win + trip / 2, 0.66), xytext=(win + trip / 2, 0.65),
                 arrowprops=dict(arrowstyle="-", color=WARN, lw=1.2))

    ax1.annotate("", xy=(0, 0.20), xytext=(t_apply, 0.20),
                 arrowprops=dict(arrowstyle="<|-|>", color="#1a202c", lw=1.5))
    ax1.text(t_apply / 2, 0.10,
             "the oldest sample in the window is 2703 ms old when the correction arrives",
             ha="center", fontsize=9)

    # ---------------- panel 2: tail expanded ----------------
    ax2.set_xlim(win - 120, t_apply + 340); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.text(win - 100, 0.98, "Tail, expanded", fontsize=10,
             fontweight="bold", color="#1a202c")

    x = win
    for w, color, lab, sub in [
            (rest, NEUTRAL, "network + broker + TLS", "425 ms  ·  not separable"),
            (comp, CLOUD, "ESKF in Lambda", "278 ms @1769 MB")]:
        seg(ax2, x, w, color, y=0.42, h=0.24)
        ax2.text(x + w / 2, 0.72, lab, ha="center", fontsize=9,
                 fontweight="bold", rotation=0)
        ax2.text(x + w / 2, 0.34, sub, ha="center", fontsize=8.5,
                 color=NEUTRAL)
        x += w

    ax2.plot([win, win], [0.20, 0.80], color=WARN, lw=2.0, ls="--")
    ax2.text(win - 10, 0.86, "valid_at_us", ha="right", fontsize=9,
             color=WARN, fontweight="bold")
    ax2.text(win - 10, 0.79, "(the instant the correction describes)", ha="right",
             fontsize=8, color=WARN)

    ax2.plot([t_apply, t_apply], [0.20, 0.80], color=WARN, lw=2.0)
    ax2.text(t_apply + 15, 0.72, "correction applied", ha="left",
             fontsize=9, color=WARN, fontweight="bold")

    ax2.annotate("", xy=(win, 0.26), xytext=(t_apply, 0.26),
                 arrowprops=dict(arrowstyle="<|-|>", color=WARN, lw=1.8))
    ax2.text((win + t_apply) / 2, 0.15,
             "703 ms stale at the moment it is applied", ha="center",
             fontsize=9.5, color=WARN, fontweight="bold")

    for t in (win, win + rest, t_apply):
        ax2.plot([t, t], [0.38, 0.42], color="#1a202c", lw=1.0)
        ax2.text(t, 0.05, "%d ms" % t, ha="center", fontsize=8, color=NEUTRAL)

    fig.text(0.5, 0.015,
             "The window dominates at 74%: making the cloud faster barely reduces staleness.\n"
             "Network and broker time falls before and after the computation; only the sum is measured, so it is drawn as one block.",
             ha="center", fontsize=9.5, style="italic", color="#1a202c")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    p = OUT / "fig_timeline.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_split_principle, fig_topology, fig_timeline):
        print("  wrote:", fn().name)

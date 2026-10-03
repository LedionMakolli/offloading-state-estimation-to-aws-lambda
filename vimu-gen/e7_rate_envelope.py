"""E7 -- Deriving the feasible loop rate instead of assuming one.

The 5 ms deadline used throughout this work is 1/200 Hz, and until now 200 Hz
was simply asserted. That is a weak point: an examiner is entitled to ask why
not 100 Hz, or 1000 Hz. This experiment replaces the assumption with an
analysis.

Every constraint below is computed from something already measured on the
hardware, or from a published device limit -- nothing is invented:

  C1  Tier-0 CPU cost        measured in E3/E5 (179 us mean, 319 us worst case
                             under crypto contention)
  C2  scheduler granularity  FreeRTOS tick is 1 kHz; a period of N ticks has
                             quantisation of roughly 1/N of the period
  C3  sensor output rate     MPU-6050 datasheet limit
  C4  message size           AWS IoT Core 128 kB cap, ~32 wire bytes per sample
  C5  integration error      measured convergence table from E2
  C6  signal bandwidth       Nyquist against the dynamics being tracked

The output is the band of rates that satisfy all of them simultaneously, and
where the chosen rate sits inside it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- measured ---
TIER0_MEAN_US = 179.0      # E5, all phases
TIER0_MAX_US = 319.0       # E5, worst case during crypto contention
TICK_HZ = 1000.0           # FreeRTOS configTICK_RATE_HZ
MPU6050_MAX_HZ = 1000.0    # datasheet, with DLPF enabled
IOT_LIMIT_B = 128 * 1024
WIRE_B_PER_SAMPLE = 32     # 6 int16 = 12 B x2 sensors = 24 B, x4/3 for base64
WINDOW_S = 2.0

# E2 convergence: final position error of the noise-free mechanisation over 30 s
E2_FS = np.array([50, 100, 200, 400, 800, 1600], dtype=float)
E2_ERR_M = np.array([1.316e-2, 3.292e-3, 8.233e-4, 2.059e-4, 5.147e-5, 1.287e-5])

# Policy thresholds -- these are engineering choices and are stated as such.
TIER0_BUDGET_FRAC = 0.25   # Tier-0 may occupy at most this share of the period
MIN_TICKS_PER_PERIOD = 5   # so tick quantisation stays ~20% of one period
DYNAMICS_HZ = 20.0         # attitude dynamics to be tracked
NYQUIST_MARGIN = 5.0       # sample at least this many times the signal bandwidth


def constraints(fs: np.ndarray) -> dict[str, np.ndarray]:
    """True where the rate fs satisfies each constraint."""
    period_us = 1e6 / fs
    return {
        "C1 CPU (Tier 0 <= 25% of the period)":
            TIER0_MAX_US <= TIER0_BUDGET_FRAC * period_us,
        # Two distinct things were previously conflated under one heading.
        #
        # C2a is a hard consequence of the tick: xTaskDelayUntil expresses the
        # period in whole ticks, so only rates of TICK_HZ/N are representable
        # exactly. At a 1 kHz tick that admits 1000, 500, 250, 200, 125, 100 Hz
        # but not, say, 300 Hz.
        #
        # C2b is a policy margin, not a scheduler limit. Note that the measured
        # jitter in E5 was 10 us against a 1000 us tick, so tick granularity is
        # demonstrably NOT the dominant jitter source; this margin is
        # conservative and is stated as a choice.
        "C2a period = whole number of ticks":
            np.isclose(TICK_HZ / fs, np.round(TICK_HZ / fs), rtol=1e-9, atol=1e-6),
        "C2b policy margin (>= %d ticks/period)" % MIN_TICKS_PER_PERIOD:
            fs <= TICK_HZ / MIN_TICKS_PER_PERIOD,
        "C3 sensor limit (MPU-6050)":
            fs <= MPU6050_MAX_HZ,
        "C4 message size (IoT Core 128 kB)":
            fs * WINDOW_S * WIRE_B_PER_SAMPLE <= IOT_LIMIT_B,
        "C6 Nyquist (>= 5x the 20 Hz dynamics)":
            fs >= DYNAMICS_HZ * NYQUIST_MARGIN,
    }


def main() -> None:
    # Include the round candidate rates exactly, so a boundary case is not
    # missed by the log grid stepping over it.
    fs = np.unique(np.concatenate([
        np.logspace(np.log10(20), np.log10(4000), 2000),
        np.array([50, 100, 125, 200, 250, 400, 500, 800, 1000, 2000], float)]))
    cons = constraints(fs)
    feasible = np.ones_like(fs, dtype=bool)
    for m in cons.values():
        feasible &= m

    lo = fs[feasible].min()
    hi = fs[feasible].max()

    print("E7 - Feasible loop-rate analysis")
    print("=" * 66)
    print("\n  Constraints, each from a measured or published quantity:\n")
    for name, mask in cons.items():
        ok = fs[mask]
        if ok.size == 0:
            rng = "none"
        elif ok.min() <= fs.min() * 1.01:
            rng = "<= %.0f Hz" % ok.max()
        elif ok.max() >= fs.max() * 0.99:
            rng = ">= %.0f Hz" % ok.min()
        else:
            rng = "%.0f - %.0f Hz" % (ok.min(), ok.max())
        print("    %-46s %s" % (name, rng))

    print("\n  Band satisfying all constraints at once: %.0f - %.0f Hz" % (lo, hi))
    print("  Chosen rate: 200 Hz  ->  %s" %
          ("inside the band" if lo <= 200 <= hi else "OUTSIDE THE BAND"))

    print("\n  Which constraint closes the upper side?")
    upper = {n: fs[m].max() for n, m in cons.items() if fs[m].size and fs[m].max() < fs.max() * 0.99}
    for n, v in sorted(upper.items(), key=lambda kv: kv[1]):
        print("    %-46s %.0f Hz" % (n, v))

    print("\n  Tier-0 cost by rate (from the 319 us worst case):")
    for f in (50, 100, 200, 400, 500, 1000):
        print("    %5d Hz  period %6.2f ms  ->  %5.1f %% of the budget"
              % (f, 1000 / f, 100 * TIER0_MAX_US / (1e6 / f)))

    print("\n  Integration error over 30 s (measured in E2):")
    for f, e in zip(E2_FS, E2_ERR_M):
        print("    %5d Hz  %.2e m" % (f, e))
    print("    -> irrelevant as a constraint: even 50 Hz gives 1.3 cm over 30 s,")
    print("       while drift caused by sensor noise is measured in metres.")

    # ------------------------------------------------------------------ plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [1.25, 1]})
    fig.suptitle("E7 — Feasible loop-rate envelope", fontsize=13,
                 fontweight="bold")

    names = list(cons.keys())
    for i, name in enumerate(names):
        m = cons[name]
        y = len(names) - i
        ax1.plot(fs, np.where(m, y, np.nan), lw=7, solid_capstyle="butt",
                 color="#2b6cb0", alpha=0.75)
        ax1.plot(fs, np.where(~m, y, np.nan), lw=7, solid_capstyle="butt",
                 color="#c53030", alpha=0.25)
        ax1.text(21, y + 0.28, name, fontsize=8.5, color="#1a202c")

    ax1.axvspan(lo, hi, color="#276749", alpha=0.10, zorder=0)
    ax1.axvline(200, color="#276749", lw=2.0, ls="--", zorder=5)
    ax1.text(205, 0.35, "200 Hz\nchosen", fontsize=9, color="#276749",
             fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(20, 4000)
    ax1.set_ylim(0, len(names) + 0.9)
    ax1.set_yticks([])
    ax1.set_xlabel("loop rate [Hz]")
    ax1.grid(True, axis="x", which="both", alpha=0.3)
    ax1.set_title("blue = constraint met    red = violated    "
                  "green = feasible band (%.0f–%.0f Hz)" % (lo, hi),
                  fontsize=9.5)

    f2 = np.logspace(np.log10(20), np.log10(4000), 500)
    ax2.semilogx(f2, 100 * TIER0_MAX_US / (1e6 / f2), color="#c53030", lw=2,
                 label="Tier 0, worst case (319 µs)")
    ax2.semilogx(f2, 100 * TIER0_MEAN_US / (1e6 / f2), color="#2b6cb0", lw=2,
                 ls="--", label="Tier 0, mean (179 µs)")
    ax2.axhline(100 * TIER0_BUDGET_FRAC, color="#4a5568", ls=":", lw=1.5)
    ax2.text(22, 26, "25% policy limit", fontsize=8, color="#4a5568")
    ax2.axhline(100, color="k", lw=1.2)
    ax2.axvspan(lo, hi, color="#276749", alpha=0.10, zorder=0)
    ax2.axvline(200, color="#276749", lw=2.0, ls="--")
    ax2.set_xlim(20, 4000)
    ax2.set_ylim(0, 130)
    ax2.set_xlabel("loop rate [Hz]")
    ax2.set_ylabel("Tier-0 load [% of the period]")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "e7_rate_envelope.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\n  figure: %s" % p.name)


if __name__ == "__main__":
    main()

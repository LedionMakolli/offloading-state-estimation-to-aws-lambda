"""Analyse the E4b capture: is the round trip continuous or quantised?

Reading a mean off the table would hide the most informative thing in it. The
measured values cluster: 566-585, 674-687, 788-801, 897. If those clusters are
separated by a constant, the round trip is not varying smoothly with load -- it
is taking a discrete number of network round trips, and the step size is one of
them. That distinction matters, because a quantised delay is predictable in a
way a continuous one is not, and an estimator can be designed around it.

    python e4b_analyse.py > ../docs/e4b_analysis.txt
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path(__file__).resolve().parents[1] / "docs"
SRC = DOCS / "e4b_results.txt"
FIG = DOCS / "figures" / "e4b_rtt_quantised.png"

ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(yes|no)\s*(.*)$")


def parse() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rtt, handler, cold = [], [], []
    for line in SRC.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW.match(line)
        if m:
            rtt.append(float(m.group(2)))
            handler.append(float(m.group(3)))
            cold.append(m.group(5) == "yes")
    return np.array(rtt), np.array(handler), np.array(cold, dtype=bool)


def main() -> None:
    rtt, handler, cold = parse()
    warm = rtt[~cold]
    print("E4b - is the round trip quantised?")
    print("=" * 66)
    print("  samples         : %d  (%d warm, %d cold)" % (rtt.size, warm.size, cold.sum()))

    # Cluster by simple gap splitting: anything further than half the smallest
    # inter-cluster distance starts a new group. With clusters this separated a
    # threshold of 30 ms is unambiguous, and the printout below shows the
    # within-cluster spread so the choice can be checked rather than trusted.
    s = np.sort(warm)
    groups, cur = [], [s[0]]
    for x in s[1:]:
        if x - cur[-1] > 30.0:
            groups.append(np.array(cur))
            cur = [x]
        else:
            cur.append(x)
    groups.append(np.array(cur))

    print("\n  Observed clusters (warm invocations only):\n")
    print("    %-6s %-8s %-10s %-10s %s" % ("n", "mean", "min", "max", "width"))
    centres = []
    for g in groups:
        centres.append(g.mean())
        print("    %-6d %-8.1f %-10.1f %-10.1f %.1f ms"
              % (g.size, g.mean(), g.min(), g.max(), g.max() - g.min()))

    centres = np.array(centres)
    steps = np.diff(centres)
    print("\n  Spacing between centres : %s ms"
          % np.array2string(np.round(steps, 1), separator=", "))
    if steps.size:
        print("  Mean step               : %.1f ms  (sd %.1f ms)"
              % (steps.mean(), steps.std()))
        widths = np.array([g.max() - g.min() for g in groups])
        print("  Mean cluster width      : %.1f ms" % widths.mean())
        print()
        if steps.std() < 0.25 * steps.mean() and widths.mean() < 0.5 * steps.mean():
            print("  -> Clusters are narrow and evenly spaced. The round trip does not")
            print("     vary continuously; it takes a whole number of steps of about")
            print("     %.0f ms." % steps.mean())
            print("     A step that size is on the scale of one or two network round")
            print("     trips, not of load variation in Lambda -- compute there varies")
            print("     by only %.0f ms between the fastest and slowest sample."
                  % (handler[~cold].max() - handler[~cold].min()))
        else:
            print("  -> Clusters are not regular enough to call the delay quantised.")

    # Does the cloud explain the steps? If handler_ms were the driver, it would
    # correlate with which group a sample lands in. Check that directly.
    hw = handler[~cold]
    lab = np.zeros(warm.size, dtype=int)
    for i, x in enumerate(np.sort(warm)):
        lab[i] = int(np.argmin(np.abs(centres - x)))
    order = np.argsort(warm)
    hw_sorted = hw[order]
    print("\n  Does Lambda explain the step? Mean handler_ms per cluster:")
    for k in range(len(centres)):
        sel = lab == k
        if sel.sum():
            print("    cluster %d (%.0f ms): handler %.1f ms  (n=%d)"
                  % (k, centres[k], hw_sorted[sel].mean(), sel.sum()))
    print("    -> if these are roughly equal, the step comes from the network,")
    print("       not from compute time.")

    # Baseline for comparison. Measured with ICMP from the PC on the same link
    # to the IoT endpoint (10 packets: min 53, avg 58, max 62 ms). It is NOT the
    # Pico's own path -- the Pico adds a WiFi hop -- so it is used only to
    # establish the order of magnitude of one network round trip, not as an
    # exact term in a budget.
    net_rtt_ms = 58.0
    if steps.size:
        print("\n  Comparison with one network round trip (%.0f ms, ICMP from the PC):"
              % net_rtt_ms)
        print("    step / network RTT      : %.2f" % (steps.mean() / net_rtt_ms))
        floor_other = warm.min() - hw.mean()
        print("    floor excluding Lambda  : %.0f ms  =  %.1f x network RTT"
              % (floor_other, floor_other / net_rtt_ms))
        print("    -> the step is one to two round trips, while the floor needs")
        print("       several: the 6.6 kB publish is split into many TCP segments,")
        print("       and the response passes through the broker again.")

    # ------------------------------------------------------------------ plot
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("E4b — round-trip time measured on the device",
                 fontsize=12, fontweight="bold")

    idx = np.arange(rtt.size)
    ax1.plot(idx[~cold], rtt[~cold], "o", ms=5, color="#2b6cb0", label="warm")
    if cold.any():
        ax1.plot(idx[cold], rtt[cold], "o", ms=7, color="#c53030", label="cold start")
    for c in centres:
        ax1.axhline(c, color="#718096", lw=0.8, ls=":", zorder=0)
    ax1.set_xlabel("window #")
    ax1.set_ylabel("RTT [ms]")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.hist(warm, bins=np.arange(warm.min() - 15, warm.max() + 30, 10),
             color="#2b6cb0", alpha=0.85)
    ax2.set_xlabel("RTT [ms], warm only")
    ax2.set_ylabel("count")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("step ≈ %.0f ms" % (steps.mean() if steps.size else 0),
                  fontsize=9.5)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print("\n  wrote %s" % FIG.relative_to(DOCS.parent).as_posix())


if __name__ == "__main__":
    main()

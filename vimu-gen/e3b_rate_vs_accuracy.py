"""E3b -- The on-device Pareto frontier: ESKF rate versus accuracy and CPU.

E3 showed a full-rate ESKF cannot run on the Pico W: a textbook implementation
at 200 Hz needs 413% of the sample period. That alone does not justify
offloading, because the obvious reply is "then run it slower". Decimating the
filter genuinely does make it fit -- so the real comparison is not *cloud
versus impossible*, it is *cloud versus a degraded on-device filter*.

This experiment measures that degradation, and pairs it with the CPU cost
measured on the actual hardware in E3. The output is the frontier of what the
device can achieve unaided, which is the baseline AWS Lambda has to beat.

Decimation model
----------------
The realistic two-rate architecture: Tier 0 (strapdown) keeps running at the
full 200 Hz IMU rate, while the ESKF covariance propagation and measurement
updates run at a lower rate on block-averaged IMU samples. Averaging over a
block is what a real implementation would do; it also loses coning and sculling
content, which is part of the accuracy cost being measured and not an artifact.

Detector scaling
----------------
The stationary detector's statistic scales as 1/dt, because residuals scale
with the per-sample noise N/sqrt(dt) while the normalisation uses the density
N. Its threshold is therefore scaled with the rate (400 * fs, which reproduces
the 8e4 default at 200 Hz exactly) and its window is held at a constant 0.2 s
of wall time rather than a constant sample count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import eskf
import noise_model as nm
import rotations as rot
import trajectory as tj

DEG, MILLI_G = nm.DEG, nm.MILLI_G

# ---------------------------------------------------------------------------
# Measured on the hardware -- see docs/e3_results.txt. Cycles on RP2040 @125MHz.
# ---------------------------------------------------------------------------
CLK_HZ = 125_000_000
CYC = {
    "predict_sparse": 155_942,
    "predict_dense": 1_058_777,
    "update_short": 282_507,
    "update_joseph": 1_521_245,
    "strapdown": 10_940,
}
IMU_HZ = 200.0
# Fraction of the mission spent stationary, so how often an update actually
# fires. Measured from the mission profile rather than assumed.
STATIONARY_DUTY = 0.25

CONST_BG = np.array([0.05, -0.03, 0.02]) * DEG
CONST_BA = np.array([2.0, -1.0, 3.0]) * MILLI_G


def cpu_fraction(eskf_hz: float, optimised: bool = True) -> float:
    """Fraction of one core consumed, from the E3 hardware measurements.

    Tier 0 runs at the full IMU rate regardless; only the ESKF is decimated.
    """
    pred = CYC["predict_sparse"] if optimised else CYC["predict_dense"]
    upd = CYC["update_short"] if optimised else CYC["update_joseph"]
    per_sec = (
        IMU_HZ * CYC["strapdown"]
        + eskf_hz * (pred + STATIONARY_DUTY * upd)
    )
    return per_sec / CLK_HZ


def decimate(x: np.ndarray, d: int) -> np.ndarray:
    """Block-average, aligned so that block k covers samples [(k-1)d, kd).

    That alignment matters: run_eskf uses sample k to propagate from k-1 to k,
    so the averaged block must describe the interval *ending* at k.
    """
    if d == 1:
        return x.copy()
    n_blocks = x.shape[0] // d
    out = np.empty((n_blocks, x.shape[1]))
    out[0] = x[0]
    for k in range(1, n_blocks):
        out[k] = x[(k - 1) * d : k * d].mean(axis=0)
    return out


def make_full_rate(traj, seed: int):
    rng = np.random.default_rng(seed)
    gyro, rg = nm.synthesise_triad(
        traj.omega_body, nm.gyro_spec(), traj.dt, rng, nm.GYRO_RANGES,
        apply_deterministic=False,
    )
    accel, ra = nm.synthesise_triad(
        traj.accel_body, nm.accel_spec(), traj.dt, rng, nm.ACCEL_RANGES,
        apply_deterministic=False,
    )
    return (gyro + CONST_BG, accel + CONST_BA,
            rg.bias_series + CONST_BG, ra.bias_series + CONST_BA)


def run_at_rate(traj, gyro, accel, tbg, tba, d: int) -> dict:
    """Run the ESKF on IMU data decimated by factor d."""
    fs = IMU_HZ / d
    dt = d / IMU_HZ

    g_dec = decimate(gyro, d)
    a_dec = decimate(accel, d)
    n = g_dec.shape[0]
    idx = np.arange(n) * d

    cfg = eskf.ESKFConfig(
        detector_window=max(4, int(round(0.2 * fs))),
        detector_threshold=400.0 * fs,
    )
    res = eskf.run_eskf(
        g_dec, a_dec, dt,
        q0=traj.quat[0], v0=traj.velocity[0], p0=traj.position[0], cfg=cfg,
    )

    half = n // 2
    pos_err = np.linalg.norm(res.position - traj.position[idx], axis=1)
    att_err = np.rad2deg(rot.angular_error(res.quat, traj.quat[idx]))
    dbg = res.bias_gyro[half:] - tbg[idx][half:]

    return {
        "fs": fs,
        "final_pos": float(pos_err[-1]),
        "rms_pos": float(np.sqrt(np.mean(pos_err**2))),
        "final_att": float(att_err[-1]),
        "rms_bg": float(np.sqrt(np.mean(dbg**2)) / DEG),
        "n_updates": sum(res.n_updates.values()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--budget", type=float, default=0.50,
                   help="CPU fraction available to the ESKF+Tier0 on-device")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "e3b_frontier.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print("E3b - on-device Pareto frontier (ESKF rate vs accuracy vs CPU)")
    print(f"CPU costs from E3 hardware measurement, RP2040 @ {CLK_HZ/1e6:.0f} MHz")
    print(f"duration {args.duration:g} s, {args.runs} runs per rate\n")

    traj = tj.MissionTrajectory().generate(args.duration, IMU_HZ)

    # Generate the full-rate streams once per seed, then decimate.
    streams = [make_full_rate(traj, args.seed + i) for i in range(args.runs)]

    decims = [1, 2, 4, 8, 10, 20, 40, 100]
    rows = []

    print(f"  {'ESKF Hz':>8}{'CPU opt':>10}{'CPU naive':>11}"
          f"{'pos err':>11}{'RMS pos':>11}{'att err':>11}{'gyro bias':>12}")
    print(f"  {'':>8}{'%':>10}{'%':>11}{'m':>11}{'m':>11}{'deg':>11}{'deg/s':>12}")
    print(f"  {'-' * 74}")

    for d in decims:
        per_seed = [run_at_rate(traj, *s, d) for s in streams]
        fs = per_seed[0]["fs"]
        row = {
            "fs": fs,
            "cpu_opt": cpu_fraction(fs, True),
            "cpu_naive": cpu_fraction(fs, False),
            "final_pos": float(np.median([r["final_pos"] for r in per_seed])),
            "rms_pos": float(np.median([r["rms_pos"] for r in per_seed])),
            "final_att": float(np.median([r["final_att"] for r in per_seed])),
            "rms_bg": float(np.median([r["rms_bg"] for r in per_seed])),
        }
        rows.append(row)
        flag = "" if row["cpu_opt"] <= args.budget else "  <-- over budget"
        print(f"  {fs:>8.0f}{row['cpu_opt'] * 100:>10.1f}{row['cpu_naive'] * 100:>11.1f}"
              f"{row['final_pos']:>11.2f}{row['rms_pos']:>11.2f}"
              f"{row['final_att']:>11.3f}{row['rms_bg']:>12.4f}{flag}")

    # Best rate that fits the on-device CPU budget.
    feasible = [r for r in rows if r["cpu_opt"] <= args.budget]
    best = min(feasible, key=lambda r: r["final_pos"]) if feasible else None
    full = rows[0]

    print(f"\n{'=' * 76}")
    print("INTERPRETATION")
    print(f"{'=' * 76}")
    if best is None:
        print(f"  No ESKF rate fits a {args.budget:.0%} CPU budget on this device.")
    else:
        print(f"  On-device budget {args.budget:.0%} of one core (leaving the rest for the")
        print(f"  sensor driver, lwIP/mbedTLS and scheduling slack).")
        print(f"    best feasible on-device rate : {best['fs']:.0f} Hz "
              f"({best['cpu_opt'] * 100:.1f}% CPU)")
        print(f"    its position error at {args.duration:.0f} s   : {best['final_pos']:.2f} m")
        print(f"    full-rate {full['fs']:.0f} Hz reference     : {full['final_pos']:.2f} m "
              f"but {full['cpu_opt'] * 100:.1f}% CPU (infeasible)")
        ratio = best["final_pos"] / max(full["final_pos"], 1e-9)
        print(f"\n  Accuracy cost of staying on-device: {ratio:.2f}x worse position error.")
        print("  That ratio is what offloading to AWS Lambda must beat, and it is the")
        print("  quantity E4-E7 trade against round-trip latency.")
    print(f"\n  Note: 'CPU naive' is the textbook implementation. At 200 Hz it is")
    print(f"  {rows[0]['cpu_naive'] * 100:.0f}% of a core -- the offload case is far stronger")
    print("  before hand-optimisation than after, so the thesis should quote the")
    print("  optimised figure as the honest one.")

    if not args.no_plot:
        plot(rows, args, best, full)

    return 0


def plot(rows, args, best, full):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = np.array([r["fs"] for r in rows])
    cpu_o = np.array([r["cpu_opt"] for r in rows]) * 100
    cpu_n = np.array([r["cpu_naive"] for r in rows]) * 100
    pos = np.array([r["final_pos"] for r in rows])
    bg = np.array([r["rms_bg"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    ax.loglog(fs, cpu_o, "o-", color="tab:green", lw=1.8, ms=5, label="optimised")
    ax.loglog(fs, cpu_n, "s--", color="tab:red", lw=1.6, ms=5, label="textbook")
    ax.axhline(args.budget * 100, color="0.35", ls=":", lw=1.5)
    ax.axhline(100, color="k", ls="-", lw=1.2)
    ax.text(fs.min() * 1.1, args.budget * 100 * 1.1, f"budget {args.budget:.0%}",
            fontsize=8, color="0.3")
    ax.text(fs.min() * 1.1, 108, "one full core", fontsize=8)
    ax.set_xlabel("ESKF rate [Hz]")
    ax.set_ylabel("CPU load [% of one M0+ core]")
    ax.set_title("A. Measured CPU cost vs filter rate\n(RP2040 @125 MHz, from E3)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(fs, pos, "o-", color="tab:blue", lw=1.8, ms=5)
    ax.set_xlabel("ESKF rate [Hz]")
    ax.set_ylabel(f"position error at {args.duration:.0f} s [m]")
    ax.set_title("B. Accuracy cost of running slower", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    feas = cpu_o <= args.budget * 100
    ax.loglog(cpu_o[feas], pos[feas], "o-", color="tab:green", lw=1.8, ms=6,
              label="fits on the device")
    ax.loglog(cpu_o[~feas], pos[~feas], "o--", color="tab:red", lw=1.6, ms=6,
              alpha=0.6, label="exceeds the budget")
    ax.axvline(args.budget * 100, color="0.35", ls=":", lw=1.5)
    if best is not None:
        ax.plot(best["cpu_opt"] * 100, best["final_pos"], "*", color="k", ms=16,
                zorder=5, label="best achievable on the device")
    ax.plot(full["cpu_opt"] * 100, full["final_pos"], "P", color="tab:orange",
            ms=12, zorder=5, label="full rate (needs Lambda)")
    for r in rows:
        ax.annotate(f"{r['fs']:.0f} Hz", (r["cpu_opt"] * 100, r["final_pos"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xlabel("CPU load [% of one M0+ core]")
    ax.set_ylabel(f"position error at {args.duration:.0f} s [m]")
    ax.set_title("C. The frontier AWS Lambda has to beat", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("E3b — On-device Pareto frontier: what the Pico W can do unaided",
                 fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())

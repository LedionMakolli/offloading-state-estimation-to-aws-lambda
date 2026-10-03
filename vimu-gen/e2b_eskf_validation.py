"""E2b -- Validation of the offloaded ESKF estimator.

Prerequisite for E4-E7: this is the payload that will run in Lambda, so it has
to be shown correct on the host before any of the systems work depends on it.

Four questions, each a panel of the figure:

A. Does the filter recover the bias that was actually injected, and is its
   reported uncertainty honest? Scored against the true instantaneous bias
   (turn-on + flicker + rate random walk), not just the constant part --
   scoring against the constant alone would make a correct filter look broken,
   because flicker dominates at long horizons.

B. How much does it help? Position and attitude error against the unaided
   dead-reckoning baseline from E2.

C. Does observability actually require attitude diversity at rest? This is the
   experiment behind the design choice in MissionTrajectory: resting at one
   attitude leaves horizontal accelerometer bias unobservable, because a tilt
   about x and a bias on y produce an identical gravity signature.

D. Which pseudo-measurement is doing the work? Ablation over ZUPT / ZARU /
   gravity, which tells the firmware author what may be dropped under deadline
   pressure and what may not.

A note on what is NOT claimed. The ESKF does not make position observable --
nothing in an IMU-only architecture can. Error is reset at each stationary
phase, so it is bounded by the *coast interval between stationary phases*
rather than by total mission time. Fitting a drift exponent to the ESKF error
is meaningless: the signal is a sawtooth, not a power law.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import eskf
import noise_model as nm
import rotations as rot
import strapdown as sd
import trajectory as tj

DEG, MILLI_G = nm.DEG, nm.MILLI_G

# A constant offset on top of the model's own flicker, representing the
# residual left by an imperfect startup calibration.
CONST_BG = np.array([0.05, -0.03, 0.02]) * DEG
CONST_BA = np.array([2.0, -1.0, 3.0]) * MILLI_G


def measurements(traj, seed: int):
    """Corrupted IMU stream plus the true instantaneous bias of each sensor."""
    rng = np.random.default_rng(seed)
    gyro, rg = nm.synthesise_triad(
        traj.omega_body, nm.gyro_spec(), traj.dt, rng, nm.GYRO_RANGES,
        apply_deterministic=False,
    )
    accel, ra = nm.synthesise_triad(
        traj.accel_body, nm.accel_spec(), traj.dt, rng, nm.ACCEL_RANGES,
        apply_deterministic=False,
    )
    return (
        gyro + CONST_BG,
        accel + CONST_BA,
        rg.bias_series + CONST_BG,
        ra.bias_series + CONST_BA,
    )


def part_a_accuracy(traj, seeds: list[int]) -> dict:
    print(f"\n{'=' * 74}\nA. BIAS RECOVERY AND FILTER CONSISTENCY\n{'=' * 74}")
    cfg = eskf.ESKFConfig()
    rcfg = cfg.resolved(traj.dt)
    print(f"  derived measurement noise: zaru {rcfg.zaru_sigma / DEG:.4f} deg/s, "
          f"gravity {rcfg.gravity_sigma:.4f} m/s^2")

    truth_mask = None
    rec = {"eg": [], "ea": [], "cg": [], "ca": [], "prec": [], "recall": []}
    keep = None
    half = len(traj) // 2

    t0 = time.perf_counter()
    for i, sd_ in enumerate(seeds):
        gyro, accel, tbg, tba = measurements(traj, sd_)
        res = eskf.run_eskf(gyro, accel, traj.dt, traj.quat[0], traj.velocity[0],
                            traj.position[0], cfg=cfg)
        if truth_mask is None:
            truth_mask = MISSION.stationary_mask(traj.t)
        det = res.stationary
        tp = int((det & truth_mask).sum())
        fp = int((det & ~truth_mask).sum())
        fn = int((~det & truth_mask).sum())
        rec["prec"].append(tp / max(tp + fp, 1))
        rec["recall"].append(tp / max(tp + fn, 1))

        dbg = res.bias_gyro[half:] - tbg[half:]
        dba = res.bias_accel[half:] - tba[half:]
        rec["eg"].append(np.sqrt(np.mean(dbg**2)) / DEG)
        rec["ea"].append(np.sqrt(np.mean(dba**2)) / MILLI_G)
        rec["cg"].append(np.mean(np.abs(dbg) / res.sigma_bias_gyro[half:]))
        rec["ca"].append(np.mean(np.abs(dba) / res.sigma_bias_accel[half:]))
        if i == 0:
            keep = (res, tbg, tba)
    print(f"  {len(seeds)} runs in {time.perf_counter() - t0:.1f} s")

    print(f"\n  {'metric':<34}{'mean':>12}{'std':>12}")
    print(f"  {'-' * 58}")
    rows = [
        ("stationary detector precision", rec["prec"], ""),
        ("stationary detector recall", rec["recall"], ""),
        ("gyro bias RMS error", rec["eg"], "deg/s"),
        ("accel bias RMS error", rec["ea"], "mg"),
        ("gyro consistency |err|/sigma", rec["cg"], ""),
        ("accel consistency |err|/sigma", rec["ca"], ""),
    ]
    for label, vals, unit in rows:
        print(f"  {label + ' ' + unit:<34}{np.mean(vals):>12.4f}{np.std(vals):>12.4f}")

    cons_ok = 0.5 <= np.mean(rec["cg"]) <= 2.0 and 0.5 <= np.mean(rec["ca"]) <= 2.0
    print(f"\n  [{'PASS' if cons_ok else 'FAIL'}] filter covariance is consistent "
          f"(both |err|/sigma within [0.5, 2.0])")
    return {"rec": rec, "example": keep, "passed": cons_ok}


def part_b_improvement(traj, seeds: list[int]) -> dict:
    print(f"\n{'=' * 74}\nB. IMPROVEMENT OVER UNAIDED DEAD RECKONING\n{'=' * 74}")
    cfg = eskf.ESKFConfig()
    dr_curves, kf_curves, dr_att, kf_att = [], [], [], []
    for sd_ in seeds:
        gyro, accel, _, _ = measurements(traj, sd_)
        dr = sd.mechanise(gyro, accel, traj.dt, traj.quat[0], traj.velocity[0],
                          traj.position[0])
        kf = eskf.run_eskf(gyro, accel, traj.dt, traj.quat[0], traj.velocity[0],
                           traj.position[0], cfg=cfg)
        dr_curves.append(np.linalg.norm(dr.position - traj.position, axis=1))
        kf_curves.append(np.linalg.norm(kf.position - traj.position, axis=1))
        dr_att.append(np.rad2deg(rot.angular_error(dr.quat, traj.quat)))
        kf_att.append(np.rad2deg(rot.angular_error(kf.quat, traj.quat)))

    dr_m = np.median(dr_curves, axis=0)
    kf_m = np.median(kf_curves, axis=0)
    dra_m = np.median(dr_att, axis=0)
    kfa_m = np.median(kf_att, axis=0)

    print(f"\n  {'horizon':>10}{'dead-reckon':>16}{'ESKF':>14}{'factor':>10}")
    print(f"  {'-' * 50}")
    fs = 1.0 / traj.dt
    for h in (10, 30, 60):
        # Index, not time, decides validity: the last sample of a 60 s run at
        # 200 Hz sits at 59.995 s, so comparing against t[-1] drops the final row.
        i = int(h * fs) - 1
        if i >= len(traj):
            continue
        print(f"  {h:>8.0f} s{dr_m[i]:>14.2f} m{kf_m[i]:>12.2f} m"
              f"{dr_m[i] / max(kf_m[i], 1e-9):>9.0f}x")
    print(f"\n  attitude error at {traj.t[-1]:.0f} s: "
          f"{dra_m[-1]:.3f} deg -> {kfa_m[-1]:.3f} deg "
          f"({dra_m[-1] / max(kfa_m[-1], 1e-9):.0f}x)")
    print("\n  Note: ESKF error is a sawtooth reset at each stationary phase, so it is")
    print("  bounded by the coast interval, not by mission time. A power-law drift")
    print("  exponent is not a meaningful description of it.")
    return {"dr": dr_m, "kf": kf_m, "dr_att": dra_m, "kf_att": kfa_m, "t": traj.t}


def part_c_observability(seeds: list[int], duration: float, fs: float) -> dict:
    print(f"\n{'=' * 74}\nC. OBSERVABILITY REQUIRES ATTITUDE DIVERSITY AT REST\n{'=' * 74}")
    cfg = eskf.ESKFConfig()
    variants = {
        "single attitude (all level)": [(0.0, 0.0, 0.0)] * 3,
        "three distinct attitudes": [(0.0, 0.0, 0.0), (25.0, -15.0, 40.0),
                                     (-20.0, 30.0, -60.0)],
    }
    out = {}
    print(f"\n  {'rest attitudes':<30}{'accel bias err [mg]':>22}{'gyro bias err [deg/s]':>24}")
    print(f"  {'-' * 76}")
    for label, poses in variants.items():
        traj = tj.MissionTrajectory(rest_attitudes=poses).generate(duration, fs)
        half = len(traj) // 2
        ea_xy, ea_z, eg = [], [], []
        for sd_ in seeds:
            gyro, accel, tbg, tba = measurements(traj, sd_)
            res = eskf.run_eskf(gyro, accel, traj.dt, traj.quat[0], traj.velocity[0],
                                traj.position[0], cfg=cfg)
            dba = (res.bias_accel[half:] - tba[half:]) / MILLI_G
            dbg = (res.bias_gyro[half:] - tbg[half:]) / DEG
            ea_xy.append(np.sqrt(np.mean(dba[:, :2] ** 2)))
            ea_z.append(np.sqrt(np.mean(dba[:, 2] ** 2)))
            eg.append(np.sqrt(np.mean(dbg**2)))
        out[label] = {"xy": np.mean(ea_xy), "z": np.mean(ea_z), "gyro": np.mean(eg)}
        print(f"  {label:<30}{'xy ' + format(np.mean(ea_xy), '.3f'):>12}"
              f"{'  z ' + format(np.mean(ea_z), '.3f'):>10}{np.mean(eg):>24.4f}")

    a = out["single attitude (all level)"]["xy"]
    b = out["three distinct attitudes"]["xy"]
    print(f"\n  horizontal accel bias error improves {a / max(b, 1e-9):.0f}x with attitude diversity")
    print("  the z axis barely changes: tilt affects it only to second order, so it")
    print("  was already observable from a single level pose")
    return out


def part_d_ablation(traj, seeds: list[int]) -> dict:
    print(f"\n{'=' * 74}\nD. ABLATION -- WHICH PSEUDO-MEASUREMENT MATTERS\n{'=' * 74}")
    cfg = eskf.ESKFConfig()
    combos = {
        "all three": (True, True, True),
        "no ZARU": (True, False, True),
        "no gravity": (True, True, False),
        "no ZUPT": (False, True, True),
        "ZUPT only": (True, False, False),
    }
    out = {}
    print(f"\n  {'enabled':<16}{'pos err [m]':>14}{'att err [deg]':>16}"
          f"{'gyro b [deg/s]':>18}")
    print(f"  {'-' * 64}")
    for label, (zu, za, gr) in combos.items():
        pe, ae, gb = [], [], []
        half = len(traj) // 2
        for sd_ in seeds:
            gyro, accel, tbg, _ = measurements(traj, sd_)
            res = eskf.run_eskf(gyro, accel, traj.dt, traj.quat[0], traj.velocity[0],
                                traj.position[0], cfg=cfg, enable_zupt=zu,
                                enable_zaru=za, enable_gravity=gr)
            pe.append(np.linalg.norm(res.position[-1] - traj.position[-1]))
            ae.append(np.rad2deg(rot.angular_error(res.quat[-1:], traj.quat[-1:]))[0])
            gb.append(np.sqrt(np.mean(((res.bias_gyro[half:] - tbg[half:]) / DEG) ** 2)))
        out[label] = {"pos": np.median(pe), "att": np.median(ae), "gyro": np.mean(gb)}
        print(f"  {label:<16}{np.median(pe):>14.2f}{np.median(ae):>16.3f}"
              f"{np.mean(gb):>18.4f}")
    return out


def plot(a: dict, b: dict, c: dict, d: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A -- bias tracking with +/- sigma band
    ax = axes[0, 0]
    res, tbg, tba = a["example"]
    t = res.t
    for axis, colour in zip(range(3), ["tab:red", "tab:green", "tab:blue"]):
        ax.plot(t, tbg[:, axis] / DEG, color=colour, lw=1.6, alpha=0.55)
        ax.plot(t, res.bias_gyro[:, axis] / DEG, color=colour, lw=1.1, ls="--")
        ax.fill_between(
            t,
            (res.bias_gyro[:, axis] - res.sigma_bias_gyro[:, axis]) / DEG,
            (res.bias_gyro[:, axis] + res.sigma_bias_gyro[:, axis]) / DEG,
            color=colour, alpha=0.15, lw=0,
        )
    ax.plot([], [], "k-", lw=1.6, alpha=0.55, label="true bias")
    ax.plot([], [], "k--", lw=1.1, label="estimate")
    ax.fill_between([], [], [], color="k", alpha=0.15, label=r"$\pm 1\sigma$")
    for x0, x1 in MISSION.motion_intervals:
        ax.axvspan(x0, x1, color="0.9", zorder=0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gyro bias [°/s]")
    ax.set_title("A. Gyro bias tracking (shaded = in motion)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # Clip to the tracking detail. The band opens at the 1 deg/s initial
    # uncertainty and collapses within the first stationary phase, which would
    # otherwise set the y-range and flatten everything of interest.
    span = float(np.abs(tbg / DEG).max()) * 2.5
    ax.set_ylim(-span, span)

    # B -- error vs dead reckoning
    ax = axes[0, 1]
    ax.semilogy(b["t"], np.maximum(b["dr"], 1e-4), "k-", lw=1.8, label="unaided dead reckoning")
    ax.semilogy(b["t"], np.maximum(b["kf"], 1e-4), color="tab:green", lw=1.8,
                label="ESKF (offloaded)")
    for x0, x1 in MISSION.motion_intervals:
        ax.axvspan(x0, x1, color="0.9", zorder=0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title(
        f"B. Position error\n{b['dr'][-1]:.0f} m → {b['kf'][-1]:.2f} m at "
        f"{b['t'][-1]:.0f} s ({b['dr'][-1] / max(b['kf'][-1], 1e-9):.0f}×)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # C -- observability
    ax = axes[1, 0]
    labels = list(c.keys())
    xy = [c[k]["xy"] for k in labels]
    z = [c[k]["z"] for k in labels]
    x = np.arange(len(labels))
    b1 = ax.bar(x - 0.2, xy, 0.4, label="horizontal (x, y)", color="tab:red")
    b2 = ax.bar(x + 0.2, z, 0.4, label="vertical (z)", color="tab:blue")
    ax.set_yscale("log")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax.set_ylabel("RMS accel bias error [mg]")
    ax.set_title("C. Horizontal accel bias is unobservable\nfrom a single stationary attitude",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # D -- ablation
    ax = axes[1, 1]
    labels = list(d.keys())
    vals = [d[k]["pos"] for k in labels]
    colours = ["tab:green" if k == "all three" else "tab:orange" for k in labels]
    ax.barh(np.arange(len(labels)), vals, color=colours)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("final position error [m]")
    ax.set_title("D. Pseudo-measurement ablation", fontsize=10)
    ax.grid(True, axis="x", which="both", alpha=0.3)

    fig.suptitle("E2b — Validation of the offloaded ESKF estimator", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {out_path}")


MISSION = tj.MissionTrajectory()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "e2b_eskf.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print("E2b - ESKF estimator validation")
    print(f"duration {args.duration:g} s, fs {args.fs:g} Hz, {args.runs} runs")

    traj = MISSION.generate(args.duration, args.fs)
    seeds = [args.seed + i for i in range(args.runs)]
    few = seeds[: max(3, args.runs // 2)]

    a = part_a_accuracy(traj, seeds)
    b = part_b_improvement(traj, few)
    c = part_c_observability(few, args.duration, args.fs)
    d = part_d_ablation(traj, few)

    if not args.no_plot:
        plot(a, b, c, d, args.out)

    print(f"\n{'=' * 74}")
    print(f"E2b RESULT: {'PASS' if a['passed'] else 'FAIL'}")
    print(f"{'=' * 74}")
    return 0 if a["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

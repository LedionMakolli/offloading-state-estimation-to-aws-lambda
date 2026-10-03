"""E2 -- Unaided dead-reckoning drift of the virtual MPU-6050.

Three parts, in order of dependency:

A. Mechanisation verification.
   Feed the *exact* analytic inertial measurements to the strapdown
   integrator and confirm the error falls as dt^2 when the rate is doubled.
   This proves the navigator is correct before any noise is introduced, so
   that everything measured afterwards is attributable to the sensor model.
   Claiming "machine precision" here would be wrong -- a second-order
   integrator has genuine truncation error. Order of convergence is the
   correct test.

B. Stochastic drift floor.
   Dead-reckon with the calibrated noise model and no bias error at all.
   This is the best case that a perfectly bias-compensated system could
   achieve, and it is still hopeless over any useful duration.

C. Drift versus residual bias.
   Sweep the residual (post-calibration) gyro and accelerometer bias and
   measure the resulting position error. This is the experiment that gives
   the thesis a *requirement* rather than an anecdote: it says how accurately
   the offloaded estimator must determine the biases to hold a given
   position accuracy for a given time, which is precisely what the cloud
   ESKF is being asked to do.

Expected physics, for reference when reading the output:
  * An uncompensated accelerometer bias integrates twice -> error ~ t^2.
  * An uncompensated gyro bias tilts the estimated gravity vector, leaking
    g*sin(theta) into horizontal acceleration -> error ~ t^3.
  The gyro term therefore dominates at long horizons even when it looks
  negligible at t = 0, which is the single most important qualitative fact
  about inertial navigation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import noise_model as nm
import rotations as rot
import strapdown as sd
from trajectory import LissajousTrajectory, peak_measurements

DEG = nm.DEG
MILLI_G = nm.MILLI_G


def make_measurements(
    traj,
    rng: np.random.Generator,
    gyro_bias: np.ndarray | float = 0.0,
    accel_bias: np.ndarray | float = 0.0,
    apply_quantisation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt the exact measurements with the noise model plus a fixed bias.

    The bias is applied explicitly rather than via the datasheet turn-on
    tolerance (+/-20 deg/s for the gyro) because any real system estimates
    and removes that during startup calibration. What matters operationally
    is the *residual* left after calibration, which is what this models.
    """
    dt = traj.dt
    gyro, _ = nm.synthesise_triad(
        traj.omega_body, nm.gyro_spec(), dt, rng, nm.GYRO_RANGES,
        apply_deterministic=False, apply_quantisation=apply_quantisation,
    )
    accel, _ = nm.synthesise_triad(
        traj.accel_body, nm.accel_spec(), dt, rng, nm.ACCEL_RANGES,
        apply_deterministic=False, apply_quantisation=apply_quantisation,
    )
    return gyro + gyro_bias, accel + accel_bias


def dead_reckon(traj, gyro: np.ndarray, accel: np.ndarray) -> sd.NavSolution:
    """Run the navigator from the true initial state."""
    return sd.mechanise(
        gyro, accel, traj.dt,
        q0=traj.quat[0], v0=traj.velocity[0], p0=traj.position[0],
    )


# --------------------------------------------------------------------------
# Part A -- mechanisation verification
# --------------------------------------------------------------------------
def part_a_convergence(duration: float, rates: list[float]) -> dict:
    print(f"\n{'=' * 74}\nA. MECHANISATION VERIFICATION (noise-free analytic input)\n{'=' * 74}")
    print("  Halving dt should quarter the error for a second-order scheme.\n")
    print(f"  {'fs [Hz]':>10}{'dt [ms]':>12}{'final pos err [m]':>22}{'ratio':>10}{'order':>9}")
    print(f"  {'-' * 63}")

    traj_gen = LissajousTrajectory()
    errors, dts = [], []
    prev = None
    for fs in rates:
        traj = traj_gen.generate(duration, fs)
        nav = dead_reckon(traj, traj.omega_body, traj.accel_body)
        err = float(np.linalg.norm(nav.position[-1] - traj.position[-1]))
        dt = 1.0 / fs
        errors.append(err)
        dts.append(dt)
        if prev is None:
            print(f"  {fs:>10.0f}{dt * 1e3:>12.3f}{err:>22.3e}{'-':>10}{'-':>9}")
        else:
            ratio = prev / err if err > 0 else np.inf
            order = np.log2(ratio) if np.isfinite(ratio) and ratio > 0 else np.nan
            print(f"  {fs:>10.0f}{dt * 1e3:>12.3f}{err:>22.3e}{ratio:>10.2f}{order:>9.2f}")
        prev = err

    dts_a = np.array(dts)
    errs_a = np.array(errors)
    valid = errs_a > 0
    order_fit = np.polyfit(np.log(dts_a[valid]), np.log(errs_a[valid]), 1)[0]
    print(f"\n  fitted order of convergence: {order_fit:.3f}  (expected ~2)")
    ok = 1.7 <= order_fit <= 2.3
    print(f"  [{'PASS' if ok else 'FAIL'}] mechanisation is second-order accurate")
    return {"dts": dts_a, "errors": errs_a, "order": order_fit, "passed": ok}


# --------------------------------------------------------------------------
# Part B -- stochastic drift floor
# --------------------------------------------------------------------------
def part_b_noise_floor(duration: float, fs: float, n_runs: int, seed: int) -> dict:
    print(f"\n{'=' * 74}\nB. STOCHASTIC DRIFT FLOOR (perfect bias compensation)\n{'=' * 74}")
    traj = LissajousTrajectory().generate(duration, fs)
    peaks = peak_measurements(traj)
    print(f"  trajectory: {duration:g} s at {fs:g} Hz, path length "
          f"{peaks['path_length_m']:.1f} m, max speed {peaks['max_speed_mps']:.2f} m/s")
    print(f"  peak |gyro| {peaks['max_gyro_dps']:.1f} deg/s (FS +/-250), "
          f"peak |accel| {peaks['max_accel_g']:.2f} g (FS +/-2)")
    if peaks["max_gyro_dps"] > 250 or peaks["max_accel_g"] > 2.0:
        print("  WARNING: trajectory exceeds the selected full-scale range; "
              "measurements will clip.")

    curves = np.empty((n_runs, len(traj)))
    t0 = time.perf_counter()
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)
        gyro, accel = make_measurements(traj, rng)
        nav = dead_reckon(traj, gyro, accel)
        curves[i] = sd.position_error(nav, traj.position)
    print(f"  {n_runs} Monte Carlo runs in {time.perf_counter() - t0:.1f} s")

    median = np.median(curves, axis=0)
    exponent, coeff = sd.fit_drift_exponent(traj.t, median, t_min=2.0)

    print(f"\n  {'horizon':>10}{'median err':>16}{'p10':>14}{'p90':>14}")
    print(f"  {'-' * 54}")
    for horizon in [1.0, 5.0, 10.0, 30.0, 60.0]:
        if horizon > duration:
            continue
        idx = int(horizon * fs) - 1
        print(f"  {horizon:>8.0f} s{median[idx]:>14.3f} m"
              f"{np.percentile(curves[:, idx], 10):>12.3f} m"
              f"{np.percentile(curves[:, idx], 90):>12.3f} m")
    print(f"\n  fitted drift law: error ~ {coeff:.3e} * t^{exponent:.2f}")
    return {"t": traj.t, "curves": curves, "median": median,
            "exponent": exponent, "traj": traj}


# --------------------------------------------------------------------------
# Part C -- drift versus residual bias
# --------------------------------------------------------------------------
def part_c_bias_sweep(
    duration: float, fs: float, n_runs: int, seed: int
) -> dict:
    print(f"\n{'=' * 74}\nC. DRIFT VERSUS RESIDUAL BIAS (the requirement curve)\n{'=' * 74}")
    traj = LissajousTrajectory().generate(duration, fs)

    # Residual gyro bias levels, in deg/s. The MPU-6050 bias instability of
    # ~0.01 deg/s is the floor a static calibration cannot beat no matter how
    # long it averages, so that is the left-hand end of the sweep.
    gyro_levels_dps = np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.5])
    accel_levels_mg = np.array([0.0, 0.04, 0.13, 0.5, 2.0, 10.0])

    results = {}
    for label, levels, unit in [
        ("gyro", gyro_levels_dps, "deg/s"),
        ("accel", accel_levels_mg, "mg"),
    ]:
        print(f"\n  residual {label} bias sweep")
        print(f"  {'bias':>14}{'err @10s':>14}{'err @30s':>14}{'err @60s':>14}{'exponent':>11}")
        print(f"  {'-' * 67}")
        finals, exps, curves_for_level = [], [], []
        for level in levels:
            gb = np.array([level * DEG, 0.0, 0.0]) if label == "gyro" else 0.0
            ab = np.array([level * MILLI_G, 0.0, 0.0]) if label == "accel" else 0.0
            runs = np.empty((n_runs, len(traj)))
            for i in range(n_runs):
                rng = np.random.default_rng(seed + 1000 + i)
                gyro, accel = make_measurements(traj, rng, gyro_bias=gb, accel_bias=ab)
                nav = dead_reckon(traj, gyro, accel)
                runs[i] = sd.position_error(nav, traj.position)
            med = np.median(runs, axis=0)
            curves_for_level.append(med)
            try:
                exp_fit, _ = sd.fit_drift_exponent(traj.t, med, t_min=2.0)
            except ValueError:
                exp_fit = np.nan
            exps.append(exp_fit)
            finals.append(med[-1])

            def at(sec: float) -> str:
                if sec > duration:
                    return f"{'-':>12}"
                return f"{med[int(sec * fs) - 1]:>12.2f}m"

            print(f"  {level:>10.3f} {unit:<3}{at(10)}{at(30)}{at(60)}{exp_fit:>11.2f}")

        results[label] = {
            "levels": levels,
            "finals": np.array(finals),
            "exponents": np.array(exps),
            "curves": np.array(curves_for_level),
            "unit": unit,
        }

    print("\n  Reading this table: the exponent tends toward 3 as gyro bias grows")
    print("  (tilt-induced gravity leakage) and toward 2 as accelerometer bias")
    print("  grows (direct double integration), matching the expected physics.")
    return {"t": traj.t, **results}


def plot(a: dict, b: dict, c: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # A -- convergence
    ax = axes[0]
    ax.loglog(a["dts"] * 1e3, a["errors"], "ko-", lw=1.6, ms=5, label="measured")
    ref = a["errors"][-1] * (a["dts"] / a["dts"][-1]) ** 2
    ax.loglog(a["dts"] * 1e3, ref, "--", color="tab:blue", lw=1.4,
              label=r"reference $\propto dt^2$")
    ax.set_xlabel("time step $dt$  [ms]")
    ax.set_ylabel("final position error  [m]")
    ax.set_title(f"A. Mechanisation convergence\nfitted order = {a['order']:.2f}",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # B -- noise floor
    ax = axes[1]
    t = b["t"]
    for row in b["curves"]:
        ax.plot(t, row, color="0.75", lw=0.7, zorder=1)
    ax.plot(t, b["median"], "k-", lw=2.0, zorder=3, label="median over runs")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("position error  [m]")
    ax.set_title(
        "B. Drift with perfect bias compensation\n"
        f"still $\\sim t^{{{b['exponent']:.1f}}}$; "
        f"{b['median'][-1]:.1f} m after {t[-1]:.0f} s",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # C -- requirement curve
    ax = axes[2]
    NAME = {"gyro": "gyro", "accel": "accel"}
    UNIT = {"deg/s": "°/s"}
    for label, colour, marker in [("gyro", "tab:red", "o"), ("accel", "tab:blue", "s")]:
        d = c[label]
        lv = d["levels"]
        nz = lv > 0
        ax.loglog(lv[nz], d["finals"][nz], marker=marker, color=colour, lw=1.6, ms=5,
                  label=f"residual {NAME[label]} bias [{UNIT.get(d['unit'], d['unit'])}]")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.4)
    ax.text(ax.get_xlim()[0] * 1.2, 1.1, "1 m", fontsize=8, color="0.3")
    ax.set_xlabel("residual bias  [°/s for gyro, mg for accel]")
    ax.set_ylabel(f"position error at t = {c['t'][-1]:.0f} s  [m]")
    ax.set_title("C. Accuracy requirement on the\noffloaded bias estimate", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "E2 — Unaided inertial-navigation drift of the virtual MPU-6050", fontsize=13
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--runs", type=int, default=15, help="Monte Carlo runs per condition")
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "e2_dead_reckoning.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print("E2 - Unaided dead-reckoning drift")
    print(f"duration {args.duration:g} s, fs {args.fs:g} Hz, "
          f"{args.runs} runs/condition, seed {args.seed}")

    a = part_a_convergence(30.0, [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
    b = part_b_noise_floor(args.duration, args.fs, args.runs, args.seed)
    c = part_c_bias_sweep(args.duration, args.fs, max(3, args.runs // 3), args.seed)

    if not args.no_plot:
        plot(a, b, c, args.out)

    print(f"\n{'=' * 74}")
    print(f"E2 RESULT: mechanisation {'PASS' if a['passed'] else 'FAIL'}")
    print(f"{'=' * 74}")
    return 0 if a["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

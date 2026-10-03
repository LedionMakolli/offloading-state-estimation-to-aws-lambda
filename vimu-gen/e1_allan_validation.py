"""E1 -- Allan-variance validation of the virtual MPU-6050 noise model.

This is the gate for the whole project. It answers the single most likely
objection to a thesis built on a simulated sensor: *how do you know your
virtual IMU behaves like the real part?*

The argument is not "trust the code". It is:

  1. The noise model is specified from published MPU-6050 parameters
     (see noise_model.py, which cites the datasheet section for each).
  2. A long stationary record is synthesised from that specification.
  3. The Allan variance of that record is computed blind, with no knowledge
     of the generating parameters.
  4. The five-term Allan model is fitted to the result, recovering N and B.
  5. Recovered values are compared against the specification they came from.

If step 5 fails, the noise model is wrong and no downstream result means
anything -- which is exactly why this runs before any firmware work.

Note the deliberate scope of the claim: this validates that the *model*
reproduces the documented stochastic signature of an MPU-6050. It is not a
claim to have characterised a physical device. State that limitation
explicitly in the thesis.

Usage:
    python e1_allan_validation.py                  # 2 h at 200 Hz
    python e1_allan_validation.py --quick          # 10 min, for fast iteration
    python e1_allan_validation.py --duration 21600 # 6 h, tighter B estimate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import allan
import noise_model as nm

# Tolerances on the recovered parameters.
#
# N is a direct consequence of the white-noise generator and should come back
# very close. B is read off the Allan floor, whose position is itself a noisy
# statistic with few independent clusters at large tau, so it earns a looser
# bound. These are acceptance thresholds for the model, not physical claims.
N_TOLERANCE = 0.05   # 5 %
B_TOLERANCE = 0.25   # 25 %
SLOPE_TOLERANCE = 0.05


def _fmt(value: float, kind: str) -> str:
    """Render an SI value in the units an IMU datasheet would use."""
    if kind == "gyro":
        return f"{value / nm.DEG:.6g} deg/s"
    return f"{value / nm.MILLI_G:.6g} mg"


def _fmt_density(value: float, kind: str) -> str:
    if kind == "gyro":
        return f"{value / nm.DEG:.6g} deg/s/sqrt(Hz)"
    return f"{value / nm.MICRO_G:.6g} ug/sqrt(Hz)"


def validate_sensor(
    name: str,
    kind: str,
    spec: nm.TriadSpec,
    range_table: dict[int, tuple[float, float]],
    duration_s: float,
    fs: float,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Synthesise a stationary record and recover its noise parameters."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    n_samples = int(round(duration_s * fs))
    print(
        f"  synthesising {duration_s / 3600:.2f} h at {fs:g} Hz "
        f"({n_samples:,} samples x 3 axes)..."
    )
    t0 = time.perf_counter()
    data = nm.synthesise_static(
        duration_s, fs, spec, range_table, rng,
        apply_deterministic=False,   # isolate the stochastic terms
        apply_quantisation=False,
    )
    print(f"  done in {time.perf_counter() - t0:.1f} s")

    dt = 1.0 / fs
    taus, adev, dof = allan.overlapping_allan_deviation(data, dt)

    # Average the Allan deviation across the three axes. They are independent
    # realisations of the same specification, so this is a variance reduction,
    # not a fudge.
    adev_mean = np.sqrt(np.mean(adev**2, axis=1))

    fit = allan.fit_allan_model(taus, adev_mean)
    b_floor, tau_floor = allan.estimate_bias_instability(taus, adev_mean)
    # Diagnose the slope over the lowest decade of tau available. Further out,
    # the flicker floor contributes enough to flatten the measured slope --
    # for the gyroscope the white/flicker crossover is only ~0.6 s, so a
    # window anywhere near 1 s reports a slope well short of -1/2.
    slope = allan.fit_slope(taus, adev_mean, tau_lo=taus[0], tau_hi=taus[0] * 10.0)

    results = {
        "spec_N": spec.noise_density,
        "spec_B": spec.bias_instability,
        "fit_N": fit["N"],
        "fit_B": fit["B"],
        "floor_B": b_floor,
        "tau_floor": tau_floor,
        "slope": slope,
        "fit": fit,
    }

    n_err = abs(fit["N"] - spec.noise_density) / spec.noise_density
    b_err = abs(fit["B"] - spec.bias_instability) / spec.bias_instability
    slope_err = abs(slope - (-0.5))

    print(f"\n  {'quantity':<26}{'specified':>22}{'recovered':>22}{'err':>9}")
    print(f"  {'-' * 77}")
    print(
        f"  {'N  noise density':<26}"
        f"{_fmt_density(spec.noise_density, kind):>22}"
        f"{_fmt_density(fit['N'], kind):>22}"
        f"{n_err * 100:>8.2f}%"
    )
    print(
        f"  {'B  bias instability':<26}"
        f"{_fmt(spec.bias_instability, kind):>22}"
        f"{_fmt(fit['B'], kind):>22}"
        f"{b_err * 100:>8.2f}%"
    )
    print(
        f"  {'B  from Allan floor':<26}"
        f"{'':>22}{_fmt(b_floor, kind):>22}"
        f"{abs(b_floor - spec.bias_instability) / spec.bias_instability * 100:>8.2f}%"
    )
    print(f"  {'white-region slope':<26}{-0.5:>22.4f}{slope:>22.4f}{slope_err:>9.4f}")
    print(f"\n  Allan floor located at tau = {tau_floor:.3g} s")
    print(
        f"  other fitted terms: Q={fit['Q']:.3e}  K={fit['K']:.3e}  R={fit['R']:.3e}"
    )

    checks = [
        (f"N within {N_TOLERANCE:.0%}", n_err <= N_TOLERANCE),
        (f"B within {B_TOLERANCE:.0%}", b_err <= B_TOLERANCE),
        (f"slope within +/-{SLOPE_TOLERANCE}", slope_err <= SLOPE_TOLERANCE),
    ]
    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    results["passed"] = all(ok for _, ok in checks)
    return results, taus, adev_mean


def plot(results: dict, curves: dict, out_path: Path) -> None:
    """Log-log Allan deviation plot with the fitted decomposition overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (name, kind) in zip(axes, [("gyroscope", "gyro"), ("accelerometer", "accel")]):
        taus, adev = curves[kind]
        res = results[kind]
        scale = nm.DEG if kind == "gyro" else nm.MILLI_G
        unit = "°/s" if kind == "gyro" else "mg"

        ax.loglog(taus, adev / scale, "k-", lw=1.8, label="simulated $\\sigma(\\tau)$", zorder=5)

        fit = res["fit"]
        # Individual fitted terms, so the decomposition is visible rather than asserted.
        term_white = fit["N"] / np.sqrt(taus)
        term_flicker = np.full_like(taus, allan.BIAS_INSTABILITY_COEFF * fit["B"])
        ax.loglog(taus, term_white / scale, "--", color="tab:blue", lw=1.3,
                  label="white noise $N$, slope $-1/2$")
        ax.loglog(taus, term_flicker / scale, "--", color="tab:red", lw=1.3,
                  label="bias instability $B$")
        if fit["K"] > 0:
            ax.loglog(taus, (fit["K"] * np.sqrt(taus / 3.0)) / scale, "--",
                      color="tab:green", lw=1.3, label="rate random walk $K$")

        ax.axvline(res["tau_floor"], color="0.7", ls=":", lw=1.2, zorder=1)
        ax.plot(res["tau_floor"], np.min(adev) / scale, "o", color="tab:red",
                ms=7, zorder=6)

        spec_n = res["spec_N"] / scale
        spec_b = res["spec_B"] / scale
        fit_n = res["fit_N"] / scale
        fit_b = res["fit_B"] / scale
        ax.set_title(
            f"{name}\n"
            f"$N$: spec {spec_n:.4g}, fit {fit_n:.4g} {unit}/$\\sqrt{{Hz}}$   |   "
            f"$B$: spec {spec_b:.4g}, fit {fit_b:.4g} {unit}",
            fontsize=10,
        )
        ax.set_xlabel(r"averaging time $\tau$  [s]")
        ax.set_ylabel(rf"Allan deviation $\sigma(\tau)$  [{unit}]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "E1 — Allan-variance validation of the virtual MPU-6050 noise model",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # 6 h is not arbitrary. The accelerometer's white/flicker crossover sits at
    # tau = (N / (0.6643 B))^2 ~= 227 s, so the flicker floor only becomes
    # visible after ~4 minutes of averaging. Resolving B needs a record long
    # enough to cover a decade beyond that with enough clusters to be stable.
    p.add_argument("--duration", type=float, default=21600.0,
                   help="stationary record length in seconds (default: 21600 = 6 h)")
    p.add_argument("--fs", type=float, default=200.0,
                   help="sample rate in Hz (default: 200)")
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--quick", action="store_true",
                   help="10-minute record for fast iteration; B will be poorly resolved")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "e1_allan.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    duration = 600.0 if args.quick else args.duration

    print("E1 - Allan-variance validation of the virtual MPU-6050")
    print(f"record: {duration / 3600:.3f} h @ {args.fs:g} Hz, seed {args.seed}")
    if args.quick:
        print("NOTE: --quick is for iteration only. The bias-instability floor")
        print("      needs a record many times longer than tau_floor to resolve;")
        print("      expect the B check to fail here. Use the full run to gate.")

    rng = np.random.default_rng(args.seed)

    results, curves = {}, {}
    for kind, name, spec, table in [
        ("gyro", "GYROSCOPE", nm.gyro_spec(), nm.GYRO_RANGES),
        ("accel", "ACCELEROMETER", nm.accel_spec(), nm.ACCEL_RANGES),
    ]:
        res, taus, adev = validate_sensor(
            name, kind, spec, table, duration, args.fs, rng
        )
        results[kind] = res
        curves[kind] = (taus, adev)

    if not args.no_plot:
        plot(results, curves, args.out)

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n{'=' * 72}")
    print(f"E1 RESULT: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'=' * 72}")
    if not all_passed and args.quick:
        print("(--quick was used; re-run without it before drawing conclusions)")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

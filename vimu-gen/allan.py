"""Overlapping Allan variance for inertial sensor characterisation.

Implements the estimator described in IEEE Std 952-1997 (Annex C) and used
throughout the inertial navigation literature to decompose an IMU error
signal into its constituent noise processes.

The five-term Allan variance model for an inertial sensor is

    sigma^2(tau) = 3Q^2/tau^2            quantisation noise      (slope -1)
                 + N^2/tau               white noise / random walk (slope -1/2)
                 + (2 ln2 / pi) B^2      bias instability        (slope  0)
                 + K^2 tau / 3           rate random walk        (slope +1/2)
                 + R^2 tau^2 / 2         rate ramp               (slope +1)

Two identities make this directly usable as a validation gate:

  * White noise:       sigma(tau) = N / sqrt(tau), so sigma(1 s) == N.
                       N is exactly the datasheet "noise spectral density"
                       in units/sqrt(Hz).
  * Bias instability:  the flat floor sits at sigma_min = sqrt(2 ln2 / pi) * B
                       ~= 0.6643 * B.

Units are carried through unchanged: feed rad/s and you get rad/s back.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

# sigma_min = BIAS_INSTABILITY_COEFF * B at the Allan deviation floor.
BIAS_INSTABILITY_COEFF = np.sqrt(2.0 * np.log(2.0) / np.pi)  # ~= 0.6643


def log_spaced_cluster_sizes(n_samples: int, points_per_decade: int = 12) -> np.ndarray:
    """Log-spaced cluster sizes m, valid for an N-sample overlapping estimate.

    The largest usable cluster is N//2 because the estimator needs three
    points spaced m apart (indices k, k+m, k+2m).
    """
    m_max = n_samples // 2
    if m_max < 1:
        raise ValueError(f"need at least 2 samples, got {n_samples}")
    n_points = max(2, int(points_per_decade * np.log10(max(m_max, 2))))
    m = np.unique(np.round(np.logspace(0, np.log10(m_max), n_points)).astype(np.int64))
    return m[m >= 1]


def overlapping_allan_deviation(
    x: np.ndarray,
    dt: float,
    cluster_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal.

    Args:
        x: (N,) or (N, C) array of sensor samples (e.g. rad/s or m/s^2).
           Multiple columns are treated as independent axes.
        dt: sample period in seconds.
        cluster_sizes: optional explicit cluster sizes m. Defaults to
            log-spaced coverage of the full usable range.

    Returns:
        taus: (M,) averaging times in seconds.
        adev: (M,) or (M, C) Allan deviation, same units as ``x``.
        dof:  (M,) number of overlapping clusters used at each tau, which
              sets the confidence of the estimate (few clusters at large tau
              is why the right-hand end of an Allan plot is always noisy).
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n_samples = x.shape[0]

    if cluster_sizes is None:
        cluster_sizes = log_spaced_cluster_sizes(n_samples)
    cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)
    cluster_sizes = cluster_sizes[(cluster_sizes >= 1) & (2 * cluster_sizes < n_samples)]
    if cluster_sizes.size == 0:
        raise ValueError("no valid cluster sizes; signal is too short")

    # theta is the integrated output (angle for a gyro, velocity for an
    # accelerometer). A leading zero keeps theta[k] = dt * sum(x[:k]).
    theta = np.empty((n_samples + 1, x.shape[1]), dtype=np.float64)
    theta[0] = 0.0
    np.cumsum(x, axis=0, out=theta[1:])
    theta *= dt

    taus = cluster_sizes * dt
    adev = np.empty((cluster_sizes.size, x.shape[1]), dtype=np.float64)
    dof = np.empty(cluster_sizes.size, dtype=np.int64)

    for i, m in enumerate(cluster_sizes):
        # Second difference of theta at lag m, over all overlapping clusters.
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[:-2 * m]
        tau = taus[i]
        avar = np.sum(d * d, axis=0) / (2.0 * tau * tau * d.shape[0])
        adev[i] = np.sqrt(avar)
        dof[i] = d.shape[0]

    if squeeze:
        adev = adev[:, 0]
    return taus, adev, dof


def fit_slope(taus: np.ndarray, adev: np.ndarray, tau_lo: float, tau_hi: float) -> float:
    """Least-squares log-log slope of the Allan deviation over a tau window."""
    sel = (taus >= tau_lo) & (taus <= tau_hi)
    if sel.sum() < 2:
        raise ValueError(f"need >=2 points in [{tau_lo}, {tau_hi}] s, got {sel.sum()}")
    slope, _ = np.polyfit(np.log10(taus[sel]), np.log10(adev[sel]), 1)
    return float(slope)


def estimate_white_noise_density(
    taus: np.ndarray,
    adev: np.ndarray,
    tau_lo: float = 0.05,
    tau_hi: float = 1.0,
) -> float:
    """Recover N (units/sqrt(Hz)) by fitting the -1/2 slope region.

    Fits log10(adev) = c - 0.5*log10(tau) with the slope held at exactly
    -1/2, then evaluates at tau = 1 s where sigma(1) == N.
    """
    sel = (taus >= tau_lo) & (taus <= tau_hi)
    if sel.sum() < 2:
        raise ValueError(f"need >=2 points in [{tau_lo}, {tau_hi}] s, got {sel.sum()}")
    # Slope is fixed at -1/2, so only the intercept is free.
    intercept = np.mean(np.log10(adev[sel]) + 0.5 * np.log10(taus[sel]))
    return float(10.0**intercept)


def fit_allan_model(
    taus: np.ndarray, adev: np.ndarray, relative_weighting: bool = True
) -> dict[str, float]:
    """Least-squares fit of the full five-term Allan variance model.

    The model is linear in the *squared* coefficients:

        sigma^2(tau) = Q^2 (3/tau^2) + N^2 (1/tau) + B^2 (2 ln2/pi)
                     + K^2 (tau/3)   + R^2 (tau^2/2)

    so fitting reduces to non-negative least squares over the design matrix
    of those five basis functions. Non-negativity is required because the
    coefficients are variances; an unconstrained fit happily returns
    negative "variances" and then sqrt() produces NaN.

    This is preferred over reading N off a windowed -1/2 slope: near tau = 1 s
    the flicker floor contributes appreciably to the total, so a windowed fit
    systematically overestimates N. Fitting all terms jointly removes that bias.

    Args:
        taus: (M,) averaging times.
        adev: (M,) Allan deviation.
        relative_weighting: weight each point by 1/sigma^2 so that the fit
            is even across the many decades the curve spans, rather than
            being dominated by the large-tau end.

    Returns:
        dict with keys Q, N, B, K, R in the signal's own units.
    """
    taus = np.asarray(taus, dtype=np.float64)
    avar = np.asarray(adev, dtype=np.float64) ** 2

    design = np.column_stack(
        [
            3.0 / taus**2,
            1.0 / taus,
            np.full_like(taus, 2.0 * np.log(2.0) / np.pi),
            taus / 3.0,
            taus**2 / 2.0,
        ]
    )
    target = avar

    if relative_weighting:
        w = 1.0 / np.where(avar > 0, avar, np.finfo(float).tiny)
        design = design * w[:, None]
        target = target * w

    coeffs, _ = nnls(design, target)
    q, n, b, k, r = np.sqrt(coeffs)
    return {"Q": float(q), "N": float(n), "B": float(b), "K": float(k), "R": float(r)}


def estimate_bias_instability(taus: np.ndarray, adev: np.ndarray) -> tuple[float, float]:
    """Recover B (same units as the signal) from the Allan deviation floor.

    Returns (B, tau_at_floor). The floor is located as the minimum of the
    Allan deviation curve, which is only meaningful if the record is long
    enough for the flicker plateau to appear.
    """
    idx = int(np.argmin(adev))
    return float(adev[idx] / BIAS_INSTABILITY_COEFF), float(taus[idx])

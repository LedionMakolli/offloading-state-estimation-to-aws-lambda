"""Strapdown inertial navigation mechanisation.

This is the Tier-0 computation: the loop that must run on the Pico at the
full sample rate under a hard deadline. It is written as an explicit
per-sample loop rather than a vectorised expression precisely because it is
the reference for the C implementation in ``firmware/src/nav_math.c`` -- the two
should be readable side by side.

Mechanisation, per step:

  1. Attitude. Integrate the angular rate as a rotation vector and apply it
     as a body-frame increment:  q_{k+1} = q_k (x) exp(w_mid dt / 2).
     The midpoint rate 0.5*(w_k + w_{k+1}) is used so the attitude update is
     second-order accurate; the naive q_k (x) exp(w_k dt / 2) is only first
     order and dominates the total error budget if used.
  2. Specific force to nav frame:  a_nav = R(q) f_body + g_nav.
  3. Velocity, trapezoidal:  v_{k+1} = v_k + (a_k + a_{k+1}) dt / 2.
  4. Position, matching order:  p_{k+1} = p_k + v_k dt + (2 a_k + a_{k+1}) dt^2 / 6.

There is no aiding of any kind here. That is the point: this is the unaided
dead-reckoning baseline whose divergence motivates offloading the bias
estimation, and it is what E2 measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import rotations as rot


@dataclass
class NavSolution:
    """Output of the mechanisation, sampled at every input step."""

    t: np.ndarray          # (N,)
    quat: np.ndarray       # (N, 4)
    velocity: np.ndarray   # (N, 3)
    position: np.ndarray   # (N, 3)


def mechanise(
    omega_body: np.ndarray,
    accel_body: np.ndarray,
    dt: float,
    q0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    t0: float = 0.0,
) -> NavSolution:
    """Integrate IMU measurements into a navigation solution.

    Args:
        omega_body: (N, 3) angular rate, rad/s, body frame.
        accel_body: (N, 3) specific force, m/s^2, body frame.
        dt: sample period, seconds.
        q0, v0, p0: initial attitude (body-to-nav quaternion), velocity and
            position in the nav frame. Supplying the true initial state is
            deliberate -- it isolates sensor-driven drift from initialisation
            error, which is a separate error source.

    Returns:
        NavSolution with one entry per input sample.
    """
    omega_body = np.asarray(omega_body, dtype=np.float64)
    accel_body = np.asarray(accel_body, dtype=np.float64)
    if omega_body.shape != accel_body.shape:
        raise ValueError(
            f"gyro {omega_body.shape} and accel {accel_body.shape} must match"
        )
    n = omega_body.shape[0]

    quat = np.empty((n, 4))
    vel = np.empty((n, 3))
    pos = np.empty((n, 3))

    quat[0] = rot.quat_normalize(np.asarray(q0, dtype=np.float64))
    vel[0] = np.asarray(v0, dtype=np.float64)
    pos[0] = np.asarray(p0, dtype=np.float64)

    # Acceleration in the nav frame at the current step, carried across
    # iterations so the trapezoidal update costs one rotation per step.
    accel_nav_k = rot.quat_to_dcm(quat[0]) @ accel_body[0] + rot.G_NAV

    half_dt = 0.5 * dt
    dt2_over_6 = dt * dt / 6.0

    for k in range(n - 1):
        # 1. Attitude, midpoint rate for second-order accuracy.
        omega_mid = 0.5 * (omega_body[k] + omega_body[k + 1])
        dq = rot.quat_from_rotvec(omega_mid * dt)
        q_next = rot.quat_normalize(rot.quat_mul(quat[k], dq))
        quat[k + 1] = q_next

        # 2. Specific force into the nav frame at the new attitude.
        accel_nav_next = rot.quat_to_dcm(q_next) @ accel_body[k + 1] + rot.G_NAV

        # 3-4. Velocity and position.
        vel[k + 1] = vel[k] + (accel_nav_k + accel_nav_next) * half_dt
        pos[k + 1] = (
            pos[k] + vel[k] * dt + (2.0 * accel_nav_k + accel_nav_next) * dt2_over_6
        )

        accel_nav_k = accel_nav_next

    t = t0 + np.arange(n, dtype=np.float64) * dt
    return NavSolution(t=t, quat=quat, velocity=vel, position=pos)


def position_error(nav: NavSolution, truth_position: np.ndarray) -> np.ndarray:
    """Per-sample Euclidean position error in metres."""
    return np.linalg.norm(nav.position - truth_position, axis=1)


def attitude_error_deg(nav: NavSolution, truth_quat: np.ndarray) -> np.ndarray:
    """Per-sample total attitude error in degrees."""
    return np.rad2deg(rot.angular_error(nav.quat, truth_quat))


def summarise_error(
    nav: NavSolution, truth_position: np.ndarray, truth_quat: np.ndarray
) -> dict[str, float]:
    """Standard error metrics for a dead-reckoning run."""
    perr = position_error(nav, truth_position)
    aerr = attitude_error_deg(nav, truth_quat)
    return {
        "final_position_error_m": float(perr[-1]),
        "rms_position_error_m": float(np.sqrt(np.mean(perr**2))),
        "max_position_error_m": float(np.max(perr)),
        "final_attitude_error_deg": float(aerr[-1]),
        "rms_attitude_error_deg": float(np.sqrt(np.mean(aerr**2))),
    }


def fit_drift_exponent(
    t: np.ndarray, error: np.ndarray, t_min: float = 1.0
) -> tuple[float, float]:
    """Fit error(t) = c * t^n and return (n, c).

    The theoretical expectation for unaided inertial dead reckoning is that
    an uncompensated accelerometer bias integrates twice into position
    (n = 2), while an uncompensated gyro bias tilts the gravity vector and
    integrates into position three times (n = 3). Real runs land between the
    two depending on which error source dominates over the window observed,
    so this returns the measured exponent rather than assuming one.
    """
    sel = (t >= t_min) & (error > 0)
    if sel.sum() < 2:
        raise ValueError("not enough positive-error samples to fit a drift exponent")
    n, log_c = np.polyfit(np.log(t[sel]), np.log(error[sel]), 1)
    return float(n), float(np.exp(log_c))

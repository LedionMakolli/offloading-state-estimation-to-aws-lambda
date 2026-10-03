"""Quaternion and Euler-angle utilities for strapdown navigation.

Conventions, fixed once and used everywhere:

  * Quaternions are scalar-first, ``q = [w, x, y, z]``, unit norm, and rotate
    a vector from the *body* frame to the *navigation* frame:
    ``v_nav = R(q) @ v_body``.
  * Euler angles are the aerospace 3-2-1 sequence (yaw about z, then pitch
    about y, then roll about x), applied as ``R = Rz(yaw) Ry(pitch) Rx(roll)``.
  * The navigation frame is ENU: x East, y North, z Up. Gravity is therefore
    ``g_nav = [0, 0, -9.80665]`` m/s^2.

The ENU choice matches EuRoC/TUM-VI ground truth, which avoids a frame
conversion at the dataset boundary. Aerospace texts more often use NED; the
mechanisation is identical, only the sign of the gravity vector and the axis
labels change.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.80665                        # m/s^2, standard gravity
G_NAV = np.array([0.0, 0.0, -GRAVITY])   # ENU: gravity points down (-z)


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q (x) r, scalar-first. Supports (4,) or (N,4)."""
    q = np.atleast_2d(q)
    r = np.atleast_2d(r)
    w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    out = np.empty((max(q.shape[0], r.shape[0]), 4))
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out[0] if (q.shape[0] == 1 and r.shape[0] == 1) else out


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate, which for a unit quaternion is the inverse rotation."""
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Renormalise to unit length, guarding against a zero quaternion."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Quaternion from a rotation vector (axis * angle, radians).

    Uses the small-angle Taylor expansion of sin(t/2)/t below a threshold so
    that the 0/0 at zero rotation is handled exactly rather than by luck.
    """
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    theta = np.linalg.norm(v, axis=1)
    out = np.empty((v.shape[0], 4))
    small = theta < 1e-8
    # sin(theta/2)/theta -> 1/2 - theta^2/48 as theta -> 0
    k = np.where(small, 0.5 - theta**2 / 48.0, np.sin(theta / 2.0) / np.where(small, 1.0, theta))
    out[:, 0] = np.cos(theta / 2.0)
    out[:, 1:] = v * k[:, None]
    return out[0] if v.shape[0] == 1 else out


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix R with v_nav = R @ v_body. Supports (4,) or (N,4)."""
    q = np.atleast_2d(quat_normalize(q))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r[0] if q.shape[0] == 1 else r


def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """3-2-1 Euler angles (radians) to a body-to-nav quaternion."""
    roll, pitch, yaw = np.atleast_1d(roll), np.atleast_1d(pitch), np.atleast_1d(yaw)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = np.empty((roll.shape[0], 4))
    q[:, 0] = cy * cp * cr + sy * sp * sr
    q[:, 1] = cy * cp * sr - sy * sp * cr
    q[:, 2] = cy * sp * cr + sy * cp * sr
    q[:, 3] = sy * cp * cr - cy * sp * sr
    return q[0] if roll.shape[0] == 1 else q


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Body-to-nav quaternion to 3-2-1 Euler angles, returned as (..., 3).

    Pitch is clamped before arcsin so that gimbal-lock inputs saturate at
    +/-90 deg instead of producing NaN.
    """
    q = np.atleast_2d(quat_normalize(q))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3))
    out[:, 0] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    out[:, 1] = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    out[:, 2] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return out[0] if q.shape[0] == 1 else out


def euler_rates_to_body_rates(
    roll: np.ndarray,
    pitch: np.ndarray,
    roll_dot: np.ndarray,
    pitch_dot: np.ndarray,
    yaw_dot: np.ndarray,
) -> np.ndarray:
    """Convert 3-2-1 Euler rates to body angular rates omega_b = [p, q, r].

        p = roll_dot                       - yaw_dot sin(pitch)
        q =  pitch_dot cos(roll) + yaw_dot cos(pitch) sin(roll)
        r = -pitch_dot sin(roll) + yaw_dot cos(pitch) cos(roll)
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    omega = np.empty((np.atleast_1d(roll).shape[0], 3))
    omega[:, 0] = roll_dot - yaw_dot * sp
    omega[:, 1] = pitch_dot * cr + yaw_dot * cp * sr
    omega[:, 2] = -pitch_dot * sr + yaw_dot * cp * cr
    return omega


def angular_error(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Total rotation angle (radians) between two attitude quaternions.

    Computed as the magnitude of the relative rotation, which is the
    frame-independent attitude error used to report AHRS accuracy.
    """
    # quat_mul collapses to 1-D when both operands are single quaternions, so
    # restore the leading axis before indexing the scalar part.
    q_err = np.atleast_2d(
        quat_mul(quat_conj(np.atleast_2d(q_true)), np.atleast_2d(q_est))
    )
    w = np.clip(np.abs(q_err[:, 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)

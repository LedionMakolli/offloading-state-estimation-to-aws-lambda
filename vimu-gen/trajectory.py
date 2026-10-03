"""Analytic 6-DoF trajectory with closed-form inertial measurements.

Why an analytic trajectory when real datasets exist:

A dataset gives you sampled *poses*. To get the specific force and angular
rate an IMU would have seen, you must differentiate those poses twice, which
injects numerical error that is indistinguishable from sensor error. That
makes it impossible to tell whether drift in the navigator comes from the
sensor model or from the differentiation.

Here, position, velocity, acceleration, attitude and angular rate are all
available in closed form, so the inertial measurements are *exact*. Any drift
observed downstream is therefore attributable to the noise model alone. This
is the reference against which the strapdown mechanisation is verified before
any real dataset is used.

The motion is a 3-D Lissajous figure with independently oscillating attitude,
chosen because it excites all three axes of both sensors continuously and
never settles into a trivially observable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import rotations as rot


@dataclass
class TrajectorySample:
    """Ground truth plus the exact inertial measurements at each time step."""

    t: np.ndarray          # (N,)   seconds
    position: np.ndarray   # (N, 3) m, ENU nav frame
    velocity: np.ndarray   # (N, 3) m/s, nav frame
    accel_nav: np.ndarray  # (N, 3) m/s^2, nav frame (true kinematic acceleration)
    quat: np.ndarray       # (N, 4) body-to-nav attitude
    euler: np.ndarray      # (N, 3) roll, pitch, yaw [rad]
    omega_body: np.ndarray  # (N, 3) rad/s, the ideal gyroscope output
    accel_body: np.ndarray  # (N, 3) m/s^2, the ideal accelerometer output (specific force)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def __len__(self) -> int:
        return self.t.shape[0]


@dataclass
class LissajousTrajectory:
    """Smooth analytic flight path with continuously excited attitude.

    Defaults describe a gentle indoor-drone-scale motion: a few metres of
    travel, sub-g accelerations and tens of degrees per second of rotation,
    which keeps the signal well inside the MPU-6050's +/-250 deg/s and +/-2 g
    default full-scale ranges so that quantisation, not clipping, is the
    limiting deterministic error.
    """

    # Translational: amplitude [m], angular frequency [rad/s], phase [rad]
    amplitude: np.ndarray = field(default_factory=lambda: np.array([3.0, 2.0, 0.5]))
    omega_pos: np.ndarray = field(default_factory=lambda: np.array([0.40, 0.80, 1.10]))
    phase_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi / 4, np.pi / 3]))
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.5]))

    # Attitude: amplitude [rad], angular frequency [rad/s]
    euler_amplitude: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([12.0, 9.0, 30.0]))
    )
    omega_euler: np.ndarray = field(default_factory=lambda: np.array([0.70, 0.50, 0.30]))
    phase_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi / 5, 0.0]))

    def sample(self, t: np.ndarray) -> TrajectorySample:
        """Evaluate the trajectory and its exact inertial measurements at times t."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))

        # --- Translation: p, v, a all differentiated in closed form ---
        arg = self.omega_pos[None, :] * t[:, None] + self.phase_pos[None, :]
        amp = self.amplitude[None, :]
        w = self.omega_pos[None, :]

        position = self.origin[None, :] + amp * np.sin(arg)
        velocity = amp * w * np.cos(arg)
        accel_nav = -amp * w**2 * np.sin(arg)

        # --- Attitude: Euler angles and their exact rates ---
        arg_e = self.omega_euler[None, :] * t[:, None] + self.phase_euler[None, :]
        amp_e = self.euler_amplitude[None, :]
        w_e = self.omega_euler[None, :]

        euler = amp_e * np.sin(arg_e)              # roll, pitch, yaw
        euler_dot = amp_e * w_e * np.cos(arg_e)

        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
        quat = rot.euler_to_quat(roll, pitch, yaw)
        quat = np.atleast_2d(quat)

        omega_body = rot.euler_rates_to_body_rates(
            roll, pitch, euler_dot[:, 0], euler_dot[:, 1], euler_dot[:, 2]
        )

        # --- Specific force: what the accelerometer actually senses ---
        # f_nav = a_nav - g_nav. At rest this is +9.81 m/s^2 on the ENU z axis,
        # which is why a level accelerometer reads +1 g upward, not zero.
        f_nav = accel_nav - rot.G_NAV[None, :]
        dcm = np.atleast_3d(rot.quat_to_dcm(quat))
        if dcm.ndim == 2:
            dcm = dcm[None, :, :]
        # Rotate nav -> body with the transpose of the body-to-nav DCM.
        accel_body = np.einsum("nji,nj->ni", dcm, f_nav)

        return TrajectorySample(
            t=t,
            position=position,
            velocity=velocity,
            accel_nav=accel_nav,
            quat=quat,
            euler=euler,
            omega_body=omega_body,
            accel_body=accel_body,
        )

    def generate(self, duration_s: float, fs: float) -> TrajectorySample:
        """Sample the trajectory uniformly at ``fs`` Hz for ``duration_s`` seconds."""
        n = int(round(duration_s * fs))
        t = np.arange(n, dtype=np.float64) / fs
        return self.sample(t)


def peak_measurements(traj: TrajectorySample) -> dict[str, float]:
    """Peak |gyro| and |accel| per axis, for checking full-scale range choice.

    Selecting a full-scale range is a real design decision: too small clips the
    signal, too large wastes ADC resolution and raises quantisation noise. This
    reports the numbers needed to justify that choice in the thesis.
    """
    return {
        "max_gyro_dps": float(np.max(np.abs(traj.omega_body)) * 180.0 / np.pi),
        "max_accel_g": float(np.max(np.abs(traj.accel_body)) / rot.GRAVITY),
        "max_speed_mps": float(np.max(np.linalg.norm(traj.velocity, axis=1))),
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(traj.position, axis=0), axis=1))
        ),
    }


# --------------------------------------------------------------------------
# Mission profile: alternating stationary and dynamic phases
# --------------------------------------------------------------------------
def _smoothstep5(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic smoothstep and its first two derivatives, clamped to [0, 1].

    The quintic 6x^5 - 15x^4 + 10x^3 is used rather than the more common cubic
    because its second derivative vanishes at both endpoints. The envelope
    multiplies position, so a discontinuous second derivative would appear as a
    step in specific force -- an artificial impulse the accelerometer would see
    but no real vehicle would produce.

    Clamping handles the outside-range case automatically: both derivatives
    evaluate to exactly zero at x = 0 and x = 1.
    """
    x = np.clip(x, 0.0, 1.0)
    s = x * x * x * (x * (6.0 * x - 15.0) + 10.0)
    ds = 30.0 * x * x * (x - 1.0) * (x - 1.0)
    dds = 60.0 * x * (x * (2.0 * x - 3.0) + 1.0)
    return s, ds, dds


def _motion_envelope(
    t: np.ndarray, intervals: list[tuple[float, float]], ramp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Envelope that is 0 when stationary and 1 during motion, with C2 ramps.

    Returns (s, s_dot, s_ddot). Intervals must not overlap.
    """
    s = np.zeros_like(t)
    sd = np.zeros_like(t)
    sdd = np.zeros_like(t)
    for t0, t1 in intervals:
        if t1 - t0 < 2 * ramp:
            raise ValueError(
                f"motion interval ({t0}, {t1}) is shorter than two ramps of {ramp} s"
            )
        a, a_d, a_dd = _smoothstep5((t - t0) / ramp)
        a_d, a_dd = a_d / ramp, a_dd / ramp**2

        b_s, b_d, b_dd = _smoothstep5((t - (t1 - ramp)) / ramp)
        b = 1.0 - b_s
        b_d, b_dd = -b_d / ramp, -b_dd / ramp**2

        s += a * b
        sd += a_d * b + a * b_d
        sdd += a_dd * b + 2.0 * a_d * b_d + a * b_dd
    return s, sd, sdd


@dataclass
class MissionTrajectory(LissajousTrajectory):
    """Lissajous motion gated by an envelope, with stationary phases between.

    Stationary phases are what make the inertial biases observable at all. With
    no aiding of any kind, an IMU-only filter cannot separate a constant
    accelerometer bias from a gravity-direction error, nor observe gyro bias --
    the states are structurally unobservable. Detected stationary intervals
    supply three pseudo-measurements that break that degeneracy:

      ZUPT     velocity is zero
      ZARU     angular rate is zero, so the gyro reads its own bias directly
      gravity  specific force is exactly -g, fixing roll and pitch

    During a stationary phase the envelope and both its derivatives are zero,
    so position holds at the origin, velocity and acceleration are identically
    zero, and attitude is exactly level. That makes the pseudo-measurements
    true by construction rather than approximately true.
    """

    motion_intervals: list[tuple[float, float]] = field(
        default_factory=lambda: [(5.0, 25.0), (30.0, 55.0)]
    )
    ramp: float = 2.0

    # Attitude held during each static phase, in degrees (roll, pitch, yaw).
    # There is one more static phase than there are motion intervals.
    #
    # These MUST differ from each other. A gravity update measures
    # h = R^T g + b_a, in which a tilt about x and a bias on y produce an
    # identical signature -- only their combination is observable. Resting at
    # a single attitude therefore leaves horizontal accelerometer bias
    # permanently unobservable no matter how many updates are applied, and the
    # filter will split the observable combination arbitrarily between tilt and
    # bias. Observing from several distinct orientations breaks the degeneracy.
    # This is exactly the principle behind classical multi-position IMU
    # calibration, applied opportunistically instead of on a rate table.
    rest_attitudes: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.0, 0.0, 0.0),
            (25.0, -15.0, 40.0),
            (-20.0, 30.0, -60.0),
        ]
    )

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        s, sd, sdd = _motion_envelope(t, self.motion_intervals, self.ramp)

        # --- Translation, with the envelope folded in via the product rule ---
        arg = self.omega_pos[None, :] * t[:, None] + self.phase_pos[None, :]
        amp = self.amplitude[None, :]
        w = self.omega_pos[None, :]
        sin_a, cos_a = np.sin(arg), np.cos(arg)

        base = amp * sin_a
        base_d = amp * w * cos_a
        base_dd = -amp * w**2 * sin_a

        position = self.origin[None, :] + s[:, None] * base
        velocity = sd[:, None] * base + s[:, None] * base_d
        accel_nav = (
            sdd[:, None] * base + 2.0 * sd[:, None] * base_d + s[:, None] * base_dd
        )

        # --- Attitude: oscillates during motion, holds a pose while at rest ---
        arg_e = self.omega_euler[None, :] * t[:, None] + self.phase_euler[None, :]
        amp_e = self.euler_amplitude[None, :]
        w_e = self.omega_euler[None, :]
        sin_e, cos_e = np.sin(arg_e), np.cos(arg_e)

        rest = self._rest_euler(t)
        euler = (1.0 - s)[:, None] * rest + s[:, None] * (amp_e * sin_e)
        euler_dot = (
            -sd[:, None] * rest
            + sd[:, None] * (amp_e * sin_e)
            + s[:, None] * (amp_e * w_e * cos_e)
        )

        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
        quat = np.atleast_2d(rot.euler_to_quat(roll, pitch, yaw))
        omega_body = rot.euler_rates_to_body_rates(
            roll, pitch, euler_dot[:, 0], euler_dot[:, 1], euler_dot[:, 2]
        )

        f_nav = accel_nav - rot.G_NAV[None, :]
        dcm = rot.quat_to_dcm(quat)
        if dcm.ndim == 2:
            dcm = dcm[None, :, :]
        accel_body = np.einsum("nji,nj->ni", dcm, f_nav)

        return TrajectorySample(
            t=t,
            position=position,
            velocity=velocity,
            accel_nav=accel_nav,
            quat=quat,
            euler=euler,
            omega_body=omega_body,
            accel_body=accel_body,
        )

    def _rest_euler(self, t: np.ndarray) -> np.ndarray:
        """Held attitude for the static phase containing each time in t.

        Poses switch at the midpoint of each motion interval, where the
        envelope is exactly 1 and both (1 - s) and its derivative vanish. The
        piecewise-constant switch is therefore invisible in the emitted
        attitude, angular rate and specific force.
        """
        poses = np.deg2rad(np.asarray(self.rest_attitudes, dtype=np.float64))
        n_phases = len(self.motion_intervals) + 1
        if poses.shape[0] < n_phases:
            raise ValueError(
                f"need {n_phases} rest attitudes for "
                f"{len(self.motion_intervals)} motion intervals, "
                f"got {poses.shape[0]}"
            )
        switches = np.array([0.5 * (a + b) for a, b in self.motion_intervals])
        return poses[np.searchsorted(switches, t)]

    def stationary_mask(self, t: np.ndarray) -> np.ndarray:
        """Ground-truth stationary flag. For scoring the detector only.

        The filter must never use this -- it has to decide from the noisy
        measurements alone, exactly as the firmware will.
        """
        s, sd, _ = _motion_envelope(np.atleast_1d(t), self.motion_intervals, self.ramp)
        return (s <= 1e-12) & (np.abs(sd) <= 1e-12)

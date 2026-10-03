"""Error-state Kalman filter for inertial bias estimation.

This is the payload that gets offloaded. It is deliberately kept as a pure
function of (IMU window, initial state) -> (corrected state, bias estimates)
so that the identical code can run in three places -- host, AWS Lambda, and
Greengrass -- which is what makes the planned tier comparison an apples-to-apples
measurement rather than a comparison of two different implementations.

Observability
-------------
An IMU-only filter cannot estimate its own biases. Accelerometer bias is
indistinguishable from a gravity-direction (tilt) error, and gyro bias is not
observable at all without something to fix the attitude against. The states are
structurally unobservable, and no amount of filtering fixes that.

Detected stationary intervals supply the missing constraints:

  ZUPT     velocity is zero               -> bounds velocity error
  ZARU     angular rate is zero, so the gyro reads its own bias directly
           -> makes gyro bias *directly* observable, no inference needed
  gravity  specific force is exactly -g   -> fixes roll and pitch

ZARU is the important one. E2 showed gyro bias is the binding constraint on
position drift (it enters as t^3 via tilt-induced gravity leakage, against t^2
for accelerometer bias), and ZARU observes it directly rather than inferring it.

What this does and does not achieve
-----------------------------------
Bounding attitude error removes the t^3 term, leaving t^2. Position still
drifts without bound -- there is no position-fixing measurement anywhere in
this system, and there cannot be one in an IMU-only architecture. The claim is
the removal of the dominant error mode, not a solution to unaided positioning.

Error-state convention
----------------------
Global (nav-frame) angular error, following Sola, "Quaternion kinematics for
the error-state Kalman filter" (2017), section 6:

    p = p_hat + dp,  v = v_hat + dv,  q = dq (x) q_hat,
    b_a = b_a_hat + db_a,  b_g = b_g_hat + db_g

Error state ordering is [dp, dv, dtheta, db_a, db_g], 15 states.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

import rotations as rot

# Error-state index blocks.
I_P = slice(0, 3)
I_V = slice(3, 6)
I_TH = slice(6, 9)
I_BA = slice(9, 12)
I_BG = slice(12, 15)
N_STATE = 15

G_UP = -rot.G_NAV  # [0, 0, +9.80665]: the specific force a level sensor reads at rest


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass
class ESKFConfig:
    """Filter tuning. Defaults are matched to the virtual MPU-6050."""

    # Process noise: IMU white-noise densities (units/sqrt(Hz)).
    gyro_noise: float = 0.005 * np.pi / 180.0
    accel_noise: float = 400e-6 * 9.80665

    # Bias process noise (units/sqrt(s)).
    #
    # The real bias process is flicker (1/f), which is NOT a random walk. The
    # filter models it as one anyway, because a random walk is Markov and
    # flicker is not -- representing flicker exactly would need an
    # infinite-dimensional state. So K has to be chosen, and it is chosen in
    # two steps:
    #
    #  1. Match Allan deviations over the interval the filter must coast
    #     between stationary phases. A random walk has sigma(tau) = K sqrt(tau/3)
    #     and the flicker floor is 0.6643 B, so K = 1.15 B / sqrt(tau_coast).
    #     With B_gyro = 0.01 deg/s and tau_coast = 25 s this gives 4.0e-5.
    #
    #  2. Inflate by ~2.6x. Matching at a single tau understates flicker's
    #     low-frequency power, so the step-1 value leaves the filter
    #     overconfident (measured |error|/sigma = 2.2 rather than 1).
    #
    # The combined rule is K = 3.0 B / sqrt(tau_coast). Verified by sweep: this
    # lands |error|/sigma at 1.12 (gyro) and 0.97 (accel), with RMS bias error
    # within 7% of its achievable minimum. Larger K makes the filter
    # conservative and measurably degrades the estimate.
    gyro_bias_rw: float = 1.047e-4      # B_gyro  = 0.01 deg/s -> 1.745e-4 rad/s
    accel_bias_rw: float = 2.352e-4     # B_accel = 0.04 mg    -> 3.92e-4 m/s^2

    # Pseudo-measurement noise.
    #
    # ZUPT sigma expresses how exactly "stationary" means zero velocity. That
    # is a modelling choice about the motion, not a sensor property.
    zupt_sigma: float = 0.01            # m/s
    #
    # ZARU and gravity noise are NOT free parameters. Each update consumes a
    # single sample of a sensor, so its uncertainty is the white-noise density
    # referred to the sample rate:  sigma = N / sqrt(dt).  At 200 Hz that is
    # 0.071 deg/s for the gyro and 0.055 m/s^2 for the accelerometer -- roughly
    # 35x larger than a plausible-looking hand-picked value. Understating them
    # makes the filter overconfident: it drives the estimate hard on every
    # sample and then reports a covariance far smaller than its actual error.
    # Left as None, both are derived from the rate in resolved().
    zaru_sigma: float | None = None     # rad/s
    gravity_sigma: float | None = None  # m/s^2
    # Allowance on the gravity update for un-modelled vibration and for the
    # fact that "stationary" is a detector decision, not a guarantee.
    gravity_margin: float = 2.0

    # Initial uncertainty.
    p0_sigma: float = 1e-3
    v0_sigma: float = 1e-3
    theta0_sigma: float = np.deg2rad(1.0)
    ba0_sigma: float = 50e-3 * 9.80665
    bg0_sigma: float = np.deg2rad(1.0)

    # Stationary detector (SHOE-style generalised likelihood ratio test).
    detector_window: int = 40           # samples
    detector_threshold: float = 8.0e4

    def resolved(self, dt: float) -> "ESKFConfig":
        """Return a copy with the rate-dependent measurement noises filled in."""
        cfg = replace(self)
        if cfg.zaru_sigma is None:
            cfg.zaru_sigma = self.gyro_noise / np.sqrt(dt)
        if cfg.gravity_sigma is None:
            cfg.gravity_sigma = self.gravity_margin * self.accel_noise / np.sqrt(dt)
        return cfg


@dataclass
class ESKFResult:
    """Per-sample filter output plus the final state."""

    t: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    quat: np.ndarray
    bias_accel: np.ndarray     # (N, 3)
    bias_gyro: np.ndarray      # (N, 3)
    sigma_bias_gyro: np.ndarray   # (N, 3) 1-sigma from P
    sigma_bias_accel: np.ndarray  # (N, 3)
    stationary: np.ndarray     # (N,) bool, as decided by the detector
    n_updates: dict[str, int] = field(default_factory=dict)
    # Final covariance. Required for windowed operation: a stateless FaaS
    # invocation cannot retain P between calls, so it has to be handed back to
    # the caller and returned with the next window.
    P_final: np.ndarray | None = None


def stationary_detector(
    gyro: np.ndarray,
    accel: np.ndarray,
    cfg: ESKFConfig,
) -> np.ndarray:
    """SHOE-style zero-velocity detector operating on measurements alone.

    Implements the generalised likelihood ratio statistic of Skog et al.,
    "Zero-velocity detection -- an algorithm evaluation" (IEEE TBME, 2010):

        T = (1/W) sum_i [ ||a_i - g * a_bar/||a_bar|| ||^2 / sigma_a^2
                          + ||w_i||^2 / sigma_g^2 ]

    The first term asks whether the accelerometer is reading pure gravity in a
    consistent direction; the second asks whether the gyro is quiet. Declaring
    stationarity requires both.

    This is deliberately causal and uses no ground truth -- it must run
    identically in the firmware.
    """
    n = gyro.shape[0]
    w = cfg.detector_window
    if n < w:
        return np.zeros(n, dtype=bool)

    var_a = cfg.accel_noise**2
    var_g = cfg.gyro_noise**2

    stat = np.full(n, np.inf)
    # Sliding window; W is small (tens of samples) so the direct form is fine
    # and mirrors what the firmware will do with a ring buffer.
    for k in range(n - w + 1):
        a_win = accel[k : k + w]
        g_win = gyro[k : k + w]
        a_bar = a_win.mean(axis=0)
        norm = np.linalg.norm(a_bar)
        if norm < 1e-9:
            continue
        resid = a_win - rot.GRAVITY * (a_bar / norm)
        t_stat = (
            np.sum(resid * resid) / var_a + np.sum(g_win * g_win) / var_g
        ) / w
        stat[k + w // 2] = t_stat

    return stat < cfg.detector_threshold


class ESKF:
    """Error-state Kalman filter over a 15-state inertial error model."""

    def __init__(
        self,
        cfg: ESKFConfig,
        q0: np.ndarray,
        v0: np.ndarray,
        p0: np.ndarray,
        bg0: np.ndarray | None = None,
        ba0: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.q = rot.quat_normalize(np.asarray(q0, dtype=np.float64).copy())
        self.v = np.asarray(v0, dtype=np.float64).copy()
        self.p = np.asarray(p0, dtype=np.float64).copy()
        self.bg = np.zeros(3) if bg0 is None else np.asarray(bg0, dtype=np.float64).copy()
        self.ba = np.zeros(3) if ba0 is None else np.asarray(ba0, dtype=np.float64).copy()

        self.P = np.diag(
            np.concatenate([
                np.full(3, cfg.p0_sigma**2),
                np.full(3, cfg.v0_sigma**2),
                np.full(3, cfg.theta0_sigma**2),
                np.full(3, cfg.ba0_sigma**2),
                np.full(3, cfg.bg0_sigma**2),
            ])
        )
        self.counts = {"zupt": 0, "zaru": 0, "gravity": 0}

    # -- propagation -------------------------------------------------------
    def predict(self, gyro_m: np.ndarray, accel_m: np.ndarray, dt: float) -> None:
        """Propagate nominal state and covariance by one IMU sample."""
        cfg = self.cfg
        omega = gyro_m - self.bg
        accel_b = accel_m - self.ba

        dcm = rot.quat_to_dcm(self.q)
        accel_n = dcm @ accel_b + rot.G_NAV

        # Nominal state: same mechanisation as strapdown.py.
        self.p = self.p + self.v * dt + 0.5 * accel_n * dt * dt
        self.v = self.v + accel_n * dt
        self.q = rot.quat_normalize(
            rot.quat_mul(self.q, rot.quat_from_rotvec(omega * dt))
        )

        # Error-state transition (global angular error).
        f = np.eye(N_STATE)
        f[I_P, I_V] = np.eye(3) * dt
        f[I_V, I_TH] = -skew(dcm @ accel_b) * dt
        f[I_V, I_BA] = -dcm * dt
        f[I_TH, I_BG] = -dcm * dt
        # f[I_TH, I_TH] stays I: with the global error definition the angular
        # error has no self-coupling term to first order.

        q_proc = np.zeros((N_STATE, N_STATE))
        q_proc[I_V, I_V] = np.eye(3) * (cfg.accel_noise**2 * dt)
        q_proc[I_TH, I_TH] = np.eye(3) * (cfg.gyro_noise**2 * dt)
        q_proc[I_BA, I_BA] = np.eye(3) * (cfg.accel_bias_rw**2 * dt)
        q_proc[I_BG, I_BG] = np.eye(3) * (cfg.gyro_bias_rw**2 * dt)

        self.P = f @ self.P @ f.T + q_proc

    # -- correction --------------------------------------------------------
    def _update(self, h: np.ndarray, residual: np.ndarray, r_meas: np.ndarray) -> None:
        """Joseph-form covariance update, then inject and reset the error state."""
        s = h @ self.P @ h.T + r_meas
        k = np.linalg.solve(s.T, (self.P @ h.T).T).T   # K = P H^T S^-1
        dx = k @ residual

        i_kh = np.eye(N_STATE) - k @ h
        # Joseph form keeps P symmetric positive-definite under repeated
        # updates; the shorter (I-KH)P form drifts asymmetric and can lose
        # positive-definiteness over a long run.
        self.P = i_kh @ self.P @ i_kh.T + k @ r_meas @ k.T
        self.P = 0.5 * (self.P + self.P.T)

        # Inject the error into the nominal state.
        self.p = self.p + dx[I_P]
        self.v = self.v + dx[I_V]
        # Global error: q <- dq (x) q_hat
        self.q = rot.quat_normalize(
            rot.quat_mul(rot.quat_from_rotvec(dx[I_TH]), self.q)
        )
        self.ba = self.ba + dx[I_BA]
        self.bg = self.bg + dx[I_BG]

    def update_zupt(self) -> None:
        """Velocity is zero while stationary."""
        h = np.zeros((3, N_STATE))
        h[:, I_V] = np.eye(3)
        self._update(h, -self.v, np.eye(3) * self.cfg.zupt_sigma**2)
        self.counts["zupt"] += 1

    def update_zaru(self, gyro_m: np.ndarray) -> None:
        """Angular rate is zero, so the gyro output *is* its bias."""
        h = np.zeros((3, N_STATE))
        h[:, I_BG] = -np.eye(3)
        residual = self.bg - gyro_m          # z - h(x) with z = 0, h = w_m - bg
        self._update(h, residual, np.eye(3) * self.cfg.zaru_sigma**2)
        self.counts["zaru"] += 1

    def update_gravity(self, accel_m: np.ndarray) -> None:
        """Specific force is exactly -g while stationary, fixing roll and pitch."""
        dcm = rot.quat_to_dcm(self.q)
        predicted = dcm.T @ G_UP + self.ba
        h = np.zeros((3, N_STATE))
        h[:, I_TH] = dcm.T @ skew(G_UP)
        h[:, I_BA] = np.eye(3)
        self._update(h, accel_m - predicted, np.eye(3) * self.cfg.gravity_sigma**2)
        self.counts["gravity"] += 1


def run_eskf(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float,
    q0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    cfg: ESKFConfig | None = None,
    stationary: np.ndarray | None = None,
    enable_zupt: bool = True,
    enable_zaru: bool = True,
    enable_gravity: bool = True,
    bg0: np.ndarray | None = None,
    ba0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> ESKFResult:
    """Run the filter over a window of IMU samples.

    Args:
        stationary: optional precomputed detector output. If None it is
            computed from the measurements, which is the operational path.
        bg0, ba0, P0: carried-in filter state. Supplying these is what makes
            windowed operation work across separate invocations -- without
            them each window would restart from a diffuse prior and a short
            window containing no stationary interval could observe nothing.
    """
    cfg = (cfg or ESKFConfig()).resolved(dt)
    n = gyro.shape[0]
    if stationary is None:
        stationary = stationary_detector(gyro, accel, cfg)

    f = ESKF(cfg, q0, v0, p0, bg0=bg0, ba0=ba0)
    if P0 is not None:
        P0 = np.asarray(P0, dtype=np.float64)
        if P0.shape != (N_STATE, N_STATE):
            raise ValueError(f"P0 must be {N_STATE}x{N_STATE}, got {P0.shape}")
        f.P = P0.copy()

    pos = np.empty((n, 3))
    vel = np.empty((n, 3))
    quat = np.empty((n, 4))
    bg = np.empty((n, 3))
    ba = np.empty((n, 3))
    sig_bg = np.empty((n, 3))
    sig_ba = np.empty((n, 3))

    for k in range(n):
        if k > 0:
            f.predict(gyro[k], accel[k], dt)

        if stationary[k]:
            if enable_zupt:
                f.update_zupt()
            if enable_zaru:
                f.update_zaru(gyro[k])
            if enable_gravity:
                f.update_gravity(accel[k])

        pos[k] = f.p
        vel[k] = f.v
        quat[k] = f.q
        bg[k] = f.bg
        ba[k] = f.ba
        diag = np.diag(f.P)
        sig_ba[k] = np.sqrt(np.maximum(diag[I_BA], 0.0))
        sig_bg[k] = np.sqrt(np.maximum(diag[I_BG], 0.0))

    return ESKFResult(
        t=np.arange(n) * dt,
        position=pos,
        velocity=vel,
        quat=quat,
        bias_accel=ba,
        bias_gyro=bg,
        sigma_bias_gyro=sig_bg,
        sigma_bias_accel=sig_ba,
        stationary=stationary,
        n_updates=dict(f.counts),
        P_final=f.P.copy(),
    )

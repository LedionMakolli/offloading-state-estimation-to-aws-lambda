"""Stochastic and deterministic error model for a virtual InvenSense MPU-6050.

The point of this module is that every parameter is traceable to either the
manufacturer datasheet or published Allan-variance characterisations of the
part. Nothing here is tuned by hand to make a plot look right; the model is
specified from the datasheet and then *verified* against it by
``e1_allan_validation.py``.

Datasheet reference
-------------------
InvenSense, "MPU-6000/MPU-6050 Product Specification", Rev 3.4 (2013),
Sections 6.1 (gyroscope) and 6.2 (accelerometer):

  Gyroscope
    Rate noise spectral density        0.005 deg/s/sqrt(Hz)
    Zero-rate output (ZRO) tolerance   +/- 20 deg/s        (initial, 25 C)
    ZRO variation over temperature     +/- 0.02 deg/s/degC (-40 to +85 C)
    Sensitivity scale factor tolerance +/- 3 %
    Cross-axis sensitivity             +/- 2 %
    ADC                                16-bit

  Accelerometer
    Noise power spectral density       400 ug/sqrt(Hz)
    Zero-g output (X, Y)               +/- 50 mg           (initial, 25 C)
    Zero-g output (Z)                  +/- 80 mg           (initial, 25 C)
    Zero-g variation over temperature  +/- 1.5 mg/degC (X,Y), +/- 2.0 mg/degC (Z)
    Sensitivity scale factor tolerance +/- 3 %
    Cross-axis sensitivity             +/- 2 %
    ADC                                16-bit

Bias instability is *not* quoted on the datasheet -- it only appears in an
Allan-variance analysis. The defaults below are mid-range values from the
published literature on the MPU-6050 and MUST be cited as such in the thesis,
not attributed to InvenSense. They are deliberately configurable because
reported values vary by roughly a factor of three across units and studies.

Noise synthesis
---------------
White noise
    A one-sided noise density N (units/sqrt(Hz)) is realised by i.i.d. samples
    of standard deviation sigma_d = N / sqrt(dt). The Allan deviation is then
    sigma(tau) = N / sqrt(tau), so sigma(1 s) == N exactly.

Bias instability (flicker)
    Fractionally integrating white noise (Kasdin & Walter, 1992) with
    alpha = 1 gives a filter magnitude |H|^2 = |2 sin(pi f dt)|^-1, hence an
    output one-sided PSD S(f) ~= sigma_w^2 / (pi f), i.e. A = sigma_w^2 / pi.
    Flicker Allan variance is sigma^2(tau) = 2 ln2 * A, and the IMU convention
    is sigma^2(tau) = (2 ln2 / pi) B^2. Equating gives A = B^2 / pi, therefore

        sigma_w = B

    So pink-filtering white noise of standard deviation B produces a bias
    instability of exactly B, with an Allan deviation floor at 0.6643 * B.
    This identity is checked numerically by the E1 gate.

Rate random walk
    b_k = b_{k-1} + K sqrt(dt) n_k with n ~ N(0,1) yields sigma^2(tau) = K^2 tau/3.

All internal computation is in SI units: rad/s for the gyroscope and m/s^2
for the accelerometer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------
DEG = np.pi / 180.0          # rad per degree
G0 = 9.80665                 # m/s^2 per g (standard gravity, CGPM 1901)
MICRO_G = 1e-6 * G0          # m/s^2 per ug
MILLI_G = 1e-3 * G0          # m/s^2 per mg

# --------------------------------------------------------------------------
# Datasheet-derived MPU-6050 constants, converted to SI
# --------------------------------------------------------------------------
GYRO_NOISE_DENSITY = 0.005 * DEG        # rad/s/sqrt(Hz)   datasheet 6.1
GYRO_ZRO_TOLERANCE = 20.0 * DEG         # rad/s   1-sigma-equivalent, datasheet 6.1
GYRO_ZRO_TEMPCO = 0.02 * DEG            # rad/s/degC       datasheet 6.1

ACCEL_NOISE_DENSITY = 400.0 * MICRO_G   # m/s^2/sqrt(Hz)   datasheet 6.2
ACCEL_ZG_TOLERANCE_XY = 50.0 * MILLI_G  # m/s^2            datasheet 6.2
ACCEL_ZG_TOLERANCE_Z = 80.0 * MILLI_G   # m/s^2            datasheet 6.2
ACCEL_ZG_TEMPCO_XY = 1.5 * MILLI_G      # m/s^2/degC       datasheet 6.2
ACCEL_ZG_TEMPCO_Z = 2.0 * MILLI_G       # m/s^2/degC       datasheet 6.2

SCALE_FACTOR_TOLERANCE = 0.03           # +/- 3 %, both sensors
CROSS_AXIS_SENSITIVITY = 0.02           # +/- 2 %, both sensors

# Literature values, NOT datasheet. Cite the source used in the thesis.
GYRO_BIAS_INSTABILITY = 0.01 * DEG      # rad/s   (~36 deg/hr)
ACCEL_BIAS_INSTABILITY = 0.04 * MILLI_G  # m/s^2  (~40 ug)
GYRO_RATE_RANDOM_WALK = 0.0002 * DEG    # rad/s/sqrt(s)
ACCEL_RATE_RANDOM_WALK = 0.0001 * MILLI_G  # m/s^2/sqrt(s)

# Selectable full-scale ranges -> (range in SI, LSB per SI unit). Datasheet 6.1/6.2.
GYRO_RANGES = {
    0: (250.0 * DEG, 131.0 / DEG),
    1: (500.0 * DEG, 65.5 / DEG),
    2: (1000.0 * DEG, 32.8 / DEG),
    3: (2000.0 * DEG, 16.4 / DEG),
}
ACCEL_RANGES = {
    0: (2.0 * G0, 16384.0 / G0),
    1: (4.0 * G0, 8192.0 / G0),
    2: (8.0 * G0, 4096.0 / G0),
    3: (16.0 * G0, 2048.0 / G0),
}


@dataclass(frozen=True)
class TriadSpec:
    """Error parameters for one three-axis sensor."""

    noise_density: float          # N, units/sqrt(Hz)
    bias_instability: float       # B, units
    rate_random_walk: float       # K, units/sqrt(s)
    turn_on_bias: np.ndarray      # (3,) 1-sigma of the constant per-power-cycle bias
    bias_tempco: np.ndarray       # (3,) units/degC
    scale_tolerance: float = SCALE_FACTOR_TOLERANCE
    cross_axis: float = CROSS_AXIS_SENSITIVITY
    fs_sel: int = 0               # index into GYRO_RANGES / ACCEL_RANGES

    def range_and_lsb(self, table: dict[int, tuple[float, float]]) -> tuple[float, float]:
        if self.fs_sel not in table:
            raise ValueError(f"invalid full-scale select {self.fs_sel}")
        return table[self.fs_sel]


def gyro_spec(fs_sel: int = 0) -> TriadSpec:
    """Default MPU-6050 gyroscope specification."""
    return TriadSpec(
        noise_density=GYRO_NOISE_DENSITY,
        bias_instability=GYRO_BIAS_INSTABILITY,
        rate_random_walk=GYRO_RATE_RANDOM_WALK,
        turn_on_bias=np.full(3, GYRO_ZRO_TOLERANCE),
        bias_tempco=np.full(3, GYRO_ZRO_TEMPCO),
        fs_sel=fs_sel,
    )


def accel_spec(fs_sel: int = 0) -> TriadSpec:
    """Default MPU-6050 accelerometer specification."""
    return TriadSpec(
        noise_density=ACCEL_NOISE_DENSITY,
        bias_instability=ACCEL_BIAS_INSTABILITY,
        rate_random_walk=ACCEL_RATE_RANDOM_WALK,
        turn_on_bias=np.array(
            [ACCEL_ZG_TOLERANCE_XY, ACCEL_ZG_TOLERANCE_XY, ACCEL_ZG_TOLERANCE_Z]
        ),
        bias_tempco=np.array(
            [ACCEL_ZG_TEMPCO_XY, ACCEL_ZG_TEMPCO_XY, ACCEL_ZG_TEMPCO_Z]
        ),
        fs_sel=fs_sel,
    )


# --------------------------------------------------------------------------
# Noise process synthesis
# --------------------------------------------------------------------------
def _pink_fir(n_taps: int, alpha: float = 1.0) -> np.ndarray:
    """Kasdin & Walter fractional-integration FIR for 1/f^alpha noise.

    h[0] = 1, h[k] = h[k-1] * (k - 1 + alpha/2) / k
    """
    h = np.empty(n_taps, dtype=np.float64)
    h[0] = 1.0
    k = np.arange(1, n_taps, dtype=np.float64)
    h[1:] = np.cumprod((k - 1.0 + alpha / 2.0) / k)
    return h


def flicker_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Flicker (1/f) noise whose Allan deviation floor is 0.6643 * sigma.

    Per the module docstring, driving the pink filter with white noise of
    standard deviation ``sigma`` gives a bias instability B == sigma.
    """
    if sigma <= 0.0:
        return np.zeros(n)
    # The FIR must be as long as the record for the low-frequency content
    # that produces the flat Allan floor to be represented at all.
    h = _pink_fir(n)
    w = rng.standard_normal(n) * sigma
    return fftconvolve(w, h, mode="full")[:n]


def white_noise(
    n: int, density: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """White noise with one-sided density ``density`` (units/sqrt(Hz))."""
    return rng.standard_normal(n) * (density / np.sqrt(dt))


def rate_random_walk(
    n: int, k: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Random walk in the rate signal, giving sigma^2(tau) = K^2 tau / 3."""
    if k <= 0.0:
        return np.zeros(n)
    return np.cumsum(rng.standard_normal(n) * (k * np.sqrt(dt)))


def misalignment_matrix(
    spec: TriadSpec, rng: np.random.Generator, enable: bool = True
) -> np.ndarray:
    """Combined scale-factor and cross-axis error matrix M.

    Measured = M @ true + bias + noise. Diagonal terms are scale-factor
    error, off-diagonal terms are axis non-orthogonality / cross-axis
    sensitivity. Both are drawn once per simulated device.
    """
    m = np.eye(3)
    if not enable:
        return m
    m[np.diag_indices(3)] += rng.uniform(-spec.scale_tolerance, spec.scale_tolerance, 3)
    off = ~np.eye(3, dtype=bool)
    m[off] += rng.uniform(-spec.cross_axis, spec.cross_axis, 6)
    return m


def quantise(x: np.ndarray, full_scale: float, lsb_per_unit: float) -> np.ndarray:
    """Clip to the selected full-scale range and quantise to the 16-bit ADC."""
    counts = np.clip(np.round(x * lsb_per_unit), -32768, 32767)
    return counts / lsb_per_unit


@dataclass
class TriadErrorRealisation:
    """The per-device constants drawn for one simulated sensor.

    Kept so that a simulated run is fully reproducible and so the true
    biases can be compared against what the cloud estimator recovers.
    """

    turn_on_bias: np.ndarray
    misalignment: np.ndarray
    seed: int
    # (N, 3) slowly-varying bias actually injected, sample by sample.
    bias_series: np.ndarray | None = None


def synthesise_triad(
    true_signal: np.ndarray,
    spec: TriadSpec,
    dt: float,
    rng: np.random.Generator,
    range_table: dict[int, tuple[float, float]],
    temperature: np.ndarray | None = None,
    apply_deterministic: bool = True,
    apply_quantisation: bool = True,
) -> tuple[np.ndarray, TriadErrorRealisation]:
    """Corrupt a clean three-axis signal with the full MPU-6050 error model.

    Args:
        true_signal: (N, 3) noise-free signal in SI units.
        spec: sensor specification.
        dt: sample period, seconds.
        rng: seeded generator.
        range_table: GYRO_RANGES or ACCEL_RANGES.
        temperature: optional (N,) temperature in degC, referenced to 25 C.
        apply_deterministic: include turn-on bias, scale factor, cross-axis.
        apply_quantisation: include ADC clipping and quantisation.

    Returns:
        (N, 3) corrupted measurements, and the realised per-device constants.
    """
    true_signal = np.atleast_2d(np.asarray(true_signal, dtype=np.float64))
    if true_signal.shape[1] != 3:
        raise ValueError(f"expected (N, 3) signal, got {true_signal.shape}")
    n = true_signal.shape[0]

    misalign = misalignment_matrix(spec, rng, enable=apply_deterministic)
    out = true_signal @ misalign.T

    turn_on = (
        rng.standard_normal(3) * spec.turn_on_bias
        if apply_deterministic
        else np.zeros(3)
    )
    # The slowly-varying error is accumulated separately from the white noise
    # so that a filter's bias estimate can be scored against the bias that was
    # actually injected. Without this the only available reference is the
    # constant turn-on term, which ignores the flicker component -- and flicker
    # is precisely what dominates at long horizons, so scoring against the
    # constant alone makes a correctly-performing filter look badly biased.
    bias = np.tile(turn_on, (n, 1))
    for axis in range(3):
        bias[:, axis] += flicker_noise(n, spec.bias_instability, rng)
        bias[:, axis] += rate_random_walk(n, spec.rate_random_walk, dt, rng)

    if temperature is not None:
        delta_t = np.asarray(temperature, dtype=np.float64) - 25.0
        bias += delta_t[:, None] * spec.bias_tempco[None, :]

    white = np.empty((n, 3))
    for axis in range(3):
        white[:, axis] = white_noise(n, spec.noise_density, dt, rng)

    out = out + bias + white

    if apply_quantisation:
        full_scale, lsb = spec.range_and_lsb(range_table)
        out = quantise(out, full_scale, lsb)

    realisation = TriadErrorRealisation(
        turn_on_bias=turn_on,
        misalignment=misalign,
        seed=-1,
        bias_series=bias,
    )
    return out, realisation


def synthesise_static(
    duration_s: float,
    fs: float,
    spec: TriadSpec,
    range_table: dict[int, tuple[float, float]],
    rng: np.random.Generator,
    apply_deterministic: bool = False,
    apply_quantisation: bool = False,
) -> np.ndarray:
    """Stationary sensor output: the classic Allan-variance test condition.

    Deterministic errors and quantisation default to *off* so that the
    recovered N and B can be compared directly against the specification
    without confounding terms. Turning them on is how their effect on the
    Allan curve is studied.
    """
    n = int(round(duration_s * fs))
    dt = 1.0 / fs
    true_signal = np.zeros((n, 3))
    out, _ = synthesise_triad(
        true_signal,
        spec,
        dt,
        rng,
        range_table,
        apply_deterministic=apply_deterministic,
        apply_quantisation=apply_quantisation,
    )
    return out

"""AWS Lambda handler: offloaded inertial bias estimation.

This is the cloud half of the integration. The device runs Tier 0 (attitude and
strapdown) under a hard deadline; this function runs the expensive Tier 1/2 work
-- the 15-state ESKF covariance propagation and the ZUPT/ZARU/gravity updates --
over a window of buffered IMU samples, and returns a bias correction.

The one field that matters most in the response is ``valid_at_us``. The
correction describes the state at the *end of the submitted window*, not at the
moment it arrives back on the device. By the time the device receives it, that
timestamp is already in the past by the full round trip -- tens of milliseconds
when warm, hundreds to thousands on a cold start. Applying such a correction to
the current state is exactly the out-of-sequence measurement problem, and
stamping it explicitly is what lets the device rewind and re-propagate instead.

Wire format
-----------
IMU samples travel as base64-encoded little-endian int16, which is the raw ADC
format the MPU-6050 produces and the Pico would forward unmodified. A 1 s window
at 200 Hz is 200 * 6 * 2 = 2400 bytes per sensor, ~3.2 kB base64 -- comfortably
inside the AWS IoT Core 128 kB message limit, and far smaller than sending
floats as JSON numbers.

Dependencies
------------
numpy only. eskf.py deliberately avoids scipy so this package stays small
enough for a plain zip deployment with the AWS-provided numpy layer; the LSB
scale constants are duplicated here rather than imported from noise_model.py,
which pulls in scipy.signal.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import numpy as np

import eskf
import rotations as rot

# Cold-start detection. Module scope runs once per execution environment, so
# this is False on the first invocation into a fresh container and True
# afterwards. E4/E5 use it to separate init cost from steady-state latency.
_WARM = False
_INIT_T0 = time.time()

DEG = np.pi / 180.0
G0 = 9.80665

# MPU-6050 full-scale selects -> LSB per SI unit (datasheet 6.1 / 6.2).
GYRO_LSB = {0: 131.0 / DEG, 1: 65.5 / DEG, 2: 32.8 / DEG, 3: 16.4 / DEG}
ACCEL_LSB = {0: 16384.0 / G0, 1: 8192.0 / G0, 2: 4096.0 / G0, 3: 2048.0 / G0}

# A window of n samples costs n*6*2 bytes raw, x4/3 for base64, for two
# sensors: ~32 n bytes on the wire. AWS IoT Core caps a message at 128 kB, so
# ~4000 samples is the hard ceiling. Operationally the window should be far
# shorter than that -- the point of offloading is a timely correction, and
# buffering 20 s before sending one defeats it.
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "4000"))
IOT_CORE_LIMIT_BYTES = 128 * 1024

# Upper-triangle packing of the 15x15 covariance: 120 unique elements rather
# than 225, and reconstruction is symmetric by construction.
_TRI_I, _TRI_J = np.triu_indices(eskf.N_STATE)


def pack_covariance(P: np.ndarray) -> str:
    """15x15 symmetric -> base64 float32 upper triangle (480 B -> 640 chars)."""
    return base64.b64encode(
        np.asarray(P, dtype=np.float64)[_TRI_I, _TRI_J].astype("<f4").tobytes()
    ).decode("ascii")


def unpack_covariance(b64: str) -> np.ndarray:
    """Inverse of pack_covariance. Rebuilds a symmetric matrix."""
    raw = base64.b64decode(b64, validate=True)
    expect = _TRI_I.size * 4
    if len(raw) != expect:
        raise BadRequest(f"P_packed must be {expect} bytes, got {len(raw)}")
    vals = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    P = np.zeros((eskf.N_STATE, eskf.N_STATE))
    P[_TRI_I, _TRI_J] = vals
    P[_TRI_J, _TRI_I] = vals
    return P


class BadRequest(Exception):
    """Malformed input. Returned as a 400-style body rather than a crash."""


# --------------------------------------------------------------------------
# Publishing the correction back to the device
#
# When this function is invoked by an AWS IoT Rule, nothing is waiting for its
# return value -- the rule fires the function and moves on. So the correction
# has to be published back onto an MQTT topic explicitly, or the device never
# receives it. That failure mode is quiet and misleading: the estimator runs
# perfectly, CloudWatch shows a clean invocation, and the device simply waits
# forever.
#
# The boto3 client is created once per execution environment rather than per
# invocation. Client construction costs tens of milliseconds, which would
# otherwise be paid on every window.
#
# WHEN that construction happens turns out to matter more than it looks. E4c
# measured the return leg at 9 ms on a warm environment but 457 ms on a cold
# one -- and none of that 448 ms difference appears in Init Duration, because
# it is spent inside the first billed invocation. CloudWatch therefore reported
# a 501 ms cold-start penalty where the device actually waited 991 ms.
#
# IOT_CLIENT_MODE selects where the cost is paid, so the three variants can be
# compared rather than assumed:
#
#   lazy     build the client on first use. The original behaviour, kept so the
#            measurement that motivated the change stays reproducible.
#   eager    build the client at import. Cheap, but only partly effective:
#            botocore opens the TLS connection lazily, on the first request.
#   prewarm  build the client AND issue one throwaway publish at import, which
#            is what actually establishes the connection.
#
# The warm-up publish goes to a topic no rule matches, so it cannot re-enter
# this function; it sits under imu/*/correction only because that is what the
# execution role is permitted to publish to.
# --------------------------------------------------------------------------
RESPONSE_TOPIC = os.environ.get("RESPONSE_TOPIC", "imu/{device_id}/correction")
PUBLISH_ENABLED = os.environ.get("PUBLISH_RESPONSE", "1") not in ("0", "false", "False")
IOT_CLIENT_MODE = os.environ.get("IOT_CLIENT_MODE", "prewarm").strip().lower()
WARMUP_TOPIC = os.environ.get("WARMUP_TOPIC", "imu/_warmup/correction")

_iot_client = None
_init_publish_ms: float | None = None   # what the warm-up cost, if it ran


def _get_iot_client():
    global _iot_client
    if _iot_client is None:
        import boto3  # available in the Lambda runtime; not needed for local tests
        _iot_client = boto3.client("iot-data")
    return _iot_client


def _warm_iot_client() -> None:
    """Move the return leg's first-call cost out of the billed invocation.

    Never raises. A warm-up is an optimisation, and an optimisation that can
    prevent the function from initialising is worse than the cost it saves --
    boto3 is absent locally, for one, and credentials may not be resolvable
    during init in some configurations.
    """
    global _init_publish_ms
    if IOT_CLIENT_MODE == "lazy" or not PUBLISH_ENABLED:
        return
    t0 = time.perf_counter()
    try:
        client = _get_iot_client()
        if IOT_CLIENT_MODE == "prewarm":
            client.publish(topic=WARMUP_TOPIC, qos=0, payload=b'{"warmup":1}')
        _init_publish_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    except Exception:  # noqa: BLE001
        _init_publish_ms = -1.0


# Run at import, which is what puts the cost inside the init phase.
_warm_iot_client()


def publish_correction(body: dict[str, Any]) -> dict[str, Any]:
    """Publish the correction to the device's topic. Never raises.

    A publish failure must not turn a successful estimate into a crash: the
    result is still worth logging, and an exception here would make the rule
    retry a computation that already succeeded.
    """
    device_id = body.get("device_id")
    if not PUBLISH_ENABLED or not device_id:
        return {"published": False, "reason": "disabled or no device_id"}

    topic = RESPONSE_TOPIC.format(device_id=device_id)
    try:
        t0 = time.perf_counter()
        _get_iot_client().publish(
            topic=topic, qos=0, payload=json.dumps(body).encode("utf-8")
        )
        return {"published": True, "topic": topic,
                "publish_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
    except Exception as exc:  # noqa: BLE001
        return {"published": False, "topic": topic,
                "error": f"{type(exc).__name__}: {exc}"}


def _decode_triad(b64: str, n: int, lsb: float) -> np.ndarray:
    """base64 little-endian int16 -> (n, 3) float array in SI units."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise BadRequest(f"base64 decode failed: {exc}") from exc
    expect = n * 3 * 2
    if len(raw) != expect:
        raise BadRequest(f"expected {expect} bytes for {n} samples, got {len(raw)}")
    counts = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return counts.reshape(n, 3) / lsb


def _get_state(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = payload.get("state") or {}
    q = np.asarray(st.get("q", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    v = np.asarray(st.get("v", [0.0, 0.0, 0.0]), dtype=np.float64)
    p = np.asarray(st.get("p", [0.0, 0.0, 0.0]), dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,) or p.shape != (3,):
        raise BadRequest("state.q must be length 4; state.v and state.p length 3")
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-6:
        raise BadRequest("state.q is not a valid quaternion")
    return rot.quat_normalize(q), v, p


def estimate(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the ESKF over one submitted window. Pure function, unit-testable."""
    n = int(payload.get("n", 0))
    if n <= 0:
        raise BadRequest("n must be a positive sample count")
    if n > MAX_SAMPLES:
        raise BadRequest(f"n={n} exceeds MAX_SAMPLES={MAX_SAMPLES}")

    fs = float(payload.get("fs", 200.0))
    if not (1.0 <= fs <= 10000.0):
        raise BadRequest(f"implausible sample rate {fs}")
    dt = 1.0 / fs

    gyro_sel = int(payload.get("gyro_fs_sel", 0))
    accel_sel = int(payload.get("accel_fs_sel", 0))
    if gyro_sel not in GYRO_LSB or accel_sel not in ACCEL_LSB:
        raise BadRequest("invalid full-scale select")

    if "gyro" not in payload or "accel" not in payload:
        raise BadRequest("payload must carry 'gyro' and 'accel'")

    gyro = _decode_triad(payload["gyro"], n, GYRO_LSB[gyro_sel])
    accel = _decode_triad(payload["accel"], n, ACCEL_LSB[accel_sel])
    q0, v0, p0 = _get_state(payload)

    # Carried-in filter state. Lambda is stateless by design, so the covariance
    # and the bias estimates have to travel with each window; otherwise every
    # invocation restarts from a diffuse prior and a short window containing no
    # stationary interval observes nothing at all. Packed P is 640 base64
    # characters, negligible beside the IMU payload.
    st = payload.get("state") or {}
    bg0 = np.asarray(st["bg"], dtype=np.float64) if "bg" in st else None
    ba0 = np.asarray(st["ba"], dtype=np.float64) if "ba" in st else None
    if bg0 is not None and bg0.shape != (3,):
        raise BadRequest("state.bg must be length 3")
    if ba0 is not None and ba0.shape != (3,):
        raise BadRequest("state.ba must be length 3")
    P0 = unpack_covariance(payload["P_packed"]) if payload.get("P_packed") else None

    cfg = eskf.ESKFConfig()
    # Keep the detector consistent with the submitted rate: its statistic
    # scales as 1/dt, and its window should be a fixed span of wall time.
    cfg.detector_window = max(4, int(round(0.2 * fs)))
    cfg.detector_threshold = 400.0 * fs

    t0 = time.perf_counter()
    res = eskf.run_eskf(gyro, accel, dt, q0=q0, v0=v0, p0=p0, cfg=cfg,
                        bg0=bg0, ba0=ba0, P0=P0)
    compute_ms = (time.perf_counter() - t0) * 1000.0

    # The window starts at t_start_us and covers n samples, so the correction
    # describes the state at the LAST sample, not at the moment of reply.
    t_start_us = int(payload.get("t_start_us", 0))
    valid_at_us = t_start_us + int(round((n - 1) * dt * 1e6))

    return {
        "device_id": payload.get("device_id"),
        "seq": payload.get("seq"),
        "valid_at_us": valid_at_us,
        "bg": res.bias_gyro[-1].tolist(),
        "ba": res.bias_accel[-1].tolist(),
        "sigma_bg": res.sigma_bias_gyro[-1].tolist(),
        "sigma_ba": res.sigma_bias_accel[-1].tolist(),
        "q": res.quat[-1].tolist(),
        "v": res.velocity[-1].tolist(),
        "p": res.position[-1].tolist(),
        "P_packed": pack_covariance(res.P_final),
        "n_stationary": int(np.count_nonzero(res.stationary)),
        "n_updates": res.n_updates,
        "compute_ms": round(compute_ms, 3),
        # Echoed straight back so the device can compute round-trip time
        # against its own clock. One-way latency is NOT derivable from this:
        # the Pico has no disciplined clock, so RTT is the honest metric.
        "echo_t_sent_us": payload.get("t_sent_us"),
    }


def lambda_handler(event, context):
    """Entry point. Accepts an IoT Rule payload, a function-URL body, or a
    direct invoke, since E4/E5 compare those transports against each other."""
    global _WARM
    was_cold = not _WARM
    _WARM = True

    wall_t0 = time.perf_counter()

    payload = event
    if isinstance(event, dict) and "body" in event and isinstance(event["body"], str):
        try:
            payload = json.loads(event["body"])
        except json.JSONDecodeError as exc:
            return _reply(400, {"error": f"body is not JSON: {exc}"}, was_cold, wall_t0)
    if not isinstance(payload, dict):
        return _reply(400, {"error": "payload must be a JSON object"}, was_cold, wall_t0)

    try:
        result = estimate(payload)
    except BadRequest as exc:
        return _reply(400, {"error": str(exc)}, was_cold, wall_t0)
    except Exception as exc:  # noqa: BLE001 - never fail silently in the field
        return _reply(500, {"error": f"{type(exc).__name__}: {exc}"}, was_cold, wall_t0)

    # Stamp the timing fields before publishing, so the device receives exactly
    # the same body the caller sees rather than a version missing them.
    result["cold_start"] = was_cold
    result["handler_ms"] = round((time.perf_counter() - wall_t0) * 1000.0, 3)
    result["publish"] = publish_correction(result)

    # When the IoT rule is the caller there is nobody to read the return value,
    # so the outcome only exists if it is logged. Keep it to one compact line:
    # CloudWatch is billed by ingested volume, and the estimate itself is
    # already on its way to the device.
    print("seq=%s n=%s cold=%s compute_ms=%s handler_ms=%s mode=%s "
          "init_pub_ms=%s publish=%s"
          % (result.get("seq"), payload.get("n"), was_cold,
             result.get("compute_ms"), result["handler_ms"],
             IOT_CLIENT_MODE, _init_publish_ms,
             json.dumps(result["publish"])))

    return _reply(200, result, was_cold, wall_t0)


def _reply(status: int, body: dict[str, Any], was_cold: bool, wall_t0: float):
    body["cold_start"] = was_cold
    body["handler_ms"] = round((time.perf_counter() - wall_t0) * 1000.0, 3)
    if was_cold:
        body["init_age_s"] = round(time.time() - _INIT_T0, 3)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

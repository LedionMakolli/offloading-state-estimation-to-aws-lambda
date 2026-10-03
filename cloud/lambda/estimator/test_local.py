"""End-to-end contract test for the Lambda handler, with no AWS involved.

Builds payloads exactly as the Pico would -- raw int16 ADC counts, base64, JSON
-- and drives the handler through a *sequence* of windows, carrying the filter
state and covariance across invocations the way a stateless FaaS forces you to.

Running this before deploying means that when the AWS side is wired up, any
failure is an AWS failure (IAM, layers, message size, timeout) and not an
algorithm failure. Those are very different debugging problems and it is worth
being able to tell them apart.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vimu-gen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_model as nm          # noqa: E402
import trajectory as tj           # noqa: E402
from handler import (             # noqa: E402
    ACCEL_LSB, GYRO_LSB, IOT_CORE_LIMIT_BYTES, lambda_handler,
)

FS = 200.0
DURATION = 60.0
WINDOW_S = 2.0                    # operational window; short so corrections are timely
GYRO_SEL, ACCEL_SEL = 0, 0


def encode(sig: np.ndarray, lsb: float) -> str:
    counts = np.clip(np.round(sig * lsb), -32768, 32767).astype("<i2")
    return base64.b64encode(counts.tobytes()).decode("ascii")


def main() -> int:
    print("Lambda handler contract test (no AWS)")
    print("=" * 70)

    traj = tj.MissionTrajectory().generate(DURATION, FS)
    rng = np.random.default_rng(7)
    gyro, rg = nm.synthesise_triad(traj.omega_body, nm.gyro_spec(), traj.dt, rng,
                                   nm.GYRO_RANGES, apply_deterministic=False)
    accel, ra = nm.synthesise_triad(traj.accel_body, nm.accel_spec(), traj.dt, rng,
                                    nm.ACCEL_RANGES, apply_deterministic=False)
    const_bg = np.array([0.05, -0.03, 0.02]) * nm.DEG
    const_ba = np.array([2.0, -1.0, 3.0]) * nm.MILLI_G
    gyro, accel = gyro + const_bg, accel + const_ba
    true_bg, true_ba = rg.bias_series + const_bg, ra.bias_series + const_ba

    w = int(WINDOW_S * FS)
    n_win = len(traj) // w
    print(f"  mission     : {DURATION:g} s @ {FS:g} Hz")
    print(f"  window      : {WINDOW_S:g} s = {w} samples, {n_win} windows")

    # Seed the chain with the true initial nominal state and no bias estimate.
    state = {
        "q": traj.quat[0].tolist(),
        "v": traj.velocity[0].tolist(),
        "p": traj.position[0].tolist(),
    }
    p_packed = None
    sizes, computes, errs_bg = [], [], []
    checks = []

    t_wall = time.perf_counter()
    for k in range(n_win):
        s0, s1 = k * w, (k + 1) * w
        payload = {
            "device_id": "pico-w-01",
            "seq": k,
            "t_start_us": int(s0 / FS * 1e6),
            "t_sent_us": int(s1 / FS * 1e6),
            "fs": FS,
            "n": w,
            "gyro_fs_sel": GYRO_SEL,
            "accel_fs_sel": ACCEL_SEL,
            "gyro": encode(gyro[s0:s1], GYRO_LSB[GYRO_SEL]),
            "accel": encode(accel[s0:s1], ACCEL_LSB[ACCEL_SEL]),
            "state": state,
        }
        if p_packed:
            payload["P_packed"] = p_packed

        wire = json.dumps(payload)
        sizes.append(len(wire))

        resp = lambda_handler(payload, None)
        body = json.loads(resp["body"])
        if resp["statusCode"] != 200:
            print(f"  window {k} FAILED: {body.get('error')}")
            return 1

        computes.append(body["compute_ms"])
        # Carry everything forward -- this is the stateless-FaaS contract.
        state = {"q": body["q"], "v": body["v"], "p": body["p"],
                 "bg": body["bg"], "ba": body["ba"]}
        p_packed = body["P_packed"]
        errs_bg.append(np.abs(np.array(body["bg"]) - true_bg[s1 - 1]).max() / nm.DEG)

        if k == 0:
            expected_valid = payload["t_start_us"] + int(round((w - 1) / FS * 1e6))
            checks.append(("valid_at_us marks window end",
                           body["valid_at_us"] == expected_valid))
            checks.append(("t_sent echoed for RTT",
                           body["echo_t_sent_us"] == payload["t_sent_us"]))
            checks.append(("first call reports cold_start", body["cold_start"] is True))
            checks.append(("P_packed is 640 chars", len(body["P_packed"]) == 640))
        if k == 1:
            checks.append(("second call reports warm", body["cold_start"] is False))

    wall = (time.perf_counter() - t_wall) * 1000.0

    max_wire = max(sizes)
    print(f"  wire/window : {max_wire / 1024:.1f} kB max "
          f"(IoT Core cap {IOT_CORE_LIMIT_BYTES // 1024} kB)")
    print(f"  P transport : 640 chars per window, {640 * 100 / max_wire:.1f}% of payload")
    print(f"  compute     : median {np.median(computes):.1f} ms, "
          f"max {max(computes):.1f} ms over {n_win} windows")
    print(f"  total wall  : {wall:.0f} ms in-process")

    final_bg = np.array(state["bg"])
    final_ba = np.array(state["ba"])
    e_bg = (final_bg - true_bg[-1]) / nm.DEG
    e_ba = (final_ba - true_ba[-1]) / nm.MILLI_G

    print(f"\n  {'':<12}{'estimate':>26}{'error':>26}")
    print(f"  {'gyro deg/s':<12}{str(np.round(final_bg / nm.DEG, 4)):>26}"
          f"{str(np.round(e_bg, 4)):>26}")
    print(f"  {'accel mg':<12}{str(np.round(final_ba / nm.MILLI_G, 3)):>26}"
          f"{str(np.round(e_ba, 3)):>26}")
    print(f"  gyro bias |err| converged {errs_bg[0]:.4f} -> {errs_bg[-1]:.4f} deg/s")

    checks += [
        ("window fits IoT Core limit", max_wire < IOT_CORE_LIMIT_BYTES),
        ("gyro bias within 0.05 deg/s", bool(np.all(np.abs(e_bg) < 0.05))),
        ("accel bias within 3 mg", bool(np.all(np.abs(e_ba) < 3.0))),
        ("bias estimate improved across windows", errs_bg[-1] < errs_bg[0]),
    ]

    # A window with no stationary interval must still work, because the prior
    # arrives with the request. This is the case that fails outright if the
    # covariance is not carried across invocations.
    print("\n  carried-state check")
    mid = int(15 * FS)
    lone = {
        "device_id": "x", "seq": 99, "t_start_us": 0, "fs": FS, "n": w,
        "gyro_fs_sel": 0, "accel_fs_sel": 0,
        "gyro": encode(gyro[mid:mid + w], GYRO_LSB[0]),
        "accel": encode(accel[mid:mid + w], ACCEL_LSB[0]),
        "state": {"q": traj.quat[mid].tolist(), "v": traj.velocity[mid].tolist(),
                  "p": traj.position[mid].tolist()},
    }
    r_cold = json.loads(lambda_handler(lone, None)["body"])
    lone_warm = {**lone, "state": {**lone["state"], "bg": final_bg.tolist(),
                                   "ba": final_ba.tolist()}, "P_packed": p_packed}
    r_warm = json.loads(lambda_handler(lone_warm, None)["body"])
    e_cold = np.abs(np.array(r_cold["bg"]) - true_bg[mid + w - 1]).max() / nm.DEG
    e_warm = np.abs(np.array(r_warm["bg"]) - true_bg[mid + w - 1]).max() / nm.DEG
    print(f"    moving-only window, {r_cold['n_stationary']} stationary samples")
    print(f"    without carried state: gyro bias err {e_cold:.4f} deg/s")
    print(f"    with    carried state: gyro bias err {e_warm:.4f} deg/s")
    checks.append(("carried state helps a window with no ZUPT", e_warm < e_cold))

    print("\n  malformed-input handling")
    base = {"device_id": "x", "seq": 1, "fs": FS, "n": w, "gyro_fs_sel": 0,
            "accel_fs_sel": 0, "gyro": encode(gyro[:w], GYRO_LSB[0]),
            "accel": encode(accel[:w], ACCEL_LSB[0]), "state": {}}
    for label, bad in [
        ("sample-count mismatch", {**base, "n": w + 1}),
        ("implausible fs", {**base, "fs": 0.0}),
        ("missing gyro", {k: v for k, v in base.items() if k != "gyro"}),
        ("bad quaternion", {**base, "state": {"q": [0, 0, 0, 0]}}),
        ("oversize window", {**base, "n": 99999}),
        ("corrupt P_packed", {**base, "P_packed": "AAAA"}),
    ]:
        r = lambda_handler(bad, None)
        ok = r["statusCode"] == 400
        msg = json.loads(r["body"]).get("error", "")[:40]
        print(f"    [{'ok' if ok else 'FAIL'}] {label:<24} -> {r['statusCode']} {msg}")
        checks.append((f"rejects {label}", ok))

    print("\n" + "=" * 70)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    allok = all(ok for _, ok in checks)
    print("=" * 70)
    print(f"RESULT: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

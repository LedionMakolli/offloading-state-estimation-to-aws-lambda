"""Invoke the DEPLOYED Lambda function with a realistic IMU window.

Run this after creating the function in the AWS console. It builds the same
payload the Pico will send, invokes the function through the AWS CLI, and checks
the response -- so a failure here is an AWS configuration problem, not an
algorithm problem, since test_local.py already proved the algorithm.

    python invoke_deployed.py [function-name]

Default function name: picoW_estimator
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vimu-gen"))

import noise_model as nm      # noqa: E402
import trajectory as tj       # noqa: E402

FUNCTION = sys.argv[1] if len(sys.argv) > 1 else "picoW_estimator"
AWS = os.environ.get("AWS_CLI", r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")

FS = 200.0
WINDOW_S = 2.0
GYRO_LSB = 131.0 / nm.DEG
ACCEL_LSB = 16384.0 / 9.80665


def encode(sig, lsb):
    counts = np.clip(np.round(sig * lsb), -32768, 32767).astype("<i2")
    return base64.b64encode(counts.tobytes()).decode("ascii")


def build_payload():
    traj = tj.MissionTrajectory().generate(WINDOW_S + 4.0, FS)
    rng = np.random.default_rng(11)
    gyro, _ = nm.synthesise_triad(traj.omega_body, nm.gyro_spec(), traj.dt, rng,
                                  nm.GYRO_RANGES, apply_deterministic=False)
    accel, _ = nm.synthesise_triad(traj.accel_body, nm.accel_spec(), traj.dt, rng,
                                   nm.ACCEL_RANGES, apply_deterministic=False)
    n = int(WINDOW_S * FS)
    return {
        "device_id": "picoW_thing",
        "seq": 1,
        "t_start_us": 0,
        "t_sent_us": int(WINDOW_S * 1e6),
        "fs": FS,
        "n": n,
        "gyro_fs_sel": 0,
        "accel_fs_sel": 0,
        "gyro": encode(gyro[:n], GYRO_LSB),
        "accel": encode(accel[:n], ACCEL_LSB),
        "state": {
            "q": traj.quat[0].tolist(),
            "v": traj.velocity[0].tolist(),
            "p": traj.position[0].tolist(),
        },
    }


def main() -> int:
    if not os.path.exists(AWS):
        print("AWS CLI not found at", AWS)
        print("set AWS_CLI environment variable to its path")
        return 1

    payload = build_payload()
    wire = json.dumps(payload)
    print("function :", FUNCTION)
    print("payload  : %d samples, %.1f kB" % (payload["n"], len(wire) / 1024))

    with tempfile.TemporaryDirectory() as td:
        req = Path(td) / "req.json"
        resp = Path(td) / "resp.json"
        req.write_text(wire, encoding="utf-8")

        # Two invocations: the first is likely a cold start, the second warm.
        for label in ("call 1 (likely cold)", "call 2 (warm)"):
            t0 = time.perf_counter()
            proc = subprocess.run(
                [AWS, "lambda", "invoke",
                 "--function-name", FUNCTION,
                 "--cli-binary-format", "raw-in-base64-out",
                 "--payload", "file://" + str(req),
                 str(resp)],
                capture_output=True, text=True)
            rtt_ms = (time.perf_counter() - t0) * 1000.0

            if proc.returncode != 0:
                print("\nINVOKE FAILED")
                print(proc.stderr.strip()[:600])
                return 2

            meta = json.loads(proc.stdout or "{}")
            if meta.get("FunctionError"):
                print("\nFUNCTION ERROR:", meta["FunctionError"])
                print(resp.read_text(encoding="utf-8")[:800])
                return 3

            outer = json.loads(resp.read_text(encoding="utf-8"))
            body = json.loads(outer["body"]) if "body" in outer else outer

            print("\n%s" % label)
            print("  status      : %s" % outer.get("statusCode"))
            print("  round trip  : %.0f ms  (includes AWS CLI startup)" % rtt_ms)
            print("  cold_start  : %s" % body.get("cold_start"))
            print("  compute_ms  : %s" % body.get("compute_ms"))
            print("  handler_ms  : %s" % body.get("handler_ms"))
            print("  valid_at_us : %s" % body.get("valid_at_us"))
            if body.get("bg"):
                bg = np.array(body["bg"]) / nm.DEG
                ba = np.array(body["ba"]) / nm.MILLI_G
                print("  gyro bias   : %s deg/s" % np.round(bg, 4))
                print("  accel bias  : %s mg" % np.round(ba, 3))
            print("  P_packed    : %d chars" % len(body.get("P_packed", "")))

    print("\nOK - the deployed function works end to end.")
    print("Note: the round-trip figure above is dominated by AWS CLI startup and")
    print("is NOT the device latency. The real E4b number must be measured on the")
    print("Pico, using the echoed t_sent_us field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

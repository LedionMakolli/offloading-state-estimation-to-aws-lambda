"""E2E test of the full AWS chain, with the PC standing in for the Pico.

This publishes a realistic IMU window to the same MQTT topic the firmware will
use. Nothing here talks to Lambda directly -- the point is to exercise the path
the device actually takes:

    MQTT topic imu/<id>/window
        -> IoT Rule
            -> Lambda picoW_estimator
                -> iot-data publish
                    -> MQTT topic imu/<id>/correction

The device is not subscribed yet (that is T7b), so the return leg is verified
from the CloudWatch log rather than by receiving the message. The log line the
handler emits carries the publish result, which is what proves the last hop.

    python publish_window.py [device-id]
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

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "picoW_thing"
AWS = os.environ.get("AWS_CLI", r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")
LOG_GROUP = "/aws/lambda/picoW_estimator"

FS = 200.0
WINDOW_S = 2.0
GYRO_LSB = 131.0 / nm.DEG
ACCEL_LSB = 16384.0 / 9.80665


def aws(*args, **kw):
    return subprocess.run([AWS, *args], capture_output=True, text=True, **kw)


def encode(sig, lsb):
    counts = np.clip(np.round(sig * lsb), -32768, 32767).astype("<i2")
    return base64.b64encode(counts.tobytes()).decode("ascii")


def build_payload(seq: int, t_start_us: int) -> dict:
    traj = tj.MissionTrajectory().generate(WINDOW_S + 4.0, FS)
    rng = np.random.default_rng(11 + seq)
    gyro, _ = nm.synthesise_triad(traj.omega_body, nm.gyro_spec(), traj.dt, rng,
                                  nm.GYRO_RANGES, apply_deterministic=False)
    accel, _ = nm.synthesise_triad(traj.accel_body, nm.accel_spec(), traj.dt, rng,
                                   nm.ACCEL_RANGES, apply_deterministic=False)
    n = int(WINDOW_S * FS)
    return {
        "device_id": DEVICE,
        "seq": seq,
        "t_start_us": t_start_us,
        "t_sent_us": t_start_us + int(WINDOW_S * 1e6),
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
        return 1

    ep = aws("iot", "describe-endpoint", "--endpoint-type", "iot:Data-ATS")
    if ep.returncode != 0:
        print("could not resolve the IoT data endpoint")
        print(ep.stderr.strip()[:400])
        return 2
    endpoint = "https://" + json.loads(ep.stdout)["endpointAddress"]
    print("endpoint : %s" % endpoint)

    topic = "imu/%s/window" % DEVICE
    reply = "imu/%s/correction" % DEVICE
    payload = build_payload(seq=1, t_start_us=int(time.time() * 1e6) % 10**12)
    wire = json.dumps(payload)
    print("topic    : %s  ->  reply on %s" % (topic, reply))
    print("payload  : %d samples, %.1f kB" % (payload["n"], len(wire) / 1024))

    # Mark the log window from just before the publish, so the search below
    # cannot pick up an invocation from an earlier run.
    t_mark_ms = int(time.time() * 1000) - 5000

    with tempfile.TemporaryDirectory() as td:
        req = Path(td) / "req.json"
        req.write_text(wire, encoding="utf-8")
        t0 = time.perf_counter()
        pub = aws("iot-data", "publish",
                  "--endpoint-url", endpoint,
                  "--topic", topic,
                  "--qos", "0",
                  "--cli-binary-format", "raw-in-base64-out",
                  "--payload", "file://" + str(req))
        pub_ms = (time.perf_counter() - t0) * 1000.0

    if pub.returncode != 0:
        print("\nPUBLISH FAILED")
        print(pub.stderr.strip()[:600])
        return 3
    print("published in %.0f ms (dominated by CLI startup)" % pub_ms)

    # The rule fires asynchronously; give it a moment, then read the log.
    print("\nwaiting for the rule to fire and the log to appear ...")
    events = []
    for attempt in range(12):
        time.sleep(5)
        ev = aws("logs", "filter-log-events",
                 "--log-group-name", LOG_GROUP,
                 "--start-time", str(t_mark_ms),
                 "--max-items", "60")
        if ev.returncode != 0:
            print("  (log group not readable yet: %s)" % ev.stderr.strip()[:120])
            continue
        events = json.loads(ev.stdout or "{}").get("events", [])
        if any("REPORT" in e["message"] for e in events):
            break
        print("  attempt %d: %d event(s) so far" % (attempt + 1, len(events)))
    else:
        print("\nNO INVOCATION SEEN in 60 s.")
        print("Most likely the IoT rule does not match '%s'." % topic)
        return 4

    print("\n--- CloudWatch ---")
    for e in events:
        print("  " + e["message"].rstrip())

    text = "\n".join(e["message"] for e in events)
    ok_invoked = "REPORT" in text
    ok_published = '"published": true' in text.lower() or "'published': True" in text
    ok_error = "Traceback" in text or "[ERROR]" in text

    print("\n--- verdict ---")
    print("  Lambda invoked by the rule      : %s" % ("YES" if ok_invoked else "NO"))
    print("  correction published back       : %s" % ("YES" if ok_published else "NO"))
    print("  errors in the log               : %s" % ("YES" if ok_error else "NO"))

    if ok_invoked and ok_published and not ok_error:
        print("\nOK - the full chain works: topic -> rule -> Lambda -> topic.")
        return 0
    if ok_invoked and not ok_published:
        print("\nLambda ran but the publish failed. Check the 'publish' line above:")
        print("  - AccessDeniedException  -> the role is missing iot:Publish")
        print("  - reason: disabled       -> PUBLISH_RESPONSE is '0'")
        print("  - no 'publish' line      -> stale zip, re-upload estimator.zip")
    return 5


if __name__ == "__main__":
    sys.exit(main())

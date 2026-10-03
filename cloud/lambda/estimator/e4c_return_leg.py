"""E4c -- Cloud-side latency decomposition on the rule-driven path.

Direct invocation measures the function's own compute. This experiment measures what happens
around it, on the path the device actually uses: an MQTT publish enters through
an IoT rule and the answer leaves through a second publish issued by the handler
itself. Neither leg appears in the Duration figure an engineer normally reads.

Two regimes are compared, because they exercise different things:

  sequential   one window per second. The rule reuses a single warm execution
               environment, so this isolates the steady-state cost.
  burst        every window published at once. Lambda scales out, each new
               environment pays initialisation, so this exposes the cold path
               without having to force it artificially by editing the function.

The quantity of interest is publish_ms -- the cost of the return leg. A cold
environment pays for boto3 client construction, credential resolution and a TLS
handshake to the iot-data endpoint on its first call, and all of that falls
OUTSIDE the Init Duration that CloudWatch reports. If that gap is large, the
reported cold-start penalty understates the staleness the device actually sees.

    python e4c_return_leg.py [n_per_phase]
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vimu-gen"))

import noise_model as nm      # noqa: E402
import trajectory as tj       # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEVICE = "picoW_thing"
AWS = os.environ.get("AWS_CLI", r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")
LOG_GROUP = "/aws/lambda/picoW_estimator"
OUT = REPO / "docs" / "e4c_results.txt"

FS, WINDOW_S = 200.0, 2.0
GYRO_LSB = 131.0 / nm.DEG
ACCEL_LSB = 16384.0 / 9.80665

# The mode/init_pub_ms fields were added when the warm-up was introduced; they
# are optional so that logs captured before that change still parse.
SEQ_RE = re.compile(
    r"seq=(?P<seq>\d+) n=(?P<n>\d+) cold=(?P<cold>True|False) "
    r"compute_ms=(?P<compute>[\d.]+) handler_ms=(?P<handler>[\d.]+) "
    r"(?:mode=(?P<mode>\S+) init_pub_ms=(?P<initpub>\S+) )?"
    r"publish=(?P<publish>\{.*\})")
REPORT_RE = re.compile(
    r"REPORT RequestId: (\S+)\s+Duration: ([\d.]+) ms\s+"
    r"Billed Duration: (\d+) ms\s+Memory Size: (\d+) MB\s+"
    r"Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?")


def encode(sig, lsb):
    counts = np.clip(np.round(sig * lsb), -32768, 32767).astype("<i2")
    return base64.b64encode(counts.tobytes()).decode("ascii")


def build_payload(seq: int) -> str:
    traj = tj.MissionTrajectory().generate(WINDOW_S + 4.0, FS)
    rng = np.random.default_rng(1000 + seq)
    gyro, _ = nm.synthesise_triad(traj.omega_body, nm.gyro_spec(), traj.dt, rng,
                                  nm.GYRO_RANGES, apply_deterministic=False)
    accel, _ = nm.synthesise_triad(traj.accel_body, nm.accel_spec(), traj.dt, rng,
                                   nm.ACCEL_RANGES, apply_deterministic=False)
    n = int(WINDOW_S * FS)
    return json.dumps({
        "device_id": DEVICE, "seq": seq,
        "t_start_us": seq * int(WINDOW_S * 1e6),
        "t_sent_us": (seq + 1) * int(WINDOW_S * 1e6),
        "fs": FS, "n": n, "gyro_fs_sel": 0, "accel_fs_sel": 0,
        "gyro": encode(gyro[:n], GYRO_LSB),
        "accel": encode(accel[:n], ACCEL_LSB),
        "state": {"q": traj.quat[0].tolist(),
                  "v": traj.velocity[0].tolist(),
                  "p": traj.position[0].tolist()},
    })


def publish(endpoint: str, path: Path) -> int:
    p = subprocess.run(
        [AWS, "iot-data", "publish", "--endpoint-url", endpoint,
         "--topic", "imu/%s/window" % DEVICE, "--qos", "0",
         "--cli-binary-format", "raw-in-base64-out",
         "--payload", "file://" + str(path)],
        capture_output=True, text=True)
    if p.returncode != 0:
        print("  publish failed:", p.stderr.strip()[:200])
    return p.returncode


def collect(t_mark_ms: int, expect: int, label: str) -> list[dict]:
    """Poll the log until `expect` invocations have reported, or time out."""
    events: list[dict] = []
    reports: list[dict] = []
    for _ in range(24):
        time.sleep(5)
        p = subprocess.run(
            [AWS, "logs", "filter-log-events", "--log-group-name", LOG_GROUP,
             "--start-time", str(t_mark_ms), "--max-items", "600"],
            capture_output=True, text=True)
        if p.returncode != 0:
            continue
        events = json.loads(p.stdout or "{}").get("events", [])
        reports = [e for e in events if e["message"].startswith("REPORT")]
        if len(reports) >= expect:
            break
        print("  %s: %d/%d reported ..." % (label, len(reports), expect))
    else:
        print("  %s: timed out with %d/%d" % (label, len(reports), expect))

    # Join the handler's own line to its REPORT through the request id, which
    # both share, so compute/publish and Duration/Init describe one invocation.
    by_req: dict[str, dict] = {}
    current: dict[str, str] = {}
    for e in sorted(events, key=lambda x: x["timestamp"]):
        msg, stream = e["message"].rstrip(), e["logStreamName"]
        if msg.startswith("START RequestId:"):
            current[stream] = msg.split()[2]
        m = SEQ_RE.search(msg)
        if m and stream in current:
            pub = json.loads(m.group("publish"))
            initpub = m.group("initpub")
            by_req.setdefault(current[stream], {}).update(
                stream=stream, seq=int(m.group("seq")),
                cold=m.group("cold") == "True",
                compute_ms=float(m.group("compute")),
                handler_ms=float(m.group("handler")),
                mode=m.group("mode") or "lazy",
                init_pub_ms=(float(initpub)
                             if initpub not in (None, "None") else float("nan")),
                published=pub.get("published"),
                publish_ms=pub.get("publish_ms", float("nan")))
        m = REPORT_RE.search(msg)
        if m:
            by_req.setdefault(m.group(1), {}).update(
                duration_ms=float(m.group(2)), billed_ms=int(m.group(3)),
                mem_mb=int(m.group(5)),
                init_ms=float(m.group(6)) if m.group(6) else None)
    return [v for v in by_req.values() if "compute_ms" in v]


def stats(xs):
    a = np.asarray([x for x in xs if x == x], dtype=float)
    if a.size == 0:
        return "n/a"
    return "n=%d  min %.1f  median %.1f  max %.1f" % (
        a.size, a.min(), np.median(a), a.max())


def main() -> int:
    if not os.path.exists(AWS):
        print("AWS CLI not found at", AWS)
        return 1
    ep = subprocess.run([AWS, "iot", "describe-endpoint", "--endpoint-type",
                         "iot:Data-ATS"], capture_output=True, text=True)
    if ep.returncode != 0:
        print("cannot resolve IoT endpoint:", ep.stderr.strip()[:200])
        return 2
    endpoint = "https://" + json.loads(ep.stdout)["endpointAddress"]

    print("E4c - Cloud-side latency breakdown")
    print("=" * 70)
    print("endpoint : %s" % endpoint)
    print("windows  : %d per phase, %d samples each\n" % (N, int(WINDOW_S * FS)))

    with tempfile.TemporaryDirectory() as td:
        files = []
        for i in range(2 * N):
            f = Path(td) / ("w%02d.json" % i)
            f.write_text(build_payload(i), encoding="utf-8")
            files.append(f)
        print("built %d windows (%.1f kB each)\n"
              % (len(files), files[0].stat().st_size / 1024))

        print("Phase A - sequential, 1 window/second")
        mark_a = int(time.time() * 1000) - 2000
        for f in files[:N]:
            publish(endpoint, f)
            time.sleep(1.0)
        rows_a = collect(mark_a, N, "A")

        # Let the sequential environments settle before the burst, so the two
        # phases do not share invocations in the log window.
        time.sleep(10)

        print("\nPhase B - burst, all at once")
        mark_b = int(time.time() * 1000) - 2000
        with ThreadPoolExecutor(max_workers=N) as pool:
            list(pool.map(lambda f: publish(endpoint, f), files[N:]))
        rows_b = collect(mark_b, N, "B")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    for label, rows in (("A (sequential)", rows_a), ("B (burst)", rows_b)):
        cold = [r for r in rows if r.get("cold")]
        warm = [r for r in rows if not r.get("cold")]
        emit()
        emit("Phase %s" % label)
        emit("-" * 70)
        emit("  invocations         : %d  (%d cold, %d warm)"
             % (len(rows), len(cold), len(warm)))
        emit("  distinct environments: %d" % len({r["stream"] for r in rows}))
        emit("  all published       : %s"
             % ("YES" if rows and all(r.get("published") for r in rows) else "NO"))
        emit("  client mode         : %s"
             % ", ".join(sorted({r.get("mode", "?") for r in rows})))
        emit("  compute_ms (ESKF)   : %s" % stats([r["compute_ms"] for r in rows]))
        emit("  publish_ms  cold    : %s" % stats([r["publish_ms"] for r in cold]))
        emit("  publish_ms  warm    : %s" % stats([r["publish_ms"] for r in warm]))
        emit("  warm-up in init     : %s"
             % stats([r.get("init_pub_ms", float("nan")) for r in cold]))
        emit("  Init Duration       : %s"
             % stats([r["init_ms"] for r in cold if r.get("init_ms")]))
        emit("  Duration (reported) : %s"
             % stats([r.get("duration_ms", float("nan")) for r in rows]))

    all_rows = rows_a + rows_b
    cold = [r for r in all_rows if r.get("cold") and r.get("init_ms")]
    warm = [r for r in all_rows if not r.get("cold")]
    emit()
    emit("True cost of a cold start")
    emit("-" * 70)
    if cold and warm:
        wp = np.median([r["publish_ms"] for r in warm])
        init = np.array([r["init_ms"] for r in cold], float)
        extra = np.array([r["publish_ms"] for r in cold], float) - wp
        emit("  reported Init Duration           : %7.1f ms (mean of %d)"
             % (init.mean(), init.size))
        emit("  extra penalty on first publish   : %7.1f ms" % extra.mean())
        emit("  true cold-start cost             : %7.1f ms"
             % (init.mean() + extra.mean()))
        emit("  CloudWatch underestimate         : %7.1f %%"
             % (100.0 * extra.mean() / (init.mean() + extra.mean())))
        emit()
        emit("  Reason: boto3 builds the client, resolves credentials and opens")
        emit("  TLS to iot-data only on the first call, i.e. after the init phase.")
        emit("  That time adds to correction staleness but appears neither in")
        emit("  Init Duration nor in the Duration of any later invocation.")
    else:
        emit("  not enough cold/warm samples in this run")

    # Keep each mode's capture separate, so the before/after comparison does not
    # depend on remembering which run produced which file.
    modes = sorted({r.get("mode", "lazy") for r in all_rows}) or ["unknown"]
    out = OUT.with_name("e4c_results_%s.txt" % "_".join(modes))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# AWS Lambda estimator — deployment

The offloaded half of the system: a stateless function that takes a window of
IMU samples plus the filter's prior, runs the 15-state ESKF, and returns a bias
correction stamped with the time it applies to.

Validated offline before deployment — `python test_local.py` drives the handler
through 30 sequential windows with no AWS involved, so that any failure after
deployment is an AWS failure (IAM, layers, message size, timeout) rather than an
algorithm failure.

## Build

    python pack.py                    # 12 kB, numpy expected from a layer
    python pack.py --with-numpy       # vendored, for x86_64
    python pack.py --with-numpy --arch manylinux2014_aarch64   # Graviton

Handler entry point: `handler.lambda_handler`

`pack.py` copies `eskf.py` and `rotations.py` out of `vimu-gen/` at package
time rather than keeping duplicates here, so the code that runs in Lambda is
provably the code that E2b validated.

**Do not bundle a Windows numpy build.** It packages cleanly and then fails at
import inside Lambda, which is a slow failure to diagnose. Either attach an
AWS-provided numpy/SciPy layer, or use `--with-numpy`, which forces
`--platform manylinux2014_*` and `--only-binary=:all:`.

## Configuration

| Setting | Value | Why |
|---------|-------|-----|
| Runtime | Python 3.12+ | matches the host validation |
| Memory | **512 MB or more** | Lambda scales CPU with memory; the filter is a per-sample Python loop and is CPU-bound, not memory-bound |
| Timeout | 30 s | generous; a 2 s window measures ~120 ms median locally |
| Architecture | either | x86_64 vs arm64 is a variable in E5, so deploy both |
| `MAX_SAMPLES` | 4000 (default) | the AWS IoT Core 128 kB message ceiling |

## Wire contract

Request:

```json
{
  "device_id": "pico-w-01", "seq": 12,
  "t_start_us": 24000000, "t_sent_us": 26000000,
  "fs": 200.0, "n": 400,
  "gyro_fs_sel": 0, "accel_fs_sel": 0,
  "gyro":  "<base64 int16 LE, n*3>",
  "accel": "<base64 int16 LE, n*3>",
  "state": {"q": [...4], "v": [...3], "p": [...3], "bg": [...3], "ba": [...3]},
  "P_packed": "<base64 float32, 120-element upper triangle>"
}
```

Response adds `valid_at_us`, `sigma_bg`, `sigma_ba`, an updated `P_packed`,
`cold_start`, `compute_ms`, `handler_ms`, and `echo_t_sent_us`.

### Two things about this contract that are not incidental

**`valid_at_us` is the OOSM hook.** The correction describes the state at the
*end of the submitted window*, not at the moment it arrives. Measured locally,
compute alone is ~120 ms median and up to ~358 ms, and the window's own span
adds seconds — so a correction is already badly stale before a single packet
crosses the network. Out-of-sequence handling on the device is mandatory, not
an optimisation, and this field is what makes it possible.

**`P_packed` exists because Lambda is stateless.** A short window often contains
no stationary interval at all, and without a prior the filter observes nothing.
Measured on a window with zero stationary samples:

| | gyro bias error |
|---|---|
| without carried state | 0.0350 deg/s |
| with carried state | 0.0217 deg/s |

The covariance therefore has to travel with every request. At 640 base64
characters it is 8.4% of a 7.5 kB window — cheap, but it is a real architectural
consequence of choosing FaaS over a long-lived service, and worth stating as
such in the thesis.

## Sizing

A 2 s window at 200 Hz is 400 samples → 7.5 kB of JSON, comfortably inside the
128 kB IoT Core limit. The ceiling is ~4000 samples (20 s); operationally the
window should stay far shorter, since the point of offloading is a *timely*
correction.

## Measurement hooks for E4/E5

Every response carries `cold_start` (module-scope flag, false only on the first
invocation into a fresh execution environment), `compute_ms` (filter only),
`handler_ms` (including decode and encode) and `echo_t_sent_us`.

Together these decompose end-to-end latency into network, cold start and
compute. Note that `echo_t_sent_us` supports **round-trip** time only: the Pico
has no disciplined clock, so one-way latency is not derivable and should not be
reported.

For cold-start statistics, deploy many identically configured functions and
invoke each once, rather than waiting for containers to be reclaimed. Every
such invocation is guaranteed cold, which turns a multi-day serial campaign
into minutes without weakening the measurement.

## Wiring to the device

AWS IoT Core rule, MQTT topic → Lambda:

    SELECT * FROM 'imu/+/window'

Reply on `imu/<device>/correction`. The `device_id` and `seq` fields are echoed
so the device can match a correction to the window that produced it — necessary
because replies can arrive out of order, which is the whole point.

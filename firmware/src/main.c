/* E3 -- Cortex-M0+ compute profile for the offloaded estimator.
 *
 * Question: is running a 15-state ESKF on the Pico W actually feasible at the
 * IMU rate, and if not, by how much does it miss?
 *
 * This matters because the entire thesis premise -- that estimation work
 * should be moved off the RTOS node -- needs a measured justification rather
 * than an assumed one. The RP2040 is a dual Cortex-M0+ at 125 MHz with no
 * floating-point unit, so every float operation is a library call. (Not pure
 * softfloat: the RP2040 bootrom carries optimised single-precision routines
 * that the SDK links by default, so these numbers are the *best* case for the
 * part, not a strawman.)
 *
 * Budget: a 200 Hz control loop has 5 ms per sample. Anything consuming a
 * large fraction of that leaves no room for the sensor driver, the network
 * stack, or any scheduling slack.
 *
 * Method: SysTick, single-cycle resolution. Each measurement is the minimum
 * over several trials, which rejects samples perturbed by interrupts -- the
 * right statistic for a floor, where the mean would be contaminated by
 * whatever else the core happened to be doing.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pico/stdlib.h"
#include "bench.h"
#include "nav_math.h"
#include "eskf_m0.h"

#define REPEATS 5
#define BUDGET_NS 5000000ull   /* 200 Hz */

typedef struct {
    const char *label;
    uint32_t cycles;
    int is_header;
} result_t;

static result_t g_results[40];
static int g_n = 0;

static volatile float g_sink_f;
static volatile double g_sink_d;

/* Buffers for the state-dimension sweep. 3 x 39^2 floats = 18 kB, in .bss. */
#define MAX_DIM 39
static float s_bigA[MAX_DIM * MAX_DIM];
static float s_bigB[MAX_DIM * MAX_DIM];
static float s_bigC[MAX_DIM * MAX_DIM];

static void add_header(const char *label) {
    g_results[g_n].label = label;
    g_results[g_n].cycles = 0;
    g_results[g_n].is_header = 1;
    g_n++;
}

static void add_result(const char *label, uint32_t total, int iters) {
    uint32_t oh = bench_overhead();
    uint32_t net = (total > oh) ? (total - oh) : 0;
    g_results[g_n].label = label;
    g_results[g_n].cycles = net / (uint32_t)iters;
    g_results[g_n].is_header = 0;
    g_n++;
}

#define BENCH(label, iters, stmt)                                   \
    do {                                                            \
        uint32_t _best = 0xFFFFFFFFu;                               \
        for (int _r = 0; _r < REPEATS; _r++) {                      \
            uint32_t _a = bench_now();                              \
            for (int _i = 0; _i < (iters); _i++) { stmt; }          \
            uint32_t _b = bench_now();                              \
            uint32_t _d = bench_delta(_a, _b);                      \
            if (_d < _best) _best = _d;                             \
        }                                                           \
        add_result((label), _best, (iters));                        \
    } while (0)

/* ------------------------------------------------------------------ */
/* Correctness: the sparse predict must reproduce the dense one.       */
/* ------------------------------------------------------------------ */
static eskf_m0_t f_dense, f_sparse;

static void seed_filter(eskf_m0_t *f) {
    eskf_m0_init(f);
    /* A non-trivial, non-symmetric-looking but symmetric P, and a tilted
     * attitude, so that a bug in any block shows up rather than cancelling. */
    for (int i = 0; i < ESKF_N; i++) {
        for (int j = 0; j <= i; j++) {
            float val = 1e-3f * (float)(1 + ((i * 7 + j * 3) % 11));
            f->P[i * ESKF_N + j] = val;
            f->P[j * ESKF_N + i] = val;
        }
    }
    float rv[3] = {0.21f, -0.35f, 0.12f};
    quat_from_rotvec(rv, f->q);
    f->ba[0] = 0.02f; f->ba[1] = -0.01f; f->ba[2] = 0.03f;
    f->bg[0] = 1e-3f; f->bg[1] = -2e-3f; f->bg[2] = 5e-4f;
}

static uint32_t verify_sparse_matches_dense(void) {
    const float gyro[3] = {0.05f, -0.03f, 0.02f};
    const float accel[3] = {0.3f, -0.2f, 9.7f};
    const float dt = 0.005f;

    seed_filter(&f_dense);
    seed_filter(&f_sparse);
    for (int step = 0; step < 3; step++) {
        eskf_m0_predict_dense(&f_dense, gyro, accel, dt);
        eskf_m0_predict_sparse(&f_sparse, gyro, accel, dt);
    }

    float max_abs = 0.0f, max_ref = 0.0f;
    for (int i = 0; i < ESKF_N * ESKF_N; i++) {
        float d = fabsf(f_dense.P[i] - f_sparse.P[i]);
        float r = fabsf(f_dense.P[i]);
        if (d > max_abs) max_abs = d;
        if (r > max_ref) max_ref = r;
    }
    /* Return relative error in parts per billion, as an integer. */
    if (max_ref <= 0.0f) return 0xFFFFFFFFu;
    return (uint32_t)((max_abs / max_ref) * 1e9f);
}

/* ------------------------------------------------------------------ */
static void run_benchmarks(void) {
    /* --- raw arithmetic, dependency-chained so latency is measured --- */
    add_header("raw arithmetic (dependency chain)");
    {
        float x = 1.0001f;
        BENCH("float add", 1000, x = x + 1.0001f);
        g_sink_f = x;
        x = 1.0001f;
        BENCH("float multiply", 1000, x = x * 1.0000001f);
        g_sink_f = x;
        x = 1.5f;
        BENCH("float divide", 500, x = 2.0f / x + 1.0f);
        g_sink_f = x;
        x = 2.0f;
        BENCH("float sqrtf", 500, x = sqrtf(x) + 1.0f);
        g_sink_f = x;
        x = 0.5f;
        BENCH("float sinf", 200, x = sinf(x) + 0.5f);
        g_sink_f = x;

        double d = 1.0001;
        BENCH("double add", 500, d = d + 1.0001);
        g_sink_d = d;
        d = 1.0001;
        BENCH("double multiply", 500, d = d * 1.0000001);
        g_sink_d = d;
    }

    /* --- quaternion primitives --- */
    add_header("quaternion primitives");
    {
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float r[4] = {0.9998f, 0.01f, 0.015f, 0.005f};
        float out[4], R[9];
        float rv[3] = {0.001f, -0.002f, 0.0015f};

        BENCH("quat_mul", 200, quat_mul(q, r, out); q[0] = out[0]);
        g_sink_f = q[0];
        BENCH("quat_from_rotvec", 200, quat_from_rotvec(rv, out); rv[0] = out[1] * 0.001f);
        g_sink_f = out[0];
        BENCH("quat_to_dcm", 200, quat_to_dcm(r, R); r[1] = R[0] * 0.01f);
        g_sink_f = R[0];
    }

    /* --- Tier 0: what stays on the device --- */
    add_header("Tier 0 (must stay on device)");
    {
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float integral[3] = {0};
        float v[3] = {0}, p[3] = {0};
        const float gyro[3] = {0.05f, -0.03f, 0.02f};
        const float accel[3] = {0.3f, -0.2f, 9.7f};

        BENCH("mahony_update", 100,
              mahony_update(q, gyro, accel, 0.005f, 1.0f, 0.05f, integral));
        g_sink_f = q[0];

        q[0] = 1.0f; q[1] = q[2] = q[3] = 0.0f;
        BENCH("strapdown_step", 100,
              strapdown_step(q, v, p, gyro, accel, 0.005f));
        g_sink_f = q[0] + v[0] + p[0];
    }

    /* --- Tier 1/2: the candidate for offload --- */
    add_header("ESKF (offload candidate)");
    {
        static float A[ESKF_N * ESKF_N], B[ESKF_N * ESKF_N], C[ESKF_N * ESKF_N];
        for (int i = 0; i < ESKF_N * ESKF_N; i++) {
            A[i] = 0.001f * (float)(i % 13);
            B[i] = 0.002f * (float)(i % 7);
        }
        BENCH("mat_mul 15x15 (3375 MAC)", 4,
              mat_mul(A, B, C, ESKF_N, ESKF_N, ESKF_N));
        g_sink_f = C[0];

        const float gyro[3] = {0.05f, -0.03f, 0.02f};
        const float accel[3] = {0.3f, -0.2f, 9.7f};

        seed_filter(&f_dense);
        BENCH("predict dense (textbook)", 4,
              eskf_m0_predict_dense(&f_dense, gyro, accel, 0.005f));
        g_sink_f = f_dense.P[0];

        seed_filter(&f_sparse);
        BENCH("predict sparse (block)", 8,
              eskf_m0_predict_sparse(&f_sparse, gyro, accel, 0.005f));
        g_sink_f = f_sparse.P[0];

        static float H[3 * ESKF_N];
        memset(H, 0, sizeof(H));
        for (int i = 0; i < 3; i++) H[i * ESKF_N + 3 + i] = 1.0f;  /* ZUPT */
        const float resid[3] = {0.01f, -0.02f, 0.005f};
        const float Rm[3] = {1e-4f, 1e-4f, 1e-4f};

        seed_filter(&f_dense);
        BENCH("update3 Joseph", 4,
              eskf_m0_update3_joseph(&f_dense, H, resid, Rm));
        g_sink_f = f_dense.P[0];

        seed_filter(&f_dense);
        BENCH("update3 short form", 8,
              eskf_m0_update3_short(&f_dense, H, resid, Rm));
        g_sink_f = f_dense.P[0];
    }

    /* --- how cost scales with state dimension ---
     *
     * A 15-state ESKF turns out to fit on this part once hand-optimised, so
     * raw compute is not by itself a reason to offload. The interesting
     * question is where the boundary actually is. Covariance propagation is
     * O(n^3), so a richer estimator -- adding scale-factor, misalignment or
     * extra-sensor states -- grows steeply. These are measured, not
     * extrapolated, so the boundary is a fact about the hardware.
     *
     * Iteration counts fall with n because SysTick is 24-bit: at 125 MHz it
     * wraps after 16.7 M cycles, and n=39 is already 9 M cycles per call. */
    add_header("state-dimension scaling (dense n x n multiply, 2n^3 = predict)");
    {
        static const int dims[] = {15, 21, 27, 33, 39};
        static const int iters[] = {4, 2, 1, 1, 1};
        static const char *labels[] = {
            "n=15  (current ESKF)", "n=21  (+scale factors)",
            "n=27  (+misalignment)", "n=33  (+2nd IMU)", "n=39  (+magnetometer)",
        };
        for (unsigned t = 0; t < sizeof(dims) / sizeof(dims[0]); t++) {
            int n = dims[t];
            for (int i = 0; i < n * n; i++) {
                s_bigA[i] = 0.001f * (float)(i % 13);
                s_bigB[i] = 0.002f * (float)(i % 7);
            }
            uint32_t best = 0xFFFFFFFFu;
            for (int r = 0; r < REPEATS; r++) {
                uint32_t a = bench_now();
                for (int i = 0; i < iters[t]; i++) {
                    mat_mul(s_bigA, s_bigB, s_bigC, n, n, n);
                }
                uint32_t b = bench_now();
                uint32_t d = bench_delta(a, b);
                if (d < best) best = d;
            }
            add_result(labels[t], best, iters[t]);
            g_sink_f = s_bigC[0];
        }
    }
}

/* ------------------------------------------------------------------ */
static void print_report(uint32_t rel_err_ppb) {
    uint32_t hz = bench_clk_hz();

    printf("\n");
    printf("========================================================================\n");
    printf(" E3 - Cortex-M0+ compute profile (RP2040 / Pico W)\n");
    printf("========================================================================\n");
    printf(" clk_sys        : %lu Hz\n", (unsigned long)hz);
    printf(" timer          : SysTick, 1-cycle resolution\n");
    printf(" timing overhead: %lu cycles (subtracted)\n",
           (unsigned long)bench_overhead());
    printf(" budget @ 200 Hz: %llu ns per sample\n", (unsigned long long)BUDGET_NS);
    printf(" build          : -O2, single precision unless stated\n");
    printf("\n");
    printf(" sparse vs dense predict: max relative difference %lu ppb %s\n",
           (unsigned long)rel_err_ppb,
           rel_err_ppb < 100000u ? "(agree)" : "(MISMATCH - results invalid)");
    printf("\n");
    printf(" %-30s %10s %12s %10s\n", "operation", "cycles", "ns", "% budget");
    printf(" %-30s %10s %12s %10s\n", "------------------------------",
           "----------", "------------", "----------");

    for (int i = 0; i < g_n; i++) {
        if (g_results[i].is_header) {
            printf("\n %s\n", g_results[i].label);
            continue;
        }
        uint64_t ns = bench_cycles_to_ns(g_results[i].cycles);
        uint32_t pct_x100 = (uint32_t)((ns * 100ull * 100ull) / BUDGET_NS);
        printf(" %-30s %10lu %12llu %7lu.%02lu\n",
               g_results[i].label,
               (unsigned long)g_results[i].cycles,
               (unsigned long long)ns,
               (unsigned long)(pct_x100 / 100),
               (unsigned long)(pct_x100 % 100));
    }
    printf("\n");
    printf("========================================================================\n");
}

int main(void) {
    stdio_init_all();

    /* Wait for the host to open the CDC port, but not forever -- the board
     * must still be usable when running detached from a terminal. */
    for (int i = 0; i < 100 && !stdio_usb_connected(); i++) {
        sleep_ms(100);
    }
    sleep_ms(500);

    bench_init();

    uint32_t rel_err_ppb = verify_sparse_matches_dense();
    run_benchmarks();

    /* Re-print on a loop so the report can be captured whenever a terminal
     * attaches, rather than only in the instant after reset. */
    while (true) {
        print_report(rel_err_ppb);
        sleep_ms(4000);
    }
    return 0;
}

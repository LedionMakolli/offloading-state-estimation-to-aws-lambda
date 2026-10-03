/* E5 -- Does a hard real-time deadline survive the cost of TLS security?
 *
 * E4a measured a 6.1 s TLS handshake to AWS IoT, of which 98% is arithmetic on
 * the Cortex-M0+ rather than network delay. Bare metal, that computation simply
 * blocks everything: during the handshake the board stopped answering USB at
 * all. Whether an RTOS changes that outcome is the central claim of the thesis,
 * so it should be measured rather than asserted.
 *
 * Method
 * ------
 * Follows the cyclictest methodology, which is the standard way to measure
 * real-time latency: run a periodic task at a priority above the load, record
 * the difference between when it *should* have woken and when it did, and
 * report a histogram rather than only a mean.
 *
 * Two tasks contend for a single core:
 *
 *   RT task      priority 5, 200 Hz, hard deadline. Runs the real Tier-0 work
 *                (Mahony attitude filter + strapdown INS step).
 *   CRYPTO task  priority 1. Performs the actual RSA-2048 private-key
 *                operation that dominates the TLS handshake.
 *
 * The load is the genuine RSA operation rather than a full handshake on
 * purpose: E4a already showed 98% of handshake time is this computation, so
 * using it directly removes network variability and isolates the scheduling
 * question, which is what actually threatens the deadline.
 *
 * Three phases let the same task be measured with and without the load:
 *   A baseline (RT alone)   B contention (RT + RSA)   C recovery (RT alone)
 *
 * Practices taken from cyclictest and applied here:
 *   - latency is measured against an absolute schedule, not a relative delay
 *   - the first second of each phase is discarded as warm-up
 *   - a 1 us-resolution histogram is kept, and percentiles come from it
 *   - the maximum is reported prominently: for a real-time system the worst
 *     case is the result, and the mean can hide it entirely
 */

#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "FreeRTOS.h"
#include "task.h"

#include "mbedtls/pk.h"
#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"
#include "mbedtls/platform_time.h"

#include "nav_math.h"
#include "aws_certs.h"

/* ---- timing symbols the shared mbedtls_config.h expects --------------------
 * No certificate date validation happens in this target, so a monotonic clock
 * satisfies both. */
time_t pico_epoch_time(time_t *timer) {
    time_t t = (time_t)(time_us_64() / 1000000ull);
    if (timer) *timer = t;
    return t;
}
mbedtls_ms_time_t mbedtls_ms_time(void) {
    return (mbedtls_ms_time_t)(time_us_64() / 1000ull);
}

/* ---- configuration --------------------------------------------------------- */
#define RT_HZ             200
#define RT_PERIOD_MS      (1000 / RT_HZ)
#define RT_PERIOD_US      (1000000 / RT_HZ)

/* A miss is a wake-up late by more than a quarter period: 1250 us at 200 Hz.
 * Loose enough that ordinary tick granularity is not counted as a failure,
 * tight enough that a genuine scheduling failure is. */
#define DEADLINE_SLACK_US (RT_PERIOD_US / 4)

#define WARMUP_SAMPLES    (RT_HZ)          /* discard the first second */
#define PHASE_A_MS        15000
#define PHASE_B_MS        60000
#define PHASE_C_MS        15000

#define PRIO_RT           5
#define PRIO_REPORT       3
#define PRIO_CRYPTO       1

/* 1 us per bucket, matching cyclictest's resolution. */
#define HIST_BUCKETS      1000

typedef enum { PH_A = 0, PH_B = 1, PH_C = 2, PH_DONE = 3 } phase_t;
static volatile phase_t g_phase = PH_A;

/* ---- per-phase statistics ---------------------------------------------------- */
typedef struct {
    uint32_t n;
    uint32_t warmup_left;
    uint32_t overflow;          /* latency >= HIST_BUCKETS us */
    uint32_t max_us;
    uint32_t min_us;
    uint64_t sum_us;
    uint32_t misses;
    uint32_t max_work_us;
    uint64_t sum_work_us;
    /* Signed deviation of the measured period from nominal. Unlike absolute
     * lateness this is immune to any offset or slow drift between the FreeRTOS
     * tick and the hardware timer, so it isolates true cycle-to-cycle jitter. */
    int32_t  per_err_min;
    int32_t  per_err_max;
    uint16_t hist[HIST_BUCKETS];
} stats_t;

static stats_t g_stats[3];
static volatile uint32_t g_rsa_ops = 0;
static volatile uint64_t g_rsa_total_us = 0;
static volatile uint32_t g_rsa_max_us = 0;

static void stats_reset(stats_t *s) {
    memset(s, 0, sizeof(*s));
    s->min_us = UINT32_MAX;
    s->per_err_min = INT32_MAX;
    s->per_err_max = INT32_MIN;
    s->warmup_left = WARMUP_SAMPLES;
}

static void stats_add(stats_t *s, int64_t late_us, int64_t per_err_us,
                      uint32_t work_us) {
    if (s->warmup_left) { s->warmup_left--; return; }
    /* cyclictest semantics: latency is how late the wake-up was. A correctly
     * scheduled task never wakes early relative to its absolute schedule, so
     * negative values are clamped rather than treated as a separate quantity. */
    uint32_t lat = (late_us > 0) ? (uint32_t)late_us : 0u;

    s->n++;
    s->sum_us += lat;
    if (lat > s->max_us) s->max_us = lat;
    if (lat < s->min_us) s->min_us = lat;
    if (lat < HIST_BUCKETS) s->hist[lat]++; else s->overflow++;
    if (lat > (uint32_t)DEADLINE_SLACK_US) s->misses++;

    if (per_err_us < s->per_err_min) s->per_err_min = (int32_t)per_err_us;
    if (per_err_us > s->per_err_max) s->per_err_max = (int32_t)per_err_us;

    if (work_us > s->max_work_us) s->max_work_us = work_us;
    s->sum_work_us += work_us;
}

/* Percentile read off the histogram, integer arithmetic only. */
static uint32_t stats_pct(const stats_t *s, uint32_t num, uint32_t den) {
    if (!s->n) return 0;
    uint32_t target = (uint32_t)(((uint64_t)s->n * num) / den);
    if (target == 0) target = 1;
    uint32_t cum = 0;
    for (uint32_t i = 0; i < HIST_BUCKETS; i++) {
        cum += s->hist[i];
        if (cum >= target) return i;
    }
    return s->max_us;
}

/* ---- the hard real-time task ------------------------------------------------- */
static void rt_task(void *arg) {
    (void)arg;
    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float integral[3] = {0}, v[3] = {0}, p[3] = {0};
    uint32_t k = 0;
    phase_t last = PH_A;

    for (int i = 0; i < 3; i++) stats_reset(&g_stats[i]);

    TickType_t wake = xTaskGetTickCount();
    /* The reference must come from a real wake-up, not from before the first
     * delay. Sampling the clock mid-tick puts the reference up to one tick
     * ahead of every wake that follows, which drives all measured lateness
     * negative and reports a flat zero -- a result that looks clean and is
     * simply wrong. */
    bool have_ref = false;
    uint64_t expected_us = 0;
    uint64_t prev_wake_us = 0;

    while (g_phase != PH_DONE) {
        xTaskDelayUntil(&wake, pdMS_TO_TICKS(RT_PERIOD_MS));
        uint64_t now = time_us_64();

        /* The real Tier-0 work: a synthetic IMU sample driven through the
         * attitude filter and the strapdown integrator. */
        uint64_t w0 = time_us_64();
        float ph = (float)k * 0.01f;
        float gyro[3]  = {0.05f * ph, -0.03f, 0.02f};
        float accel[3] = {0.30f, -0.20f, 9.70f + 0.01f * ph};
        mahony_update(q, gyro, accel, 1.0f / RT_HZ, 1.0f, 0.05f, integral);
        strapdown_step(q, v, p, gyro, accel, 1.0f / RT_HZ);
        uint32_t work_us = (uint32_t)(time_us_64() - w0);
        k++;

        phase_t cur = g_phase;
        if (cur != last) {
            last = cur;
            have_ref = false;          /* re-establish the reference */
        }
        if (cur >= PH_DONE) break;

        if (!have_ref) {
            expected_us = now;
            prev_wake_us = now;
            have_ref = true;
            continue;                  /* first wake defines t0, never scored */
        }

        expected_us += RT_PERIOD_US;
        int64_t late_us    = (int64_t)(now - expected_us);
        int64_t per_err_us = (int64_t)(now - prev_wake_us) - RT_PERIOD_US;
        prev_wake_us = now;

        stats_add(&g_stats[cur], late_us, per_err_us, work_us);
    }
    vTaskDelete(NULL);
}

/* ---- the low-priority crypto load ------------------------------------------- */
static mbedtls_pk_context g_pk;
static mbedtls_entropy_context g_entropy;
static mbedtls_ctr_drbg_context g_drbg;

static void crypto_task(void *arg) {
    (void)arg;
    while (g_phase == PH_A) vTaskDelay(pdMS_TO_TICKS(20));

    unsigned char hash[32];
    unsigned char sig[512];
    size_t sig_len = 0;
    for (int i = 0; i < 32; i++) hash[i] = (unsigned char)(i * 7 + 1);

    while (g_phase == PH_B) {
        uint64_t t0 = time_us_64();
        int rc = mbedtls_pk_sign(&g_pk, MBEDTLS_MD_SHA256, hash, sizeof(hash),
                                 sig, sizeof(sig), &sig_len,
                                 mbedtls_ctr_drbg_random, &g_drbg);
        uint32_t dt = (uint32_t)(time_us_64() - t0);
        if (rc == 0) {
            g_rsa_ops++;
            g_rsa_total_us += dt;
            if (dt > g_rsa_max_us) g_rsa_max_us = dt;
        } else {
            printf("[crypto] pk_sign failed: -0x%04x\n", (unsigned)(-rc));
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
    vTaskDelete(NULL);
}

/* ---- reporting --------------------------------------------------------------- */
static const char *phase_name(int i) {
    return i == 0 ? "A baseline (RT alone)"
         : i == 1 ? "B contention (RT + RSA)"
                  : "C recovery (RT alone)";
}

static void print_hist(const stats_t *s) {
    /* Coarse buckets for readability; the fine histogram backs the percentiles. */
    static const uint32_t edges[] = {0, 50, 100, 200, 400, 800, HIST_BUCKETS};
    printf("     latency distribution: ");
    for (unsigned e = 0; e + 1 < sizeof(edges) / sizeof(edges[0]); e++) {
        uint32_t c = 0;
        for (uint32_t i = edges[e]; i < edges[e + 1]; i++) c += s->hist[i];
        if (c) printf("[%lu-%lu us: %lu] ",
                      (unsigned long)edges[e], (unsigned long)edges[e + 1] - 1,
                      (unsigned long)c);
    }
    if (s->overflow) printf("[>=%d us: %lu] ", HIST_BUCKETS, (unsigned long)s->overflow);
    printf("\n");
}

static void print_report(void) {
    printf("\n========================================================================\n");
    printf(" E5 - Real-time deadline under TLS crypto load (FreeRTOS, RP2040)\n");
    printf("========================================================================\n");
    printf(" FreeRTOS      : %s, %d core, tick %d Hz\n",
           tskKERNEL_VERSION_NUMBER, (int)configNUMBER_OF_CORES,
           (int)configTICK_RATE_HZ);
    printf(" RT task       : %d Hz (period %d us), priority %d\n",
           RT_HZ, RT_PERIOD_US, PRIO_RT);
    printf(" load          : RSA-2048 private key op, priority %d\n", PRIO_CRYPTO);
    printf(" deadline slack: %d us (quarter period)\n", DEADLINE_SLACK_US);
    printf(" warm-up       : %d samples discarded per phase\n", WARMUP_SAMPLES);
    if (g_rsa_ops) {
        printf(" RSA-2048 ops  : %lu completed, mean %lu ms, max %lu ms\n",
               (unsigned long)g_rsa_ops,
               (unsigned long)(g_rsa_total_us / g_rsa_ops / 1000),
               (unsigned long)(g_rsa_max_us / 1000));
    }

    printf("\n %-25s %7s %6s %6s %6s %6s %7s %7s\n",
           "phase", "n", "mean", "p50", "p99", "p99.9", "MAX", "misses");
    printf(" %-25s %7s %6s %6s %6s %6s %7s %7s\n",
           "-------------------------", "-------", "------", "------",
           "------", "------", "-------", "-------");
    for (int i = 0; i < 3; i++) {
        stats_t *s = &g_stats[i];
        if (!s->n) continue;
        printf(" %-25s %7lu %6lu %6lu %6lu %6lu %7lu %7lu\n",
               phase_name(i), (unsigned long)s->n,
               (unsigned long)(s->sum_us / s->n),
               (unsigned long)stats_pct(s, 50, 100),
               (unsigned long)stats_pct(s, 99, 100),
               (unsigned long)stats_pct(s, 999, 1000),
               (unsigned long)s->max_us,
               (unsigned long)s->misses);
    }
    printf("  (all latency values in microseconds)\n");

    /* Period error is the drift-immune view: it compares consecutive wake-ups
     * against the nominal period, so a constant offset, or a slow mismatch
     * between the FreeRTOS tick and the hardware timer, cannot hide jitter. */
    printf("\n %-25s %12s %12s\n", "phase", "period min", "period max");
    printf(" %-25s %12s %12s\n", "-------------------------",
           "------------", "------------");
    for (int i = 0; i < 3; i++) {
        stats_t *s = &g_stats[i];
        if (!s->n) continue;
        printf(" %-25s %9ld us %9ld us\n", phase_name(i),
               (long)s->per_err_min, (long)s->per_err_max);
    }
    printf("  (deviation of the measured period from the nominal %d us)\n",
           RT_PERIOD_US);

    for (int i = 0; i < 3; i++) {
        if (!g_stats[i].n) continue;
        printf("\n %s\n", phase_name(i));
        print_hist(&g_stats[i]);
        printf("     Tier-0 work: mean %lu us, max %lu us (of %d us budget)\n",
               (unsigned long)(g_stats[i].sum_work_us / g_stats[i].n),
               (unsigned long)g_stats[i].max_work_us, RT_PERIOD_US);
    }

    uint32_t total_miss = g_stats[0].misses + g_stats[1].misses + g_stats[2].misses;
    printf("\n VERDICT: %s\n", total_miss == 0
           ? "no deadline missed in any phase - preemption protected the RT loop"
           : "deadlines were missed - see the misses column");
    printf("========================================================================\n");
}

static void report_task(void *arg) {
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(PHASE_A_MS));
    g_phase = PH_B;
    printf("[phase] B: RSA-2048 load ON\n");

    vTaskDelay(pdMS_TO_TICKS(PHASE_B_MS));
    g_phase = PH_C;
    printf("[phase] C: RSA load OFF\n");

    vTaskDelay(pdMS_TO_TICKS(PHASE_C_MS));
    g_phase = PH_DONE;
    vTaskDelay(pdMS_TO_TICKS(200));

    while (true) {
        print_report();
        vTaskDelay(pdMS_TO_TICKS(8000));
    }
}

/* configSUPPORT_STATIC_ALLOCATION obliges the application to supply the memory
 * for the idle and timer tasks; the kernel will not allocate them itself. */
static StaticTask_t g_idle_tcb;
static StackType_t  g_idle_stack[configMINIMAL_STACK_SIZE];
void vApplicationGetIdleTaskMemory(StaticTask_t **tcb, StackType_t **stack,
                                   configSTACK_DEPTH_TYPE *size) {
    *tcb = &g_idle_tcb;
    *stack = g_idle_stack;
    *size = configMINIMAL_STACK_SIZE;
}

static StaticTask_t g_timer_tcb;
static StackType_t  g_timer_stack[configTIMER_TASK_STACK_DEPTH];
void vApplicationGetTimerTaskMemory(StaticTask_t **tcb, StackType_t **stack,
                                    configSTACK_DEPTH_TYPE *size) {
    *tcb = &g_timer_tcb;
    *stack = g_timer_stack;
    *size = configTIMER_TASK_STACK_DEPTH;
}

void vApplicationStackOverflowHook(TaskHandle_t task, char *name) {
    (void)task;
    panic("stack overflow in task %s", name);
}
void vApplicationMallocFailedHook(void) { panic("FreeRTOS malloc failed"); }

int main(void) {
    stdio_init_all();
    for (int i = 0; i < 300 && !stdio_usb_connected(); i++) sleep_ms(100);
    sleep_ms(300);

    printf("\n[boot] E5: FreeRTOS real-time deadline under crypto load\n");

    /* Load the same RSA-2048 key the TLS handshake uses, so the measured cost
     * is that of the real credential rather than a synthetic one. */
    mbedtls_pk_init(&g_pk);
    mbedtls_entropy_init(&g_entropy);
    mbedtls_ctr_drbg_init(&g_drbg);

    const char *pers = "e5-rtos";
    int rc = mbedtls_ctr_drbg_seed(&g_drbg, mbedtls_entropy_func, &g_entropy,
                                   (const unsigned char *)pers, strlen(pers));
    if (rc) { printf("[fail] ctr_drbg_seed -0x%04x\n", (unsigned)(-rc)); while (1) sleep_ms(1000); }

    rc = mbedtls_pk_parse_key(&g_pk, (const unsigned char *)AWS_PRIVATE_KEY,
                              AWS_PRIVATE_KEY_LEN, NULL, 0,
                              mbedtls_ctr_drbg_random, &g_drbg);
    if (rc) { printf("[fail] pk_parse_key -0x%04x\n", (unsigned)(-rc)); while (1) sleep_ms(1000); }
    printf("[init] RSA-%u key loaded\n", (unsigned)mbedtls_pk_get_bitlen(&g_pk));
    printf("[phase] A: baseline, RT task alone (%d s)\n", PHASE_A_MS / 1000);

    /* Stack depths are in words. The crypto task gets 16 kB because mbedTLS
     * bignum work needs it -- and unlike the bare-metal core0 stack, capped at
     * the 4 kB of SCRATCH_Y, a FreeRTOS task stack comes from the kernel heap
     * in main SRAM. The RTOS removes that constraint rather than adding one. */
    xTaskCreate(rt_task,     "rt",     1024, NULL, PRIO_RT,     NULL);
    xTaskCreate(crypto_task, "crypto", 4096, NULL, PRIO_CRYPTO, NULL);
    xTaskCreate(report_task, "report", 1024, NULL, PRIO_REPORT, NULL);

    vTaskStartScheduler();
    panic("scheduler returned");
    return 0;
}

#include "bench.h"
#include "hardware/clocks.h"

static uint32_t s_overhead = 0;

void bench_init(void) {
    /* RVR to full scale, then CSR = ENABLE | CLKSOURCE(processor clock).
     * TICKINT stays clear: we poll CVR, we do not want an interrupt every
     * 134 ms perturbing the very measurements we are taking. */
    systick_hw->rvr = BENCH_MASK;
    systick_hw->cvr = 0;          /* any write clears the counter */
    systick_hw->csr = 0x5u;

    /* Calibrate the cost of the measurement itself. Taking the minimum of
     * several trials rejects any sample perturbed by an interrupt, which is
     * the right statistic for a floor measurement. */
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 32; i++) {
        uint32_t a = bench_now();
        uint32_t b = bench_now();
        uint32_t d = bench_delta(a, b);
        if (d < best) best = d;
    }
    s_overhead = best;
}

uint32_t bench_overhead(void) { return s_overhead; }

uint32_t bench_clk_hz(void) { return clock_get_hz(clk_sys); }

uint64_t bench_cycles_to_ns(uint32_t cycles) {
    return ((uint64_t)cycles * 1000000000ull) / (uint64_t)clock_get_hz(clk_sys);
}

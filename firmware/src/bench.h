/* Cycle-accurate timing for the RP2040.
 *
 * The Cortex-M0+ has no DWT cycle counter -- that is an M3/M4/M7 feature -- so
 * the usual CoreDebug->DEMCR / DWT->CYCCNT recipe does not exist here. What the
 * core does have is SysTick, a 24-bit down-counter clocked from the processor
 * clock, which gives single-cycle resolution.
 *
 * At the default 125 MHz, 24 bits wraps every 134 ms, comfortably longer than
 * any single operation measured in E3. The alternative, time_us_64(), has only
 * microsecond resolution and would quantise a 3 us operation into 3 buckets.
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include "hardware/structs/systick.h"

#define BENCH_MASK 0x00FFFFFFu

void bench_init(void);

/* Measured cost of an empty start/stop pair, subtracted from every result. */
uint32_t bench_overhead(void);

uint32_t bench_clk_hz(void);
uint64_t bench_cycles_to_ns(uint32_t cycles);

/* SysTick counts DOWN, so elapsed is start-minus-end. */
static inline uint32_t bench_now(void) { return systick_hw->cvr; }

static inline uint32_t bench_delta(uint32_t start, uint32_t end) {
    return (start - end) & BENCH_MASK;
}

#endif /* BENCH_H */

/* FreeRTOS configuration for the RP2040 real-time experiment.
 *
 * Single core on purpose. The RP2040 has two Cortex-M0+ cores, and running the
 * crypto task on the second one would make the contention question disappear
 * by construction. The experiment asks whether a hard 5 ms deadline survives a
 * multi-second RSA operation *on the same core*, so the answer has to be earned
 * by the scheduler rather than handed over by the hardware. Using the second
 * core is a mitigation to evaluate afterwards, not the baseline.
 */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#include "pico/types.h"

/* --- scheduler ---------------------------------------------------------- */
#define configNUMBER_OF_CORES                   1
#define configUSE_PREEMPTION                    1
#define configUSE_TIME_SLICING                  1
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 0
#define configUSE_TICKLESS_IDLE                 0

/* 1 kHz tick. The real-time task runs at 200 Hz, so a 1 ms tick gives five
 * ticks per period -- enough resolution that scheduler granularity is not the
 * dominant term in the jitter being measured. */
#define configTICK_RATE_HZ                      ( ( TickType_t ) 1000 )
#define configCPU_CLOCK_HZ                      125000000

#define configMAX_PRIORITIES                    8
#define configMINIMAL_STACK_SIZE                ( ( configSTACK_DEPTH_TYPE ) 256 )
#define configMAX_TASK_NAME_LEN                 16
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1
#define configSTACK_DEPTH_TYPE                  uint32_t

/* --- memory -------------------------------------------------------------
 * Task stacks come out of this heap, which lives in main SRAM. That matters:
 * bare metal, the core0 stack sits in SCRATCH_Y and is capped at 4 kB, which
 * is uncomfortably tight for mbedTLS bignum work. Under FreeRTOS a task can
 * simply be given a 16 kB stack. The RTOS removes a hardware constraint here
 * rather than adding one. */
#define configSUPPORT_DYNAMIC_ALLOCATION        1
#define configSUPPORT_STATIC_ALLOCATION         1
#define configTOTAL_HEAP_SIZE                   ( 96 * 1024 )
#define configAPPLICATION_ALLOCATED_HEAP        0

/* --- hooks and checks ---------------------------------------------------- */
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
#define configUSE_MALLOC_FAILED_HOOK            1
/* Stack overflow checking stays on: this experiment deliberately runs an
 * RSA-2048 operation inside a task, and silent stack corruption would look
 * like a scheduling anomaly and corrupt the measurement. */
#define configCHECK_FOR_STACK_OVERFLOW          2

/* --- features actually used ---------------------------------------------- */
#define configUSE_MUTEXES                       1
#define configUSE_RECURSIVE_MUTEXES             1
#define configUSE_COUNTING_SEMAPHORES           1
#define configUSE_TASK_NOTIFICATIONS            1
#define configUSE_TIMERS                        1
#define configTIMER_TASK_PRIORITY               ( configMAX_PRIORITIES - 1 )
#define configTIMER_QUEUE_LENGTH                8
#define configTIMER_TASK_STACK_DEPTH            1024
#define configQUEUE_REGISTRY_SIZE               8
#define configUSE_TRACE_FACILITY                1
#define configUSE_STATS_FORMATTING_FUNCTIONS    0
#define configGENERATE_RUN_TIME_STATS           0

/* --- API functions ------------------------------------------------------- */
#define INCLUDE_vTaskPrioritySet                1
#define INCLUDE_uxTaskPriorityGet               1
#define INCLUDE_vTaskDelete                     1
#define INCLUDE_vTaskSuspend                    1
#define INCLUDE_xTaskDelayUntil                 1
#define INCLUDE_vTaskDelay                      1
#define INCLUDE_xTaskGetSchedulerState          1
#define INCLUDE_uxTaskGetStackHighWaterMark     1
#define INCLUDE_xTaskGetIdleTaskHandle          1
#define INCLUDE_eTaskGetState                   1
#define INCLUDE_xTaskAbortDelay                 1
#define INCLUDE_xTaskGetHandle                  1
/* The RP2040 port's FIFO interrupt handler calls xEventGroupSetBitsFromISR,
 * which event_groups.c only compiles when configUSE_TRACE_FACILITY,
 * configUSE_TIMERS and this flag are all set. It defaults to 0, so leaving
 * it out links cleanly right up to the final step and then fails with an
 * undefined reference from inside the port. */
#define INCLUDE_xTimerPendFunctionCall          1

/* --- RP2040 port specifics ----------------------------------------------- */
/* Lets FreeRTOS tasks block on SDK synchronisation primitives and on SDK
 * timing calls (sleep_ms and friends) without spinning. */
#define configSUPPORT_PICO_SYNC_INTEROP         1
#define configSUPPORT_PICO_TIME_INTEROP         1
#define configUSE_DYNAMIC_EXCEPTION_HANDLERS    0

/* Cortex-M0+ has no BASEPRI, so the port masks interrupts wholesale; there is
 * no interrupt priority configuration to set here. */

/* panic() already masks interrupts, and taskDISABLE_INTERRUPTS is not visible
 * this early -- it lives in task.h, which is included after this file. */
#define configASSERT(x) if ((x) == 0) { panic("FreeRTOS assert %s:%d", __FILE__, __LINE__); }

#endif /* FREERTOS_CONFIG_H */

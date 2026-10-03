cmake_minimum_required(VERSION 3.13...3.27)

# Board is the Pico W (RP2040 + CYW43439). E3 measures compute only and does
# not bring up WiFi, but the board definition should still match the hardware
# so that clock configuration and pin mapping are correct.
set(PICO_BOARD pico_w CACHE STRING "Target board")

include(pico_sdk_import.cmake)

project(e3_compute_profile C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(e3_compute_profile
    src/main.c
    src/bench.c
    src/nav_math.c
    src/eskf_m0.c
)

target_include_directories(e3_compute_profile PRIVATE src)

# -O2 is what a real deployment would ship. Benchmarking at -O0 would measure
# the compiler's debug output rather than the cost of the algorithm.
target_compile_options(e3_compute_profile PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

target_link_libraries(e3_compute_profile
    pico_stdlib
)

# Report over USB CDC. UART is disabled so that nothing competes for the pins
# and so the benchmark is not paying for a 115200-baud blocking write.
pico_enable_stdio_usb(e3_compute_profile 1)
pico_enable_stdio_uart(e3_compute_profile 0)

pico_add_extra_outputs(e3_compute_profile)

# --------------------------------------------------------------------------
# E4 stage 1: WiFi + SNTP + DNS + mutual TLS to AWS IoT Core.
#
# Built only when the private config exists, so a fresh clone still builds E3
# without anyone's credentials. Copy aws_config.h.template to aws_config.h and
# run tools/gen_certs.py to produce aws_certs.h.
# --------------------------------------------------------------------------
if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/aws_config.h AND
    EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/aws_certs.h)

    add_executable(e4_net_test src/net_test_main.c)

    target_include_directories(e4_net_test PRIVATE src)
    target_compile_options(e4_net_test PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

    # The SDK default stack is 2 kB. mbedTLS performs RSA-2048 operations from
    # inside lwIP callbacks, which on this arch run on the main stack, and the
    # bignum temporaries are tight there.
    #
    # 4096 is the ceiling, not a preference: on RP2040 the core0 stack lives in
    # SCRATCH_Y, which is exactly 4 kB. Asking for 16 kB fails at link time with
    # "region SCRATCH_Y overflowed by 12288 bytes". Going beyond this needs a
    # custom linker script that relocates the stack into main SRAM.
    target_compile_definitions(e4_net_test PRIVATE
        PICO_STACK_SIZE=4096
        # Give stdio time to attach before the heavy work starts, so the log of
        # a failing run can actually be captured.
        PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS=30000
    )

    target_link_libraries(e4_net_test
        pico_stdlib
        pico_cyw43_arch_lwip_threadsafe_background
        pico_lwip_sntp
        pico_lwip_mbedtls
        pico_mbedtls
    )

    pico_enable_stdio_usb(e4_net_test 1)
    pico_enable_stdio_uart(e4_net_test 0)
    pico_add_extra_outputs(e4_net_test)

    # ----------------------------------------------------------------------
    # E4b: the closed loop. MQTT over the same mutual-TLS connection, with a
    # real window published and the correction received back.
    #
    # Needs aws_window.h from tools/gen_window.py in addition to the
    # credentials, so it is gated separately.
    # ----------------------------------------------------------------------
    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/aws_window.h)
        add_executable(e4b_mqtt src/e4b_mqtt_main.c)

        target_include_directories(e4b_mqtt PRIVATE src)
        target_compile_options(e4b_mqtt PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

        target_compile_definitions(e4b_mqtt PRIVATE
            PICO_STACK_SIZE=4096
            PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS=30000
        )

        target_link_libraries(e4b_mqtt
            pico_stdlib
            pico_cyw43_arch_lwip_threadsafe_background
            pico_lwip_sntp
            pico_lwip_mqtt
            pico_lwip_mbedtls
            pico_mbedtls
        )

        pico_enable_stdio_usb(e4b_mqtt 1)
        pico_enable_stdio_uart(e4b_mqtt 0)
        pico_add_extra_outputs(e4b_mqtt)
    else()
        message(WARNING
            "e4b_mqtt skipped: src/aws_window.h missing. "
            "Run: python tools/gen_window.py")
    endif()
else()
    message(WARNING
        "e4_net_test skipped: src/aws_config.h and/or src/aws_certs.h missing. "
        "Copy aws_config.h.template and run: python tools/gen_certs.py")
endif()

# --------------------------------------------------------------------------
# E5: FreeRTOS real-time deadline under TLS crypto load.
#
# Needs aws_certs.h for the real RSA-2048 key, so it is gated the same way as
# e4_net_test. FREERTOS_KERNEL_PATH points at the kernel unpacked next to the
# SDK; override on the command line to relocate it.
# --------------------------------------------------------------------------
if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/aws_certs.h)
    if (NOT DEFINED FREERTOS_KERNEL_PATH)
        set(FREERTOS_KERNEL_PATH "$ENV{USERPROFILE}/.pico-sdk/FreeRTOS-Kernel")
    endif()

    if (EXISTS ${FREERTOS_KERNEL_PATH}/portable/ThirdParty/GCC/RP2040/FreeRTOS_Kernel_import.cmake)
        include(${FREERTOS_KERNEL_PATH}/portable/ThirdParty/GCC/RP2040/FreeRTOS_Kernel_import.cmake)

        add_executable(e5_rtos_jitter
            src/rtos_jitter_main.c
            src/nav_math.c
        )
        target_include_directories(e5_rtos_jitter PRIVATE src)
        target_compile_options(e5_rtos_jitter PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

        target_link_libraries(e5_rtos_jitter
            pico_stdlib
            pico_mbedtls
            FreeRTOS-Kernel
            FreeRTOS-Kernel-Heap4
        )

        pico_enable_stdio_usb(e5_rtos_jitter 1)
        pico_enable_stdio_uart(e5_rtos_jitter 0)
        pico_add_extra_outputs(e5_rtos_jitter)
    else()
        message(WARNING "e5_rtos_jitter skipped: FreeRTOS kernel not at ${FREERTOS_KERNEL_PATH}")
    endif()
endif()

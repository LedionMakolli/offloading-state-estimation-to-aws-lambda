/* lwIP configuration for the Pico W AWS IoT client.
 *
 * NO_SYS=1: lwIP runs without an RTOS, driven from the cyw43_arch
 * threadsafe_background poll. FreeRTOS integration comes later; keeping the
 * network stack bare-metal for now means the connectivity measurements are not
 * confounded by scheduler behaviour.
 */
#ifndef _LWIPOPTS_H
#define _LWIPOPTS_H

/* lwipopts.h is pulled in by lwip/opt.h before any lwIP type is defined, so
 * the SNTP callback declared at the bottom of this file has nothing to spell
 * uint32_t with unless stdint comes first. */
#include <stdint.h>

#define NO_SYS                      1
#define LWIP_SOCKET                 0
#define LWIP_NETCONN                0

#define MEM_LIBC_MALLOC             0
#define MEM_ALIGNMENT               4
/* TLS needs far more heap than a plain TCP example: the record buffers, the
 * certificate chain during verification, and the RSA bignum scratch all live
 * here. 16 kB is comfortable on a 264 kB part. */
#define MEM_SIZE                    16000

#define MEMP_NUM_TCP_SEG            32
#define MEMP_NUM_ARP_QUEUE          10
#define PBUF_POOL_SIZE              24

#define LWIP_ARP                    1
#define LWIP_ETHERNET               1
#define LWIP_ICMP                   1
#define LWIP_RAW                    1
#define LWIP_IPV4                   1
#define LWIP_IPV6                   0
#define LWIP_TCP                    1
#define LWIP_UDP                    1
#define LWIP_DNS                    1
#define LWIP_DHCP                   1

#define TCP_MSS                     1460
#define TCP_WND                     (8 * TCP_MSS)
#define TCP_SND_BUF                 (8 * TCP_MSS)
#define TCP_SND_QUEUELEN            ((4 * (TCP_SND_BUF) + (TCP_MSS - 1)) / (TCP_MSS))

#define LWIP_NETIF_STATUS_CALLBACK  1
#define LWIP_NETIF_LINK_CALLBACK    1
#define LWIP_NETIF_HOSTNAME         1
#define LWIP_NETIF_TX_SINGLE_PBUF   1
#define LWIP_TCP_KEEPALIVE          1
#define DHCP_DOES_ARP_CHECK         0
#define LWIP_DHCP_DOES_ACD_CHECK    0

#define LWIP_CHKSUM_ALGORITHM       3

/* TLS over TCP. altcp is the abstraction layer that lets the same code path
 * carry either plain TCP or TLS. */
#define LWIP_ALTCP                  1
#define LWIP_ALTCP_TLS              1
#define LWIP_ALTCP_TLS_MBEDTLS      1

/* SNTP. The clock has to be correct before TLS, because certificate validity
 * dates are checked against it. A Pico boots believing it is 1970, which makes
 * every certificate look "not yet valid" -- an error that gives no hint that
 * the real problem is the clock. */
/* lwIP sizes MEMP_NUM_SYS_TIMEOUT automatically, but the formula counts only
 * its INTERNAL modules (TCP, ARP, DHCP, DNS, ...). SNTP lives in apps/ and is
 * not counted, yet it takes a slot. Enabling it therefore silently exhausts
 * the pool, and the next module to ask for a timeout -- TCP, on connect --
 * dies with:
 *
 *     *** PANIC *** sys_timeout: timeout != NULL, pool MEMP_SYS_TIMEOUT is empty
 *
 * which names SNTP nowhere and points at TCP, where the problem is not.
 * Each entry costs about a dozen bytes, so headroom is cheap insurance. */
#define MEMP_NUM_SYS_TIMEOUT        16

/* MQTT.
 *
 * The output ring buffer must hold an ENTIRE publish, header included:
 * mqtt_publish() checks for space up front and returns ERR_MEM rather than
 * streaming a message larger than the buffer. A two-second window at 200 Hz is
 * 400 samples x 6 int16 = 4800 bytes, which base64 expands to 6400 characters,
 * plus roughly 300 bytes of JSON scaffolding. The default 256 bytes is short by
 * a factor of twenty-five.
 *
 * This buffer lives inside mqtt_client_t. It is deliberately NOT allocated with
 * mqtt_client_new(), which would take it from the 16 kB lwIP heap; the client is
 * a static instead, so the cost is visible in the link map. */
#define MQTT_OUTPUT_RINGBUF_SIZE    8192

/* Holds the topic name of an incoming publish. "imu/picoW_thing/correction" is
 * 26 characters, so the default is ample; pinned here so a longer device name
 * later fails loudly at compile time rather than truncating a topic. */
#define MQTT_VAR_HEADER_BUFFER_LEN  128

#define SNTP_SERVER_DNS             1
#define SNTP_STARTUP_DELAY          0
#define SNTP_UPDATE_DELAY           3600000
#define SNTP_SET_SYSTEM_TIME(sec)   sntp_set_system_time_wrapper(sec)
void sntp_set_system_time_wrapper(uint32_t sec);

/* Statistics off: they cost RAM and we are not debugging lwIP itself. */
#define MEM_STATS                   0
#define SYS_STATS                   0
#define MEMP_STATS                  0
#define LINK_STATS                  0

/* Uncomment for verbose lwIP tracing when something refuses to connect.
#define LWIP_DEBUG                  1
#define ALTCP_MBEDTLS_DEBUG         LWIP_DBG_ON
#define DNS_DEBUG                   LWIP_DBG_ON
*/

#endif /* _LWIPOPTS_H */

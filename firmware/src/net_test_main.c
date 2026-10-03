/* E4 stage 1 -- WiFi, SNTP, DNS and mutual-TLS connectivity to AWS IoT Core.
 *
 * Proves the hard half of the integration works before any MQTT or IMU data is
 * involved: the credentials parse, the clock is right, AWS accepts our client
 * certificate, and the handshake completes. If this runs, the remaining work
 * is protocol plumbing rather than security.
 *
 * It is also the E4 measurement. Each stage is timed, and plain TCP is
 * connected separately from TLS so the handshake can be isolated:
 *
 *     TLS handshake ~= (TCP+TLS connect) - (plain TCP connect)
 *
 * That subtraction matters because a naive "TLS connect took 3 s" figure
 * conflates an RSA-2048 private-key operation on a 125 MHz core with several
 * round trips to Stockholm. Separating them is the difference between a
 * measurement and an anecdote.
 *
 * Why raw altcp rather than lwIP's MQTT client: mqtt_client_connect() creates
 * the TLS pcb internally and connects immediately, leaving no window to call
 * mbedtls_ssl_set_hostname(). AWS IoT requires SNI, so the pcb has to be owned
 * here. MQTT layers on afterwards.
 */

#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"

#include "lwip/altcp.h"
#include "lwip/altcp_tcp.h"
#include "lwip/altcp_tls.h"
#include "lwip/dns.h"
#include "lwip/apps/sntp.h"

#include "mbedtls/ssl.h"
#include "mbedtls/platform_time.h"
#include "mbedtls/error.h"

#include "aws_config.h"
#include "aws_certs.h"

#define STAGE_TIMEOUT_MS 40000

/* ---------------------------------------------------------------- clock -- */
static volatile uint32_t g_epoch_base = 0;
static volatile uint64_t g_epoch_ref_us = 0;
static volatile bool g_time_valid = false;

/* Called by lwIP's SNTP client via SNTP_SET_SYSTEM_TIME in lwipopts.h. */
void sntp_set_system_time_wrapper(uint32_t sec) {
    g_epoch_base = sec;
    g_epoch_ref_us = time_us_64();
    g_time_valid = true;
}

/* Wired into mbedTLS by MBEDTLS_PLATFORM_TIME_MACRO. Certificate validity
 * dates are checked against this, so it must be real wall-clock time. */
time_t pico_epoch_time(time_t *timer) {
    time_t t = (time_t)(g_epoch_base +
                        (uint32_t)((time_us_64() - g_epoch_ref_us) / 1000000ull));
    if (timer) *timer = t;
    return t;
}

/* Monotonic millisecond clock for mbedTLS (MBEDTLS_PLATFORM_MS_TIME_ALT).
 * Must be monotonic, not wall clock: mbedTLS uses it for timeouts, and a
 * clock that jumps when SNTP lands would corrupt them. time_us_64() counts
 * from boot and never steps. */
mbedtls_ms_time_t mbedtls_ms_time(void) {
    return (mbedtls_ms_time_t)(time_us_64() / 1000ull);
}

/* ------------------------------------------------------------- helpers -- */
static volatile bool g_dns_done = false;
static volatile bool g_dns_ok = false;
static ip_addr_t g_server_ip;

static void dns_cb(const char *name, const ip_addr_t *ipaddr, void *arg) {
    (void)name; (void)arg;
    if (ipaddr) { g_server_ip = *ipaddr; g_dns_ok = true; }
    g_dns_done = true;
}

typedef enum { CONN_PENDING, CONN_OK, CONN_FAILED } conn_state_t;
static volatile conn_state_t g_conn = CONN_PENDING;
static volatile err_t g_conn_err = ERR_OK;

static err_t on_connected(void *arg, struct altcp_pcb *pcb, err_t err) {
    (void)arg; (void)pcb;
    g_conn_err = err;
    g_conn = (err == ERR_OK) ? CONN_OK : CONN_FAILED;
    return ERR_OK;
}

static void on_err(void *arg, err_t err) {
    (void)arg;
    g_conn_err = err;
    g_conn = CONN_FAILED;
}

static err_t on_recv(void *arg, struct altcp_pcb *pcb, struct pbuf *p, err_t err) {
    (void)arg; (void)err;
    if (p) { altcp_recved(pcb, p->tot_len); pbuf_free(p); }
    return ERR_OK;
}

/* Busy-wait on a volatile flag. threadsafe_background runs lwIP off an
 * interrupt, so the main thread only has to yield. */
static bool wait_flag(volatile bool *flag, uint32_t timeout_ms) {
    absolute_time_t deadline = make_timeout_time_ms(timeout_ms);
    while (!*flag) {
        if (absolute_time_diff_us(get_absolute_time(), deadline) < 0) return false;
        sleep_ms(2);
    }
    return true;
}

static bool wait_conn(uint32_t timeout_ms) {
    absolute_time_t deadline = make_timeout_time_ms(timeout_ms);
    while (g_conn == CONN_PENDING) {
        if (absolute_time_diff_us(get_absolute_time(), deadline) < 0) return false;
        sleep_ms(2);
    }
    return g_conn == CONN_OK;
}

/* ----------------------------------------------------------------- LED -- */
/* Stage signalling on the onboard LED, so a run that hangs before the serial
 * port can be opened is still diagnosable: the blink count says which stage
 * was entered last. On a board where USB may stop being serviced, this is the
 * only channel guaranteed to keep working. */
#define LED_WIFI  1
#define LED_SNTP  2
#define LED_DNS   3
#define LED_TCP   4
#define LED_PARSE 5
#define LED_TLS   6

static void led_blink(int n) {
    for (int i = 0; i < n; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(150);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(150);
    }
    sleep_ms(700);
}

/* ------------------------------------------------------------- results -- */
typedef struct { const char *label; int64_t us; bool ok; const char *note; } stage_t;
static stage_t g_stage[10];
static int g_nstage = 0;

static void record(const char *label, int64_t us, bool ok, const char *note) {
    if (g_nstage < (int)(sizeof(g_stage) / sizeof(g_stage[0]))) {
        g_stage[g_nstage++] = (stage_t){label, us, ok, note};
    }
}

/* --------------------------------------------------------------- stages -- */
static bool connect_once(bool use_tls, struct altcp_tls_config *tls_cfg, int64_t *out_us) {
    g_conn = CONN_PENDING;
    g_conn_err = ERR_OK;

    cyw43_arch_lwip_begin();
    struct altcp_pcb *pcb = use_tls ? altcp_tls_new(tls_cfg, IPADDR_TYPE_V4)
                                    : altcp_tcp_new_ip_type(IPADDR_TYPE_V4);
    if (!pcb) { cyw43_arch_lwip_end(); return false; }

    if (use_tls) {
        /* SNI. AWS IoT requires it, and lwIP's altcp_tls never sets it -- there
         * is no mbedtls_ssl_set_hostname call anywhere in altcp_tls_mbedtls.c.
         * This also enables hostname verification against the server cert. */
        mbedtls_ssl_context *ssl = (mbedtls_ssl_context *)altcp_tls_context(pcb);
        if (ssl) mbedtls_ssl_set_hostname(ssl, AWS_IOT_ENDPOINT);
    }

    altcp_err(pcb, on_err);
    altcp_recv(pcb, on_recv);

    absolute_time_t t0 = get_absolute_time();
    err_t e = altcp_connect(pcb, &g_server_ip, AWS_IOT_PORT, on_connected);
    cyw43_arch_lwip_end();

    if (e != ERR_OK) { *out_us = 0; return false; }

    bool ok = wait_conn(STAGE_TIMEOUT_MS);
    *out_us = absolute_time_diff_us(t0, get_absolute_time());

    cyw43_arch_lwip_begin();
    altcp_close(pcb);
    cyw43_arch_lwip_end();
    return ok;
}

static void print_report(void) {
    printf("\n========================================================================\n");
    printf(" E4 stage 1 - AWS IoT connectivity from Pico W\n");
    printf("========================================================================\n");
    printf(" endpoint : %s:%d\n", AWS_IOT_ENDPOINT, AWS_IOT_PORT);
    printf(" thing    : %s\n", AWS_THING_NAME);
    printf(" cert     : RSA-2048, %u B of PEM embedded in flash\n",
           (unsigned)(AWS_ROOT_CA_LEN + AWS_DEVICE_CERT_LEN + AWS_PRIVATE_KEY_LEN));
    if (g_time_valid) {
        /* newlib on this toolchain makes time_t 64-bit, so widen explicitly
         * rather than truncating into a long. */
        long long now = (long long)pico_epoch_time(NULL);
        printf(" clock    : unix %lld (SNTP)\n", now);
    } else {
        printf(" clock    : NOT SET\n");
    }
    printf("\n %-34s %12s %8s  %s\n", "stage", "ms", "result", "note");
    printf(" %-34s %12s %8s  %s\n", "----------------------------------",
           "------------", "--------", "----");

    int64_t tcp_us = -1, tls_us = -1;
    for (int i = 0; i < g_nstage; i++) {
        printf(" %-34s %9lld.%03lld %8s  %s\n",
               g_stage[i].label,
               (long long)(g_stage[i].us / 1000), (long long)(g_stage[i].us % 1000),
               g_stage[i].ok ? "ok" : "FAIL",
               g_stage[i].note ? g_stage[i].note : "");
        if (strstr(g_stage[i].label, "TCP connect")) tcp_us = g_stage[i].us;
        if (strstr(g_stage[i].label, "TCP+TLS")) tls_us = g_stage[i].us;
    }

    if (tcp_us > 0 && tls_us > 0) {
        int64_t hs = tls_us - tcp_us;
        printf("\n TLS handshake alone      = %lld.%03lld ms  (TCP+TLS minus plain TCP)\n",
               (long long)(hs / 1000), (long long)(hs % 1000));
        printf(" of which the RSA-2048 private-key operation on a 125 MHz\n");
        printf(" Cortex-M0+ with no crypto accelerator is the dominant term.\n");
    }
    printf("========================================================================\n");
}

int main(void) {
    stdio_init_all();
    /* 30 s, so a terminal can attach before the heavy work begins. */
    for (int i = 0; i < 300 && !stdio_usb_connected(); i++) sleep_ms(100);
    sleep_ms(500);

    printf("\n[boot] E4 stage 1: AWS IoT connectivity test\n");

    if (cyw43_arch_init_with_country(WIFI_COUNTRY)) {
        printf("[fail] cyw43_arch_init failed\n");
        while (true) sleep_ms(1000);
    }
    cyw43_arch_enable_sta_mode();

    /* --- WiFi ---------------------------------------------------------- */
    led_blink(LED_WIFI);
    printf("[wifi] connecting to \"%s\" ...\n", WIFI_SSID);
    absolute_time_t t0 = get_absolute_time();
    int rc = cyw43_arch_wifi_connect_timeout_ms(WIFI_SSID, WIFI_PASSWORD,
                                                CYW43_AUTH_WPA2_AES_PSK, 30000);
    int64_t dt = absolute_time_diff_us(t0, get_absolute_time());
    record("WiFi associate + DHCP", dt, rc == 0, rc == 0 ? "" : "check SSID/password/2.4GHz");
    if (rc) {
        printf("[fail] wifi connect rc=%d\n", rc);
        print_report();
        while (true) sleep_ms(5000);
    }
    printf("[wifi] ip = %s\n", ip4addr_ntoa(netif_ip4_addr(netif_default)));

    /* --- SNTP ---------------------------------------------------------- */
    led_blink(LED_SNTP);
    printf("[sntp] syncing clock (needed before certificate dates can be checked)\n");
    t0 = get_absolute_time();
    cyw43_arch_lwip_begin();
    sntp_setoperatingmode(SNTP_OPMODE_POLL);
    sntp_setservername(0, "pool.ntp.org");
    sntp_init();
    cyw43_arch_lwip_end();
    bool time_ok = wait_flag(&g_time_valid, 20000);
    dt = absolute_time_diff_us(t0, get_absolute_time());
    record("SNTP time sync", dt, time_ok, time_ok ? "" : "no NTP reply");
    if (!time_ok) {
        printf("[fail] SNTP timed out; TLS would reject every certificate as not-yet-valid\n");
        print_report();
        while (true) sleep_ms(5000);
    }
    printf("[sntp] unix time = %lld\n", (long long)pico_epoch_time(NULL));

    /* --- DNS ----------------------------------------------------------- */
    led_blink(LED_DNS);
    printf("[dns ] resolving %s\n", AWS_IOT_ENDPOINT);
    t0 = get_absolute_time();
    cyw43_arch_lwip_begin();
    err_t de = dns_gethostbyname(AWS_IOT_ENDPOINT, &g_server_ip, dns_cb, NULL);
    cyw43_arch_lwip_end();
    if (de == ERR_OK) { g_dns_ok = true; g_dns_done = true; }
    else if (de == ERR_INPROGRESS) wait_flag(&g_dns_done, 15000);
    dt = absolute_time_diff_us(t0, get_absolute_time());
    record("DNS resolve", dt, g_dns_ok, g_dns_ok ? "" : "DNS failure");
    if (!g_dns_ok) { print_report(); while (true) sleep_ms(5000); }
    printf("[dns ] %s\n", ipaddr_ntoa(&g_server_ip));

    /* --- plain TCP, to isolate network RTT from TLS work ---------------- */
    led_blink(LED_TCP);
    printf("[tcp ] plain TCP connect (baseline)\n");
    int64_t tcp_us = 0;
    bool tcp_ok = connect_once(false, NULL, &tcp_us);
    record("plain TCP connect", tcp_us, tcp_ok, tcp_ok ? "network RTT baseline" : "port blocked?");

    /* --- mutual TLS ----------------------------------------------------- */
    led_blink(LED_PARSE);
    printf("[tls ] building 2-way-auth config from embedded PEM\n");
    /* Deliberately NOT inside cyw43_arch_lwip_begin/end.
     *
     * In threadsafe_background mode that lock blocks the async context, and
     * TinyUSB is serviced from the same context. Parsing an RSA-2048 private
     * key validates its CRT parameters, which on a 125 MHz M0+ with no crypto
     * accelerator takes seconds -- long enough that USB stops responding and
     * the host sees an enumerated but dead CDC device. This call touches only
     * mbedTLS state, never lwIP's, so it does not need the lock at all. */
    absolute_time_t tp0 = get_absolute_time();
    struct altcp_tls_config *cfg = altcp_tls_create_config_client_2wayauth(
        (const u8_t *)AWS_ROOT_CA,       AWS_ROOT_CA_LEN,
        (const u8_t *)AWS_PRIVATE_KEY,   AWS_PRIVATE_KEY_LEN,
        (const u8_t *)"",                0,          /* key has no passphrase */
        (const u8_t *)AWS_DEVICE_CERT,   AWS_DEVICE_CERT_LEN);
    int64_t parse_us = absolute_time_diff_us(tp0, get_absolute_time());

    if (!cfg) {
        printf("[fail] altcp_tls_create_config_client_2wayauth returned NULL\n");
        printf("       usually a PEM parse failure: check the *_LEN macros include the NUL\n");
        record("parse PEM + build TLS config", parse_us, false, "PEM parse failed");
        print_report();
        while (true) sleep_ms(5000);
    }
    record("parse PEM + build TLS config", parse_us, true,
           "RSA-2048 key validation on M0+");

    led_blink(LED_TLS);
    printf("[tls ] connecting with SNI + client certificate ...\n");
    int64_t tls_us = 0;
    bool tls_ok = connect_once(true, cfg, &tls_us);
    record("TCP+TLS connect (handshake)", tls_us, tls_ok,
           tls_ok ? "AWS accepted our certificate"
                  : "handshake rejected - see notes below");

    if (!tls_ok) {
        printf("\n[fail] TLS handshake failed, lwIP err=%d\n", (int)g_conn_err);
        printf("       likely causes, in order of frequency:\n");
        printf("        1. endpoint is not the -ats one (wrong root CA)\n");
        printf("        2. certificate not ACTIVE, or no policy attached\n");
        printf("        3. clock wrong -> certificate looks not-yet-valid\n");
        printf("        4. enable ALTCP_MBEDTLS_DEBUG in lwipopts.h for detail\n");
    }

    print_report();
    /* Finished: slow heartbeat, report reprinted every 10 s. */
    while (true) {
        for (int i = 0; i < 10; i++) {
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1); sleep_ms(60);
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0); sleep_ms(940);
        }
        print_report();
    }
    return 0;
}

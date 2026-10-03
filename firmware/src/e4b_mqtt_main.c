/* E4b -- the closed loop, measured from the device.
 *
 * E4 stage 1 proved the mutual-TLS handshake completes. This closes the loop:
 * the Pico publishes a virtual-IMU window to AWS IoT Core, an IoT rule invokes
 * the Lambda estimator, the estimator publishes a bias correction back, and the
 * Pico receives it. The quantity of interest is how long that takes.
 *
 * WHAT IS AND IS NOT MEASURED
 *
 * Round-trip time only. The Pico has no disciplined clock -- SNTP gives it the
 * date, not sub-millisecond alignment with AWS -- so a one-way figure cannot be
 * obtained by subtracting two clocks, and halving the round trip would assume a
 * symmetry that a request of 6.6 kB and a reply of 1.5 kB plainly do not have.
 * The round trip is stamped entirely with time_us_64(), one clock, both ends.
 *
 * What CAN be separated is the part AWS accounts for itself. The correction
 * carries handler_ms, measured inside the Lambda, so:
 *
 *     RTT = handler_ms + everything else
 *
 * where "everything else" is TLS record work on both sides, the WiFi link, the
 * path to Stockholm, the broker, rule dispatch, and the return publish. That
 * decomposition is reported below, and it is honest about which term is
 * measured and which is a remainder.
 *
 * SNI, AND WHY THE CLIENT IS SET UP THE WAY IT IS
 *
 * AWS IoT requires SNI, and lwIP's altcp_tls layer never calls
 * mbedtls_ssl_set_hostname() -- there is no such call anywhere in
 * altcp_tls_mbedtls.c. mqtt_client_connect() creates the TLS pcb internally,
 * which appears to leave no window to set it.
 *
 * There is one, and it is safe: mqtt_client_connect() ends by calling
 * altcp_connect(), which only issues a SYN. The TLS handshake does not begin
 * until the TCP connection completes, in a callback. lwIP here runs in
 * threadsafe_background mode, so that callback cannot run while this thread
 * holds the lock. Setting the hostname immediately after mqtt_client_connect()
 * returns therefore lands before the ClientHello is built. Reaching the pcb
 * needs mqtt_priv.h, which is why that private header is included.
 *
 * The alternative -- reimplementing MQTT over an altcp pcb we create ourselves
 * -- is a few hundred lines of packet encoding to avoid three.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"

#include "lwip/altcp.h"
#include "lwip/altcp_tls.h"
#include "lwip/dns.h"
#include "lwip/apps/sntp.h"
#include "lwip/apps/mqtt.h"
#include "lwip/apps/mqtt_priv.h"   /* for client->conn, to set SNI -- see above */

#include "mbedtls/ssl.h"
#include "mbedtls/platform_time.h"

#include "aws_config.h"
#include "aws_certs.h"
#include "aws_window.h"

#define N_TRIALS        30
#define TRIAL_GAP_MS    3000     /* pacing between windows */
#define REPLY_TIMEOUT_MS 20000   /* generous: a cold start alone can be 1 s */

#define PAYLOAD_MAX     8192
#define REPLY_MAX       3072     /* correction measured at ~1460 bytes */

/* ---------------------------------------------------------------- clock -- */
static volatile uint32_t g_epoch_base = 0;
static volatile uint64_t g_epoch_ref_us = 0;
static volatile bool g_time_valid = false;

void sntp_set_system_time_wrapper(uint32_t sec) {
    g_epoch_base = sec;
    g_epoch_ref_us = time_us_64();
    g_time_valid = true;
}

time_t pico_epoch_time(time_t *timer) {
    time_t t = (time_t)(g_epoch_base +
                        (uint32_t)((time_us_64() - g_epoch_ref_us) / 1000000ull));
    if (timer) *timer = t;
    return t;
}

mbedtls_ms_time_t mbedtls_ms_time(void) {
    return (mbedtls_ms_time_t)(time_us_64() / 1000ull);
}

/* ------------------------------------------------------------- plumbing -- */
static volatile bool g_dns_done = false, g_dns_ok = false;
static ip_addr_t g_server_ip;

static void dns_cb(const char *name, const ip_addr_t *ipaddr, void *arg) {
    (void)name; (void)arg;
    if (ipaddr) { g_server_ip = *ipaddr; g_dns_ok = true; }
    g_dns_done = true;
}

static bool wait_flag(volatile bool *flag, uint32_t timeout_ms) {
    absolute_time_t deadline = make_timeout_time_ms(timeout_ms);
    while (!*flag) {
        if (absolute_time_diff_us(get_absolute_time(), deadline) < 0) return false;
        sleep_ms(2);
    }
    return true;
}

/* ---------------------------------------------------------------- MQTT --- */
/* Static rather than mqtt_client_new(): with an 8 kB output ring buffer this
 * structure is larger than half the lwIP heap, and a link-time allocation
 * fails loudly at build time instead of returning NULL at run time. */
static mqtt_client_t g_client;

static volatile bool g_mqtt_settled = false;   /* connect callback has fired */
static volatile mqtt_connection_status_t g_mqtt_status = MQTT_CONNECT_DISCONNECTED;
static volatile bool g_sub_done = false;
static volatile err_t g_sub_err = ERR_OK;

static char g_reply[REPLY_MAX];
static volatile uint32_t g_reply_len = 0;
static volatile bool g_reply_ready = false;
static volatile uint64_t g_reply_us = 0;      /* stamped in the callback */
static volatile uint32_t g_reply_truncated = 0;

static void on_mqtt_conn(mqtt_client_t *c, void *arg, mqtt_connection_status_t st) {
    (void)c; (void)arg;
    g_mqtt_status = st;
    g_mqtt_settled = true;
}

static void on_sub(void *arg, err_t err) {
    (void)arg;
    g_sub_err = err;
    g_sub_done = true;
}

static void on_inpub(void *arg, const char *topic, u32_t tot_len) {
    (void)arg; (void)topic; (void)tot_len;
    g_reply_len = 0;
    g_reply_truncated = 0;
}

/* Runs in interrupt context. Copy and flag; parse on the main thread. */
static void on_indata(void *arg, const u8_t *data, u16_t len, u8_t flags) {
    (void)arg;
    uint32_t space = REPLY_MAX - 1 - g_reply_len;
    uint32_t take = (len < space) ? len : space;
    if (take < len) g_reply_truncated = 1;
    memcpy(g_reply + g_reply_len, data, take);
    g_reply_len += take;
    if (flags & MQTT_DATA_FLAG_LAST) {
        g_reply[g_reply_len] = '\0';
        g_reply_us = time_us_64();
        g_reply_ready = true;
    }
}

static void on_pub_done(void *arg, err_t err) {
    (void)arg;
    if (err != ERR_OK) printf("[mqtt] publish ack error %d\n", (int)err);
}

/* ------------------------------------------------------- tiny JSON scan -- */
/* The reply is generated by json.dumps() and is only ever read for a handful of
 * scalars, so a full parser would cost more flash than it earns. Keys are
 * matched with their quotes, which is what keeps "t_sent_us" from matching
 * inside "echo_t_sent_us". */
static const char *json_value(const char *buf, const char *key) {
    char pat[48];
    int n = snprintf(pat, sizeof pat, "\"%s\"", key);
    if (n <= 0 || n >= (int)sizeof pat) return NULL;
    const char *p = strstr(buf, pat);
    if (!p) return NULL;
    p = strchr(p + n, ':');
    if (!p) return NULL;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

static bool json_u64(const char *buf, const char *key, uint64_t *out) {
    const char *p = json_value(buf, key);
    if (!p) return false;
    *out = strtoull(p, NULL, 10);
    return true;
}

static bool json_double(const char *buf, const char *key, double *out) {
    const char *p = json_value(buf, key);
    if (!p) return false;
    *out = strtod(p, NULL);
    return true;
}

static bool json_bool(const char *buf, const char *key, bool *out) {
    const char *p = json_value(buf, key);
    if (!p) return false;
    *out = (strncmp(p, "true", 4) == 0);
    return true;
}

/* ------------------------------------------------------ payload builder -- */
static char g_payload[PAYLOAD_MAX];

static int build_payload(uint32_t seq, uint64_t t_start_us, uint64_t t_sent_us) {
    /* Built by hand rather than with one big snprintf: the two base64 blobs are
     * 3200 characters each, and the SDK's formatter copies %s a character at a
     * time. memcpy for the bulk keeps the cost of assembling a window well
     * under the network time it is meant to measure. */
    int k = snprintf(g_payload, PAYLOAD_MAX,
        "{\"device_id\":\"%s\",\"seq\":%lu,\"t_start_us\":%llu,\"t_sent_us\":%llu,"
        "\"fs\":%.1f,\"n\":%d,\"gyro_fs_sel\":0,\"accel_fs_sel\":0,"
        "\"state\":{\"q\":[%s],\"v\":[%s],\"p\":[%s]},\"gyro\":\"",
        AWS_THING_NAME, (unsigned long)seq,
        (unsigned long long)t_start_us, (unsigned long long)t_sent_us,
        AWS_WINDOW_FS, AWS_WINDOW_N,
        AWS_WINDOW_Q0, AWS_WINDOW_V0, AWS_WINDOW_P0);
    if (k < 0 || k >= PAYLOAD_MAX) return -1;

    static const char gyro_b64[]  = AWS_WINDOW_GYRO_B64;
    static const char accel_b64[] = AWS_WINDOW_ACCEL_B64;
    const int glen = (int)sizeof(gyro_b64) - 1;
    const int alen = (int)sizeof(accel_b64) - 1;
    const char mid[] = "\",\"accel\":\"";
    const char end[] = "\"}";

    if (k + glen + (int)sizeof(mid) + alen + (int)sizeof(end) > PAYLOAD_MAX) return -1;

    memcpy(g_payload + k, gyro_b64, glen);            k += glen;
    memcpy(g_payload + k, mid, sizeof(mid) - 1);      k += sizeof(mid) - 1;
    memcpy(g_payload + k, accel_b64, alen);           k += alen;
    memcpy(g_payload + k, end, sizeof(end) - 1);      k += sizeof(end) - 1;
    g_payload[k] = '\0';
    return k;
}

/* ---------------------------------------------------------- statistics --- */
static int cmp_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static int64_t pct(int64_t *sorted, int n, double q) {
    if (n <= 0) return 0;
    int i = (int)(q * (n - 1) + 0.5);
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    return sorted[i];
}

/* --------------------------------------------------------------- main ---- */
static int64_t rtt_us[N_TRIALS];
static double  handler_ms[N_TRIALS];
static double  compute_ms[N_TRIALS];
static bool    was_cold[N_TRIALS];
static int     n_ok = 0, n_timeout = 0, n_mismatch = 0;

static void led(int n) {
    for (int i = 0; i < n; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1); sleep_ms(150);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0); sleep_ms(150);
    }
    sleep_ms(700);
}

static void die(const char *msg) {
    printf("\n[fail] %s\n", msg);
    while (true) sleep_ms(5000);
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);

    printf("\n==================================================================\n");
    printf(" E4b - Round trip Pico -> IoT Core -> Lambda -> Pico\n");
    printf("==================================================================\n");
    printf(" endpoint : %s:%d\n", AWS_IOT_ENDPOINT, AWS_IOT_PORT);
    printf(" publish  : %s\n", TOPIC_WINDOW);
    printf(" subscribe: %s\n", TOPIC_CORRECTION);
    printf(" window   : %d samples at %.1f Hz\n\n", AWS_WINDOW_N, AWS_WINDOW_FS);

    if (cyw43_arch_init_with_country(WIFI_COUNTRY)) die("cyw43_arch_init failed");
    cyw43_arch_enable_sta_mode();

    led(1);
    printf("[wifi] associating with \"%s\"\n", WIFI_SSID);
    if (cyw43_arch_wifi_connect_timeout_ms(WIFI_SSID, WIFI_PASSWORD,
                                           CYW43_AUTH_WPA2_AES_PSK, 30000))
        die("wifi connect failed - check SSID/password and that it is 2.4 GHz");
    printf("[wifi] ip = %s\n", ip4addr_ntoa(netif_ip4_addr(netif_default)));

    led(2);
    printf("[sntp] syncing clock\n");
    cyw43_arch_lwip_begin();
    sntp_setoperatingmode(SNTP_OPMODE_POLL);
    sntp_setservername(0, "pool.ntp.org");
    sntp_init();
    cyw43_arch_lwip_end();
    if (!wait_flag(&g_time_valid, 20000))
        die("SNTP timed out - every certificate would look not-yet-valid");
    printf("[sntp] unix time = %lld\n", (long long)pico_epoch_time(NULL));

    led(3);
    printf("[dns ] resolving %s\n", AWS_IOT_ENDPOINT);
    cyw43_arch_lwip_begin();
    err_t de = dns_gethostbyname(AWS_IOT_ENDPOINT, &g_server_ip, dns_cb, NULL);
    cyw43_arch_lwip_end();
    if (de == ERR_OK) { g_dns_ok = true; g_dns_done = true; }
    else if (de == ERR_INPROGRESS) wait_flag(&g_dns_done, 15000);
    if (!g_dns_ok) die("DNS resolve failed");
    printf("[dns ] %s\n", ipaddr_ntoa(&g_server_ip));

    /* Outside the lwIP lock on purpose: validating the RSA-2048 CRT parameters
     * takes seconds on this core, and holding the lock stalls the async context
     * that also services USB, killing the serial log mid-run. */
    led(4);
    printf("[tls ] building 2-way-auth config\n");
    absolute_time_t t0 = get_absolute_time();
    struct altcp_tls_config *cfg = altcp_tls_create_config_client_2wayauth(
        (const u8_t *)AWS_ROOT_CA,     AWS_ROOT_CA_LEN,
        (const u8_t *)AWS_PRIVATE_KEY, AWS_PRIVATE_KEY_LEN,
        (const u8_t *)"", 0,
        (const u8_t *)AWS_DEVICE_CERT, AWS_DEVICE_CERT_LEN);
    printf("[tls ] config built in %lld ms\n",
           absolute_time_diff_us(t0, get_absolute_time()) / 1000);
    if (!cfg) die("altcp_tls_create_config_client_2wayauth returned NULL"
                  " - check the *_LEN macros include the NUL");

    led(5);
    struct mqtt_connect_client_info_t ci;
    memset(&ci, 0, sizeof ci);
    ci.client_id  = AWS_CLIENT_ID;   /* must equal the Thing name for the policy */
    ci.keep_alive = 60;
    ci.tls_config = cfg;

    memset(&g_client, 0, sizeof g_client);
    mqtt_set_inpub_callback(&g_client, on_inpub, on_indata, NULL);

    printf("[mqtt] connecting (TLS handshake starts after the SYN completes)\n");
    t0 = get_absolute_time();
    cyw43_arch_lwip_begin();
    err_t ce = mqtt_client_connect(&g_client, &g_server_ip, AWS_IOT_PORT,
                                   on_mqtt_conn, NULL, &ci);
    /* The one-line reason mqtt_priv.h is included. Must happen before the TCP
     * connect callback fires, which the lock guarantees. */
    if (ce == ERR_OK && g_client.conn) {
        mbedtls_ssl_context *ssl = (mbedtls_ssl_context *)altcp_tls_context(g_client.conn);
        if (ssl) mbedtls_ssl_set_hostname(ssl, AWS_IOT_ENDPOINT);
        else printf("[warn] no mbedTLS context on the pcb; SNI not set\n");
    }
    cyw43_arch_lwip_end();
    if (ce != ERR_OK) {
        printf("[fail] mqtt_client_connect returned %d\n", (int)ce);
        die("connect refused locally");
    }

    if (!wait_flag(&g_mqtt_settled, 60000)) die("MQTT connect timed out");
    int64_t connect_us = absolute_time_diff_us(t0, get_absolute_time());
    if (g_mqtt_status != MQTT_CONNECT_ACCEPTED) {
        printf("[fail] MQTT status %d ", (int)g_mqtt_status);
        printf("(256=disconnected: usually the policy or a client_id mismatch)\n");
        die("broker refused the connection");
    }
    printf("[mqtt] connected in %lld ms (TLS handshake included)\n", connect_us / 1000);

    led(6);
    cyw43_arch_lwip_begin();
    err_t se = mqtt_subscribe(&g_client, TOPIC_CORRECTION, 0, on_sub, NULL);
    cyw43_arch_lwip_end();
    if (se != ERR_OK) die("mqtt_subscribe failed to enqueue");
    if (!wait_flag(&g_sub_done, 15000)) die("SUBACK timed out");
    if (g_sub_err != ERR_OK) die("subscribe denied - check iot:Subscribe/Receive in the policy");
    printf("[mqtt] subscribed to %s\n\n", TOPIC_CORRECTION);

    printf(" trial   RTT(ms)  handler(ms)  compute(ms)  cold  note\n");
    printf(" ------  -------  -----------  -----------  ----  ----------------\n");

    for (int i = 0; i < N_TRIALS; i++) {
        g_reply_ready = false;

        uint64_t t_sent = time_us_64();
        int len = build_payload((uint32_t)i,
                                (uint64_t)i * AWS_WINDOW_N * AWS_WINDOW_DT_US,
                                t_sent);
        if (len < 0) die("payload buffer too small");

        cyw43_arch_lwip_begin();
        err_t pe = mqtt_publish(&g_client, TOPIC_WINDOW, g_payload, (u16_t)len,
                                0 /* qos */, 0 /* retain */, on_pub_done, NULL);
        cyw43_arch_lwip_end();

        if (pe != ERR_OK) {
            /* ERR_MEM here means MQTT_OUTPUT_RINGBUF_SIZE is smaller than the
             * message: mqtt_publish refuses rather than streaming. */
            printf(" %5d   publish failed, err=%d%s\n", i, (int)pe,
                   pe == ERR_MEM ? " (ring buffer too small)" : "");
            n_timeout++;
            sleep_ms(TRIAL_GAP_MS);
            continue;
        }

        if (!wait_flag(&g_reply_ready, REPLY_TIMEOUT_MS)) {
            printf(" %5d   -        -            -            -     no reply\n", i);
            n_timeout++;
            if (!mqtt_client_is_connected(&g_client)) die("connection dropped");
            continue;
        }

        int64_t rtt = (int64_t)(g_reply_us - t_sent);

        uint64_t echo = 0;
        double h = -1.0, c = -1.0;
        bool cold = false;
        bool have_echo = json_u64(g_reply, "echo_t_sent_us", &echo);
        json_double(g_reply, "handler_ms", &h);
        json_double(g_reply, "compute_ms", &c);
        json_bool(g_reply, "cold_start", &cold);

        const char *note = "";
        if (g_reply_truncated) note = "reply truncated";
        else if (!have_echo)   note = "no echo field";
        else if (echo != t_sent) { note = "stale reply"; n_mismatch++; }

        printf(" %5d  %7lld  %11.1f  %11.1f  %4s  %s\n",
               i, rtt / 1000, h, c, cold ? "yes" : "no", note);

        if (have_echo && echo == t_sent && !g_reply_truncated) {
            rtt_us[n_ok] = rtt;
            handler_ms[n_ok] = h;
            compute_ms[n_ok] = c;
            was_cold[n_ok] = cold;
            n_ok++;
        }
        sleep_ms(TRIAL_GAP_MS);
    }

    /* ------------------------------------------------------------ report -- */
    printf("\n==================================================================\n");
    printf(" Results\n");
    printf("==================================================================\n");
    printf("  windows sent         : %d\n", N_TRIALS);
    printf("  valid replies        : %d\n", n_ok);
    printf("  no reply             : %d\n", n_timeout);
    printf("  stale replies        : %d\n", n_mismatch);

    if (n_ok == 0) { printf("\n  no samples - no statistics\n"); while (true) sleep_ms(5000); }

    int64_t sorted[N_TRIALS];
    memcpy(sorted, rtt_us, sizeof(int64_t) * n_ok);
    qsort(sorted, n_ok, sizeof(int64_t), cmp_i64);

    double mean = 0.0, hmean = 0.0, cmean = 0.0;
    int n_cold = 0;
    for (int i = 0; i < n_ok; i++) {
        mean  += (double)rtt_us[i];
        hmean += handler_ms[i];
        cmean += compute_ms[i];
        if (was_cold[i]) n_cold++;
    }
    mean /= n_ok; hmean /= n_ok; cmean /= n_ok;

    printf("\n  Round trip (measured on a single clock, time_us_64)\n");
    printf("    min    %8.1f ms\n", sorted[0] / 1000.0);
    printf("    p50    %8.1f ms\n", pct(sorted, n_ok, 0.50) / 1000.0);
    printf("    mean   %8.1f ms\n", mean / 1000.0);
    printf("    p90    %8.1f ms\n", pct(sorted, n_ok, 0.90) / 1000.0);
    printf("    p99    %8.1f ms\n", pct(sorted, n_ok, 0.99) / 1000.0);
    printf("    max    %8.1f ms\n", sorted[n_ok - 1] / 1000.0);

    printf("\n  Breakdown (means)\n");
    printf("    RTT total                %8.1f ms   100.0 %%\n", mean / 1000.0);
    printf("      of which: ESKF in Lambda %6.1f ms  %6.1f %%\n",
           cmean, 100.0 * cmean / (mean / 1000.0));
    printf("      full handler           %8.1f ms  %6.1f %%\n",
           hmean, 100.0 * hmean / (mean / 1000.0));
    printf("      remainder (net+broker) %8.1f ms  %6.1f %%\n",
           mean / 1000.0 - hmean, 100.0 * (mean / 1000.0 - hmean) / (mean / 1000.0));
    printf("\n    The remainder is NOT just network: it also includes TLS on both\n");
    printf("    ends, the broker, rule dispatch and the return publish.\n");
    printf("    It is reported as a remainder because that is what one clock can measure.\n");

    printf("\n  Cold start: %d of %d invocations\n", n_cold, n_ok);
    if (n_cold > 0 && n_cold < n_ok) {
        double rc = 0, rw = 0;
        int nc = 0, nw = 0;
        for (int i = 0; i < n_ok; i++) {
            if (was_cold[i]) { rc += rtt_us[i]; nc++; } else { rw += rtt_us[i]; nw++; }
        }
        printf("    mean RTT, cold    : %8.1f ms  (n=%d)\n", rc / nc / 1000.0, nc);
        printf("    mean RTT, warm    : %8.1f ms  (n=%d)\n", rw / nw / 1000.0, nw);
        printf("    cold-start penalty: %8.1f ms\n",
               (rc / nc - rw / nw) / 1000.0);
    }

    printf("\n  Correction staleness on arrival\n");
    printf("    window                 %8.1f ms\n", 1000.0 * AWS_WINDOW_N / AWS_WINDOW_FS);
    printf("    round trip             %8.1f ms\n", mean / 1000.0);
    printf("    total                  %8.1f ms\n",
           1000.0 * AWS_WINDOW_N / AWS_WINDOW_FS + mean / 1000.0);
    printf("    -> this is how far the current state is ahead of the one the\n");
    printf("       correction describes. It is what makes this an OOSM problem.\n");

    printf("\n[done]\n");
    cyw43_arch_lwip_begin();
    mqtt_disconnect(&g_client);
    cyw43_arch_lwip_end();

    while (true) { led(1); sleep_ms(4000); }
}

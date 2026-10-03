/* mbedTLS configuration for AWS IoT Core on the RP2040.
 *
 * Trimmed to what AWS IoT actually negotiates, because every enabled feature
 * costs flash and, for the record buffers, RAM. The SDK's own
 * pico_mbedtls_config.h is only a shim that includes this file, so this is the
 * real configuration.
 *
 * AWS IoT Core requires TLS 1.2 or better and offers ECDHE key exchange with
 * RSA or ECDSA certificates. The device certificate downloaded from the
 * console is RSA-2048, so both RSA (for our client auth and the server's
 * signature) and ECDHE/ECDSA (for the key exchange and some chain elements)
 * have to be present.
 */
#ifndef MBEDTLS_CONFIG_H
#define MBEDTLS_CONFIG_H

/* Some mbedTLS sources use INT_MAX without including limits.h. */
#include <limits.h>

/* Entropy comes from the RP2040's ROSC-based hardware RNG, which the SDK
 * exposes to mbedTLS as mbedtls_hardware_poll() in pico_mbedtls.c. Without
 * this pair, mbedTLS looks for a platform entropy source that does not exist
 * on bare metal and fails at init. */
#define MBEDTLS_NO_PLATFORM_ENTROPY
#define MBEDTLS_ENTROPY_HARDWARE_ALT

/* Record buffers. The default is 16 kB each way, which is wasteful here.
 * Inbound must hold the largest handshake flight -- the AWS server's
 * certificate chain, a few kB -- so 8 kB is the safe floor. Outbound must hold
 * our largest publish; a 2 s IMU window is ~7.5 kB, so 8 kB fits with room. */
#define MBEDTLS_SSL_IN_CONTENT_LEN     8192
#define MBEDTLS_SSL_OUT_CONTENT_LEN    8192

/* lwIP's altcp_tls reaches into mbedTLS structures directly. */
#define MBEDTLS_ALLOW_PRIVATE_ACCESS

/* Certificate validity dates are checked, which is why SNTP runs first.
 * MBEDTLS_HAVE_TIME_DATE is what turns that check on; without it an expired
 * or not-yet-valid certificate would be accepted. */
#include <time.h>

#define MBEDTLS_HAVE_TIME
#define MBEDTLS_HAVE_TIME_DATE

/* Supply the clock function but NOT the type. mbedTLS passes mbedtls_time_t*
 * straight to gmtime(), which takes time_t*, so overriding
 * MBEDTLS_PLATFORM_TIME_TYPE_MACRO to anything else breaks that call. Leaving
 * the type alone makes mbedtls_time_t an alias of time_t and everything lines
 * up. */
#define MBEDTLS_PLATFORM_TIME_MACRO        pico_epoch_time
time_t pico_epoch_time(time_t *timer);

/* mbedTLS 3.6 also wants a monotonic millisecond clock, and pico_mbedtls.c
 * does not provide one -- without this the build stops at
 * "#error No mbedtls_ms_time available". We supply it from time_us_64(). */
#define MBEDTLS_PLATFORM_MS_TIME_ALT

/* --- TLS 1.2 client ---------------------------------------------------- */
#define MBEDTLS_SSL_TLS_C
#define MBEDTLS_SSL_CLI_C
#define MBEDTLS_SSL_PROTO_TLS1_2
/* AWS IoT requires the client to send SNI. */
#define MBEDTLS_SSL_SERVER_NAME_INDICATION

/* --- key exchange ------------------------------------------------------ */
#define MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED
#define MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED
#define MBEDTLS_ECDH_C
#define MBEDTLS_ECDSA_C
#define MBEDTLS_ECP_C
/* AWS uses P-256 and P-384; the rest of the curves are dead weight. */
#define MBEDTLS_ECP_DP_SECP256R1_ENABLED
#define MBEDTLS_ECP_DP_SECP384R1_ENABLED

/* --- our client certificate is RSA-2048 -------------------------------- */
#define MBEDTLS_RSA_C
#define MBEDTLS_PKCS1_V15
#define MBEDTLS_PKCS1_V21
#define MBEDTLS_BIGNUM_C

/* --- bulk crypto and hashing ------------------------------------------- */
#define MBEDTLS_AES_C
#define MBEDTLS_GCM_C
#define MBEDTLS_CIPHER_C
#define MBEDTLS_MD_C
#define MBEDTLS_SHA224_C
#define MBEDTLS_SHA256_C
#define MBEDTLS_SHA384_C
#define MBEDTLS_SHA512_C
/* Halves the SHA-256 tables at a small speed cost. Worth it here: the M0+ is
 * already the bottleneck for the RSA operation, not for hashing. */
#define MBEDTLS_SHA256_SMALLER
#define MBEDTLS_AES_FEWER_TABLES

/* --- RNG --------------------------------------------------------------- */
#define MBEDTLS_ENTROPY_C
#define MBEDTLS_CTR_DRBG_C

/* --- X.509 and PEM ----------------------------------------------------- */
#define MBEDTLS_X509_USE_C
#define MBEDTLS_X509_CRT_PARSE_C
#define MBEDTLS_PK_C
#define MBEDTLS_PK_PARSE_C
#define MBEDTLS_ASN1_PARSE_C
#define MBEDTLS_ASN1_WRITE_C
#define MBEDTLS_OID_C
/* Parses the .pem files as downloaded -- no format conversion needed. */
#define MBEDTLS_PEM_PARSE_C
#define MBEDTLS_BASE64_C

#define MBEDTLS_PLATFORM_C

/* Turns error codes into readable strings. Costs ~5 kB of flash and saves
 * hours: without it a failed handshake reports only a negative number. */
#define MBEDTLS_ERROR_C

#endif /* MBEDTLS_CONFIG_H */

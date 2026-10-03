/* 15-state error-state Kalman filter operations for the Cortex-M0+.
 *
 * Mirrors vimu-gen/eskf.py. The point of building it here is not to run it in
 * production on the Pico -- E3 exists to measure whether that is even possible
 * -- but to obtain an honest cost for the operation that the thesis proposes to
 * offload. A cost quoted from a desktop profile, or estimated from a flop
 * count, would not survive review.
 *
 * The covariance propagation P <- F P F' + Q is the expensive step: a dense
 * 15x15 triple product is 2 * 15^3 = 6750 multiply-accumulates, every sample.
 *
 * Two implementations of each heavy step are provided, because the gap between
 * them is itself a result:
 *   *_dense   the textbook form, written straight from the equations
 *   *_sparse  exploiting the block structure of F, which is I plus four
 *             nonzero 3x3 blocks
 */
#ifndef ESKF_M0_H
#define ESKF_M0_H

#include <stdint.h>

#define ESKF_N 15

typedef struct {
    float P[ESKF_N * ESKF_N];
    float q[4];
    float v[3];
    float p[3];
    float ba[3];
    float bg[3];
    float Q_diag[ESKF_N];
} eskf_m0_t;

void eskf_m0_init(eskf_m0_t *f);

/* Generic dense multiply, C(n x p) = A(n x m) B(m x p). Exposed so main.c can
 * time the bare primitive separately from the filter that uses it. */
void mat_mul(const float *A, const float *B, float *C, int n, int m, int p);

/* Covariance propagation, textbook dense form. */
void eskf_m0_predict_dense(eskf_m0_t *f, const float gyro[3],
                           const float accel[3], float dt);

/* Covariance propagation exploiting the sparsity of F.
 * F = I + G, and G has only four nonzero 3x3 blocks, so
 *     F P F' = P + G P + (G P)' + (G P) G'
 * using the symmetry of P to avoid forming P G' separately. */
void eskf_m0_predict_sparse(eskf_m0_t *f, const float gyro[3],
                            const float accel[3], float dt);

/* Three-row measurement update.
 * joseph form:  P <- (I-KH) P (I-KH)' + K R K'   -- numerically robust, 2x15^3
 * short form:   P <- P - K (H P)                 -- ~1350 MACs, 5x cheaper */
void eskf_m0_update3_joseph(eskf_m0_t *f, const float H[3 * ESKF_N],
                            const float resid[3], const float Rm[3]);
void eskf_m0_update3_short(eskf_m0_t *f, const float H[3 * ESKF_N],
                           const float resid[3], const float Rm[3]);

#endif /* ESKF_M0_H */

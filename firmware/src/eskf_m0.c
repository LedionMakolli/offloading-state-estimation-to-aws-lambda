#include "eskf_m0.h"
#include "nav_math.h"
#include <string.h>

#define N ESKF_N

/* Error-state index blocks, matching eskf.py: [dp, dv, dtheta, dba, dbg]. */
#define I_P 0
#define I_V 3
#define I_TH 6
#define I_BA 9
#define I_BG 12

/* Scratch lives in .bss rather than on the stack. Three 15x15 float buffers is
 * 2.7 kB, which would overflow the default per-core stack. */
static float s_F[N * N];
static float s_M[N * N];
static float s_T[N * N];

void mat_mul(const float *A, const float *B, float *C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float acc = 0.0f;
            for (int k = 0; k < m; k++) {
                acc += A[i * m + k] * B[k * p + j];
            }
            C[i * p + j] = acc;
        }
    }
}

void eskf_m0_init(eskf_m0_t *f) {
    memset(f, 0, sizeof(*f));
    f->q[0] = 1.0f;
    for (int i = 0; i < N; i++) {
        f->P[i * N + i] = 1e-2f;
        f->Q_diag[i] = 1e-6f;
    }
}

/* Fill A = -skew(R a_b) dt, and Rdt = -R dt, the two distinct 3x3 blocks of G. */
static void build_blocks(const eskf_m0_t *f, const float accel[3], float dt,
                         float A[9], float Rdt[9], float R[9]) {
    quat_to_dcm(f->q, R);

    float ab[3] = {accel[0] - f->ba[0], accel[1] - f->ba[1], accel[2] - f->ba[2]};
    float Ra[3];
    mat3_vec(R, ab, Ra);

    float S[9];
    skew3(Ra, S);
    for (int i = 0; i < 9; i++) {
        A[i] = -S[i] * dt;
        Rdt[i] = -R[i] * dt;
    }
}

void eskf_m0_predict_dense(eskf_m0_t *f, const float gyro[3],
                           const float accel[3], float dt) {
    float A[9], Rdt[9], R[9];
    build_blocks(f, accel, dt, A, Rdt, R);

    /* F = I + G */
    memset(s_F, 0, sizeof(s_F));
    for (int i = 0; i < N; i++) s_F[i * N + i] = 1.0f;
    for (int i = 0; i < 3; i++) {
        s_F[(I_P + i) * N + (I_V + i)] = dt;
        for (int j = 0; j < 3; j++) {
            s_F[(I_V + i) * N + (I_TH + j)] = A[i * 3 + j];
            s_F[(I_V + i) * N + (I_BA + j)] = Rdt[i * 3 + j];
            s_F[(I_TH + i) * N + (I_BG + j)] = Rdt[i * 3 + j];
        }
    }

    /* M = F P */
    mat_mul(s_F, f->P, s_M, N, N, N);

    /* P = M F'  -- transpose folded into the index expression */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) {
                acc += s_M[i * N + k] * s_F[j * N + k];
            }
            s_T[i * N + j] = acc;
        }
    }

    for (int i = 0; i < N * N; i++) f->P[i] = s_T[i];
    for (int i = 0; i < N; i++) f->P[i * N + i] += f->Q_diag[i] * dt;
}

void eskf_m0_predict_sparse(eskf_m0_t *f, const float gyro[3],
                            const float accel[3], float dt) {
    float A[9], Rdt[9], R[9];
    build_blocks(f, accel, dt, A, Rdt, R);

    const float *P = f->P;

    /* M = G P. Only block-rows 0..2 of G are nonzero, so only rows 0..8 of M
     * are nonzero and rows 9..14 are never touched. */
    for (int j = 0; j < N; j++) {
        /* rows 0..2:  G[P,V] = I dt  ->  M = dt * P[V, :] */
        for (int i = 0; i < 3; i++) {
            s_M[(0 + i) * N + j] = dt * P[(I_V + i) * N + j];
        }
        /* rows 3..5:  G[V,TH] = A, G[V,BA] = Rdt */
        for (int i = 0; i < 3; i++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) {
                acc += A[i * 3 + k] * P[(I_TH + k) * N + j];
                acc += Rdt[i * 3 + k] * P[(I_BA + k) * N + j];
            }
            s_M[(3 + i) * N + j] = acc;
        }
        /* rows 6..8:  G[TH,BG] = Rdt */
        for (int i = 0; i < 3; i++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) {
                acc += Rdt[i * 3 + k] * P[(I_BG + k) * N + j];
            }
            s_M[(6 + i) * N + j] = acc;
        }
    }

    /* T = M G'. G' has nonzero columns 0..8 only, mirroring the above. */
    for (int i = 0; i < 9; i++) {
        const float *Mi = &s_M[i * N];
        /* cols 0..2 */
        for (int j = 0; j < 3; j++) {
            s_T[i * N + j] = dt * Mi[I_V + j];
        }
        /* cols 3..5 */
        for (int j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) {
                acc += A[j * 3 + k] * Mi[I_TH + k];
                acc += Rdt[j * 3 + k] * Mi[I_BA + k];
            }
            s_T[i * N + 3 + j] = acc;
        }
        /* cols 6..8 */
        for (int j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) {
                acc += Rdt[j * 3 + k] * Mi[I_BG + k];
            }
            s_T[i * N + 6 + j] = acc;
        }
        for (int j = 9; j < N; j++) s_T[i * N + j] = 0.0f;
    }

    /* P <- P + M + M' + T, with M and T zero below row 9. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = f->P[i * N + j];
            if (i < 9) acc += s_M[i * N + j] + s_T[i * N + j];
            if (j < 9) acc += s_M[j * N + i];
            f->P[i * N + j] = acc;
        }
    }
    for (int i = 0; i < N; i++) f->P[i * N + i] += f->Q_diag[i] * dt;
}

/* 3x3 inverse by cofactors. Returns 0 if singular. */
static int inv3(const float m[9], float out[9]) {
    float c0 = m[4] * m[8] - m[5] * m[7];
    float c1 = m[5] * m[6] - m[3] * m[8];
    float c2 = m[3] * m[7] - m[4] * m[6];
    float det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    if (det > -1e-20f && det < 1e-20f) return 0;
    float inv = 1.0f / det;
    out[0] = c0 * inv;
    out[1] = (m[2] * m[7] - m[1] * m[8]) * inv;
    out[2] = (m[1] * m[5] - m[2] * m[4]) * inv;
    out[3] = c1 * inv;
    out[4] = (m[0] * m[8] - m[2] * m[6]) * inv;
    out[5] = (m[2] * m[3] - m[0] * m[5]) * inv;
    out[6] = c2 * inv;
    out[7] = (m[1] * m[6] - m[0] * m[7]) * inv;
    out[8] = (m[0] * m[4] - m[1] * m[3]) * inv;
    return 1;
}

/* Shared front half of both update forms: HP (3xN), S (3x3), K (Nx3), dx. */
static int update_gain(eskf_m0_t *f, const float H[3 * N], const float Rm[3],
                       float HP[3 * N], float K[N * 3]) {
    const float *P = f->P;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += H[i * N + k] * P[k * N + j];
            HP[i * N + j] = acc;
        }
    }

    float S[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += HP[i * N + k] * H[j * N + k];
            S[i * 3 + j] = acc + (i == j ? Rm[i] : 0.0f);
        }
    }

    float Sinv[9];
    if (!inv3(S, Sinv)) return 0;

    /* K = P H' S^-1, computed as (HP)' S^-1 using the symmetry of P. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc += HP[k * N + i] * Sinv[k * 3 + j];
            K[i * 3 + j] = acc;
        }
    }
    return 1;
}

void eskf_m0_update3_short(eskf_m0_t *f, const float H[3 * N],
                           const float resid[3], const float Rm[3]) {
    static float HP[3 * N];
    static float K[N * 3];
    if (!update_gain(f, H, Rm, HP, K)) return;

    /* P <- P - K (H P): two thin products, ~1350 MACs total. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc += K[i * 3 + k] * HP[k * N + j];
            f->P[i * N + j] -= acc;
        }
    }
    (void)resid;
}

void eskf_m0_update3_joseph(eskf_m0_t *f, const float H[3 * N],
                            const float resid[3], const float Rm[3]) {
    static float HP[3 * N];
    static float K[N * 3];
    if (!update_gain(f, H, Rm, HP, K)) return;

    /* IKH = I - K H, dense 15x15 */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc += K[i * 3 + k] * H[k * N + j];
            s_F[i * N + j] = (i == j ? 1.0f : 0.0f) - acc;
        }
    }

    mat_mul(s_F, f->P, s_M, N, N, N);          /* M = IKH P */

    for (int i = 0; i < N; i++) {              /* T = M IKH' */
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += s_M[i * N + k] * s_F[j * N + k];
            s_T[i * N + j] = acc;
        }
    }

    for (int i = 0; i < N; i++) {              /* + K R K' */
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc += K[i * 3 + k] * Rm[k] * K[j * 3 + k];
            f->P[i * N + j] = s_T[i * N + j] + acc;
        }
    }
    (void)resid;
}

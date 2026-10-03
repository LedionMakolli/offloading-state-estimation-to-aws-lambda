#include "nav_math.h"
#include <math.h>

void vec3_cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

float vec3_norm(const float a[3]) {
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

void skew3(const float v[3], float S[9]) {
    S[0] = 0.0f;   S[1] = -v[2]; S[2] =  v[1];
    S[3] =  v[2];  S[4] = 0.0f;  S[5] = -v[0];
    S[6] = -v[1];  S[7] =  v[0]; S[8] = 0.0f;
}

void mat3_vec(const float M[9], const float v[3], float out[3]) {
    out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

void quat_mul(const float q[4], const float r[4], float out[4]) {
    const float w1 = q[0], x1 = q[1], y1 = q[2], z1 = q[3];
    const float w2 = r[0], x2 = r[1], y2 = r[2], z2 = r[3];
    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

void quat_normalize(float q[4]) {
    float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 0.0f) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

void quat_from_rotvec(const float v[3], float q[4]) {
    float theta = vec3_norm(v);
    float k;
    if (theta < 1e-8f) {
        /* sin(theta/2)/theta -> 1/2 - theta^2/48. Taking the limit explicitly
         * avoids a 0/0 and, more importantly here, skips two transcendentals
         * in the common near-stationary case. */
        k = 0.5f - theta * theta * (1.0f / 48.0f);
        q[0] = 1.0f - theta * theta * 0.125f;
    } else {
        k = sinf(theta * 0.5f) / theta;
        q[0] = cosf(theta * 0.5f);
    }
    q[1] = v[0] * k;
    q[2] = v[1] * k;
    q[3] = v[2] * k;
}

void quat_to_dcm(const float q[4], float R[9]) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - wz);
    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);
    R[7] = 2.0f * (yz + wx);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

void mahony_update(float q[4], const float gyro[3], const float accel[3],
                   float dt, float kp, float ki, float integral[3]) {
    float omega[3] = {gyro[0], gyro[1], gyro[2]};

    /* Only use the accelerometer when it is plausibly measuring gravity.
     * During high dynamics it measures the vehicle's own acceleration, and
     * feeding that in as a gravity reference actively corrupts attitude. */
    float an = vec3_norm(accel);
    if (an > 1e-3f) {
        float inv = 1.0f / an;
        float a[3] = {accel[0] * inv, accel[1] * inv, accel[2] * inv};

        /* Gravity direction predicted by the current attitude, in body frame:
         * the third ROW of R, since v_body = R^T v_nav and g_nav is +z here. */
        float R[9];
        quat_to_dcm(q, R);
        float g_body[3] = {R[6], R[7], R[8]};

        float err[3];
        vec3_cross(a, g_body, err);

        integral[0] += ki * err[0] * dt;
        integral[1] += ki * err[1] * dt;
        integral[2] += ki * err[2] * dt;

        omega[0] += kp * err[0] + integral[0];
        omega[1] += kp * err[1] + integral[1];
        omega[2] += kp * err[2] + integral[2];
    }

    float dtheta[3] = {omega[0] * dt, omega[1] * dt, omega[2] * dt};
    float dq[4], qn[4];
    quat_from_rotvec(dtheta, dq);
    quat_mul(q, dq, qn);
    q[0] = qn[0]; q[1] = qn[1]; q[2] = qn[2]; q[3] = qn[3];
    quat_normalize(q);
}

void strapdown_step(float q[4], float v[3], float p[3],
                    const float gyro[3], const float accel[3], float dt) {
    float dtheta[3] = {gyro[0] * dt, gyro[1] * dt, gyro[2] * dt};
    float dq[4], qn[4];
    quat_from_rotvec(dtheta, dq);
    quat_mul(q, dq, qn);
    q[0] = qn[0]; q[1] = qn[1]; q[2] = qn[2]; q[3] = qn[3];
    quat_normalize(q);

    float R[9];
    quat_to_dcm(q, R);
    float a_nav[3];
    mat3_vec(R, accel, a_nav);
    a_nav[2] -= NAV_GRAVITY;   /* ENU: g_nav = [0, 0, -g] */

    const float half_dt2 = 0.5f * dt * dt;
    for (int i = 0; i < 3; i++) {
        p[i] += v[i] * dt + a_nav[i] * half_dt2;
        v[i] += a_nav[i] * dt;
    }
}

/* Vector, quaternion and Tier-0 navigation maths for the RP2040.
 *
 * Mirrors vimu-gen/rotations.py and vimu-gen/strapdown.py so the C and Python
 * implementations can be read side by side and cross-checked numerically.
 *
 * Conventions match the host code exactly:
 *   - quaternions are scalar-first [w, x, y, z], unit norm, body -> nav
 *   - navigation frame is ENU, so gravity is [0, 0, -9.80665]
 *   - all state is single-precision float; see main.c for the measured cost
 *     of double on this part
 */
#ifndef NAV_MATH_H
#define NAV_MATH_H

#include <stdint.h>

#define NAV_GRAVITY 9.80665f

void vec3_cross(const float a[3], const float b[3], float out[3]);
float vec3_norm(const float a[3]);
void skew3(const float v[3], float S[9]);
void mat3_vec(const float M[9], const float v[3], float out[3]);

void quat_mul(const float q[4], const float r[4], float out[4]);
void quat_normalize(float q[4]);
void quat_from_rotvec(const float v[3], float q[4]);
void quat_to_dcm(const float q[4], float R[9]);

/* Tier-0: the complementary attitude filter that stays on the device.
 * integral[3] is the filter's persistent integral term, owned by the caller. */
void mahony_update(float q[4], const float gyro[3], const float accel[3],
                   float dt, float kp, float ki, float integral[3]);

/* Tier-0: one strapdown INS step (attitude, velocity, position). */
void strapdown_step(float q[4], float v[3], float p[3],
                    const float gyro[3], const float accel[3], float dt);

#endif /* NAV_MATH_H */

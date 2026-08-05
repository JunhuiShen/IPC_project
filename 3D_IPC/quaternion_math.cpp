#include "quaternion_math.h"
#include <cmath>
#include <stdexcept>

int quaternion_product_tensor(int alpha, int beta, int gamma) {
    if (alpha < 0 || alpha > 3 || beta < 0 || beta > 3 || gamma < 0 || gamma > 3)
        throw std::invalid_argument("quaternion_product_tensor: indices must be in [0, 3]");

    if (alpha == 0 && beta == 0 && gamma == 0) return 1;
    if (alpha == 0 && beta == gamma && beta != 0) return -1;
    if (alpha != 0 && beta == 0 && alpha == gamma) return 1;
    if (alpha != 0 && gamma == 0 && alpha == beta) return 1;
    if (alpha == 1 && beta == 2 && gamma == 3) return 1;
    if (alpha == 2 && beta == 3 && gamma == 1) return 1;
    if (alpha == 3 && beta == 1 && gamma == 2) return 1;
    if (alpha == 1 && beta == 3 && gamma == 2) return -1;
    if (alpha == 2 && beta == 1 && gamma == 3) return -1;
    if (alpha == 3 && beta == 2 && gamma == 1) return -1;
    return 0;
}

Vec4 quaternion_multiply(const Vec4& a, const Vec4& b) {
    return Vec4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

// q^-1 = q* for a unit quaternion.
Vec4 quaternion_conjugate(const Vec4& quat) {
    return Vec4(quat[0], -quat[1], -quat[2], -quat[3]);
}

Vec4 quaternion_inverse(const Vec4& quat) {
    const double norm_squared = quat.squaredNorm();
    if (!(norm_squared > 0.0) || !std::isfinite(norm_squared))
        throw std::invalid_argument("quaternion_inverse requires a finite nonzero quaternion");
    return quaternion_conjugate(quat) / norm_squared;
}

Vec4 quaternion_normalize(const Vec4& quat) {
    const double norm = quat.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::invalid_argument("quaternion_normalize requires a finite nonzero quaternion");
    return quat / norm;
}

// q_dot = 1/2 (0, omega) * q.
Vec4 quaternion_time_derivative(const Vec4& q, const Vec3& omega) {
    const Vec4 omega_quaternion(0.0, omega[0], omega[1], omega[2]);
    return 0.5 * quaternion_multiply(omega_quaternion, q);
}

Vec3 quaternion_rotate(const Vec4& quat, const Vec3& vector) {
    const Vec4 vector_quaternion(0.0, vector[0], vector[1], vector[2]);
    return quaternion_multiply(quat, quaternion_multiply(vector_quaternion, quaternion_conjugate(quat))).tail<3>();
}

Vec3 quaternion_inverse_rotate(const Vec4& quat, const Vec3& vector) {
    return quaternion_rotate(quaternion_conjugate(quat), vector);
}

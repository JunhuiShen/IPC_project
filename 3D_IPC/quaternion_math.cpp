#include "quaternion_math.h"
#include "algebra/algebra.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

int quaternion_product_tensor(int alpha, int beta, int gamma) {
    assert(0 <= alpha && alpha < 4);
    assert(0 <= beta  && beta  < 4);
    assert(0 <= gamma && gamma < 4);

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
    return Rigid_Body::ALGEBRA::QuaternionMultiply(a, b);
}

// q^-1 = q* for a unit quaternion.
Vec4 quaternion_conjugate(const Vec4& quat) {
    return Rigid_Body::ALGEBRA::ConjugateQuaternion(quat);
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

// q and -q represent the same rotation.
Vec4 quaternion_align_sign(const Vec4& quat, const Vec4& reference) {
    return quat.dot(reference) < 0.0 ? -quat : quat;
}

Vec3 quaternion_rotate(const Vec4& quat, const Vec3& vector) {
    return Rigid_Body::ALGEBRA::QuaternionRotate(quat, vector);
}

Vec3 quaternion_inverse_rotate(const Vec4& quat, const Vec3& vector) {
    return quaternion_rotate(quaternion_conjugate(quat), vector);
}

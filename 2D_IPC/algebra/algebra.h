#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

namespace Rigid_Body {

inline void Assert(bool success, const std::string& flag) {
    if (!success) {
        throw std::runtime_error(flag);
    }
}

namespace ALGEBRA {

// Levi-Civita symbol for indices in {0,1,2}
inline int LeviCivita(int alpha, int beta, int gamma) {
    Assert((0 <= alpha) && (alpha <= 2), "ALGEBRA: Invalid first argument for LeviCivita 3D.");
    Assert((0 <= beta) && (beta <= 2), "ALGEBRA: Invalid second argument for LeviCivita 3D.");
    Assert((0 <= gamma) && (gamma <= 2), "ALGEBRA: Invalid third argument for LeviCivita 3D.");
    if (alpha == 0 && beta == 1 && gamma == 2)
        return 1;
    else if (alpha == 1 && beta == 2 && gamma == 0)
        return 1;
    else if (alpha == 2 && beta == 0 && gamma == 1)
        return 1;
    else if (alpha == 2 && beta == 1 && gamma == 0)
        return -1;
    else if (alpha == 1 && beta == 0 && gamma == 2)
        return -1;
    else if (alpha == 0 && beta == 2 && gamma == 1)
        return -1;
    else
        return 0;
}

// Levi-Civita symbol for indices in {0,1}
inline int LeviCivita(int alpha, int beta) {
    Assert((0 <= alpha) && (alpha <= 1), "ALGEBRA: Invalid first argument for LeviCivita 2D.");
    Assert((0 <= beta) && (beta <= 1), "ALGEBRA: Invalid second argument for LeviCivita 2D.");
    if (alpha == 0 && beta == 1)
        return 1;
    else if (alpha == 1 && beta == 0)
        return -1;
    else
        return 0;
}

// Quaternions are stored as (w, x, y, z)
inline Eigen::Vector4d ConjugateQuaternion(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]);
}

inline Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    return Eigen::Vector4d(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

inline Eigen::Vector3d QuaternionRotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    Eigen::Vector4d a = q;
    Eigen::Vector4d v_quat(double(0), v[0], v[1], v[2]);
    Eigen::Vector4d b = QuaternionMultiply(v_quat, ConjugateQuaternion(q));
    Eigen::Vector4d quat_product(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
    return Eigen::Vector3d(quat_product[1], quat_product[2], quat_product[3]);
}

// Exponential map: angular velocity * dt -> unit quaternion
inline Eigen::Vector4d QuaternionFromVector(const Eigen::Vector3d& w) {
    double omega = w.norm();
    if (omega < 1e-10) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    double s = std::sin(double(0.5) * omega);
    return Eigen::Vector4d(std::cos(double(0.5) * omega), s * w[0] / omega, s * w[1] / omega, s * w[2] / omega);
}

}  // namespace ALGEBRA
}  // namespace Rigid_Body

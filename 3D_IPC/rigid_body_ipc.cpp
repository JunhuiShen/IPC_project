#include "rigid_body_ipc.h"
#include <cmath>

namespace {

constexpr double kSmallAngleThreshold = 1.0e-4;

}  // namespace

// Let a = dt / 2, r = ||omega||, and E(omega) = [cos(a r); (sin(a r) / r) * omega]
Vec4 exp(const Vec3& omega, double dt) {
    const double r = omega.norm();
    const double half_dt = 0.5 * dt;
    const double angle = half_dt * r;
    double vector_coefficient;
    if (std::abs(angle) < kSmallAngleThreshold) {
        const double angle2 = angle * angle;
        const double angle4 = angle2 * angle2;
        vector_coefficient = half_dt * (1.0 - angle2 / 6.0 + angle4 / 120.0);
    } else {
        vector_coefficient = std::sin(angle) / r;
    }

    Vec4 result;
    result[0] = std::cos(angle);
    result[1] = vector_coefficient * omega[0];
    result[2] = vector_coefficient * omega[1];
    result[3] = vector_coefficient * omega[2];
    return result;
}

// For |a r| >= 1e-4 and r != 0, we have
// dE / d omega = [ -a sin(a r) omega^T / r ]; [ (sin(a r) / r) I  + (a cos(a r) / r^2 - sin(a r) / r^3) omega omega^T ]]
// For |a r| < 1e-4, evaluate these coefficients by Taylor expansion to avoid cancellation in (a r cos(a r) - sin(a r)) / r^3. At r = 0, the limit is [0; a I]
Mat43 dexp_domega(const Vec3& omega, double dt) {
    const double r = omega.norm();
    const double half_dt = 0.5 * dt;
    Mat43 Dexp = Mat43::Zero();
    const double angle = half_dt * r;
    double C1;
    double C2;
    double C3;

    if (std::abs(angle) < kSmallAngleThreshold) {
        const double angle2 = angle * angle;
        const double angle4 = angle2 * angle2;
        const double sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0; //sin(ar)/r = a - a^3 r^2/6 + a^5 r^4/120
        C1 = -half_dt * half_dt * sinc;
        C2 = half_dt * sinc;
        C3 = half_dt * half_dt * half_dt * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);
    } else {
        const double sin_angle = std::sin(angle);
        const double cos_angle = std::cos(angle);
        C1 = -half_dt * sin_angle / r;
        C2 = sin_angle / r;
        C3 = (half_dt * r * cos_angle - sin_angle) / (r * r * r);
    }

    for (int beta = 0; beta < 3; ++beta) {
        Dexp(0, beta) = C1 * omega[beta];
        for (int i = 0; i < 3; ++i) {
            Dexp(i + 1, beta) = C2 * kronecker_delta(i, beta) + C3 * omega[i] * omega[beta];
        }
    }
    return Dexp;
}

// Let a = dt / 2, r = ||omega||, s = sin(a r), c = cos(a r), and u = omega
// This returns H_gamma = d2 E_gamma / d omega2 for E(omega) = [c; (s / r) u]. 
// For r != 0, H_0 = (-a s / r) I + (-a^2 c / r^2 + a s / r^3) u u^T, and
// H_i(j,k) = A (delta_ij u_k + delta_ik u_j + delta_jk u_i) + B u_i u_j u_k for  i = 1,2,3,
// where A = (a r c - s) / r^3 and  B = -a^2 s / r^3 - 3 (a r c - s) / r^5
// For |a r| < 1e-4, Taylor coefficients are used to avoid cancellation. At r = 0,
// H_0 = -a^2 I and H_1 = H_2 = H_3 = 0.
std::array<Mat33, 4> d2exp_domega2(const Vec3& omega, double dt) {
    std::array<Mat33, 4> result = {Mat33::Zero(), Mat33::Zero(), Mat33::Zero(), Mat33::Zero()};
    const double r = omega.norm();
    const double half_dt = 0.5 * dt;
    const double angle = half_dt * r;
    const Mat33 omega_outer = omega * omega.transpose();
    double C1;
    double C2;
    double A;
    double B;

    if (std::abs(angle) < kSmallAngleThreshold) {
        const double angle2 = angle * angle;
        const double angle4 = angle2 * angle2;
        const double sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0;
        C1 = -half_dt * half_dt * sinc;
        C2 = std::pow(half_dt, 4) * (1.0 / 3.0 - angle2 / 30.0 + angle4 / 840.0);
        A = half_dt * half_dt * half_dt * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);
        B = std::pow(half_dt, 5)  * (1.0 / 15.0 - angle2 / 210.0 + angle4 / 7560.0);
    } else {
        const double sin_angle = std::sin(angle);
        const double cos_angle = std::cos(angle);
        C1 = -half_dt * sin_angle / r;
        C2 = -half_dt * half_dt * cos_angle / (r * r) + half_dt * sin_angle / (r * r * r);
        const double numerator = half_dt * r * cos_angle - sin_angle;
        A = numerator / (r * r * r);
        B = -half_dt * half_dt * sin_angle / (r * r * r) - 3.0 * numerator / (r * r * r * r * r);
    }

    // d2 E_0 / (d omega_beta d omega_gamma).
    for (int beta = 0; beta < 3; ++beta) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            result[0](beta, gamma) = C1 * kronecker_delta(beta, gamma) + C2 * omega[beta] * omega[gamma];
        }
    }

    // exp_i = g(r) * omega_i, with g(r) = sin(half_dt * r) / r.
    for (int i = 0; i < 3; ++i) {
        Mat33 H = B * omega[i] * omega_outer;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                H(j, k) += A * (kronecker_delta(i, j) * omega[k] + kronecker_delta(i, k) * omega[j] + kronecker_delta(j, k) * omega[i]);
            }
        }
        result[1 + i] = H;
    }
    return result;
}

// q(omega) = E(omega, dt) * q0; exp() evaluates E and includes the dt/2 factor.
Vec4 quaternion_from_angular_velocity(const Vec4& q0, const Vec3& omega, double dt) {
    return quaternion_multiply(exp(omega, dt), q0);
}

Vec3 world_space_position(const Vec3& X_centered, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt) {
    const Vec4 quat = quaternion_from_angular_velocity(q0, omega, dt);
    return x_com + quaternion_rotate(quat, X_centered);
}

Vec3 material_space_position( const Vec3& x, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt) {
    const Vec4 quat = quaternion_from_angular_velocity(q0, omega, dt);
    return quaternion_inverse_rotate(quat, x - x_com);
}

// J(c,beta) = d x_c / d q_beta for x = x_com + R(q) X_centered.
// Here c is a spatial index, beta is a quaternion index, and q = (w,v)
// The translation Jacobian is the identity and is omitted here
Mat34 dx_dq(const Vec3& X_centered, const Vec4& quat) {
    const double w = quat[0];
    const Vec3 v(quat[1], quat[2], quat[3]);
    Mat34 J = Mat34::Zero();

    for (int c = 0; c < 3; ++c) {
        J(c, 0) = 2.0 * w * X_centered[c];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                J(c, 0) += 2.0 * levi_civita(c, i, j) * v[i] * X_centered[j];
        }

        for (int a = 0; a < 3; ++a) {
            J(c, a + 1) = 2.0 * (-v[a] * X_centered[c] + v[c] * X_centered[a]);
            for (int j = 0; j < 3; ++j) {
                J(c, a + 1) += 2.0 * (kronecker_delta(c, a) * v[j] * X_centered[j] + w * levi_civita(c, a, j) * X_centered[j]);
            }
        }
    }
    return J;
}

// H[c](beta,gamma) = d2 x_c / (d q_beta d q_gamma).
// Translation is affine, so every second derivative involving it is zero.
std::array<Mat44, 3> d2x_dq2(const Vec3& X_centered) {
    std::array<Mat44, 3> H = {
        Mat44::Zero(), Mat44::Zero(), Mat44::Zero()
    };

    for (int c = 0; c < 3; ++c) {
        H[c](0, 0) = 2.0 * X_centered[c];

        for (int a = 0; a < 3; ++a) {
            for (int j = 0; j < 3; ++j) {
                const double value = 2.0 * levi_civita(c, a, j) * X_centered[j];
                H[c](0, a + 1) += value;
                H[c](a + 1, 0) += value;
            }
        }

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                H[c](a + 1, b + 1) = 2.0 * (-kronecker_delta(a, b) * X_centered[c] + kronecker_delta(a, c) * X_centered[b] + kronecker_delta(b, c) * X_centered[a]);
            }
        }
    }

    return H;
}

// For q(omega) = E(omega) * q0, with q0 fixed and Q the quternion product tensor, the Jacobian is
// dq_alpha / d omega_beta = sum_{gamma,delta} Q(alpha,gamma,delta) * (d E_gamma / d omega_beta) * q0_delta.
Mat43 dq_domega(const Vec4& q0, const Vec3& omega, double dt) {
    const Mat43 d_exp_domega = dexp_domega(omega, dt);
    Mat43 result = Mat43::Zero();
    for (int alpha = 0; alpha < 4; ++alpha) {
        for (int beta = 0; beta < 3; ++beta) {
            for (int gamma = 0; gamma < 4; ++gamma) {
                for (int delta = 0; delta < 4; ++delta) {
                    result(alpha, beta) += quaternion_product_tensor(alpha, gamma, delta) * d_exp_domega(gamma, beta) * q0[delta];
                }
            }
        }
    }
    return result;
}

// The corresponding Hessian of each quaternion component is
// d2 q_alpha / (d omega_beta d omega_eta) = sum_{gamma,delta} Q(alpha,gamma,delta) * (d2 E_gamma / (d omega_beta d omega_eta)) * q0_delta
std::array<Mat33, 4> d2q_domega2(const Vec4& q0, const Vec3& omega, double dt) {
    const std::array<Mat33, 4> d2exp = d2exp_domega2(omega, dt);
    std::array<Mat33, 4> result = {Mat33::Zero(), Mat33::Zero(), Mat33::Zero(), Mat33::Zero()};
    for (int alpha = 0; alpha < 4; ++alpha) {
        for (int gamma = 0; gamma < 4; ++gamma) {
            for (int delta = 0; delta < 4; ++delta) {
                result[alpha] += quaternion_product_tensor(alpha, gamma, delta) * d2exp[gamma] * q0[delta];
            }
        }
    }
    return result;
}

// For each x_c, dx / d omega = (dx / dq) (dq / d omega)
Mat33 dx_domega(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt) {
    const Vec4 q = quaternion_from_angular_velocity(q0, omega, dt);
    return dx_dq(X_centered, q) * dq_domega(q0, omega, dt);
}

// For each x_c, the omega Hessian is d2x_c / d omega2 = (dq/domega)^T (d2x_c/dq2) (dq/domega) + sum_alpha (dx_c/dq_alpha) (d2q_alpha/domega2)
std::array<Mat33, 3> d2x_domega2(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt) {
    std::array<Mat33, 3> result;
    const Vec4 q = quaternion_from_angular_velocity(q0, omega, dt);
    const Mat34 J_xq = dx_dq(X_centered, q);
    const std::array<Mat44, 3> H_xq = d2x_dq2(X_centered);
    const Mat43 J_qomega = dq_domega(q0, omega, dt);
    const std::array<Mat33, 4> H_qomega = d2q_domega2(q0, omega, dt);

    for (int c = 0; c < 3; ++c) {
        result[c] = J_qomega.transpose() * H_xq[c] * J_qomega;
        for (int alpha = 0; alpha < 4; ++alpha)
            result[c] += J_xq(c, alpha) * H_qomega[alpha];
        result[c] = 0.5 * (result[c] + result[c].transpose());
    }
    return result;
}

// For x(t, omega), the translation gradient contribution is g_t = gx
Vec3 rigid_node_translation_gradient(const Vec3& gx) {
    return gx;
}

// For x(t, omega), the angular-velocity gradient contribution is g_omega = (dx/domega)^T gx
Vec3 rigid_node_omega_gradient(const Vec3& gx, const Mat33& dx_domega) {
    return dx_domega.transpose() * gx;
}

// The translation Hessian contribution is H_tt = sym(Hx).
Mat33 rigid_node_translation_hessian(const Mat33& Hx) {
    return 0.5 * (Hx + Hx.transpose());
}

// Add the mixed terms sum_{i != j} J_i^T H_ij J_j separately when assembling a coupled multi-node rigid-body Hessian.
// The per-node angular-velocity Hessian contribution is H_omegaomega = J_xomega^T sym(Hx) J_xomega + sum_c gx[c] d2x_c / d omega2.
Mat33 rigid_node_omega_hessian(const Vec3& gx, const Mat33& Hx, const Mat33& dx_domega, const std::array<Mat33, 3>& d2x_domega2) {
    const Mat33 Hx_symmetric = rigid_node_translation_hessian(Hx);
    Mat33 omega_hessian = dx_domega.transpose() * Hx_symmetric * dx_domega;

    for (int c = 0; c < 3; ++c) {
        omega_hessian += gx[c] * d2x_domega2[c];
    }
    return 0.5 * (omega_hessian + omega_hessian.transpose());
}

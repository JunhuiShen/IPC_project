#include "rigid_body_ipc.h"
#include <cassert>
#include <cmath>
#include <algebra/algebra.h>

namespace {

constexpr double kSmallAngleThreshold = 1.0e-4;

// G_alpha(q) = partial R_h(q) / partial q_alpha, where G_0 = 2 q_s I_3 + 2 [q_v]_x and
// G_l = -2 q_l I_3 + 2 e_l q_v^T + 2 q_v e_l^T + 2 q_s [e_l]_x for l = 1, 2, 3.
std::array<Mat33, 4> dRh_dq(const Vec4& q) {
    const double q_s = q[0];
    const Vec3 q_v = q.tail<3>();
    std::array<Mat33, 4> G;
    G[0] = 2.0 * q_s * Mat33::Identity() + 2.0 * skew_matrix(q_v);
    for (int l = 0; l < 3; ++l) {
        const Vec3 e_l = Vec3::Unit(l);
        G[l + 1] = -2.0 * q_v[l] * Mat33::Identity() + 2.0 * e_l * q_v.transpose() + 2.0 * q_v * e_l.transpose() + 2.0 * q_s * skew_matrix(e_l);
    }
    return G;
}

// R_h(q) = (q_s^2 - q_v^T q_v) I_3 + 2 q_v q_v^T + 2 q_s [q_v]_x represents q (0, r) q^*.
// The physical rotation q (0, r) q^{-1} is R(q) = R_h(q) / ||q||^2.
// Since q_n is unit, its directional derivative is R_dot_n = G_alpha(q_n) q_dot_n_alpha - 2 (q_n^T q_dot_n) R_h(q_n).
// D = R(q(omega)) - R_hat, where R_hat = R(q_n) + dt R_dot_n, and q_dot_n = 1/2 (0, omega_n) * q_nm1.
Mat33 rotation_residual(const Vec3& omega, const Vec4& q_n, const Vec4& q_nm1, const Vec3& omega_n, double dt) {
    const auto homogeneous_quaternion_rotation_matrix = [](const Vec4& q) -> Mat33 {
        const double q_s = q[0];
        const Vec3 q_v = q.tail<3>();
        return (q_s * q_s - q_v.squaredNorm()) * Mat33::Identity() + 2.0 * q_v * q_v.transpose() + 2.0 * q_s * skew_matrix(q_v);
    };
    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt);
    const Mat33 R_candidate = homogeneous_quaternion_rotation_matrix(q);
    const Vec4 q_dot_n = quaternion_time_derivative(q_nm1, omega_n);
    const std::array<Mat33, 4> G_n = dRh_dq(q_n);
    const Mat33 R_n = homogeneous_quaternion_rotation_matrix(q_n);
    Mat33 R_dot_n = Mat33::Zero();
    for (int alpha = 0; alpha < 4; ++alpha)
        R_dot_n += q_dot_n[alpha] * G_n[alpha];
    R_dot_n -= 2.0 * q_n.dot(q_dot_n) * R_n;
    const Mat33 R_hat_np1 = R_n + dt * R_dot_n;
    return R_candidate - R_hat_np1;
}

// For E_q = 1/2 tr(D I_hat D^T), (g_q)_alpha = tr(D^T G_alpha I_hat) and
// (H_qq)_{alpha,beta} = tr(G_alpha^T G_beta I_hat) + tr(D^T K_{alpha,beta} I_hat).
std::pair<Vec4, Mat44> quaternion_inertia_derivatives(const Vec3& omega, const Vec4& q_n, const Vec4& q_nm1, const Vec3& omega_n, double dt, const Mat33& I_hat) {
    Vec4 g_q = Vec4::Zero(); // g_q = partial E_q / partial q.
    Mat44 H_qq = Mat44::Zero(); // H_qq = partial^2 E_q / partial q^2.
    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt);
    const Mat33 D = rotation_residual(omega, q_n, q_nm1, omega_n, dt); // D(q) R_p = b_p.
    const std::array<Mat33, 4> G = dRh_dq(q); // G_alpha = partial D / partial q_alpha.

    for (int alpha = 0; alpha < 4; ++alpha) {
        g_q[alpha] = (D.transpose() * G[alpha] * I_hat).trace();
        for (int beta = 0; beta < 4; ++beta) {
            // K_00 = 2 I_3, K_0l = K_l0 = 2 [e_l]_x, and
            // K_lm = -2 delta_lm I_3 + 2 e_l e_m^T + 2 e_m e_l^T.
            Mat33 K_alpha_beta;
            if (alpha == 0 && beta == 0) {
                K_alpha_beta = 2.0 * Mat33::Identity();
            } else if (alpha == 0 || beta == 0) {
                K_alpha_beta = 2.0 * skew_matrix(Vec3::Unit((alpha == 0 ? beta : alpha) - 1));
            } else {
                const int l = alpha - 1;
                const int m = beta - 1;
                const Vec3 e_l = Vec3::Unit(l);
                const Vec3 e_m = Vec3::Unit(m);
                K_alpha_beta = -2.0 * kronecker_delta(l, m) * Mat33::Identity() + 2.0 * e_l * e_m.transpose() + 2.0 * e_m * e_l.transpose();
            }
            H_qq(alpha, beta) = (G[alpha].transpose() * G[beta] * I_hat).trace() + (D.transpose() * K_alpha_beta * I_hat).trace();
        }
    }

    return {g_q, H_qq};
}

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

// J(c,beta) = d x_c / d q_beta for x = x_com + vec(q * (0, X_centered) * q^*).
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

// I_hat = sum_p m_p R_p R_p^T is constant in the body frame and is precomputed once.
Mat33 body_second_moment(const std::vector<double>& masses, const std::vector<Vec3>& R_p) {
    assert(masses.size() == R_p.size());
    Mat33 I_hat = Mat33::Zero();
    for (std::size_t p = 0; p < R_p.size(); ++p)
        I_hat += masses[p] * R_p[p] * R_p[p].transpose();
    return I_hat;
}

// E_in = 1/2 m ||x_com - x_com_n - dt v_com_n||^2 + 1/2 tr(D I_hat D^T).
double incremental_potential_energy(const Vec3& x_com, const Vec3& omega, const Vec3& x_com_n, const Vec3& v_com_n, const Vec4& q_n, const Vec4& q_nm1, const Vec3& omega_n, double dt, double total_mass, const Mat33& I_hat) {
    assert(std::abs(q_n.squaredNorm() - 1.0) < 1.0e-10 && "q_n must be a unit quaternion");
    assert(std::abs(q_nm1.squaredNorm() - 1.0) < 1.0e-10 && "q_nm1 must be a unit quaternion");
    // a = x_com - x_hat_com = x_com - x_com_n - dt v_com_n is the translational inertial residual.
    const Vec3 a = x_com - x_com_n - dt * v_com_n;
    double energy = 0.0;

    // E_x = (1/2) m a_i a_i.
    for (int i = 0; i < 3; ++i)
        energy += 0.5 * total_mass * a[i] * a[i];

    const Mat33 D = rotation_residual(omega, q_n, q_nm1, omega_n, dt); // D = R(q) - R_hat.
    energy += 0.5 * (D * I_hat * D.transpose()).trace(); // E_q = 1/2 tr(D I_hat D^T).

    return energy;
}

// g_x = m (x_com - x_com_n - dt v_com_n).
Vec3 inertia_translation_gradient(const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n, double dt, double total_mass) {
    Vec3 gradient = Vec3::Zero();

    // (g_x)_alpha = m (x_com_alpha - x_com_n_alpha - dt v_com_n_alpha).
    for (int alpha = 0; alpha < 3; ++alpha) {
        gradient[alpha] = total_mass * (x_com[alpha] - x_com_n[alpha] - dt * v_com_n[alpha]);
    }
    return gradient;
}

// H_xx = m I_3.
Mat33 inertia_translation_hessian(double total_mass) {
    Mat33 hessian = Mat33::Zero();

    // (H_x)_{alpha,beta} = m delta_{alpha,beta}.
    for (int alpha = 0; alpha < 3; ++alpha) {
        for (int beta = 0; beta < 3; ++beta) {
            hessian(alpha, beta) = total_mass * kronecker_delta(alpha, beta);
        }
    }
    return hessian;
}

// g_omega = J_qomega^T g_q and H_omegaomega = J_qomega^T H_qq J_qomega + sum_alpha (g_q)_alpha H_omegaomega^{q_alpha}.
std::pair<Vec3, Mat33> inertia_rotation_gradient_hessian(const Vec3& omega, const Vec4& q_n, const Vec4& q_nm1, const Vec3& omega_n, double dt, const Mat33& I_hat) {
    assert(std::abs(q_n.squaredNorm() - 1.0) < 1.0e-10 && "q_n must be a unit quaternion");
    assert(std::abs(q_nm1.squaredNorm() - 1.0) < 1.0e-10 && "q_nm1 must be a unit quaternion");
    const auto [g_q, H_qq] = quaternion_inertia_derivatives(omega, q_n, q_nm1, omega_n, dt, I_hat);
    const Mat43 J_qomega = dq_domega(q_n, omega, dt);
    const std::array<Mat33, 4> H_qomega = d2q_domega2(q_n, omega, dt);
    Vec3 gradient = Vec3::Zero();
    Mat33 hessian = Mat33::Zero();

    // (g_omega)_beta = (g_q)_alpha (J_qomega)_{alpha,beta}.
    for (int beta = 0; beta < 3; ++beta) {
        for (int alpha = 0; alpha < 4; ++alpha) {
            gradient[beta] += g_q[alpha] * J_qomega(alpha, beta);
        }
    }

    // (H_omegaomega)_{beta,kappa} = (J_qomega)_{alpha,beta} (H_qq)_{alpha,delta} (J_qomega)_{delta,kappa}  + (g_q)_alpha (H_q_alpha)_{beta,kappa}.
    for (int beta = 0; beta < 3; ++beta) {
        for (int kappa = 0; kappa < 3; ++kappa) {
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int delta = 0; delta < 4; ++delta) {
                    hessian(beta, kappa) += J_qomega(alpha, beta) * H_qq(alpha, delta) * J_qomega(delta, kappa);
                }
                hessian(beta, kappa) += g_q[alpha] * H_qomega[alpha](beta, kappa);
            }
        }
    }
    return {gradient, hessian};
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


Mat16 InertiaC4(const std::vector<Vec3>& R, const std::vector<double>& nodal_mass){
    // R material space position
    Mat16 IC4 = Mat16::Zero();
    Mat44 I_hat = Mat44::Zero();
    for(size_t i = 0; i < R.size(); i++){
        for(size_t alpha = 1; alpha < 4; alpha++){
            for(size_t beta = 1; beta < 4; beta++){
                I_hat(alpha,beta) += nodal_mass[i] * R[i][alpha-1] * R[i][beta-1];
            }
        }
    }

    for(size_t beta = 0; beta < 4; beta++){
        for(size_t sigma = 0; sigma < 4; sigma++){
            for(size_t epsilon = 0; epsilon < 4; epsilon++){
                for(size_t theta = 0; theta < 4; theta++){
                    for(size_t alpha = 1; alpha < 4; alpha++){
                        for(size_t delta = 1; delta < 4; delta++){
                            for(size_t rho = 1; rho < 4; rho++){
                                IC4(beta * 4 + sigma, epsilon * 4 + theta) +=
                                    I_hat(delta, rho)
                                    * QPT_QPT(alpha, beta, delta, epsilon)
                                    * QPT_QPT(alpha, sigma, rho, theta);
                            }
                        }
                    }
                }
            }
        }
    }

    return IC4;
}

double incremental_potential_energy(const Vec3& x_com, const Vec4& q, const Vec3& x_com_n,
                                    const Vec3& v_com_n, const Vec3& omega_n, const Vec4& q_n,
                                    const double dt, const double total_mass, const Mat16& IC4
                                    ){
                                        double IPE = double(0);
                                        Vec3 x_hat_com_n = x_com_n + dt * v_com_n;
                                        // linear part
                                        IPE += double(.5) * total_mass * (x_com - x_hat_com_n).squaredNorm();

                                        // angular part
                                        Vec4 q_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q);
                                        Vec4 q_n_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
                                        Vec3 angle = (double(-1)*dt) * omega_n;
                                        Vec4 omega_n_4d = Vec4(double(0), omega_n[0], omega_n[1], omega_n[2]);
                                        Vec4 q_nm1 = Rigid_Body::ALGEBRA::QuaternionMultiply(
                                            Rigid_Body::ALGEBRA::QuaternionFromVector(angle), q_n);
                                        Vec4 q_n_dot = double(.5) * Rigid_Body::ALGEBRA::QuaternionMultiply(omega_n_4d,q_nm1);
                                        Vec4 q_n_inv_dot = double(-1) * Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_inv, Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_dot,q_n_inv));

                                        for (size_t beta = 0; beta < 4; beta++){
                                            for(size_t sigma = 0; sigma < 4; sigma++){
                                                for(size_t epsilon = 0; epsilon < 4; epsilon++){
                                                    for(size_t theta = 0; theta < 4; theta++){
                                                        IPE += IC4(beta*4 + sigma, epsilon * 4 + theta) * q[beta] * q_inv[epsilon] *
                                                        (double(.5) * q[sigma] * q_inv[theta]
                                                        - q_n[sigma] * q_n_inv[theta]
                                                        - dt * q_n_dot[sigma] * q_n_inv[theta]
                                                        - dt * q_n[sigma] * q_n_inv_dot[theta]);
                                                    }
                                                }
                                            }
                                        }

                                        // angular part constant terms
                                        for (size_t beta = 0; beta < 4; beta++){
                                            for(size_t sigma = 0; sigma < 4; sigma++){
                                                for(size_t epsilon = 0; epsilon < 4; epsilon++){
                                                    for(size_t theta = 0; theta < 4; theta++){
                                                        IPE += IC4(beta*4 + sigma, epsilon * 4 + theta)*
                                                        (dt * q_n[beta] * q_n_inv[epsilon] * q_n_dot[sigma] * q_n_inv[theta]
                                                        + dt * q_n[beta] * q_n_inv_dot[epsilon] * q_n[sigma] * q_n_inv[theta]
                                                        + dt * dt * q_n_dot[beta] * q_n_inv[epsilon] * q_n[sigma] * q_n_inv_dot[theta]
                                                        + double(.5) * q_n[beta] * q_n_inv[epsilon] * q_n[sigma] * q_n_inv[theta]
                                                        + double(.5) * dt * dt * q_n_dot[beta] * q_n_inv[epsilon] * q_n_dot[sigma] * q_n_inv[theta]
                                                        + double(.5) * dt * dt * q_n[beta] * q_n_inv_dot[epsilon] * q_n[sigma] * q_n_inv_dot[theta]);
                                                    }
                                                }
                                            }
                                        }

                                        return IPE;
                                    }

Vec3 incremental_potential_translation_gradient(const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n, const double total_mass,const double dt){
    Vec3 g = Vec3::Zero();
    Vec3 x_hat_com_n = x_com_n + dt * v_com_n;
    g = total_mass * (x_com - x_hat_com_n);
    return g;
}

Mat33 incremental_potential_translation_hessian(const double total_mass){
    Mat33 H = total_mass * Mat33::Identity();
    return H;
}

Vec3 incremental_potential_orientation_gradient(const Vec4& q, const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, const Mat16& IC4, const double dt){
    Vec4 g_q = Vec4::Zero();

    Eigen::Matrix<double,4,3> dqdw = Rigid_Body::ALGEBRA::DqDw(dt, omega, q_n);
    Vec4 q_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q);
    Vec4 q_n_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec3 angle = (double(-1)*dt) * omega_n;
    Vec4 omega_n_4d = Vec4(double(0), omega_n[0], omega_n[1], omega_n[2]);
    Vec4 q_nm1 = Rigid_Body::ALGEBRA::QuaternionMultiply(
        Rigid_Body::ALGEBRA::QuaternionFromVector(angle), q_n);
    Vec4 q_n_dot = double(.5) * Rigid_Body::ALGEBRA::QuaternionMultiply(omega_n_4d,q_nm1);
    Vec4 q_n_inv_dot = double(-1) * Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_inv, Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_dot,q_n_inv));

    Mat44 Dq = Mat44::Identity(), Dq_inv;
    Dq_inv = -Dq;
    Dq_inv(0,0) = double(1); // Diag(Dq_inv) = [1,-1,-1,-1]

    for(size_t alpha = 0; alpha < 4; alpha++){
        for(size_t beta = 0; beta < 4; beta++){
            for(size_t sigma = 0; sigma < 4; sigma++){
                for(size_t epsilon = 0; epsilon < 4; epsilon++){
                    for(size_t theta = 0; theta < 4; theta++){
                        g_q[alpha] += IC4(beta*4 + sigma, epsilon*4 + theta) * (Dq(alpha,beta) * q_inv[epsilon] + q[beta] * Dq_inv(alpha,epsilon))
                        * (q[sigma] * q_inv[theta] - q_n[sigma] * q_n_inv[theta] - dt * q_n_dot[sigma] * q_n_inv[theta] - dt * q_n[sigma] * q_n_inv_dot[theta]);
                    }
                }
            }
        }
    }


    Vec3 g_w = dqdw.transpose() * g_q;

    return g_w;
}

void incremental_potential_orientation_gradient_hessian(const Vec4& q, const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, const Mat16& IC4, const double dt, Mat33& H_w, Vec3& g_w){
    g_w.setZero();
    H_w.setZero();

    Eigen::Matrix<double,4,3> dqdw = Rigid_Body::ALGEBRA::DqDw(dt, omega, q_n);
    std::array<Eigen::Matrix<double,4,3>,3> d2qdw2 = Rigid_Body::ALGEBRA::D2qDw2(dt,omega,q_n);
    Vec4 q_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q);
    Vec4 q_n_inv = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec3 angle = (double(-1)*dt) * omega_n;
    Vec4 omega_n_4d = Vec4(double(0), omega_n[0], omega_n[1], omega_n[2]);
    Vec4 q_nm1 = Rigid_Body::ALGEBRA::QuaternionMultiply(
        Rigid_Body::ALGEBRA::QuaternionFromVector(angle), q_n);
    Vec4 q_n_dot = double(.5) * Rigid_Body::ALGEBRA::QuaternionMultiply(omega_n_4d,q_nm1);
    Vec4 q_n_inv_dot = double(-1) * Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_inv, Rigid_Body::ALGEBRA::QuaternionMultiply(q_n_dot,q_n_inv));

    Mat44 Dq = Mat44::Identity(), Dq_inv;
    Dq_inv = -Dq;
    Dq_inv(0,0) = double(1); // Diag(Dq_inv) = [1,-1,-1,-1]

    // Gradient
    Vec4 g_q = Vec4::Zero();
    for(size_t alpha = 0; alpha < 4; alpha++){
        for(size_t beta = 0; beta < 4; beta++){
            for(size_t sigma = 0; sigma < 4; sigma++){
                for(size_t epsilon = 0; epsilon < 4; epsilon++){
                    for(size_t theta = 0; theta < 4; theta++){
                        g_q[alpha] += IC4(beta*4 + sigma, epsilon*4 + theta) * (Dq(alpha,beta) * q_inv[epsilon] + q[beta] * Dq_inv(alpha,epsilon))
                        * (q[sigma] * q_inv[theta] - q_n[sigma] * q_n_inv[theta] - dt * q_n_dot[sigma] * q_n_inv[theta] - dt * q_n[sigma] * q_n_inv_dot[theta]);
                    }
                }
            }
        }
    }
    g_w = dqdw.transpose() * g_q;

    // Hessian
    Mat44 H_q = Mat44::Zero();
    for(size_t alpha = 0; alpha < 4; alpha++){
        for(size_t gamma = 0; gamma < 4; gamma++){
            for(size_t beta = 0; beta < 4; beta++){
                for(size_t sigma = 0; sigma < 4; sigma++){
                    for(size_t epsilon = 0; epsilon < 4; epsilon++){
                        for(size_t theta = 0; theta < 4; theta++){
                            H_q(alpha, gamma) += IC4(beta*4 + sigma, epsilon*4 + theta)
                            * (
                                (Dq(alpha,beta) * Dq_inv(gamma,epsilon) + Dq(gamma,beta) * Dq_inv(alpha,epsilon))
                                * (q[sigma] * q_inv[theta] - q_n[sigma] * q_n_inv[theta] - dt * q_n_dot[sigma] * q_n_inv(theta) - dt * q_n[sigma] * q_n_inv_dot[theta])
                                + (Dq(alpha,beta) * q_inv[epsilon] + q[beta] * Dq_inv(alpha,epsilon))
                                * (Dq(gamma,sigma) * q_inv[theta] + q[sigma] * Dq_inv(gamma,theta))
                            );
                        }
                    }
                }
            }
        }
    }

    H_w = dqdw.transpose() * H_q * dqdw;

    for(size_t alpha = 0; alpha < 3; alpha++){
        for(size_t gamma = 0; gamma < 3; gamma++){
            for(size_t beta = 0; beta < 4; beta++){
                H_w(alpha,gamma) += d2qdw2[gamma](beta,alpha) * g_q[beta];
            }
        }
    }
}

double gravitational_potential(const Vec3& x_com, const double total_mass, const double gravity, const double dt){
    return dt * dt * total_mass * gravity * x_com[1];
}

Vec3 gravitational_potential_gradient(const double total_mass, const double gravity, const double dt){
    return {double(0),dt*dt*total_mass*gravity,double(0)};
}

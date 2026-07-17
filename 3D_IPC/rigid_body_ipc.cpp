#include "rigid_body_ipc.h"
#include <cassert>
#include <cmath>

namespace {

constexpr double kSmallAngleThreshold = 1.0e-4;

// r_hat_p = r_p(q_n) + dt r_dot_p^n, where r_p(q_n) = vec(q_n * (0, R_p) * (q_n)^*) and
// r_dot_p^n = vec(q_dot_n * (0, R_p) * (q_n)^* + q_n * (0, R_p) * (q_dot_n)^*).
Vec3 build_rhat(const Vec3& R_p, const Vec4& q_n, const Vec4& q_dot_n, double dt) {
    const Vec4 R_p_quaternion(0.0, R_p[0], R_p[1], R_p[2]);
    const Vec4 first_term = quaternion_multiply(quaternion_multiply(q_dot_n, R_p_quaternion), quaternion_conjugate(q_n));
    const Vec4 second_term = quaternion_multiply(quaternion_multiply(q_n, R_p_quaternion), quaternion_conjugate(q_dot_n));
    const Vec3 r_dot_p_n = (first_term + second_term).tail<3>();
    return quaternion_rotate(q_n, R_p) + dt * r_dot_p_n;
}

// (J_p)_{i,0} = 2 q_s R_{p,i} + 2 epsilon_{ijk} q_j R_{p,k} and (J_p)_{i,l} = 2 (-q_l R_{p,i} + delta_{il} q_j R_{p,j} + q_i R_{p,l} + q_s epsilon_{ilk} R_{p,k}).
Mat34 build_Jp(const Vec3& R_p, const Vec4& q) {
    const double q_s = q[0];
    const Vec3 q_v(q[1], q[2], q[3]);
    Mat34 J_p = Mat34::Zero();

    for (int i = 0; i < 3; ++i) {
        J_p(i, 0) = 2.0 * q_s * R_p[i];
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                J_p(i, 0) += 2.0 * levi_civita(i, j, k) * q_v[j] * R_p[k];
            }
        }

        for (int l = 0; l < 3; ++l) {
            J_p(i, l + 1) = 2.0 * (-q_v[l] * R_p[i] + q_v[i] * R_p[l]);
            for (int j = 0; j < 3; ++j) {
                J_p(i, l + 1) += 2.0 * kronecker_delta(i, l) * q_v[j] * R_p[j];
            }
            for (int k = 0; k < 3; ++k) {
                J_p(i, l + 1) += 2.0 * q_s * levi_civita(i, l, k) * R_p[k];
            }
        }
    }
    return J_p;
}

// B_p[i] is the 4x4 matrix with entries (B_p)_{i,alpha,beta} = partial^2 b_{p,i} / (partial q_alpha partial q_beta).
std::array<Mat44, 3> build_Bp(const Vec3& R_p) {
    std::array<Mat44, 3> B_p = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};

    for (int i = 0; i < 3; ++i) {
        B_p[i](0, 0) = 2.0 * R_p[i];

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                const double value = 2.0 * levi_civita(i, l, k) * R_p[k];
                B_p[i](0, l + 1) += value;
                B_p[i](l + 1, 0) += value;
            }
        }

        for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
                B_p[i](l + 1, m + 1) = 2.0 * (-kronecker_delta(l, m) * R_p[i] + kronecker_delta(i, l) * R_p[m] + kronecker_delta(i, m) * R_p[l]);
            }
        }
    }
    return B_p;
}

// Assemble g_q and H_qq from b_p = r_p(q(omega)) - r_hat_p, where r_p(q) = vec(q * (0, R_p) * q^*) and r_hat_p = r_p(q_n) + dt r_dot_p^n.
std::pair<Vec4, Mat44> quaternion_inertia_derivatives(const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, double dt, const std::vector<double>& masses, const std::vector<Vec3>& R_p) {
    Vec4 g_q = Vec4::Zero(); // g_q = partial E_q / partial q.
    Mat44 H_qq = Mat44::Zero(); // H_qq = partial^2 E_q / partial q^2.
    // q = q(omega) is the candidate unit quaternion at time n+1.
    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt);
    // q_dot_n = 1/2 (0, omega_n) * q_n is the quaternion velocity in the note.
    const Vec4 q_dot_n = quaternion_time_derivative(q_n, omega_n);

    for (std::size_t p = 0; p < R_p.size(); ++p) {
        // r_p = vec(q * (0, R_p) * q^*)
        const Vec3 r_p = quaternion_rotate(q, R_p[p]);
        // r_hat_p = r_p(q_n) + dt r_dot_p^n is fixed
        const Vec3 r_hat_p = build_rhat(R_p[p], q_n, q_dot_n, dt);
        // b_p = r_p - r_hat_p
        const Vec3 b_p = r_p - r_hat_p;
        // J_p = partial b_p / partial q is assembled directly from the indexed formula above.
        const Mat34 J_p = build_Jp(R_p[p], q);

        // (g_q)_alpha = sum_p m_p b_{p,i} (J_p)_{i,alpha}.
        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int i = 0; i < 3; ++i) {
                g_q[alpha] += masses[p] * b_p[i] * J_p(i, alpha);
            }
        }

        // B_p[i] = partial^2 b_{p,i} / partial q^2 is assembled directly from the indexed piecewise formula above.
        const std::array<Mat44, 3> B_p = build_Bp(R_p[p]);

        // (H_qq)_{alpha,beta} = sum_p m_p [(J_p)_{i,alpha} (J_p)_{i,beta}  + b_{p,i} (B_p)_{i,alpha,beta} ].
        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int beta = 0; beta < 4; ++beta) {
                for (int i = 0; i < 3; ++i) {
                    H_qq(alpha, beta) += masses[p] * (J_p(i, alpha) * J_p(i, beta) + b_p[i] * B_p[i](alpha, beta));
                }
            }
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

// E_in = 1/2 m ||x_com - x_com_n - dt v_com_n||^2 + 1/2 sum_p m_p ||r_p(q(omega)) - r_hat_p||^2.
double incremental_potential_energy(const Vec3& x_com, const Vec3& omega, const Vec3& x_com_n, const Vec3& v_com_n, const Vec4& q_n, const Vec3& omega_n, double dt, double total_mass, const std::vector<double>& masses, const std::vector<Vec3>& R_p) {
    assert(std::abs(q_n.squaredNorm() - 1.0) < 1.0e-10 && "q_n must be a unit quaternion");
    // a = x_com - x_hat_com = x_com - x_com_n - dt v_com_n is the translational inertial residual.
    const Vec3 a = x_com - x_com_n - dt * v_com_n;
    double energy = 0.0;

    // E_x = (1/2) m a_i a_i.
    for (int i = 0; i < 3; ++i)
        energy += 0.5 * total_mass * a[i] * a[i];

    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt); // Candidate q^{n+1} = q(omega).
    const Vec4 q_dot_n = quaternion_time_derivative(q_n, omega_n); // q_dot_n = 1/2 (0, omega_n) * q_n.

    // E_q = (1/2) sum_p m_p b_{p,i} b_{p,i}.
    for (std::size_t p = 0; p < R_p.size(); ++p) {
        // r_p, r_hat_p, and b_p are the candidate offset, predicted offset, and their residual.
        const Vec3 r_p = quaternion_rotate(q, R_p[p]);
        const Vec3 r_hat_p = build_rhat(R_p[p], q_n, q_dot_n, dt);
        const Vec3 b_p = r_p - r_hat_p;
        for (int i = 0; i < 3; ++i)
            energy += 0.5 * masses[p] * b_p[i] * b_p[i];
    }

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
std::pair<Vec3, Mat33> inertia_rotation_gradient_hessian(const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, double dt, const std::vector<double>& masses, const std::vector<Vec3>& R_p) {
    assert(std::abs(q_n.squaredNorm() - 1.0) < 1.0e-10 && "q_n must be a unit quaternion");
    // H_qq is exact and includes the residual-curvature B_p term.
    const auto [g_q, H_qq] = quaternion_inertia_derivatives(omega, q_n, omega_n, dt, masses, R_p);
    // J_qomega = partial q / partial omega maps the ambient quaternion derivatives to omega coordinates.
    const Mat43 J_qomega = dq_domega(q_n, omega, dt);
    // H_qomega[alpha] = partial^2 q_alpha / partial omega^2 supplies the second-order chain-rule term.
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

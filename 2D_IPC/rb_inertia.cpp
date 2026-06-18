#include "rb_inertia.h"

#include <cmath>

namespace {

Mat2 R(double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    return {c, -s, s, c};
}

Mat2 C() {
    return {0.0, -1.0, 1.0, 0.0};
}

}  // namespace

Mat2 inertia_body_tensor(const Vec& U, const std::vector<double>& masses) {
    Mat2 inertia_tensor{0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < U.size(); ++i) {
        for (int gamma = 0; gamma < 2; ++gamma) {
            for (int beta = 0; beta < 2; ++beta) {
                const double value = mat_entry(inertia_tensor, gamma, beta) + masses[i] * vec_entry(U[i], gamma) * vec_entry(U[i], beta);
                set_mat_entry(inertia_tensor, gamma, beta, value);
            }
        }
    }
    return inertia_tensor;
}

Vec2 inertia_translation_gradient(const Vec2& y, const Vec2& y_n, const Vec2& vhat_n, double dt, double m_total) {
    Vec2 grad{0.0, 0.0};
    for (int alpha = 0; alpha < 2; ++alpha) {
        const double value = m_total * (vec_entry(y, alpha) - vec_entry(y_n, alpha) - dt * vec_entry(vhat_n, alpha));
        set_vec_entry(grad, alpha, value);
    }
    return grad;
}

Mat2 inertia_translation_hessian(double m_total) {
    Mat2 hessian{0.0, 0.0, 0.0, 0.0};
    for (int xi = 0; xi < 2; ++xi) {
        for (int eta = 0; eta < 2; ++eta) {
            set_mat_entry(hessian, xi, eta, m_total * kronecker_delta(eta, xi));
        }
    }
    return hessian;
}

double inertia_rotation_gradient(double theta, double theta_n, double omega_n, const Mat2& inertia_tensor, double dt) {
    const Mat2 R_n = R(theta_n);
    const Mat2 R_theta = R(theta);
    const Mat2 C_mat = C();

    double first_sum = 0.0;
    double second_sum = 0.0;
    for (int alpha = 0; alpha < 2; ++alpha) {
        for (int gamma = 0; gamma < 2; ++gamma) {
            for (int lambda = 0; lambda < 2; ++lambda) {
                for (int beta = 0; beta < 2; ++beta) {
                    first_sum += mat_entry(R_n, alpha, gamma) * mat_entry(C_mat, alpha, lambda) * mat_entry(R_theta, lambda, beta) * mat_entry(inertia_tensor, gamma, beta);
                }
            }

            for (int beta = 0; beta < 2; ++beta) {
                second_sum += mat_entry(R_n, alpha, gamma) * mat_entry(R_theta, alpha, beta) * mat_entry(inertia_tensor, gamma, beta);
            }
        }
    }

    return -first_sum - dt * omega_n * second_sum;
}

double inertia_rotation_hessian(double theta, double theta_n, double omega_n, const Mat2& inertia_tensor, double dt) {
    const Mat2 R_n = R(theta_n);
    const Mat2 R_theta = R(theta);
    const Mat2 C_mat = C();

    double first_sum = 0.0;
    double second_sum = 0.0;
    for (int alpha = 0; alpha < 2; ++alpha) {
        for (int gamma = 0; gamma < 2; ++gamma) {
            for (int beta = 0; beta < 2; ++beta) {
                first_sum += mat_entry(R_n, alpha, gamma) * mat_entry(R_theta, alpha, beta) * mat_entry(inertia_tensor, gamma, beta);
            }

            for (int kappa = 0; kappa < 2; ++kappa) {
                for (int beta = 0; beta < 2; ++beta) {
                    second_sum += mat_entry(C_mat, alpha, kappa) * mat_entry(R_n, kappa, gamma) *  mat_entry(R_theta, alpha, beta) *  mat_entry(inertia_tensor, gamma, beta);
                }
            }
        }
    }

    return first_sum + dt * omega_n * second_sum;
}

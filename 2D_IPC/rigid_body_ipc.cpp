#include "rigid_body_ipc.h"

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

Vec2 world_space_position(const Vec2& X, const Vec2& x_com, const double theta){
    Vec2 x = x_com;

    Vec2 U{double(0),double(0)}, r{double(0),double(0)};
    for(size_t alpha = 0; alpha < 2; alpha++){
        set_vec_entry(U,alpha,vec_entry(X,alpha));
    }
    r = matvec(R(theta),U);
    for(size_t alpha = 0; alpha < 2; alpha++){
        set_vec_entry(x,alpha, vec_entry(x,alpha) + vec_entry(r,alpha));
    }

    return x;
}

Vec2 material_space_position(const Vec2& x, const Vec2& x_com, const double theta){
    // U = X - Y. Assume Y is at the origin.
    // Takes in a world space particle x and transform into material space position X
    Vec2 r{double(0),double(0)};
    for(size_t alpha = 0; alpha < 2; alpha++){
        set_vec_entry(r,alpha,vec_entry(x,alpha)-vec_entry(x_com,alpha));
    }

    Vec2 X = matvec(R(-theta),r);
    return X;
}


double incremental_potential_energy(const Vec2& y, const double theta, const Vec2& y_n, const double theta_n, const Vec2& vhat_n, const double omega_n, const double dt, const double m_total, const Mat2& I){
    double result = double(0);

    // Translation
    for(size_t alpha = 0; alpha < 2; alpha++){
        result += mat_entry(I,alpha,alpha) + (m_total/double(2)) * vec_entry(y,alpha) * vec_entry(y,alpha)
                - m_total * vec_entry(y_n,alpha) * vec_entry(y,alpha) - m_total * dt * vec_entry(vhat_n,alpha) * vec_entry(y, alpha)
                + (m_total/double(2)) * vec_entry(y_n,alpha) * vec_entry(y_n,alpha) + (dt * m_total) * vec_entry(y_n,alpha) * vec_entry(vhat_n,alpha)
                + (dt * dt * double(.5)) * omega_n * omega_n * mat_entry(I,alpha,alpha) + dt * dt * double(.5) * m_total * vec_entry(vhat_n,alpha) * vec_entry(vhat_n,alpha);
    }

    // Orientation
    const Mat2 R_theta = R(theta);
    const Mat2 R_n = R(theta_n);
    const Mat2 C_mat = C();
    for(size_t alpha = 0; alpha < 2; alpha++){
        for(size_t beta = 0; beta < 2; beta++){
            for(size_t delta = 0; delta < 2; delta++){
                result -= mat_entry(R_theta,alpha,beta) * mat_entry(I,beta,delta) * mat_entry(R_n,alpha,delta);
            }
        }
    }

    for(size_t alpha = 0; alpha < 2; alpha++){
        for(size_t beta = 0; beta < 2; beta++){
            for(size_t delta = 0; delta < 2; delta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    result -= dt * omega_n * mat_entry(I,beta,gamma) * mat_entry(R_theta, alpha, beta) * mat_entry(C_mat,alpha,delta) * mat_entry(R_n, delta,gamma);
                }
            }
        }
    }


    return result;
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

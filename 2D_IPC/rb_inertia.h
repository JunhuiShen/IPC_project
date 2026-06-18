#pragma once

#include "ipc_math.h"

// Translation gradient from the reduced rigid-body inertia:
// grad_y I_b = M_b * (y - (y_n + dt * vhat_n)).
Vec2 inertia_translation_gradient(const Vec2& y, const Vec2& y_n, const Vec2& vhat_n, double dt, double m_total);

// Translation Hessian from the reduced rigid-body inertia:
// H_{xi eta} = partial^2 I_b / (partial y_xi partial y_eta) = M_b * delta_{eta xi}.
Mat2 inertia_translation_hessian(double m_total);

// Rotation gradient from the reduced rigid-body inertia:
// grad_theta_Ib =
//     -sum_i m_i R_{alpha gamma}(theta_n) C_{alpha lambda}
//        R_{lambda beta}(theta) U_{i gamma} U_{i beta}
//     -dt * omega_n * sum_i m_i R_{alpha gamma}(theta_n)
//        R_{alpha beta}(theta) U_{i gamma} U_{i beta}.
double inertia_rotation_gradient(double theta, double theta_n, double omega_n, const Vec& U, const std::vector<double>& masses, double dt);

// Rotation Hessian from the reduced rigid-body inertia:
// H_theta_theta =
//     sum_i m_i R_{alpha gamma}(theta_n) R_{alpha beta}(theta)
//        U_{i gamma} U_{i beta}
//     +dt * omega_n * sum_i m_i C_{alpha kappa} R_{kappa gamma}(theta_n)
//        R_{alpha beta}(theta) U_{i gamma} U_{i beta}.
double inertia_rotation_hessian(double theta, double theta_n, double omega_n, const Vec& U, const std::vector<double>& masses, double dt);

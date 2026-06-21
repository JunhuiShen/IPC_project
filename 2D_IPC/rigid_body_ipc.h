#pragma once

#include "ipc_math.h"

// Translation gradient from the reduced rigid-body inertia:
// grad_y I_b = M_b * (y - (y_n + dt * vhat_n)).
Vec2 inertia_translation_gradient(const Vec2& y, const Vec2& y_n, const Vec2& vhat_n, double dt, double m_total);

// Translation Hessian from the reduced rigid-body inertia:
// H_{xi eta} = partial^2 I_b / (partial y_xi partial y_eta) = M_b * delta_{eta xi}.
Mat2 inertia_translation_hessian(double m_total);

// Body-space inertia tensor
// inertia_tensor_{gamma beta} = sum_i m_i U_{i gamma} U_{i beta}.
Mat2 inertia_body_tensor(const Vec& U, const std::vector<double>& masses);


// U = X - Y. Assume Y is at the origin.
// Takes in a material space particle X and transform into world space position x
Vec2 world_space_position(const Vec2& X, const Vec2& x_com, const double theta);

// U = X - Y. Assume Y is at the origin.
// Takes in a world space particle x and transform into material space position X
Vec2 material_space_position(const Vec2& x, const Vec2& x_com, const double theta);

// Take in rigid body stuff 
// (world space particles, v_com_input, theta_input, omega_input, total_mass)
// compute center of mass, inertia tensor, material space particles
// and store them into ref_mesh and deformed state
void create_rigid_body(const Vec& x, const Vec2& v_com_input, const double theta_input, const double omega_input, const double total_mass, Vec2& x_com, Vec2& v_com, double& theta, double& omega, Mat2& I, Vec& ref_positions);

// Rotation gradient from the reduced rigid-body inertia:
// grad_theta_Ib =
//     -R_{alpha gamma}(theta_n) C_{alpha lambda}
//        R_{lambda beta}(theta) inertia_tensor_{gamma beta}
//     -dt * omega_n * R_{alpha gamma}(theta_n)
//        R_{alpha beta}(theta) inertia_tensor_{gamma beta}.
double inertia_rotation_gradient(double theta, double theta_n, double omega_n, const Mat2& inertia_tensor, double dt);

// Rotation Hessian from the reduced rigid-body inertia:
// H_theta_theta =
//     R_{alpha gamma}(theta_n) R_{alpha beta}(theta) inertia_tensor_{gamma beta}
//     +dt * omega_n * C_{alpha kappa} R_{kappa gamma}(theta_n)
//        R_{alpha beta}(theta) inertia_tensor_{gamma beta}.
double inertia_rotation_hessian(double theta, double theta_n, double omega_n, const Mat2& inertia_tensor, double dt);

double incremental_potential_energy(const Vec2& y, double theta,
                                    const Vec2& y_n, double theta_n, const Vec2& vhat_n,
                                    double omega_n, double dt, double m_total,
                                    const Mat2& I);


double gravitational_potential(const Vec2& y, double m_total, const double gravity, const double dt);

Vec2 gravitational_gradient(const double m_total, const double gravity, const double dt);


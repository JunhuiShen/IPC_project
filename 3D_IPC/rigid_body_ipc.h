#pragma once

#include "quaternion_math.h"
#include <array>

// Quaternion exponential E(omega) = exp((dt / 2) * omega) and derivatives
Vec4 exp(const Vec3& omega, double dt);
Mat43 dexp_domega(const Vec3& omega, double dt);
std::array<Mat33, 4> d2exp_domega2(const Vec3& omega, double dt);

// Directly evaluate q(omega) = E(omega) * q0 and its Jacobian
Vec4 quaternion_from_angular_velocity(const Vec4& q0, const Vec3& omega, double dt);
Mat43 dq_domega(const Vec4& q0, const Vec3& omega, double dt);
std::array<Mat33, 4> d2q_domega2(const Vec4& q0, const Vec3& omega, double dt);

// X_centered is a fixed body-space position measured from the center of mass.
// x = x_com + R(q) X_centered with q(omega) = E(omega, dt) * q0.
// X_centered = R(q)^T (x - x_com) with q(omega) = E(omega, dt) * q0.
Vec3 world_space_position(const Vec3& X_centered, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt);
Vec3 material_space_position(const Vec3& x, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt);

// Node Jacobian and coordinate Hessians with respect to quaternion coordinates
Mat34 dx_dq(const Vec3& X_centered, const Vec4& quat);
std::array<Mat44, 3> d2x_dq2(const Vec3& X_centered);

// Exact omega-coordinate derivatives of x(t, omega) = t + R(q(omega)) X_centered
Mat33 dx_domega(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt);
std::array<Mat33, 3> d2x_domega2(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt);

// Per-node translation and angular-velocity gradient contributions.
Vec3 rigid_node_translation_gradient(const Vec3& gx);
Vec3 rigid_node_omega_gradient(const Vec3& gx, const Mat33& dx_domega);

// Per-node translation and angular-velocity Hessian contributions.
Mat33 rigid_node_translation_hessian(const Mat33& Hx);
Mat33 rigid_node_omega_hessian(const Vec3& gx, const Mat33& Hx, const Mat33& dx_domega, const std::array<Mat33, 3>& d2x_domega2);

#pragma once

#include "quaternion_math.h"
#include <array>
#include <utility>
#include <vector>

struct DeformedState;
struct RefMesh;

// Quaternion exponential E(omega) = exp((dt / 2) * omega) and derivatives
Vec4 exp(const Vec3& omega, double dt);
Mat43 dexp_domega(const Vec3& omega, double dt);
std::array<Mat33, 4> d2exp_domega2(const Vec3& omega, double dt);

// Directly evaluate q(omega) = E(omega) * q0 and its Jacobian
Vec4 quaternion_from_angular_velocity(const Vec4& q0, const Vec3& omega, double dt);
Mat43 dq_domega(const Vec4& q0, const Vec3& omega, double dt);
std::array<Mat33, 4> d2q_domega2(const Vec4& q0, const Vec3& omega, double dt);

// X_centered is a fixed body-space position measured from the center of mass.
// Direct transforms using an already-evaluated orientation quaternion.
Vec3 world_space_position(
    const Vec3& X, const Vec3& x_com, const Vec4& orientation);
Vec3 material_space_position(
    const Vec3& x, const Vec3& x_com, const Vec4& orientation);

// x = x_com + vec(q * (0, X_centered) * q^*) with q(omega) = E(omega, dt) * q0.
// X_centered = vec(q^* * (0, x - x_com) * q) with q(omega) = E(omega, dt) * q0.
Vec3 world_space_position(const Vec3& X_centered, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt);
Vec3 material_space_position(const Vec3& x, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt);

// Node Jacobian and coordinate Hessians with respect to quaternion coordinates
Mat34 dx_dq(const Vec3& X_centered, const Vec4& quat);
std::array<Mat44, 3> d2x_dq2(const Vec3& X_centered);

// Precompute I_hat = sum_p m_p R_p R_p^T once for each rigid body.
// R_p[p] = X_p - X_com is the fixed body-reference COM offset of point p.
Mat33 body_second_moment(const std::vector<double>& masses, const std::vector<Vec3>& R_p);

// Register one rigid body from its particles' world-space positions. The
// particles are appended to state.deformed_positions and receive equal mass
// totaling total_mass.
// The input orientation and omega use the scalar-first quaternion and
// world-space angular-velocity conventions used by this file. Returns the
// newly assigned rigid-body index.
int create_rigid_body(
    const std::vector<Vec3>& x,
    const Vec3& v_com_input, const Vec4& orientation_input,
    const Vec3& omega_input, double total_mass,
    RefMesh& ref_mesh, DeformedState& state);

// Rigid-body inertial incremental potential aggregated with I_hat.
// The candidate quaternion is q(omega) = exp((dt / 2) * omega) * q_n.
// The fixed predictor reconstructs q_nm1 = exp((-dt / 2) * omega_n) * q_n internally.
double incremental_potential_energy(const Vec3& x_com, const Vec3& omega, const Vec3& x_com_n, const Vec3& v_com_n, const Vec4& q_n, const Vec3& omega_n, double dt, double total_mass, const Mat33& I_hat);

Vec3 inertia_translation_gradient(const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n, double dt, double total_mass);

Mat33 inertia_translation_hessian(double total_mass);

// Exact gradient and Hessian of the rotational inertial term with respect to omega.
std::pair<Vec3, Mat33> inertia_rotation_gradient_hessian(const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, double dt, const Mat33& I_hat);

// Exact omega-coordinate derivatives of x(t, omega) = t + R(q(omega)) X_centered
Mat33 dx_domega(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt);
std::array<Mat33, 3> d2x_domega2(const Vec3& X_centered, const Vec4& q0, const Vec3& omega, double dt);

// Per-node translation and angular-velocity gradient contributions.
Vec3 rigid_node_translation_gradient(const Vec3& gx);
Vec3 rigid_node_omega_gradient(const Vec3& gx, const Mat33& dx_domega);

// Per-node translation and angular-velocity Hessian contributions.
Mat33 rigid_node_translation_hessian(const Mat33& Hx);
Mat33 rigid_node_omega_hessian(const Vec3& gx, const Mat33& Hx, const Mat33& dx_domega, const std::array<Mat33, 3>& d2x_domega2);

#if 0
// Traditional rigid-body inertial potential using the fourth-order
// quaternion inertia tensor. For omega derivatives, q must satisfy
// q = QuaternionFromVector(dt * omega) * q_n.
Mat16 InertiaC4(const std::vector<Vec3>& R, const std::vector<double>& nodal_mass);

double incremental_potential_energy(
    const Vec3& x_com, const Vec4& q,
    const Vec3& x_com_n, const Vec3& v_com_n,
    const Vec3& omega_n, const Vec4& q_n,
    double dt, double total_mass, const Mat16& IC4);

Vec3 incremental_potential_translation_gradient(
    const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n,
    double total_mass, double dt);

Mat33 incremental_potential_translation_hessian(double total_mass);

Vec3 incremental_potential_orientation_gradient(
    const Vec4& q, const Vec3& omega,
    const Vec4& q_n, const Vec3& omega_n,
    const Mat16& IC4, double dt);

void incremental_potential_orientation_gradient_hessian(
    const Vec4& q, const Vec3& omega,
    const Vec4& q_n, const Vec3& omega_n,
    const Mat16& IC4, double dt,
    Mat33& H_w, Vec3& g_w);
#endif


double gravitational_potential(
    const Vec3& x_com, const double total_mass, const double gravity, const double dt);

Vec3 gravitational_potential_gradient(
    const double total_mass, const double gravity, const double dt);

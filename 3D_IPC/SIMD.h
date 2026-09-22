#pragma once

#include "physics.h"

// Element lanes are independent triangle/hinge contributions. Point terms pack
// xyz components. The caller owns accumulation and scheduling.
namespace ipc_simd {

// V2's organizer gathers AoS entries before calling these kernels. Each call
// transposes only a local tile and returns private AoS contributions. They do
// not read mesh indices or scatter into shared vertices. Outputs include rest
// area/coefficient but not dt^2, which is applied during ordered accumulation.
inline constexpr std::size_t tile_width = 8;
const char* tile_backend_name();
void corotated_derivatives_tile(
    const Vec3* positions, const Mat22* dm_inverse, const double* areas,
    const Vec2* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* hessians);
void bending_derivatives_tile(
    const Vec3* positions, const int* active_nodes, const double* coefficients,
    const double* rest_angles, std::size_t count, double stiffness,
    Vec3* gradients, Mat33* hessians);

int lane_width();
const char* backend_name();

// Batch evaluation of the same packed atan2 kernel used by hinge bending.
// Finite inputs stay in SIMD; axes, signed zero, infinities and NaNs follow
// the std::atan2 value semantics.
void atan2_batch(const double* y, const double* x, double* angles, std::size_t count);

// Add inertia, gravity and an optional pin spring using packed xyz arithmetic.
void accumulate_point_terms(
    double mass, const Vec3& x, const Vec3& xhat, const Vec3& gravity,
    const Vec3* pin_target, double kpin, double dt2,
    Vec3& gradient, Mat33& hessian);

// Add corotated elasticity to existing local accumulators in incident order.
// This computes the same corotated gradient and exact self Hessian as the
// scalar assembly; contact and the local solve remain with the caller.
void accumulated_corotated_elasticity(
    const RefMesh& mesh, const std::vector<Vec3>& positions,
    const IncidentTriangles& incident,
    const std::vector<ShapeGrads>* rest_shape_grads,
    double mu, double lambda, double dt2, Vec3& gradient, Mat33& hessian);

// Add the existing hinge gradient and PSD Gauss-Newton self Hessian in incident
// order, with one hinge per SIMD lane (including its signed dihedral atan2).
void accumulate_bending(
    const RefMesh& mesh, const std::vector<Vec3>& positions,
    const std::vector<std::pair<int, int>>& incident,
    double kB, double dt2, Vec3& gradient, Mat33& hessian);

} // namespace ipc_simd

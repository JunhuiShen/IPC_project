#pragma once

#include "physics.h"

// Element lanes are independent incident triangles/hinges for one active
// vertex. Point terms pack xyz components. The caller owns g/H and scheduling.
namespace ipc_simd {

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

// Add membrane elasticity to existing local accumulators in incident order.
// This computes the same corotated gradient and exact self Hessian as the
// scalar assembly; contact and the local solve remain with the caller.
void accumulate_membrane(
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

#pragma once

#include "barrier_energy.h"
#include "sdf_penalty_energy.h"

#include <array>
#include <utility>

// Geometry and normal-force data frozen at a contact evaluation point. For
// mesh contacts, the weights are signed closest-feature weights and sum to
// zero. An SDF contact has one dynamic role with weights[0] = 1.
struct FrozenFrictionContact {
    std::array<double, 4> weights{{0.0, 0.0, 0.0, 0.0}};
    Vec3 normal = Vec3::Zero();
    Mat33 projector = Mat33::Zero();
    Vec3 tangential_displacement = Vec3::Zero();
    double normal_force = 0.0;
    double eps_u = 0.0;
    bool active = false;
};

// Position ordering is the same as the barrier kernels:
// node--triangle: [node, triangle vertex 1, vertex 2, vertex 3]
// segment--segment: [segment 1 endpoint 1, endpoint 2,
//                    segment 2 endpoint 1, endpoint 2]
FrozenFrictionContact make_node_triangle_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        double d_hat, double k_barrier, double dt, double eps_v,
        double eps = 1.0e-12,
        const NodeTriangleDistanceResult* precomputed_dr = nullptr);

FrozenFrictionContact make_node_triangle_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const NodeTriangleContactEvaluation& evaluation,
        double dt, double eps_v);

FrozenFrictionContact make_segment_segment_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        double d_hat, double k_barrier, double dt, double eps_v,
        double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

FrozenFrictionContact make_segment_segment_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const SegmentSegmentContactEvaluation& evaluation,
        double dt, double eps_v);

// One particle against an analytic SDF. The SDF evaluation must be taken at
// current_position and carries the prescribed previous/current material poses.
// The corresponding obstacle displacement is evaluated lazily here so normal
// SDF contact does not inspect material motion when friction is disabled. The
// SDF penalty supplies the (unscaled) normal force; frozen_friction_* applies
// the surrounding dt^2 scaling.
FrozenFrictionContact make_sdf_frozen_friction_contact(
        const Vec3& current_position,
        const Vec3& previous_position,
        const SDFEvaluation& sdf,
        double k_sdf, double eps_sdf, double dt, double eps_v,
        double eps = 1.0e-12,
        const Vec3* precomputed_penalty_gradient = nullptr);

// Smooth approximation of ||u|| and f_1(||u||) / ||u||, respectively.
double smooth_friction_potential(double slip, double eps_u);
double smooth_friction_mollifier_over_slip(double slip, double eps_u);

// Incremental-potential contribution. dt2 is supplied explicitly because the
// surrounding objectives already control time-step scaling.
double frozen_friction_energy(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2);

// Derivatives with respect to the common relative displacement. The Hessian
// is the frozen, positive-semidefinite approximation used by block solvers.
Vec3 frozen_friction_relative_gradient(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2);

Mat33 frozen_friction_relative_hessian(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2);

// Computes the common derivative scale once when both relative derivatives
// are needed.
std::pair<Vec3, Mat33> frozen_friction_relative_gradient_and_hessian(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2);

Vec3 frozen_friction_role_gradient(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2);

Mat33 frozen_friction_role_hessian(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2);

std::pair<Vec3, Mat33> frozen_friction_role_gradient_and_hessian(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2);

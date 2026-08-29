#pragma once

#include "IPC_math.h"
#include "rigid_body_ipc.h"

//  SDF-based penalty energy

//  Heaviside and its derivative with respect to z.
double sdf_heaviside(double z, double eps);

double sdf_heaviside_gradient(double z, double eps);

// A material pose maps a collider-local material point to world space:
//     x_world = rotation * x_material + translation.
// Geometry remains stored directly in world space in the primitive structs
// below. These poses only supply the otherwise-underdetermined material motion
// needed by friction (for example, spin of a geometrically unchanged sphere or
// tangential translation of an infinite plane).
struct SDFMaterialPose {
    Mat33 rotation = Mat33::Identity();
    Vec3 translation = Vec3::Zero();
};

// For a substep n -> n+1, `previous` and `current` are the material-to-world
// poses at its beginning and end. Equal poses mean a stationary material
// surface. Keeping both transforms explicit also makes prescribed motion
// reconstructible from absolute time after a restart.
struct SDFMaterialMotion {
    SDFMaterialPose previous;
    SDFMaterialPose current;
};

struct SDFEvaluation {
    double phi;
    Vec3   grad_phi;
    Mat33  hess_phi;
    // Closest point on the current analytic surface. Material motion is copied
    // without being inspected; only an enabled friction term maps this point
    // through the poses and validates them.
    Vec3 surface_point;
    SDFMaterialMotion material_motion;

    SDFEvaluation()
        : phi(0.0), grad_phi(Vec3::Zero()), hess_phi(Mat33::Zero()),
          surface_point(Vec3::Zero()) {}
};

// Map a point on the current material surface to the corresponding point at
// the beginning of the substep. The rotations must be finite, right-handed,
// and orthonormal when the two poses differ.
Vec3 sdf_previous_material_point(
    const SDFMaterialMotion& motion,
    const Vec3& current_material_point);

//  Infinite half-space
struct PlaneSDF {
    Vec3 point;
    Vec3 normal;   //  must be unit length
    SDFMaterialMotion material_motion;
};

SDFEvaluation evaluate_sdf(const PlaneSDF& s, const Vec3& x);

//  Infinite solid cylinder
struct CylinderSDF {
    Vec3   point;
    Vec3   axis;     //  must be unit length
    double radius;
    SDFMaterialMotion material_motion;
};

SDFEvaluation evaluate_sdf(const CylinderSDF& s, const Vec3& x);

//  Solid sphere
struct SphereSDF {
    Vec3   center;
    double radius;
    SDFMaterialMotion material_motion;
};

SDFEvaluation evaluate_sdf(const SphereSDF& s, const Vec3& x);

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps);

Vec3   sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps);

Mat33  sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps,
                           bool include_curvature = true);

RigidEnergyDerivatives sdf_penalty_derivatives_rb(
        const SDFEvaluation& sdf, const Vec3& X_centered,
        const QuaternionOmegaKinematics& kinematics,
        double k, double eps,
        bool include_sdf_curvature = true,
        bool include_rigid_curvature = true);

// Cached-input overload. precomputed_hessian must have been evaluated with
// the same include_sdf_curvature setting used for this call.
RigidEnergyDerivatives sdf_penalty_derivatives_rb(
        const SDFEvaluation& sdf, const Vec3& X_centered,
        const QuaternionOmegaKinematics& kinematics,
        double k, double eps,
        bool include_sdf_curvature,
        bool include_rigid_curvature,
        const Vec3* precomputed_gradient,
        const Mat33* precomputed_hessian,
        const Mat33* precomputed_position_jacobian);

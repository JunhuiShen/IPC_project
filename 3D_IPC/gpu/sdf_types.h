#pragma once

// Plain SDF records shared by host API and device evaluators.
namespace ipc_gpu {
namespace solver {

enum class SdfKind : unsigned { Plane, Cylinder, Sphere };
enum class SdfStatus : unsigned { Ok, InvalidInput, InvalidMaterialPose, NonFiniteDerivative };

struct SdfPose {
    double rotation[9]{1,0,0,0,1,0,0,0,1}; // row-major material->world
    double translation[3]{};
};
struct SdfCollider {
    double point[3]{}; // plane/cylinder point or sphere center
    double axis[3]{0,1,0}; // plane normal or infinite-cylinder axis (unit)
    double radius=0;
    SdfPose previous_pose,current_pose;
    SdfKind kind=SdfKind::Plane;
};
struct SdfParameters {
    double stiffness=0,epsilon=0,dt=1.0/300.0,friction=0,eps_v=.001;
    // Production solver uses false for both deformables and rigid bodies.
    unsigned include_curvature=0;
};
struct SdfOutput {
    double phi=0,grad_phi[3]{},hess_phi[9]{},surface_point[3]{};
    double gradient[3]{},hessian[9]{}; // includes stiffness, NOT dt²
    double friction_gradient[3]{},friction_hessian[9]{}; // includes dt²
    int collider=-1;
    SdfStatus status=SdfStatus::Ok;
};

} // namespace solver
} // namespace ipc_gpu

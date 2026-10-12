#pragma once

#include "kernels.h"
#include <type_traits>

namespace ipc_gpu {

enum RigidFlags : unsigned {
    RigidTranslationGradient = 1u << 0,
    RigidOrientationGradient = 1u << 1,
    RigidTranslationHessian = 1u << 2,
    RigidOrientationHessian = 1u << 3,
    RigidMixedHessian = 1u << 4,
    RigidUpperCrossBlocks = 1u << 5,
    RigidFriction = 1u << 6,
};

// CPU geometry preparation freezes the exact closest feature and Cartesian
// derivatives. All arrays are row-major and contain no Eigen objects/pointers.
// CUDA performs the complete coupled COM/omega pullback, including quaternion
// curvature and signed multi-role friction. Barrier remains unscaled;
// friction_gradient/hessian already include dt^2 and friction coefficient.
struct RigidPreparedContact {
    double gradients[4][3];
    double jacobians[4][9];
    double hessians[4][4][9];
    double curvature[4][3][9];
    double translation_hessian[9];
    double friction_weights[4];
    double friction_gradient[3];
    double friction_hessian[9];
    unsigned flags;
    unsigned update_mask; // bit 0: COM, bit 1: omega (frozen friction only)
    unsigned segment_segment;
};

struct RigidDerivatives {
    double translation_gradient[3];
    double orientation_gradient[3];
    double translation_translation_hessian[9];
    double translation_orientation_hessian[9];
    double orientation_orientation_hessian[9];
};

struct RigidResult {
    RigidDerivatives barrier;
    RigidDerivatives friction;
};

struct RigidInertia {
    double position[3], previous_position[3], previous_velocity[3];
    double mass, gravity, dt;
    double orientation[4];
    double orientation_jacobian[4][3];
    double orientation_hessians[4][9];
    double second_moment[9], rotation_predictor[9];
    unsigned rotational; // zero: COM inertia/gravity, one: rotational inertia
};

static_assert(std::is_trivially_copyable<RigidPreparedContact>::value);
static_assert(std::is_standard_layout<RigidPreparedContact>::value);
static_assert(std::is_trivially_copyable<RigidResult>::value);
static_assert(std::is_trivially_copyable<RigidInertia>::value);

// One transfer/launch covers all admitted contacts of one body and phase.
// CUDA errors are reported; these entry points never silently compute on CPU.
void compute_rigid(const RigidPreparedContact*, std::size_t, RigidResult*);
void compute_rigid_inertia(const RigidInertia*, std::size_t, Derivatives*);

} // namespace ipc_gpu

#pragma once

#include "SIMD.h"
#include "barrier_energy.h"
#include "broad_phase.h"

#include <functional>
#include <vector>

namespace ipc_simd {

// An immutable contact snapshot. Barrier derivatives include every role on
// side; body_role_mask identifies the actual owned roles for frozen friction.
// A zero mask selects the same roles as side. Kinematics is required for
// orientation modes, with second derivatives for an orientation Hessian.
struct RigidContactInput {
    std::array<Vec3, 4> positions;
    std::array<Vec3, 4> previous_positions;
    std::array<Vec3, 4> body_references;
    bool segment_segment = false;
    bool exact_computation_fallback = true;
    RigidBarrierSide side = RigidBarrierSide::FirstPrimitive;
    const QuaternionOmegaKinematics* kinematics = nullptr;
    RigidBodyUpdateMode update_mode = RigidBodyUpdateMode::TranslationAndOrientation;
    unsigned body_role_mask = 0;
};

struct RigidContactOutput {
    RigidEnergyDerivatives barrier;
    RigidEnergyDerivatives friction;
};

// Scalar closest-feature selection and exact Cartesian cross-Hessian
// preparation, followed by SIMD coupled rigid pullback and frozen friction.
// Independent contacts occupy lanes. No mesh lookup or shared accumulation is
// performed here. Barrier is unscaled; friction already includes dt^2.
void rigid_contact_derivatives_tile(const RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidContactOutput* outputs);

// Gather contacts in NT-then-SS order from one fixed body configuration,
// compute private tiles, join helpers, and distribute in original order.
// The caller retains separate COM and omega phases and applies dt^2*k_barrier
// to the barrier result. leader_work must not mutate the contact snapshot.
RigidContactOutput rigid_contact_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& cache,
    const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices,
    const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* kinematics = nullptr,
    bool cooperative = false,
    const std::function<void()>* leader_work = nullptr);

} // namespace ipc_simd

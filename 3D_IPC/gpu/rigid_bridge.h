#pragma once

#include "rigid_types.h"
#include "general_simd_rigid.h"

namespace ipc_gpu {

// Host-only robust geometry and frozen-friction preparation. This performs no
// generalized-coordinate contraction and has no fixed CPU tile size.
void prepare_rigid_contacts(const ipc_simd::RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidPreparedContact* outputs);

ipc_simd::RigidContactOutput unpack_rigid(const RigidResult& value);

// Drop-in body contact organizer for the isolated GPU solver overlay. Each
// invocation gathers the current positions independently, so an omega phase
// observes the preceding committed COM update. CPU CCD/box scheduling stays
// with the existing solver. The cooperative argument preserves its interface;
// a whole body is submitted in one CUDA call, never in 32-contact CPU tiles.
ipc_simd::RigidContactOutput rigid_contact_derivatives(
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

std::pair<Vec3, Mat33> rigid_translation_inertia_gravity(
    const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n,
    double dt, double total_mass, double gravity);

std::pair<Vec3, Mat33> rigid_rotation_inertia(
    const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, double dt,
    const Mat33& I_hat, const QuaternionOmegaKinematics* kinematics = nullptr,
    const Mat33* rotation_predictor = nullptr);

} // namespace ipc_gpu

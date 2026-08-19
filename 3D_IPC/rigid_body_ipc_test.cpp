#include "rigid_body_ipc.h"

#include "parallel_helper.h"
#include "physics.h"
#include "safe_step.h"
#include "simulation.h"
#include "solver.h"
#include "mesh_utils.h"
#include "time_integration.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDt = 0.03;

Vec4 reference_quaternion_from_rotation_vector(const Vec3& rotation_vector) {
    const double angle = rotation_vector.norm();
    if (angle == 0.0)
        return Vec4(1.0, 0.0, 0.0, 0.0);
    const double sin_half_angle = std::sin(0.5 * angle);
    return Vec4(std::cos(0.5 * angle), sin_half_angle * rotation_vector[0] / angle, sin_half_angle * rotation_vector[1] / angle, sin_half_angle * rotation_vector[2] / angle);
}

Vec4 reference_quaternion_multiply(const Vec4& a, const Vec4& b) {
    return Vec4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

struct OmegaNodeEnergy {
    double energy = 0.0;
    Vec3 gradient = Vec3::Zero();
    Mat33 hessian = Mat33::Zero();
};

struct QuaternionNodeEnergy {
    double energy = 0.0;
    Vec4 gradient = Vec4::Zero();
    Mat44 hessian = Mat44::Zero();
};

struct RigidSafeStepCandidates {
    BroadPhase::Cache cache;
    std::vector<std::vector<int>> body_nt_pair_indices;
    std::vector<std::vector<int>> body_ss_pair_indices;
};

RigidSafeStepCandidates build_rigid_safe_step_candidates(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, int num_vertices, int num_rbs) {
    RigidSafeStepCandidates candidates;
    for (int node = 0; node < num_vertices; ++node) {
        for (int tri = 0; tri < num_tris(ref_mesh); ++tri) {
            const int v0 = tri_vertex(ref_mesh, tri, 0);
            const int v1 = tri_vertex(ref_mesh, tri, 1);
            const int v2 = tri_vertex(ref_mesh, tri, 2);
            if (node != v0 && node != v1 && node != v2)
                candidates.cache.nt_pairs.push_back(NodeTrianglePair{node, {v0, v1, v2}});
        }
    }
    for (int first = 0; first < static_cast<int>(edges.size()); ++first) {
        for (int second = first + 1; second < static_cast<int>(edges.size()); ++second) {
            const int a0 = edges[first][0];
            const int a1 = edges[first][1];
            const int b0 = edges[second][0];
            const int b1 = edges[second][1];
            if (a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1)
                candidates.cache.ss_pairs.push_back(SegmentSegmentPair{{a0, a1, b0, b1}});
        }
    }
    std::vector<std::vector<int>> adjacency;
    build_rb_contact_adj(candidates.cache, ref_mesh.node_to_rb, num_rbs, candidates.body_nt_pair_indices, candidates.body_ss_pair_indices, adjacency);
    return candidates;
}

double translation_safe_step_for_test(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& positions, int rb, const Vec3& displacement, double safety = 0.9) {
    const RigidSafeStepCandidates candidates = build_rigid_safe_step_candidates(ref_mesh, edges, static_cast<int>(positions.size()), rb + 1);
    return per_rigid_body_translation_safe_step(ref_mesh, candidates.cache, candidates.body_nt_pair_indices[rb], candidates.body_ss_pair_indices[rb], positions, rb, displacement, safety);
}

double rotation_safe_step_for_test(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& positions, int rb, const Vec3& x_com, const Vec4& q_current, const Vec4& q_target, double safety = 0.9) {
    const RigidSafeStepCandidates candidates = build_rigid_safe_step_candidates(ref_mesh, edges, static_cast<int>(positions.size()), rb + 1);
    return per_rigid_body_rotation_safe_step(ref_mesh, candidates.cache, candidates.body_nt_pair_indices[rb], candidates.body_ss_pair_indices[rb], positions, rb, x_com, q_current, q_target, safety);
}

OmegaNodeEnergy evaluate_omega_node_energy(const Vec3& X_centered, const Vec3& target, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt) {
    const Vec3 x = world_space_position(X_centered, x_com, q0, omega, dt);
    const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(q0, omega, dt, true);
    const Mat33 J_xomega = dx_domega(X_centered, kinematics);
    const std::array<Mat33, 3> H_xomega = d2x_domega2(X_centered, kinematics);
    const Vec3 gx = x - target;
    const Vec3 omega_gradient = rigid_node_omega_gradient(gx, J_xomega);
    const Mat33 omega_hessian = rigid_node_omega_hessian(gx, Mat33::Identity(), J_xomega, H_xomega);

    OmegaNodeEnergy result;
    result.energy = 0.5 * gx.squaredNorm();
    result.gradient = omega_gradient;
    result.hessian = omega_hessian;
    return result;
}

QuaternionNodeEnergy evaluate_quaternion_node_energy(const Vec3& X_centered, const Vec3& target, const Vec3& x_com, const Vec4& quat) {
    const Vec3 gx = x_com + quaternion_rotate(quat, X_centered) - target;
    const Mat34 J_xq = dx_dq(X_centered, quat);
    const std::array<Mat44, 3> H_xq = d2x_dq2(X_centered);

    QuaternionNodeEnergy result;
    result.energy = 0.5 * gx.squaredNorm();
    result.gradient = J_xq.transpose() * gx;
    result.hessian = J_xq.transpose() * J_xq;
    for (int c = 0; c < 3; ++c)
        result.hessian += gx[c] * H_xq[c];
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
    return result;
}

const std::vector<double> kConvergenceHs = {
    1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4
};

void expect_quadratic_convergence(
    const std::vector<double>& hs,
    const std::vector<double>& errors) {
    ASSERT_EQ(hs.size(), errors.size());

    constexpr double noise_floor = 1.0e-10;
    bool all_below_noise = true;
    bool saw_reliable_slope = false;

    for (std::size_t i = 1; i < errors.size(); ++i) {
        if (errors[i - 1] < noise_floor || errors[i] < noise_floor)
            continue;

        all_below_noise = false;
        if (errors[i] == 0.0)
            continue;

        const double slope = std::log(errors[i - 1] / errors[i])
            / std::log(hs[i - 1] / hs[i]);
        saw_reliable_slope = true;
        EXPECT_GT(slope, 1.99);
        EXPECT_LT(slope, 2.01);
    }

    EXPECT_TRUE(all_below_noise || saw_reliable_slope)
        << "no reliable finite-difference slope data";
}

TEST(RigidBodyIPCCreation, ComputesAndStoresRigidBodyState) {
    const Vec3 center(0.7, -0.4, 1.2);
    const Vec3 v_com(-0.3, 0.5, 0.2);
    const Vec3 omega(0.4, -0.2, 0.3);
    const Vec4 orientation =
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const std::vector<Vec3> body_offsets = {
        Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0), Vec3(0.0, -2.0, 0.0),
        Vec3(0.0, 0.0, 0.5), Vec3(0.0, 0.0, -0.5)
    };

    DeformedState state;
    state.deformed_positions.push_back(Vec3(9.0, 8.0, 7.0));
    std::vector<Vec3> x;
    for (const Vec3& offset : body_offsets) {
        x.push_back(
            center + quaternion_rotate(orientation, offset));
    }

    RefMesh ref_mesh;
    constexpr double total_mass = 12.0;
    const int rb = create_rigid_body(
        x, v_com, 2.0 * orientation, omega, total_mass,
        ref_mesh, state);
    const std::vector<int>& nodes = ref_mesh.rb_nodes[rb];

    ASSERT_EQ(rb, 0);
    ASSERT_EQ(state.x_coms.size(), 1u);
    ASSERT_EQ(ref_mesh.ref_positions.size(), 1u);
    ASSERT_EQ(ref_mesh.rb_nodes.size(), 1u);
    ASSERT_EQ(ref_mesh.num_positions, state.deformed_positions.size());
    ASSERT_EQ(ref_mesh.node_to_rb.size(), state.deformed_positions.size());

    EXPECT_TRUE(state.x_coms[rb].isApprox(center, 1.0e-14));
    EXPECT_TRUE(state.v_coms[rb].isApprox(v_com, 1.0e-14));
    EXPECT_TRUE(state.orientations[rb].isApprox(orientation, 1.0e-14));
    EXPECT_TRUE(state.omega[rb].isApprox(omega, 1.0e-14));
    EXPECT_DOUBLE_EQ(ref_mesh.total_mass[rb], total_mass);
    EXPECT_EQ(ref_mesh.rb_nodes[rb], nodes);
    EXPECT_EQ(ref_mesh.node_to_rb[0], -1);

    const double nodal_mass = total_mass / body_offsets.size();
    const std::vector<double> masses(body_offsets.size(), nodal_mass);
    const Mat33 expected_I_hat = body_second_moment(masses, body_offsets);
    EXPECT_TRUE(ref_mesh.I_hat[rb].isApprox(expected_I_hat, 1.0e-14));

    for (std::size_t local = 0; local < nodes.size(); ++local) {
        const int node = nodes[local];
        const Vec3 world_offset = quaternion_rotate(
            orientation, body_offsets[local]);
        EXPECT_TRUE(ref_mesh.ref_positions[rb][local].isApprox(
            body_offsets[local], 1.0e-14));
        EXPECT_DOUBLE_EQ(ref_mesh.mass[node], nodal_mass);
        EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
        EXPECT_TRUE(state.velocities[node].isApprox(
            v_com + omega.cross(world_offset), 1.0e-14));
    }
}

TEST(RigidBodyIPCCreation, StoresUpdateModesAndCanonicalizesDisabledVelocities) {
    const std::array<RigidBodyUpdateMode, 4> modes = {
        RigidBodyUpdateMode::TranslationAndOrientation,
        RigidBodyUpdateMode::TranslationOnly,
        RigidBodyUpdateMode::OrientationOnly,
        RigidBodyUpdateMode::None,
    };
    const Vec3 input_v_com(0.7, -0.4, 0.2);
    const Vec3 input_omega(-0.3, 0.5, 0.8);
    const Vec4 orientation = quaternion_normalize(
        Vec4(0.9, -0.1, 0.2, 0.3));
    const std::vector<Vec3> offsets = {
        Vec3(-0.4, -0.3, -0.2),
        Vec3(0.6, -0.2, -0.1),
        Vec3(-0.1, 0.7, -0.3),
        Vec3(-0.1, -0.2, 0.6),
    };

    RefMesh ref_mesh;
    DeformedState state;
    for (int body = 0; body < static_cast<int>(modes.size()); ++body) {
        const Vec3 center(4.0 * body, 1.0, -2.0);
        std::vector<Vec3> positions;
        positions.reserve(offsets.size());
        for (const Vec3& offset : offsets) {
            positions.push_back(
                center + quaternion_rotate(orientation, offset));
        }

        const int rb = create_rigid_body(
            positions, input_v_com, orientation, input_omega, 4.0,
            ref_mesh, state, modes[body]);
        ASSERT_EQ(rb, body);
        ASSERT_EQ(ref_mesh.rb_update_modes.size(),
            static_cast<std::size_t>(body + 1));
        EXPECT_EQ(ref_mesh.rb_update_modes[rb], modes[body]);

        const Vec3 expected_v_com = updates_rigid_translation(modes[body])
            ? input_v_com : Vec3::Zero();
        const Vec3 expected_omega = updates_rigid_orientation(modes[body])
            ? input_omega : Vec3::Zero();
        EXPECT_TRUE(state.v_coms[rb].isApprox(expected_v_com, 1.0e-14));
        EXPECT_TRUE(state.omega[rb].isApprox(expected_omega, 1.0e-14));

        for (const int node : ref_mesh.rb_nodes[rb]) {
            const Vec3 world_offset =
                state.deformed_positions[node] - state.x_coms[rb];
            EXPECT_TRUE(state.velocities[node].isApprox(
                expected_v_com + expected_omega.cross(world_offset),
                1.0e-14));
        }
    }
}

TEST(RigidBodyIPCUpdateModes, FixedBodyStaysBitwiseFixedWhileDefaultBodyFalls) {
    const Vec4 fixed_orientation = quaternion_normalize(
        Vec4(0.8, -0.2, 0.3, 0.4));
    const std::vector<Vec3> tetra_offsets = {
        Vec3(-0.4, -0.3, -0.2),
        Vec3(0.6, -0.2, -0.1),
        Vec3(-0.1, 0.7, -0.3),
        Vec3(-0.1, -0.2, 0.6),
    };

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec3> fixed_positions;
    fixed_positions.reserve(tetra_offsets.size());
    for (const Vec3& offset : tetra_offsets) {
        fixed_positions.push_back(
            Vec3(0.0, 1.0, 0.0)
            + quaternion_rotate(fixed_orientation, offset));
    }
    const int fixed_rb = create_rigid_body(
        fixed_positions, Vec3(1.0, 2.0, 3.0), fixed_orientation,
        Vec3(-0.5, 0.7, 0.2), 4.0, ref_mesh, state,
        RigidBodyUpdateMode::None);

    std::vector<Vec3> dynamic_positions;
    dynamic_positions.reserve(tetra_offsets.size());
    for (const Vec3& offset : tetra_offsets)
        dynamic_positions.push_back(Vec3(10.0, 3.0, 0.0) + offset);
    const int dynamic_rb = create_rigid_body(
        dynamic_positions, Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0),
        Vec3::Zero(), 4.0, ref_mesh, state);

    ASSERT_EQ(ref_mesh.rb_update_modes[fixed_rb],
        RigidBodyUpdateMode::None);
    ASSERT_EQ(ref_mesh.rb_update_modes[dynamic_rb],
        RigidBodyUpdateMode::TranslationAndOrientation);

    // Snapshot a canonical proxy configuration. Reinjecting forbidden rigid
    // velocities below checks that both the frame driver and proxy sync erase
    // them without perturbing any fixed generalized coordinate or proxy.
    sync_rigid_body_particles(ref_mesh, state);
    const Vec3 fixed_com_before = state.x_coms[fixed_rb];
    const Vec4 fixed_orientation_before = state.orientations[fixed_rb];
    std::vector<Vec3> fixed_proxies_before;
    fixed_proxies_before.reserve(ref_mesh.rb_nodes[fixed_rb].size());
    for (const int node : ref_mesh.rb_nodes[fixed_rb])
        fixed_proxies_before.push_back(state.deformed_positions[node]);

    state.v_coms[fixed_rb] = Vec3(4.0, -5.0, 6.0);
    state.omega[fixed_rb] = Vec3(-1.0, 2.0, -3.0);
    for (const int node : ref_mesh.rb_nodes[fixed_rb])
        state.velocities[node] = Vec3(7.0, 8.0, 9.0);

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 1.0;
    params.node_box_min = 0.5;
    params.node_box_max = 0.5;
    params.theta_box_min = 0.5;
    params.theta_box_max = 0.5;
    params.node_box_update_count = 1;
    params.d_hat = 0.0;
    params.k_barrier = 0.0;
    params.use_parallel = false;

    const Vec3 dynamic_com_before = state.x_coms[dynamic_rb];
    const SolverResult result = advance_one_frame_rb(state, ref_mesh, params);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.iterations, 1);

    EXPECT_EQ(std::memcmp(state.x_coms[fixed_rb].data(),
                  fixed_com_before.data(), sizeof(double) * 3),
        0);
    EXPECT_EQ(std::memcmp(state.orientations[fixed_rb].data(),
                  fixed_orientation_before.data(), sizeof(double) * 4),
        0);
    EXPECT_TRUE(state.v_coms[fixed_rb].isZero(0.0));
    EXPECT_TRUE(state.omega[fixed_rb].isZero(0.0));
    for (int local = 0;
         local < static_cast<int>(ref_mesh.rb_nodes[fixed_rb].size());
         ++local) {
        const int node = ref_mesh.rb_nodes[fixed_rb][local];
        EXPECT_EQ(std::memcmp(state.deformed_positions[node].data(),
                      fixed_proxies_before[local].data(), sizeof(double) * 3),
            0);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
    }

    const double dt = params.dt();
    const Vec3 expected_dynamic_com =
        dynamic_com_before + dt * dt * params.gravity;
    EXPECT_TRUE(state.x_coms[dynamic_rb].isApprox(
        expected_dynamic_com, 1.0e-14));
    EXPECT_LT(state.x_coms[dynamic_rb].y(), dynamic_com_before.y());
    EXPECT_TRUE(state.v_coms[dynamic_rb].isApprox(
        dt * params.gravity, 1.0e-14));
}

TEST(RigidBodyIPCUpdateModes, PartialModesAdvanceOnlyTheirEnabledCoordinates) {
    const std::vector<Vec3> offsets = {
        Vec3(-0.4, -0.3, -0.2),
        Vec3(0.6, -0.2, -0.1),
        Vec3(-0.1, 0.7, -0.3),
        Vec3(-0.1, -0.2, 0.6),
    };
    const Vec4 translation_only_orientation = quaternion_normalize(
        Vec4(0.9, 0.1, -0.2, 0.3));
    const Vec4 orientation_only_orientation = quaternion_normalize(
        Vec4(0.8, -0.3, 0.1, 0.4));

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec3> translation_only_positions;
    std::vector<Vec3> orientation_only_positions;
    for (const Vec3& offset : offsets) {
        translation_only_positions.push_back(
            Vec3(0.0, 3.0, 0.0)
            + quaternion_rotate(translation_only_orientation, offset));
        orientation_only_positions.push_back(
            Vec3(10.0, 2.0, 0.0)
            + quaternion_rotate(orientation_only_orientation, offset));
    }
    const int translation_only_rb = create_rigid_body(
        translation_only_positions, Vec3::Zero(),
        translation_only_orientation, Vec3(0.3, -0.2, 0.4), 4.0,
        ref_mesh, state, RigidBodyUpdateMode::TranslationOnly);
    const int orientation_only_rb = create_rigid_body(
        orientation_only_positions, Vec3(1.0, -2.0, 3.0),
        orientation_only_orientation, Vec3(0.3, -0.2, 0.4), 4.0,
        ref_mesh, state, RigidBodyUpdateMode::OrientationOnly);

    const Vec3 translation_com_before = state.x_coms[translation_only_rb];
    const Vec4 translation_q_before = state.orientations[translation_only_rb];
    const Vec3 orientation_com_before = state.x_coms[orientation_only_rb];
    const Vec4 orientation_q_before = state.orientations[orientation_only_rb];

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 1.0;
    params.node_box_min = 0.5;
    params.node_box_max = 0.5;
    params.theta_box_min = 0.5;
    params.theta_box_max = 0.5;
    params.node_box_update_count = 1;
    params.d_hat = 0.0;
    params.k_barrier = 0.0;
    params.use_parallel = false;

    const SolverResult result = advance_one_frame_rb(state, ref_mesh, params);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.iterations, 1);

    const double dt = params.dt();
    EXPECT_TRUE(state.x_coms[translation_only_rb].isApprox(
        translation_com_before + dt * dt * params.gravity, 1.0e-14));
    EXPECT_TRUE(state.v_coms[translation_only_rb].isApprox(
        dt * params.gravity, 1.0e-14));
    EXPECT_EQ(std::memcmp(state.orientations[translation_only_rb].data(),
                  translation_q_before.data(), sizeof(double) * 4),
        0);
    EXPECT_TRUE(state.omega[translation_only_rb].isZero(0.0));

    EXPECT_EQ(std::memcmp(state.x_coms[orientation_only_rb].data(),
                  orientation_com_before.data(), sizeof(double) * 3),
        0);
    EXPECT_TRUE(state.v_coms[orientation_only_rb].isZero(0.0));
    EXPECT_NE(std::memcmp(state.orientations[orientation_only_rb].data(),
                  orientation_q_before.data(), sizeof(double) * 4),
        0);
    EXPECT_GT(state.omega[orientation_only_rb].norm(), 1.0e-8);
    EXPECT_TRUE(state.orientations[orientation_only_rb].isApprox(
        quaternion_from_angular_velocity(
            orientation_q_before, state.omega[orientation_only_rb], dt),
        1.0e-13));
}

TEST(RigidBodyIPCPositionHelpers, OrientationOverloadsRoundTrip) {
    const Vec3 x_com(0.7, -0.4, 1.2);
    const Vec4 orientation =
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 X(0.3, -0.6, 0.2);

    const Vec3 x = world_space_position(
        X, x_com, orientation);
    const Vec3 recovered_X = material_space_position(
        x, x_com, orientation);

    EXPECT_TRUE(recovered_X.isApprox(X, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionWrappers, MultiplyAndConjugateUseScalarFirstHamiltonConvention) {
    const Vec4 a(0.5, -0.2, 0.4, 0.1);
    const Vec4 b(-0.3, 0.6, -0.1, 0.2);

    EXPECT_TRUE(quaternion_multiply(a, b).isApprox(Vec4(-0.01, 0.45, -0.07, -0.15), 1.0e-14));
    EXPECT_TRUE(quaternion_conjugate(a).isApprox(Vec4(0.5, 0.2, -0.4, -0.1), 1.0e-14));
}

TEST(RigidBodyIPCQuaternionWrappers, InverseAndNormalizeHandleNonunitQuaternion) {
    const Vec4 quat(2.0, -1.0, 3.0, 4.0);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);

    EXPECT_TRUE(quaternion_multiply(quat, quaternion_inverse(quat)).isApprox(identity, 1.0e-14));
    EXPECT_NEAR(quaternion_normalize(quat).norm(), 1.0, 1.0e-14);
    EXPECT_THROW(quaternion_inverse(Vec4::Zero()), std::invalid_argument);
    EXPECT_THROW(quaternion_normalize(Vec4::Zero()), std::invalid_argument);
}

TEST(RigidBodyIPCQuaternionWrappers, FullArcInterpolationPreserves270DegreeBranch) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec3 full_rotation(0.0, 0.0, 1.5 * M_PI);
    const Vec4 target = quaternion_from_angular_velocity(
        identity, full_rotation, 1.0);

    const Vec4 halfway = interpolate_orientation_full_arc(
        identity, target, 0.5);
    const Vec4 expected_halfway = quaternion_from_angular_velocity(
        identity, Vec3(0.0, 0.0, 0.75 * M_PI), 1.0);

    EXPECT_TRUE(halfway.isApprox(expected_halfway, 1.0e-14));
    EXPECT_TRUE(interpolate_orientation_full_arc(
        identity, target, 1.0).isApprox(
            quaternion_normalize(target), 1.0e-14));
}

TEST(RigidBodyIPCQuaternionWrappers, FullArcTreatsOppositeTargetSignsAsComplementaryPaths) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target_270 = quaternion_from_angular_velocity(
        identity, Vec3(0.0, 0.0, 1.5 * M_PI), 1.0);

    const Vec4 halfway_270 = interpolate_orientation_full_arc(
        identity, target_270, 0.5);
    const Vec4 halfway_minus_90 = interpolate_orientation_full_arc(
        identity, -target_270, 0.5);
    const Vec4 expected_135 = quaternion_from_angular_velocity(
        identity, Vec3(0.0, 0.0, 0.75 * M_PI), 1.0);
    const Vec4 expected_minus_45 = quaternion_from_angular_velocity(
        identity, Vec3(0.0, 0.0, -0.25 * M_PI), 1.0);

    EXPECT_TRUE(halfway_270.isApprox(expected_135, 1.0e-14));
    EXPECT_TRUE(halfway_minus_90.isApprox(expected_minus_45, 1.0e-14));
    EXPECT_LT(std::abs(halfway_270.dot(halfway_minus_90)), 1.0e-14);
}

TEST(RigidBodyIPCQuaternionWrappers, NearlyFullArcDoesNotUseShortestArcFallback) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const double angle = 2.0 * M_PI - 4.0e-13;
    const Vec4 target = quaternion_from_angular_velocity(identity, angle * Vec3::UnitZ(), 1.0);
    const Vec4 interpolated = interpolate_orientation_full_arc(identity, target, 0.25);
    const Vec4 expected = quaternion_from_angular_velocity(identity, 0.25 * angle * Vec3::UnitZ(), 1.0);
    EXPECT_NEAR(std::abs(interpolated.dot(expected)), 1.0, 1.0e-13);
}

TEST(RigidBodyIPCQuaternionWrappers, FullArcAngularVelocityRecoveryPreserves270DegreeBranch) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    constexpr double dt = 0.25;
    const Vec3 omega(0.0, 0.0, 1.5 * M_PI / dt);
    const Vec4 target = quaternion_from_angular_velocity(
        identity, omega, dt);

    EXPECT_TRUE(angular_velocity_from_orientation_full_arc(
        target, identity, dt).isApprox(omega, 1.0e-13));
    EXPECT_TRUE(angular_velocity_from_orientation_full_arc(
        -target, identity, dt).isApprox(
            Vec3(0.0, 0.0, -0.5 * M_PI / dt), 1.0e-13));
}

TEST(RigidBodyIPCQuaternionWrappers, FullArcHelpersValidateInputsAndAmbiguousFullTurn) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);

    EXPECT_THROW(
        interpolate_orientation_full_arc(identity, identity, -0.1),
        std::invalid_argument);
    EXPECT_THROW(
        interpolate_orientation_full_arc(
            identity, identity,
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
    EXPECT_THROW(
        angular_velocity_from_orientation_full_arc(
            identity, identity, 0.0),
        std::invalid_argument);
    EXPECT_THROW(
        angular_velocity_from_orientation_full_arc(
            identity, identity,
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);

    // The antipodal endpoint specifies a 360-degree angle but not its axis.
    EXPECT_THROW(
        interpolate_orientation_full_arc(identity, -identity, 0.5),
        std::invalid_argument);
    EXPECT_THROW(
        angular_velocity_from_orientation_full_arc(
            -identity, identity, 1.0),
        std::invalid_argument);

    // A full turn built with trigonometric functions is only numerically
    // antipodal; its roundoff-sized vector part is not a reliable axis.
    const Vec4 numerical_full_turn =
        quaternion_from_angular_velocity(
            identity, Vec3(0.0, 0.0, 2.0 * M_PI), 1.0);
    EXPECT_THROW(
        interpolate_orientation_full_arc(
            identity, numerical_full_turn, 0.5),
        std::invalid_argument);
    EXPECT_THROW(
        angular_velocity_from_orientation_full_arc(
            numerical_full_turn, identity, 1.0),
        std::invalid_argument);
}

TEST(RigidBodyIPCQuaternionWrappers, TimeDerivativeUsesWorldSpaceAngularVelocity) {
    const Vec4 q = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec4 omega_quaternion(0.0, omega[0], omega[1], omega[2]);
    const Vec4 expected = 0.5 * quaternion_multiply(omega_quaternion, q);
    const Vec4 q_dot = quaternion_time_derivative(q, omega);

    EXPECT_TRUE(q_dot.isApprox(expected, 1.0e-14));
    EXPECT_NEAR(q.dot(q_dot), 0.0, 1.0e-14);
}

TEST(RigidBodyIPCQuaternionWrappers, ForwardAndInverseRotationRoundTrip) {
    const double half_angle = 0.25 * std::acos(-1.0);
    const Vec4 quat(std::cos(half_angle), 0.0, 0.0, std::sin(half_angle));
    const Vec3 vector(1.0, 0.0, 0.0);

    const Vec3 rotated = quaternion_rotate(quat, vector);
    EXPECT_TRUE(rotated.isApprox(Vec3(0.0, 1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(quaternion_inverse_rotate(quat, rotated).isApprox(vector, 1.0e-14));
}

TEST(RigidBodyIPCOmegaNodeKinematics, MaterialAndWorldSpacePositionRoundTrip) {
    const Vec3 X_centered(0.7, -0.4, 1.2);
    const Vec3 x_com(-0.3, 0.8, 1.5);
    const Vec4 q0 = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec3 x = world_space_position(X_centered, x_com, q0, omega, kDt);
    const Vec3 recovered = material_space_position(x, x_com, q0, omega, kDt);

    EXPECT_TRUE(recovered.isApprox(X_centered, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, ProductTensorMatchesHamiltonProduct) {
    const Vec4 a(0.5, -0.2, 0.4, 0.1);
    const Vec4 b(-0.3, 0.6, -0.1, 0.2);
    Vec4 product_from_tensor = Vec4::Zero();
    for (int alpha = 0; alpha < 4; ++alpha) {
        for (int beta = 0; beta < 4; ++beta) {
            for (int gamma = 0; gamma < 4; ++gamma) {
                product_from_tensor[alpha] +=
                    quaternion_product_tensor(alpha, beta, gamma) * a[beta] * b[gamma];
            }
        }
    }

    const Vec4 expected = reference_quaternion_multiply(a, b);
    EXPECT_TRUE(product_from_tensor.isApprox(expected, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionExp, FirstOrderTaylorRemainderConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec3 direction = Vec3(0.3, -0.4, 0.5).normalized();
    constexpr double dt = 1.3;
    const Vec4 value = exp(omega, dt);
    const Mat43 J = dexp_domega(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const Vec4 remainder = exp(omega + h * direction, dt) - value - h * J * direction;
        errors[hi] = remainder.norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, JacobianConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat43 exact = dexp_domega(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat43 finite_difference = Mat43::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) =
                (exp(omega + step, dt) - exp(omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, SecondDerivativeConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 4> exact = d2exp_domega2(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat43 finite_difference =
                (dexp_domega(omega + step, dt) - dexp_domega(omega - step, dt)) / (2.0 * h);
            for (int alpha = 0; alpha < 4; ++alpha) {
                const Vec3 error = finite_difference.row(alpha).transpose()   - exact[alpha].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, SmallAngleTaylorBranchMatchesLimit) {
    constexpr double dt = 1.3;
    const double a = 0.5 * dt;
    const Vec3 omega(3.0e-5, -4.0e-5, 2.0e-5);
    ASSERT_GT(a * omega.norm(), 1.0e-6);
    ASSERT_LT(a * omega.norm(), 1.0e-4);

    const double angle = a * omega.norm();
    const double angle2 = angle * angle;
    const double angle4 = angle2 * angle2;
    const double sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0;
    const Mat33 omega_outer = omega * omega.transpose();
    const double C1 = -a * a * sinc;
    const double C2 = a * sinc;
    const double C3 = a * a * a * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);

    Mat43 expected_dexp = Mat43::Zero();
    expected_dexp.row(0) = C1 * omega.transpose();
    expected_dexp.bottomRows<3>() = C2 * Mat33::Identity() + C3 * omega_outer;
    EXPECT_TRUE(dexp_domega(omega, dt).isApprox(expected_dexp, 1.0e-14));

    const std::array<Mat33, 4> d2exp = d2exp_domega2(omega, dt);
    const double scalar_outer_coefficient = std::pow(a, 4) * (1.0 / 3.0 - angle2 / 30.0 + angle4 / 840.0);
    const Mat33 expected_scalar_hessian = C1 * Mat33::Identity() + scalar_outer_coefficient * omega_outer;
    EXPECT_TRUE(d2exp[0].isApprox(expected_scalar_hessian, 1.0e-14));

    const double A = C3;
    const double B = std::pow(a, 5) * (1.0 / 15.0 - angle2 / 210.0 + angle4 / 7560.0);
    for (int i = 0; i < 3; ++i) {
        Mat33 expected_vector_hessian = Mat33::Zero();
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                expected_vector_hessian(j, k) = B * omega[i] * omega[j] * omega[k]
                    + A * ((i == j ? omega[k] : 0.0) + (i == k ? omega[j] : 0.0) + (j == k ? omega[i] : 0.0));
            }
        }
        EXPECT_TRUE(d2exp[1 + i].isApprox(expected_vector_hessian, 1.0e-14));
    }
}

TEST(RigidBodyIPCQuaternionOmega, MatchesLeftQuaternionUpdate) {
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);

    const Vec4 expected = reference_quaternion_multiply(reference_quaternion_from_rotation_vector(kDt * omega), q0);
    const Vec4 actual = quaternion_from_angular_velocity(q0, omega, kDt);

    EXPECT_TRUE(actual.isApprox(expected, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, ZeroAngularVelocityTaylorLimit) {
    const Vec4 q0(1.0, 0.0, 0.0, 0.0);
    const Vec3 zero_omega = Vec3::Zero();

    EXPECT_TRUE(exp(zero_omega, kDt).isApprox(q0, 1.0e-14));
    EXPECT_TRUE(quaternion_from_angular_velocity(q0, zero_omega, kDt).isApprox(q0, 1.0e-14));

    const Mat43 J = dq_domega(q0, zero_omega, kDt);
    Mat43 expected = Mat43::Zero();
    expected.bottomRows<3>() = 0.5 * kDt * Mat33::Identity();
    EXPECT_TRUE(J.isApprox(expected, 1.0e-14));

    const std::array<Mat33, 4> H_exp = d2exp_domega2(zero_omega, kDt);
    EXPECT_TRUE(H_exp[0].isApprox(-0.25 * kDt * kDt * Mat33::Identity(), 1.0e-14));
    for (int alpha = 1; alpha < 4; ++alpha)
        EXPECT_TRUE(H_exp[alpha].isZero(1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, JacobianConvergesQuadratically) {
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(-0.3, 0.2, 0.1));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat43 exact = dq_domega(q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat43 finite_difference = Mat43::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) = (quaternion_from_angular_velocity(q0, omega + step, dt) - quaternion_from_angular_velocity(q0, omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionOmega, SecondDerivativeConvergesQuadratically) {
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(-0.3, 0.2, 0.1));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 4> exact = d2q_domega2(q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());
    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat43 plus = dq_domega(q0, omega + step, dt);
            const Mat43 minus = dq_domega(q0, omega - step, dt);
            const Mat43 finite_difference = (plus - minus) / (2.0 * h);
            for (int alpha = 0; alpha < 4; ++alpha) {
                const Vec3 error = finite_difference.row(alpha).transpose()  - exact[alpha].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionDerivatives, DxDqTaylorRemainderConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q(0.8, -0.3, 0.4, 0.2);
    const Vec4 direction = Vec4(0.2, -0.3, 0.4, -0.5).normalized();
    const Mat34 J = dx_dq(X_centered, q);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const Vec3 remainder = quaternion_rotate(q + h * direction, X_centered) - quaternion_rotate(q, X_centered) - h * J * direction;
        errors[hi] = remainder.norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionDerivatives, D2xDq2MatchesFiniteDifferenceOfDxDq) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q(0.8, -0.3, 0.4, 0.2);
    const std::array<Mat44, 3> exact = d2x_dq2(X_centered);

    for (double h : kConvergenceHs) {
        for (int gamma = 0; gamma < 4; ++gamma) {
            Vec4 step = Vec4::Zero();
            step[gamma] = h;
            const Mat34 finite_difference =
                (dx_dq(X_centered, q + step) - dx_dq(X_centered, q - step)) / (2.0 * h);

            for (int c = 0; c < 3; ++c) {
                const Vec4 error = finite_difference.row(c).transpose()
                    - exact[c].col(gamma);
                EXPECT_LT(error.norm(), 1.0e-10)
                    << "h=" << h << ", output coordinate=" << c
                    << ", quaternion coordinate=" << gamma;
            }
        }
    }
}

TEST(RigidBodyIPCQuaternionDerivatives, NodeEnergyDerivativesConvergeQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 target(-0.2, 0.6, 0.3);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 quat(0.8, -0.3, 0.4, 0.2);
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const QuaternionNodeEnergy exact =
            evaluate_quaternion_node_energy(X_centered, target, x_com, quat);
        Vec4 gradient_fd = Vec4::Zero();
        Mat44 hessian_fd = Mat44::Zero();
        for (int beta = 0; beta < 4; ++beta) {
            Vec4 step = Vec4::Zero();
            step[beta] = h;
            const QuaternionNodeEnergy plus =
                evaluate_quaternion_node_energy(X_centered, target, x_com, quat + step);
            const QuaternionNodeEnergy minus =
                evaluate_quaternion_node_energy(X_centered, target, x_com, quat - step);
            gradient_fd[beta] = (plus.energy - minus.energy) / (2.0 * h);
            hessian_fd.col(beta) = (plus.gradient - minus.gradient) / (2.0 * h);
        }
        gradient_errors[hi] = (gradient_fd - exact.gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact.hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCOmegaNodeKinematics, JacobianConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat33 exact = dx_domega(X_centered, quaternion_omega_kinematics(q0, omega, dt));
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat33 finite_difference = Mat33::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) =
                (world_space_position(X_centered, x_com, q0, omega + step, dt)
                 - world_space_position(X_centered, x_com, q0, omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCOmegaNodeKinematics, SecondDerivativeConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 3> exact = d2x_domega2(X_centered, quaternion_omega_kinematics(q0, omega, dt, true));
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const QuaternionOmegaKinematics plus_kinematics = quaternion_omega_kinematics(q0, omega + step, dt);
            const QuaternionOmegaKinematics minus_kinematics = quaternion_omega_kinematics(q0, omega - step, dt);
            const Mat33 finite_difference =
                (dx_domega(X_centered, plus_kinematics)
                 - dx_domega(X_centered, minus_kinematics)) / (2.0 * h);
            for (int c = 0; c < 3; ++c) {
                const Vec3 error = finite_difference.row(c).transpose()
                    - exact[c].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCOmegaNodeKinematics, QuaternionDerivativeCacheMatchesPrimitiveEvaluationsExactly) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;

    const QuaternionOmegaKinematics first_order = quaternion_omega_kinematics(q0, omega, dt);
    EXPECT_FALSE(first_order.has_second_derivatives);
    const Vec4 direct_orientation = quaternion_from_angular_velocity(q0, omega, dt);
    const Mat43 direct_orientation_jacobian = dq_domega(q0, omega, dt);
    for (int row = 0; row < 4; ++row) {
        EXPECT_EQ(first_order.orientation[row], direct_orientation[row]);
        for (int column = 0; column < 3; ++column)
            EXPECT_EQ(first_order.orientation_jacobian(row, column), direct_orientation_jacobian(row, column));
    }

    const QuaternionOmegaKinematics second_order = quaternion_omega_kinematics(q0, omega, dt, true);
    EXPECT_TRUE(second_order.has_second_derivatives);
    const std::array<Mat33, 4> direct_orientation_hessians = d2q_domega2(q0, omega, dt);
    for (int coordinate = 0; coordinate < 4; ++coordinate) {
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column)
                EXPECT_EQ(second_order.orientation_hessians[coordinate](row, column), direct_orientation_hessians[coordinate](row, column));
        }
    }

    EXPECT_TRUE(dx_domega(X_centered, first_order).allFinite());
    for (const Mat33& hessian : d2x_domega2(X_centered, second_order))
        EXPECT_TRUE(hessian.allFinite());
}

TEST(RigidBodyIPCOmegaNodeDerivatives, CenteredDifferenceConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 target(-0.2, 0.6, 0.3);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 q0 = reference_quaternion_from_rotation_vector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());
    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const OmegaNodeEnergy exact =
            evaluate_omega_node_energy(X_centered, target, x_com, q0, omega, dt);
        Vec3 gradient_fd = Vec3::Zero();
        Mat33 hessian_fd = Mat33::Zero();

        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            const OmegaNodeEnergy plus =
                evaluate_omega_node_energy(X_centered, target, x_com, q0, omega + step, dt);
            const OmegaNodeEnergy minus =
                evaluate_omega_node_energy(X_centered, target, x_com, q0, omega - step, dt);
            gradient_fd[beta] = (plus.energy - minus.energy) / (2.0 * h);
            hessian_fd.col(beta) = (plus.gradient - minus.gradient) / (2.0 * h);
        }

        gradient_errors[hi] = (gradient_fd - exact.gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact.hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCInertialEnergy, TranslationDerivativesMatchEnergy) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com(0.31, -0.42, 0.18);
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    constexpr double dt = 0.31;
    const Mat33 I_hat = body_second_moment(masses, R_p);

    ASSERT_TRUE((masses[0] * R_p[0] + masses[1] * R_p[1] + masses[2] * R_p[2]).isZero(1.0e-14));

    const Vec3 exact_gradient = inertia_translation_gradient(x_com, x_com_n, v_com_n, dt, total_mass);
    const Mat33 exact_hessian = inertia_translation_hessian(total_mass);
    Vec3 gradient_fd = Vec3::Zero();
    Mat33 hessian_fd = Mat33::Zero();
    constexpr double h = 1.0e-5;

    for (int alpha = 0; alpha < 3; ++alpha) {
        Vec3 step = Vec3::Zero();
        step[alpha] = h;
        const double plus_energy = incremental_potential_energy(x_com + step, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, I_hat);
        const double minus_energy = incremental_potential_energy(x_com - step, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, I_hat);
        gradient_fd[alpha] = (plus_energy - minus_energy) / (2.0 * h);

        const Vec3 plus_gradient = inertia_translation_gradient(x_com + step, x_com_n, v_com_n, dt, total_mass);
        const Vec3 minus_gradient = inertia_translation_gradient(x_com - step, x_com_n, v_com_n, dt, total_mass);
        hessian_fd.col(alpha) = (plus_gradient - minus_gradient) / (2.0 * h);
    }

    EXPECT_TRUE(gradient_fd.isApprox(exact_gradient, 1.0e-9));
    EXPECT_TRUE(hessian_fd.isApprox(exact_hessian, 1.0e-9));
}

TEST(RigidBodyIPCInertialEnergy, OmegaDerivativesConvergeQuadratically) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    constexpr double dt = 0.31;
    const Vec3 x_com = x_com_n + dt * v_com_n;
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    const Mat33 I_hat = body_second_moment(masses, R_p);
    const auto [exact_gradient, exact_hessian] = inertia_rotation_gradient_hessian(omega, q_n, omega_n, dt, I_hat);
    const Vec3 gradient_only = inertia_rotation_gradient(omega, q_n, omega_n, dt, I_hat);
    for (int coordinate = 0; coordinate < 3; ++coordinate)
        EXPECT_EQ(gradient_only[coordinate], exact_gradient[coordinate]);
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Vec3 gradient_fd = Vec3::Zero();
        Mat33 hessian_fd = Mat33::Zero();

        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            const double plus_energy = incremental_potential_energy(x_com, omega + step, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, I_hat);
            const double minus_energy = incremental_potential_energy(x_com, omega - step, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, I_hat);
            gradient_fd[beta] = (plus_energy - minus_energy) / (2.0 * h);

            const Vec3 plus_gradient = inertia_rotation_gradient_hessian(omega + step, q_n, omega_n, dt, I_hat).first;
            const Vec3 minus_gradient = inertia_rotation_gradient_hessian(omega - step, q_n, omega_n, dt, I_hat).first;
            hessian_fd.col(beta) = (plus_gradient - minus_gradient) / (2.0 * h);
        }

        gradient_errors[hi] = (gradient_fd - exact_gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact_hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCInertialEnergy, CachedKinematicsAndPredictorAreBitwiseEquivalent) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> references = {Vec3(-0.8, 0.35, 0.2), Vec3(0.45, -0.55, 0.9), Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053)};
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    constexpr double dt = 0.31;
    const Mat33 I_hat = body_second_moment(masses, references);
    const Vec3 uncached_gradient = inertia_rotation_gradient(omega, q_n, omega_n, dt, I_hat);
    const auto [uncached_full_gradient, uncached_hessian] = inertia_rotation_gradient_hessian(omega, q_n, omega_n, dt, I_hat);
    const Mat33 predictor = rigid_rotation_predictor(q_n, omega_n, dt);
    const QuaternionOmegaKinematics first_order = quaternion_omega_kinematics(q_n, omega, dt);
    const QuaternionOmegaKinematics second_order = quaternion_omega_kinematics(q_n, omega, dt, true);
    const Vec3 cached_gradient = inertia_rotation_gradient(omega, q_n, omega_n, dt, I_hat, &first_order, &predictor);
    const auto [cached_full_gradient, cached_hessian] = inertia_rotation_gradient_hessian(omega, q_n, omega_n, dt, I_hat, &second_order, &predictor);
    EXPECT_EQ(std::memcmp(cached_gradient.data(), uncached_gradient.data(), sizeof(double) * static_cast<std::size_t>(cached_gradient.size())), 0);
    EXPECT_EQ(std::memcmp(cached_full_gradient.data(), uncached_full_gradient.data(), sizeof(double) * static_cast<std::size_t>(cached_full_gradient.size())), 0);
    EXPECT_EQ(std::memcmp(cached_hessian.data(), uncached_hessian.data(), sizeof(double) * static_cast<std::size_t>(cached_hessian.size())), 0);
}

TEST(RigidBodyIPCInertialEnergy, ReducedEnergyMatchesFullNodalMassQuadratic) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com(0.31, -0.42, 0.18);
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    constexpr double dt = 0.31;
    const Mat33 I_hat = body_second_moment(masses, R_p);

    ASSERT_TRUE((masses[0] * R_p[0] + masses[1] * R_p[1] + masses[2] * R_p[2]).isZero(1.0e-14));

    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt);
    const Vec4 q_dot_n = quaternion_time_derivative(quaternion_from_angular_velocity(q_n, -omega_n, dt), omega_n);
    const Vec4 q_n_inverse = quaternion_inverse(q_n);
    const Vec4 q_n_inverse_dot = -quaternion_multiply(
        q_n_inverse, quaternion_multiply(q_dot_n, q_n_inverse));
    double full_nodal_energy = 0.0;

    ASSERT_GT(std::abs(q_n.dot(q_dot_n)), 1.0e-3);

    for (std::size_t p = 0; p < R_p.size(); ++p) {
        const Vec3 r_p = quaternion_rotate(q, R_p[p]);
        const Vec3 r_p_n = quaternion_rotate(q_n, R_p[p]);
        const Vec4 R_p_quaternion(0.0, R_p[p][0], R_p[p][1], R_p[p][2]);
        const Vec4 first_term = quaternion_multiply(
            quaternion_multiply(q_dot_n, R_p_quaternion), q_n_inverse);
        const Vec4 second_term = quaternion_multiply(
            quaternion_multiply(q_n, R_p_quaternion), q_n_inverse_dot);
        const Vec3 r_dot_p_n = (first_term + second_term).tail<3>();
        const Vec3 x_p = x_com + r_p;
        const Vec3 v_p_n = v_com_n + r_dot_p_n;
        const Vec3 x_hat_p = x_com_n + r_p_n + dt * v_p_n;
        const Vec3 residual = x_p - x_hat_p;
        full_nodal_energy += 0.5 * masses[p] * residual.squaredNorm();
    }

    const double reduced_energy = incremental_potential_energy(x_com, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, I_hat);

    EXPECT_NEAR(reduced_energy, full_nodal_energy, 1.0e-14);
}

TEST(RigidBodyBroadPhase, BuildsBoxesCandidatesAndConflictGraph) {
    RefMesh ref_mesh;
    DeformedState state;
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const int first_rb = create_rigid_body({Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}, Vec3::Zero(), identity, Vec3::Zero(), 3.0, ref_mesh, state);
    const int second_rb = create_rigid_body({Vec3(0.2, 0.2, 0.1), Vec3(1.2, 0.2, 0.1), Vec3(0.2, 1.2, 0.1)}, Vec3::Zero(), identity, Vec3::Zero(), 3.0, ref_mesh, state);
    ref_mesh.tris = {ref_mesh.rb_nodes[first_rb][0], ref_mesh.rb_nodes[first_rb][1], ref_mesh.rb_nodes[first_rb][2], ref_mesh.rb_nodes[second_rb][0], ref_mesh.rb_nodes[second_rb][1], ref_mesh.rb_nodes[second_rb][2]};

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.d_hat = 0.15;
    params.node_box_min = 0.01;
    params.node_box_max = 0.01;

    std::vector<AABB> blue_boxes(state.deformed_positions.size());
    build_blue_boxes_rb(state.x_coms, state.orientations, std::vector<double>(2, M_PI), std::vector<double>(2, 0.01), ref_mesh, blue_boxes);
    BroadPhase broad_phase;
    broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat);

    const BroadPhase::Cache& cache = broad_phase.cache();
    ASSERT_EQ(cache.node_boxes.size(), state.deformed_positions.size());
    ASSERT_EQ(cache.tri_boxes.size(), 2);
    ASSERT_EQ(cache.red_edge_boxes.size(), 6);
    ASSERT_EQ(cache.edge_boxes.size(), 6);
    for (int node = 0; node < static_cast<int>(state.deformed_positions.size()); ++node) {
        EXPECT_TRUE(cache.node_boxes[node].min.isApprox(blue_boxes[node].min, 1.0e-14));
        EXPECT_TRUE(cache.node_boxes[node].max.isApprox(blue_boxes[node].max, 1.0e-14));
    }
    AABB expected_triangle_box = blue_boxes[0];
    expected_triangle_box.expand(blue_boxes[1]);
    expected_triangle_box.expand(blue_boxes[2]);
    expected_triangle_box.min -= Vec3::Constant(params.d_hat);
    expected_triangle_box.max += Vec3::Constant(params.d_hat);
    EXPECT_TRUE(cache.tri_boxes[0].min.isApprox(expected_triangle_box.min, 1.0e-14));
    EXPECT_TRUE(cache.tri_boxes[0].max.isApprox(expected_triangle_box.max, 1.0e-14));

    const auto edge = std::find(cache.edges.begin(), cache.edges.end(), std::array<int, 2>{0, 1});
    ASSERT_NE(edge, cache.edges.end());
    const int edge_index = static_cast<int>(edge - cache.edges.begin());
    AABB expected_red_edge_box = blue_boxes[0];
    expected_red_edge_box.expand(blue_boxes[1]);
    EXPECT_TRUE(cache.red_edge_boxes[edge_index].min.isApprox(expected_red_edge_box.min, 1.0e-14));
    EXPECT_TRUE(cache.red_edge_boxes[edge_index].max.isApprox(expected_red_edge_box.max, 1.0e-14));
    const AABB expected_green_edge_box(expected_red_edge_box.min - Vec3::Constant(params.d_hat), expected_red_edge_box.max + Vec3::Constant(params.d_hat));
    EXPECT_TRUE(cache.edge_boxes[edge_index].min.isApprox(expected_green_edge_box.min, 1.0e-14));
    EXPECT_TRUE(cache.edge_boxes[edge_index].max.isApprox(expected_green_edge_box.max, 1.0e-14));

    const int second_node = ref_mesh.rb_nodes[second_rb][0];
    EXPECT_NE(std::find_if(cache.nt_pairs.begin(), cache.nt_pairs.end(), [&](const NodeTrianglePair& pair) { return pair.node == second_node && pair.tri_v[0] == 0 && pair.tri_v[1] == 1 && pair.tri_v[2] == 2; }), cache.nt_pairs.end());
    std::vector<std::vector<int>> body_nt_pair_indices;
    std::vector<std::vector<int>> body_ss_pair_indices;
    std::vector<std::vector<int>> adjacency;
    build_rb_contact_adj(cache, ref_mesh.node_to_rb, 2, body_nt_pair_indices, body_ss_pair_indices, adjacency);
    EXPECT_EQ(adjacency, (std::vector<std::vector<int>>{{1}, {0}}));
    EXPECT_FALSE(body_nt_pair_indices[0].empty());
    EXPECT_FALSE(body_nt_pair_indices[1].empty());
}

TEST(RigidBodyIPCSolver, AddsNaiveRigidBarrierTranslationAndOrientationTerms) {
    RefMesh ref_mesh;
    DeformedState state;
    state.deformed_positions = {Vec3(-1.0, -1.0, 0.0), Vec3(3.0, -1.0, 0.0), Vec3(-1.0, 3.0, 0.0)};
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.tris = {0, 1, 2};

    const std::vector<Vec3> rigid_nodes = {Vec3(0.0, 0.0, 0.2), Vec3(3.0, 0.0, 1.0), Vec3(0.0, 3.0, 1.0), Vec3(3.0, 3.0, 1.0)};
    constexpr double total_mass = 4.0;
    const int rb = create_rigid_body(rigid_nodes, Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(), total_mass, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.d_hat = 0.5;
    params.k_barrier = 100.0;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 1.0;
    params.node_box_min = 10.0;
    params.node_box_max = 10.0;
    params.theta_box_min = M_PI;
    params.theta_box_max = M_PI;
    const double dt = params.dt();
    const double barrier_scale = dt * dt * params.k_barrier;

    {
        SimParams residual_params = params;
        residual_params.fixed_iters = false;
        residual_params.max_global_iters = 0;
        std::vector<Vec3> residual_x_com_new = state.x_coms;
        std::vector<Vec4> residual_q_new = state.orientations;
        std::vector<Vec3> residual_omega_new(
            state.omega.size(), Vec3::Zero());
        const SolverResult residual_result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, residual_params, residual_x_com_new, residual_q_new, residual_omega_new);
        EXPECT_TRUE(residual_result.has_residual);
        EXPECT_GT(residual_result.initial_residual, 1.0e-8);
    }

    const auto barrier_at = [&](const Vec3& x_com, const Vec3& omega) {
        RigidEnergyDerivatives total;
        for (int local = 0; local < static_cast<int>(ref_mesh.ref_positions[rb].size()); ++local) {
            const Vec3 x = world_space_position(ref_mesh.ref_positions[rb][local], x_com, state.orientations[rb], omega, dt);
            const std::array<Vec3, 4> references = {ref_mesh.ref_positions[rb][local], Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
            const RigidEnergyDerivatives contribution = node_triangle_barrier_rb(x, state.deformed_positions[0], state.deformed_positions[1], state.deformed_positions[2], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega, dt, params.d_hat);
            total.translation_gradient += contribution.translation_gradient;
            total.orientation_gradient += contribution.orientation_gradient;
            total.translation_translation_hessian += contribution.translation_translation_hessian;
            total.translation_orientation_hessian += contribution.translation_orientation_hessian;
            total.orientation_orientation_hessian += contribution.orientation_orientation_hessian;
        }
        return total;
    };

    const Vec3 initial_com = state.x_coms[rb];
    const Vec3 initial_omega = state.omega[rb];
    const RigidEnergyDerivatives initial_barrier = barrier_at(initial_com, initial_omega);
    Vec3 expected_com_gradient = inertia_translation_gradient(initial_com, state.x_coms[rb], state.v_coms[rb], dt, total_mass);
    Mat33 expected_com_hessian = inertia_translation_hessian(total_mass);
    expected_com_gradient += barrier_scale * initial_barrier.translation_gradient;
    expected_com_hessian += barrier_scale * initial_barrier.translation_translation_hessian;
    const Vec3 expected_com = initial_com - expected_com_hessian.ldlt().solve(expected_com_gradient);

    auto [expected_omega_gradient, expected_omega_hessian] = inertia_rotation_gradient_hessian(initial_omega, state.orientations[rb], state.omega[rb], dt, ref_mesh.I_hat[rb]);
    const RigidEnergyDerivatives updated_barrier = barrier_at(expected_com, initial_omega);
    expected_omega_gradient += barrier_scale * updated_barrier.orientation_gradient;
    expected_omega_hessian += barrier_scale * updated_barrier.orientation_orientation_hessian;
    const Vec3 expected_newton_omega =
        initial_omega - expected_omega_hessian.ldlt().solve(expected_omega_gradient);
    const Vec4 q_n = quaternion_normalize(state.orientations[rb]);
    const Vec4 expected_q = quaternion_normalize(
        quaternion_from_angular_velocity(q_n, expected_newton_omega, dt));

    std::vector<Vec3> x_com_new = state.x_coms;
    std::vector<Vec4> q_new = state.orientations;
    std::vector<Vec3> omega_new(state.omega.size(), Vec3::Zero());
    const SolverResult result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, x_com_new, q_new, omega_new);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_GT(x_com_new[rb].z(), initial_com.z());
    EXPECT_GT(omega_new[rb].norm(), 1.0e-8);
    EXPECT_TRUE(x_com_new[rb].isApprox(expected_com, 1.0e-11));
    EXPECT_TRUE(q_new[rb].isApprox(expected_q, 1.0e-11));
    EXPECT_TRUE(omega_new[rb].isApprox(
        expected_newton_omega, 1.0e-11));
}

TEST(GeneralSolver, AdvancesDeformableAndRigidInertiaInOneSweep) {
    RefMesh ref_mesh;
    DeformedState state;
    state.deformed_positions = {
        Vec3(-1.0, -1.0, 0.0),
        Vec3( 3.0, -1.0, 0.0),
        Vec3(-1.0,  3.0, 0.0),
    };
    state.velocities = {
        Vec3(0.2, -0.3, 0.1),
        Vec3(0.2, -0.3, 0.1),
        Vec3(0.2, -0.3, 0.1),
    };
    ref_mesh.tris = {0, 1, 2};
    ref_mesh.initialize(
        {Vec2(-1.0, -1.0), Vec2(3.0, -1.0), Vec2(-1.0, 3.0)},
        state.deformed_positions);
    ASSERT_TRUE(ref_mesh.Dm_inverse[0].allFinite());
    ref_mesh.mass.assign(3, 2.0);
    ref_mesh.node_to_rb.assign(3, -1);

    const Vec3 rigid_velocity(-0.4, 0.25, 0.3);
    const Vec3 rigid_omega(0.35, -0.2, 0.15);
    const int rb = create_rigid_body(
        {Vec3(10.0, 0.0, 0.0), Vec3(11.0, 0.0, 0.0),
         Vec3(10.0, 1.0, 0.0), Vec3(10.0, 0.0, 1.0)},
        rigid_velocity, Vec4(1.0, 0.0, 0.0, 0.0), rigid_omega,
        4.0, ref_mesh, state);
    ref_mesh.build_deformable_nodes();

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 1.0;
    params.node_box_min = 2.0;
    params.node_box_max = 2.0;
    params.theta_box_min = M_PI;
    params.theta_box_max = M_PI;
    params.node_box_update_count = 1;
    params.use_ccd = false;
    params.use_parallel = true;
    params.gravity = Vec3(0.0, -1.1, 0.0);
    const double dt = params.dt();

    const VertexTriangleMap adj =
        build_incident_triangle_map(ref_mesh.tris);
    std::vector<Vec3> xhat;
    build_xhat(
        xhat, state.deformed_positions, state.velocities, dt);
    // General rigid motion must not depend on per-proxy xhat values.
    for (const int node : ref_mesh.rb_nodes[rb])
        xhat[node] = Vec3(123.0, -456.0, 789.0);

    std::vector<Vec3> xnew = state.deformed_positions;
    std::vector<Vec3> x_com_new = state.x_coms;
    std::vector<Vec4> q_new = state.orientations;
    std::vector<Vec3> omega_new(state.omega.size(), Vec3::Zero());
    BroadPhase broad_phase;

    const auto [omega_gradient, omega_hessian] =
        inertia_rotation_gradient_hessian(
            Vec3::Zero(), state.orientations[rb], state.omega[rb],
            dt, ref_mesh.I_hat[rb]);
    const Vec3 expected_omega =
        -omega_hessian.ldlt().solve(omega_gradient);
    const Vec4 expected_orientation = quaternion_normalize(
        quaternion_from_angular_velocity(
            state.orientations[rb], expected_omega, dt));
    const Vec3 expected_com =
        state.x_coms[rb] + dt * state.v_coms[rb]
        + dt * dt * params.gravity;

    const SolverResult result =
        global_gauss_seidel_solver_basic_general(
            ref_mesh, state, adj, {}, params, xnew, xhat,
            x_com_new, q_new, omega_new, broad_phase);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    for (int node = 0; node < 3; ++node) {
        const Vec3 expected_position =
            xhat[node] + dt * dt * params.gravity;
        EXPECT_TRUE(xnew[node].isApprox(
            expected_position, 1.0e-12));
    }
    EXPECT_TRUE(x_com_new[rb].isApprox(expected_com, 1.0e-12));
    EXPECT_TRUE(omega_new[rb].isApprox(expected_omega, 1.0e-11));
    EXPECT_TRUE(q_new[rb].isApprox(expected_orientation, 1.0e-11));
    for (int local = 0;
         local < static_cast<int>(ref_mesh.rb_nodes[rb].size());
         ++local) {
        const Vec3 expected_position = world_space_position(
            ref_mesh.ref_positions[rb][local], expected_com,
            expected_orientation);
        EXPECT_TRUE(xnew[ref_mesh.rb_nodes[rb][local]].isApprox(
            expected_position, 1.0e-11));
    }

    // The frame driver commits both subsystems only after the shared solve,
    // reconstructs their velocities, and then synchronizes rigid proxies.
    DeformedState committed_state = state;
    std::vector<Pin> no_pins;
    BroadPhase frame_broad_phase;
    const SolverResult frame_result = advance_one_frame_general(
        committed_state, ref_mesh, adj, no_pins, params,
        frame_broad_phase);
    EXPECT_TRUE(frame_result.converged);
    for (int node = 0; node < 3; ++node) {
        const Vec3 expected_position =
            xhat[node] + dt * dt * params.gravity;
        const Vec3 expected_velocity =
            state.velocities[node] + dt * params.gravity;
        EXPECT_TRUE(committed_state.deformed_positions[node].isApprox(
            expected_position, 1.0e-12));
        EXPECT_TRUE(committed_state.velocities[node].isApprox(
            expected_velocity, 1.0e-12));
    }
    EXPECT_TRUE(committed_state.x_coms[rb].isApprox(
        expected_com, 1.0e-12));
    EXPECT_TRUE(committed_state.v_coms[rb].isApprox(
        state.v_coms[rb] + dt * params.gravity, 1.0e-12));
    EXPECT_TRUE(committed_state.orientations[rb].isApprox(
        expected_orientation, 1.0e-11));
    EXPECT_TRUE(committed_state.omega[rb].isApprox(
        expected_omega, 1.0e-11));
    for (int local = 0;
         local < static_cast<int>(ref_mesh.rb_nodes[rb].size());
         ++local) {
        const int node = ref_mesh.rb_nodes[rb][local];
        const Vec3 expected_position = world_space_position(
            ref_mesh.ref_positions[rb][local], expected_com,
            expected_orientation);
        EXPECT_TRUE(committed_state.deformed_positions[node].isApprox(
            expected_position, 1.0e-11));
    }
}

TEST(GeneralSolver, RelativeToleranceCannotHideUnconvergedCloth) {
    RefMesh ref_mesh;
    DeformedState state;
    state.deformed_positions = {
        Vec3(-1.0, -1.0, 0.0),
        Vec3( 1.0, -1.0, 0.0),
        Vec3(-1.0,  1.0, 0.0),
    };
    state.velocities.assign(3, Vec3::UnitX());
    ref_mesh.tris = {0, 1, 2};
    ref_mesh.initialize(
        {Vec2(-1.0, -1.0), Vec2(1.0, -1.0), Vec2(-1.0, 1.0)},
        state.deformed_positions);
    ref_mesh.mass.assign(3, 1.0);
    ref_mesh.node_to_rb.assign(3, -1);

    constexpr double rigid_mass = 1.0e6;
    create_rigid_body(
        {Vec3(10.0, 0.0, 0.0), Vec3(11.0, 0.0, 0.0),
         Vec3(10.0, 1.0, 0.0), Vec3(10.0, 0.0, 1.0)},
        Vec3(0.01, 0.0, 0.0),
        Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(),
        rigid_mass, ref_mesh, state);
    ref_mesh.build_deformable_nodes();

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.max_global_iters = 1;
    params.fixed_iters = false;
    params.tol_abs = 0.0;
    params.tol_rel = 1.0e-3;
    params.damping = 1.0;
    params.node_box_min = 0.002;
    params.node_box_max = 0.002;
    params.theta_box_min = 0.01;
    params.theta_box_max = 0.01;
    params.node_box_update_count = 1;
    params.use_ccd = false;

    const DeformedState initial_state = state;
    const VertexTriangleMap adj =
        build_incident_triangle_map(ref_mesh.tris);
    BroadPhase broad_phase;
    std::vector<Pin> pins;
    const SolverResult result = advance_one_frame_general(
        state, ref_mesh, adj, pins, params, broad_phase);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_TRUE(result.has_residual_components);
    EXPECT_DOUBLE_EQ(
        result.initial_residual,
        result.initial_cloth_residual
            + result.initial_rigid_residual);
    EXPECT_DOUBLE_EQ(
        result.final_residual,
        result.final_cloth_residual
            + result.final_rigid_residual);
    EXPECT_EQ(result.iterations, 1);
    ASSERT_EQ(state.deformed_positions.size(),
              initial_state.deformed_positions.size());
    for (int node = 0;
         node < static_cast<int>(state.deformed_positions.size());
         ++node) {
        EXPECT_TRUE(state.deformed_positions[node].isApprox(
            initial_state.deformed_positions[node], 0.0));
        EXPECT_TRUE(state.velocities[node].isApprox(
            initial_state.velocities[node], 0.0));
    }
    ASSERT_EQ(state.x_coms.size(), initial_state.x_coms.size());
    for (int rb = 0; rb < static_cast<int>(state.x_coms.size()); ++rb) {
        EXPECT_TRUE(state.x_coms[rb].isApprox(
            initial_state.x_coms[rb], 0.0));
        EXPECT_TRUE(state.v_coms[rb].isApprox(
            initial_state.v_coms[rb], 0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            initial_state.orientations[rb], 0.0));
        EXPECT_TRUE(state.omega[rb].isApprox(
            initial_state.omega[rb], 0.0));
    }
}

TEST(GeneralSolver, ClothRigidBarrierMovesBothSidesApart) {
    RefMesh ref_mesh;
    DeformedState state;
    state.deformed_positions = {
        Vec3(-1.0, -1.0, 0.0),
        Vec3( 3.0, -1.0, 0.0),
        Vec3(-1.0,  3.0, 0.0),
    };
    state.velocities.assign(3, Vec3::Zero());
    ref_mesh.tris = {0, 1, 2};
    ref_mesh.initialize(
        {Vec2(-1.0, -1.0), Vec2(3.0, -1.0), Vec2(-1.0, 3.0)},
        state.deformed_positions);
    ref_mesh.mass.assign(3, 10.0);
    ref_mesh.node_to_rb.assign(3, -1);

    const Vec3 center(0.0, 0.0, 0.2);
    const int rb = create_rigid_body(
        {center,
         center + 5.0 * Vec3::UnitX(), center - 5.0 * Vec3::UnitX(),
         center + 5.0 * Vec3::UnitY(), center - 5.0 * Vec3::UnitY(),
         center + 5.0 * Vec3::UnitZ(), center - 5.0 * Vec3::UnitZ()},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(),
        7.0, ref_mesh, state);
    ref_mesh.build_deformable_nodes();
    const int contact_node = ref_mesh.rb_nodes[rb][0];

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.d_hat = 0.5;
    params.k_barrier = 100.0;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 0.25;
    params.node_box_min = 1.0;
    params.node_box_max = 1.0;
    params.theta_box_min = M_PI;
    params.theta_box_max = M_PI;
    params.node_box_update_count = 1;
    params.use_ccd = false;
    params.use_parallel = true;

    const VertexTriangleMap adj =
        build_incident_triangle_map(ref_mesh.tris);
    std::vector<Vec3> xhat = state.deformed_positions;
    std::vector<Vec3> xnew = state.deformed_positions;
    std::vector<Vec3> x_com_new = state.x_coms;
    std::vector<Vec4> q_new = state.orientations;
    std::vector<Vec3> omega_new(state.omega.size(), Vec3::Zero());
    BroadPhase broad_phase;

    const Vec3 initial_cloth_centroid =
        (xnew[0] + xnew[1] + xnew[2]) / 3.0;
    const Vec3 initial_com = x_com_new[rb];
    const double initial_distance = node_triangle_distance(
        xnew[contact_node], xnew[0], xnew[1], xnew[2]).distance;

    const SolverResult result =
        global_gauss_seidel_solver_basic_general(
            ref_mesh, state, adj, {}, params, xnew, xhat,
            x_com_new, q_new, omega_new, broad_phase);

    const Vec3 final_cloth_centroid =
        (xnew[0] + xnew[1] + xnew[2]) / 3.0;
    const double final_distance = node_triangle_distance(
        xnew[contact_node], xnew[0], xnew[1], xnew[2]).distance;
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LT(final_cloth_centroid.z(), initial_cloth_centroid.z());
    EXPECT_GT(x_com_new[rb].z(), initial_com.z());
    EXPECT_GT(final_distance, initial_distance);
    EXPECT_NEAR(omega_new[rb].norm(), 0.0, 1.0e-11);
    for (int local = 0;
         local < static_cast<int>(ref_mesh.rb_nodes[rb].size());
         ++local) {
        const Vec3 expected_position = world_space_position(
            ref_mesh.ref_positions[rb][local], x_com_new[rb], q_new[rb]);
        EXPECT_TRUE(xnew[ref_mesh.rb_nodes[rb][local]].isApprox(
            expected_position, 1.0e-11));
    }
}

TEST(DeformableResidual, UsesOnlySuppliedDeformableNodes) {
    RefMesh ref_mesh;
    ref_mesh.mass = {2.0, 4.0};
    ref_mesh.node_to_rb = {-1, 0};

    VertexTriangleMap adj;
    adj.emplace(0, IncidentTriangles{});
    adj.emplace(1, IncidentTriangles{});

    const std::vector<Vec3> xhat = {Vec3::Zero(), Vec3::Zero()};
    const std::vector<Vec3> x = {Vec3::Zero(), 9.0 * Vec3::UnitX()};
    const std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    BroadPhase broad_phase;

    const double deformable_only = compute_global_deformable_residual(
        ref_mesh, adj, pins, params, x, xhat, broad_phase,
        std::vector<int>{0});
    const double both_nodes = compute_global_deformable_residual(
        ref_mesh, adj, pins, params, x, xhat, broad_phase,
        std::vector<int>{0, 1});

    EXPECT_DOUBLE_EQ(deformable_only, 0.0);
    EXPECT_DOUBLE_EQ(both_nodes, 9.0);
}

TEST(DeformableResidual, CachedIncidenceAndShapeGradientsAreEquivalent) {
    RefMesh ref_mesh;
    ref_mesh.tris = {0, 1, 2, 1, 3, 2};
    const std::vector<Vec3> x_rest = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
    };
    ref_mesh.initialize({Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.0, 1.0), Vec2(1.0, 1.0)}, x_rest);
    ref_mesh.mass = {2.0, 3.0, 4.0, 5.0};

    std::vector<Vec3> x = x_rest;
    x[0] += Vec3(0.03, -0.02, 0.04);
    x[1] += Vec3(-0.01, 0.05, -0.02);
    x[2] += Vec3(0.02, 0.01, 0.03);
    x[3] += Vec3(-0.04, -0.01, 0.02);
    std::vector<Vec3> xhat = x_rest;
    xhat[0] += Vec3(-0.01, 0.02, 0.01);
    xhat[3] += Vec3(0.02, -0.03, -0.01);

    const std::vector<Pin> pins = {
        Pin{1, x_rest[1] + Vec3(0.02, -0.01, 0.03)},
    };
    SimParams params = SimParams::zeros();
    params.fps = 7.0;
    params.substeps = 2;
    params.mu = 2.7;
    params.lambda = 4.1;
    params.gravity = Vec3(0.3, -1.2, 0.4);
    params.kpin = 3.6;

    const VertexTriangleMap adjacency = build_incident_triangle_map(ref_mesh.tris);
    std::vector<IncidentTriangles> incident_triangles(x.size());
    for (const auto& [node, incident] : adjacency)
        incident_triangles[static_cast<std::size_t>(node)] = incident;
    std::vector<ShapeGrads> rest_shape_grads(ref_mesh.Dm_inverse.size());
    for (std::size_t triangle = 0; triangle < ref_mesh.Dm_inverse.size(); ++triangle) {
        rest_shape_grads[triangle] = shape_function_gradients(ref_mesh.Dm_inverse[triangle]);
    }
    const PinMap pin_map = build_pin_map(pins, static_cast<int>(x.size()));
    const std::vector<int> deformable_nodes = {0, 1, 2, 3};
    BroadPhase broad_phase;

    const double uncached = compute_global_deformable_residual(ref_mesh, adjacency, pins, params, x, xhat, broad_phase, deformable_nodes, &pin_map);
    const VertexTriangleMap empty_adjacency;
    const double cached = compute_global_deformable_residual(ref_mesh, empty_adjacency, pins, params, x, xhat, broad_phase, deformable_nodes, &pin_map, &incident_triangles, &rest_shape_grads);

    EXPECT_DOUBLE_EQ(cached, uncached);
}

TEST(DeformableResidual, FrozenWorkspaceIsBitwiseEquivalentForElasticBendingAndContactGradients) {
    RefMesh ref_mesh;
    ref_mesh.tris = {0, 1, 2, 1, 3, 2};
    const std::vector<Vec3> x_rest = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(1.0, 1.0, 0.0)};
    ref_mesh.initialize({Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.0, 1.0), Vec2(1.0, 1.0)}, x_rest);
    ref_mesh.num_positions = 8;
    ref_mesh.mass = {2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5};

    const std::vector<Vec3> x = {Vec3(0.03, -0.02, 0.04), Vec3(0.99, 0.05, -0.02), Vec3(0.02, 1.01, 0.03), Vec3(0.96, 0.99, 0.14), Vec3(0.25, 0.25, 0.18), Vec3(0.75, 0.25, 0.18), Vec3(0.50, 0.05, 0.34), Vec3(0.50, 0.75, 0.34)};
    std::vector<Vec3> xhat = x;
    for (int node = 0; node < static_cast<int>(xhat.size()); ++node) xhat[static_cast<std::size_t>(node)] += static_cast<double>(node + 1) * Vec3(0.001, -0.002, 0.0015);

    SimParams params = SimParams::zeros();
    params.fps = 8.0;
    params.substeps = 2;
    params.mu = 2.7;
    params.lambda = 4.1;
    params.kB = 1.9;
    params.d_hat = 0.4;
    params.k_barrier = 7.3;
    params.gravity = Vec3(0.3, -1.2, 0.4);
    params.kpin = 3.6;
    params.use_parallel = false;
    const std::vector<Pin> pins = {Pin{1, x_rest[1] + Vec3(0.02, -0.01, 0.03)}};
    const PinMap pin_map = build_pin_map(pins, static_cast<int>(x.size()));

    const VertexTriangleMap adjacency = build_incident_triangle_map(ref_mesh.tris);
    std::vector<IncidentTriangles> incident_triangles(x.size());
    for (const auto& [node, incident] : adjacency) incident_triangles[static_cast<std::size_t>(node)] = incident;
    std::vector<ShapeGrads> rest_shape_grads(ref_mesh.Dm_inverse.size());
    for (std::size_t triangle = 0; triangle < rest_shape_grads.size(); ++triangle) rest_shape_grads[triangle] = shape_function_gradients(ref_mesh.Dm_inverse[triangle]);

    BroadPhase broad_phase;
    BroadPhase::Cache& broad_phase_cache = broad_phase.mutable_cache();
    broad_phase_cache.vertex_nt.resize(x.size());
    broad_phase_cache.vertex_ss.resize(x.size());
    broad_phase_cache.nt_pairs.push_back(NodeTrianglePair{4, {0, 1, 2}});
    const int nt_nodes[4] = {4, 0, 1, 2};
    for (int role = 0; role < 4; ++role) broad_phase_cache.vertex_nt[static_cast<std::size_t>(nt_nodes[role])].push_back({0, role});
    broad_phase_cache.ss_pairs.push_back(SegmentSegmentPair{{4, 5, 6, 7}});
    for (int role = 0; role < 4; ++role) broad_phase_cache.vertex_ss[static_cast<std::size_t>(4 + role)].push_back({0, role});

    FrozenResidualWorkspace workspace;
    build_frozen_residual_workspace(ref_mesh, params, x, broad_phase, workspace, &rest_shape_grads);
    ASSERT_EQ(workspace.cloth_triangle_gradients.size(), 2u);
    ASSERT_EQ(workspace.hinge_gradients.size(), 1u);
    ASSERT_EQ(workspace.nt_gradients.size(), 1u);
    ASSERT_EQ(workspace.ss_gradients.size(), 1u);
    ASSERT_EQ(workspace.nt_aabb_active[0], 1);
    ASSERT_EQ(workspace.ss_aabb_active[0], 1);
    ASSERT_EQ(workspace.nt_barrier_active[0], 1);
    ASSERT_EQ(workspace.ss_barrier_active[0], 1);
    ASSERT_EQ(workspace.nt_gradient_cached[0], 1);
    ASSERT_EQ(workspace.ss_gradient_cached[0], 1);

    for (int triangle = 0; triangle < num_tris(ref_mesh); ++triangle) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, triangle);
        Mat32 Ds;
        Ds.col(0) = def.x[1] - def.x[0];
        Ds.col(1) = def.x[2] - def.x[0];
        const Mat32 F = Ds * ref_mesh.Dm_inverse[static_cast<std::size_t>(triangle)];
        const CorotatedCache32 cache = buildCorotatedCache(F);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        for (int role = 0; role < 3; ++role) {
            const Vec3 expected = corotated_node_gradient(P, ref_mesh.area[static_cast<std::size_t>(triangle)], rest_shape_grads[static_cast<std::size_t>(triangle)], role);
            for (int component = 0; component < 3; ++component) EXPECT_EQ(workspace.cloth_triangle_gradients[static_cast<std::size_t>(triangle)][static_cast<std::size_t>(role)][component], expected[component]);
        }
    }
    const Hinge& hinge = ref_mesh.hinges[0];
    HingeDef hinge_def;
    for (int role = 0; role < 4; ++role) hinge_def.x[role] = x[static_cast<std::size_t>(hinge.v[role])];
    for (int role = 0; role < 4; ++role) {
        const Vec3 expected = bending_node_gradient(hinge_def, params.kB, hinge.c_e, hinge.bar_theta, role);
        for (int component = 0; component < 3; ++component) EXPECT_EQ(workspace.hinge_gradients[0][static_cast<std::size_t>(role)][component], expected[component]);
    }
    const NodeTrianglePair& nt_pair = broad_phase_cache.nt_pairs[0];
    const SegmentSegmentPair& ss_pair = broad_phase_cache.ss_pairs[0];
    for (int role = 0; role < 4; ++role) {
        const Vec3 expected_nt = node_triangle_barrier_gradient(x[static_cast<std::size_t>(nt_pair.node)], x[static_cast<std::size_t>(nt_pair.tri_v[0])], x[static_cast<std::size_t>(nt_pair.tri_v[1])], x[static_cast<std::size_t>(nt_pair.tri_v[2])], params.d_hat, role);
        const Vec3 expected_ss = segment_segment_barrier_gradient(x[static_cast<std::size_t>(ss_pair.v[0])], x[static_cast<std::size_t>(ss_pair.v[1])], x[static_cast<std::size_t>(ss_pair.v[2])], x[static_cast<std::size_t>(ss_pair.v[3])], params.d_hat, role);
        for (int component = 0; component < 3; ++component) {
            EXPECT_EQ(workspace.nt_gradients[0][static_cast<std::size_t>(role)][component], expected_nt[component]);
            EXPECT_EQ(workspace.ss_gradients[0][static_cast<std::size_t>(role)][component], expected_ss[component]);
        }
    }

    for (int node = 0; node < static_cast<int>(x.size()); ++node) {
        const std::vector<int> one_node = {node};
        const double uncached = compute_global_deformable_residual(ref_mesh, adjacency, pins, params, x, xhat, broad_phase, one_node, &pin_map, &incident_triangles, &rest_shape_grads);
        const double cached = compute_global_deformable_residual(ref_mesh, adjacency, pins, params, x, xhat, broad_phase, one_node, &pin_map, &incident_triangles, &rest_shape_grads, &workspace);
        EXPECT_EQ(cached, uncached) << "node " << node;
    }
    const std::vector<int> all_nodes = {0, 1, 2, 3, 4, 5, 6, 7};
    const double uncached = compute_global_deformable_residual(ref_mesh, adjacency, pins, params, x, xhat, broad_phase, all_nodes, &pin_map, &incident_triangles, &rest_shape_grads);
    const double cached = compute_global_deformable_residual(ref_mesh, adjacency, pins, params, x, xhat, broad_phase, all_nodes, &pin_map, &incident_triangles, &rest_shape_grads, &workspace);
    EXPECT_EQ(cached, uncached);
}

TEST(DeformableResidual, FrozenWorkspaceDistinguishesAabbCandidatesAtBarrierBoundary) {
    RefMesh ref_mesh;
    const std::vector<Vec3> x = {Vec3(0.25, 0.25, 0.5), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.5), Vec3(1.0, 0.0, 0.5)};
    SimParams params = SimParams::zeros();
    params.d_hat = 0.5;
    params.k_barrier = 1.0;
    params.use_parallel = false;
    BroadPhase broad_phase;
    broad_phase.mutable_cache().nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
    broad_phase.mutable_cache().ss_pairs.push_back(SegmentSegmentPair{{4, 5, 6, 7}});

    FrozenResidualWorkspace workspace;
    build_frozen_residual_workspace(ref_mesh, params, x, broad_phase, workspace);

    ASSERT_EQ(workspace.nt_aabb_active.size(), 1u);
    ASSERT_EQ(workspace.ss_aabb_active.size(), 1u);
    EXPECT_EQ(workspace.nt_aabb_active[0], 1);
    EXPECT_EQ(workspace.ss_aabb_active[0], 1);
    EXPECT_EQ(workspace.nt_barrier_active[0], 0);
    EXPECT_EQ(workspace.ss_barrier_active[0], 0);
    EXPECT_EQ(workspace.nt_gradient_cached[0], 1);
    EXPECT_EQ(workspace.ss_gradient_cached[0], 1);
    for (int role = 0; role < 4; ++role) {
        EXPECT_TRUE(workspace.nt_gradients[0][static_cast<std::size_t>(role)].isZero(0.0));
        EXPECT_TRUE(workspace.ss_gradients[0][static_cast<std::size_t>(role)].isZero(0.0));
    }
}

TEST(DeformableResidual, FrozenWorkspaceDefersZeroDistanceGradientFailureUntilConsumed) {
    RefMesh ref_mesh;
    ref_mesh.mass.assign(8, 1.0);
    const std::vector<Vec3> x = {Vec3(0.25, 0.25, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.5, -0.5, 0.0), Vec3(0.5, 0.5, 0.0)};
    VertexTriangleMap adjacency;
    for (int node = 0; node < static_cast<int>(x.size()); ++node) adjacency.emplace(node, IncidentTriangles{});
    SimParams params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.d_hat = 0.5;
    params.k_barrier = 1.0;
    params.use_parallel = false;
    BroadPhase broad_phase;
    BroadPhase::Cache& broad_phase_cache = broad_phase.mutable_cache();
    broad_phase_cache.vertex_nt.resize(x.size());
    broad_phase_cache.vertex_ss.resize(x.size());
    broad_phase_cache.nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
    const int nt_nodes[4] = {0, 1, 2, 3};
    for (int role = 0; role < 4; ++role) broad_phase_cache.vertex_nt[static_cast<std::size_t>(nt_nodes[role])].push_back({0, role});
    broad_phase_cache.ss_pairs.push_back(SegmentSegmentPair{{4, 5, 6, 7}});
    for (int role = 0; role < 4; ++role) broad_phase_cache.vertex_ss[static_cast<std::size_t>(4 + role)].push_back({0, role});

    FrozenResidualWorkspace workspace;
    EXPECT_NO_THROW(build_frozen_residual_workspace(ref_mesh, params, x, broad_phase, workspace));
    ASSERT_EQ(workspace.nt_aabb_active[0], 1);
    ASSERT_EQ(workspace.ss_aabb_active[0], 1);
    ASSERT_EQ(workspace.nt_barrier_active[0], 1);
    ASSERT_EQ(workspace.ss_barrier_active[0], 1);
    ASSERT_EQ(workspace.nt_gradient_cached[0], 0);
    ASSERT_EQ(workspace.ss_gradient_cached[0], 0);

    const std::vector<int> nt_consumer = {0};
    const std::vector<int> ss_consumer = {4};
    const std::string original_death_test_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH({ (void)compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, x, broad_phase, nt_consumer); }, "delta must be nonzero");
    EXPECT_DEATH({ (void)compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, x, broad_phase, nt_consumer, nullptr, nullptr, nullptr, &workspace); }, "delta must be nonzero");
    EXPECT_DEATH({ (void)compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, x, broad_phase, ss_consumer); }, "delta must be nonzero");
    EXPECT_DEATH({ (void)compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, x, broad_phase, ss_consumer, nullptr, nullptr, nullptr, &workspace); }, "delta must be nonzero");
    GTEST_FLAG_SET(death_test_style, original_death_test_style);
}

TEST(DeformableResidual, FarCachedContactsLeaveResidualUnchanged) {
    constexpr int num_nodes = 8;
    RefMesh ref_mesh;
    ref_mesh.mass.assign(num_nodes, 1.0);

    const std::vector<Vec3> x = {
        Vec3(100.0, 100.0, 100.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(100.0, 0.0, 0.0),
        Vec3(101.0, 0.0, 0.0),
        Vec3(0.0, 100.0, 0.0),
        Vec3(1.0, 100.0, 0.0),
    };
    std::vector<Vec3> xhat = x;
    for (int node = 0; node < num_nodes; ++node) {
        const double scale = static_cast<double>(node + 1);
        xhat[static_cast<std::size_t>(node)] += scale * Vec3(0.01, -0.02, 0.03);
    }

    VertexTriangleMap adjacency;
    for (int node = 0; node < num_nodes; ++node)
        adjacency.emplace(node, IncidentTriangles{});
    const std::vector<int> deformable_nodes = {0, 1, 2, 3, 4, 5, 6, 7};

    BroadPhase broad_phase;
    BroadPhase::Cache& cache = broad_phase.mutable_cache();
    cache.vertex_nt.resize(num_nodes);
    cache.vertex_ss.resize(num_nodes);
    cache.nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
    const int nt_nodes[4] = {0, 1, 2, 3};
    for (int dof = 0; dof < 4; ++dof)
        cache.vertex_nt[nt_nodes[dof]].push_back({0, dof});
    cache.ss_pairs.push_back(SegmentSegmentPair{{4, 5, 6, 7}});
    for (int dof = 0; dof < 4; ++dof)
        cache.vertex_ss[4 + dof].push_back({0, dof});

    SimParams params = SimParams::zeros();
    params.d_hat = 0.5;
    params.k_barrier = 9.0;
    const double with_far_candidates = compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, xhat, broad_phase, deformable_nodes);
    params.k_barrier = 0.0;
    const double without_barrier = compute_global_deformable_residual(ref_mesh, adjacency, {}, params, x, xhat, broad_phase, deformable_nodes);

    EXPECT_DOUBLE_EQ(with_far_candidates, without_barrier);
    EXPECT_DOUBLE_EQ(with_far_candidates, 0.24);
}

TEST(BoundQuaternion, ClipsAtFirstExitOfLongArc) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target_270 = quaternion_from_angular_velocity(identity, 1.5 * M_PI * Vec3::UnitZ(), 1.0);
    const Vec4 expected_90 = quaternion_from_angular_velocity(identity, 0.5 * M_PI * Vec3::UnitZ(), 1.0);

    EXPECT_TRUE(bound_quaternion(identity, identity, target_270, 0.5 * M_PI).isApprox(expected_90, 1.0e-14));
    EXPECT_TRUE(bound_quaternion(identity, identity, -target_270, 0.5 * M_PI).isApprox(-target_270, 1.0e-14));
}

TEST(BoundQuaternion, AllowsInwardMotionBeforeFirstExit) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 current = quaternion_from_angular_velocity(identity, (80.0 * M_PI / 180.0) * Vec3::UnitZ(), 1.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, (-120.0 * M_PI / 180.0) * Vec3::UnitZ(), 1.0);
    const Vec4 outward = quaternion_from_angular_velocity(identity, (120.0 * M_PI / 180.0) * Vec3::UnitZ(), 1.0);
    const Vec4 expected = interpolate_orientation_full_arc(current, target, 0.85);
    const Vec4 boundary = quaternion_from_angular_velocity(identity, 0.5 * M_PI * Vec3::UnitZ(), 1.0);
    const Vec4 tiny_outward = quaternion_from_angular_velocity(identity, (0.5 * M_PI + 1.0e-13) * Vec3::UnitZ(), 1.0);

    EXPECT_TRUE(bound_quaternion(identity, current, target, 0.5 * M_PI).isApprox(expected, 1.0e-14));
    EXPECT_TRUE(bound_quaternion(identity, boundary, outward, 0.5 * M_PI).isApprox(boundary, 1.0e-14));
    EXPECT_TRUE(bound_quaternion(identity, boundary, tiny_outward, 0.5 * M_PI).isApprox(boundary, 1.0e-14));
    EXPECT_TRUE(bound_quaternion(identity, identity, identity, 0.0).isApprox(identity, 1.0e-14));
    EXPECT_TRUE(bound_quaternion(identity, identity, target, M_PI).isApprox(target, 1.0e-14));
    EXPECT_THROW(bound_quaternion(identity, identity, -identity, 0.5 * M_PI), std::invalid_argument);
    EXPECT_THROW(bound_quaternion(identity, identity, -identity, M_PI), std::invalid_argument);
}

TEST(RigidBodyIPCSolver, ParallelBodiesStayInsideCachedBlueBoxes) {
    RefMesh ref_mesh;
    DeformedState state;
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const std::vector<Vec3> first_nodes = {Vec3(-0.5, 0.0, 0.0), Vec3(0.5, 0.0, 0.0), Vec3(0.0, 0.5, 0.0)};
    const std::vector<Vec3> second_nodes = {Vec3(9.5, 0.0, 0.0), Vec3(10.5, 0.0, 0.0), Vec3(10.0, 0.5, 0.0)};
    const int first_rb = create_rigid_body(first_nodes, Vec3::UnitX(), identity, 2.0 * Vec3::UnitZ(), 1.0, ref_mesh, state);
    const int second_rb = create_rigid_body(second_nodes, Vec3::UnitX(), identity, 2.0 * Vec3::UnitZ(), 1.0, ref_mesh, state);
    ref_mesh.tris = {ref_mesh.rb_nodes[first_rb][0], ref_mesh.rb_nodes[first_rb][1], ref_mesh.rb_nodes[first_rb][2], ref_mesh.rb_nodes[second_rb][0], ref_mesh.rb_nodes[second_rb][1], ref_mesh.rb_nodes[second_rb][2]};

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.max_global_iters = 3;
    params.fixed_iters = true;
    params.damping = 1.0;
    params.d_hat = 0.01;
    params.node_box_min = 0.025;
    params.node_box_max = 0.025;
    params.theta_box_min = 0.05;
    params.theta_box_max = 0.05;
    params.node_box_update_count = 10;
    params.use_parallel = true;

    std::vector<Vec3> parallel_coms = state.x_coms;
    std::vector<Vec4> parallel_orientations = state.orientations;
    std::vector<Vec3> parallel_omega(state.omega.size(), Vec3::Zero());
    const SolverResult parallel_result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, parallel_coms, parallel_orientations, parallel_omega);
    EXPECT_TRUE(parallel_result.converged);

    for (int rb = 0; rb < 2; ++rb) {
        EXPECT_LE((parallel_coms[rb] - state.x_coms[rb]).cwiseAbs().maxCoeff(), params.node_box_max + 1.0e-14);
        Vec4 relative = quaternion_normalize(quaternion_multiply(parallel_orientations[rb], quaternion_conjugate(state.orientations[rb])));
        if (relative[0] < 0.0)
            relative = -relative;
        const double theta = 2.0 * std::atan2(relative.tail<3>().norm(), relative[0]);
        EXPECT_LE(theta, params.theta_box_max + 1.0e-13);
    }

    params.use_parallel = false;
    std::vector<Vec3> serial_coms = state.x_coms;
    std::vector<Vec4> serial_orientations = state.orientations;
    std::vector<Vec3> serial_omega(state.omega.size(), Vec3::Zero());
    const SolverResult serial_result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, serial_coms, serial_orientations, serial_omega);
    EXPECT_TRUE(serial_result.converged);
    for (int rb = 0; rb < 2; ++rb) {
        EXPECT_TRUE(parallel_coms[rb].isApprox(serial_coms[rb], 1.0e-13));
        EXPECT_TRUE(parallel_orientations[rb].isApprox(serial_orientations[rb], 1.0e-13));
        EXPECT_TRUE(parallel_omega[rb].isApprox(serial_omega[rb], 1.0e-13));
    }

    params.use_parallel = true;
    params.node_box_update_count = 1;
    std::vector<Vec3> rebuilt_coms = state.x_coms;
    std::vector<Vec4> rebuilt_orientations = state.orientations;
    std::vector<Vec3> rebuilt_omega(state.omega.size(), Vec3::Zero());
    global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, rebuilt_coms, rebuilt_orientations, rebuilt_omega);
    for (int rb = 0; rb < 2; ++rb) {
        EXPECT_GT((rebuilt_coms[rb] - state.x_coms[rb]).cwiseAbs().maxCoeff(), params.node_box_max);
        EXPECT_LE((rebuilt_coms[rb] - state.x_coms[rb]).cwiseAbs().maxCoeff(), 3.0 * params.node_box_max + 1.0e-14);
        Vec4 relative = quaternion_normalize(quaternion_multiply(rebuilt_orientations[rb], quaternion_conjugate(state.orientations[rb])));
        if (relative[0] < 0.0)
            relative = -relative;
        const double theta = 2.0 * std::atan2(relative.tail<3>().norm(), relative[0]);
        EXPECT_GT(theta, params.theta_box_max);
        EXPECT_LE(theta, 3.0 * params.theta_box_max + 1.0e-13);
    }
}

TEST(RigidBodyTranslationSafeStep, MovingNodeUsesLinearCCD) {
    const std::vector<Vec3> x = {
        Vec3(0.25, 0.25, 1.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};

    const double alpha = translation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3(0.0, 0.0, -2.0), 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyTranslationSafeStep, RepeatedCandidatePreservesTOIExactly) {
    const std::vector<Vec3> x = {Vec3(0.25, 0.25, 1.0), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};
    ref_mesh.rb_nodes = {{0}};
    const RigidSafeStepCandidates candidates = build_rigid_safe_step_candidates(ref_mesh, {}, static_cast<int>(x.size()), 1);
    const std::vector<int>& single = candidates.body_nt_pair_indices[0];
    ASSERT_EQ(single.size(), 1U);
    std::vector<int> repeated = single;
    repeated.insert(repeated.end(), single.begin(), single.end());

    const double single_toi = per_rigid_body_translation_safe_step(ref_mesh, candidates.cache, single, {}, x, 0, Vec3(0.0, 0.0, -2.0), 1.0);
    const double repeated_toi = per_rigid_body_translation_safe_step(ref_mesh, candidates.cache, repeated, {}, x, 0, Vec3(0.0, 0.0, -2.0), 1.0);

    EXPECT_DOUBLE_EQ(repeated_toi, single_toi);
}

TEST(RigidBodyTranslationSafeStep, MovingTriangleUsesRelativeNodeMotion) {
    const std::vector<Vec3> x = {
        Vec3(0.25, 0.25, 1.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {-1, 0, 0, 0};

    const double alpha = translation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3(0.0, 0.0, 2.0), 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyTranslationSafeStep, MovingFirstEdgeUsesTranslatingEdgeCCD) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 1.0),
        Vec3(1.0, 0.0, 1.0),
        Vec3(0.5, -1.0, 0.0),
        Vec3(0.5, 1.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {0, 0, -1, -1};
    const std::vector<std::array<int, 2>> edges = {{0, 1}, {2, 3}};

    const double alpha = translation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3(0.0, 0.0, -2.0), 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyTranslationSafeStep, MovingSecondEdgeIsReorderedForCCD) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 1.0),
        Vec3(1.0, 0.0, 1.0),
        Vec3(0.5, -1.0, 0.0),
        Vec3(0.5, 1.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {0, 0, -1, -1};
    const std::vector<std::array<int, 2>> edges = {{2, 3}, {0, 1}};

    const double alpha = translation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3(0.0, 0.0, -2.0), 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyTranslationSafeStep, SkipsInternalAndUnrelatedPairs) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 1.0),
        Vec3(1.0, 0.0, 1.0),
        Vec3(0.5, -1.0, 0.0),
        Vec3(0.5, 1.0, 0.0)
    };
    const std::vector<std::array<int, 2>> edges = {{0, 1}, {2, 3}};
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {0, 0, 0, 0};
    EXPECT_DOUBLE_EQ(translation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3(0.0, 0.0, -2.0)), 1.0);

    ref_mesh.node_to_rb = {-1, -1, -1, -1};
    EXPECT_DOUBLE_EQ(translation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3(0.0, 0.0, -2.0)), 1.0);
}

TEST(RigidBodyRotationSafeStep, MovingNodeUsesRotationCCD) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -2.0, 0.0),
        Vec3(1.0, 0.0, -1.0),
        Vec3(3.0, 0.0, -1.0),
        Vec3(2.0, 0.0, 1.0)
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, M_PI), 1.0);

    const double alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, target, 0.9);
    const Vec4 cap_before_collision = bound_quaternion(identity, identity, target, 0.25 * M_PI);
    const double cap_before_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, cap_before_collision, 0.9);
    const Vec4 cap_after_collision = bound_quaternion(identity, identity, target, 0.75 * M_PI);
    const double cap_after_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, cap_after_collision, 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
    EXPECT_DOUBLE_EQ(cap_before_alpha, 1.0);
    EXPECT_NEAR(cap_after_alpha, 0.6, 1.0e-12);
    EXPECT_TRUE(interpolate_orientation_full_arc(identity, cap_after_collision, cap_after_alpha).isApprox(interpolate_orientation_full_arc(identity, target, alpha), 1.0e-12));
}

TEST(RigidBodyRotationSafeStep, RepeatedCandidatePreservesTOIExactly) {
    const std::vector<Vec3> x = {Vec3(0.0, -2.0, 0.0), Vec3(1.0, 0.0, -1.0), Vec3(3.0, 0.0, -1.0), Vec3(2.0, 0.0, 1.0)};
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};
    ref_mesh.rb_nodes = {{0}};
    const RigidSafeStepCandidates candidates = build_rigid_safe_step_candidates(ref_mesh, {}, static_cast<int>(x.size()), 1);
    const std::vector<int>& single = candidates.body_nt_pair_indices[0];
    ASSERT_EQ(single.size(), 1U);
    std::vector<int> repeated = single;
    repeated.insert(repeated.end(), single.begin(), single.end());
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, M_PI), 1.0);

    const double single_toi = per_rigid_body_rotation_safe_step(ref_mesh, candidates.cache, single, {}, x, 0, Vec3::Zero(), identity, target, 1.0);
    const double repeated_toi = per_rigid_body_rotation_safe_step(ref_mesh, candidates.cache, repeated, {}, x, 0, Vec3::Zero(), identity, target, 1.0);

    EXPECT_DOUBLE_EQ(repeated_toi, single_toi);
}

TEST(RigidBodyRotationSafeStep, PreservesFull270DegreeTargetArc) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -2.0, 0.0),
        Vec3(-1.0, 2.0, -1.0),
        Vec3(1.0, 2.0, -1.0),
        Vec3(0.0, 2.0, 1.0),
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 full_target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, 1.5 * M_PI), 1.0);

    const double full_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, full_target, 0.9);
    const double complementary_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, -full_target, 0.9);

    // Contact occurs after 180 / 270 = 2/3 of the full arc.
    EXPECT_NEAR(full_alpha, 0.9 * (2.0 / 3.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(complementary_alpha, 1.0);
}

TEST(RigidBodyRotationSafeStep, NoncollinearOmegaEndpointsUseQuaternionPath) {
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 current = quaternion_from_angular_velocity(identity, Vec3(0.5 * M_PI, 0.0, 0.0), 1.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.5 * M_PI, 0.0), 1.0);
    const Vec4 halfway = interpolate_orientation_full_arc(current, target, 0.5);
    const Vec3 moving_start = quaternion_rotate(current, Vec3::UnitZ());
    const Vec3 contact = quaternion_rotate(halfway, Vec3::UnitZ());

    const Vec4 relative = quaternion_normalize(quaternion_multiply(target, quaternion_conjugate(current)));
    const Vec3 axis = relative.tail<3>().normalized();
    const Vec3 plane_normal = axis.cross(contact).normalized();
    const Vec3 plane_axis = plane_normal.cross(axis).normalized();
    const std::vector<Vec3> x = {
        moving_start,
        contact + 0.4 * axis,
        contact - 0.2 * axis + 0.35 * plane_axis,
        contact - 0.2 * axis - 0.35 * plane_axis
    };

    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {0, -1, -1, -1};
    const double alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), current, target, 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-10);
}

TEST(RigidBodyRotationSafeStep, MovingTriangleUsesReverseRotation) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -2.0, 0.0),
        Vec3(1.0, 0.0, -1.0),
        Vec3(3.0, 0.0, -1.0),
        Vec3(2.0, 0.0, 1.0)
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {-1, 0, 0, 0};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, -M_PI), 1.0);

    const double alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, target, 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyRotationSafeStep, MovingTriangleReverseCasePreservesFullArc) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -2.0, 0.0),
        Vec3(-1.0, 2.0, -1.0),
        Vec3(1.0, 2.0, -1.0),
        Vec3(0.0, 2.0, 1.0),
    };
    RefMesh ref_mesh;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.node_to_rb = {-1, 0, 0, 0};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 full_target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, 1.5 * M_PI), 1.0);

    const double full_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, full_target, 0.9);
    const double complementary_alpha = rotation_safe_step_for_test(ref_mesh, {}, x, 0, Vec3::Zero(), identity, -full_target, 0.9);

    EXPECT_NEAR(full_alpha, 0.9 * (2.0 / 3.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(complementary_alpha, 1.0);
}

TEST(RigidBodyRotationSafeStep, MovingFirstEdgeUsesRotationCCD) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -1.0, 0.0),
        Vec3(0.0, -2.0, 0.0),
        Vec3(1.5, 0.0, 0.0),
        Vec3(3.0, 0.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {0, 0, -1, -1};
    const std::vector<std::array<int, 2>> edges = {{0, 1}, {2, 3}};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, M_PI), 1.0);

    const double alpha = rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, target, 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyRotationSafeStep, MovingSecondEdgeUsesReverseRotation) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -1.0, 0.0),
        Vec3(0.0, -2.0, 0.0),
        Vec3(1.5, 0.0, 0.0),
        Vec3(3.0, 0.0, 0.0)
    };
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {0, 0, -1, -1};
    const std::vector<std::array<int, 2>> edges = {{2, 3}, {0, 1}};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, M_PI), 1.0);

    const double alpha = rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, target, 0.9);

    EXPECT_NEAR(alpha, 0.45, 1.0e-12);
}

TEST(RigidBodyRotationSafeStep, MovingSecondEdgeReverseCasePreservesFullArc) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -1.0, 0.0),
        Vec3(0.0, -2.0, 0.0),
        Vec3(0.0, 1.5, -1.0),
        Vec3(0.0, 1.5, 1.0),
    };
    RefMesh ref_mesh;
    ref_mesh.node_to_rb = {-1, -1, 0, 0};
    const std::vector<std::array<int, 2>> edges = {{0, 1}, {2, 3}};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 full_target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, 1.5 * M_PI), 1.0);

    const double full_alpha = rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, full_target, 0.9);
    const double complementary_alpha = rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, -full_target, 0.9);

    EXPECT_NEAR(full_alpha, 0.9 * (2.0 / 3.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(complementary_alpha, 1.0);
}

TEST(RigidBodyRotationSafeStep, SkipsInternalAndUnrelatedPairs) {
    const std::vector<Vec3> x = {
        Vec3(0.0, -1.0, 0.0),
        Vec3(0.0, -2.0, 0.0),
        Vec3(1.5, 0.0, 0.0),
        Vec3(3.0, 0.0, 0.0)
    };
    const std::vector<std::array<int, 2>> edges = {{0, 1}, {2, 3}};
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 target = quaternion_from_angular_velocity(identity, Vec3(0.0, 0.0, M_PI), 1.0);
    RefMesh ref_mesh;

    ref_mesh.node_to_rb = {0, 0, 0, 0};
    EXPECT_DOUBLE_EQ(rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, target), 1.0);

    ref_mesh.node_to_rb = {-1, -1, -1, -1};
    EXPECT_DOUBLE_EQ(rotation_safe_step_for_test(ref_mesh, edges, x, 0, Vec3::Zero(), identity, target), 1.0);
}

}  // namespace

#include "general_simd_rigid.h"

#include "friction_energy.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>

namespace {

constexpr double dt = 0.031, d_hat = 0.55, stiffness = 370.0;

ipc_simd::RigidContactOutput reference(const ipc_simd::RigidContactInput& input,
    RigidDerivativeMode mode, double friction) {
    ipc_simd::RigidContactOutput result;
    const auto& x = input.positions;
    FrozenFrictionContact contact;
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 omega = Vec3::Zero();
    if (input.segment_segment) {
        if (friction != 0.0) {
            const auto evaluation = make_segment_segment_contact_evaluation(x, d_hat, stiffness);
            result.barrier = segment_segment_barrier_rb(x[0], x[1], x[2], x[3],
                input.body_references, input.side, q, omega, dt, d_hat, mode,
                1e-12, input.kinematics, &evaluation.dr, &evaluation.b_prime, &evaluation.b_double_prime);
            contact = make_segment_segment_frozen_friction_contact(x, input.previous_positions, evaluation, dt, 0.1);
        } else result.barrier = segment_segment_barrier_rb(x[0], x[1], x[2], x[3],
            input.body_references, input.side, q, omega, dt, d_hat, mode, 1e-12, input.kinematics);
    } else {
        if (friction != 0.0) {
            const auto evaluation = make_node_triangle_contact_evaluation(x, d_hat, stiffness);
            result.barrier = node_triangle_barrier_rb(x[0], x[1], x[2], x[3],
                input.body_references, input.side, q, omega, dt, d_hat, mode,
                1e-12, input.kinematics, &evaluation.dr, &evaluation.b_prime, &evaluation.b_double_prime);
            contact = make_node_triangle_frozen_friction_contact(x, input.previous_positions, evaluation, dt, 0.1);
        } else result.barrier = node_triangle_barrier_rb(x[0], x[1], x[2], x[3],
            input.body_references, input.side, q, omega, dt, d_hat, mode, 1e-12, input.kinematics);
    }
    if (!contact.active) return result;
    const int first = input.side == RigidBarrierSide::FirstPrimitive ? 0 : (input.segment_segment ? 2 : 1);
    const int last = input.side == RigidBarrierSide::FirstPrimitive ? (input.segment_segment ? 1 : 0) : 3;
    const unsigned mask = input.body_role_mask ? input.body_role_mask : ((1u << (last + 1)) - (1u << first));
    double weight = 0.0;
    Mat33 J = Mat33::Zero();
    for (int role = 0; role < 4; ++role) {
        if (!(mask & (1u << role))) continue;
        weight += contact.weights[role];
        if (mode != RigidDerivativeMode::TranslationHessian && updates_rigid_orientation(input.update_mode))
            J += contact.weights[role] * dx_domega(input.body_references[role], *input.kinematics);
    }
    const Mat33 T = weight * Mat33::Identity();
    const auto [g, H] = frozen_friction_relative_gradient_and_hessian(contact, friction, dt * dt);
    const bool translation = updates_rigid_translation(input.update_mode);
    const bool orientation = updates_rigid_orientation(input.update_mode);
    if (translation && mode != RigidDerivativeMode::OrientationHessian)
        result.friction.translation_gradient = T.transpose() * g;
    if (orientation && mode != RigidDerivativeMode::TranslationHessian)
        result.friction.orientation_gradient = J.transpose() * g;
    if (translation && (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::TranslationHessian))
        result.friction.translation_translation_hessian = T.transpose() * H * T;
    if (orientation && (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian))
        result.friction.orientation_orientation_hessian = J.transpose() * H * J;
    if (translation && orientation && mode == RigidDerivativeMode::Full)
        result.friction.translation_orientation_hessian = T.transpose() * H * J;
    return result;
}

void compare(const RigidEnergyDerivatives& actual, const RigidEnergyDerivatives& expected) {
    const auto check = [](const auto& a, const auto& b) {
        EXPECT_TRUE(a.allFinite());
        EXPECT_LE((a - b).norm(), 3e-12 * (1.0 + b.norm()));
    };
    check(actual.translation_gradient, expected.translation_gradient);
    check(actual.orientation_gradient, expected.orientation_gradient);
    check(actual.translation_translation_hessian, expected.translation_translation_hessian);
    check(actual.translation_orientation_hessian, expected.translation_orientation_hessian);
    check(actual.orientation_orientation_hessian, expected.orientation_orientation_hessian);
}

void add(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& value) {
    total.translation_gradient += value.translation_gradient;
    total.orientation_gradient += value.orientation_gradient;
    total.translation_translation_hessian += value.translation_translation_hessian;
    total.translation_orientation_hessian += value.translation_orientation_hessian;
    total.orientation_orientation_hessian += value.orientation_orientation_hessian;
}

ipc_simd::RigidContactInput fixture(int index, const QuaternionOmegaKinematics& kinematics) {
    ipc_simd::RigidContactInput input;
    input.segment_segment = (index / 2) % 2 != 0;
    input.side = index % 2 == 0 ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
    input.kinematics = &kinematics;
    const int feature = (index / 4) % 6;
    if (!input.segment_segment) {
        input.positions = {Vec3(0.2, 0.25, 0.17), Vec3::Zero(), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
        if (feature == 1) input.positions[0] = Vec3(0.4, -0.13, 0.16);
        if (feature == 2) input.positions[0] = Vec3(-0.12, -0.11, 0.16);
        if (feature == 3) input.positions[0][2] *= -1.0;
        if (feature == 4) input.positions[3] = Vec3(0.6, 0.0, 0.0);
        if (feature == 5) input.positions[0][2] = 0.9;
    } else {
        input.positions = {Vec3(-0.7, 0.0, 0.0), Vec3(0.7, 0.0, 0.0), Vec3(0.0, -0.6, 0.17), Vec3(0.0, 0.6, 0.17)};
        if (feature == 1) { input.positions[2][1] = 0.12; input.positions[3][1] = 0.9; }
        if (feature == 2) { input.positions[2] = Vec3(0.9, 0.12, 0.17); input.positions[3] = Vec3(0.9, 0.9, 0.17); }
        if (feature == 3) { input.positions[2] = Vec3(-0.4, 0.12, 0.17); input.positions[3] = Vec3(0.5, 0.12, 0.17); }
        if (feature == 4) input.positions[1] = input.positions[0];
        if (feature == 5) { input.positions[2][2] = 0.9; input.positions[3][2] = 0.9; }
    }
    for (int i = 0; i < 4; ++i) {
        input.body_references[i] = input.positions[i] + Vec3(0.4, -0.7, 0.9);
        input.previous_positions[i] = input.positions[i] - (i + 1) * Vec3(0.001, -0.003, 0.002);
    }
    return input;
}

} // namespace

TEST(GeneralSIMDRigid, CoupledBarrierFrictionFeaturesSidesModesAndTailsMatchScalar) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(0.8, 0.2, -0.4, 0.4).normalized(), Vec3(1.2, -0.9, 1.7), dt, true);
    std::array<ipc_simd::RigidContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::RigidContactOutput, ipc_simd::contact_tile_width> outputs;
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = fixture(static_cast<int>(i), kinematics);
    for (auto mode : {RigidDerivativeMode::Full, RigidDerivativeMode::Gradient,
             RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian})
        for (double friction : {0.0, 0.37})
            for (std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(3), std::size_t(17), inputs.size()}) {
                ipc_simd::rigid_contact_derivatives_tile(inputs.data(), count, d_hat,
                    stiffness, friction, dt, 0.1, mode, outputs.data());
                for (std::size_t i = 0; i < count; ++i) {
                    SCOPED_TRACE(::testing::Message() << "entry=" << i << " count=" << count << " mode=" << int(mode) << " friction=" << friction);
                    const auto expected = reference(inputs[i], mode, friction);
                    compare(outputs[i].barrier, expected.barrier);
                    compare(outputs[i].friction, expected.friction);
                }
            }
}

TEST(GeneralSIMDRigid, FrictionUsesSignedMultiRoleJacobianAndUpdateMode) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3(0.5, 1.1, -0.9), dt, true);
    std::array<ipc_simd::RigidContactInput, 4> inputs;
    std::array<ipc_simd::RigidContactOutput, 4> outputs;
    for (int i = 0; i < 4; ++i) {
        inputs[i] = fixture(1, kinematics);
        inputs[i].update_mode = static_cast<RigidBodyUpdateMode>(i);
    }
    // Include roles from both sides: their signed weights cancel in COM.
    inputs[0].body_role_mask = 15;
    ipc_simd::rigid_contact_derivatives_tile(inputs.data(), inputs.size(), d_hat,
        stiffness, 0.45, dt, 0.1, RigidDerivativeMode::Full, outputs.data());
    for (int i = 0; i < 4; ++i) {
        const auto expected = reference(inputs[i], RigidDerivativeMode::Full, 0.45);
        compare(outputs[i].barrier, expected.barrier);
        compare(outputs[i].friction, expected.friction);
    }
    EXPECT_LT(outputs[0].friction.translation_gradient.norm(), 1e-14);
    EXPECT_LT(outputs[0].friction.translation_translation_hessian.norm(), 1e-14);
    EXPECT_GT(outputs[1].friction.translation_translation_hessian.norm(), 1e-5);
    EXPECT_GT(outputs[2].friction.orientation_orientation_hessian.norm(), 1e-8);
}

TEST(GeneralSIMDRigid, TriangleOrientationRequiresCrossBlocksAndQuaternionCurvature) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3(3.0, 2.0, -1.0), dt, true);
    const auto input = fixture(1, kinematics);
    ipc_simd::RigidContactOutput output;
    ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat, stiffness, 0.0,
        dt, 0.1, RigidDerivativeMode::OrientationHessian, &output);
    const auto expected = reference(input, RigidDerivativeMode::OrientationHessian, 0.0);
    compare(output.barrier, expected.barrier);
    const auto& x = input.positions;
    Mat33 diagonal_only = Mat33::Zero(), curvature = Mat33::Zero();
    for (int i = 1; i < 4; ++i) {
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], d_hat, i);
        const Mat33 J = dx_domega(input.body_references[i], kinematics);
        diagonal_only += J.transpose() * H * J;
        const auto second = d2x_domega2(input.body_references[i], kinematics);
        for (int c = 0; c < 3; ++c) curvature += g[c] * second[c];
    }
    EXPECT_GT(curvature.norm(), 1e-5);
    EXPECT_GT((output.barrier.orientation_orientation_hessian - diagonal_only - curvature).norm(), 1e-5);
}

TEST(GeneralSIMDRigid, OrderedAccumulatorMatchesTilesAndCooperativeHelpers) {
    RefMesh mesh;
    DeformedState state;
    state.orientations = {Vec4(1, 0, 0, 0)};
    const std::vector<Vec3> omega = {Vec3(0.4, -0.8, 0.6)};
    const auto kinematics = quaternion_omega_kinematics(state.orientations[0], omega[0], dt, true);
    const auto nt = fixture(1, kinematics), ss = fixture(2, kinematics);
    std::vector<Vec3> positions(nt.positions.begin(), nt.positions.end());
    positions.insert(positions.end(), ss.positions.begin(), ss.positions.end());
    state.deformed_positions.assign(nt.previous_positions.begin(), nt.previous_positions.end());
    state.deformed_positions.insert(state.deformed_positions.end(), ss.previous_positions.begin(), ss.previous_positions.end());
    mesh.node_to_rb = {-1, 0, 0, 0, 0, 0, -1, -1};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    mesh.ref_positions = {{nt.body_references[1], nt.body_references[2], nt.body_references[3], ss.body_references[0], ss.body_references[1]}};
    const std::vector<int> local = {-1, 0, 1, 2, 3, 4, -1, -1};
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(40);
    for (auto& pair : cache.nt_pairs) { pair.node = 0; pair.tri_v[0] = 1; pair.tri_v[1] = 2; pair.tri_v[2] = 3; }
    cache.ss_pairs.resize(37);
    for (auto& pair : cache.ss_pairs) for (int role = 0; role < 4; ++role) pair.v[role] = role + 4;
    std::vector<int> nt_indices(40), ss_indices(37);
    std::iota(nt_indices.begin(), nt_indices.end(), 0);
    std::iota(ss_indices.begin(), ss_indices.end(), 0);
    SimParams params;
    params.d_hat = d_hat;
    params.k_barrier = stiffness;
    params.friction_coefficient = 0.37;
    params.friction_velocity_epsilon = 0.1;
    for (auto mode : {RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian}) {
        ipc_simd::RigidContactOutput expected;
        const auto nt_value = reference(nt, mode, params.friction_coefficient);
        const auto ss_value = reference(ss, mode, params.friction_coefficient);
        for (int i = 0; i < 40; ++i) { add(expected.barrier, nt_value.barrier); add(expected.friction, nt_value.friction); }
        for (int i = 0; i < 37; ++i) { add(expected.barrier, ss_value.barrier); add(expected.friction, ss_value.friction); }
        const auto serial = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
            nt_indices, ss_indices, local, positions, omega, params, dt, mode, &kinematics);
        compare(serial.barrier, expected.barrier);
        compare(serial.friction, expected.friction);
        ipc_simd::RigidContactOutput parallel;
        int leader_calls = 0;
        const std::function<void()> leader = [&] { ++leader_calls; };
        #pragma omp parallel num_threads(4)
        {
            #pragma omp single
            parallel = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
                nt_indices, ss_indices, local, positions, omega, params, dt, mode,
                &kinematics, true, &leader);
        }
        EXPECT_EQ(leader_calls, 1);
        compare(parallel.barrier, serial.barrier);
        compare(parallel.friction, serial.friction);
    }
    // Friction-free calls do not read previous positions, even if absent.
    params.friction_coefficient = 0.0;
    state.deformed_positions.clear();
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
        nt_indices, ss_indices, local, positions, omega, params, dt,
        RigidDerivativeMode::TranslationHessian));
}

TEST(GeneralSIMDRigid, RejectedContactsSkipExactOverlapAndPreviousPositionAccess) {
    RefMesh mesh;
    mesh.node_to_rb = {0, 0, 0, 0};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    DeformedState state;
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(1);
    auto& pair = cache.nt_pairs[0];
    pair.node = 0; pair.tri_v[0] = 1; pair.tri_v[1] = 2; pair.tri_v[2] = 3;
    SimParams params;
    params.d_hat = d_hat; params.k_barrier = stiffness; params.friction_coefficient = 0.4;
    const std::vector<Vec3> positions(4, Vec3::Zero());
    // Same-body self pairs are rejected before exact distance/previous reads.
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
        {0}, {}, {}, positions, {}, params, dt, RigidDerivativeMode::TranslationHessian));
    // An admitted exact intersection keeps the scalar exception contract.
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3::Zero(), dt, true);
    auto input = fixture(0, kinematics);
    input.positions[0][2] = 0.0;
    ipc_simd::RigidContactOutput output;
    EXPECT_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.0, dt, 0.1, RigidDerivativeMode::Full, &output), std::runtime_error);
    // An admitted inactive contact still validates the frozen builder's
    // smoothing parameters, while unneeded previous coordinates may be NaN.
    input.positions[0][2] = 0.9;
    for (auto& p : input.previous_positions) p.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.4, dt, 0.1, RigidDerivativeMode::Full, &output));
    EXPECT_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.4, dt, 0.0, RigidDerivativeMode::Full, &output), std::invalid_argument);
}

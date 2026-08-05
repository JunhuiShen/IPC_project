#include "rigid_body_ipc.h"

#include "parallel_helper.h"
#include "physics.h"
#include "safe_step.h"
#include "solver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
    const Mat33 J_xomega = dx_domega(X_centered, q0, omega, dt);
    const std::array<Mat33, 3> H_xomega = d2x_domega2(X_centered, q0, omega, dt);
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
    const Mat33 exact = dx_domega(X_centered, q0, omega, dt);
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
    const std::array<Mat33, 3> exact = d2x_domega2(X_centered, q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat33 finite_difference =
                (dx_domega(X_centered, q0, omega + step, dt)
                 - dx_domega(X_centered, q0, omega - step, dt)) / (2.0 * h);
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
        std::vector<Vec3> residual_omega_new = state.omega;
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
    const Vec4 expected_q_dot = (expected_q - q_n) / dt;
    const Vec3 expected_finite_difference_omega =
        (2.0 * quaternion_multiply(
            expected_q_dot, quaternion_inverse(expected_q))).tail<3>();

    std::vector<Vec3> x_com_new = state.x_coms;
    std::vector<Vec4> q_new = state.orientations;
    std::vector<Vec3> omega_new = state.omega;
    const SolverResult result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, x_com_new, q_new, omega_new);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_GT(x_com_new[rb].z(), initial_com.z());
    EXPECT_GT(omega_new[rb].norm(), 1.0e-8);
    EXPECT_TRUE(x_com_new[rb].isApprox(expected_com, 1.0e-11));
    EXPECT_TRUE(q_new[rb].isApprox(expected_q, 1.0e-11));
    EXPECT_TRUE(omega_new[rb].isApprox(
        expected_finite_difference_omega, 1.0e-11));
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
    std::vector<Vec3> parallel_omega = state.omega;
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
    std::vector<Vec3> serial_omega = state.omega;
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
    std::vector<Vec3> rebuilt_omega = state.omega;
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

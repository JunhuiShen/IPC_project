#include "friction_energy.h"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 1.0e-11;

void expect_vec_near(const Vec3& actual, const Vec3& expected, double tolerance = kTolerance) {
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance);
}

void expect_mat_near(const Mat33& actual, const Mat33& expected, double tolerance = kTolerance) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(actual(i, j), expected(i, j), tolerance);
}

void expect_weights_near(
        const std::array<double, 4>& actual,
        const std::array<double, 4>& expected,
        double tolerance = kTolerance) {
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance);
}

Vec3 weighted_position(
        const std::array<Vec3, 4>& positions,
        const std::array<double, 4>& weights) {
    Vec3 result = Vec3::Zero();
    for (int i = 0; i < 4; ++i)
        result += weights[i] * positions[i];
    return result;
}

std::array<Vec3, 4> previous_with_node_slip(
        const std::array<Vec3, 4>& current) {
    std::array<Vec3, 4> previous = current;
    previous[0] -= Vec3(0.031, -0.023, 0.017);
    return previous;
}

std::array<Vec3, 4> previous_with_general_slip(
        const std::array<Vec3, 4>& current) {
    const std::array<Vec3, 4> displacements{{
        Vec3(0.031, -0.023, 0.017),
        Vec3(-0.011, 0.019, 0.007),
        Vec3(0.013, 0.005, -0.029),
        Vec3(-0.021, -0.017, 0.011)}};
    std::array<Vec3, 4> previous = current;
    for (int role = 0; role < 4; ++role)
        previous[role] -= displacements[role];
    return previous;
}

void expect_contact_exact(
        const FrozenFrictionContact& actual,
        const FrozenFrictionContact& expected) {
    EXPECT_EQ(actual.weights, expected.weights);
    EXPECT_EQ(
        std::memcmp(actual.normal.data(), expected.normal.data(),
                    sizeof(double) * 3),
        0);
    EXPECT_EQ(
        std::memcmp(actual.projector.data(), expected.projector.data(),
                    sizeof(double) * 9),
        0);
    EXPECT_EQ(
        std::memcmp(actual.tangential_displacement.data(),
                    expected.tangential_displacement.data(),
                    sizeof(double) * 3),
        0);
    EXPECT_EQ(actual.normal_force, expected.normal_force);
    EXPECT_EQ(actual.eps_u, expected.eps_u);
    EXPECT_EQ(actual.active, expected.active);
}

} // namespace

TEST(SmoothFriction, PotentialAndMollifierAreContinuousAtThreshold) {
    const double eps_u = 0.2;
    EXPECT_DOUBLE_EQ(smooth_friction_potential(0.0, eps_u), 0.0);
    EXPECT_DOUBLE_EQ(
            smooth_friction_mollifier_over_slip(0.0, eps_u),
            2.0 / eps_u);

    const double expected_at_threshold = 2.0 * eps_u / 3.0;
    EXPECT_NEAR(
            smooth_friction_potential(eps_u, eps_u),
            expected_at_threshold, 1.0e-15);
    EXPECT_NEAR(
            smooth_friction_potential(
                    std::nextafter(eps_u, 0.0), eps_u),
            expected_at_threshold, 1.0e-14);
    EXPECT_NEAR(
            smooth_friction_mollifier_over_slip(eps_u, eps_u),
            1.0 / eps_u, 1.0e-14);
    EXPECT_NEAR(
            smooth_friction_mollifier_over_slip(
                    std::nextafter(eps_u, 0.0), eps_u),
            1.0 / eps_u, 1.0e-13);

    for (double slip : {0.04, 0.11, 0.35}) {
        const double h = 1.0e-7;
        const double finite_difference =
                (smooth_friction_potential(slip + h, eps_u)
                 - smooth_friction_potential(slip - h, eps_u))
                / (2.0 * h);
        EXPECT_NEAR(
                finite_difference,
                slip * smooth_friction_mollifier_over_slip(slip, eps_u),
                2.0e-9);
    }
}

TEST(SmoothFriction, SaturatedForceHasCoulombMagnitude) {
    FrozenFrictionContact contact;
    contact.normal = Vec3::UnitZ();
    contact.projector = Mat33::Identity() - contact.normal * contact.normal.transpose();
    contact.tangential_displacement = Vec3(0.3, 0.4, 0.0);
    contact.normal_force = 7.0;
    contact.eps_u = 0.1;
    contact.active = true;

    const double mu = 0.6;
    const double dt2 = 0.04;
    const Vec3 gradient = frozen_friction_relative_gradient(contact, mu, dt2);
    EXPECT_NEAR(gradient.norm(), dt2 * mu * contact.normal_force, 1.0e-14);
    EXPECT_NEAR(gradient.dot(contact.normal), 0.0, 1.0e-14);
}

TEST(FrozenFrictionContact, NodeTriangleFeaturesMatchBarrier) {
    struct Case {
        std::string name;
        std::array<Vec3, 4> positions;
        NodeTriangleRegion expected_region;
        std::array<double, 4> expected_weights;
    };

    const std::vector<Case> cases = {
        {
            "face",
            {{Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
              Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}},
            NodeTriangleRegion::FaceInterior,
            {{1.0, -0.5, -0.2, -0.3}}
        },
        {
            "edge",
            {{Vec3(0.4, -0.2, 0.1), Vec3(0.0, 0.0, 0.0),
              Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}},
            NodeTriangleRegion::Edge12,
            {{1.0, -0.6, -0.4, 0.0}}
        },
        {
            "vertex",
            {{Vec3(-0.2, -0.3, 0.1), Vec3(0.0, 0.0, 0.0),
              Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}},
            NodeTriangleRegion::Vertex1,
            {{1.0, -1.0, 0.0, 0.0}}
        },
        {
            "degenerate",
            {{Vec3(0.25, 0.2, 0.0), Vec3(0.0, 0.0, 0.0),
              Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)}},
            NodeTriangleRegion::DegenerateTriangle,
            {{1.0, -0.75, -0.25, 0.0}}
        }
    };

    constexpr double d_hat = 0.5;
    constexpr double k_barrier = 23.0;
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const auto& x = test_case.positions;
        const NodeTriangleDistanceResult dr =
                node_triangle_distance(x[0], x[1], x[2], x[3]);
        ASSERT_EQ(dr.region, test_case.expected_region);

        const FrozenFrictionContact contact =
                make_node_triangle_frozen_friction_contact(
                        x, previous_with_node_slip(x), d_hat, k_barrier,
                        0.01, 1.0, 1.0e-12, &dr);
        ASSERT_TRUE(contact.active);
        expect_weights_near(contact.weights, test_case.expected_weights);
        EXPECT_NEAR(
                contact.normal_force,
                -k_barrier * scalar_barrier_gradient(dr.distance, d_hat),
                1.0e-10);

        const Vec3 separation = weighted_position(x, contact.weights);
        expect_vec_near(contact.normal, separation.normalized());
        expect_mat_near(
                contact.projector,
                Mat33::Identity() - contact.normal * contact.normal.transpose());

        const double barrier_derivative =
                scalar_barrier_gradient(dr.distance, d_hat);
        for (int role = 0; role < 4; ++role) {
            const Vec3 barrier_gradient = node_triangle_barrier_gradient(
                    x[0], x[1], x[2], x[3], d_hat, role,
                    1.0e-12, &dr, &barrier_derivative);
            expect_vec_near(
                    barrier_gradient,
                    barrier_derivative * contact.weights[role] * contact.normal,
                    2.0e-10);
        }
    }
}

TEST(FrozenFrictionContact, SegmentSegmentAndParallelFeaturesMatchBarrier) {
    struct Case {
        std::string name;
        std::array<Vec3, 4> positions;
        SegmentSegmentRegion expected_region;
        std::array<double, 4> expected_weights;
    };

    const std::vector<Case> cases = {
        {
            "interior",
            {{Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
              Vec3(0.3, -1.0, 0.2), Vec3(0.3, 1.0, 0.2)}},
            SegmentSegmentRegion::Interior,
            {{0.7, 0.3, -0.5, -0.5}}
        },
        {
            "parallel",
            {{Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
              Vec3(0.2, 0.2, 0.0), Vec3(0.8, 0.2, 0.0)}},
            SegmentSegmentRegion::ParallelSegments,
            {{0.8, 0.2, -1.0, 0.0}}
        }
    };

    constexpr double d_hat = 0.5;
    constexpr double k_barrier = 31.0;
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const auto& x = test_case.positions;
        const SegmentSegmentDistanceResult dr =
                segment_segment_distance(x[0], x[1], x[2], x[3]);
        ASSERT_EQ(dr.region, test_case.expected_region);

        const FrozenFrictionContact contact =
                make_segment_segment_frozen_friction_contact(
                        x, previous_with_node_slip(x), d_hat, k_barrier,
                        0.01, 1.0, 1.0e-12, &dr);
        ASSERT_TRUE(contact.active);
        expect_weights_near(contact.weights, test_case.expected_weights);

        const Vec3 separation = weighted_position(x, contact.weights);
        expect_vec_near(contact.normal, separation.normalized());

        const double barrier_derivative =
                scalar_barrier_gradient(dr.distance, d_hat);
        for (int role = 0; role < 4; ++role) {
            const Vec3 barrier_gradient = segment_segment_barrier_gradient(
                    x[0], x[1], x[2], x[3], d_hat, role,
                    1.0e-12, &dr, &barrier_derivative);
            expect_vec_near(
                    barrier_gradient,
                    barrier_derivative * contact.weights[role] * contact.normal,
                    2.0e-10);
        }
    }
}

TEST(FrictionCacheIntegration,
     CachedMeshContactLocalAssemblyMatchesUncachedFormulas) {
    constexpr double d_hat = 0.5;
    constexpr double k_barrier = 23.0;
    constexpr double dt = 0.01;
    constexpr double eps_v = 0.5;
    constexpr double mu = 0.37;
    constexpr double dt2 = dt * dt;

    const std::array<Vec3, 4> nt_current{{
        Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}};
    const std::array<Vec3, 4> nt_previous =
        previous_with_node_slip(nt_current);
    const NodeTriangleDistanceResult nt_distance = node_triangle_distance(
        nt_current[0], nt_current[1], nt_current[2], nt_current[3]);
    const double nt_scalar_gradient =
        scalar_barrier_gradient(nt_distance.distance, d_hat);
    const double nt_scalar_hessian =
        scalar_barrier_hessian(nt_distance.distance, d_hat);
    const FrozenFrictionContact nt_uncached_contact =
        make_node_triangle_frozen_friction_contact(
            nt_current, nt_previous, d_hat, k_barrier, dt, eps_v);
    const FrozenFrictionContact nt_cached_contact =
        make_node_triangle_frozen_friction_contact(
            nt_current, nt_previous, d_hat, k_barrier, dt, eps_v,
            1.0e-12, &nt_distance);

    for (int role = 0; role < 4; ++role) {
        const auto uncached_barrier =
            node_triangle_barrier_self_gradient_and_hessian(
                nt_current[0], nt_current[1], nt_current[2], nt_current[3],
                d_hat, role);
        const auto cached_barrier =
            node_triangle_barrier_self_gradient_and_hessian(
                nt_current[0], nt_current[1], nt_current[2], nt_current[3],
                d_hat, role, 1.0e-12, &nt_distance,
                &nt_scalar_gradient, &nt_scalar_hessian);
        const auto uncached_friction =
            frozen_friction_role_gradient_and_hessian(
                nt_uncached_contact, role, mu, dt2);
        const auto cached_friction =
            frozen_friction_role_gradient_and_hessian(
                nt_cached_contact, role, mu, dt2);

        expect_vec_near(
            dt2 * k_barrier * cached_barrier.first
                + cached_friction.first,
            dt2 * k_barrier * uncached_barrier.first
                + uncached_friction.first,
            0.0);
        expect_mat_near(
            dt2 * k_barrier * cached_barrier.second
                + cached_friction.second,
            dt2 * k_barrier * uncached_barrier.second
                + uncached_friction.second,
            0.0);
    }

    const std::array<Vec3, 4> ss_current{{
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
        Vec3(0.3, -1.0, 0.2), Vec3(0.3, 1.0, 0.2)}};
    const std::array<Vec3, 4> ss_previous =
        previous_with_node_slip(ss_current);
    const SegmentSegmentDistanceResult ss_distance = segment_segment_distance(
        ss_current[0], ss_current[1], ss_current[2], ss_current[3]);
    const double ss_scalar_gradient =
        scalar_barrier_gradient(ss_distance.distance, d_hat);
    const double ss_scalar_hessian =
        scalar_barrier_hessian(ss_distance.distance, d_hat);
    const FrozenFrictionContact ss_uncached_contact =
        make_segment_segment_frozen_friction_contact(
            ss_current, ss_previous, d_hat, k_barrier, dt, eps_v);
    const FrozenFrictionContact ss_cached_contact =
        make_segment_segment_frozen_friction_contact(
            ss_current, ss_previous, d_hat, k_barrier, dt, eps_v,
            1.0e-12, &ss_distance);

    for (int role = 0; role < 4; ++role) {
        const auto uncached_barrier =
            segment_segment_barrier_self_gradient_and_hessian(
                ss_current[0], ss_current[1], ss_current[2], ss_current[3],
                d_hat, role);
        const auto cached_barrier =
            segment_segment_barrier_self_gradient_and_hessian(
                ss_current[0], ss_current[1], ss_current[2], ss_current[3],
                d_hat, role, 1.0e-12, &ss_distance,
                &ss_scalar_gradient, &ss_scalar_hessian);
        const auto uncached_friction =
            frozen_friction_role_gradient_and_hessian(
                ss_uncached_contact, role, mu, dt2);
        const auto cached_friction =
            frozen_friction_role_gradient_and_hessian(
                ss_cached_contact, role, mu, dt2);

        expect_vec_near(
            dt2 * k_barrier * cached_barrier.first
                + cached_friction.first,
            dt2 * k_barrier * uncached_barrier.first
                + uncached_friction.first,
            0.0);
        expect_mat_near(
            dt2 * k_barrier * cached_barrier.second
                + cached_friction.second,
            dt2 * k_barrier * uncached_barrier.second
                + uncached_friction.second,
            0.0);
    }
}

TEST(FrictionCacheIntegration,
     SharedNodeTriangleEvaluationMatchesLegacyAcrossAllFeatures) {
    struct Case {
        const char* name;
        std::array<Vec3, 4> positions;
        double d_hat;
        NodeTriangleRegion expected_region;
    };
    const std::vector<Case> cases{
        {"face_interior",
         {Vec3(0.25, 0.25, 0.3), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::FaceInterior},
        {"edge_12",
         {Vec3(0.5, -0.2, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Edge12},
        {"edge_23",
         {Vec3(0.7, 0.7, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Edge23},
        {"edge_31",
         {Vec3(-0.15, 0.5, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Edge31},
        {"vertex_1",
         {Vec3(-0.2, -0.3, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Vertex1},
        {"vertex_2",
         {Vec3(1.4, -0.1, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Vertex2},
        {"vertex_3",
         {Vec3(-0.1, 1.4, 0.1), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)},
         1.0, NodeTriangleRegion::Vertex3},
        {"degenerate_triangle",
         {Vec3(0.25, 0.2, 0.0), Vec3(0.0, 0.0, 0.0),
          Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)},
         1.0, NodeTriangleRegion::DegenerateTriangle},
    };

    constexpr double k_barrier = 37.0;
    constexpr double dt = 0.017;
    constexpr double eps_v = 0.41;
    constexpr double mu = 0.53;
    constexpr double dt2 = dt * dt;
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const std::array<Vec3, 4> previous =
            previous_with_general_slip(test_case.positions);
        const NodeTriangleContactEvaluation evaluation =
            make_node_triangle_contact_evaluation(
                test_case.positions, test_case.d_hat, k_barrier);
        ASSERT_EQ(evaluation.dr.region, test_case.expected_region);
        ASSERT_TRUE(evaluation.active);

        const FrozenFrictionContact legacy_contact =
            make_node_triangle_frozen_friction_contact(
                test_case.positions, previous, test_case.d_hat,
                k_barrier, dt, eps_v);
        const FrozenFrictionContact shared_contact =
            make_node_triangle_frozen_friction_contact(
                test_case.positions, previous, evaluation, dt, eps_v);
        expect_contact_exact(shared_contact, legacy_contact);

        for (int role = 0; role < 4; ++role) {
            const auto legacy_normal =
                node_triangle_barrier_self_gradient_and_hessian(
                    test_case.positions[0], test_case.positions[1],
                    test_case.positions[2], test_case.positions[3],
                    test_case.d_hat, role);
            const auto shared_normal =
                node_triangle_barrier_self_gradient_and_hessian(
                    test_case.positions[0], test_case.positions[1],
                    test_case.positions[2], test_case.positions[3],
                    role, evaluation);
            const auto legacy_friction =
                frozen_friction_role_gradient_and_hessian(
                    legacy_contact, role, mu, dt2);
            const auto shared_friction =
                frozen_friction_role_gradient_and_hessian(
                    shared_contact, role, mu, dt2);

            expect_vec_near(
                shared_friction.first, legacy_friction.first, 0.0);
            expect_mat_near(
                shared_friction.second, legacy_friction.second, 0.0);
            expect_vec_near(
                dt2 * k_barrier * shared_normal.first
                    + shared_friction.first,
                dt2 * k_barrier * legacy_normal.first
                    + legacy_friction.first,
                0.0);
            expect_mat_near(
                dt2 * k_barrier * shared_normal.second
                    + shared_friction.second,
                dt2 * k_barrier * legacy_normal.second
                    + legacy_friction.second,
                0.0);

            const auto zero_mu =
                frozen_friction_role_gradient_and_hessian(
                    shared_contact, role, 0.0, dt2);
            EXPECT_TRUE(zero_mu.first.isZero(0.0));
            EXPECT_TRUE(zero_mu.second.isZero(0.0));
        }
    }
}

TEST(FrictionCacheIntegration,
     SharedSegmentSegmentEvaluationMatchesLegacyAcrossAllFeatures) {
    struct Case {
        const char* name;
        std::array<Vec3, 4> positions;
        double d_hat;
        SegmentSegmentRegion expected_region;
    };
    const std::vector<Case> cases{
        {"interior",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(0.5, -1.0, 0.5), Vec3(0.5, 1.0, 0.5)},
         2.0, SegmentSegmentRegion::Interior},
        {"edge_s0",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(-1.0, -1.0, 0.3), Vec3(-1.0, 1.0, 0.3)},
         3.0, SegmentSegmentRegion::Edge_s0},
        {"edge_s1",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(2.0, -1.0, 0.3), Vec3(2.0, 1.0, 0.3)},
         3.0, SegmentSegmentRegion::Edge_s1},
        {"edge_t0",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(0.5, 0.3, 0.3), Vec3(0.5, 1.3, 0.3)},
         2.0, SegmentSegmentRegion::Edge_t0},
        {"edge_t1",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(0.5, -1.3, 0.3), Vec3(0.5, -0.3, 0.3)},
         2.0, SegmentSegmentRegion::Edge_t1},
        {"corner_s0t0",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(-0.5, -0.5, 0.3), Vec3(-0.5, -1.5, 0.3)},
         2.0, SegmentSegmentRegion::Corner_s0t0},
        {"corner_s0t1",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(-1.5, -1.5, 0.3), Vec3(-0.5, -0.5, 0.3)},
         2.0, SegmentSegmentRegion::Corner_s0t1},
        {"corner_s1t0",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(1.5, -0.5, 0.3), Vec3(1.5, -1.5, 0.3)},
         2.0, SegmentSegmentRegion::Corner_s1t0},
        {"corner_s1t1",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(2.5, -1.5, 0.3), Vec3(1.5, -0.5, 0.3)},
         2.0, SegmentSegmentRegion::Corner_s1t1},
        {"parallel_segments",
         {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
          Vec3(0.2, 0.2, 0.0), Vec3(0.8, 0.2, 0.0)},
         1.0, SegmentSegmentRegion::ParallelSegments},
    };

    constexpr double k_barrier = 41.0;
    constexpr double dt = 0.017;
    constexpr double eps_v = 0.41;
    constexpr double mu = 0.53;
    constexpr double dt2 = dt * dt;
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const std::array<Vec3, 4> previous =
            previous_with_general_slip(test_case.positions);
        const SegmentSegmentContactEvaluation evaluation =
            make_segment_segment_contact_evaluation(
                test_case.positions, test_case.d_hat, k_barrier);
        ASSERT_EQ(evaluation.dr.region, test_case.expected_region);
        ASSERT_TRUE(evaluation.active);

        const FrozenFrictionContact legacy_contact =
            make_segment_segment_frozen_friction_contact(
                test_case.positions, previous, test_case.d_hat,
                k_barrier, dt, eps_v);
        const FrozenFrictionContact shared_contact =
            make_segment_segment_frozen_friction_contact(
                test_case.positions, previous, evaluation, dt, eps_v);
        expect_contact_exact(shared_contact, legacy_contact);

        for (int role = 0; role < 4; ++role) {
            const auto legacy_normal =
                segment_segment_barrier_self_gradient_and_hessian(
                    test_case.positions[0], test_case.positions[1],
                    test_case.positions[2], test_case.positions[3],
                    test_case.d_hat, role);
            const auto shared_normal =
                segment_segment_barrier_self_gradient_and_hessian(
                    test_case.positions[0], test_case.positions[1],
                    test_case.positions[2], test_case.positions[3],
                    role, evaluation);
            const auto legacy_friction =
                frozen_friction_role_gradient_and_hessian(
                    legacy_contact, role, mu, dt2);
            const auto shared_friction =
                frozen_friction_role_gradient_and_hessian(
                    shared_contact, role, mu, dt2);

            expect_vec_near(
                shared_friction.first, legacy_friction.first, 0.0);
            expect_mat_near(
                shared_friction.second, legacy_friction.second, 0.0);
            expect_vec_near(
                dt2 * k_barrier * shared_normal.first
                    + shared_friction.first,
                dt2 * k_barrier * legacy_normal.first
                    + legacy_friction.first,
                0.0);
            expect_mat_near(
                dt2 * k_barrier * shared_normal.second
                    + shared_friction.second,
                dt2 * k_barrier * legacy_normal.second
                    + legacy_friction.second,
                0.0);

            const auto zero_mu =
                frozen_friction_role_gradient_and_hessian(
                    shared_contact, role, 0.0, dt2);
            EXPECT_TRUE(zero_mu.first.isZero(0.0));
            EXPECT_TRUE(zero_mu.second.isZero(0.0));
        }
    }
}

TEST(FrictionCacheIntegration,
     SharedInactiveEvaluationsPreserveLegacyZeroDerivatives) {
    constexpr double d_hat = 0.5;
    constexpr double k_barrier = 23.0;
    constexpr double dt = 0.02;
    constexpr double eps_v = 0.3;
    constexpr double mu = 0.6;

    const std::array<Vec3, 4> nt_current{
        Vec3(0.2, 0.3, 0.8), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const std::array<Vec3, 4> nt_previous =
        previous_with_general_slip(nt_current);
    const NodeTriangleContactEvaluation nt_evaluation =
        make_node_triangle_contact_evaluation(
            nt_current, d_hat, k_barrier);
    ASSERT_FALSE(nt_evaluation.active);
    const FrozenFrictionContact nt_legacy =
        make_node_triangle_frozen_friction_contact(
            nt_current, nt_previous, d_hat, k_barrier, dt, eps_v);
    const FrozenFrictionContact nt_shared =
        make_node_triangle_frozen_friction_contact(
            nt_current, nt_previous, nt_evaluation, dt, eps_v);
    expect_contact_exact(nt_shared, nt_legacy);

    const std::array<Vec3, 4> ss_current{
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
        Vec3(0.2, 0.8, 0.0), Vec3(0.8, 0.8, 0.0)};
    const std::array<Vec3, 4> ss_previous =
        previous_with_general_slip(ss_current);
    const SegmentSegmentContactEvaluation ss_evaluation =
        make_segment_segment_contact_evaluation(
            ss_current, d_hat, k_barrier);
    ASSERT_FALSE(ss_evaluation.active);
    const FrozenFrictionContact ss_legacy =
        make_segment_segment_frozen_friction_contact(
            ss_current, ss_previous, d_hat, k_barrier, dt, eps_v);
    const FrozenFrictionContact ss_shared =
        make_segment_segment_frozen_friction_contact(
            ss_current, ss_previous, ss_evaluation, dt, eps_v);
    expect_contact_exact(ss_shared, ss_legacy);

    for (int role = 0; role < 4; ++role) {
        const auto nt_derivatives =
            frozen_friction_role_gradient_and_hessian(
                nt_shared, role, mu, dt * dt);
        EXPECT_TRUE(nt_derivatives.first.isZero(0.0));
        EXPECT_TRUE(nt_derivatives.second.isZero(0.0));
        const auto ss_derivatives =
            frozen_friction_role_gradient_and_hessian(
                ss_shared, role, mu, dt * dt);
        EXPECT_TRUE(ss_derivatives.first.isZero(0.0));
        EXPECT_TRUE(ss_derivatives.second.isZero(0.0));
    }
}

TEST(FrictionCacheIntegration,
     SharedZeroStiffnessCoincidentContactsSkipFrictionGeometry) {
    constexpr double d_hat = 0.5;
    constexpr double dt = 0.02;
    constexpr double eps_v = 0.3;

    const std::array<Vec3, 4> nt_current{
        Vec3(0.2, 0.3, 0.0), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const NodeTriangleContactEvaluation nt_evaluation =
        make_node_triangle_contact_evaluation(
            nt_current, d_hat, 0.0);
    ASSERT_DOUBLE_EQ(nt_evaluation.dr.distance, 0.0);
    const FrozenFrictionContact nt_shared =
        make_node_triangle_frozen_friction_contact(
            nt_current, nt_current, nt_evaluation, dt, eps_v);
    EXPECT_FALSE(nt_shared.active);
    EXPECT_DOUBLE_EQ(nt_shared.normal_force, 0.0);
    EXPECT_DOUBLE_EQ(nt_shared.eps_u, dt * eps_v);

    const std::array<Vec3, 4> ss_current{
        Vec3(-1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const SegmentSegmentContactEvaluation ss_evaluation =
        make_segment_segment_contact_evaluation(
            ss_current, d_hat, 0.0);
    ASSERT_DOUBLE_EQ(ss_evaluation.dr.distance, 0.0);
    const FrozenFrictionContact ss_shared =
        make_segment_segment_frozen_friction_contact(
            ss_current, ss_current, ss_evaluation, dt, eps_v);
    EXPECT_FALSE(ss_shared.active);
    EXPECT_DOUBLE_EQ(ss_shared.normal_force, 0.0);
    EXPECT_DOUBLE_EQ(ss_shared.eps_u, dt * eps_v);
}

TEST(FrozenSdfFrictionContact,
     StaticPlaneUsesPenaltyLoadAndParticleTangentialSlip) {
    constexpr double k_sdf = 80.0;
    constexpr double eps_sdf = 0.10;
    constexpr double dt = 0.02;
    constexpr double eps_v = 0.5;
    const PlaneSDF plane{Vec3::Zero(), Vec3::UnitY()};
    const Vec3 current(0.08, 0.04, -0.06);
    const Vec3 particle_displacement(0.03, 0.02, -0.04);
    const Vec3 previous = current - particle_displacement;
    const SDFEvaluation sdf = evaluate_sdf(plane, current);

    const FrozenFrictionContact contact =
        make_sdf_frozen_friction_contact(
            current, previous, sdf, k_sdf, eps_sdf, dt, eps_v);

    ASSERT_TRUE(contact.active);
    expect_weights_near(contact.weights, {{1.0, 0.0, 0.0, 0.0}});
    expect_vec_near(contact.normal, Vec3::UnitY());
    expect_mat_near(
        contact.projector,
        Mat33::Identity() - Vec3::UnitY() * Vec3::UnitY().transpose());
    expect_vec_near(
        contact.tangential_displacement, Vec3(0.03, 0.0, -0.04));
    EXPECT_DOUBLE_EQ(contact.eps_u, dt * eps_v);
    EXPECT_NEAR(
        contact.normal_force,
        -sdf_penalty_gradient(sdf, k_sdf, eps_sdf).dot(Vec3::UnitY()),
        1.0e-13);
    EXPECT_GT(contact.normal_force, 0.0);

    constexpr double mu = 0.35;
    constexpr double dt2 = dt * dt;
    const Vec3 gradient =
        frozen_friction_role_gradient(contact, 0, mu, dt2);
    EXPECT_GT(gradient.dot(contact.tangential_displacement), 0.0);
    EXPECT_NEAR(gradient.dot(contact.normal), 0.0, 1.0e-14);

    // Freeze the SDF evaluation, normal load, and tangent plane, as the
    // production block solver does, and differentiate only the particle slip.
    constexpr double h = 1.0e-7;
    for (int component = 0; component < 3; ++component) {
        Vec3 plus = current;
        Vec3 minus = current;
        plus[component] += h;
        minus[component] -= h;
        const FrozenFrictionContact plus_contact =
            make_sdf_frozen_friction_contact(
                plus, previous, sdf, k_sdf, eps_sdf, dt, eps_v);
        const FrozenFrictionContact minus_contact =
            make_sdf_frozen_friction_contact(
                minus, previous, sdf, k_sdf, eps_sdf, dt, eps_v);
        const double finite_difference =
            (frozen_friction_energy(plus_contact, mu, dt2)
             - frozen_friction_energy(minus_contact, mu, dt2))
            / (2.0 * h);
        EXPECT_NEAR(finite_difference, gradient[component], 2.0e-9)
            << "component=" << component;
    }

    const Mat33 hessian =
        frozen_friction_role_hessian(contact, 0, mu, dt2);
    EXPECT_TRUE(hessian.isApprox(hessian.transpose(), 1.0e-14));
    EXPECT_TRUE((hessian * contact.normal).isZero(1.0e-14));
    Eigen::SelfAdjointEigenSolver<Mat33> eigen_solver(hessian);
    ASSERT_EQ(eigen_solver.info(), Eigen::Success);
    EXPECT_GE(eigen_solver.eigenvalues().minCoeff(), -1.0e-13);
}

TEST(FrozenSdfFrictionContact,
     TranslatingPlaneSubtractsPrescribedSurfaceMotion) {
    PlaneSDF plane{Vec3::Zero(), Vec3::UnitY()};
    const Vec3 surface_translation(0.07, 0.0, -0.03);
    plane.material_motion.previous.translation = Vec3::Zero();
    plane.material_motion.current.translation = surface_translation;

    const Vec3 current(0.31, 0.04, -0.16);
    const Vec3 comoving_previous = current - surface_translation;
    const SDFEvaluation sdf = evaluate_sdf(plane, current);
    const FrozenFrictionContact comoving =
        make_sdf_frozen_friction_contact(
            current, comoving_previous, sdf,
            90.0, 0.10, 0.02, 0.5);
    ASSERT_TRUE(comoving.active);
    expect_vec_near(comoving.tangential_displacement, Vec3::Zero(), 1.0e-14);

    const FrozenFrictionContact stationary_particle =
        make_sdf_frozen_friction_contact(
            current, current, sdf, 90.0, 0.10, 0.02, 0.5);
    ASSERT_TRUE(stationary_particle.active);
    expect_vec_near(
        stationary_particle.tangential_displacement,
        -surface_translation, 1.0e-14);
    const Vec3 gradient = frozen_friction_relative_gradient(
        stationary_particle, 0.6, 0.02 * 0.02);
    EXPECT_GT(gradient.dot(-surface_translation), 0.0);
}

TEST(FrozenSdfFrictionContact,
     SpinningSphereTracksTheClosestMaterialPoint) {
    constexpr double half_pi = 1.57079632679489661923;
    Mat33 current_rotation;
    current_rotation <<
        std::cos(half_pi), -std::sin(half_pi), 0.0,
        std::sin(half_pi),  std::cos(half_pi), 0.0,
        0.0,                0.0,               1.0;

    SphereSDF sphere{Vec3::Zero(), 1.0};
    sphere.material_motion.current.rotation = current_rotation;
    const Vec3 current(1.04, 0.0, 0.0);
    const SDFEvaluation sdf = evaluate_sdf(sphere, current);
    expect_vec_near(sdf.surface_point, Vec3::UnitX(), 1.0e-14);

    // Current +x material came from -y before the +90 degree spin.
    const Vec3 surface_displacement(1.0, 1.0, 0.0);
    const FrozenFrictionContact comoving =
        make_sdf_frozen_friction_contact(
            current, current - surface_displacement, sdf,
            50.0, 0.10, 0.01, 0.2);
    ASSERT_TRUE(comoving.active);
    expect_vec_near(comoving.tangential_displacement, Vec3::Zero(), 1.0e-13);

    const FrozenFrictionContact stationary_particle =
        make_sdf_frozen_friction_contact(
            current, current, sdf, 50.0, 0.10, 0.01, 0.2);
    ASSERT_TRUE(stationary_particle.active);
    expect_vec_near(
        stationary_particle.tangential_displacement,
        Vec3(0.0, -1.0, 0.0), 1.0e-13);
}

TEST(FrozenSdfFrictionContact,
     InactiveDegenerateAndInvalidInputsAreHandled) {
    const PlaneSDF plane{Vec3::Zero(), Vec3::UnitY()};
    const Vec3 outside(0.2, 0.2, -0.1);
    const SDFEvaluation outside_sdf = evaluate_sdf(plane, outside);
    EXPECT_FALSE(make_sdf_frozen_friction_contact(
        outside, outside - Vec3::UnitX(), outside_sdf,
        30.0, 0.1, 0.01, 0.5).active);
    EXPECT_FALSE(make_sdf_frozen_friction_contact(
        outside, outside - Vec3::UnitX(), outside_sdf,
        0.0, 0.1, 0.01, 0.5).active);

    const SphereSDF sphere{Vec3::Zero(), 1.0};
    const SDFEvaluation center_sdf = evaluate_sdf(sphere, Vec3::Zero());
    EXPECT_FALSE(make_sdf_frozen_friction_contact(
        Vec3::Zero(), -Vec3::UnitX(), center_sdf,
        30.0, 0.1, 0.01, 0.5).active);

    EXPECT_THROW(make_sdf_frozen_friction_contact(
        outside, outside, outside_sdf, -1.0, 0.1, 0.01, 0.5),
        std::invalid_argument);
    EXPECT_THROW(make_sdf_frozen_friction_contact(
        outside, outside, outside_sdf, 1.0, 0.1, 0.0, 0.5),
        std::invalid_argument);
    EXPECT_THROW(make_sdf_frozen_friction_contact(
        outside, outside, outside_sdf, 1.0, 0.1, 0.01, 0.0),
        std::invalid_argument);

    PlaneSDF invalid_motion{Vec3::Zero(), Vec3::UnitY()};
    invalid_motion.material_motion.current.translation = Vec3::UnitX();
    invalid_motion.material_motion.current.rotation(0, 0) = 2.0;
    const Vec3 active_position(0.0, 0.04, 0.0);
    const SDFEvaluation active_sdf =
        evaluate_sdf(invalid_motion, active_position);
    EXPECT_THROW(make_sdf_frozen_friction_contact(
        active_position, active_position, active_sdf,
        30.0, 0.1, 0.01, 0.5), std::invalid_argument);
}

TEST(FrozenFrictionDerivatives, FrozenEnergyGradientFiniteDifference) {
    const std::array<Vec3, 4> current = {{
        Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
    }};
    FrozenFrictionContact contact = make_node_triangle_frozen_friction_contact(
            current, previous_with_node_slip(current),
            0.5, 17.0, 0.01, 0.5);
    ASSERT_TRUE(contact.active);
    ASSERT_GT(contact.tangential_displacement.norm(), contact.eps_u);

    constexpr double mu = 0.42;
    constexpr double dt2 = 0.0001;
    constexpr double h = 1.0e-7;
    for (int role = 0; role < 4; ++role) {
        const Vec3 gradient = frozen_friction_role_gradient(
                contact, role, mu, dt2);
        for (int component = 0; component < 3; ++component) {
            FrozenFrictionContact plus = contact;
            FrozenFrictionContact minus = contact;
            const Vec3 frozen_change =
                    contact.weights[role] * contact.projector.col(component) * h;
            plus.tangential_displacement += frozen_change;
            minus.tangential_displacement -= frozen_change;
            const double finite_difference =
                    (frozen_friction_energy(plus, mu, dt2)
                     - frozen_friction_energy(minus, mu, dt2))
                    / (2.0 * h);
            EXPECT_NEAR(finite_difference, gradient[component], 2.0e-9);
        }
    }
}

TEST(FrozenFrictionDerivatives, HessiansAreSymmetricPositiveSemidefinite) {
    FrozenFrictionContact contact;
    contact.weights = {{0.7, 0.3, -0.4, -0.6}};
    contact.normal = Vec3(1.0, 2.0, 3.0).normalized();
    contact.projector = Mat33::Identity() - contact.normal * contact.normal.transpose();
    contact.tangential_displacement = contact.projector * Vec3(0.4, -0.2, 0.1);
    contact.normal_force = 9.0;
    contact.eps_u = 0.1;
    contact.active = true;

    const Mat33 relative_hessian =
            frozen_friction_relative_hessian(contact, 0.5, 0.04);
    expect_mat_near(relative_hessian, relative_hessian.transpose(), 1.0e-14);
    Eigen::SelfAdjointEigenSolver<Mat33> relative_solver(relative_hessian);
    ASSERT_EQ(relative_solver.info(), Eigen::Success);
    EXPECT_GE(relative_solver.eigenvalues().minCoeff(), -1.0e-13);

    for (int role = 0; role < 4; ++role) {
        const Mat33 role_hessian = frozen_friction_role_hessian(
                contact, role, 0.5, 0.04);
        Eigen::SelfAdjointEigenSolver<Mat33> role_solver(role_hessian);
        ASSERT_EQ(role_solver.info(), Eigen::Success);
        EXPECT_GE(role_solver.eigenvalues().minCoeff(), -1.0e-13);

        const auto combined = frozen_friction_role_gradient_and_hessian(
                contact, role, 0.5, 0.04);
        expect_vec_near(
                combined.first,
                frozen_friction_role_gradient(contact, role, 0.5, 0.04));
        expect_mat_near(combined.second, role_hessian);
    }
}

TEST(FrozenFrictionDerivatives,
     CombinedRelativeEvaluationMatchesSeparateCallsAcrossSlipRegimes) {
    FrozenFrictionContact contact;
    contact.weights = {{0.7, 0.3, -0.4, -0.6}};
    contact.normal = Vec3::UnitZ();
    contact.projector =
        Mat33::Identity() - contact.normal * contact.normal.transpose();
    contact.normal_force = 9.0;
    contact.eps_u = 0.1;
    contact.active = true;

    const std::array<Vec3, 3> slips{{
        Vec3::Zero(),
        Vec3(0.03, -0.04, 0.0),
        Vec3(0.3, 0.4, 0.0)}};
    for (const Vec3& slip : slips) {
        SCOPED_TRACE(slip.transpose());
        contact.tangential_displacement = slip;
        const auto combined =
            frozen_friction_relative_gradient_and_hessian(
                contact, 0.37, 0.0025);
        expect_vec_near(
            combined.first,
            frozen_friction_relative_gradient(contact, 0.37, 0.0025),
            0.0);
        expect_mat_near(
            combined.second,
            frozen_friction_relative_hessian(contact, 0.37, 0.0025),
            0.0);
    }

    contact.tangential_displacement = Vec3(0.3, 0.4, 0.0);
    contact.active = false;
    const auto inactive = frozen_friction_relative_gradient_and_hessian(
        contact, 0.37, 0.0025);
    expect_vec_near(inactive.first, Vec3::Zero(), 0.0);
    expect_mat_near(inactive.second, Mat33::Zero(), 0.0);

    contact.active = true;
    const auto zero_coefficient =
        frozen_friction_relative_gradient_and_hessian(
            contact, 0.0, 0.0025);
    expect_vec_near(zero_coefficient.first, Vec3::Zero(), 0.0);
    expect_mat_near(zero_coefficient.second, Mat33::Zero(), 0.0);
}

TEST(FrozenSdfFrictionContact,
     PrecomputedPenaltyGradientMatchesUncachedBuilder) {
    PlaneSDF plane{Vec3::Zero(), Vec3::UnitY()};
    plane.material_motion.previous.translation = Vec3(-0.02, 0.0, 0.01);
    plane.material_motion.current.translation = Vec3(0.03, 0.0, -0.04);

    const Vec3 current(0.31, 0.04, -0.16);
    const Vec3 previous(0.27, 0.03, -0.11);
    const SDFEvaluation sdf = evaluate_sdf(plane, current);
    constexpr double k_sdf = 90.0;
    constexpr double eps_sdf = 0.10;
    constexpr double dt = 0.02;
    constexpr double eps_v = 0.5;
    const Vec3 penalty_gradient =
        sdf_penalty_gradient(sdf, k_sdf, eps_sdf);

    const FrozenFrictionContact uncached =
        make_sdf_frozen_friction_contact(
            current, previous, sdf, k_sdf, eps_sdf, dt, eps_v);
    const FrozenFrictionContact cached =
        make_sdf_frozen_friction_contact(
            current, previous, sdf, k_sdf, eps_sdf, dt, eps_v,
            1.0e-12, &penalty_gradient);

    ASSERT_TRUE(uncached.active);
    ASSERT_EQ(cached.active, uncached.active);
    expect_weights_near(cached.weights, uncached.weights, 0.0);
    expect_vec_near(cached.normal, uncached.normal, 0.0);
    expect_mat_near(cached.projector, uncached.projector, 0.0);
    expect_vec_near(
        cached.tangential_displacement,
        uncached.tangential_displacement, 0.0);
    EXPECT_DOUBLE_EQ(cached.normal_force, uncached.normal_force);
    EXPECT_DOUBLE_EQ(cached.eps_u, uncached.eps_u);

    // A deliberately scaled cached gradient verifies that the supplied value
    // is consumed instead of silently recomputing the penalty gradient.
    const Vec3 twice_penalty_gradient = 2.0 * penalty_gradient;
    const FrozenFrictionContact scaled =
        make_sdf_frozen_friction_contact(
            current, previous, sdf, k_sdf, eps_sdf, dt, eps_v,
            1.0e-12, &twice_penalty_gradient);
    ASSERT_TRUE(scaled.active);
    EXPECT_DOUBLE_EQ(scaled.normal_force, 2.0 * uncached.normal_force);
    expect_vec_near(
        scaled.tangential_displacement,
        uncached.tangential_displacement, 0.0);
}

TEST(FrozenFrictionDerivatives, ActionReactionAndCommonTranslationCancel) {
    const std::array<Vec3, 4> current = {{
        Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
    }};
    const FrozenFrictionContact contact =
            make_node_triangle_frozen_friction_contact(
                    current, previous_with_node_slip(current),
                    0.5, 11.0, 0.02, 0.5);
    ASSERT_TRUE(contact.active);

    double weight_sum = 0.0;
    Vec3 gradient_sum = Vec3::Zero();
    for (int role = 0; role < 4; ++role) {
        weight_sum += contact.weights[role];
        gradient_sum += frozen_friction_role_gradient(
                contact, role, 0.3, 0.0004);
    }
    EXPECT_NEAR(weight_sum, 0.0, 1.0e-14);
    expect_vec_near(gradient_sum, Vec3::Zero(), 1.0e-13);

    const Vec3 common_increment(0.2, -0.1, 0.4);
    FrozenFrictionContact translated = contact;
    translated.tangential_displacement +=
            weight_sum * contact.projector * common_increment;
    EXPECT_DOUBLE_EQ(
            frozen_friction_energy(translated, 0.3, 0.0004),
            frozen_friction_energy(contact, 0.3, 0.0004));
}

TEST(FrozenFrictionDerivatives, EvaluationIsLinearInDtSquared) {
    FrozenFrictionContact contact;
    contact.weights = {{1.0, -1.0, 0.0, 0.0}};
    contact.normal = Vec3::UnitZ();
    contact.projector = Mat33::Identity() - contact.normal * contact.normal.transpose();
    contact.tangential_displacement = Vec3(0.2, -0.1, 0.0);
    contact.normal_force = 5.0;
    contact.eps_u = 0.01;
    contact.active = true;

    const double energy_1 = frozen_friction_energy(contact, 0.4, 0.01);
    const double energy_2 = frozen_friction_energy(contact, 0.4, 0.07);
    EXPECT_NEAR(energy_2, 7.0 * energy_1, 1.0e-14);
    expect_vec_near(
            frozen_friction_relative_gradient(contact, 0.4, 0.07),
            7.0 * frozen_friction_relative_gradient(contact, 0.4, 0.01),
            1.0e-14);
    expect_mat_near(
            frozen_friction_relative_hessian(contact, 0.4, 0.07),
            7.0 * frozen_friction_relative_hessian(contact, 0.4, 0.01),
            1.0e-14);
}

TEST(FrozenFrictionContact, InactiveContactsAndZeroCoefficientAreZero) {
    const std::array<Vec3, 4> far = {{
        Vec3(0.2, 0.3, 0.8), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
    }};
    const FrozenFrictionContact outside =
            make_node_triangle_frozen_friction_contact(
                    far, previous_with_node_slip(far),
                    0.5, 10.0, 0.01, 1.0);
    EXPECT_FALSE(outside.active);
    EXPECT_DOUBLE_EQ(frozen_friction_energy(outside, 0.5, 0.0001), 0.0);
    expect_vec_near(
            frozen_friction_relative_gradient(outside, 0.5, 0.0001),
            Vec3::Zero(), 0.0);
    expect_mat_near(
            frozen_friction_relative_hessian(outside, 0.5, 0.0001),
            Mat33::Zero(), 0.0);

    const std::array<Vec3, 4> near = {{
        Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
    }};
    const FrozenFrictionContact no_barrier =
            make_node_triangle_frozen_friction_contact(
                    near, previous_with_node_slip(near),
                    0.5, 0.0, 0.01, 1.0);
    EXPECT_FALSE(no_barrier.active);

    const FrozenFrictionContact active =
            make_node_triangle_frozen_friction_contact(
                    near, previous_with_node_slip(near),
                    0.5, 10.0, 0.01, 1.0);
    ASSERT_TRUE(active.active);
    EXPECT_DOUBLE_EQ(frozen_friction_energy(active, 0.0, 0.0001), 0.0);
    expect_vec_near(
            frozen_friction_relative_gradient(active, 0.0, 0.0001),
            Vec3::Zero(), 0.0);
}

TEST(FrozenFrictionValidation, RejectsInvalidPhysicalParameters) {
    const std::array<Vec3, 4> positions = {{
        Vec3(0.2, 0.3, 0.1), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
    }};
    const auto previous = previous_with_node_slip(positions);

    EXPECT_THROW(
            make_node_triangle_frozen_friction_contact(
                    positions, previous, 0.5, 10.0, 0.0, 1.0),
            std::invalid_argument);
    EXPECT_THROW(
            make_node_triangle_frozen_friction_contact(
                    positions, previous, 0.5, 10.0, 0.01, 0.0),
            std::invalid_argument);
    EXPECT_THROW(
            make_node_triangle_frozen_friction_contact(
                    positions, previous, 0.5, -1.0, 0.01, 1.0),
            std::invalid_argument);
    EXPECT_THROW(
            smooth_friction_potential(-1.0, 0.1),
            std::invalid_argument);
    EXPECT_THROW(
            smooth_friction_mollifier_over_slip(0.1, 0.0),
            std::invalid_argument);

    const FrozenFrictionContact contact =
            make_node_triangle_frozen_friction_contact(
                    positions, previous, 0.5, 10.0, 0.01, 1.0);
    EXPECT_THROW(
            frozen_friction_energy(contact, -0.1, 0.0001),
            std::invalid_argument);
    EXPECT_THROW(
            frozen_friction_relative_gradient(contact, 0.1, -0.0001),
            std::invalid_argument);
    EXPECT_THROW(
            frozen_friction_role_hessian(contact, 4, 0.1, 0.0001),
            std::invalid_argument);
    EXPECT_THROW(
            make_node_triangle_frozen_friction_contact(
                    positions, previous, 0.5,
                    std::numeric_limits<double>::infinity(), 0.01, 1.0),
            std::invalid_argument);
}

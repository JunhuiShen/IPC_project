#include "ogc_contact.h"
#include "friction_energy.h"
#include "gpu/resident_dat_math.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

using Points = std::array<Vec3, 4>;

Points obtuse_triangle(const Vec3& point = Vec3(.5,-.01,.1)) {
    return {point, Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)};
}

NodeTriangleDistanceResult corrected(const Points& x) {
    return ogc_detail::node_triangle_distance(x[0], x[1], x[2], x[3]);
}

void expect_gpu_contact_matches_cpu(const Points& x) {
    namespace resident = ipc_gpu::resident;
    resident::ContactInput input;
    input.d_hat = 2.0;
    input.k_barrier = 100.0;
    input.dt = .01;
    input.friction = .3;
    input.eps_v = .001;
    Points previous = x;
    previous[0] -= Vec3(.01,.005,0);
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        input.positions[i][k] = x[i][k];
        input.previous[i][k] = previous[i][k];
    }
    const auto gpu = resident::dat::evaluate(input);
    ASSERT_EQ(gpu.status, resident::ContactStatus::Ok);
    ASSERT_TRUE(gpu.active);
    const auto dr = corrected(x);
    const auto evaluation = make_node_triangle_contact_evaluation(x, input.d_hat, input.k_barrier, input.eps, &dr, false);
    const auto friction = make_node_triangle_frozen_friction_contact(x, previous, evaluation, input.dt, input.eps_v);
    const auto [fg, fH] = frozen_friction_relative_gradient_and_hessian(friction, input.friction, input.dt * input.dt);
    EXPECT_NEAR(gpu.distance, dr.distance, 1e-12);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(gpu.weights[i], friction.weights[i], 1e-12);
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], i, evaluation);
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(gpu.barrier_gradient[i][k], g[k], 1e-8 * std::max(1.0, std::abs(g[k])));
            EXPECT_NEAR(gpu.friction_relative_gradient[k], fg[k], 1e-10 * std::max(1.0, std::abs(fg[k])));
            for (int l = 0; l < 3; ++l) {
                EXPECT_NEAR(gpu.barrier_hessian[i][i][3*k+l], H(k,l), 1e-7 * std::max(1.0, std::abs(H(k,l))));
                EXPECT_NEAR(gpu.friction_relative_hessian[3*k+l], fH(k,l), 1e-9 * std::max(1.0, std::abs(fH(k,l))));
            }
        }
    }
}

} // namespace

TEST(OGCContact, SharedGSDistanceRetainsItsExistingBehavior) {
    const auto x = obtuse_triangle();
    const auto legacy = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
    EXPECT_EQ(legacy.region, NodeTriangleRegion::Vertex1);
    EXPECT_EQ(legacy.closest_point, x[1]);
    EXPECT_NEAR(legacy.distance, .51, 1e-14);
    const auto fixed = corrected(x);
    EXPECT_EQ(fixed.region, NodeTriangleRegion::Edge12);
    EXPECT_TRUE(fixed.closest_point.isApprox(Vec3(.5,0,0), 1e-13));
    EXPECT_NEAR(fixed.distance, std::sqrt(.0101), 1e-14);
}

TEST(OGCContact, ObtuseClosestEdgeIsIndependentOfVertexOrderAndOrientation) {
    const auto initial = obtuse_triangle();
    const Mat33 rotation = Eigen::AngleAxisd(.73, Vec3(1,2,3).normalized()).toRotationMatrix();
    for (bool transformed : {false, true}) {
        const auto transform = [&](const Vec3& p) -> Vec3 { return transformed ? Vec3(rotation * p + Vec3(2,-3,4)) : p; };
        std::array<int, 3> order{1,2,3};
        do {
            const Points x{transform(initial[0]), transform(initial[order[0]]), transform(initial[order[1]]), transform(initial[order[2]])};
            const auto dr = corrected(x);
            EXPECT_LE((dr.closest_point - transform(Vec3(.5,0,0))).norm(), 1e-12);
            EXPECT_NEAR(dr.distance, std::sqrt(.0101), 1e-12);
            EXPECT_TRUE(dr.region == NodeTriangleRegion::Edge12 || dr.region == NodeTriangleRegion::Edge23 || dr.region == NodeTriangleRegion::Edge31);
            expect_gpu_contact_matches_cpu(x);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST(OGCContact, ClampedEndpointsUseVertexDerivatives) {
    const auto x = obtuse_triangle(Vec3(1.5,-.1,.1));
    const auto dr = corrected(x);
    ASSERT_EQ(dr.region, NodeTriangleRegion::Vertex2);
    EXPECT_TRUE(dr.closest_point.isApprox(x[2]));
    EXPECT_NEAR(dr.distance, std::sqrt(.27), 1e-14);
    constexpr double step = 1e-6;
    for (int role = 0; role < 4; ++role) {
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], 2.0, role, 1e-12, &dr);
        for (int k = 0; k < 3; ++k) {
            auto plus = x, minus = x;
            plus[role][k] += step;
            minus[role][k] -= step;
            const auto dp = corrected(plus), dm = corrected(minus);
            const double difference = (scalar_barrier(dp.distance, 2.0) - scalar_barrier(dm.distance, 2.0)) / (2*step);
            EXPECT_NEAR(g[k], difference, 1e-7);
            const Vec3 gp = node_triangle_barrier_gradient(plus[0], plus[1], plus[2], plus[3], 2.0, role, 1e-12, &dp);
            const Vec3 gm = node_triangle_barrier_gradient(minus[0], minus[1], minus[2], minus[3], 2.0, role, 1e-12, &dm);
            EXPECT_LE((H.col(k) - (gp-gm)/(2*step)).norm(), 1e-6);
        }
    }
    expect_gpu_contact_matches_cpu(x);
}

TEST(OGCContact, FaceAndDegenerateMetadataArePreserved) {
    const std::array<Points, 2> cases{Points{Vec3(.2,.3,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(0,1,0)}, Points{Vec3(.5,1,.2), Vec3::Zero(), Vec3(1,0,0), Vec3(2,0,0)}};
    for (const auto& x : cases) {
        const auto original = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
        const auto fixed = corrected(x);
        EXPECT_EQ(fixed.region, original.region);
        EXPECT_EQ(fixed.closest_point, original.closest_point);
        EXPECT_EQ(fixed.distance, original.distance);
        EXPECT_EQ(fixed.phi, original.phi);
        EXPECT_EQ(fixed.normal, original.normal);
        EXPECT_EQ(fixed.tilde_x, original.tilde_x);
        EXPECT_EQ(fixed.barycentric_tilde_x, original.barycentric_tilde_x);
    }
    expect_gpu_contact_matches_cpu(cases[0]);
}

TEST(OGCContact, GPUPlanarGeometryUsesCorrectedClosestPoint) {
    const auto x = obtuse_triangle();
    double positions[4][3];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) positions[i][k] = x[i][k];
    const auto g = ipc_gpu::resident::dat::geometry(positions, true, .15);
    ASSERT_TRUE(g.near);
    ASSERT_TRUE(g.valid);
    EXPECT_NEAR(g.distance, std::sqrt(.0101), 1e-14);
    const Vec3 normal = Vec3(0,-.01,.1).normalized();
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(g.normal[k], normal[k], 1e-14);
}

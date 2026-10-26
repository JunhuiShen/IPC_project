#include "ogc_contact.h"
#include "friction_energy.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

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
}

namespace {

struct OgcSolverFixture {
    RefMesh mesh;
    SimParams params = SimParams::zeros();
    std::vector<Vec3> initial{Vec3(.5,-.01,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)};
    std::vector<Vec3> target = initial;
    std::vector<Vec3> velocity = std::vector<Vec3>(initial.size(), Vec3::Zero());

    OgcSolverFixture() {
        mesh.num_positions = initial.size();
        mesh.mass.assign(initial.size(), 1.0);
        mesh.node_to_rb.assign(initial.size(), -1);
        mesh.tris = {1,2,3};
        params.fixed_iters = true;
        params.max_global_iters = 1;
        params.damping = 1.0;
        params.d_hat = 1.0;
        params.k_barrier = 0.0;
        params.node_box_min = params.node_box_max = 1.0;
    }

    std::vector<Vec3> solve() {
        auto x = initial;
        const auto result = global_gauss_seidel_solver_ogc(mesh, {}, {}, params, x, target, velocity, "", &initial);
        EXPECT_TRUE(result.converged);
        return x;
    }
};

} // namespace

TEST(OGCSolver, ObtuseTriangleUsesCorrectedDistanceForMotionBound) {
    OgcSolverFixture scene;
    const Vec3 proposal(0,.02,-.2);
    scene.target[0] += proposal;
    const auto x = scene.solve();
    EXPECT_LE((x[0] - (scene.initial[0] + .2*proposal)).norm(), 1e-13);
    EXPECT_NEAR((x[0]-scene.initial[0]).norm(), .4*std::sqrt(.0101), 1e-13);
    for (int i = 1; i < 4; ++i) EXPECT_EQ(x[i], scene.initial[i]);
}

TEST(OGCSolver, CorrectedDistanceActivatesBarrierAndFrictionAssembly) {
    for (double friction : {0.0,.3}) {
        OgcSolverFixture scene;
        scene.params.d_hat = .2;
        scene.params.k_barrier = 100;
        scene.params.friction_coefficient = friction;
        const auto x = scene.solve();
        EXPECT_GT(Vec3(0,-.01,.1).dot(x[0]-scene.initial[0]), 0.0);
    }
}

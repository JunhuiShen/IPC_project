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

NodeTriangleDistanceResult cached_distance(const Points& x) {
    RefMesh mesh;
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    mesh.node_to_rb.assign(x.size(), -1);
    mesh.tris = {1,2,3};
    const std::vector<Vec3> positions(x.begin(), x.end());
    std::vector<AABB> boxes;
    for (const auto& point : positions) boxes.emplace_back(point, point);
    BroadPhase phase;
    phase.initialize(boxes, mesh, 2.0);
    ogc_detail::ContactCache contacts;
    contacts.evaluate(phase, positions, false, true);
    EXPECT_EQ(contacts.nt_distances.size(), 1);
    return contacts.nt_distances.at(0);
}

void expect_same_distance(const NodeTriangleDistanceResult& cached, const NodeTriangleDistanceResult& shared) {
    EXPECT_EQ(cached.region, shared.region);
    EXPECT_EQ(cached.closest_point, shared.closest_point);
    EXPECT_EQ(cached.distance, shared.distance);
    EXPECT_EQ(cached.phi, shared.phi);
    EXPECT_EQ(cached.normal, shared.normal);
    EXPECT_EQ(cached.tilde_x, shared.tilde_x);
    EXPECT_EQ(cached.barycentric_tilde_x, shared.barycentric_tilde_x);
}

} // namespace

TEST(OGCContact, OGCAndGSUseTheSameDistanceBehavior) {
    const auto x = obtuse_triangle();
    const auto shared = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
    EXPECT_EQ(shared.region, NodeTriangleRegion::Edge12);
    EXPECT_LE((shared.closest_point - Vec3(.5,0,0)).norm(), 1e-13);
    EXPECT_NEAR(shared.distance, std::sqrt(.0101), 1e-14);
    expect_same_distance(cached_distance(x), shared);
}

TEST(OGCContact, CachedDistanceMatchesSharedAcrossVertexOrdersAndOrientations) {
    const auto initial = obtuse_triangle();
    const Mat33 rotation = Eigen::AngleAxisd(.73, Vec3(1,2,3).normalized()).toRotationMatrix();
    for (bool transformed : {false, true}) {
        const auto transform = [&](const Vec3& p) -> Vec3 { return transformed ? Vec3(rotation * p + Vec3(2,-3,4)) : p; };
        std::array<int, 3> order{1,2,3};
        do {
            const Points x{transform(initial[0]), transform(initial[order[0]]), transform(initial[order[1]]), transform(initial[order[2]])};
            const auto shared = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
            const auto expected_edge = order[2] == 3 ? NodeTriangleRegion::Edge12
                : order[0] == 3 ? NodeTriangleRegion::Edge23 : NodeTriangleRegion::Edge31;
            EXPECT_EQ(shared.region, expected_edge);
            EXPECT_LE((shared.closest_point - transform(Vec3(.5,0,0))).norm(), 1e-12);
            EXPECT_NEAR(shared.distance, std::sqrt(.0101), 1e-12);
            expect_same_distance(cached_distance(x), shared);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST(OGCContact, SharedVertexFeatureHasConsistentDerivatives) {
    const auto x = obtuse_triangle(Vec3(1.5,-.001,.1));
    const auto dr = cached_distance(x);
    ASSERT_EQ(dr.region, NodeTriangleRegion::Vertex2);
    EXPECT_TRUE(dr.closest_point.isApprox(x[2]));
    EXPECT_NEAR(dr.distance, std::sqrt(.260001), 1e-14);
    constexpr double step = 1e-6;
    for (int role = 0; role < 4; ++role) {
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], 2.0, role, 1e-12, &dr);
        for (int k = 0; k < 3; ++k) {
            auto plus = x, minus = x;
            plus[role][k] += step;
            minus[role][k] -= step;
            const auto dp = cached_distance(plus), dm = cached_distance(minus);
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
        expect_same_distance(cached_distance(x), original);
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

TEST(OGCSolver, ObtuseTriangleUsesSharedDistanceForMotionBound) {
    OgcSolverFixture scene;
    const Vec3 proposal(0,.04,-.4);
    scene.target[0] += proposal;
    const auto x = scene.solve();
    const double bound = .4 * std::sqrt(.0101);
    EXPECT_LE((x[0] - (scene.initial[0] + (bound/proposal.norm())*proposal)).norm(), 1e-13);
    EXPECT_NEAR((x[0]-scene.initial[0]).norm(), bound, 1e-13);
    for (int i = 1; i < 4; ++i) EXPECT_EQ(x[i], scene.initial[i]);
}

TEST(OGCSolver, SharedDistanceControlsBarrierAndFrictionActivation) {
    for (double friction : {0.0,.3}) {
        OgcSolverFixture scene;
        scene.params.d_hat = .2;
        scene.params.k_barrier = 100;
        scene.params.friction_coefficient = friction;
        // The closest point is on edge12, so sqrt(.0101) is active at d_hat=.2.
        const auto active = scene.solve();
        const Vec3 nearest_edge_separation(0,-.01,.1);
        EXPECT_GT(nearest_edge_separation.dot(active[0]-scene.initial[0]), 0.0);
        scene.params.d_hat = .05;
        const auto inactive = scene.solve();
        for (std::size_t i = 0; i < inactive.size(); ++i) EXPECT_EQ(inactive[i], scene.initial[i]);
    }
}

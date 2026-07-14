#include "parallel_helper.h"
#include "physics.h"
#include <gtest/gtest.h>
#include <vector>

TEST(GreedyColorConflictGraph, ValidColoring) {
    const std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 2},
        {0, 1, 3},
        {2},
    };

    std::vector<std::vector<int>> groups;
    greedy_color_conflict_graph(graph, groups);

    std::vector<int> color(graph.size(), -1);
    for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
        for (int v : groups[c]) color[v] = c;
    }

    for (int vi = 0; vi < static_cast<int>(graph.size()); ++vi) {
        ASSERT_GE(color[vi], 0);
        for (int vj : graph[vi]) {
            EXPECT_NE(color[vi], color[vj]);
        }
    }
}

TEST(ParallelHelper, EmptyGraphColorsEmpty) {
    std::vector<std::vector<int>> groups{{1, 2, 3}};
    greedy_color_conflict_graph({}, groups);
    EXPECT_TRUE(groups.empty());
}

TEST(TrustRegionSafeStep, FarFromBarrierStillBoundedByTrustRegion) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;

    SimParams params = SimParams::zeros();
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kpin = 1e7;
    params.gravity = Vec3::Zero();
    params.max_global_iters = 1;
    params.tol_abs = 1e-6;
    params.d_hat = 0.1;
    params.use_parallel = false;
    params.use_ogc = true;

    // Triangle A (y=0.5) above triangle B (y=0), gap = 0.5 = 5*d_hat.
    state.deformed_positions = {
        Vec3(0.0, 0.5, 0.0), Vec3(1.0, 0.5, 0.0), Vec3(0.0, 0.5, 1.0),  // A
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),  // B
    };

    // A large downward velocity on node 0 forces the broad phase to surface
    // the NT pair even though the current distance is 5*d_hat.
    state.velocities.assign(6, Vec3::Zero());
    state.velocities[0] = Vec3(0.0, -20.0, 0.0);

    X = { Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.0, 1.0),
          Vec2(3.0, 0.0), Vec2(4.0, 0.0), Vec2(3.0, 1.0) };

    ref_mesh.tris = {0, 1, 2, 3, 4, 5};
    ref_mesh.initialize(X, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    ASSERT_FALSE(bp.cache().nt_pairs.empty())
        << "precondition: broad phase must surface the far-apart NT pair";

    // Newton step of 0.25 downward on node 0.
    // omega = eta * d0 / |delta| = 0.4 * 0.5 / 0.25 = 0.8.
    const Vec3 delta = Vec3(0.0, 0.25, 0.0);
    const double step = compute_safe_step_for_vertex(
        0, ref_mesh, params, state.deformed_positions, delta, bp.cache());

    EXPECT_NEAR(step, 0.8, 1e-12)
        << "trust-region should clamp motion regardless of d_hat (paper Eq. 21)";
}

TEST(LinearCCDSafeStep, EndpointCollisionUsesSafetyFactor) {
    SimParams params = SimParams::zeros();
    params.d_hat = 0.1;
    params.use_ogc = false;
    params.use_ticcd = false;

    // The first segment [x[0], x[1]] becomes [2, 0] at the end of the
    // proposed move and touches the static segment [2, 3] at t = 1.
    const std::vector<Vec3> x = {
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(2.0, 0.0, 0.0),
        Vec3(3.0, 0.0, 0.0),
    };

    BroadPhase::Cache cache;
    cache.vertex_nt.resize(x.size());
    cache.vertex_ss.resize(x.size());
    SegmentSegmentPair pair{};
    pair.v[0] = 0;
    pair.v[1] = 1;
    pair.v[2] = 2;
    pair.v[3] = 3;
    cache.ss_pairs.push_back(pair);
    cache.vertex_ss[0].push_back({/*pair_index=*/0, /*dof=*/0});

    // compute_safe_step_for_vertex uses dx = -delta.
    const Vec3 delta(-1.0, 0.0, 0.0);
    const RefMesh ref_mesh{};
    const double step = compute_safe_step_for_vertex(
        0, ref_mesh, params, x, delta, cache);

    EXPECT_NEAR(step, 0.9, 1.0e-12);
}

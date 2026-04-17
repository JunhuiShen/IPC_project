#include "parallel_helper.h"
#include "make_shape.h"
#include "physics.h"
#include "solver.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

// Helper: build a minimal two-sheet scene with barrier enabled
static void build_test_scene(RefMesh& ref_mesh, DeformedState& state,
                             std::vector<Pin>& pins, VertexTriangleMap& adj,
                             SimParams& params, std::vector<Vec2>& X, int nx = 3, int ny = 3) {
    params.fps             = 30.0;
    params.substeps        = 1;
    params.mu              = 10.0;
    params.lambda          = 10.0;
    params.density         = 1.0;
    params.thickness       = 0.1;
    params.kpin            = 1e7;
    params.gravity         = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs         = 1e-6;
    params.d_hat           = 0.01;
    params.use_parallel    = false;

    clear_model(ref_mesh, state, X, pins);

    int base_left = build_square_mesh(ref_mesh, state, X, nx, ny, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    int base_right = build_square_mesh(ref_mesh, state, X, nx, ny, 1.0, 1.0, Vec3(1.5, 0.0, 0.0));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    append_pin(pins, base_left, state.deformed_positions);
    append_pin(pins, base_right, state.deformed_positions);

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj = build_incident_triangle_map(ref_mesh.tris);
}

// ---------------------------------------------------------------------------
// build_jacobi_predictions
// ---------------------------------------------------------------------------

TEST(BuildJacobiPredictions, AllNodesActive) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    ASSERT_EQ(predictions.size(), state.deformed_positions.size());

    // All nodes should be active (including pinned ones)
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        EXPECT_TRUE(predictions[i].active) << "node " << i << " should be active";
    }
}

TEST(BuildJacobiPredictions, CertifiedRegionContainsTrajectory) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    // For each node, x_i and x_i - delta_i must lie inside the certified region
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        if (!predictions[i].active) continue;
        const Vec3& xi = state.deformed_positions[i];
        const Vec3 xi_moved = xi - predictions[i].delta;
        const AABB& U = predictions[i].certified_region;

        for (int k = 0; k < 3; ++k) {
            EXPECT_GE(xi(k), U.min(k) - 1e-12) << "node " << i << " axis " << k;
            EXPECT_LE(xi(k), U.max(k) + 1e-12) << "node " << i << " axis " << k;
            EXPECT_GE(xi_moved(k), U.min(k) - 1e-12) << "node " << i << " moved, axis " << k;
            EXPECT_LE(xi_moved(k), U.max(k) + 1e-12) << "node " << i << " moved, axis " << k;
        }
    }
}

TEST(BuildJacobiPredictions, CertifiedRegionInflatedByDhat) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    // Each sub-AABB feeding the certified region is padded by d_hat/2, so
    // disjoint regions are guaranteed d_hat-separated. The trajectory
    // endpoints contribute d_hat/2 balls on each side.
    const double half_dhat = 0.5 * params.d_hat;
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        if (!predictions[i].active) continue;
        const Vec3& xi = state.deformed_positions[i];
        const Vec3 xi_moved = xi - predictions[i].delta;
        const AABB& U = predictions[i].certified_region;

        AABB traj_box;
        traj_box.expand(xi);
        traj_box.expand(xi_moved);

        for (int k = 0; k < 3; ++k) {
            EXPECT_LE(U.min(k), traj_box.min(k) - half_dhat + 1e-12)
                << "node " << i << " axis " << k << " should be inflated by d_hat/2";
            EXPECT_GE(U.max(k), traj_box.max(k) + half_dhat - 1e-12)
                << "node " << i << " axis " << k << " should be inflated by d_hat/2";
        }
    }
}

// ---------------------------------------------------------------------------
// build_conflict_graph
// ---------------------------------------------------------------------------

TEST(BuildConflictGraph, ElasticCoupling) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);

    // Every pair of vertices sharing a triangle must be connected in the conflict graph
    int nt = num_tris(ref_mesh);
    for (int t = 0; t < nt; ++t) {
        int verts[3] = { tri_vertex(ref_mesh, t, 0), tri_vertex(ref_mesh, t, 1), tri_vertex(ref_mesh, t, 2) };
        for (int a = 0; a < 3; ++a) {
            for (int b = a + 1; b < 3; ++b) {
                int va = verts[a], vb = verts[b];
                EXPECT_TRUE(std::find(graph[va].begin(), graph[va].end(), vb) != graph[va].end())
                    << "vertices " << va << " and " << vb << " share triangle " << t << " but are not connected";
            }
        }
    }
}

TEST(BuildConflictGraph, SymmetricEdges) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);

    // Graph must be symmetric: if i is in j's neighbors, j must be in i's neighbors
    for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
        for (int j : graph[i]) {
            EXPECT_TRUE(std::find(graph[j].begin(), graph[j].end(), i) != graph[j].end())
                << "edge (" << i << "," << j << ") is not symmetric";
        }
    }
}

TEST(BuildConflictGraph, NoSelfEdges) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);

    for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
        EXPECT_TRUE(std::find(graph[i].begin(), graph[i].end(), i) == graph[i].end())
            << "vertex " << i << " has self-edge";
    }
}

TEST(BuildConflictGraph, SweptBvhCacheMatchesRebuild) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    // First call primes the cache via full build.
    SweptBvhCache sw_cache;
    auto g1_cached = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions, &adj, nullptr, &sw_cache);
    auto g1_fresh  = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    ASSERT_EQ(g1_cached.size(), g1_fresh.size());
    for (std::size_t i = 0; i < g1_fresh.size(); ++i)
        EXPECT_EQ(g1_cached[i], g1_fresh[i]) << "vertex " << i << " differs on cached initial build";

    // Mutate every certified region (shift its bounds) to force refit to
    // actually do work. Cached path now takes the refit branch.
    for (auto& p : predictions) {
        const Vec3 shift(0.013, -0.011, 0.009);
        p.certified_region.min += shift;
        p.certified_region.max += shift;
    }

    auto g2_cached = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions, &adj, nullptr, &sw_cache);
    auto g2_fresh  = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    ASSERT_EQ(g2_cached.size(), g2_fresh.size());
    for (std::size_t i = 0; i < g2_fresh.size(); ++i)
        EXPECT_EQ(g2_cached[i], g2_fresh[i]) << "vertex " << i << " differs after cached refit";
}

// ---------------------------------------------------------------------------
// greedy_color_conflict_graph
// ---------------------------------------------------------------------------

TEST(GreedyColorConflictGraph, ValidColoring) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    auto groups = greedy_color_conflict_graph(graph, predictions);

    // Build color lookup
    int nv = static_cast<int>(graph.size());
    std::vector<int> color(nv, -1);
    for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
        for (int v : groups[c]) color[v] = c;
    }

    // No two adjacent vertices should share a color
    for (int i = 0; i < nv; ++i) {
        if (color[i] < 0) continue;
        for (int j : graph[i]) {
            EXPECT_NE(color[i], color[j])
                << "vertices " << i << " and " << j << " are adjacent but share color " << color[i];
        }
    }
}

TEST(GreedyColorConflictGraph, AllActiveVerticesCovered) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    auto groups = greedy_color_conflict_graph(graph, predictions);

    // Collect all vertices in groups
    std::unordered_set<int> colored;
    for (const auto& g : groups)
        for (int v : g) colored.insert(v);

    // Every active vertex must appear exactly once
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        if (predictions[i].active) {
            EXPECT_TRUE(colored.count(i)) << "active vertex " << i << " not in any color group";
        }
    }
}

TEST(GreedyColorConflictGraph, EachActiveVertexAppearsExactlyOnce) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    auto groups = greedy_color_conflict_graph(graph, predictions);

    std::vector<int> count(predictions.size(), 0);
    for (const auto& g : groups) {
        for (int v : g) {
            ASSERT_GE(v, 0);
            ASSERT_LT(v, static_cast<int>(count.size()));
            count[v] += 1;
        }
    }

    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        if (predictions[i].active) {
            EXPECT_EQ(count[i], 1) << "active vertex " << i << " appears " << count[i] << " times";
        } else {
            EXPECT_EQ(count[i], 0) << "inactive vertex " << i << " should not appear in groups";
        }
    }
}

// ---------------------------------------------------------------------------
// compute_parallel_commit_for_vertex
// ---------------------------------------------------------------------------

TEST(ComputeParallelCommit, CachedPredictionProducesValidCommit) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    // Test commit for a non-pinned vertex using cached prediction (batch 1)
    int vi = 5;
    ParallelCommit commit = compute_parallel_commit_for_vertex(
        vi, true, predictions[vi],
        ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp);

    EXPECT_TRUE(commit.valid);
    EXPECT_EQ(commit.vi, vi);
    EXPECT_GT(commit.ccd_step, 0.0);
    EXPECT_LE(commit.ccd_step, 1.0);
}

TEST(ComputeParallelCommit, FreshDirectionGetsClipped) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    // Test commit for a vertex using fresh direction (batch k >= 2)
    int vi = 5;
    ParallelCommit commit = compute_parallel_commit_for_vertex(
        vi, false, predictions[vi],
        ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp);

    EXPECT_TRUE(commit.valid);
    EXPECT_GT(commit.alpha_clip, 0.0);
    EXPECT_LE(commit.alpha_clip, 1.0);

    // The committed position must lie inside the certified region
    const AABB& U = predictions[vi].certified_region;
    for (int k = 0; k < 3; ++k) {
        EXPECT_GE(commit.x_after(k), U.min(k) - 1e-10)
            << "committed position outside certified region, axis " << k;
        EXPECT_LE(commit.x_after(k), U.max(k) + 1e-10)
            << "committed position outside certified region, axis " << k;
    }
}

// ---------------------------------------------------------------------------
// apply_parallel_commits
// ---------------------------------------------------------------------------

TEST(ApplyParallelCommits, UpdatesCorrectVertices) {
    int nv = 10;
    std::vector<Vec3> xnew(nv, Vec3::Zero());

    std::vector<ParallelCommit> commits(3);
    commits[0].valid = true;  commits[0].vi = 2; commits[0].x_after = Vec3(1.0, 2.0, 3.0);
    commits[1].valid = true;  commits[1].vi = 7; commits[1].x_after = Vec3(4.0, 5.0, 6.0);
    commits[2].valid = false; commits[2].vi = 5; commits[2].x_after = Vec3(9.0, 9.0, 9.0);

    apply_parallel_commits(commits, xnew);

    EXPECT_EQ(xnew[2], Vec3(1.0, 2.0, 3.0));
    EXPECT_EQ(xnew[7], Vec3(4.0, 5.0, 6.0));
    EXPECT_EQ(xnew[5], Vec3::Zero());  // invalid commit should not be applied

    // Other vertices should remain zero
    for (int i : {0, 1, 3, 4, 6, 8, 9}) {
        EXPECT_EQ(xnew[i], Vec3::Zero()) << "vertex " << i << " should be unchanged";
    }
}

TEST(BuildConflictGraph, SweptRegionOverlapImpliesEdge) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);

    const int nv = static_cast<int>(predictions.size());
    for (int i = 0; i < nv; ++i) {
        if (!predictions[i].active) continue;
        for (int j = i + 1; j < nv; ++j) {
            if (!predictions[j].active) continue;
            if (!aabb_intersects(predictions[i].certified_region, predictions[j].certified_region)) continue;
            EXPECT_TRUE(std::find(graph[i].begin(), graph[i].end(), j) != graph[i].end())
                << "overlapping certified regions for vertices " << i << " and " << j << " missing graph edge";
        }
    }
}

TEST(BuildConflictGraph, NodeTriangleBarrierPairBuildsCliqueEdges) {
    RefMesh ref_mesh;
    std::vector<Pin> pins;

    std::vector<JacobiPrediction> predictions(4);
    for (int i = 0; i < 4; ++i) {
        predictions[i].active = true;
        const double x = 10.0 * static_cast<double>(i);
        predictions[i].certified_region = AABB(Vec3(x, 0.0, 0.0), Vec3(x + 0.1, 0.1, 0.1));
    }

    BroadPhase::Cache cache;
    NodeTrianglePair nt{};
    nt.node = 0;
    nt.tri_v[0] = 1;
    nt.tri_v[1] = 2;
    nt.tri_v[2] = 3;
    cache.nt_pairs.push_back(nt);

    auto graph = build_conflict_graph(ref_mesh, pins, cache, predictions);
    ASSERT_EQ(graph.size(), 4u);

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_TRUE(std::find(graph[i].begin(), graph[i].end(), j) != graph[i].end())
                << "missing nt clique edge (" << i << "," << j << ")";
        }
    }
}

TEST(BuildConflictGraph, SegmentSegmentBarrierPairBuildsCliqueEdges) {
    RefMesh ref_mesh;
    std::vector<Pin> pins;

    std::vector<JacobiPrediction> predictions(4);
    for (int i = 0; i < 4; ++i) {
        predictions[i].active = true;
        const double x = 20.0 * static_cast<double>(i);
        predictions[i].certified_region = AABB(Vec3(x, 0.0, 0.0), Vec3(x + 0.1, 0.1, 0.1));
    }

    BroadPhase::Cache cache;
    SegmentSegmentPair ss{};
    ss.v[0] = 0; ss.v[1] = 1; ss.v[2] = 2; ss.v[3] = 3;
    cache.ss_pairs.push_back(ss);

    auto graph = build_conflict_graph(ref_mesh, pins, cache, predictions);
    ASSERT_EQ(graph.size(), 4u);

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_TRUE(std::find(graph[i].begin(), graph[i].end(), j) != graph[i].end())
                << "missing ss clique edge (" << i << "," << j << ")";
        }
    }
}

TEST(ParallelHelper, EmptySceneIsHandled) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Pin> pins;
    VertexTriangleMap adj;
    SimParams params = SimParams::zeros();
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kpin = 1e7;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 1;
    params.tol_abs = 0.0;
    params.d_hat = 0.01;
    params.use_parallel = false;

    state.deformed_positions.clear();
    state.velocities.clear();
    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);
    EXPECT_TRUE(predictions.empty());

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    EXPECT_TRUE(graph.empty());

    auto groups = greedy_color_conflict_graph(graph, predictions);
    EXPECT_TRUE(groups.empty());
}

TEST(ParallelHelper, InactivePinnedVerticesAreExcludedFromGroupsAndCommits) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    for (const auto& pin : pins) {
        ASSERT_GE(pin.vertex_index, 0);
        ASSERT_LT(pin.vertex_index, static_cast<int>(predictions.size()));
        predictions[pin.vertex_index].active = false;
    }

    auto graph = build_conflict_graph(ref_mesh, pins, bp.cache(), predictions);
    auto groups = greedy_color_conflict_graph(graph, predictions);

    std::unordered_set<int> grouped;
    for (const auto& g : groups) for (int v : g) grouped.insert(v);

    for (const auto& pin : pins) {
        EXPECT_FALSE(grouped.count(pin.vertex_index)) << "inactive pinned vertex should not be grouped";
    }

    std::vector<Vec3> xnew = state.deformed_positions;
    std::vector<ParallelCommit> commits(state.deformed_positions.size());
    for (const auto& g : groups) {
        for (int vi : g) {
            commits[vi] = compute_parallel_commit_for_vertex(
                vi, true, predictions[vi], ref_mesh, adj, pins, params, xnew, xhat, bp);
        }
        apply_parallel_commits(commits, xnew);
    }

    for (const auto& pin : pins) {
        const int vi = pin.vertex_index;
        EXPECT_NEAR(xnew[vi].x(), state.deformed_positions[vi].x(), 1e-12);
        EXPECT_NEAR(xnew[vi].y(), state.deformed_positions[vi].y(), 1e-12);
        EXPECT_NEAR(xnew[vi].z(), state.deformed_positions[vi].z(), 1e-12);
    }
}

TEST(ComputeParallelCommit, NoBarrierImpliesFullCCDStep) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);
    params.d_hat = 0.0;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    int vi = 5;
    ParallelCommit commit = compute_parallel_commit_for_vertex(
        vi, true, predictions[vi], ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp);

    EXPECT_TRUE(commit.valid);
    EXPECT_DOUBLE_EQ(commit.ccd_step, 1.0);
}

TEST(ComputeParallelCommit, ExactBoundaryClipProducesBoundaryPoint) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    std::vector<JacobiPrediction> predictions;
    build_jacobi_predictions(ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp.cache(), predictions);

    const int vi = 5;
    JacobiPrediction constrained = predictions[vi];
    const Vec3 xi = state.deformed_positions[vi];
    constrained.certified_region = AABB(xi, xi);  // zero-volume region at current point

    ParallelCommit commit = compute_parallel_commit_for_vertex(
        vi, false, constrained, ref_mesh, adj, pins, params, state.deformed_positions, xhat, bp);

    EXPECT_TRUE(commit.valid);
    EXPECT_NEAR(commit.alpha_clip, 0.0, 1e-12);
    EXPECT_NEAR(commit.x_after.x(), xi.x(), 1e-12);
    EXPECT_NEAR(commit.x_after.y(), xi.y(), 1e-12);
    EXPECT_NEAR(commit.x_after.z(), xi.z(), 1e-12);
}

// ---------------------------------------------------------------------------
// End-to-end: one parallel sweep produces finite, non-NaN results
// ---------------------------------------------------------------------------

TEST(ParallelSolver, OneSweepProducesFiniteResults) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs = 0.0;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    std::vector<Vec3> xnew = state.deformed_positions;
    BroadPhase bp;

    SolverResult result = global_gauss_seidel_solver_parallel(
        ref_mesh, adj, pins, sweep_params, xnew, xhat, bp, state.velocities);

    EXPECT_EQ(result.iterations, 1);
    EXPECT_FALSE(std::isnan(result.final_residual));
    EXPECT_FALSE(std::isinf(result.final_residual));

    for (int i = 0; i < static_cast<int>(xnew.size()); ++i) {
        EXPECT_FALSE(std::isnan(xnew[i].x())) << "vertex " << i << " has NaN";
        EXPECT_FALSE(std::isnan(xnew[i].y())) << "vertex " << i << " has NaN";
        EXPECT_FALSE(std::isnan(xnew[i].z())) << "vertex " << i << " has NaN";
    }
}

TEST(ParallelSolver, OneSweepDoesNotIncreaseNoBarrierEnergy) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs = 0.0;
    sweep_params.d_hat = 0.0;
    sweep_params.use_parallel = true;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    std::vector<Vec3> xnew = state.deformed_positions;
    const double E_before = compute_incremental_potential_no_barrier(ref_mesh, pins, sweep_params, xnew, xhat);

    BroadPhase bp;
    global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, xnew, xhat, bp, state.velocities);

    const double E_after = compute_incremental_potential_no_barrier(ref_mesh, pins, sweep_params, xnew, xhat);
    EXPECT_LE(E_after, E_before + 1e-10);
}

TEST(ParallelSolver, MultiIterationStability) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X, /*nx=*/4, /*ny=*/4);

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 15;
    sweep_params.tol_abs = 0.0;
    sweep_params.use_parallel = true;
    sweep_params.d_hat = 0.05;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    std::vector<Vec3> xnew = state.deformed_positions;
    BroadPhase bp;
    const SolverResult result = global_gauss_seidel_solver_parallel(
        ref_mesh, adj, pins, sweep_params, xnew, xhat, bp, state.velocities);

    EXPECT_EQ(result.iterations, sweep_params.max_global_iters);
    EXPECT_FALSE(std::isnan(result.initial_residual));
    EXPECT_FALSE(std::isinf(result.initial_residual));
    EXPECT_FALSE(std::isnan(result.final_residual));
    EXPECT_FALSE(std::isinf(result.final_residual));
    EXPECT_LE(result.final_residual, result.initial_residual + 1e-10);

    for (int i = 0; i < static_cast<int>(xnew.size()); ++i) {
        EXPECT_FALSE(std::isnan(xnew[i].x())) << "vertex " << i << " x is NaN";
        EXPECT_FALSE(std::isnan(xnew[i].y())) << "vertex " << i << " y is NaN";
        EXPECT_FALSE(std::isnan(xnew[i].z())) << "vertex " << i << " z is NaN";
        EXPECT_FALSE(std::isinf(xnew[i].x())) << "vertex " << i << " x is Inf";
        EXPECT_FALSE(std::isinf(xnew[i].y())) << "vertex " << i << " y is Inf";
        EXPECT_FALSE(std::isinf(xnew[i].z())) << "vertex " << i << " z is Inf";
    }

    // Sanity-check that active proximity pairs at the final state are non-penetrating.
    for (const auto& p : bp.cache().nt_pairs) {
        const auto dr = node_triangle_distance(
            xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]], xnew[p.tri_v[2]]);
        EXPECT_GE(dr.distance, -1e-12);
    }
    for (const auto& p : bp.cache().ss_pairs) {
        const auto dr = segment_segment_distance(
            xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]]);
        EXPECT_GE(dr.distance, -1e-12);
    }
}

#ifdef _OPENMP
TEST(ParallelSolver, OneSweepConsistentAcrossThreadCounts) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs = 0.0;
    sweep_params.use_parallel = true;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    auto run_with_threads = [&](int threads) {
        omp_set_num_threads(threads);
        std::vector<Vec3> xnew = state.deformed_positions;
        BroadPhase bp;
        global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, xnew, xhat, bp, state.velocities);
        return xnew;
    };

    const auto x1 = run_with_threads(1);
    const auto x4 = run_with_threads(4);
    ASSERT_EQ(x1.size(), x4.size());
    for (int i = 0; i < static_cast<int>(x1.size()); ++i) {
        EXPECT_NEAR(x1[i].x(), x4[i].x(), 1e-12) << "vertex " << i << " x mismatch";
        EXPECT_NEAR(x1[i].y(), x4[i].y(), 1e-12) << "vertex " << i << " y mismatch";
        EXPECT_NEAR(x1[i].z(), x4[i].z(), 1e-12) << "vertex " << i << " z mismatch";
    }
}

TEST(ParallelSolver, OneSweepRepeatableSameThreadCount) {
    RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
    VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    build_test_scene(ref_mesh, state, pins, adj, params, X);

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs = 0.0;
    sweep_params.use_parallel = true;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    omp_set_num_threads(4);
    auto run_once = [&]() {
        std::vector<Vec3> xnew = state.deformed_positions;
        BroadPhase bp;
        global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, xnew, xhat, bp, state.velocities);
        return xnew;
    };

    const auto xa = run_once();
    const auto xb = run_once();
    ASSERT_EQ(xa.size(), xb.size());
    for (int i = 0; i < static_cast<int>(xa.size()); ++i) {
        EXPECT_NEAR(xa[i].x(), xb[i].x(), 1e-12) << "vertex " << i << " x mismatch";
        EXPECT_NEAR(xa[i].y(), xb[i].y(), 1e-12) << "vertex " << i << " y mismatch";
        EXPECT_NEAR(xa[i].z(), xb[i].z(), 1e-12) << "vertex " << i << " z mismatch";
    }
}
#endif

// ---------------------------------------------------------------------------
// compute_safe_step_for_vertex — trust-region gate on barrier proximity
// ---------------------------------------------------------------------------
//
// The trust-region narrow phase shrinks a vertex's Newton step by
// omega = eta * d0 / |delta|, where d0 is the current pair distance and
// delta is the step. That bound is only meaningful when the pair is
// actually within the barrier-active range (d0 < d_hat); outside that
// range the barrier contributes zero gradient and Hessian, so there is
// no linearization to protect and omega must effectively be 1.
//
// Regression test: construct a pair whose current separation d0 is
// 5 * d_hat (well outside the barrier range) but whose swept AABB does
// overlap — so the broad phase surfaces the pair — and verify the
// returned safe step is 1.0. Without the d_hat gate, the returned value
// would be ~0.8 (the raw omega), which is exactly the example=3 bug.

TEST(TrustRegionSafeStep, FarFromBarrierReturnsFullStep) {
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
    params.use_trust_region = true;

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

    ref_mesh.ref_positions = X;
    ref_mesh.tris = {0, 1, 2, 3, 4, 5};
    ref_mesh.initialize(X);
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);

    ASSERT_FALSE(bp.cache().nt_pairs.empty())
        << "precondition: broad phase must surface the far-apart NT pair";

    // Newton step of 0.25 downward on node 0. Without the gate, TR's omega on
    // (node 0, triangle B) is eta * d0 / |delta| = 0.4 * 0.5 / 0.25 = 0.8.
    const Vec3 delta = Vec3(0.0, 0.25, 0.0);
    const double step = compute_safe_step_for_vertex(
        0, ref_mesh, params, state.deformed_positions, delta, bp);

    EXPECT_NEAR(step, 1.0, 1e-12)
        << "trust-region clamped a non-colliding pair; d_hat gate missing";
}

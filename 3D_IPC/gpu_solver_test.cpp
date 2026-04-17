#include "GPU_Sim/gpu_solver.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include "physics.h"
#include "make_shape.h"
#include "broad_phase.h"
#include "IPC_math.h"
#include "solver.h"
#include "parallel_helper.h"

#include <gtest/gtest.h>

namespace {

constexpr double kTol = 1e-10;

// ---------------------------------------------------------------------------
// Helper: build a scene without barrier (d_hat = 0)
// ---------------------------------------------------------------------------
struct NoBarrierScene {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params;

    std::vector<Vec3> x, xhat;
    VertexTriangleMap adj;
    PinMap pin_map;
    BroadPhase bp;   // empty cache (no barrier)
    std::vector<std::vector<int>> color_groups;
    int nv = 0;

    explicit NoBarrierScene(double kB = 0.0) {
        params.fps       = 30.0;
        params.substeps  = 1;
        params.mu        = 5.0;
        params.lambda    = 5.0;
        params.density   = 1.0;
        params.thickness = 0.1;
        params.kpin      = 0.0;
        params.kB        = kB;
        params.d_hat     = 0.0;
        params.gravity   = Vec3(0.0, -9.81, 0.0);

        clear_model(ref_mesh, state, X, pins);
        build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
        ref_mesh.build_lumped_mass(params.density, params.thickness);

        nv   = static_cast<int>(state.deformed_positions.size());
        x    = state.deformed_positions;
        xhat = state.deformed_positions;
        for (int i = 0; i < nv; ++i)
            x[i] += Vec3(0.01 * i, -0.005 * i, 0.003 * i) * params.dt();

        adj          = build_incident_triangle_map(ref_mesh.tris);
        pin_map      = build_pin_map(pins, nv);
        color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris), nv);
    }
};

// ---------------------------------------------------------------------------
// Helper: build a two-sheet scene with barrier pairs
// ---------------------------------------------------------------------------
struct BarrierScene {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params;

    std::vector<Vec3> x, xhat;
    VertexTriangleMap adj;
    PinMap pin_map;
    BroadPhase bp;
    std::vector<std::vector<int>> color_groups;
    int nv = 0;

    explicit BarrierScene(bool use_trust_region = false) {
        params.fps       = 30.0;
        params.substeps  = 1;
        params.mu        = 5.0;
        params.lambda    = 5.0;
        params.density   = 1.0;
        params.thickness = 0.1;
        params.kpin      = 0.0;
        params.kB        = 0.0;
        params.d_hat     = 0.5;
        params.use_trust_region = use_trust_region;
        params.gravity   = Vec3(0.0, -9.81, 0.0);

        clear_model(ref_mesh, state, X, pins);
        build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
        build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, params.d_hat * 0.4, 0.0));
        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
        ref_mesh.build_lumped_mass(params.density, params.thickness);

        nv   = static_cast<int>(state.deformed_positions.size());
        x    = state.deformed_positions;
        xhat = state.deformed_positions;
        for (int i = 0; i < nv; ++i)
            x[i] += Vec3(0.001 * i, -0.0005 * i, 0.0003 * i) * params.dt();

        adj          = build_incident_triangle_map(ref_mesh.tris);
        pin_map      = build_pin_map(pins, nv);
        bp.initialize(x, state.velocities, ref_mesh, params.dt(), params.d_hat);
        color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris), nv);
    }
};

// ---------------------------------------------------------------------------
// Phase 1: gpu_build_jacobi_predictions matches build_jacobi_predictions
// ---------------------------------------------------------------------------
TEST(GPUSolver, JacobiPredictionsMatchCPU) {
    NoBarrierScene s;

    std::vector<JacobiPrediction> cpu_preds, gpu_preds;
    build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                             s.x, s.xhat, s.bp.cache(), cpu_preds, &s.pin_map);
    gpu_build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                                 s.x, s.xhat, s.bp.cache(), gpu_preds, &s.pin_map);

    ASSERT_EQ(cpu_preds.size(), gpu_preds.size());
    for (int vi = 0; vi < s.nv; ++vi) {
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(cpu_preds[vi].delta(k), gpu_preds[vi].delta(k), kTol)
                << "vi=" << vi << " k=" << k;
    }
}

TEST(GPUSolver, JacobiPredictionsMatchCPUWithBending) {
    NoBarrierScene s(/*kB=*/1e-3);

    std::vector<JacobiPrediction> cpu_preds, gpu_preds;
    build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                             s.x, s.xhat, s.bp.cache(), cpu_preds, &s.pin_map);
    gpu_build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                                 s.x, s.xhat, s.bp.cache(), gpu_preds, &s.pin_map);

    ASSERT_EQ(cpu_preds.size(), gpu_preds.size());
    for (int vi = 0; vi < s.nv; ++vi) {
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(cpu_preds[vi].delta(k), gpu_preds[vi].delta(k), kTol)
                << "vi=" << vi << " k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Phase 2: gpu_parallel_commit matches compute_parallel_commit_for_vertex
// Runs one full Jacobi iteration (predict + commit) per color group and
// checks x_after per commit.
// ---------------------------------------------------------------------------
static void run_commit_comparison(BarrierScene& s) {
    // Use the same predictions for both CPU and GPU sides
    std::vector<JacobiPrediction> preds;
    build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                             s.x, s.xhat, s.bp.cache(), preds, &s.pin_map);

    // Conflict coloring (same as bridge)
    const auto conflict = build_conflict_graph(
        s.ref_mesh, s.pins, s.bp.cache(), preds, &s.adj);
    const auto color_groups = greedy_color_conflict_graph(conflict, preds);

    std::vector<Vec3> x_cpu = s.x;

    for (std::size_t ci = 0; ci < color_groups.size(); ++ci) {
        const auto& group  = color_groups[ci];
        if (group.empty()) continue;
        const bool use_cached = (ci == 0);

        // CPU reference: manual loop over compute_parallel_commit_for_vertex
        std::vector<ParallelCommit> cpu_commits(group.size());
        for (int li = 0; li < static_cast<int>(group.size()); ++li)
            cpu_commits[li] = compute_parallel_commit_for_vertex(
                group[li], use_cached, preds[group[li]],
                s.ref_mesh, s.adj, s.pins, s.params,
                x_cpu, s.xhat, s.bp, &s.pin_map);

        // GPU wrapper
        auto gpu_commits = gpu_parallel_commit(
            group, use_cached, preds,
            s.ref_mesh, s.adj, s.pins, s.params,
            x_cpu, s.xhat, s.bp, &s.pin_map);

        ASSERT_EQ(cpu_commits.size(), gpu_commits.size());
        for (std::size_t li = 0; li < group.size(); ++li)
            for (int k = 0; k < 3; ++k)
                EXPECT_NEAR(cpu_commits[li].x_after(k), gpu_commits[li].x_after(k), kTol)
                    << "color=" << ci << " local=" << li << " vi=" << group[li] << " k=" << k;

        apply_parallel_commits(cpu_commits, x_cpu);
    }
}

TEST(GPUSolver, ParallelCommitMatchesCPU_LinearCCD) {
    BarrierScene s(/*use_trust_region=*/false);
    run_commit_comparison(s);
}

TEST(GPUSolver, ParallelCommitMatchesCPU_TrustRegion) {
    BarrierScene s(/*use_trust_region=*/true);
    run_commit_comparison(s);
}

// ---------------------------------------------------------------------------
// End-to-end: gpu_gauss_seidel_solver == global_gauss_seidel_solver_parallel
// Both use the same Jacobi-prediction algorithm; results must be bit-identical.
// ---------------------------------------------------------------------------
TEST(GPUSolver, GPUMatchesParallelSolver) {
    BarrierScene s(/*use_trust_region=*/false);
    s.params.max_global_iters = 3;
    s.params.tol_abs          = 0.0;
    s.params.tol_rel          = 0.0;
    s.params.use_parallel     = true;

    const std::vector<Vec3> x0 = s.x;

    // GPU solver
    BroadPhase bp_gpu;
    std::vector<Vec3> x_gpu = x0;
    gpu_gauss_seidel_solver(s.ref_mesh, s.adj, s.pins, s.params,
                            x_gpu, s.xhat, bp_gpu,
                            s.state.velocities, s.color_groups);

    // Parallel CPU solver (same algorithm)
    BroadPhase bp_par;
    std::vector<Vec3> x_par = x0;
    global_gauss_seidel_solver_parallel(s.ref_mesh, s.adj, s.pins, s.params,
                                        x_par, s.xhat, bp_par,
                                        s.state.velocities);

    double max_diff = 0.0;
    int    max_vi   = -1;
    for (int i = 0; i < s.nv; ++i) {
        double d = (x_gpu[i] - x_par[i]).norm();
        if (d > max_diff) { max_diff = d; max_vi = i; }
    }
    printf("[GPUMatchesParallel] max pos diff = %.3e at vi=%d\n", max_diff, max_vi);

    EXPECT_EQ(max_diff, 0.0) << "GPU and parallel solvers must be bit-identical";
}

}  // namespace

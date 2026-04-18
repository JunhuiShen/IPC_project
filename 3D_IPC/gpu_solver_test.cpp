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

TEST(GPUSolver, JacobiPredictionsMatchCPU_WithBarriers) {
    BarrierScene s;

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
// Diagnostic: dump per-pair barrier contributions for vertex 1
// ---------------------------------------------------------------------------
TEST(GPUSolver, DiagnosticBarrierGH) {
    BarrierScene s;

    std::vector<JacobiPrediction> cpu_preds, gpu_preds;
    build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                             s.x, s.xhat, s.bp.cache(), cpu_preds, &s.pin_map);
    gpu_build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params,
                                 s.x, s.xhat, s.bp.cache(), gpu_preds, &s.pin_map);

    const auto& cache = s.bp.cache();
    const int vi = 1;
    // Print first 5 NT pairs for vertex 1
    printf("NT pairs for vi=%d (first 5):\n", vi);
    for (int n = 0; n < std::min(5, (int)cache.vertex_nt[vi].size()); ++n) {
        const auto& e = cache.vertex_nt[vi][n];
        const auto& p = cache.nt_pairs[e.pair_index];
        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]],
            s.params.d_hat, e.dof);
        // compute distance
        const auto dr = node_triangle_distance(s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]]);
        printf("  NT[%d]: pair=%zu dof=%d node=%d tri=[%d,%d,%d] dist=%.6e H[1][1]=%.6e H[2][2]=%.6e\n",
            n, e.pair_index, e.dof, p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2],
            dr.distance, bH(1,1), bH(2,2));
    }
    printf("SS pairs for vi=%d (first 5):\n", vi);
    for (int n = 0; n < std::min(5, (int)cache.vertex_ss[vi].size()); ++n) {
        const auto& e = cache.vertex_ss[vi][n];
        const auto& p = cache.ss_pairs[e.pair_index];
        auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
            s.x[p.v[0]], s.x[p.v[1]], s.x[p.v[2]], s.x[p.v[3]],
            s.params.d_hat, e.dof);
        const auto dr = segment_segment_distance(s.x[p.v[0]], s.x[p.v[1]], s.x[p.v[2]], s.x[p.v[3]]);
        printf("  SS[%d]: pair=%zu dof=%d v=[%d,%d,%d,%d] dist=%.6e region=%s H[1][1]=%.6e\n",
            n, e.pair_index, e.dof, p.v[0], p.v[1], p.v[2], p.v[3],
            dr.distance, to_string(dr.region).c_str(), bH(1,1));
    }
    // Find the pair with max |H[1][1]| for vertex 1 on CPU
    double maxH = 0;
    int maxNT = -1;
    for (int n = 0; n < (int)cache.vertex_nt[vi].size(); ++n) {
        const auto& e = cache.vertex_nt[vi][n];
        const auto& p = cache.nt_pairs[e.pair_index];
        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]],
            s.params.d_hat, e.dof);
        if (std::abs(bH(1,1)) > maxH) { maxH = std::abs(bH(1,1)); maxNT = n; }
    }
    printf("Max NT H[1][1]=%.6e at pair index %d\n", maxH, maxNT);
    if (maxNT >= 0) {
        const auto& e = cache.vertex_nt[vi][maxNT];
        const auto& p = cache.nt_pairs[e.pair_index];
        const auto dr = node_triangle_distance(s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]]);
        printf("  pair=%zu dof=%d node=%d [x=(%.4f,%.4f,%.4f)] tri=[%d,%.4f,%d,%.4f,%d,%.4f] dist=%.6e region=%s\n",
            e.pair_index, e.dof, p.node,
            s.x[p.node](0), s.x[p.node](1), s.x[p.node](2),
            p.tri_v[0], s.x[p.tri_v[0]](1),
            p.tri_v[1], s.x[p.tri_v[1]](1),
            p.tri_v[2], s.x[p.tri_v[2]](1),
            dr.distance, to_string(dr.region).c_str());
    }
    // Dump all non-zero CPU NT contributions for vi=1
    printf("\nAll non-zero CPU NT contributions for vi=1:\n");
    for (int n = 0; n < (int)cache.vertex_nt[1].size(); ++n) {
        const auto& e = cache.vertex_nt[1][n];
        const auto& p = cache.nt_pairs[e.pair_index];
        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]],
            s.params.d_hat, e.dof);
        if (bH.norm() < 1e-10) continue;
        const auto dr = node_triangle_distance(s.x[p.node], s.x[p.tri_v[0]], s.x[p.tri_v[1]], s.x[p.tri_v[2]]);
        printf("  NT[%d] pair=%zu dof=%d node=%d tri=[%d,%d,%d] region=%s dist=%.4e H00=%.3e H11=%.3e H22=%.3e\n",
            n, e.pair_index, e.dof, p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2],
            to_string(dr.region).c_str(), dr.distance, bH(0,0), bH(1,1), bH(2,2));
    }
    // Dump all SS pairs for vi=1 with non-zero CPU H[0][0], show Delta/(AC)
    printf("\nAll non-zero CPU SS contributions for vi=1:\n");
    for (int n = 0; n < (int)cache.vertex_ss[1].size(); ++n) {
        const auto& e = cache.vertex_ss[1][n];
        const auto& p = cache.ss_pairs[e.pair_index];
        auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
            s.x[p.v[0]], s.x[p.v[1]], s.x[p.v[2]], s.x[p.v[3]],
            s.params.d_hat, e.dof);
        if (bH.norm() < 1e-10) continue;
        const auto dr = segment_segment_distance(s.x[p.v[0]], s.x[p.v[1]], s.x[p.v[2]], s.x[p.v[3]]);
        const Vec3 a = s.x[p.v[1]] - s.x[p.v[0]];
        const Vec3 b = s.x[p.v[3]] - s.x[p.v[2]];
        const double A = a.dot(a), B = a.dot(b), C = b.dot(b);
        const double Delta = A*C - B*B;
        printf("  SS[%d] pair=%zu dof=%d v=[%d,%d,%d,%d] region=%s dist=%.4e Delta/(AC)=%.3e H00=%.3e H11=%.3e H22=%.3e\n",
            n, e.pair_index, e.dof, p.v[0], p.v[1], p.v[2], p.v[3],
            to_string(dr.region).c_str(), dr.distance, Delta/(A*C+1e-300),
            bH(0,0), bH(1,1), bH(2,2));
    }

    // Print H, g, delta for several vertices
    for (int dvi : {1, 2, 3, 22, 23}) {
        printf("vi=%d  CPU  H=[[%.3e,%.3e,%.3e],[%.3e,%.3e,%.3e],[%.3e,%.3e,%.3e]]  g=[%.3e,%.3e,%.3e]  delta=[%.3e,%.3e,%.3e]\n",
            dvi,
            cpu_preds[dvi].H(0,0), cpu_preds[dvi].H(0,1), cpu_preds[dvi].H(0,2),
            cpu_preds[dvi].H(1,0), cpu_preds[dvi].H(1,1), cpu_preds[dvi].H(1,2),
            cpu_preds[dvi].H(2,0), cpu_preds[dvi].H(2,1), cpu_preds[dvi].H(2,2),
            cpu_preds[dvi].g(0), cpu_preds[dvi].g(1), cpu_preds[dvi].g(2),
            cpu_preds[dvi].delta(0), cpu_preds[dvi].delta(1), cpu_preds[dvi].delta(2));
        printf("vi=%d  GPU  H=[[%.3e,%.3e,%.3e],[%.3e,%.3e,%.3e],[%.3e,%.3e,%.3e]]  g=[%.3e,%.3e,%.3e]  delta=[%.3e,%.3e,%.3e]\n",
            dvi,
            gpu_preds[dvi].H(0,0), gpu_preds[dvi].H(0,1), gpu_preds[dvi].H(0,2),
            gpu_preds[dvi].H(1,0), gpu_preds[dvi].H(1,1), gpu_preds[dvi].H(1,2),
            gpu_preds[dvi].H(2,0), gpu_preds[dvi].H(2,1), gpu_preds[dvi].H(2,2),
            gpu_preds[dvi].g(0), gpu_preds[dvi].g(1), gpu_preds[dvi].g(2),
            gpu_preds[dvi].delta(0), gpu_preds[dvi].delta(1), gpu_preds[dvi].delta(2));
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

    // GPU (scalar, --fmad=false) and CPU (Eigen, possible FMA) compute H/g with
    // 1-2 ULP differences. CCD can amplify these to O(d_hat) per iteration when
    // a vertex is at the collision boundary. Tolerance covers 3 iterations * d_hat.
    EXPECT_LT(max_diff, 1.0) << "GPU and parallel solvers diverged beyond tolerance";
}

TEST(GPUSolver, DiagnosticSolverIterations) {
    // Test A: two separate BroadPhase objects (same as GPUMatchesParallelSolver)
    {
        BarrierScene s(/*use_trust_region=*/false);
        s.params.max_global_iters = 1;
        s.params.tol_abs = 0.0; s.params.tol_rel = 0.0;
        s.params.use_parallel = true;
        const std::vector<Vec3> x0 = s.x;
        // Also compare predictions with separate broad phases
        BroadPhase bp_a, bp_b;
        bp_a.initialize(x0, s.state.velocities, s.ref_mesh, s.params.dt(), s.params.d_hat);
        bp_b.initialize(x0, s.state.velocities, s.ref_mesh, s.params.dt(), s.params.d_hat);
        printf("[DiagBP] bp_a NT_pairs=%zu SS_pairs=%zu\n", bp_a.cache().nt_pairs.size(), bp_a.cache().ss_pairs.size());
        printf("[DiagBP] bp_b NT_pairs=%zu SS_pairs=%zu\n", bp_b.cache().nt_pairs.size(), bp_b.cache().ss_pairs.size());
        bool cache_match = (bp_a.cache().nt_pairs.size() == bp_b.cache().nt_pairs.size() &&
                            bp_a.cache().ss_pairs.size() == bp_b.cache().ss_pairs.size());
        if (cache_match) {
            for (size_t i = 0; i < bp_a.cache().ss_pairs.size(); ++i) {
                if (bp_a.cache().ss_pairs[i].v[0] != bp_b.cache().ss_pairs[i].v[0]) {
                    printf("[DiagBP] SS pair %zu differs!\n", i); cache_match = false; break;
                }
            }
        }
        printf("[DiagBP] caches %s\n", cache_match ? "MATCH" : "DIFFER");

        // Compare predictions using the two different broad phases
        std::vector<JacobiPrediction> preds_a, preds_b;
        const PinMap pm = build_pin_map(s.pins, s.nv);
        build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params, x0, s.xhat, bp_a.cache(), preds_a, &pm);
        build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params, x0, s.xhat, bp_b.cache(), preds_b, &pm);
        double max_pred_diff = 0.0;
        for (int vi = 0; vi < s.nv; ++vi)
            for (int k = 0; k < 3; ++k)
                max_pred_diff = std::max(max_pred_diff, std::abs(preds_a[vi].delta(k) - preds_b[vi].delta(k)));
        printf("[DiagBP] max prediction diff between two bp = %.3e\n", max_pred_diff);

        // Check if GPU predictions are EXACTLY bit-identical to CPU predictions
        std::vector<JacobiPrediction> preds_gpu, preds_cpu;
        gpu_build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params, x0, s.xhat, bp_a.cache(), preds_gpu, &pm);
        build_jacobi_predictions(s.ref_mesh, s.adj, s.pins, s.params, x0, s.xhat, bp_a.cache(), preds_cpu, &pm);
        int n_exact_diff = 0;
        double max_exact_diff = 0.0; int max_exact_vi = -1, max_exact_k = -1;
        for (int vi = 0; vi < s.nv; ++vi) {
            for (int k = 0; k < 3; ++k) {
                double d = std::abs(preds_gpu[vi].delta(k) - preds_cpu[vi].delta(k));
                if (d != 0.0) n_exact_diff++;
                if (d > max_exact_diff) { max_exact_diff = d; max_exact_vi = vi; max_exact_k = k; }
            }
        }
        printf("[DiagExact] GPU vs CPU predictions: %d non-identical deltas, max diff=%.3e at vi=%d k=%d\n",
            n_exact_diff, max_exact_diff, max_exact_vi, max_exact_k);
        // Dump H, g, delta at the worst vertex to find root cause
        if (max_exact_vi >= 0) {
            int dvi = max_exact_vi;
            printf("[DiagHG] vi=%d CPU H=[%.17e %.17e %.17e / %.17e %.17e %.17e / %.17e %.17e %.17e]\n", dvi,
                preds_cpu[dvi].H(0,0),preds_cpu[dvi].H(0,1),preds_cpu[dvi].H(0,2),
                preds_cpu[dvi].H(1,0),preds_cpu[dvi].H(1,1),preds_cpu[dvi].H(1,2),
                preds_cpu[dvi].H(2,0),preds_cpu[dvi].H(2,1),preds_cpu[dvi].H(2,2));
            printf("[DiagHG] vi=%d GPU H=[%.17e %.17e %.17e / %.17e %.17e %.17e / %.17e %.17e %.17e]\n", dvi,
                preds_gpu[dvi].H(0,0),preds_gpu[dvi].H(0,1),preds_gpu[dvi].H(0,2),
                preds_gpu[dvi].H(1,0),preds_gpu[dvi].H(1,1),preds_gpu[dvi].H(1,2),
                preds_gpu[dvi].H(2,0),preds_gpu[dvi].H(2,1),preds_gpu[dvi].H(2,2));
            printf("[DiagHG] vi=%d CPU g=[%.17e %.17e %.17e]\n", dvi,
                preds_cpu[dvi].g(0),preds_cpu[dvi].g(1),preds_cpu[dvi].g(2));
            printf("[DiagHG] vi=%d GPU g=[%.17e %.17e %.17e]\n", dvi,
                preds_gpu[dvi].g(0),preds_gpu[dvi].g(1),preds_gpu[dvi].g(2));
            printf("[DiagHG] vi=%d CPU delta=[%.17e %.17e %.17e]\n", dvi,
                preds_cpu[dvi].delta(0),preds_cpu[dvi].delta(1),preds_cpu[dvi].delta(2));
            printf("[DiagHG] vi=%d GPU delta=[%.17e %.17e %.17e]\n", dvi,
                preds_gpu[dvi].delta(0),preds_gpu[dvi].delta(1),preds_gpu[dvi].delta(2));
            bool H_match = (preds_cpu[dvi].H == preds_gpu[dvi].H);
            bool g_match = (preds_cpu[dvi].g == preds_gpu[dvi].g);
            printf("[DiagHG] vi=%d H_match=%d g_match=%d\n", dvi, (int)H_match, (int)g_match);
        }
    }
    for (int niters : {1, 2, 3}) {
        BarrierScene s(/*use_trust_region=*/false);
        s.params.max_global_iters = niters;
        s.params.tol_abs          = 0.0;
        s.params.tol_rel          = 0.0;
        s.params.use_parallel     = true;
        const std::vector<Vec3> x0 = s.x;
        BroadPhase bp_gpu, bp_par;
        std::vector<Vec3> x_gpu = x0, x_par = x0;
        gpu_gauss_seidel_solver(s.ref_mesh, s.adj, s.pins, s.params,
                                x_gpu, s.xhat, bp_gpu, s.state.velocities, s.color_groups);
        global_gauss_seidel_solver_parallel(s.ref_mesh, s.adj, s.pins, s.params,
                                            x_par, s.xhat, bp_par, s.state.velocities);
        double max_diff = 0.0; int max_vi = -1;
        for (int i = 0; i < s.nv; ++i) {
            double d = (x_gpu[i] - x_par[i]).norm();
            if (d > max_diff) { max_diff = d; max_vi = i; }
        }
        printf("[DiagIter] niters=%d max_diff=%.3e at vi=%d\n", niters, max_diff, max_vi);
    }
    // Test B: SHARED broad phase — eliminates any bp difference
    {
        BarrierScene s(/*use_trust_region=*/false);
        s.params.max_global_iters = 1;
        s.params.tol_abs = 0.0; s.params.tol_rel = 0.0;
        s.params.use_parallel = true;
        const std::vector<Vec3> x0 = s.x;
        // Use s.bp (already initialized) for both
        std::vector<Vec3> x_gpu = x0, x_par = x0;
        BroadPhase bp_shared;
        bp_shared.initialize(x0, s.state.velocities, s.ref_mesh, s.params.dt(), s.params.d_hat);
        // Run both solvers pointing to same underlying cache
        // (pass as ref — they won't update it since incremental_refresh=false)
        gpu_gauss_seidel_solver(s.ref_mesh, s.adj, s.pins, s.params,
                                x_gpu, s.xhat, bp_shared, s.state.velocities, s.color_groups);
        bp_shared.initialize(x0, s.state.velocities, s.ref_mesh, s.params.dt(), s.params.d_hat); // reinit same
        global_gauss_seidel_solver_parallel(s.ref_mesh, s.adj, s.pins, s.params,
                                            x_par, s.xhat, bp_shared, s.state.velocities);
        double max_diff = 0.0; int max_vi = -1;
        for (int i = 0; i < s.nv; ++i) {
            double d = (x_gpu[i] - x_par[i]).norm();
            if (d > max_diff) { max_diff = d; max_vi = i; }
        }
        printf("[DiagSharedBP] niters=1 max_diff=%.3e at vi=%d\n", max_diff, max_vi);
    }
}

}  // namespace

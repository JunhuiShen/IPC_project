// gpu_solver_bridge.cpp
// Implements gpu_gauss_seidel_solver using the Jacobi-prediction algorithm,
// identical to global_gauss_seidel_solver_parallel.
//
// The two GPU-accelerated phases are dispatched through gpu_solver.h:
//   gpu_build_jacobi_predictions  — Phase 1 (one thread per vertex)
//   gpu_parallel_commit           — Phase 2 (one thread per group member)
//
// On a CPU-stub build (no CUDA) those functions run via OpenMP.
// On a real CUDA build they become kernel launches; everything else
// (conflict graph, coloring, apply_parallel_commits) stays on the CPU.

#include "gpu_solver_bridge.h"
#include "gpu_solver.h"            // gpu_build_jacobi_predictions, gpu_parallel_commit
#include "../parallel_helper.h"    // build_conflict_graph, greedy_color_conflict_graph, apply_parallel_commits
#include "../make_shape.h"         // build_pin_map

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

SolverResult gpu_gauss_seidel_solver(
    const RefMesh&                       ref_mesh,
    const VertexTriangleMap&             adj,
    const std::vector<Pin>&              pins,
    const SimParams&                     params,
    std::vector<Vec3>&                   xnew,
    const std::vector<Vec3>&             xhat,
    BroadPhase&                          broad_phase,
    const std::vector<Vec3>&             v,
    const std::vector<std::vector<int>>& /*color_groups*/,  // ignored — rebuilt dynamically
    std::vector<double>*                 residual_history)
{
    const double dt          = params.dt();
    const double dhat        = params.d_hat;
    const bool   use_barrier = dhat > 0.0;

    // Force the parallel flag so OpenMP branches fire in the CPU stub
    // (on real CUDA these become kernel launches regardless of this flag).
    SimParams p = params;
    p.use_parallel = true;

    const PinMap pm = build_pin_map(pins, static_cast<int>(xnew.size()));

    // -------------------------------------------------------------------
    // Broad-phase initialization
    // -------------------------------------------------------------------
    if (use_barrier)
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);

    // Upload all static-per-sweep data once. Persists across iterations +
    // color groups so predict/commit only re-upload x_current.
    gpu_solver_begin_sweep(ref_mesh, adj, pins, p, xhat, broad_phase.cache(), &pm);

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        const double r_gpu = gpu_compute_global_residual(xnew);
        if (r_gpu >= 0.0) return r_gpu;  // session active — GPU kernel used
        return compute_global_residual(ref_mesh, adj, pins, p,
                                       xnew, xhat, broad_phase, &pm);
    };

    SolverResult result;
    result.initial_residual = eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    const double effective_tol = std::max(p.tol_abs,
                                          p.tol_rel * result.initial_residual);

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < effective_tol) {
        result.converged = true;
        gpu_solver_end_sweep();
        return result;
    }

    // -------------------------------------------------------------------
    // Main iteration loop
    // -------------------------------------------------------------------
    double total_predict_ms = 0, total_commit_ms = 0, total_conflict_ms = 0, total_resid_ms = 0;
    auto t_start_loop = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= p.max_global_iters; ++iter) {

        // --- Phase 1: Jacobi prediction ---
        // GPU kernel target: one thread per vertex.
        // CPU stub: OpenMP parallel for inside gpu_build_jacobi_predictions.
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<JacobiPrediction> predictions;
        gpu_build_jacobi_predictions(ref_mesh, adj, pins, p,
                                     xnew, xhat, broad_phase.cache(),
                                     predictions, &pm);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_predict_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- Conflict graph + coloring ---
        // Conflict graph runs on GPU when a session is active (the swept-
        // region BVH build is still CPU — 1566 boxes, <5ms). Coloring stays
        // on CPU: it's inherently serial (greedy) and already very fast.
        auto t2 = std::chrono::high_resolution_clock::now();
        auto conflict_graph = gpu_build_conflict_graph(predictions);
        if (conflict_graph.empty()) {
            conflict_graph = build_conflict_graph(
                ref_mesh, pins, broad_phase.cache(), predictions, &adj);
        }
        const auto color_groups = greedy_color_conflict_graph(conflict_graph, predictions);
        auto t3 = std::chrono::high_resolution_clock::now();
        total_conflict_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // --- Phase 2: colored Gauss-Seidel ---
        // Fused sweep: one cooperative kernel launch processes every color
        // group with grid-wide barriers between them. Falls back to the
        // per-color path on stub builds / cooperative-launch failure.
        auto t4 = std::chrono::high_resolution_clock::now();
        if (!gpu_fused_sweep(predictions, color_groups, xnew)) {
            for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
                const auto& group = color_groups[color_idx];
                if (group.empty()) continue;

                const bool use_cached = (color_idx == 0);
                const auto commits = gpu_parallel_commit(
                    group, use_cached, predictions,
                    ref_mesh, adj, pins, p,
                    xnew, xhat, broad_phase, &pm);

                // Serial write-back (stays on CPU even on CUDA build)
                apply_parallel_commits(commits, xnew);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        total_commit_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        result.last_num_colors = static_cast<int>(color_groups.size());
        auto t6 = std::chrono::high_resolution_clock::now();
        result.final_residual  = eval_residual();
        auto t7 = std::chrono::high_resolution_clock::now();
        total_resid_ms += std::chrono::duration<double, std::milli>(t7 - t6).count();
        result.iterations      = iter;

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < effective_tol) {
            result.converged = true;
            break;
        }
    }

    auto t_end_loop = std::chrono::high_resolution_clock::now();
    double total_loop_ms = std::chrono::duration<double, std::milli>(t_end_loop - t_start_loop).count();
    fprintf(stderr, "[gpu-prof] iters=%d  total=%.1fms  predict=%.1fms  conflict=%.1fms  commit=%.1fms  resid=%.1fms\n",
            result.iterations, total_loop_ms, total_predict_ms, total_conflict_ms, total_commit_ms, total_resid_ms);

    gpu_solver_end_sweep();
    return result;
}

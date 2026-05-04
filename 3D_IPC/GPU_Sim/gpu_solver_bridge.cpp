// gpu_solver_bridge.cpp
// Implements gpu_gauss_seidel_solver using the Jacobi-prediction algorithm
// (predict + colored-commit certified-region pipeline).
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
#include "../parallel_helper.h"    // build_conflict_graph, greedy_color_conflict_graph
#include "../make_shape.h"         // build_pin_map

#include <algorithm>
#include <vector>

namespace {

void apply_parallel_commits_cpu(const std::vector<ParallelCommit>& commits, std::vector<Vec3>& xnew) {
    for (const auto& commit : commits) {
        if (!commit.valid) continue;
        xnew[commit.vi] = commit.x_after;
    }
}

} // namespace

SolverResult gpu_gauss_seidel_solver(
    const RefMesh&                       ref_mesh,
    const VertexTriangleMap&             adj,
    const std::vector<Pin>&              pins,
    const SimParams&                     params,
    std::vector<Vec3>&                   xnew,
    const std::vector<Vec3>&             xhat,
    BroadPhase&                          broad_phase,
    const std::vector<Vec3>&             v)
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

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, p,
                                       xnew, xhat, broad_phase, &pm);
    };

    SolverResult result;
    result.iterations = 0;

    const double initial_residual = p.fixed_iters ? 0.0 : eval_residual();
    const double effective_tol    = std::max(p.tol_abs, p.tol_rel * initial_residual);

    if (!p.fixed_iters && initial_residual < effective_tol) {
        result.converged = true;
        return result;
    }

    // -------------------------------------------------------------------
    // Main iteration loop
    // -------------------------------------------------------------------
    for (int iter = 1; iter <= p.max_global_iters; ++iter) {

        // --- Phase 1: Jacobi prediction ---
        // GPU kernel target: one thread per vertex.
        // CPU stub: OpenMP parallel for inside gpu_build_jacobi_predictions.
        std::vector<JacobiPrediction> predictions;
        gpu_build_jacobi_predictions(ref_mesh, adj, pins, p,
                                     xnew, xhat, broad_phase.cache(),
                                     predictions, &pm);

        // --- Conflict graph + coloring (CPU, stays CPU on CUDA build) ---
        const auto conflict_graph = build_conflict_graph(
            ref_mesh, pins, broad_phase.cache(), predictions, &adj);
        const auto color_groups = greedy_color_conflict_graph(conflict_graph, predictions);

        // --- Phase 2: colored Gauss-Seidel ---
        // GPU kernel target: one thread per group member per color.
        // CPU stub: OpenMP parallel for inside gpu_parallel_commit.
        for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
            const auto& group = color_groups[color_idx];
            if (group.empty()) continue;

            const bool use_cached = (color_idx == 0);
            const auto commits = gpu_parallel_commit(
                group, use_cached, predictions,
                ref_mesh, adj, pins, p,
                xnew, xhat, broad_phase, &pm);

            // Serial write-back (stays on CPU even on CUDA build)
            apply_parallel_commits_cpu(commits, xnew);
        }

        result.iterations = iter;

        const double cur_residual = p.fixed_iters ? 0.0 : eval_residual();
        if (!p.fixed_iters && cur_residual < effective_tol) {
            result.converged = true;
            break;
        }
    }

    if (p.fixed_iters) result.converged = true;
    return result;
}

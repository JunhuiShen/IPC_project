// gpu_solver_stub.cpp
// CPU stub for gpu_solver.h.
// Wraps the parallel_helper functions with OpenMP so the stub runs on all
// cores when use_parallel is set — identical throughput to --use_parallel.
// On a real CUDA machine replace this file with gpu_solver.cu.

#include "gpu_solver.h"
#include "../parallel_helper.h"

void gpu_build_jacobi_predictions(
    const RefMesh&                 ref_mesh,
    const VertexTriangleMap&       adj,
    const std::vector<Pin>&        pins,
    const SimParams&               params,
    const std::vector<Vec3>&       x,
    const std::vector<Vec3>&       xhat,
    const BroadPhase::Cache&       bp_cache,
    std::vector<JacobiPrediction>& predictions,
    const PinMap*                  pin_map)
{
    // CPU equivalent of the Jacobi predict kernel.
    // build_jacobi_predictions uses #pragma omp parallel for if(use_parallel).
    build_jacobi_predictions(ref_mesh, adj, pins, params, x, xhat, bp_cache, predictions, pin_map);
}

std::vector<ParallelCommit> gpu_parallel_commit(
    const std::vector<int>&              group,
    bool                                 use_cached_prediction,
    const std::vector<JacobiPrediction>& predictions,
    const RefMesh&                       ref_mesh,
    const VertexTriangleMap&             adj,
    const std::vector<Pin>&              pins,
    const SimParams&                     params,
    const std::vector<Vec3>&             x,
    const std::vector<Vec3>&             xhat,
    const BroadPhase&                    broad_phase,
    const PinMap*                        pin_map)
{
    // CPU equivalent of the commit kernel.
    std::vector<ParallelCommit> commits(group.size());

    #pragma omp parallel for schedule(static) if(params.use_parallel && group.size() >= 16)
    for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
        const int vi = group[local_idx];
        commits[local_idx] = compute_parallel_commit_for_vertex(
            vi, use_cached_prediction, predictions[vi],
            ref_mesh, adj, pins, params, x, xhat, broad_phase, pin_map);
    }

    return commits;
}

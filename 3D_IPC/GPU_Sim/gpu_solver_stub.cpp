// gpu_solver_stub.cpp
// CPU stub for gpu_solver.h.
// Wraps the parallel_helper functions with OpenMP so the stub runs on all
// cores when use_parallel is set — identical throughput to --use_parallel.
// On a real CUDA machine replace this file with gpu_solver.cu.

#include "gpu_solver.h"
#include "../parallel_helper.h"

namespace {

ParallelCommit compute_cpu_parallel_commit_for_vertex(
    int vi, bool use_cached_prediction, const JacobiPrediction& prediction,
    const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
    const SimParams& params, const std::vector<Vec3>& x_current, const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase, const PinMap* pin_map)
{
    const auto& bp_cache = broad_phase.cache();
    ParallelCommit out;
    out.vi = vi;

    Vec3 delta = prediction.delta;

    if (!use_cached_prediction) {
        Vec3 fresh_gradient, fresh_delta;
        Mat33 fresh_hessian;
        compute_local_newton_direction(
            vi, ref_mesh, adj, pins, params, x_current, xhat, bp_cache,
            fresh_gradient, fresh_hessian, fresh_delta, pin_map);
        out.alpha_clip = clip_step_to_certified_region(vi, x_current, fresh_delta, prediction.certified_region);
        delta = out.alpha_clip * fresh_delta;
    }

    out.delta = delta;
    out.ccd_step = compute_safe_step_for_vertex(vi, ref_mesh, params, x_current, delta, bp_cache);
    out.x_after = x_current[vi] - out.ccd_step * delta;
    out.valid = true;
    return out;
}

} // namespace

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
    build_jacobi_prediction_deltas(ref_mesh, adj, pins, params, x, xhat, bp_cache, predictions, pin_map);
    build_blue_boxes(x, params.use_parallel, predictions);
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
        commits[local_idx] = compute_cpu_parallel_commit_for_vertex(
            vi, use_cached_prediction, predictions[vi],
            ref_mesh, adj, pins, params, x, xhat, broad_phase, pin_map);
    }

    return commits;
}

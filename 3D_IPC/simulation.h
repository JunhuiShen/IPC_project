#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include "GPU_Elastic/gpu_elastic.h"
#include <vector>

struct TwistSpec;

// Advance one frame across all substeps; returns accumulated stats
// (initial_residual from first substep, final_residual from last, sum of
// iterations / violation counts across all substeps).
inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    const std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase) {
    SolverResult agg;
    // Fast path: elastic-only, all substeps run on-device.
    if (params.use_gpu_elastic) {
        gpu_elastic_begin_frame(state.deformed_positions, state.velocities);
        for (int sub = 0; sub < params.substeps; ++sub) {
            gpu_elastic_run_substep_device(params.max_global_iters);
        }
        gpu_elastic_end_frame(state.deformed_positions, state.velocities);
        // Residuals only become valid after end_frame drains the stream.
        for (int sub = 0; sub < params.substeps; ++sub) {
            SolverResult sub_result;
            sub_result.iterations       = params.max_global_iters;
            sub_result.final_residual   = gpu_elastic_substep_residual(sub);
            sub_result.initial_residual = sub_result.final_residual;
            sub_result.converged        = (sub_result.final_residual >= 0.0);
            sub_result.last_num_colors  = 0;
            accumulate_solver_result(agg, sub_result, sub == 0);
        }
        return agg;
    }

    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

        std::vector<Vec3> xnew = choose_initial_guess(
            state.deformed_positions, xhat, ref_mesh, params);

        SolverResult sub_result;
        if (params.use_gpu)
            sub_result = gpu_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
        else if (params.use_parallel)
            sub_result = global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities);
        else
            sub_result = global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
        accumulate_solver_result(agg, sub_result, sub == 0);

        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
        state.deformed_positions = xnew;
    }
    return agg;
}

// Same as advance_one_frame, but refreshes rotating pin targets from
// twist_spec before every substep. frame_index is 1-based and is combined
// with params.substeps and params.dt() to compute the absolute time at which
// each substep ends.
SolverResult advance_one_frame_twisting(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase, const TwistSpec& twist_spec, int frame_index);

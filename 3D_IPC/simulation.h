#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include <vector>

struct TwistSpec;

// Advance one frame across all substeps; returns accumulated stats
// (initial_residual from first substep, final_residual from last, sum of
// iterations / violation counts across all substeps).
inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    const std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase) {
    SolverResult agg;
    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

        std::vector<Vec3> xnew;
        if (params.use_trust_region)
            xnew = trust_region_initial_guess(state.deformed_positions, xhat, ref_mesh, params.d_hat);
        else if (params.use_ccd_guess)
            xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);
        else
            xnew = state.deformed_positions;

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

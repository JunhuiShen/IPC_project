#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include <vector>

struct TwistSpec;

// Advance one frame across all substeps; returns the last substep's result.
inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    const std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase) {
    SolverResult result;
    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

        std::vector<Vec3> xnew = params.use_trust_region
            ? trust_region_initial_guess(state.deformed_positions, xhat, ref_mesh, params.d_hat)
            : ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);

        if (params.use_parallel)
            result = global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities);
        else
            result = global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
        state.deformed_positions = xnew;
    }
    return result;
}

// Same as advance_one_frame, but refreshes rotating pin targets from
// twist_spec before every substep. frame_index is 1-based and is combined
// with params.substeps and params.dt() to compute the absolute time at which
// each substep ends.
SolverResult advance_one_frame_twisting(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase, const TwistSpec& twist_spec, int frame_index);

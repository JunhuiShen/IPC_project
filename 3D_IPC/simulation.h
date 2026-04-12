#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include <vector>

// Advance the simulation by one frame (all substeps).
// Returns the SolverResult from the last substep.
inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj, 
    const std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups, 
    BroadPhase& broad_phase) {
    SolverResult result;
    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

        const double dt2 = params.dt2();
        std::vector<Vec3> x_guess(xhat.size());
        for (int i = 0; i < static_cast<int>(xhat.size()); ++i)
            x_guess[i] = xhat[i] + dt2 * params.gravity;

        std::vector<Vec3> xnew = ccd_initial_guess(state.deformed_positions,  x_guess, ref_mesh);

        result = global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
        state.deformed_positions = xnew;
    }
    return result;
}

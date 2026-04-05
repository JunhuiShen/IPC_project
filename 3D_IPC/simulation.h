#pragma once
#include "physics.h"
#include "solver.h"
#include <vector>

// Advance the simulation by one frame (all substeps).
// Returns the SolverResult from the last substep.
inline SolverResult advance_one_frame(
        DeformedState& state,
        const RefMesh& ref_mesh,
        const VertexTriangleMap& adj,
        const std::vector<Pin>& pins,
        const SimParams& params,
        const std::vector<std::vector<int>>& color_groups,
        const std::vector<NodeTrianglePair>& nt_pairs,
        const std::vector<SegmentSegmentPair>& ss_pairs) {
    SolverResult result;
    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());
        std::vector<Vec3> xnew = state.deformed_positions;
        result = global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat,
                                            nt_pairs, ss_pairs, color_groups);
        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
        state.deformed_positions = xnew;
    }
    return result;
}

#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include <functional>
#include <string>
#include <vector>

struct TwistSpec;
using PinTargetUpdater = void (*)(std::vector<Pin>& pins, const TwistSpec& spec, double t);

// Advance one frame across all substeps; returns accumulated stats
// (initial_residual from first substep, final_residual from last, sum of
// iterations across all substeps).
// Called after each substep with the global substep index (0-based) and current positions.
using SubstepCallback = std::function<void(int, const std::vector<Vec3>&)>;

inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params,
    BroadPhase& broad_phase, const TwistSpec* twist_spec = nullptr, int frame_index = 1,
    PinTargetUpdater pin_updater = nullptr, SubstepCallback on_substep = nullptr, const std::string& outdir = "") {
    SolverResult agg;
    const double dt = params.dt();
    for (int sub = 0; sub < params.substeps; ++sub) {
        if (twist_spec && pin_updater) {
            const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
            pin_updater(pins, *twist_spec, t_next);
        }

        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, dt);

        std::vector<Vec3> xnew;

        if (params.use_ogc || params.use_ogc_solver)
            xnew = state.deformed_positions;
        else if (params.use_ccd_guess)
            xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);
        else
            xnew = state.deformed_positions;

        SolverResult sub_result;
        if (params.use_gpu)
            sub_result = gpu_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities);
        else if (params.use_ogc_solver)
            sub_result = global_gauss_seidel_solver_ogc(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, nullptr, outdir);
        else
            sub_result = global_gauss_seidel_solver_basic(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, nullptr, outdir);
        accumulate_solver_result(agg, sub_result, sub == 0);

        update_velocity(state.velocities, xnew, state.deformed_positions, dt);
        state.deformed_positions = xnew;

        if (on_substep) {
            const int global_sub = (frame_index - 1) * params.substeps + sub;
            on_substep(global_sub, state.deformed_positions);
        }
    }
    return agg;
}

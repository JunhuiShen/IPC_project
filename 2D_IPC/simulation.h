#pragma once

#include "broad_phase.h"
#include "initial_guess/initial_guess.h"
#include "physics.h"
#include "solver.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

struct AdvanceResult2D {
    double first_initial_residual = 0.0;
    double max_final_residual = 0.0;
    int    total_iterations = 0;
    int    substeps_completed = 0;
};

using SubstepCallback2D = std::function<void(int, const Vec&)>;

inline AdvanceResult2D advance_one_frame(
        DeformedState& state, const RefMesh& ref_mesh, const std::vector<Pin>& pins,
        const SimParams2D& params, BroadPhase& broad_phase,
        int frame_index, SubstepCallback2D on_substep = nullptr) {
    const double dt = params.substep_dt();
    const int substeps = std::max(1, params.substeps);
    AdvanceResult2D aggregate;

    for (int substep = 0; substep < substeps; ++substep) {
        Vec xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, dt);

        Vec xnew;
        apply_initial_guess(params.initial_guess_type, state, ref_mesh, pins, xnew, dt, params.eta);

        std::vector<double> residual_history;
        const SolveResult substep_result = global_gauss_seidel_solver_basic(
                ref_mesh, pins, state, xhat, xnew,
                dt, params.k_spring, params.gravity,
                params.d_hat, params.k_barrier,
                params.max_substep_iters, params.tol_abs, params.eta,
                params.node_box_min, params.node_box_max,
                params.node_box_update_count, broad_phase,
                params.use_ccd_step_policy, params.use_parallel,
                &residual_history);

        if (aggregate.substeps_completed == 0 && !residual_history.empty()) {
            aggregate.first_initial_residual = residual_history.front();
        }
        aggregate.max_final_residual = std::max(aggregate.max_final_residual, substep_result.final_residual);
        aggregate.total_iterations += substep_result.iterations_used;
        aggregate.substeps_completed += 1;

        update_velocity(state.velocities, xnew, state.deformed_positions, dt);
        state.deformed_positions = std::move(xnew);

        if (on_substep) {
            const int global_substep = (frame_index - 1) * substeps + substep;
            on_substep(global_substep, state.deformed_positions);
        }
    }

    return aggregate;
}

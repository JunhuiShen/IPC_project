#pragma once

#include "broad_phase.h"
#include "physics.h"
#include "solver.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

struct AdvanceResult2D {
    double first_initial_residual = 0.0;
    double max_final_residual = 0.0;
    bool   has_rigid_residual = false;
    double max_final_rigid_residual = 0.0;
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
        if (params.initial_guess_type == InitialGuessType::Trivial) {
            xnew = initial_guess::trivial_initial_guess(state);
        } else if (params.initial_guess_type == InitialGuessType::CCD) {
            xnew = initial_guess::ccd_initial_guess(state, ref_mesh, pins, dt, params.eta);
        } else if (params.initial_guess_type == InitialGuessType::Verlet) {
            xnew = initial_guess::verlet_initial_guess(
                    state, ref_mesh, pins, dt, params.eta, params.gravity);
        } else {
            xnew = initial_guess::trivial_initial_guess(state);
        }
        std::vector<double> residual_history;
        const SolveResult substep_result = global_gauss_seidel_solver_basic(
                ref_mesh, pins, state, xhat, xnew, params,
                broad_phase, &residual_history);

        if (aggregate.substeps_completed == 0 && !residual_history.empty()) {
            aggregate.first_initial_residual = residual_history.front();
        }
        aggregate.max_final_residual = std::max(aggregate.max_final_residual, substep_result.final_residual);
        if (substep_result.has_rigid_residual) {
            aggregate.has_rigid_residual = true;
            aggregate.max_final_rigid_residual = std::max(aggregate.max_final_rigid_residual, substep_result.final_rigid_residual);
        }
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

inline AdvanceResult2D advance_one_frame_rb(DeformedState& state, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams2D& params, 
    BroadPhase& broad_phase, int frame_index, SubstepCallback2D on_substep = nullptr) {
    const double dt = params.substep_dt();
    const int substeps = std::max(1, params.substeps);
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());
    AdvanceResult2D aggregate;

    for (int substep = 0; substep < substeps; ++substep) {
        Vec xnew = initial_guess::trivial_initial_guess(state);

        Vec y_current = state.x_coms;
        std::vector<double> theta_current = state.theta;

        std::vector<double> residual_history;
        const SolveResult substep_result = global_gauss_seidel_solver_rb(ref_mesh, pins, state, xnew, y_current, theta_current, params, broad_phase, &residual_history);

        if (aggregate.substeps_completed == 0 && !residual_history.empty())
            aggregate.first_initial_residual = residual_history.front();
        aggregate.max_final_residual = std::max(aggregate.max_final_residual, substep_result.final_residual);
        if (substep_result.has_rigid_residual) {
            aggregate.has_rigid_residual = true;
            aggregate.max_final_rigid_residual = std::max(aggregate.max_final_rigid_residual, substep_result.final_rigid_residual);
        }
        aggregate.total_iterations += substep_result.iterations_used;
        aggregate.substeps_completed += 1;

        // Update free-node velocities
        update_velocity(state.velocities, xnew, state.deformed_positions, dt);

        // Update rb velocities from the change in COM and theta
        for (int rb = 0; rb < num_rbs; ++rb) {
            state.v_coms[rb].x = (y_current[rb].x - state.x_coms[rb].x) / dt;
            state.v_coms[rb].y = (y_current[rb].y - state.x_coms[rb].y) / dt;
            state.omega[rb] = (theta_current[rb] - state.theta[rb]) / dt;
            // Normalize theta to [0, 2pi) after omega is computed
            theta_current[rb] = std::fmod(theta_current[rb], 2.0 * M_PI);
            if (theta_current[rb] < 0.0) theta_current[rb] += 2.0 * M_PI;
        }

        // Commit rb DOFs
        state.x_coms = std::move(y_current);
        state.theta  = std::move(theta_current);
        state.deformed_positions = std::move(xnew);

        if (on_substep) {
            const int global_substep = (frame_index - 1) * substeps + substep;
            on_substep(global_substep, state.deformed_positions);
        }
    }

    return aggregate;
}

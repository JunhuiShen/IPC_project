#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "initial_guess.h"
#include "rigid_body_ipc.h"
#include "time_integration.h"
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Per-substep pin-target updater. The closure captures whichever spec the
// caller wants (TwistSpec, CylinderTwistSpec, etc.).
using PinTargetUpdater = std::function<void(std::vector<Pin>& pins, double t)>;

// Advance one frame across all substeps; returns accumulated stats
// Called after each substep with the global substep index (0-based) and current positions.
using SubstepCallback = std::function<void(int, const std::vector<Vec3>&)>;

inline void sync_rigid_body_particles(
    const RefMesh& ref_mesh, DeformedState& state) {
    if (ref_mesh.rb_nodes.empty())
        return;

    const std::size_t num_rbs = state.x_coms.size();
    if (ref_mesh.rb_nodes.size() != num_rbs
        || ref_mesh.ref_positions.size() != num_rbs
        || state.v_coms.size() != num_rbs
        || state.orientations.size() != num_rbs
        || state.omega.size() != num_rbs) {
        throw std::invalid_argument(
            "sync_rigid_body_particles: inconsistent rigid-body array sizes");
    }

    if (state.velocities.size() < state.deformed_positions.size())
        state.velocities.resize(state.deformed_positions.size(), Vec3::Zero());

    for (std::size_t rb = 0; rb < num_rbs; ++rb) {
        const auto& nodes = ref_mesh.rb_nodes[rb];
        const auto& ref_positions = ref_mesh.ref_positions[rb];
        if (nodes.size() != ref_positions.size()) {
            throw std::invalid_argument(
                "sync_rigid_body_particles: node and reference-position counts differ");
        }

        for (std::size_t local = 0; local < nodes.size(); ++local) {
            const int node = nodes[local];
            if (node < 0
                || node >= static_cast<int>(state.deformed_positions.size())) {
                throw std::out_of_range(
                    "sync_rigid_body_particles: node index is out of range");
            }
            const Vec3 world_offset = quaternion_rotate(
                state.orientations[rb], ref_positions[local]);
            state.deformed_positions[node] = state.x_coms[rb] + world_offset;
            state.velocities[node] =
                state.v_coms[rb] + state.omega[rb].cross(world_offset);
        }
    }
}

// Advance a rigid-body-only frame. Collision handling is intentionally absent.
inline SolverResult advance_one_frame_rb(
    DeformedState& state, const RefMesh& ref_mesh,
    const SimParams& params, int frame_index = 1,
    SubstepCallback on_substep = nullptr) {
    SolverResult agg;
    const double dt = params.dt();

    for (int sub = 0; sub < params.substeps; ++sub) {
        std::vector<Vec3> x_coms_new = state.x_coms;
        std::vector<Vec4> orientations_new = state.orientations;
        std::vector<Vec3> omega_new = state.omega;

        const SolverResult sub_result = global_gauss_seidel_solver_basic_rb(
            ref_mesh, state, params, x_coms_new,
            orientations_new, omega_new, params.verbose);
        accumulate_solver_result(agg, sub_result, sub == 0);

        if (!sub_result.converged)
            return agg;

        update_velocity(state.v_coms, x_coms_new, state.x_coms, dt);
        state.x_coms = x_coms_new;
        state.orientations = orientations_new;
        state.omega = omega_new;
        sync_rigid_body_particles(ref_mesh, state);

        if (on_substep) {
            const int global_sub =
                (frame_index - 1) * params.substeps + sub;
            on_substep(global_sub, state.deformed_positions);
        }
    }
    return agg;
}

inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params,
    BroadPhase& broad_phase, int frame_index = 1,
    PinTargetUpdater pin_updater = nullptr, SubstepCallback on_substep = nullptr, const std::string& outdir = "") {
    SolverResult agg;
    const double dt = params.dt();
    for (int sub = 0; sub < params.substeps; ++sub) {
        if (pin_updater) {
            const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
            pin_updater(pins, t_next);
        }

        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, dt);

        std::vector<Vec3> xnew;

        if (params.use_ogc || params.use_ogc_solver)
            xnew = state.deformed_positions;
        else if (params.use_verlet_guess)
            xnew = verlet_initial_guess(state.deformed_positions, xhat, ref_mesh, params, &broad_phase);
        else if (params.use_ccd_guess)
            xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh, &broad_phase);
        else if (params.use_translation_guess){
            xnew = translation_initial_guess(state.deformed_positions, xhat, ref_mesh, pins, params);
        }
        else
            xnew = state.deformed_positions;

        SolverResult sub_result;
        if (params.use_ogc_solver)
            sub_result = global_gauss_seidel_solver_ogc(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, outdir);
        else
            sub_result = global_gauss_seidel_solver_basic(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, params.verbose);
        
        accumulate_solver_result(agg, sub_result, sub == 0);

        if (!sub_result.converged)
            return agg;

        update_velocity(state.velocities, xnew, state.deformed_positions, dt);
        state.deformed_positions = xnew;

        if (on_substep) {
            const int global_sub = (frame_index - 1) * params.substeps + sub;
            on_substep(global_sub, state.deformed_positions);
        }
    }
    return agg;
}

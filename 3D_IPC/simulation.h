#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "initial_guess.h"
#include "rigid_body_ipc.h"
#include "time_integration.h"
#include "contact_scheduling.h"
#include <algorithm>
#include <optional>
#include <cstdio>
#include <functional>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Per-substep pin-target updater. The closure captures whichever spec the
// caller wants (TwistSpec, CylinderTwistSpec, etc.).
using PinTargetUpdater = std::function<void(std::vector<Pin>& pins, double t)>;

// Advance one frame across all substeps; returns accumulated stats
// Called after each substep with the global substep index (0-based) and current positions.
using SubstepCallback = std::function<void(int, const std::vector<Vec3>&)>;

inline void apply_prescribed_pins(DeformedState& state, const std::vector<Pin>& pins) {
    if (state.velocities.size() != state.deformed_positions.size())
        throw std::invalid_argument("prescribed pins require consistent particle arrays");
    for (const auto& pin : pins) if (pin.prescribed) {
        if (pin.vertex_index < 0 || std::size_t(pin.vertex_index) >= state.deformed_positions.size()
            || !pin.target_position.allFinite())
            throw std::invalid_argument("invalid prescribed pin");
    }
    for (const auto& pin : pins) if (pin.prescribed) {
        state.deformed_positions[pin.vertex_index] = pin.target_position;
        state.velocities[pin.vertex_index].setZero();
    }
}


inline void validate_dat_scene(const RefMesh& ref_mesh, const SimParams& params,
    const DeformedState* state = nullptr) {
    params.validate_dat_parameters();
    if (params.use_ogc
        && (!ref_mesh.rb_nodes.empty() || !ref_mesh.total_mass.empty()
            || !ref_mesh.rb_update_modes.empty() || !ref_mesh.ref_positions.empty()
            || std::any_of(ref_mesh.node_to_rb.begin(), ref_mesh.node_to_rb.end(),
                [](int owner) { return owner >= 0; })
            || (state && (!state->x_coms.empty() || !state->v_coms.empty()
                || !state->orientations.empty() || !state->omega.empty()))))
        throw std::invalid_argument("--use_ogc supports cloth/solid particles only; rigid bodies and rigid proxy nodes are unsupported");
}

inline void validate_jacobi_level_based_ccd_scene(const RefMesh& ref_mesh, const SimParams& params) {
    params.validate_jacobi_level_based_ccd_parameters();
    if (params.use_jacobi_level_based_ccd
        && (!ref_mesh.rb_nodes.empty() || !ref_mesh.total_mass.empty()
            || !ref_mesh.rb_update_modes.empty() || !ref_mesh.ref_positions.empty()
            || std::any_of(ref_mesh.node_to_rb.begin(), ref_mesh.node_to_rb.end(),
                [](int owner) { return owner >= 0; }))) {
        throw std::invalid_argument("--use_jacobi_level_based_ccd supports deformable cloth/solid scenes only; rigid bodies and rigid proxy nodes are unsupported");
    }
}

inline void sync_rigid_body_particles(const RefMesh& ref_mesh, DeformedState& state,
    const std::vector<Vec3>* accepted_positions = nullptr) {
    if (ref_mesh.rb_nodes.empty())
        return;

    const std::size_t num_rbs = state.x_coms.size();
    if (ref_mesh.rb_nodes.size() != num_rbs
        || ref_mesh.ref_positions.size() != num_rbs
        || ref_mesh.rb_update_modes.size() != num_rbs
        || state.v_coms.size() != num_rbs
        || state.orientations.size() != num_rbs
        || state.omega.size() != num_rbs) {
        throw std::invalid_argument("sync_rigid_body_particles: inconsistent rigid-body array sizes");
    }

    if (accepted_positions
        && accepted_positions->size() != state.deformed_positions.size())
        throw std::invalid_argument("sync_rigid_body_particles: accepted-position count differs");

    if (state.velocities.size() < state.deformed_positions.size())
        state.velocities.resize(state.deformed_positions.size(), Vec3::Zero());

    for (std::size_t rb = 0; rb < num_rbs; ++rb) {
        const RigidBodyUpdateMode update_mode =
            ref_mesh.rb_update_modes[rb];
        if (!updates_rigid_translation(update_mode))
            state.v_coms[rb] = Vec3::Zero();
        if (!updates_rigid_orientation(update_mode))
            state.omega[rb] = Vec3::Zero();
        const auto& nodes = ref_mesh.rb_nodes[rb];
        const auto& ref_positions = ref_mesh.ref_positions[rb];
        if (nodes.size() != ref_positions.size()) {
            throw std::invalid_argument("sync_rigid_body_particles: node and reference-position counts differ");
        }

        for (std::size_t local = 0; local < nodes.size(); ++local) {
            const int node = nodes[local];
            if (node < 0
                || node >= static_cast<int>(state.deformed_positions.size())) {
                throw std::out_of_range("sync_rigid_body_particles: node index is out of range");
            }
            // The modern solver publishes the positions it actually accepted.
            // Reconstructing them from a rounded pose can reintroduce an intersection.
            const Vec3 world_offset = accepted_positions
                ? (*accepted_positions)[node] - state.x_coms[rb]
                : quaternion_rotate(state.orientations[rb], ref_positions[local]);
            state.deformed_positions[node] = accepted_positions
                ? (*accepted_positions)[node] : state.x_coms[rb] + world_offset;
            state.velocities[node] = state.v_coms[rb] + state.omega[rb].cross(world_offset);
        }
    }
}

// Publish one solver's generalized coordinates and particle proxies together.
// General and DAT return the collision configuration they actually accepted;
// callers must pass that array rather than project its rounded pose again. The
// published positions also become the next substep's friction history.
inline void commit_rigid_body_solver_state(const RefMesh& ref_mesh,
    DeformedState& state, const std::vector<Vec3>& centers,
    const std::vector<Vec4>& orientations, const std::vector<Vec3>& angular_velocities,
    double dt, const std::vector<Vec3>* accepted_positions = nullptr) {
    update_velocity(state.v_coms, centers, state.x_coms, dt);
    state.x_coms = centers;
    state.orientations = orientations;
    state.omega = angular_velocities;
    sync_rigid_body_particles(ref_mesh, state, accepted_positions);
}

// The caller owns storage for this simulation and joins each frame before reuse.
// Refresh topology at the frame boundary just as the fresh-workspace entry does;
// every substep still rebuilds its current motion boxes and contact candidates.
inline SolverResult advance_one_frame_rb(DeformedState& state, const RefMesh& ref_mesh, const SimParams& params,
    BroadPhase& broad_phase, int frame_index = 1, SubstepCallback on_substep = nullptr) {
    validate_dat_scene(ref_mesh, params, &state);
    validate_jacobi_level_based_ccd_scene(ref_mesh, params);
    if (params.use_jacobi_level_based_ccd)
        throw std::invalid_argument("advance_one_frame_rb does not support Jacobi level-based CCD; use a deformable frame driver");
    SolverResult agg;
    const double dt = params.dt();
    broad_phase.invalidate_mesh_topology();

    std::optional<SimParams> driven_params;
    if (!params.rigid_pins.empty()) driven_params.emplace(params);
    for (int sub = 0; sub < params.substeps; ++sub) {
        if (driven_params) driven_params->rigid_pin_time =
            ((frame_index - 1) * params.substeps + sub + 1) * dt;
        const SimParams& active_params = driven_params ? *driven_params : params;
        // Start at the previous pose with zero displacement/angular increment.
        // Physical velocities remain in state and still drive rigid inertia.
        std::vector<Vec3> x_com_new = state.x_coms;
        std::vector<Vec4> q_new = state.orientations;
        std::vector<Vec3> omega_new(state.omega.size(), Vec3::Zero());

        std::vector<Vec3> accepted_rigid_positions;
        SolverResult sub_result;
        if (params.use_dat) {
            std::vector<Vec3> xnew=state.deformed_positions;
            const std::vector<Vec3> xhat=state.deformed_positions;
            const VertexTriangleMap adjacency;
            const std::vector<Pin> pins;
            SimParams solver_params=active_params;
            solver_params.write_substeps=false;
            sub_result = global_gauss_seidel_solver_dat(ref_mesh,state,adjacency,pins,
                solver_params,xnew,xhat,x_com_new,q_new,omega_new,broad_phase);
            accepted_rigid_positions = std::move(xnew);
        } else {
            sub_result = global_gauss_seidel_solver_rb(ref_mesh, state,
                active_params, x_com_new, q_new, omega_new,
                active_params.use_simd ? &accepted_rigid_positions : nullptr,
                &broad_phase);
        }
        accumulate_solver_result(agg, sub_result, sub == 0);

        if (!sub_result.converged)
            return agg;

        commit_rigid_body_solver_state(ref_mesh, state, x_com_new, q_new, omega_new, dt,
            (active_params.use_dat || active_params.use_simd)
                ? &accepted_rigid_positions : nullptr);

        if (on_substep) {
            const int global_sub =
                (frame_index - 1) * params.substeps + sub;
            on_substep(global_sub, state.deformed_positions);
        }
    }
    return agg;
}

// Preserve the existing fresh-workspace API for callers that do not own one.
inline SolverResult advance_one_frame_rb(DeformedState& state, const RefMesh& ref_mesh, const SimParams& params,
    int frame_index = 1, SubstepCallback on_substep = nullptr) {
    BroadPhase broad_phase;
    return advance_one_frame_rb(state, ref_mesh, params, broad_phase,
        frame_index, std::move(on_substep));
}

// Advance a frame containing both independently deformable nodes and rigid
// bodies. The previous-step state remains fixed throughout each nonlinear
// solve; cloth/solid positions and rigid generalized coordinates are committed
// together only after convergence.
inline SolverResult advance_one_frame_general(
    DeformedState& state, const RefMesh& ref_mesh,
    const VertexTriangleMap& adj, std::vector<Pin>& pins,
    const SimParams& params, BroadPhase& broad_phase,
    int frame_index = 1, PinTargetUpdater pin_updater = nullptr,
    SubstepCallback on_substep = nullptr,
    const std::string& outdir = "") {
    validate_dat_scene(ref_mesh, params, &state);
    validate_jacobi_level_based_ccd_scene(ref_mesh, params);
    params.validate_colored_ccd_guess_parameters();
    if (ref_mesh.node_to_rb.size() != state.deformed_positions.size()
        || state.velocities.size() != state.deformed_positions.size()) {
        throw std::invalid_argument(
            "advance_one_frame_general: inconsistent particle arrays");
    }

    SolverResult aggregate;
    const double dt = params.dt();
    const int nv = static_cast<int>(state.deformed_positions.size());

    for (int substep = 0; substep < params.substeps; ++substep) {
        if (pin_updater) {
            const double t_next =
                ((frame_index - 1) * params.substeps + substep + 1) * dt;
            pin_updater(pins, t_next);
        }
        if (params.use_dat || params.use_ogc)
            apply_prescribed_pins(state, pins);

        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, dt);
        // Rigid inertia is represented by COM/orientation variables. Do not
        // also give rigid proxies independent particle predictors.
        for (int node = 0; node < nv; ++node) {
            if (ref_mesh.node_to_rb[node] >= 0)
                xhat[node] = state.deformed_positions[node];
        }

        // DAT/OGC filter their inertial predictors internally. Start their
        // input iterate at the physical previous configuration and retain xhat.
        // The opt-in guess moves cloth/solid nodes only. Rigid proxies remain
        // at their previous positions and participate as fixed CCD obstacles;
        // COM/orientation are left to the general solver. Legacy particle
        // guesses remain disabled here because they do not preserve rigidity.
        std::vector<Vec3> xnew = state.deformed_positions;
        if (!params.use_dat && !params.use_jacobi_level_based_ccd
            && (params.use_colored_ccd_guess || params.use_colored_ccd_guess_level_based)
            && !params.use_ogc) {
            std::vector<Vec3> intended_displacement(nv, Vec3::Zero());
            for (int node = 0; node < nv; ++node) {
                if (ref_mesh.node_to_rb[node] < 0) {
                    intended_displacement[node] = xhat[node] - state.deformed_positions[node];
                    // Scale the predictor target once; retain the solver's xhat.
                    if (params.colored_ccd_guess_fraction != 1.0)
                        intended_displacement[node] *= params.colored_ccd_guess_fraction;
                }
            }
            const auto colored_guess = params.use_colored_ccd_guess_level_based
                ? collision_colored_ccd_initial_guess_level_based : collision_colored_ccd_initial_guess;
            xnew = colored_guess(state.deformed_positions,
                intended_displacement, ref_mesh, params, params.colored_ccd_guess_iters,
                &broad_phase, nullptr, &xnew);
        }
        std::vector<Vec3> x_com_new = state.x_coms;
        std::vector<Vec4> q_new = state.orientations;
        std::vector<Vec3> omega_new(state.omega.size(), Vec3::Zero());

        SolverResult substep_result;
        if (params.use_dat)
            substep_result = global_gauss_seidel_solver_dat(ref_mesh, state, adj, pins,
                params, xnew, xhat, x_com_new, q_new, omega_new, broad_phase, outdir);
        else if (params.use_ogc)
            substep_result = global_gauss_seidel_solver_ogc(ref_mesh, adj, pins,
                params, xnew, xhat, state.velocities, outdir, &state.deformed_positions);
        else if (params.use_jacobi_level_based_ccd)
            substep_result = global_jacobi_solver_level_based_ccd(ref_mesh, adj, pins,
                params, xnew, xhat, state.velocities, broad_phase, outdir,
                &state.deformed_positions);
        else
            substep_result = global_gauss_seidel_solver_general(ref_mesh, state, adj, pins,
                params, xnew, xhat, x_com_new, q_new, omega_new, broad_phase, outdir);
        accumulate_solver_result(aggregate, substep_result, substep == 0);
        if (!substep_result.converged)
            return aggregate;

        // Commit independent cloth/solid nodes only. Rigid proxy particles are
        // synchronized from generalized state below.
        #pragma omp parallel for schedule(static)
        for (int node = 0; node < nv; ++node) {
            if (ref_mesh.node_to_rb[node] >= 0)
                continue;
            state.velocities[node] = (xnew[node] - state.deformed_positions[node]) / dt;
            state.deformed_positions[node] = xnew[node];
        }

        commit_rigid_body_solver_state(ref_mesh, state, x_com_new, q_new, omega_new, dt,
            (!params.use_ogc && !params.use_jacobi_level_based_ccd) ? &xnew : nullptr);

        if (on_substep) {
            const int global_substep = (frame_index - 1) * params.substeps + substep;
            on_substep(global_substep, state.deformed_positions);
        }
    }
    return aggregate;
}

inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params,
    BroadPhase& broad_phase, int frame_index = 1,
    PinTargetUpdater pin_updater = nullptr, SubstepCallback on_substep = nullptr, const std::string& outdir = "") {
    validate_dat_scene(ref_mesh, params, &state);
    validate_jacobi_level_based_ccd_scene(ref_mesh, params);
    if(params.use_dat && !ref_mesh.rb_nodes.empty())
        return advance_one_frame_general(state,ref_mesh,adj,pins,params,broad_phase,
            frame_index,pin_updater,on_substep,outdir);
    params.validate_cloth_grid_parameters();
    params.validate_colored_ccd_guess_parameters();
    const bool has_noncloth_geometry = !ref_mesh.rb_nodes.empty() || !ref_mesh.tets.empty()
        || !ref_mesh.tet_nodes.empty()
        || std::any_of(ref_mesh.node_to_rb.begin(), ref_mesh.node_to_rb.end(),
            [](int owner) { return owner >= 0; });
    if (!params.use_dat && !params.use_ogc && !params.use_jacobi_level_based_ccd
        && (params.use_colored_ccd_guess || params.use_colored_ccd_guess_level_based)
        && has_noncloth_geometry) {
        throw std::invalid_argument("advance_one_frame is cloth-only; use advance_one_frame_general for solid or mixed scenes");
    }
    SolverResult agg;
    const double dt = params.dt();
    struct StepScratch {
        std::vector<Vec3> xhat, xnew, intended_displacement;
        BroadPhase predictor_broad_phase;
    };
    // Retain allocations between frames. A scoped loan keeps nested substep
    // callbacks from overwriting the current frame's working arrays.
    solver_detail::BorrowedContactBuffer<StepScratch, std::allocator<StepScratch>> storage;
    storage.resize(1);
    auto& workspace = storage.values()[0];
    auto& predictor_broad_phase = workspace.predictor_broad_phase;
    // Refresh topology at frame boundaries for callers which mutate their mesh.
    predictor_broad_phase.invalidate_mesh_topology();
    auto& xhat = workspace.xhat;
    auto& xnew = workspace.xnew;
    for (int sub = 0; sub < params.substeps; ++sub) {
        if (pin_updater) {
            const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
            pin_updater(pins, t_next);
        }
        if (params.use_dat || params.use_ogc)
            apply_prescribed_pins(state, pins);

        build_xhat(xhat, state.deformed_positions, state.velocities, dt);

        // DAT/OGC perform their own filtered predictor. Keep physical velocity
        // in xhat and pass the previous configuration as their input iterate.
        if (params.use_dat || params.use_jacobi_level_based_ccd
            || params.use_ogc)
            xnew = state.deformed_positions;
        else if (params.use_colored_ccd_guess || params.use_colored_ccd_guess_level_based) {
            // Only the predictor targets a fraction of the displacement to
            // xhat. Scale once here; every sweep (including the level trial)
            // retries toward that same target. Keep the solver's xhat intact.
            auto& intended_displacement = workspace.intended_displacement;
            intended_displacement.resize(xhat.size());
            for (std::size_t vertex = 0; vertex < xhat.size(); ++vertex) {
                intended_displacement[vertex] = xhat[vertex] - state.deformed_positions[vertex];
                if (params.colored_ccd_guess_fraction != 1.0)
                    intended_displacement[vertex] *= params.colored_ccd_guess_fraction;
            }
            const auto colored_guess = params.use_colored_ccd_guess_level_based
                ? collision_colored_ccd_initial_guess_level_based : collision_colored_ccd_initial_guess;
            xnew = colored_guess(state.deformed_positions,
                intended_displacement, ref_mesh, params, params.colored_ccd_guess_iters,
                &broad_phase, nullptr, &xnew);
        }
        else if (params.use_ccd_guess)
            xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh, &predictor_broad_phase, &xnew);
        else
            xnew = state.deformed_positions;

        SolverResult sub_result;
        if (params.use_dat) {
            std::vector<Vec3> centers, omega;
            std::vector<Vec4> orientations;
            sub_result = global_gauss_seidel_solver_dat(ref_mesh, state, adj, pins,
                params, xnew, xhat, centers, orientations, omega, broad_phase, outdir);
        }
        else if (params.use_jacobi_level_based_ccd)
            sub_result = global_jacobi_solver_level_based_ccd(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        else if (params.use_ogc)
            sub_result = global_gauss_seidel_solver_ogc(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, outdir, &state.deformed_positions);
        else if (params.use_cloth_grid)
            sub_result = global_gauss_seidel_solver_ambient_grid(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        else
            sub_result = global_gauss_seidel_solver_cloth(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        
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

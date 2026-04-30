#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include "GPU_Elastic/gpu_elastic.h"
#include "GPU_Elastic/gpu_hash_grid.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

struct TwistSpec;
using PinTargetUpdater = void (*)(std::vector<Pin>& pins, const TwistSpec& spec, double t);

// Advance one frame across all substeps; returns accumulated stats
// (initial_residual from first substep, final_residual from last, sum of
// iterations / violation counts across all substeps).
// Called after each substep with the global substep index (0-based) and current positions.
using SubstepCallback = std::function<void(int, const std::vector<Vec3>&)>;

inline SolverResult advance_one_frame(DeformedState& state, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    std::vector<Pin>& pins, const SimParams& params, const std::vector<std::vector<int>>& color_groups,
    BroadPhase& broad_phase, const TwistSpec* twist_spec = nullptr, int frame_index = 1,
    PinTargetUpdater pin_updater = nullptr, SubstepCallback on_substep = nullptr, const std::string& outdir = "") {
    SolverResult agg;
    const double dt = params.dt();

    // Fast path: elastic-only, all substeps device-resident. x/v never touch
    // the host mid-frame; pin targets refresh per substep when twisting.
    if (params.use_gpu_elastic) {
        gpu_elastic_begin_frame(state.deformed_positions, state.velocities);
        // Per-vertex prev_disp tracking mirrors gauss_seidel_basic at
        // solver.cpp:261-269. Initialized to node_box_max on first call so
        // substep 0 uses the upper-bound radius; subsequent substeps clamp
        // (prev_disp * padding) to [node_box_min, node_box_max].
        const int nv = static_cast<int>(state.deformed_positions.size());
        static std::vector<double> prev_disp;
        if (static_cast<int>(prev_disp.size()) != nv)
            prev_disp.assign(nv, params.node_box_max);
        constexpr double node_box_padding = 1.2;
        std::vector<Vec3> bp_x_scratch;
        std::vector<Vec3> bp_x_prev;
        std::vector<double> bp_radii(nv);
        for (int sub = 0; sub < params.substeps; ++sub) {
            if (twist_spec && pin_updater) {
                const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
                pin_updater(pins, *twist_spec, t_next);
                gpu_elastic_set_pin_targets(pins);
            }
            // Capture pre-substep positions to compute displacement after.
            if (params.use_broadphase || params.use_cpu_broadphase) {
                gpu_elastic_peek_positions(bp_x_prev);
                for (int i = 0; i < nv; ++i) {
                    bp_radii[i] = std::clamp(prev_disp[i] * node_box_padding,
                                             params.node_box_min,
                                             params.node_box_max);
                }
            }
            gpu_elastic_run_substep_device(params.max_global_iters);
            if (params.use_broadphase || params.use_cpu_broadphase) {
                gpu_elastic_peek_positions(bp_x_scratch);
                if (params.use_broadphase) {
                    auto pairs = gpu_hash_grid_build_pairs(
                        bp_x_prev, ref_mesh, bp_radii, params.d_hat);
                    (void)pairs;
                } else {
                    std::vector<AABB> blue_boxes(nv);
                    for (int i = 0; i < nv; ++i) {
                        blue_boxes[i] = AABB(
                            bp_x_prev[i] - Vec3::Constant(bp_radii[i]),
                            bp_x_prev[i] + Vec3::Constant(bp_radii[i]));
                    }
                    BroadPhase bp;
                    bp.initialize(blue_boxes, ref_mesh, params.d_hat);
                    (void)bp;
                }
                // Update prev_disp for next substep from this substep's
                // actual displacement.
                for (int i = 0; i < nv; ++i)
                    prev_disp[i] = (bp_x_scratch[i] - bp_x_prev[i]).norm();
            }
        }
        gpu_elastic_end_frame(state.deformed_positions, state.velocities);
        for (int sub = 0; sub < params.substeps; ++sub) {
            SolverResult sub_result;
            sub_result.iterations       = params.max_global_iters;
            sub_result.final_residual   = gpu_elastic_substep_residual(sub);
            sub_result.initial_residual = sub_result.final_residual;
            sub_result.converged        = (sub_result.final_residual >= 0.0);
            sub_result.last_num_colors  = 0;
            accumulate_solver_result(agg, sub_result, sub == 0);
        }
        // Substep callback intentionally not fired in the device-resident
        // fast path — intermediate positions never come back to the host.
        return agg;
    }

    for (int sub = 0; sub < params.substeps; ++sub) {
        if (twist_spec && pin_updater) {
            const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
            pin_updater(pins, *twist_spec, t_next);
        }

        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, dt);

        std::vector<Vec3> xnew;
        if (params.use_trust_region)
            xnew = trust_region_initial_guess(state.deformed_positions, xhat, ref_mesh, params.d_hat);
        else if (params.use_ccd_guess)
            xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);
        else
            xnew = state.deformed_positions;

        SolverResult sub_result;
        if (params.experimental)
            sub_result = global_gauss_seidel_solver_basic(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, nullptr, outdir);
        else if (params.use_gpu)
            sub_result = gpu_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
        else if (params.use_parallel)
            sub_result = global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities);
        else
            sub_result = global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, state.velocities, color_groups);
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

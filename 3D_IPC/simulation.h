#pragma once
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "GPU_Sim/gpu_solver_bridge.h"
#include "GPU_Elastic/gpu_elastic.h"
#include "GPU_Elastic/gpu_hash_grid.h"
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
        std::vector<Vec3> bp_x_scratch;  // reused across substeps when use_broadphase is on
        for (int sub = 0; sub < params.substeps; ++sub) {
            if (twist_spec && pin_updater) {
                const double t_next = ((frame_index - 1) * params.substeps + (sub + 1)) * dt;
                pin_updater(pins, *twist_spec, t_next);
                gpu_elastic_set_pin_targets(pins);
            }
            gpu_elastic_run_substep_device(params.max_global_iters);
            if (params.use_broadphase || params.use_cpu_broadphase) {
                // Snapshot current device positions, then run either GPU or
                // CPU broad phase. Pair list discarded — measures overhead
                // before the GS sweep consumes the lists.
                gpu_elastic_peek_positions(bp_x_scratch);
                if (params.use_broadphase) {
                    auto pairs = gpu_hash_grid_build_pairs(
                        bp_x_scratch, ref_mesh,
                        params.node_box_max, params.d_hat);
                    (void)pairs;
                } else {
                    std::vector<AABB> blue_boxes(bp_x_scratch.size());
                    for (std::size_t i = 0; i < bp_x_scratch.size(); ++i) {
                        blue_boxes[i] = AABB(
                            bp_x_scratch[i] - Vec3::Constant(params.node_box_max),
                            bp_x_scratch[i] + Vec3::Constant(params.node_box_max));
                    }
                    BroadPhase bp;
                    bp.initialize(blue_boxes, ref_mesh, params.d_hat);
                    (void)bp;
                }
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

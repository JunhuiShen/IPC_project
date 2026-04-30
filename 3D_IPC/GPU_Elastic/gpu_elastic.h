#pragma once
// ============================================================================
// GPU Elastic Solver — no-collision cloth / shell solver.
//
// Scope: Newton / colored Gauss-Seidel for elastic (corotated StVK) + bending +
// pin + inertia. No barrier energy, no CCD, no certified-region clipping, no
// swept-region BVH, no JP-at-runtime coloring.
//
// Why a separate file: the collision path in GPU_Sim/gpu_solver.cu carries a
// lot of machinery — NT/SS barrier pair CSRs, LBVH rebuild each iter, JP
// coloring + compaction every iter, CCD safe-step per vertex. For no-collision
// scenes none of that is needed; the conflict graph IS the mesh adjacency and
// the coloring is fixed per simulation. This module is a minimal baseline from
// which collision can be layered back in carefully.
//
// Lifecycle:
//     gpu_elastic_init(mesh, adj, pins, params)    // once, at sim start
//     per substep:
//         gpu_elastic_set_pin_targets(pins)        // if pin targets animate
//         gpu_elastic_run_substep(x, xhat, max_iters, x_out)
//         residual = gpu_elastic_last_residual()
//     gpu_elastic_shutdown()                        // once, at sim end
// ============================================================================

#include "../physics.h"
#include <vector>

// Once-per-simulation static setup. Uploads mesh + adjacency + pin map,
// precomputes the color groups (natural greedy over the mesh's vertex-
// adjacency graph — fixed since no collision can add dynamic conflict edges),
// allocates all per-vertex buffers. Safe to call repeatedly: tears down any
// prior session first.
void gpu_elastic_init(
    const RefMesh&           ref_mesh,
    const VertexTriangleMap& adj,
    const std::vector<Pin>&  pins,
    const SimParams&         params);

// Full teardown (stream, buffers, graph capture state, etc.).
void gpu_elastic_shutdown();

// Update pin targets if they've changed (e.g. twist spec moves them each
// substep). Re-uploads only the target positions; pin indices / pin_map are
// considered structurally static across a simulation.
void gpu_elastic_set_pin_targets(const std::vector<Pin>& pins);

// Integrate one substep: Newton-GS for `max_iters` iterations with fixed
// coloring. On entry d_x is uploaded from `x_in`; the captured graph now
// builds xhat on-device from (x, v) so the host-supplied `xhat` is ignored.
// On return, `x_out` holds the final per-vertex positions. The last
// residual is stashed and retrievable via gpu_elastic_last_residual().
// Returns true on success, false if no session.
bool gpu_elastic_run_substep(
    const std::vector<Vec3>& x_in,
    const std::vector<Vec3>& xhat,
    int                      max_iters,
    std::vector<Vec3>&       x_out);

// Fast-path device-resident API: bracket a frame's substep loop with
// gpu_elastic_begin_frame (one H2D of x, v) and gpu_elastic_end_frame (one
// D2H pair). Between them, gpu_elastic_run_substep_device runs entirely on
// the GPU — the captured graph builds xhat from (x, v), saves x_prev, runs
// the solve, updates v = (x_new - x_prev)/dt, and reduces the residual.
// gpu_elastic_set_pin_targets still refreshes pin targets between substeps.
void gpu_elastic_begin_frame(const std::vector<Vec3>& x, const std::vector<Vec3>& v);
bool gpu_elastic_run_substep_device(int max_iters);
void gpu_elastic_end_frame(std::vector<Vec3>& x, std::vector<Vec3>& v);

// Per-substep residual recorded during the current frame. Only valid after
// gpu_elastic_end_frame has drained the stream. sub is in [0, substeps).
double gpu_elastic_substep_residual(int sub);

// Snapshot device positions to host without ending the frame. Forces a
// stream sync, copies d_x to `x`, returns. Velocities, d_x_prev, etc. are
// left untouched, so the substep loop can continue. Used by external
// per-substep consumers (e.g. broad phase) that need positions mid-frame.
void gpu_elastic_peek_positions(std::vector<Vec3>& x);

// Residual from the most recent run_substep (reduction over d_x using the
// final state). Mass-normalized when params.mass_normalize_residual is set.
double gpu_elastic_last_residual();

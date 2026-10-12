#pragma once

#include <memory>

namespace ipc_gpu::resident {

struct DeformableScene;
struct DeformableState;
struct BpAabb;

// Persistent device-only history for the deformable basic/basic-v2 node-box
// policy. It does not change rigid proxy boxes or initial-guess motion boxes.
// All numerical pointers are device pointers; methods enqueue default-stream
// kernels and never download positions, velocities, radii, or displacements.
class ResidentAdaptiveBoxes {
public:
    ResidentAdaptiveBoxes();
    ~ResidentAdaptiveBoxes();
    ResidentAdaptiveBoxes(const ResidentAdaptiveBoxes&) = delete;
    ResidentAdaptiveBoxes& operator=(const ResidentAdaptiveBoxes&) = delete;

    // Call AFTER applying the initial guess, immediately before the solver
    // iteration loop. The anchor is the guessed position, NOT x_n. Incoming
    // velocities are still the previous substep's velocities. On the first
    // call after construction/reset (or a node-count change), prev_disp starts
    // at maximum_radius; otherwise its device history survives substeps.
    //
    // inertial_disp = norm(v_n) * dt
    // radius = clamp(1.2 * max(prev_disp, inertial_disp), minimum, maximum)
    //
    // The radius remains fixed within this solver call; each build recenters
    // the box at the current position, matching the CPU's rebuild behavior.
    void begin_solver(const DeformableScene&, const DeformableState&,
                      double dt, double minimum_radius, double maximum_radius,
                      int* device_error);
    void build(const DeformableScene&, const DeformableState&,
               BpAabb* device_boxes, int* device_error);

    // Call after the final solver iteration and before velocity integration.
    // Stores norm(final_position - solver_entry_position), including soft-pin
    // movement. Initial-guess displacement is deliberately not included.
    void finish_solver(const DeformableScene&, const DeformableState&,
                       int* device_error);

    // Invalidate history when replacing topology. Existing allocation is
    // retained; the next begin_solver initializes prev_disp to its maximum.
    // The caller must finish/abandon earlier queued work before topology reuse.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::resident

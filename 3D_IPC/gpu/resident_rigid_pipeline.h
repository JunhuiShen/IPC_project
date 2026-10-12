#pragma once

#include "resident_scene.h"

namespace ipc_gpu::resident {

enum RigidPipelineError : int {
    RigidInvalidState = 40,
    RigidInvalidDerivative = 41,
    RigidSolveFailed = 42,
    RigidInvalidBounds = 43,
    RigidRotationFailure = 44,
    RigidContactOwnership = 45,
};

// All storage pointers are DEVICE pointers. These functions enqueue kernels;
// no host physics, geometry gathering, solve, or CCD fallback is performed.
// The caller clears error at the substep boundary and checks it before output.
// A nonzero error invalidates the substep and blocks subsequent commits.
void begin_rigid_substep(const RigidScene&, const RigidState&, int* error,
                        cudaStream_t stream = nullptr);

// body_boxes are allowed COM regions until the next rebuild. Rotating proxy
// vertices receive a spherical cap around the rebuild orientation plus the
// COM region. If provided, anchor_quats records the cap anchors, and actual
// represented_positions conservatively pads any rounded pose discrepancy.
// A restricted cap requires anchors in every subsequent angular proposal.
void build_rigid_blue_boxes(const RigidScene&, const RigidState&,
                           std::size_t total_nodes, double com_radius,
                           BpAabb* body_boxes, BpAabb* node_boxes, int* error,
                           double max_angle = 3.14159265358979323846,
                           double (*anchor_quats)[4] = nullptr,
                           const double (*represented_positions)[3] = nullptr,
                           cudaStream_t stream = nullptr);

// Initialize per-body systems; contact/SDF kernels ADD their contributions
// after this call. false = COM solve; true = angular-velocity solve.
void assemble_rigid_inertia_color(const RigidScene&, const RigidState&,
                                 const RigidWorkspace&, const int* block_colors,
                                 int color, bool orientation_phase, double dt,
                                 double gravity_y, int* error,
                                 cudaStream_t stream = nullptr);
// Angular proposals are shortened on-device to max_rotation radians per
// update, including huge finite Newton increments. This is an update cap,
// If anchors are supplied, proposals also stay inside the angular cap fixed
// at the last broad-phase rebuild (its radius must equal max_rotation).
void propose_rigid_color(const RigidScene&, const RigidState&,
                        const RigidWorkspace&, const int* block_colors,
                        int color, bool orientation_phase,
                        const BpAabb* body_boxes, double dt, double damping,
                        int* error, double max_rotation = .1,
                        const double (*anchor_quats)[4] = nullptr,
                        cudaStream_t stream = nullptr);

// Translation CCD is owned by the resident CCD queue. It fills accepted_com
// and the exact accepted_proxy_positions. Commit copies them verbatim; it
// MUST NOT re-form validated endpoints from a fraction or COM displacement.
void commit_rigid_translation_color(const RigidScene&, const RigidState&,
                                   const RigidWorkspace&, const DeformableState&,
                                   const int* block_colors, int color,
                                   const double (*accepted_proxy_positions)[3],
                                   int* error, cudaStream_t stream = nullptr);

// Must run after COM commit and after a fresh orientation system/proposal.
// Current analytic curved-trajectory CPU CCD branches are evaluated on GPU
// for every incident candidate; 0.9 * minimum TOI clips a hit. Ambiguous arcs
// are explicit errors, never interpreted as collision-free. Accepted proxy
// endpoints are computed once and copied unchanged by the following commit.
void clip_rigid_rotation_color(const RigidScene&, const RigidState&,
                               const RigidWorkspace&, const DeformableState&,
                               const BroadPhaseView&, const int* block_colors,
                               int color, double dt, double safety,
                               bool use_ccd, double (*accepted_proxy_positions)[3],
                               int* error, cudaStream_t stream = nullptr);
void commit_rigid_rotation_color(const RigidScene&, const RigidState&,
                                 const RigidWorkspace&, const DeformableState&,
                                 const int* block_colors, int color,
                                 const double (*accepted_proxy_positions)[3],
                                 int* error, cudaStream_t stream = nullptr);
void finish_rigid_substep(const RigidScene&, const RigidState&,
                         const DeformableState&, double dt, int* error,
                         cudaStream_t stream = nullptr);

} // namespace ipc_gpu::resident

#pragma once
// ============================================================================
// GPU Hash Grid Broad Phase — node-triangle pair extraction.
//
// Purpose: GPU-side equivalent of the CPU `BroadPhase::initialize(blue_boxes,
// ref_mesh, d_hat)` that powers gauss_seidel_basic. Uses a uniform spatial
// hash grid keyed on cell coordinates derived from the mesh AABB.
//
// First cut: NT (node-triangle) pairs only. SS (segment-segment) pairs to
// follow once NT correctness vs CPU is established.
//
// Cell size = 2 * node_box_size + d_hat (worst-case overlap diameter).
// On non-CUDA builds the stub forwards to the CPU BroadPhase so callers and
// tests compile unchanged.
// ============================================================================

#include "../physics.h"
#include <vector>

struct GpuBroadPhaseResult {
    std::vector<NodeTrianglePair>     nt_pairs;
    std::vector<SegmentSegmentPair>   ss_pairs;
};

// Inputs mirror BroadPhase::initialize(blue_boxes, mesh, d_hat) but with
// per-vertex blue box half-extents passed in `per_vertex_radii` (length nv).
// Mirrors gauss_seidel_basic's per-vertex `node_box_size_fn(vi)` clamping
// of prev_disp * padding to [node_box_min, node_box_max].
GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>&    positions,
    const RefMesh&              ref_mesh,
    const std::vector<double>&  per_vertex_radii,
    double                      d_hat);

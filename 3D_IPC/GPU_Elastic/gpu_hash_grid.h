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

// Inputs mirror BroadPhase::initialize(blue_boxes, mesh, d_hat) modulo the
// fact that blue boxes here are reconstructed internally from positions +
// node_box_size (uniform half-extent, basic-solver style).
GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>& positions,
    const RefMesh&           ref_mesh,
    double                   node_box_size,
    double                   d_hat);

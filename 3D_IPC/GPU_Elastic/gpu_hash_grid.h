#pragma once
// ============================================================================
// GPU Hash Grid Broad Phase — node-triangle and segment-segment pair extraction.
//
// Mirrors CPU `BroadPhase::initialize(blue_boxes, mesh, d_hat)` but keeps the
// resulting pair lists on the device. Future GS sweep kernels read them
// directly without an intermediate host roundtrip.
//
// Pair keys (packed uint64):
//   NT: (node << 32) | tri_index_in_mesh
//   SS: (edge_a << 32) | edge_b
//
// SS edge indices reference d_edges[] in the result (also pointed at the
// internal cache so the GS sweep can resolve edge_id → (v0, v1)).
//
// On non-CUDA builds the stub leaves all pointers null and counts at 0 — the
// hash grid is a CUDA-only feature in this branch.
// ============================================================================

#include "../physics.h"
#include <vector>

struct GpuBroadPhaseResult {
    // Device-resident packed keys, owned. Freed by destructor.
    unsigned long long* d_nt_keys  = nullptr;
    int                 n_nt_pairs = 0;
    unsigned long long* d_ss_keys  = nullptr;
    int                 n_ss_pairs = 0;

    // Edge table (device side, layout: 2*i = v0, 2*i+1 = v1). NOT owned —
    // points into the persistent edge cache and lives across calls. Indexes
    // referenced by d_ss_keys are valid into this array.
    const int*          d_edges  = nullptr;
    int                 n_edges  = 0;

    GpuBroadPhaseResult() = default;
    ~GpuBroadPhaseResult();
    GpuBroadPhaseResult(const GpuBroadPhaseResult&) = delete;
    GpuBroadPhaseResult& operator=(const GpuBroadPhaseResult&) = delete;
    GpuBroadPhaseResult(GpuBroadPhaseResult&& o) noexcept { *this = static_cast<GpuBroadPhaseResult&&>(o); }
    GpuBroadPhaseResult& operator=(GpuBroadPhaseResult&& o) noexcept {
        d_nt_keys = o.d_nt_keys; o.d_nt_keys = nullptr;
        n_nt_pairs = o.n_nt_pairs; o.n_nt_pairs = 0;
        d_ss_keys = o.d_ss_keys; o.d_ss_keys = nullptr;
        n_ss_pairs = o.n_ss_pairs; o.n_ss_pairs = 0;
        d_edges = o.d_edges; o.d_edges = nullptr;
        n_edges = o.n_edges; o.n_edges = 0;
        return *this;
    }
};

GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>&    positions,
    const RefMesh&              ref_mesh,
    const std::vector<double>&  per_vertex_radii,
    double                      d_hat);

// Test helpers: D2H copy of the unique key arrays and the edge table.
// Empty vectors on non-CUDA builds.
std::vector<unsigned long long> gpu_broad_phase_nt_keys_to_host(const GpuBroadPhaseResult&);
std::vector<unsigned long long> gpu_broad_phase_ss_keys_to_host(const GpuBroadPhaseResult&);
std::vector<int>                gpu_broad_phase_edges_to_host(const GpuBroadPhaseResult&);

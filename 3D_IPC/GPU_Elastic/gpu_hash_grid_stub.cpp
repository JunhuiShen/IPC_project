// CPU stub for non-CUDA builds. Returns empty result; tests that depend on
// real GPU output should be CUDA-gated. The hash grid is a CUDA-only feature
// in this branch.
#include "gpu_hash_grid.h"

GpuBroadPhaseResult::~GpuBroadPhaseResult() {
    // Nothing to free — d_nt_keys / d_ss_keys are always null on this path.
}

GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>&,
    const RefMesh&,
    const std::vector<double>&,
    double)
{
    return {};
}

std::vector<unsigned long long> gpu_broad_phase_nt_keys_to_host(const GpuBroadPhaseResult&) {
    return {};
}
std::vector<unsigned long long> gpu_broad_phase_ss_keys_to_host(const GpuBroadPhaseResult&) {
    return {};
}
std::vector<int> gpu_broad_phase_edges_to_host(const GpuBroadPhaseResult&) {
    return {};
}

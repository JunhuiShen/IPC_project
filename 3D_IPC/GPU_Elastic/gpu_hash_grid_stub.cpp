// CPU stub: forwards to CPU BroadPhase so non-CUDA builds compile and tests
// trivially pass. Real implementation lives in gpu_hash_grid.cu.
#include "gpu_hash_grid.h"
#include "../broad_phase.h"

GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>& positions,
    const RefMesh&           ref_mesh,
    double                   node_box_size,
    double                   d_hat)
{
    std::vector<AABB> blue_boxes(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        blue_boxes[i] = AABB(positions[i] - Vec3::Constant(node_box_size),
                             positions[i] + Vec3::Constant(node_box_size));
    }
    BroadPhase bp;
    bp.initialize(blue_boxes, ref_mesh, d_hat);

    GpuBroadPhaseResult r;
    r.nt_pairs = bp.cache().nt_pairs;
    r.ss_pairs = bp.cache().ss_pairs;
    return r;
}

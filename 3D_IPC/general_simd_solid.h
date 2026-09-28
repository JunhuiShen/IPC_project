#pragma once

#include "SIMD.h"

namespace ipc_simd {

// One gathered AoS record represents one tetrahedron and its active node.
// positions contains four consecutive vertices per record, and shape_gradients
// contains that node's TetRestData::grad_N. Only these local records are read.
// count must be at most tile_width; zero permits null input/output pointers.
// Outputs include rest measure, but not dt^2. Blocks are the scalar solid
// solver's TGSL PSD PBGS approximation, not exact energy Hessians. The caller
// owns ordered accumulation into its private per-node systems.
void solid_derivatives_tile(
    const Vec3* positions, const Mat33* dm_inverse, const double* measures,
    const Vec3* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* pbgs_blocks);

} // namespace ipc_simd

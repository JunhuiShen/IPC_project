#pragma once

#include "SIMD.h"

namespace ipc_simd {

// The general membrane variant preserves the original scalar/Eigen rounding
// boundaries while retaining independent-element SIMD lanes. Bending uses
// bending_derivatives_tile from SIMD.h, shared with cloth experimental v2.
void general_corotated_derivatives_tile(
    const Vec3* positions, const Mat22* dm_inverse, const double* areas,
    const Vec2* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* hessians);

} // namespace ipc_simd

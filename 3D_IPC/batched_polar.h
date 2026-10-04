#pragma once
#include "IPC_math.h"
#include <cstddef>

namespace volumetric_detail {
// Independent matrices; the signed QR-SVD algorithm and each lane's rounding
// order match the scalar volumetric reference. No iteration tolerance changes.
void batched_signed_polar(const Mat33* inputs, Mat33* rotations, std::size_t count);

// Nine row-major matrix entries each occupy eight doubles; only the first
// count lanes are read/written. Count must be <= 8; zero permits null pointers.
// The signed SVD and rounding order are identical to batched_signed_polar.
void signed_polar_soa_tile(const double* inputs, double* rotations, std::size_t count);
}

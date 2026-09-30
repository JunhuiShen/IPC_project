#pragma once
#include "IPC_math.h"
#include <cstddef>

namespace volumetric_detail {
// Independent matrices; the signed QR-SVD algorithm and each lane's rounding
// order match the scalar volumetric reference. No iteration tolerance changes.
void batched_signed_polar(const Mat33* inputs, Mat33* rotations, std::size_t count);
}

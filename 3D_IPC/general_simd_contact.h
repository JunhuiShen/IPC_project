#pragma once

#include "SIMD.h"

namespace ipc_simd {

// General-solver contact arithmetic follows the original general reference's
// rounding order. The existing cloth-v2 entry point keeps its own arithmetic.
void general_mesh_contact_derivatives_tile(const MeshContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, MeshContactOutput* outputs,
    unsigned char* derivative_active = nullptr);

} // namespace ipc_simd

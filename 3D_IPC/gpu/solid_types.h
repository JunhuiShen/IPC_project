#pragma once

#include "kernels.h"

namespace ipc_gpu {

// One gathered tetrahedron/active-node record. The active node is represented
// by its rest-space shape gradient; no mesh or host pointers enter the kernel.
struct Solid {
    double positions[12]; // four vertex-major xyz positions
    double dm_inverse[9]; // row-major inverse rest edge matrix
    double shape[3];      // TetRestData::grad_N[active_node]
    double measure;
};

struct SolidParameters {
    double mu, lambda;
};

// Outputs include rest measure, but exclude dt^2. hessian is the TGSL PSD
// PBGS elasticity block, not the exact energy Hessian. The complete signed
// polar decomposition and elasticity arithmetic execute on the device.
// Zero count permits null pointers. Errors throw; there is no CPU fallback.
void compute_solids(const Solid*, std::size_t, const SolidParameters&, Derivatives*);

} // namespace ipc_gpu

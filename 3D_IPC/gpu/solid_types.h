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

} // namespace ipc_gpu

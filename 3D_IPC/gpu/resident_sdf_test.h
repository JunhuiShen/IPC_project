#pragma once
#include "resident_sdf.cuh"
#include <cstddef>

struct ResidentSdfTestInput {
    ipc_gpu::resident::SdfCollider colliders[3];
    ipc_gpu::resident::SdfParameters parameters;
    double position[3]{},previous[3]{},jacobian[9]{};
    int count=1;
};
struct ResidentSdfTestOutput {
    ipc_gpu::resident::SdfOutput point;
    ipc_gpu::RigidDerivatives rigid;
};
void resident_sdf_compute_gpu(const ResidentSdfTestInput*,ResidentSdfTestOutput*,std::size_t);

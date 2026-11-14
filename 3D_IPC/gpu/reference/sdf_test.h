#pragma once
#include "sdf.cuh"
#include <cstddef>

struct GpuSdfTestInput {
    ipc_gpu::solver::SdfCollider colliders[3];
    ipc_gpu::solver::SdfParameters parameters;
    double position[3]{},previous[3]{},jacobian[9]{};
    int count=1;
};
struct GpuSdfTestOutput {
    ipc_gpu::solver::SdfOutput point;
    ipc_gpu::RigidDerivatives rigid;
};
void gpu_sdf_compute_gpu(const GpuSdfTestInput*,GpuSdfTestOutput*,std::size_t);

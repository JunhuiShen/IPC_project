#pragma once
#include "contact.cuh"
#include <cstddef>

struct GpuContactRejectionResult {
    unsigned certified=0;
    ipc_gpu::solver::ContactOutput output;
};

void gpu_contact_rejection_gpu(const ipc_gpu::solver::ContactInput* input,
    GpuContactRejectionResult* output,std::size_t count);

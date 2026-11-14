#pragma once
#include "ccd.cuh"
#include "exact.cuh"
#include <cstddef>

// Test-only wrapper. Input/reference management is host-side; every queried
// predicate itself is evaluated in CUDA with no host numerical callback.
void gpu_ccd_cuda_validate(const ipc_gpu::ccd::LinearQuery* queries,
    std::size_t count,ipc_gpu::ccd::Status* certificates,
    ipc_gpu::exact::Status* exact_results);

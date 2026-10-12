#pragma once
#include "resident_ccd.cuh"
#include "resident_exact.cuh"
#include <cstddef>

// Test-only wrapper. Input/reference management is host-side; every queried
// predicate itself is evaluated in CUDA with no host numerical callback.
void resident_ccd_cuda_validate(const ipc_gpu::resident_ccd::LinearQuery* queries,
    std::size_t count,ipc_gpu::resident_ccd::Status* certificates,
    ipc_gpu::resident_exact::Status* exact_results);

#include "original_contact.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void require_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string("CUDA original contact test: ")
                + cudaGetErrorString(status));
    }
}

__global__ void original_contact_records(
        const ipc_gpu::solver::ContactInput* input,
        ipc_gpu::solver::ContactOutput* output,
        std::size_t count) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = ipc_gpu::solver::evaluate_contact_original(input[i]);
}

}  // namespace

// Actual device execution of the isolated legacy component. There is no host
// fallback: the caller receives only values produced by the kernel launch.
// Every record of the host test (including the additive near-parallel
// small-Gram family, its rigid transforms and the diagnostic records) is
// evaluated by this kernel with the production ContactInput/ContactOutput
// types and the component's normal validation; the host side only marshals
// memory and compares the returned values with the real CPU false oracle.
void gpu_original_contact_compute_gpu(
        const ipc_gpu::solver::ContactInput* input,
        ipc_gpu::solver::ContactOutput* output,
        std::size_t count) {
    if (count == 0) return;
    if (!input || !output)
        throw std::invalid_argument(
                "CUDA original contact test: null record pointer");

    ipc_gpu::solver::ContactInput* device_input = nullptr;
    ipc_gpu::solver::ContactOutput* device_output = nullptr;
    try {
        require_cuda(cudaMalloc(&device_input,
                                count * sizeof(*device_input)));
        require_cuda(cudaMalloc(&device_output,
                                count * sizeof(*device_output)));
        require_cuda(cudaMemcpy(device_input, input,
                                count * sizeof(*device_input),
                                cudaMemcpyHostToDevice));
        original_contact_records<<<(count + 63) / 64, 64>>>(
                device_input, device_output, count);
        require_cuda(cudaGetLastError());
        require_cuda(cudaDeviceSynchronize());
        require_cuda(cudaMemcpy(output, device_output,
                                count * sizeof(*device_output),
                                cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_output);
        throw;
    }
    require_cuda(cudaFree(device_input));
    require_cuda(cudaFree(device_output));
}

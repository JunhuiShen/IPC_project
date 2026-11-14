#include "box_certificate_test.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("movement-box certificate CUDA: ") + cudaGetErrorString(status));
}
__global__ void evaluate(const GpuBoxCertificateInput* input,
    GpuBoxCertificateResult* output, std::size_t count) {
    const auto i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i].certified = ipc_gpu::solver::certify_movement_box_contact(
        input[i].boxes, input[i].positions, input[i].vertex_face, input[i].d_hat, &output[i].witness);
}
}

void gpu_box_certificate_gpu(const GpuBoxCertificateInput* input,
    GpuBoxCertificateResult* output, std::size_t count) {
    if (!count) return;
    GpuBoxCertificateInput* device_input = nullptr;
    GpuBoxCertificateResult* device_output = nullptr;
    try {
        checked(cudaMalloc(&device_input, count*sizeof(*device_input)));
        checked(cudaMalloc(&device_output, count*sizeof(*device_output)));
        checked(cudaMemcpy(device_input, input, count*sizeof(*device_input), cudaMemcpyHostToDevice));
        evaluate<<<(count+127)/128,128>>>(device_input, device_output, count);
        checked(cudaGetLastError());
        checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(output, device_output, count*sizeof(*device_output), cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_input); cudaFree(device_output); throw;
    }
    checked(cudaFree(device_input)); checked(cudaFree(device_output));
}

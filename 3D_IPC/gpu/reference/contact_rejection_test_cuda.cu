#include "contact_rejection_test.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
void checked(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(std::string("contact rejection CUDA: ")+cudaGetErrorString(result));
}
__global__ void evaluate(const ipc_gpu::solver::ContactInput* input,
    GpuContactRejectionResult* output,std::size_t count) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) {
        output[i].certified=ipc_gpu::solver::certified_contact_inactive(input[i]);
        output[i].output=ipc_gpu::solver::evaluate_contact(input[i]);
    }
}
}

void gpu_contact_rejection_gpu(const ipc_gpu::solver::ContactInput* input,
    GpuContactRejectionResult* output,std::size_t count) {
    if(!count)return;
    ipc_gpu::solver::ContactInput* device_input=nullptr;
    GpuContactRejectionResult* device_output=nullptr;
    try {
        checked(cudaMalloc(&device_input,count*sizeof(*device_input)));
        checked(cudaMalloc(&device_output,count*sizeof(*device_output)));
        checked(cudaMemcpy(device_input,input,count*sizeof(*device_input),cudaMemcpyHostToDevice));
        evaluate<<<(count+63)/64,64>>>(device_input,device_output,count);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(output,device_output,count*sizeof(*device_output),cudaMemcpyDeviceToHost));
    } catch(...) {cudaFree(device_input);cudaFree(device_output);throw;}
    checked(cudaFree(device_input));checked(cudaFree(device_output));
}

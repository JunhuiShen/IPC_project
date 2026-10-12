#include "resident_sdf_test.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
void checked(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string("resident SDF CUDA: ")+cudaGetErrorString(error));
}
__global__ void sdf_records(const ResidentSdfTestInput* input,ResidentSdfTestOutput* output,std::size_t count) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto& in=input[i];
    output[i].point=ipc_gpu::resident::evaluate_sdf_contact(in.colliders,in.count,in.position,in.previous,in.parameters);
    output[i].rigid=ipc_gpu::resident::sdf_rigid_pullback(output[i].point,in.jacobian,in.parameters.dt*in.parameters.dt);
}
}
void resident_sdf_compute_gpu(const ResidentSdfTestInput* input,ResidentSdfTestOutput* output,std::size_t count) {
    if(count==0)return;
    if(!input||!output)throw std::invalid_argument("resident SDF CUDA: null records");
    ResidentSdfTestInput* device_input=nullptr;ResidentSdfTestOutput* device_output=nullptr;
    try {
        checked(cudaMalloc(&device_input,count*sizeof(*device_input)));
        checked(cudaMalloc(&device_output,count*sizeof(*device_output)));
        checked(cudaMemcpy(device_input,input,count*sizeof(*device_input),cudaMemcpyHostToDevice));
        sdf_records<<<(count+63)/64,64>>>(device_input,device_output,count);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(output,device_output,count*sizeof(*device_output),cudaMemcpyDeviceToHost));
    } catch(...) {cudaFree(device_input);cudaFree(device_output);throw;}
    checked(cudaFree(device_input));checked(cudaFree(device_output));
}

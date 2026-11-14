#include "contact.cuh"
#include "contact_exact.cuh"
#include "contact_selected_ordinary.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <mutex>

namespace {
void require_cuda(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA contact test: ")+cudaGetErrorString(status));
}
__global__ void contact_records(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::ContactOutput* output,std::size_t count) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=ipc_gpu::solver::evaluate_contact(input[i]);
}
__global__ void exact_contact_records(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::ContactOutput* output,std::size_t count,
    std::uint32_t* words,int* free_slots,int* diagnostics) {
    // Keep this reference helper serial; the production parallel queue is
    // checked independently against serial exact arithmetic in its own test.
    if(blockIdx.x!=0||threadIdx.x!=0)return;
    for(std::size_t i=0;i<count;++i) {
        ipc_gpu::exact::DevicePool pool;
        ipc_gpu::exact::initialize_pool(pool,words,free_slots,1600,512);
        output[i]=ipc_gpu::solver::evaluate_contact_exact<256>(input[i]);
        if(output[i].status==ipc_gpu::solver::ContactStatus::ExactCapacityExceeded&&!pool.failed)
            output[i]=ipc_gpu::solver::evaluate_contact_exact<1600>(input[i]);
        if(pool.peak>diagnostics[0])diagnostics[0]=pool.peak;
        if(pool.failed)diagnostics[1]=1;
        if(pool.available!=pool.slot_count)diagnostics[2]=1;
        ipc_gpu::exact::release_pool();
    }
}
__global__ void selected_contact_records(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::selected_ordinary::Output* output,std::size_t count) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        const int role=ipc_gpu::solver::compact_contact_role(input[i].hessian_blocks);
        output[i]=ipc_gpu::solver::selected_ordinary::evaluate_eligible(input[i],role);
    }
}
}

void gpu_contact_compute_selected_gpu(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::selected_ordinary::Output* output,std::size_t count) {
    if(!count)return;
    if(!input||!output)throw std::invalid_argument("CUDA selected contact test: null record pointer");
    ipc_gpu::solver::ContactInput* device_input=nullptr;
    ipc_gpu::solver::selected_ordinary::Output* device_output=nullptr;
    try {
        require_cuda(cudaMalloc(&device_input,count*sizeof(*device_input)));
        require_cuda(cudaMalloc(&device_output,count*sizeof(*device_output)));
        require_cuda(cudaMemcpy(device_input,input,count*sizeof(*device_input),cudaMemcpyHostToDevice));
        selected_contact_records<<<(count+63)/64,64>>>(device_input,device_output,count);
        require_cuda(cudaGetLastError());require_cuda(cudaDeviceSynchronize());
        require_cuda(cudaMemcpy(output,device_output,count*sizeof(*device_output),cudaMemcpyDeviceToHost));
    }catch(...){cudaFree(device_input);cudaFree(device_output);throw;}
    require_cuda(cudaFree(device_input));require_cuda(cudaFree(device_output));
}

void gpu_contact_compute_gpu(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::ContactOutput* output,std::size_t count) {
    if(count==0)return;
    if(!input||!output)throw std::invalid_argument("CUDA contact test: null record pointer");
    ipc_gpu::solver::ContactInput* device_input=nullptr;
    ipc_gpu::solver::ContactOutput* device_output=nullptr;
    try {
        require_cuda(cudaMalloc(&device_input,count*sizeof(*device_input)));
        require_cuda(cudaMalloc(&device_output,count*sizeof(*device_output)));
        require_cuda(cudaMemcpy(device_input,input,count*sizeof(*device_input),cudaMemcpyHostToDevice));
        contact_records<<<(count+63)/64,64>>>(device_input,device_output,count);
        require_cuda(cudaGetLastError());require_cuda(cudaDeviceSynchronize());
        require_cuda(cudaMemcpy(output,device_output,count*sizeof(*device_output),cudaMemcpyDeviceToHost));
    } catch(...) {cudaFree(device_input);cudaFree(device_output);throw;}
    require_cuda(cudaFree(device_input));require_cuda(cudaFree(device_output));
}

void gpu_contact_compute_exact_gpu(const ipc_gpu::solver::ContactInput* input,
    ipc_gpu::solver::ContactOutput* output,std::size_t count) {
    if(count==0)return;
    if(!input||!output)throw std::invalid_argument("CUDA exact contact test: null record pointer");
    // Preserve the deliberately serialized reference-test helper. Production
    // exact evaluation now has isolated per-worker allocator contexts.
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    ipc_gpu::solver::ContactInput* device_input=nullptr;
    ipc_gpu::solver::ContactOutput* device_output=nullptr;
    std::uint32_t* words=nullptr;int* free_slots=nullptr;int* diagnostics=nullptr;
    try {
        require_cuda(cudaMalloc(&device_input,count*sizeof(*device_input)));
        require_cuda(cudaMalloc(&device_output,count*sizeof(*device_output)));
        require_cuda(cudaMalloc(&words,1600*512*sizeof(std::uint32_t)));
        require_cuda(cudaMalloc(&free_slots,512*sizeof(int)));
        require_cuda(cudaMalloc(&diagnostics,3*sizeof(int)));
        require_cuda(cudaMemset(diagnostics,0,3*sizeof(int)));
        require_cuda(cudaMemcpy(device_input,input,count*sizeof(*device_input),cudaMemcpyHostToDevice));
        exact_contact_records<<<1,1>>>(device_input,device_output,count,words,free_slots,diagnostics);
        require_cuda(cudaGetLastError());require_cuda(cudaDeviceSynchronize());
        require_cuda(cudaMemcpy(output,device_output,count*sizeof(*device_output),cudaMemcpyDeviceToHost));
        int checked[3]{};require_cuda(cudaMemcpy(checked,diagnostics,sizeof(checked),cudaMemcpyDeviceToHost));
        if(checked[1]||checked[2])throw std::runtime_error("CUDA exact contact arena exhaustion/leak (peak "+std::to_string(checked[0])+")");
    } catch(...) {cudaFree(device_input);cudaFree(device_output);cudaFree(words);cudaFree(free_slots);cudaFree(diagnostics);throw;}
    require_cuda(cudaFree(device_input));require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(words));require_cuda(cudaFree(free_slots));require_cuda(cudaFree(diagnostics));
}

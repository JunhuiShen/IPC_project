#include "solve.cuh"
#include "test_support.h"
#include <array>
struct SolveRecord {double h[9],g[3];};
struct SolveResult {double x[3];bool valid;};
__global__ void run_solves(const SolveRecord* in,SolveResult* out,std::size_t count) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)out[i].valid=ipc_gpu::solver::solve3(in[i].h,in[i].g,out[i].x);
}
void gpu_test_solves(const SolveRecord* inputs,SolveResult* outputs,std::size_t count) {
    if(!count)return;
    GpuTestBuffer<SolveRecord> in(std::vector<SolveRecord>(inputs,inputs+count));
    GpuTestBuffer<SolveResult> out(count);
    run_solves<<<(count+127)/128,128>>>(in.data(),out.data(),count);
    gpu_cuda_check(cudaGetLastError());const auto result=out.download();
    for(std::size_t i=0;i<count;++i)outputs[i]=result[i];
}

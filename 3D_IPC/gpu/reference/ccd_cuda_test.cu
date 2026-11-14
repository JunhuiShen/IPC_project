#include "ccd_cuda_test.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
namespace cert=ipc_gpu::ccd;
namespace exact = ipc_gpu::exact;

__device__ bool allocator_self_test(std::uint32_t*words,int*free_slots) {
    exact::DevicePool pool;
    exact::initialize_pool(pool,words,free_slots,128,4);
    bool ok=true;
    {
        exact::Integer<128> a(7),b(a),c(-9),d;
        a=a;c=b;d=c;
        b.limb[0]=99;
        ok=ok&&a.valid&&b.valid&&c.valid&&d.valid
            &&a.limb[0]==7&&b.limb[0]==99&&c.limb[0]==7&&d.limb[0]==7
            &&pool.available==0&&pool.peak==4;
    }
    ok=ok&&pool.available==4&&!pool.failed;
    {
        exact::Integer<128> a(1),b(2),c(3),d(4),overflow(5);
        ok=ok&&a.valid&&b.valid&&c.valid&&d.valid&&!overflow.valid&&pool.failed;
    }
    ok=ok&&pool.available==4;
    exact::release_pool();
    return ok;
}

__global__ void device_certificates(const cert::LinearQuery*queries,
    cert::Status*certified,exact::Status*precise,std::size_t count,
    std::uint32_t*words,int*free_slots) {
    if(blockIdx.x||threadIdx.x) return;
    if(!allocator_self_test(words,free_slots)) {
        for(std::size_t i=0;i<count;++i) { certified[i]=cert::Status::Invalid;precise[i]=exact::Status::Invalid; }
        return;
    }
    exact::DevicePool pool;
    exact::initialize_pool(pool,words,free_slots,2048,512);
    for(std::size_t i=0;i<count;++i) {
        certified[i]=cert::validate_linear(queries[i]);
        precise[i]=exact::validate_step<128>(queries[i].start,queries[i].end,
            queries[i].vertex_face,queries[i].gap_floor);
        if(pool.failed||pool.available!=pool.slot_count)
            precise[i]=exact::Status::CapacityExceeded;
    }
    exact::release_pool();
}
__global__ void device_full_range_retry(const cert::LinearQuery*queries,
    exact::Status*precise,std::size_t count,std::uint32_t*words,int*free_slots) {
    // Cold queue is deliberately one thread: maximum-range integer temporaries
    // must not be replicated across a full GPU-sized launch. Results stay on
    // device between the fast and full-range kernels.
    if(blockIdx.x||threadIdx.x) return;
    exact::DevicePool pool;
    exact::initialize_pool(pool,words,free_slots,2048,512);
    for(std::size_t i=0;i<count;++i)
        if(precise[i]==exact::Status::CapacityExceeded) {
            precise[i]=exact::validate_step<2048>(queries[i].start,queries[i].end,
                queries[i].vertex_face,queries[i].gap_floor);
            if(pool.failed||pool.available!=pool.slot_count)
                precise[i]=exact::Status::CapacityExceeded;
        }
    exact::release_pool();
}
void check(cudaError_t error) {
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct DeviceBuffer {
    T* pointer=nullptr;
    explicit DeviceBuffer(std::size_t n) { check(cudaMalloc(&pointer,n*sizeof(T))); }
    ~DeviceBuffer() { if(pointer) cudaFree(pointer); }
};
}

void gpu_ccd_cuda_validate(const cert::LinearQuery*queries,std::size_t count,
    cert::Status*certificates,exact::Status*exact_results) {
    if(!count)return;
    DeviceBuffer<cert::LinearQuery> input(count);
    DeviceBuffer<cert::Status> certificate(count);
    DeviceBuffer<exact::Status> precise(count);
    DeviceBuffer<std::uint32_t> scratch(2048*512);
    DeviceBuffer<int> free_slots(512);
    check(cudaMemcpy(input.pointer,queries,count*sizeof(*queries),cudaMemcpyHostToDevice));
    device_certificates<<<1,1>>>(input.pointer,certificate.pointer,precise.pointer,count,
        scratch.pointer,free_slots.pointer);
    check(cudaGetLastError());
    device_full_range_retry<<<1,1>>>(input.pointer,precise.pointer,count,scratch.pointer,free_slots.pointer);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(certificates,certificate.pointer,count*sizeof(*certificates),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(exact_results,precise.pointer,count*sizeof(*exact_results),cudaMemcpyDeviceToHost));
}

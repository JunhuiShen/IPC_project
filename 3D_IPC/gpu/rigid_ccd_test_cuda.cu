#include "rigid_ccd.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

using namespace ipc_gpu::solver;
static void cuda_check(cudaError_t error, const char* label) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(error));
}
template<class T> struct TestBuffer {
    T* p = nullptr;
    explicit TestBuffer(std::size_t count) {
        if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"rotation test allocation");
    }
    ~TestBuffer() { if (p) cudaFree(p); }
};

// Parameters: alpha, dt, bound, X[3], anchor[4]. Values: interpolated q[4],
// bounded q[4], recovered omega[3], cap lo[3], cap hi[3]. All preparation and
// geometry run in this CUDA kernel; host transfers are only test I/O.
__global__ void rotation_test_kernel(const RigidRotationQuery* queries,
    const double* parameters, std::size_t count, RigidRotationResult* results,
    double* values, RigidRotationStatus* statuses) {
    for (std::size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count;
         i += blockDim.x*gridDim.x) {
        const auto& q = queries[i]; const double* p = parameters+10*i;
        double* v = values+17*i; RigidRotationStatus* s = statuses+4*i;
        for (int k = 0; k < 17; ++k) v[k] = 0.0;
        results[i] = rigid_rotation_ccd(q);
        s[0] = interpolate_rigid_orientation(q.current,q.target,p[0],v);
        s[1] = bound_rigid_orientation(p+6,q.current,q.target,p[2],v+4);
        s[2] = rigid_angular_velocity_from_orientation(q.target,q.current,p[1],v+8);
        BpAabb box{};
        s[3] = rigid_spherical_cap_box(q.com,q.current,p+3,p[2],box);
        for (int k = 0; k < 3; ++k) { v[11+k] = box.lo[k]; v[14+k] = box.hi[k]; }
    }
}
void gpu_rigid_ccd_test_cuda(const RigidRotationQuery* queries, const double* parameters,
    std::size_t count, RigidRotationResult* results, double* values, RigidRotationStatus* statuses) {
    if (!count) return;
    TestBuffer<RigidRotationQuery> input(count);
    TestBuffer<double> params(10*count), output(17*count);
    TestBuffer<RigidRotationResult> result(count);
    TestBuffer<RigidRotationStatus> status(4*count);
    cuda_check(cudaMemcpy(input.p,queries,count*sizeof(RigidRotationQuery),cudaMemcpyHostToDevice),"rotation queries upload");
    cuda_check(cudaMemcpy(params.p,parameters,10*count*sizeof(double),cudaMemcpyHostToDevice),"rotation params upload");
    rotation_test_kernel<<<static_cast<unsigned>((count+127)/128),128>>>(input.p,params.p,count,result.p,output.p,status.p);
    cuda_check(cudaGetLastError(),"rotation test kernel");
    cuda_check(cudaMemcpy(results,result.p,count*sizeof(RigidRotationResult),cudaMemcpyDeviceToHost),"rotation results download");
    cuda_check(cudaMemcpy(values,output.p,17*count*sizeof(double),cudaMemcpyDeviceToHost),"rotation values download");
    cuda_check(cudaMemcpy(statuses,status.p,4*count*sizeof(RigidRotationStatus),cudaMemcpyDeviceToHost),"rotation status download");
}

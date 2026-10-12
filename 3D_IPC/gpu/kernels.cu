#include "kernels.h"
#include "cloth_math.cuh"
#include "solid_math.cuh"
#include "rigid_math.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ipc_gpu {
static void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + operation + ": " + cudaGetErrorString(result));
}
template<class T> struct Buffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~Buffer() { if (data) cudaFree(data); }
    void reserve(std::size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw std::overflow_error("CUDA allocation size overflow");
        T* next = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&next), count*sizeof(T)), "allocate");
        if (data) cudaFree(data);
        data = next; capacity = count;
    }
};
struct Workspace {
    Buffer<Point> points;
    Buffer<Triangle> triangles;
    Buffer<Hinge> hinges;
    Buffer<Solid> solids;
    Buffer<RigidPreparedContact> rigid_contacts;
    Buffer<RigidInertia> rigid_inertia;
    Buffer<Derivatives> point_out, triangle_out, hinge_out;
    Buffer<Derivatives> solid_out, rigid_inertia_out;
    Buffer<RigidResult> rigid_out;
    Buffer<int> invalid;
    Statistics stats;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[4]{};
    std::string name;
    std::mutex mutex;
    Workspace() {
        cudaDeviceProp properties{};
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        name = properties.name;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        for (auto& event : events) check(cudaEventCreate(&event), "create event");
        invalid.reserve(1);
    }
    ~Workspace() {
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }
};
static Workspace& workspace() { static Workspace w; return w; }

// The initial mapping is one CUDA thread per gathered AoS record. No shared
// memory, global SoA copy, atomics to vertices, or changes to GS coloring.
// Consecutive threads access strided fields of adjacent records; this is NOT
// claimed to be optimally coalesced. Profile before adding a tile transpose.
struct NoParameters {};
__device__ static RigidResult evaluate(const RigidPreparedContact& x, NoParameters) { return evaluate_rigid(x); }
__device__ static Derivatives evaluate(const RigidInertia& x, NoParameters) { return evaluate_rigid_inertia(x); }
__device__ static bool finite_output(const Derivatives& value) {
    bool valid=true;
    for (int i=0; i<3; ++i) valid=valid && isfinite(value.gradient[i]);
    for (int i=0; i<9; ++i) valid=valid && isfinite(value.hessian[i]);
    return valid;
}
__device__ static bool finite_rigid_derivatives(const RigidDerivatives& x) {
    bool valid=true;
    for (int i=0; i<3; ++i) valid=valid && isfinite(x.translation_gradient[i]) && isfinite(x.orientation_gradient[i]);
    for (int i=0; i<9; ++i) valid=valid && isfinite(x.translation_translation_hessian[i])
        && isfinite(x.translation_orientation_hessian[i]) && isfinite(x.orientation_orientation_hessian[i]);
    return valid;
}
__device__ static bool finite_output(const RigidResult& x) {
    return finite_rigid_derivatives(x.barrier) && finite_rigid_derivatives(x.friction);
}
template<class Input,class Params,class Output>
__global__ void evaluate_records(const Input* inputs, std::size_t count,
                                Params parameters, Output* outputs,
                                int* invalid) {
    for (std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         index < count; index += static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const auto value = evaluate(inputs[index], parameters);
        if (!finite_output(value)) atomicExch(invalid, 1);
        outputs[index] = value;
    }
}
template<class T> static void upload(Buffer<T>& buffer, const T* input,
                                    std::size_t count, cudaStream_t stream) {
    if (!count) return;
    if (!input) throw std::invalid_argument("CUDA input pointer is null");
    buffer.reserve(count);
    check(cudaMemcpyAsync(buffer.data, input, count*sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
}
template<class T,class Params,class Output> static void launch(const Buffer<T>& input, std::size_t count,
        const Params& params, Buffer<Output>& output, Workspace& w) {
    if (!count) return;
    constexpr unsigned block = 128;
    const auto grid = static_cast<unsigned>(std::min<std::size_t>((count+block-1)/block, 65535));
    evaluate_records<<<grid,block,0,w.stream>>>(input.data,count,params,output.data,w.invalid.data);
    check(cudaGetLastError(), "launch derivatives");
}
template<class Output> static void download(const Buffer<Output>& source, Output* target,
                     std::size_t count, cudaStream_t stream) {
    if (count) check(cudaMemcpyAsync(target, source.data, count*sizeof(Output),
                                   cudaMemcpyDeviceToHost, stream), "download");
}
void compute(const Point* p, std::size_t np, const Triangle* t, std::size_t nt,
             const Hinge* h, std::size_t nh, const Parameters& params,
             Derivatives* po, Derivatives* to, Derivatives* ho) {
    if (!(np || nt || nh)) return;
    if ((np && (!p || !po)) || (nt && (!t || !to)) || (nh && (!h || !ho)))
        throw std::invalid_argument("CUDA nonempty input/output pointer is null");
    auto& w = workspace();
    std::lock_guard<std::mutex> lock(w.mutex);
    try {
        w.point_out.reserve(np); w.triangle_out.reserve(nt); w.hinge_out.reserve(nh);
        check(cudaEventRecord(w.events[0],w.stream), "upload start");
        upload(w.points,p,np,w.stream); upload(w.triangles,t,nt,w.stream); upload(w.hinges,h,nh,w.stream);
        check(cudaMemsetAsync(w.invalid.data,0,sizeof(int),w.stream), "clear invalid flag");
        check(cudaEventRecord(w.events[1],w.stream), "compute start");
        launch(w.points,np,params,w.point_out,w);
        launch(w.triangles,nt,params,w.triangle_out,w);
        launch(w.hinges,nh,params,w.hinge_out,w);
        check(cudaEventRecord(w.events[2],w.stream), "download start");
        download(w.point_out,po,np,w.stream); download(w.triangle_out,to,nt,w.stream); download(w.hinge_out,ho,nh,w.stream);
        int invalid = 0;
        check(cudaMemcpyAsync(&invalid,w.invalid.data,sizeof(int),cudaMemcpyDeviceToHost,w.stream), "invalid flag");
        check(cudaEventRecord(w.events[3],w.stream), "download end");
        check(cudaStreamSynchronize(w.stream), "synchronize results");
        if (invalid) throw std::runtime_error("CUDA derivatives are nonfinite; no vertex updates committed for this color");
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed,w.events[0],w.events[1]), "upload timer"); w.stats.host_to_device_ms += elapsed;
        check(cudaEventElapsedTime(&elapsed,w.events[1],w.events[2]), "compute timer"); w.stats.kernel_ms += elapsed;
        check(cudaEventElapsedTime(&elapsed,w.events[2],w.events[3]), "download timer"); w.stats.device_to_host_ms += elapsed;
        ++w.stats.calls;
        w.stats.launches += (np>0)+(nt>0)+(nh>0);
        w.stats.points += np; w.stats.triangles += nt; w.stats.hinges += nh;
    } catch (...) {
        cudaStreamSynchronize(w.stream);
        throw;
    }
}
template<class Input,class Params,class Output>
static void compute_family(const Input* input,std::size_t count,const Params& params,Output* output,
                           Buffer<Input>& input_buffer,Buffer<Output>& output_buffer,
                           std::uint64_t& records,Workspace& w) {
    if (!count) return;
    if (!input || !output) throw std::invalid_argument("CUDA nonempty input/output pointer is null");
    std::lock_guard<std::mutex> lock(w.mutex);
    try {
        output_buffer.reserve(count);
        check(cudaEventRecord(w.events[0],w.stream),"upload start");
        upload(input_buffer,input,count,w.stream);
        check(cudaMemsetAsync(w.invalid.data,0,sizeof(int),w.stream),"clear invalid flag");
        check(cudaEventRecord(w.events[1],w.stream),"compute start");
        launch(input_buffer,count,params,output_buffer,w);
        check(cudaEventRecord(w.events[2],w.stream),"download start");
        download(output_buffer,output,count,w.stream);
        int invalid=0;
        check(cudaMemcpyAsync(&invalid,w.invalid.data,sizeof(int),cudaMemcpyDeviceToHost,w.stream),"invalid flag");
        check(cudaEventRecord(w.events[3],w.stream),"download end");
        check(cudaStreamSynchronize(w.stream),"synchronize results");
        if (invalid) throw std::runtime_error("CUDA solid/rigid derivatives are nonfinite; outputs rejected");
        float elapsed=0;
        check(cudaEventElapsedTime(&elapsed,w.events[0],w.events[1]),"upload timer"); w.stats.host_to_device_ms+=elapsed;
        check(cudaEventElapsedTime(&elapsed,w.events[1],w.events[2]),"compute timer"); w.stats.kernel_ms+=elapsed;
        check(cudaEventElapsedTime(&elapsed,w.events[2],w.events[3]),"download timer"); w.stats.device_to_host_ms+=elapsed;
        ++w.stats.calls; ++w.stats.launches; records+=count;
    } catch (...) { cudaStreamSynchronize(w.stream); throw; }
}
void compute_solids(const Solid* input,std::size_t count,const SolidParameters& params,Derivatives* output) {
    if (!count) return;
    auto& w=workspace();
    compute_family(input,count,params,output,w.solids,w.solid_out,w.stats.solids,w);
}
void compute_rigid(const RigidPreparedContact* input,std::size_t count,RigidResult* output) {
    if (!count) return;
    auto& w=workspace();
    compute_family(input,count,NoParameters{},output,w.rigid_contacts,w.rigid_out,w.stats.rigid_contacts,w);
}
void compute_rigid_inertia(const RigidInertia* input,std::size_t count,Derivatives* output) {
    if (!count) return;
    auto& w=workspace();
    compute_family(input,count,NoParameters{},output,w.rigid_inertia,w.rigid_inertia_out,w.stats.rigid_inertia,w);
}
const char* device_name() { return workspace().name.c_str(); }
Statistics statistics() { auto& w=workspace(); std::lock_guard<std::mutex> lock(w.mutex); return w.stats; }
void reset_statistics() { auto& w=workspace(); std::lock_guard<std::mutex> lock(w.mutex); w.stats={}; }
} // namespace ipc_gpu

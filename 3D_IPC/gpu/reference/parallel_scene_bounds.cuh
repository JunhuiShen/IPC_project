#pragma once
// Parallel BVH bounds reduction; include after empty_box()/join().
// Normal finite inputs have unique ordered extrema (equal normal values have
// equal bits). For zeros/subnormals/infinities/NaNs use the original serial
// expression and order, including its NaN and signed-zero behavior.
#if defined(__CUDACC__)
#define IPC_SCENE_BOUNDS_DEVICE __device__
#else
#define IPC_SCENE_BOUNDS_DEVICE
#endif

// Explicit reserved word keeps complete struct copies initialized; it replaces
// ABI padding and has no numerical meaning.
struct SceneBoundsPartial { BpAabb box;unsigned exceptional,reserved; };
static_assert(sizeof(SceneBoundsPartial)==sizeof(BpAabb)+2*sizeof(unsigned));

IPC_SCENE_BOUNDS_DEVICE inline bool scene_bounds_exceptional(double value) {
    std::uint64_t bits;
#if defined(__CUDA_ARCH__)
    bits=static_cast<std::uint64_t>(__double_as_longlong(value));
#else
    std::memcpy(&bits,&value,sizeof(bits));
#endif
    const auto exponent=(bits>>52)&0x7ffu;
    return exponent==0||exponent==0x7ffu;
}

IPC_SCENE_BOUNDS_DEVICE inline SceneBoundsPartial scene_bounds_partial(
    SceneBoundsPartial a,const SceneBoundsPartial& b) {
    a.box=join(a.box,b.box);a.exceptional|=b.exceptional;return a;
}

IPC_SCENE_BOUNDS_DEVICE inline SceneBoundsPartial scene_bounds_add(
    SceneBoundsPartial a,const BpAabb& box) {
    for(int axis=0;axis<3;++axis)
        a.exceptional|=scene_bounds_exceptional(box.lo[axis])||scene_bounds_exceptional(box.hi[axis]);
    a.box=join(a.box,box);return a;
}

#if defined(__CUDACC__)
__global__ void scene_bounds_partials(const BpAabb* boxes,int count,SceneBoundsPartial* output) {
    __shared__ SceneBoundsPartial values[128];
    const unsigned lane=threadIdx.x;
    SceneBoundsPartial value{empty_box(),0,0};
    const auto stride=std::size_t(blockDim.x)*gridDim.x;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+lane;i<std::size_t(count);i+=stride)
        value=scene_bounds_add(value,boxes[i]);
    values[lane]=value;
    __syncthreads();
    for(unsigned width=64;width;width/=2) {
        if(lane<width)values[lane]=scene_bounds_partial(values[lane],values[lane+width]);
        __syncthreads();
    }
    if(lane==0)output[blockIdx.x]=values[0];
}

__global__ void scene_bounds_finish(const BpAabb* boxes,int count,
    const SceneBoundsPartial* partials,int partial_count,BpAabb* result) {
    __shared__ SceneBoundsPartial values[128];
    const unsigned lane=threadIdx.x;
    SceneBoundsPartial value{empty_box(),0,0};
    for(int i=int(lane);i<partial_count;i+=128)value=scene_bounds_partial(value,partials[i]);
    values[lane]=value;
    __syncthreads();
    for(unsigned width=64;width;width/=2) {
        if(lane<width)values[lane]=scene_bounds_partial(values[lane],values[lane+width]);
        __syncthreads();
    }
    if(lane==0) {
        if(values[0].exceptional) {
            // Exact original scene_bounds body: no altered identity, grouping,
            // tolerance, exceptional-value policy or source order is used.
            BpAabb total = empty_box();
            for (int i = 0; i < count; ++i) total = join(total, boxes[i]);
            *result = total;
        } else *result=values[0].box;
    }
}

inline bool parallel_scene_bounds_option() {
    const char* value=std::getenv("IPC_GPU_PARALLEL_SCENE_BOUNDS");
    if(!value)return true;
    if(std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::invalid_argument("IPC_GPU_PARALLEL_SCENE_BOUNDS must be 0 or 1");
}
#endif
#undef IPC_SCENE_BOUNDS_DEVICE

#include "dat.h"
#include "dat.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ipc_gpu::solver {
namespace {
constexpr unsigned threads=128;
constexpr int exact_slots=512;
constexpr int small_limbs=256,large_limbs=1600;
constexpr unsigned max_small_workers=64,max_large_workers=2;
constexpr unsigned small_worker_threads=8;
void checked(cudaError_t status,const char* what) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU DAT ")+what+": "+cudaGetErrorString(status));
}
unsigned blocks(std::size_t count) { return static_cast<unsigned>((count+threads-1)/threads); }
template<class T> struct Buffer {
    T* data=nullptr;std::size_t capacity=0;
    ~Buffer(){if(data)cudaFree(data);}
    void reserve(std::size_t n) {
        if(n<=capacity)return;
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("GPU DAT allocation extent");
        T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),n*sizeof(T)),"allocate");
        if(data)checked(cudaFree(data),"release old allocation");
        data=replacement;capacity=n;
    }
};
using V=dat_detail::Vec;
static_assert(sizeof(V)==3*sizeof(double));
__device__ void fail(int* status,int value){atomicCAS(status,0,value);}
__device__ int failed(int* status){return atomicAdd(status,0);}
__device__ void atomic_min_factor(double* address,double value) {
    auto* bits=reinterpret_cast<unsigned long long*>(address);
    auto previous=atomicCAS(bits,0ULL,0ULL);
    while(value<__longlong_as_double(static_cast<long long>(previous))) {
        const auto expected=previous;
        previous=atomicCAS(bits,expected,static_cast<unsigned long long>(__double_as_longlong(value)));
        if(previous==expected)break;
    }
}
__global__ void prepare_kernel(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],double (*capped)[3],double* factors,BpAabb* boxes,
    double limit,double radius,int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    const V x=dat_detail::load(positions[node]),update=dat_detail::load(raw[node]);
    if(!dat_detail::finite(x)||!dat_detail::finite(update)){fail(status,DatInvalidInput);return;}
    const V value=dat_detail::capped_update(update,limit);
    for(int k=0;k<3;++k) {
        const double lo=x.v[k]-radius,hi=x.v[k]+radius;
        if(!dat_detail::finite(lo)||!dat_detail::finite(hi)){fail(status,DatInvalidInput);return;}
        capped[node][k]=value.v[k];boxes[node].lo[k]=lo;boxes[node].hi[k]=hi;
    }
    factors[node]=1;
}
struct Work {
    std::size_t nodes,pairs;
    const double (*positions)[3];
    const double (*capped)[3];
    const BpContact* contacts;
    double* factors;
    unsigned *exact_count,*retry_count;
    int *exact_queue,*retry_queue;
    std::uint32_t* exact_words;
    int* exact_free_slots;
    DatParameters parameters;
};
__device__ bool gather(const Work& w,std::size_t index,V positions[4],V updates[4],int* status) {
    const auto& pair=w.contacts[index];
    if(pair.kind!=0&&pair.kind!=1){fail(status,DatInvalidInput);return false;}
    for(int i=0;i<4;++i) {
        const int node=pair.vertices[i];
        if(node<0||std::size_t(node)>=w.nodes){fail(status,DatInvalidInput);return false;}
        for(int j=0;j<i;++j)if(node==pair.vertices[j]){fail(status,DatInvalidInput);return false;}
        positions[i]=dat_detail::load(w.positions[node]);updates[i]=dat_detail::load(w.capped[node]);
    }
    return true;
}
__device__ void constrain(const Work& w,std::size_t index,const V x[4],const V update[4],V separation,int* status) {
    double factors[4];
    const int code=dat_detail::plane_factors(x,update,w.contacts[index].kind==0?1:2,separation,w.parameters,factors);
    if(code){fail(status,code);return;}
    for(int i=0;i<4;++i)atomic_min_factor(w.factors+w.contacts[index].vertices[i],factors[i]);
}
__global__ void ordinary_pairs(Work work,int* status) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=work.pairs||failed(status))return;
    V positions[4],updates[4],separation;
    if(!gather(work,index,positions,updates,status))return;
    const int code=work.contacts[index].kind==0
        ?dat_detail::vertex_triangle(positions,separation):dat_detail::edge_edge(positions,separation);
    if(code==dat_detail::NeedsExact) {
        const unsigned next=atomicAdd(work.exact_count,1u);work.exact_queue[next]=static_cast<int>(index);
    } else if(code)fail(status,code);
    else constrain(work,index,positions,updates,separation,status);
}
template<int Limbs,bool Retry> __global__ void exact_pairs(Work work,unsigned workers,int* status) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=workers||failed(status))return;
    const unsigned count=Retry?*work.retry_count:*work.exact_count;
    const int* queue=Retry?work.retry_queue:work.exact_queue;
    for(unsigned ordinal=worker;ordinal<count;ordinal+=workers) {
        if(failed(status))return;
        const int index=queue[ordinal];V positions[4],updates[4],separation;
        if(!gather(work,std::size_t(index),positions,updates,status))return;
        exact::DevicePool pool;
        exact::initialize_pool(pool,work.exact_words+std::size_t(worker)*Limbs*exact_slots,
            work.exact_free_slots+std::size_t(worker)*exact_slots,Limbs,exact_slots);
        const int code=dat_detail::exact_edge_edge<Limbs>(positions,separation);
        const bool arena_failed=pool.failed||pool.available!=pool.slot_count;
        exact::release_pool();
        if(arena_failed){fail(status,DatExactCapacity);return;}
        if(code==DatExactCapacity&&!Retry) {
            const unsigned next=atomicAdd(work.retry_count,1u);work.retry_queue[next]=index;
        } else if(code){fail(status,code);return;}
        else constrain(work,std::size_t(index),positions,updates,separation,status);
    }
}
__global__ void finish_kernel(std::size_t count,const double (*capped)[3],const double* factors,
    double (*accepted)[3],int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    for(int k=0;k<3;++k) {
        const double value=capped[node][k]*factors[node];
        if(!dat_detail::finite(value)){fail(status,DatNonfiniteOutput);return;}
        accepted[node][k]=value;
    }
}
} // namespace

struct GpuDatFilter::Impl {
    Buffer<V> capped;
    Buffer<double> factors;
    Buffer<BpAabb> boxes;
    Buffer<int> exact_queue,retry_queue,exact_free_slots;
    Buffer<unsigned> exact_count,retry_count;
    Buffer<std::uint32_t> exact_words;
    std::size_t count=0;
    const double (*positions)[3]=nullptr;
    const double (*raw)[3]=nullptr;
    int* error=nullptr;
    DatParameters parameters;
    cudaStream_t stream=nullptr;
    bool ready=false;
};

GpuDatFilter::GpuDatFilter():impl_(new Impl){}
GpuDatFilter::~GpuDatFilter()=default;

const BpAabb* GpuDatFilter::prepare(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],const DatParameters& parameters,int* error,cudaStream_t stream) {
    auto& p=*impl_;p.ready=false;
    const double limit=(0.5*parameters.gamma)*parameters.query_radius;
    const double radius=parameters.query_radius*parameters.node_box_padding;
    if(!std::isfinite(parameters.gamma)||parameters.gamma<=0||parameters.gamma>=1
       ||!std::isfinite(parameters.query_radius)||parameters.query_radius<=0
       ||!std::isfinite(parameters.node_box_padding)||parameters.node_box_padding<1
       ||!(limit>0)||!std::isfinite(radius))throw std::invalid_argument("GPU DAT requires 0<gamma<1, positive finite radius/limit and finite padding>=1");
    if(count>INT_MAX||!error||(count&&(!positions||!raw)))throw std::invalid_argument("GPU DAT invalid node extent/pointer");
    p.count=count;p.positions=positions;p.raw=raw;p.error=error;p.parameters=parameters;p.stream=stream;
    p.capped.reserve(count);p.factors.reserve(count);p.boxes.reserve(count);
    if(count)prepare_kernel<<<blocks(count),threads,0,stream>>>(count,positions,raw,
        reinterpret_cast<double(*)[3]>(p.capped.data),p.factors.data,p.boxes.data,limit,radius,error);
    checked(cudaGetLastError(),"prepare launch");p.ready=true;return p.boxes.data;
}

void GpuDatFilter::filter(const BroadPhaseView& broad,double (*accepted)[3],int* error,cudaStream_t stream) {
    auto& p=*impl_;
    const auto overlaps=[&](const void* source) {
        const auto a=reinterpret_cast<std::uintptr_t>(accepted),b=reinterpret_cast<std::uintptr_t>(source);
        const auto bytes=p.count*sizeof(V);
        return p.count&&(a<=b?b-a<bytes:a-b<bytes);
    };
    if(!p.ready||error!=p.error||stream!=p.stream||(p.count&&!accepted)
       ||overlaps(p.positions)||overlaps(p.raw)
       ||broad.contact_count>INT_MAX||(broad.contact_count&&!broad.contacts))
        throw std::invalid_argument("GPU DAT requires one matching prepared view/stream and valid candidates");
    p.ready=false;
    const auto pairs=broad.contact_count;
    if(pairs) {
        p.exact_queue.reserve(pairs);p.retry_queue.reserve(pairs);p.exact_count.reserve(1);p.retry_count.reserve(1);
        const auto small=std::min<std::size_t>(max_small_workers,pairs),large=std::min<std::size_t>(max_large_workers,pairs);
        p.exact_words.reserve(std::max(small*small_limbs,large*large_limbs)*exact_slots);
        p.exact_free_slots.reserve(small*exact_slots);
        checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned),stream),"clear exact queue");
        checked(cudaMemsetAsync(p.retry_count.data,0,sizeof(unsigned),stream),"clear retry queue");
        const Work work{p.count,pairs,p.positions,reinterpret_cast<const double(*)[3]>(p.capped.data),
            broad.contacts,p.factors.data,p.exact_count.data,p.retry_count.data,p.exact_queue.data,p.retry_queue.data,
            p.exact_words.data,p.exact_free_slots.data,p.parameters};
        ordinary_pairs<<<blocks(pairs),threads,0,stream>>>(work,error);
        exact_pairs<small_limbs,false><<<static_cast<unsigned>((small+small_worker_threads-1)/small_worker_threads),small_worker_threads,0,stream>>>(work,static_cast<unsigned>(small),error);
        exact_pairs<large_limbs,true><<<static_cast<unsigned>(large),1,0,stream>>>(work,static_cast<unsigned>(large),error);
    }
    if(p.count)finish_kernel<<<blocks(p.count),threads,0,stream>>>(p.count,
        reinterpret_cast<const double(*)[3]>(p.capped.data),p.factors.data,accepted,error);
    checked(cudaGetLastError(),"filter launch");
}

} // namespace ipc_gpu::solver

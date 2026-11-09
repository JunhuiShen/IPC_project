#include <cuda_runtime.h>
#include "exact.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Standalone allocator/concurrency regression: no CPU numerical solver or
// numerical reference fallback is linked. The host checks explicit identities
// and known exact results produced independently by each CUDA worker.
namespace {
namespace exact = ipc_gpu::exact;
constexpr int capacity=1600;
constexpr int slots_per_worker=64;

enum Failure : unsigned {
    Context=1u<<0, Copies=1u<<1, NarrowCapacity=1u<<2,
    FullCapacity=1u<<3, Restored=1u<<4, Exhaustion=1u<<5,
    Recovered=1u<<6, InvalidInstall=1u<<7, Released=1u<<8,
    FreeList=1u<<9, LiveOwnership=1u<<10, ArithmeticValues=1u<<11,
    ArithmeticFailure=1u<<12
};

// Cross a non-inlined call boundary too: the accessor must refer to the same
// shared symbol as initialize_pool, Integer's constructor, and verdict.
__device__ __noinline__ bool context_is(exact::DevicePool* expected) {
    exact::Arithmetic<capacity> arithmetic;
    return exact::current_pool()==expected
        &&exact::verdict(arithmetic,exact::Status::Safe)==exact::Status::Safe;
}

// Balanced counts alone miss a live slot that was prematurely returned.
__device__ bool unique_free_slots(const exact::DevicePool& pool) {
    if(!pool.free_slots||pool.slot_count<=0||pool.available<0||pool.available>pool.slot_count)return false;
    for(int i=0;i<pool.available;++i) {
        if(pool.free_slots[i]<0||pool.free_slots[i]>=pool.slot_count)return false;
        for(int j=0;j<i;++j)if(pool.free_slots[i]==pool.free_slots[j])return false;
    }
    return true;
}
template<int C> __device__ bool live_owners(const exact::DevicePool& pool,
    const exact::Integer<C>*const* values,int count) {
#if defined(__CUDA_ARCH__)
    if(!unique_free_slots(pool))return false;
    for(int i=0;i<count;++i) {
        const auto& value=*values[i];
        if(value.slot<0) {if(value.valid)return false;continue;}
        // A failed arithmetic result may still own its allocated slot.
        if(value.pool!=&pool||value.slot>=pool.slot_count
            ||value.limb!=pool.words+std::size_t(value.slot)*pool.words_per_slot)return false;
        for(int j=0;j<pool.available;++j)if(pool.free_slots[j]==value.slot)return false;
        for(int j=0;j<i;++j)if(values[j]->slot>=0
            &&(values[j]->slot==value.slot||values[j]->limb==value.limb))return false;
    }
    return true;
#else
    return true;
#endif
}
template<int C> __device__ bool integer_is(const exact::Integer<C>& value,int expected) {
    if(!value.valid)return false;
    if(!expected)return value.sign==0&&value.used==0;
    return value.used==1&&value.sign==(expected<0?-1:1)
        &&value.limb[0]==std::uint32_t(expected<0?-expected:expected);
}
template<int C> __device__ bool point_is(const exact::Point<C>& point,int x,int y,int z) {
    return integer_is(point.v[0],x)&&integer_is(point.v[1],y)&&integer_is(point.v[2],z);
}
template<int C> __device__ bool live_points(const exact::DevicePool& pool,
    const exact::Point<C>*const* points,int count) {
    const exact::Integer<C>* values[42];
    if(count>14)return false;
    int size=0;for(int i=0;i<count;++i)for(int axis=0;axis<3;++axis)values[size++]=&points[i]->v[axis];
    return live_owners(pool,values,size);
}

__device__ bool arithmetic_point_lifetimes(exact::DevicePool& pool) {
    using I=exact::Integer<128>;using P=exact::Point<128>;
    exact::Arithmetic<128> arithmetic;
    P a,b,c,ma,mb,mc;
    a.v[0]=I(1);a.v[1]=I(2);b.v[0]=I(5);b.v[1]=I(7);c.v[0]=I(-2);c.v[1]=I(4);
    ma.v[0]=I(-3);ma.v[1]=I(6);
    const P* live[14]={&a,&b,&c,&ma,&mb,&mc};int count=6;
    if(!live_points(pool,live,count))return false;
    const auto edge=exact::difference(arithmetic,b,a),velocity=exact::difference(arithmetic,mb,ma);
    live[count++]=&edge;live[count++]=&velocity;
    if(!live_points(pool,live,count))return false;
    const auto other=exact::difference(arithmetic,c,a),other_velocity=exact::difference(arithmetic,mc,ma);
    live[count++]=&other;live[count++]=&other_velocity;
    if(!live_points(pool,live,count))return false;
    const auto value=exact::cross(arithmetic,edge,other);live[count++]=&value;
    if(!live_points(pool,live,count))return false;
    const auto first=exact::cross(arithmetic,velocity,other);live[count++]=&first;
    if(!live_points(pool,live,count))return false;
    const auto second=exact::cross(arithmetic,edge,other_velocity);live[count++]=&second;
    if(!live_points(pool,live,count))return false;
    P derivative;for(int axis=0;axis<3;++axis)derivative.v[axis]=arithmetic.sum(first.v[axis],second.v[axis]);
    live[count++]=&derivative;
    return live_points(pool,live,count)&&point_is(a,1,2,0)&&point_is(b,5,7,0)
        &&point_is(c,-2,4,0)&&point_is(ma,-3,6,0)&&point_is(mb,0,0,0)&&point_is(mc,0,0,0)
        &&point_is(edge,4,5,0)&&point_is(velocity,3,-6,0)&&point_is(other,-3,2,0)
        &&point_is(other_velocity,3,-6,0)&&point_is(value,0,0,23)
        &&point_is(first,0,0,-12)&&point_is(second,0,0,-39)&&point_is(derivative,0,0,-51)
        &&exact::verdict(arithmetic,exact::Status::Safe)==exact::Status::Safe;
}

__global__ void exercise_pools(std::uint32_t* scratch,int* free_slots,
    unsigned* failures,int* peaks,std::size_t records,int seed) {
    const unsigned local=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    const unsigned block=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
    const unsigned block_size=blockDim.x*blockDim.y*blockDim.z;
    const unsigned worker=block*block_size+local;
    const unsigned workers=gridDim.x*gridDim.y*gridDim.z*block_size;
    auto* words=scratch+std::size_t(worker)*capacity*slots_per_worker;
    auto* available=free_slots+std::size_t(worker)*slots_per_worker;
    exact::DevicePool pool;
    unsigned failure=0;
    int peak=0;
    exact::initialize_pool(pool,words,available,capacity,slots_per_worker);
    // This barrier is only a test adversary: it makes every worker install
    // first, exposing the former device-global pointer race deterministically
    // within a block. The production allocator never requires a barrier.
    __syncthreads();
    if(!context_is(&pool))failure|=Context;
    {
        exact::Integer<capacity> a(seed+int(worker)+7),b(a),c(-9),d;
        a=a;c=b;d=c;
        const auto original=a.valid?a.limb[0]:0u;
        if(!a.valid||!b.valid||!c.valid||!d.valid
            ||a.limb==b.limb||c.limb==d.limb)failure|=Copies;
#if defined(__CUDA_ARCH__)
        if(a.pool!=&pool||b.pool!=&pool)failure|=Copies;
#endif
        if(b.valid)b.limb[0]^=0x55u;
        if(!a.valid||!c.valid||!d.valid||a.limb[0]!=original
            ||c.limb[0]!=original||d.limb[0]!=original)failure|=Copies;
        {
            exact::Integer<capacity> nested(c);
            if(!nested.valid||nested.limb==c.limb
                ||nested.limb[0]!=original)failure|=Copies;
        }
    }
    if(pool.available!=slots_per_worker||pool.failed)failure|=Restored;
    if(!arithmetic_point_lifetimes(pool))failure|=LiveOwnership;
    if(pool.available!=slots_per_worker||pool.failed||!unique_free_slots(pool))failure|=FreeList;
    {
        using I=exact::Integer<2>;exact::Arithmetic<2> arithmetic;
        const auto wide=exact::shift_left(I(1),62);const I multiplier(8);
        const auto invalid=arithmetic.mul(wide,multiplier);
        const I* live[]={&wide,&multiplier,&invalid};
        if(invalid.valid||pool.failed||!live_owners(pool,live,3)
            ||exact::verdict(arithmetic,exact::Status::Safe)!=exact::Status::CapacityExceeded)
            failure|=ArithmeticFailure;
#if defined(__CUDA_ARCH__)
        if(invalid.slot<0)failure|=ArithmeticFailure;
#endif
    }
    if(pool.available!=slots_per_worker||pool.failed||!unique_free_slots(pool))failure|=FreeList;
    {
        exact::Arithmetic<capacity> arithmetic;
        const auto recovered=arithmetic.sum(exact::Integer<capacity>(17),exact::Integer<capacity>(19));
        if(!integer_is(recovered,36)||exact::verdict(arithmetic,exact::Status::Safe)!=exact::Status::Safe)
            failure|=ArithmeticFailure;
    }
    if(pool.available!=slots_per_worker||pool.failed||!unique_free_slots(pool))failure|=FreeList;

    // More records than CUDA workers exercise repeated ownership and scratch
    // reuse. Different seeds are used in overlapping nonblocking streams.
    for(std::size_t record=worker;record<records;record+=workers) {
        {
            exact::Integer<256> small(seed+std::int64_t(record)+3);
            const auto too_wide=exact::shift_left(small,20000);
            if(!small.valid||too_wide.valid||pool.failed)failure|=NarrowCapacity;
            exact::Arithmetic<256> arithmetic;
            arithmetic.ok=too_wide.valid;
            if(exact::verdict(arithmetic,exact::Status::Safe)
                !=exact::Status::CapacityExceeded)failure|=NarrowCapacity;
        }
        if(pool.available!=slots_per_worker||pool.failed)failure|=Restored;
        {
            const std::int64_t value=seed+std::int64_t(record)+3;
            const auto a=exact::shift_left(exact::Integer<capacity>(value),20000);
            const auto b=exact::shift_left(exact::Integer<capacity>(value+2),15000);
            const auto product=exact::multiply(a,b);
            bool ok=true;
            const double result=exact::ratio_to_double(product,b,-20000,&ok);
            const auto difference=exact::subtract(a,b);
            const auto restored=exact::add(difference,b);
            if(!a.valid||!b.valid||!product.valid||!difference.valid||!restored.valid
                ||!ok||result!=double(value)||exact::compare(a,restored)!=0
                ||!context_is(&pool))failure|=FullCapacity;
            exact::Arithmetic<capacity> arithmetic;
            const auto actual_product=arithmetic.mul(a,b),actual_difference=arithmetic.sub(a,b);
            const auto actual_restored=arithmetic.sum(actual_difference,b);
            const exact::Integer<capacity>* live[]={&a,&b,&product,&difference,&restored,
                &actual_product,&actual_difference,&actual_restored};
            if(!live_owners(pool,live,8))failure|=LiveOwnership;
            if(arithmetic.cmp(actual_product,product)!=0||arithmetic.cmp(actual_difference,difference)!=0
                ||arithmetic.cmp(actual_restored,a)!=0
                ||exact::verdict(arithmetic,exact::Status::Safe)!=exact::Status::Safe)
                failure|=ArithmeticValues;
        }
        if(pool.available!=slots_per_worker||pool.failed)failure|=Restored;
        if(!unique_free_slots(pool))failure|=FreeList;
    }
    peak=pool.peak;
    exact::release_pool();
    if(exact::current_pool()!=nullptr)failure|=Released;

    // Only selected workers exhaust their allocators; neighbors and other
    // blocks/streams must retain their own successful verdicts.
    const bool exhaust=worker%7==0;
    const int limit=exhaust?2:slots_per_worker;
    exact::initialize_pool(pool,words,available,capacity,limit);
    __syncthreads();
    {
        exact::Integer<capacity> a(11),b(13),c(17);
        exact::Arithmetic<capacity> arithmetic;
        const auto status=exact::verdict(arithmetic,exact::Status::Safe);
        if(!a.valid||!b.valid||c.valid==exhaust||pool.failed!=exhaust
            ||exact::current_pool()!=&pool
            ||status!=(exhaust?exact::Status::CapacityExceeded:exact::Status::Safe))
            failure|=Exhaustion;
        const auto result=arithmetic.sum(a,b);
        const exact::Integer<capacity>* live[]={&a,&b,&c,&result};
        if(!live_owners(pool,live,4)||result.valid==exhaust
            ||(!exhaust&&!integer_is(result,24))
            ||exact::verdict(arithmetic,exact::Status::Safe)
                !=(exhaust?exact::Status::CapacityExceeded:exact::Status::Safe))
            failure|=ArithmeticFailure;
    }
    if(pool.available!=limit)failure|=Restored;
    if(!unique_free_slots(pool))failure|=FreeList;
    exact::release_pool();
    if(exact::current_pool()!=nullptr)failure|=Released;

    // Invalid installation must not leave a stale allocator pointer behind.
    exact::initialize_pool(pool,nullptr,available,capacity,slots_per_worker);
    {
        exact::Integer<capacity> invalid(5);
        exact::Arithmetic<capacity> arithmetic;
        if(invalid.valid||!pool.failed||exact::current_pool()!=nullptr
            ||exact::verdict(arithmetic,exact::Status::Safe)
                !=exact::Status::CapacityExceeded)failure|=InvalidInstall;
    }
    exact::release_pool();

    // Reinitialization after failure restores the same worker's scratch.
    exact::initialize_pool(pool,words,available,capacity,slots_per_worker);
    {
        const auto result=exact::add(exact::Integer<capacity>(17),exact::Integer<capacity>(19));
        if(!result.valid||result.used!=1||result.limb[0]!=36||!context_is(&pool))
            failure|=Recovered;
    }
    if(pool.available!=slots_per_worker||pool.failed)failure|=Recovered;
    peak=peak>pool.peak?peak:pool.peak;
    exact::release_pool();
    if(exact::current_pool()!=nullptr)failure|=Released;
    failures[worker]=failure;peaks[worker]=peak;
}

__global__ void exercise_last_slot(std::uint32_t* scratch,int* free_slots,
    unsigned* failures) {
    const unsigned local=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    const unsigned worker=blockIdx.x*1024+local;
    exact::DevicePool pool;
    exact::initialize_pool(pool,scratch+std::size_t(worker)*8,
        free_slots+std::size_t(worker)*4,2,4);
    unsigned failure=exact::current_pool()==&pool?0u:unsigned(Context);
    // There must be no implicit block barrier in allocator operations: some
    // workers deliberately finish while others still construct Integers.
    if(worker%5==0) {
        exact::release_pool();
        failures[worker]=failure|(exact::current_pool()?unsigned(Released):0u);
        return;
    }
    {
        exact::Integer<2> value(std::int64_t(worker)+11),copy(value);
        if(!value.valid||!copy.valid
            ||value.limb==copy.limb||copy.limb[0]!=worker+11)failure|=Copies;
#if defined(__CUDA_ARCH__)
        if(value.pool!=&pool||copy.pool!=&pool)failure|=Copies;
#endif
    }
    if(pool.available!=4||pool.failed)failure|=Restored;
    exact::release_pool();
    if(exact::current_pool()!=nullptr)failure|=Released;
    failures[worker]=failure;
}

void check(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
template<class T> struct DeviceBuffer {
    T* data=nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)),"cudaMalloc");
    }
    ~DeviceBuffer() {if(data)cudaFree(data);}
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
struct Stream {
    cudaStream_t stream{};
    Stream(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create nonblocking stream");}
    ~Stream(){if(stream)cudaStreamDestroy(stream);}
};
struct WorkerStorage {
    DeviceBuffer<std::uint32_t> words;
    DeviceBuffer<int> free_slots;
    DeviceBuffer<unsigned> failures;
    DeviceBuffer<int> peaks;
    explicit WorkerStorage(unsigned workers):words(std::size_t(workers)*capacity*slots_per_worker),
        free_slots(std::size_t(workers)*slots_per_worker),failures(workers),peaks(workers){}
};
void inspect(WorkerStorage& storage,unsigned workers,int seed) {
    std::vector<unsigned> failures(workers);
    std::vector<int> peaks(workers);
    check(cudaMemcpy(failures.data(),storage.failures.data,workers*sizeof(unsigned),cudaMemcpyDeviceToHost),"copy failures");
    check(cudaMemcpy(peaks.data(),storage.peaks.data,workers*sizeof(int),cudaMemcpyDeviceToHost),"copy peaks");
    for(unsigned worker=0;worker<workers;++worker)
        if(failures[worker])throw std::runtime_error("seed "+std::to_string(seed)
            +", worker "+std::to_string(worker)+", failure mask "+std::to_string(failures[worker]));
    std::printf("seed=%d workers=%u all passed; maximum simultaneous slots=%d/%d\n",
        seed,workers,*std::max_element(peaks.begin(),peaks.end()),slots_per_worker);
}
} // namespace

int main() {
    try {
        // All three dimensions participate in both flattened indices.
        const dim3 grid(2,2,2),block(4,4,2);
        constexpr unsigned workers=256;
        constexpr std::size_t records=3*workers+17;
        Stream first,second;
        WorkerStorage a(workers),b(workers);
        cudaFuncAttributes attributes{};
        check(cudaFuncGetAttributes(&attributes,exercise_pools),"kernel attributes");
        std::printf("pool test: static shared=%zu bytes/block; scratch=%zu bytes/stream\n",
            attributes.sharedSizeBytes,std::size_t(workers)*capacity*slots_per_worker*sizeof(std::uint32_t));
        // One 1024-pointer shared symbol, not a duplicate for every inline call.
        if(attributes.sharedSizeBytes!=1024*sizeof(exact::DevicePool*))
            throw std::runtime_error("unexpected shared-pool symbol count/size");
        exercise_pools<<<grid,block,0,first.stream>>>(a.words.data,a.free_slots.data,
            a.failures.data,a.peaks.data,records,101);
        check(cudaGetLastError(),"first pool launch");
        exercise_pools<<<grid,block,0,second.stream>>>(b.words.data,b.free_slots.data,
            b.failures.data,b.peaks.data,records,2003);
        check(cudaGetLastError(),"second pool launch");
        check(cudaStreamSynchronize(first.stream),"first stream synchronize");
        check(cudaStreamSynchronize(second.stream),"second stream synchronize");
        inspect(a,workers,101);inspect(b,workers,2003);
        // Reuse the same allocations in a later launch with a different layout.
        exercise_pools<<<dim3(4,1,1),dim3(8,4,2),0,first.stream>>>(a.words.data,a.free_slots.data,
            a.failures.data,a.peaks.data,records,7001);
        check(cudaGetLastError(),"reused pool launch");
        check(cudaStreamSynchronize(first.stream),"reused stream synchronize");
        inspect(a,workers,7001);
        // Exercise slot 1023, plus early-returning workers, in several blocks.
        constexpr unsigned boundary_workers=3*1024;
        DeviceBuffer<std::uint32_t> boundary_words(boundary_workers*8);
        DeviceBuffer<int> boundary_slots(boundary_workers*4);
        DeviceBuffer<unsigned> boundary_failures(boundary_workers);
        exercise_last_slot<<<3,dim3(8,8,16),0,first.stream>>>(boundary_words.data,
            boundary_slots.data,boundary_failures.data);
        check(cudaGetLastError(),"last slot launch");
        check(cudaStreamSynchronize(first.stream),"last slot synchronize");
        std::vector<unsigned> boundary_result(boundary_workers);
        check(cudaMemcpy(boundary_result.data(),boundary_failures.data,
            boundary_workers*sizeof(unsigned),cudaMemcpyDeviceToHost),"copy last slot results");
        for(unsigned worker=0;worker<boundary_workers;++worker)
            if(boundary_result[worker])throw std::runtime_error("last-slot worker "
                +std::to_string(worker)+", failure mask "+std::to_string(boundary_result[worker]));
        std::puts("GPU exact Arithmetic live ownership, unique free slots, and failure cleanup passed");
        std::puts("GPU exact pool concurrency tests passed");
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"GPU exact pool concurrency test: %s\n",error.what());
        return 1;
    }
}

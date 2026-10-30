#include "global_guess.h"
#include "global_guess.cuh"
#include "device_memory.h"
#include "profile.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <string>

namespace ipc_gpu::solver {
namespace {
using global_guess_detail::Cell;
using global_guess_detail::Input;
using Counter=unsigned long long;
using Point=std::array<double,3>;
constexpr std::size_t scratch_budget=64u*1024u*1024u;
constexpr int maximum_capacity=1<<20;
unsigned blocks(std::size_t n){return static_cast<unsigned>((n+127)/128);}

// Read once per instance. These options change scheduling and unused graph
// construction only; every query retains its arithmetic, heap and retry order.
struct GuessSchedule {
    int workers=512,threads=32;
    bool candidates_only=false;
    static int integer(const char* name,int fallback,int maximum) {
        const char* value=std::getenv(name);
        if(!value)return fallback;
        const auto invalid=[&](){return std::invalid_argument(std::string(name)
            +" must be an integer in [1,"+std::to_string(maximum)+"]");};
        if(!*value)throw invalid();
        int result=0;
        for(const char* digit=value;*digit;++digit) {
            if(*digit<'0'||*digit>'9'||result>(maximum-(*digit-'0'))/10)throw invalid();
            result=result*10+(*digit-'0');
            if(result>maximum)throw invalid();
        }
        if(!result)throw invalid();
        return result;
    }
    static GuessSchedule from_environment() {
        GuessSchedule result;
        result.workers=integer("IPC_GPU_GLOBAL_GUESS_WORKERS",512,8192);
        result.threads=integer("IPC_GPU_GLOBAL_GUESS_THREADS",32,32);
        if(result.threads&(result.threads-1))
            throw std::invalid_argument("IPC_GPU_GLOBAL_GUESS_THREADS must be one of 1,2,4,8,16,32");
        if(const char* value=std::getenv("IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY")) {
            if(std::string(value)!="0"&&std::string(value)!="1")
                throw std::invalid_argument("IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY must be 0 or 1");
            result.candidates_only=*value=='1';
        }
        return result;
    }
};

__global__ void prepare(DeformableScene scene,DeformableState state,
                         double (*ends)[3],BpAabb* boxes,int* error) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=scene.nodes||atomicAdd(error,0))return;
    for(int a=0;a<3;++a) {
        const double start=state.previous[n][a];
        const double d=scene.deformable[n]?state.predicted[n][a]-start:0.0;
        // CPU first forms dx=xhat-x and passes x+dx, not xhat, to TICCD.
        const double end=start+d;ends[n][a]=end;
        boxes[n].lo[a]=fmin(start,end)-1e-10;
        boxes[n].hi[a]=fmax(start,end)+1e-10;
        if(!isfinite(start)||!isfinite(end)||!isfinite(boxes[n].lo[a])||!isfinite(boxes[n].hi[a]))
            atomicCAS(error,0,50);
    }
}
__global__ void query_kernel(const double (*start)[3],const double (*end)[3],int nodes,
                             const BpContact* contacts,std::size_t count,
                             const std::size_t* indices,double* tois,
                             Cell* heaps,int capacity,int workers,
                             Counter* cursor,std::size_t* retry,Counter* retry_count,
                             int* error) {
    const int worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=workers)return;
    Cell* heap=heaps+static_cast<std::size_t>(worker)*capacity;
    while(!atomicAdd(error,0)) {
        const auto job=atomicAdd(cursor,Counter(1));if(job>=count)return;
        const auto index=indices?indices[job]:job;
        const auto contact=contacts[index];
        if(contact.kind!=0&&contact.kind!=1){atomicCAS(error,0,50);return;}
        Input input;input.vertex_face=contact.kind==0;
        for(int v=0;v<4;++v) {
            const int node=contact.vertices[v];
            if(node<0||node>=nodes){atomicCAS(error,0,50);return;}
            for(int a=0;a<3;++a){input.start[v][a]=start[node][a];input.end[v][a]=end[node][a];}
        }
        const auto result=global_guess_detail::solve(input,heap,capacity);
        if(result.status){atomicCAS(error,0,result.status);return;}
        if(result.capacity&&capacity==maximum_capacity){atomicCAS(error,0,52);return;}
        if(result.capacity)retry[atomicAdd(retry_count,Counter(1))]=index;
        else tois[index]=result.toi;
    }
}
__device__ void atomic_minimum(double* result,double value) {
    auto p=reinterpret_cast<unsigned long long*>(result);
    auto old=atomicCAS(p,0ull,0ull);
    while(__longlong_as_double(old)>value) {
        const auto assumed=old;old=atomicCAS(p,assumed,__double_as_longlong(value));
        if(old==assumed)break;
    }
}
__global__ void initialize_minimum(double* toi){*toi=1;}
__global__ void reduce_minimum(const double* tois,std::size_t count,double* minimum,int* error) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&!atomicAdd(error,0))atomic_minimum(minimum,tois[i]);
}
__global__ void commit(DeformableScene scene,DeformableState state,const double* minimum,int* error) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=scene.nodes||!scene.deformable[n]||atomicAdd(error,0))return;
    const double toi=*minimum,omega=toi>=1?1:.9*toi;
    for(int a=0;a<3;++a) {
        const double start=state.previous[n][a];
        state.positions[n][a]=start+omega*(state.predicted[n][a]-start);
    }
}
}

struct GpuGlobalGuess::Impl {
    GuessSchedule schedule=GuessSchedule::from_environment();
    GpuBroadPhase broadphase;
    DeviceArray<Point> ends;
    DeviceArray<BpAabb> boxes;
    DeviceArray<Cell> heaps;
    DeviceArray<std::size_t> retries[2];
    DeviceArray<Counter> cursor{1},retry_count{1};
    DeviceArray<double> tois,minimum{1};
};
GpuGlobalGuess::GpuGlobalGuess():impl_(new Impl){}
GpuGlobalGuess::~GpuGlobalGuess()=default;

void GpuGlobalGuess::query(const double (*start)[3],const double (*end)[3],int nodes,
                               const BpContact* contacts,std::size_t count,double* tois,int* error) {
    GpuProfileScope profile("query");
    if(!error||nodes<0||(!start&&nodes)||(!end&&nodes)||(!contacts&&count)||(!tois&&count))
        throw std::invalid_argument("GPU global guess query pointers/extent");
    if(count==0)return;
    auto& p=*impl_;
    p.retries[0].resize(count);p.retries[1].resize(count);
    const std::size_t* indices=nullptr;
    std::size_t pending=count;
    int destination=0;
    for(int capacity=256;pending;capacity*=2) {
        GpuProfileScope batch_profile(capacity==256?"initial_capacity":"capacity_retry");
        if(capacity>maximum_capacity)throw std::runtime_error("GPU global guess exceeded BFS queue proof bound");
        const int workers=static_cast<int>(std::min({pending,std::size_t(p.schedule.workers),
            std::max(std::size_t(1),scratch_budget/(std::size_t(capacity)*sizeof(Cell)))}));
        const auto required=static_cast<std::size_t>(workers)*capacity;
        gpu_profile_counter(GpuProfileCounter::GlobalQueryBatches);
        gpu_profile_counter(GpuProfileCounter::GlobalQueryJobs,pending);
        if(capacity>256)gpu_profile_counter(GpuProfileCounter::GlobalRetryJobs,pending);
        gpu_profile_counter(GpuProfileCounter::GlobalQueueCapacityMax,capacity,true);
        gpu_profile_counter(GpuProfileCounter::GlobalWorkersMax,workers,true);
        if(p.heaps.size()<required)p.heaps.resize(required);
        p.cursor.clear();p.retry_count.clear();
        query_kernel<<<(workers+p.schedule.threads-1)/p.schedule.threads,p.schedule.threads>>>(start,end,nodes,contacts,pending,indices,tois,
            p.heaps.data(),capacity,workers,p.cursor.data(),p.retries[destination].data(),
            p.retry_count.data(),error);
        cuda_checked(cudaGetLastError());
        // Counts/status only: no geometry, query results, or minimum returned
        // to the host. Failed-capacity queries restart on a larger GPU heap.
        pending=static_cast<std::size_t>(p.retry_count.download()[0]);
        if(pending>count)throw std::runtime_error("GPU global guess invalid retry count");
        indices=p.retries[destination].data();destination=1-destination;
    }
}

void GpuGlobalGuess::apply(const DeformableScene& scene,const DeformableState& state,
                               const BroadPhaseTopology& topology,int* error) {
    if(scene.nodes>INT_MAX||topology.vertex_count!=static_cast<int>(scene.nodes)||!error)
        throw std::invalid_argument("GPU global guess scene/topology extent");
    if(!scene.nodes)return;
    auto& p=*impl_;p.ends.resize(scene.nodes);p.boxes.resize(scene.nodes);
    auto ends=reinterpret_cast<double(*)[3]>(p.ends.data());
    int status=0;
    {
    GpuProfileScope prepare_profile("prepare");
    prepare<<<blocks(scene.nodes),128>>>(scene,state,ends,p.boxes.data(),error);
    cuda_checked(cudaGetLastError());
    // A failed preparation cannot feed invalid boxes into sorting/BVH. Only
    // error status is read; caller's rollback policy owns the numerical state.
    cuda_checked(cudaMemcpy(&status,error,sizeof(status),cudaMemcpyDeviceToHost));
    }
    if(status)return;
    auto query_topology=topology;
    query_topology.dependency_count=0;query_topology.dependencies=nullptr;
    query_topology.include_triangle_dependencies=false;
    query_topology.surface_query=nullptr; // CPU includes interior NT sources.
    query_topology.protect_linear_gap=false;
    const auto broad=p.broadphase.rebuild(query_topology,p.boxes.data(),0.0,
        p.schedule.candidates_only?BroadPhaseMode::CandidatesOnly:BroadPhaseMode::Solver);
    p.tois.resize(broad.contact_count);
    query(state.previous,ends,static_cast<int>(scene.nodes),broad.contacts,broad.contact_count,p.tois.data(),error);
    GpuProfileScope reduce_profile("reduce_commit");
    initialize_minimum<<<1,1>>>(p.minimum.data());
    if(broad.contact_count)reduce_minimum<<<blocks(broad.contact_count),128>>>(p.tois.data(),broad.contact_count,p.minimum.data(),error);
    commit<<<blocks(scene.nodes),128>>>(scene,state,p.minimum.data(),error);
    cuda_checked(cudaGetLastError());
}
} // namespace ipc_gpu::solver

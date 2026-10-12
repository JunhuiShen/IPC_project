#include "resident_elastic.h"
#include "resident_profile.h"
#include "cloth_math.cuh"
#include "solid_math.cuh"
#include "resident_solve.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ipc_gpu::resident {
namespace {
constexpr unsigned width = 128;
unsigned blocks(std::size_t count) { return static_cast<unsigned>((count + width - 1) / width); }
unsigned list_blocks(std::size_t count,unsigned threads) {
    const auto result=count/threads+(count%threads!=0);
    if(result>static_cast<std::size_t>(INT_MAX))throw std::overflow_error("Resident elastic worklist grid exceeds supported extent");
    return static_cast<unsigned>(result);
}
void checked(cudaError_t status,const char* what) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("Resident elastic ")+what+": "+cudaGetErrorString(status));
}
void checked_launch() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Resident elastic CUDA launch: ") + cudaGetErrorString(status));
}
__device__ bool active(const DeformableScene& s, const int* colors, int color, std::size_t node) {
    return s.deformable[node] && colors[s.node_blocks[node]] == color;
}
__device__ std::size_t work_index(ElasticIndexRange range) {
    const auto ordinal=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(ordinal>=range.count)return static_cast<std::size_t>(-1);
    return range.indices?range.indices[ordinal]:ordinal;
}
__device__ bool finite(const Derivatives& d) {
    for (int i=0; i<3; ++i) if (!isfinite(d.gradient[i])) return false;
    for (int i=0; i<9; ++i) if (!isfinite(d.hessian[i])) return false;
    return true;
}
__global__ void begin(DeformableScene scene, DeformableState state, double dt) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes) return;
    for(int a=0;a<3;++a) {
        state.previous[node][a]=state.positions[node][a];
        state.predicted[node][a]=state.positions[node][a]
            +(scene.deformable[node] ? dt*state.velocities[node][a] : 0.0);
    }
}
__global__ void gather_points(DeformableScene s, DeformableState x, ElasticWorkspace w, const int* colors,int color,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n)) return;
    auto& p=w.points[n]; p.mass=s.masses[n]; p.pinned=s.pinned[n]; p.solid=s.solid[n];
    for(int a=0;a<3;++a) {
        p.position[a]=x.positions[n][a]; p.predicted[a]=x.predicted[n][a];
        p.target[a]=s.pin_targets[n][a];
    }
}
template<class Incident,class Gathered,int Corners>
__global__ void gather_elements(const Incident* records,std::size_t count,Gathered* output,
                               DeformableScene s,DeformableState x,const int* colors,int color,ElasticIndexRange range) {
    const auto e=work_index(range);
    if(e>=count || !active(s,colors,color,records[e].owner)) return;
    output[e]=records[e].material;
    for(int c=0;c<Corners;++c) for(int a=0;a<3;++a)
        output[e].positions[3*c+a]=x.positions[records[e].nodes[c]][a];
}
__global__ void compute_points(DeformableScene s,ElasticWorkspace w,const int* colors,int color,Parameters params,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n)) return;
    auto result=evaluate(w.points[n],params);
    if(!finite(result)) atomicCAS(w.error,0,1);
    w.point_outputs[n]=result;
}
template<class Incident,class Gathered,class Params>
__global__ void compute_elements(const Incident* records,std::size_t count,const Gathered* input,
    Derivatives* output,DeformableScene s,const int* colors,int color,Params params,int* error,ElasticIndexRange range) {
    const auto e=work_index(range);
    if(e>=count || !active(s,colors,color,records[e].owner)) return;
    auto result=evaluate(input[e],params);
    if(!finite(result)) atomicCAS(error,0,1);
    output[e]=result;
}
__device__ void accumulate(Derivatives& sum,const Derivatives& value,double dt2) {
    for(int a=0;a<3;++a) sum.gradient[a]+=dt2*value.gradient[a];
    for(int a=0;a<9;++a) sum.hessian[a]+=dt2*value.hessian[a];
}
__global__ void reduce(DeformableScene s,ElasticWorkspace w,const int* colors,int color,double dt2,bool bending,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n)) return;
    auto value=w.point_outputs[n];
    for(auto i=s.triangle_offsets[n];i<s.triangle_offsets[n+1];++i) accumulate(value,w.triangle_outputs[i],dt2);
    for(auto i=s.solid_offsets[n];i<s.solid_offsets[n+1];++i) accumulate(value,w.solid_outputs[i],dt2);
    if(bending) for(auto i=s.hinge_offsets[n];i<s.hinge_offsets[n+1];++i) accumulate(value,w.hinge_outputs[i],dt2);
    if(!finite(value)) atomicCAS(w.error,0,1);
    w.systems[n]=value;
}
__global__ void propose(DeformableScene s,DeformableState x,ElasticWorkspace w,const int* colors,int color,double damping,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n) || atomicAdd(w.error,0)) return;
    const auto& q=w.systems[n]; double delta[3];
    if(!solve3(q.hessian,q.gradient,delta)){atomicCAS(w.error,0,2);return;}
    for(int a=0;a<3;++a) {
        const double value=x.positions[n][a]-damping*delta[a];
        if(!isfinite(value)) atomicCAS(w.error,0,2);
        w.proposed[n][a]=value;
    }
}
__global__ void validate_fractions(DeformableScene s,ElasticWorkspace w,const int* colors,int color,const double* fractions,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n)) return;
    if(!isfinite(fractions[n]) || fractions[n]<0.0 || fractions[n]>1.0) atomicCAS(w.error,0,2);
}
__global__ void commit(DeformableScene s,DeformableState x,ElasticWorkspace w,const int* colors,int color,const double* fractions,ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n) || *w.error) return;
    for(int a=0;a<3;++a) x.positions[n][a]+=fractions[n]*(w.proposed[n][a]-x.positions[n][a]);
}
__global__ void finish(DeformableScene s,DeformableState x,double dt) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=s.nodes || !s.deformable[n]) return;
    for(int a=0;a<3;++a) x.velocities[n][a]=(x.positions[n][a]-x.previous[n][a])/dt;
}
__global__ void validate_endpoints(DeformableScene s,ElasticWorkspace w,const int* colors,int color,const double (*accepted)[3],ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n))return;
    for(int a=0;a<3;++a)if(!isfinite(accepted[n][a]))atomicCAS(w.error,0,2);
}
__global__ void commit_endpoints(DeformableScene s,DeformableState x,ElasticWorkspace w,const int* colors,int color,const double (*accepted)[3],ElasticIndexRange range) {
    const auto n=work_index(range);
    if(n>=s.nodes || !active(s,colors,color,n) || *w.error)return;
    for(int a=0;a<3;++a)x.positions[n][a]=accepted[n][a];
}

template<class T>struct ListBuffer {
    T* data=nullptr;std::size_t capacity=0;
    ~ListBuffer(){cudaFree(data);}
    void reserve(std::size_t count) {
        if(count<=capacity)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("Resident elastic worklist allocation overflow");
        T* replacement=nullptr;
        checked(cudaMalloc(reinterpret_cast<void**>(&replacement),count*sizeof(T)),"allocate worklist");
        const auto release=cudaFree(data);
        if(release!=cudaSuccess){cudaFree(replacement);checked(release,"release worklist");}
        data=replacement;capacity=count;
    }
};
struct IndexStorage {ListBuffer<int> keys;ListBuffer<std::size_t> indices;};
bool worklists_enabled() {
    const char* value=std::getenv("IPC_GPU_ELASTIC_COLOR_WORKLISTS");
    if(!value||std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("IPC_GPU_ELASTIC_COLOR_WORKLISTS must be 0 or 1");
}
bool valid_threads(unsigned value){return value&&value<=128&&(value&(value-1))==0;}
unsigned worklist_threads(const char* name,unsigned fallback) {
    const char* value=std::getenv(name);
    if(!value)return fallback;
    for(unsigned threads=1;threads<=128;threads*=2)
        if(std::to_string(threads)==value)return threads;
    throw std::invalid_argument(std::string(name)+" must be one of 1,2,4,8,16,32,64,128");
}
__device__ int owner_color(DeformableScene scene,std::size_t node,const int* colors,
                          int block_count,int color_count,int* invalid) {
    if(node>=scene.nodes){atomicCAS(invalid,0,1);return INT_MAX;}
    if(!scene.deformable[node])return INT_MAX;
    const int block=scene.node_blocks[node];
    if(block<0||block>=block_count){atomicCAS(invalid,0,2);return INT_MAX;}
    const int color=colors[block];
    if(color<0)return INT_MAX;
    if(color>=color_count){atomicCAS(invalid,0,3);return INT_MAX;}
    return color;
}
__global__ void index_nodes(DeformableScene scene,const int* colors,int block_count,
    int color_count,int* keys,std::size_t* indices,int* invalid) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes)return;
    indices[node]=node;keys[node]=owner_color(scene,node,colors,block_count,color_count,invalid);
}
template<class Incident>
__global__ void index_elements(const Incident* records,std::size_t count,DeformableScene scene,
    const int* colors,int block_count,int color_count,int* keys,std::size_t* indices,int* invalid) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    indices[index]=index;
    keys[index]=owner_color(scene,static_cast<std::size_t>(records[index].owner),colors,block_count,color_count,invalid);
}
__global__ void index_offsets(const int* keys,std::size_t count,int colors,std::size_t* offsets) {
    const auto color=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(color>static_cast<std::size_t>(colors))return;
    std::size_t begin=0,end=count;
    while(begin<end){const auto mid=begin+(end-begin)/2;if(keys[mid]<static_cast<int>(color))begin=mid+1;else end=mid;}
    offsets[color]=begin;
}
void validate_worklist(const ElasticColorWorklist& list,const int* colors,int color) {
    if(!list.enabled)return;
    if(list.block_colors!=colors||list.color!=color)throw std::invalid_argument("Resident elastic worklist belongs to another color view");
    if(!valid_threads(list.node_threads)||!valid_threads(list.element_threads))throw std::invalid_argument("Invalid resident elastic worklist launch width");
    for(const auto range:{list.nodes,list.triangles,list.solids,list.hinges})
        if(range.count&&!range.indices)throw std::invalid_argument("Missing resident elastic worklist indices");
}
} // namespace

struct ResidentElasticWorklists::Impl {
    const bool enabled=worklists_enabled();
    const unsigned node_threads=enabled?worklist_threads("IPC_GPU_ELASTIC_NODE_THREADS",1):1;
    const unsigned element_threads=enabled?worklist_threads("IPC_GPU_ELASTIC_ELEMENT_THREADS",8):8;
    std::array<IndexStorage,4> lists;
    ListBuffer<std::size_t> offsets;
    ListBuffer<int> invalid;
    std::vector<std::size_t> host_offsets;
    const int* colors=nullptr;
    int color_count=0;
    bool ready=false;
};
ResidentElasticWorklists::ResidentElasticWorklists():impl_(std::make_unique<Impl>()){}
ResidentElasticWorklists::~ResidentElasticWorklists()=default;
bool ResidentElasticWorklists::enabled()const{return impl_->enabled;}
void ResidentElasticWorklists::rebuild(const DeformableScene& scene,const int* colors,
    int block_count,int color_count,cudaStream_t stream) {
    auto& p=*impl_;p.ready=false;p.colors=nullptr;p.color_count=0;
    if(!p.enabled)return;
    ResidentProfileScope profile("elastic_worklists",stream);
    if(block_count<0||color_count<0||color_count>block_count||(block_count&&!colors)
        ||(scene.nodes&&(!scene.node_blocks||!scene.deformable))
        ||(scene.triangle_count&&!scene.triangles)||(scene.solid_count&&!scene.solids)
        ||(scene.hinge_count&&!scene.hinges))throw std::invalid_argument("Invalid resident elastic worklist topology");
    const std::size_t stride=static_cast<std::size_t>(color_count)+1;
    if(stride>std::numeric_limits<std::size_t>::max()/4)throw std::overflow_error("Resident elastic color offsets overflow");
    const std::array<std::size_t,4> counts{{scene.nodes,scene.triangle_count,scene.solid_count,scene.hinge_count}};
    p.offsets.reserve(4*stride);p.invalid.reserve(1);p.host_offsets.resize(4*stride);
    checked(cudaMemsetAsync(p.invalid.data,0,sizeof(int),stream),"clear worklist validation");
    for(std::size_t kind=0;kind<4;++kind) {
        auto& list=p.lists[kind];list.keys.reserve(counts[kind]);list.indices.reserve(counts[kind]);
        if(counts[kind]) {
            const auto grid=list_blocks(counts[kind],width);
            if(kind==0)index_nodes<<<grid,width,0,stream>>>(scene,colors,block_count,color_count,list.keys.data,list.indices.data,p.invalid.data);
            if(kind==1)index_elements<<<grid,width,0,stream>>>(scene.triangles,counts[kind],scene,colors,block_count,color_count,list.keys.data,list.indices.data,p.invalid.data);
            if(kind==2)index_elements<<<grid,width,0,stream>>>(scene.solids,counts[kind],scene,colors,block_count,color_count,list.keys.data,list.indices.data,p.invalid.data);
            if(kind==3)index_elements<<<grid,width,0,stream>>>(scene.hinges,counts[kind],scene,colors,block_count,color_count,list.keys.data,list.indices.data,p.invalid.data);
            checked_launch();
            // Input IDs start in ascending original-array order. Stable sort
            // by color preserves that order, including repeated owner records.
            thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                thrust::device_pointer_cast(list.keys.data),thrust::device_pointer_cast(list.keys.data+counts[kind]),
                thrust::device_pointer_cast(list.indices.data));
        }
        index_offsets<<<list_blocks(stride,width),width,0,stream>>>(list.keys.data,counts[kind],color_count,p.offsets.data+kind*stride);
        checked_launch();
    }
    int invalid=0;
    checked(cudaMemcpyAsync(p.host_offsets.data(),p.offsets.data,4*stride*sizeof(std::size_t),cudaMemcpyDeviceToHost,stream),"download worklist launch counts");
    checked(cudaMemcpyAsync(&invalid,p.invalid.data,sizeof(invalid),cudaMemcpyDeviceToHost,stream),"download worklist validation");
    checked(cudaStreamSynchronize(stream),"complete worklist rebuild");
    if(invalid)throw std::invalid_argument("Invalid resident elastic worklist owner/block/color, status "+std::to_string(invalid));
    p.colors=colors;p.color_count=color_count;p.ready=true;
}
ElasticColorWorklist ResidentElasticWorklists::color(int color)const {
    const auto& p=*impl_;if(!p.enabled)return {};
    if(!p.ready)throw std::logic_error("Resident elastic worklists must be rebuilt before use");
    if(color<0||color>=p.color_count)throw std::out_of_range("Resident elastic worklist color out of range");
    ElasticColorWorklist result;result.enabled=true;result.node_threads=p.node_threads;result.element_threads=p.element_threads;
    result.block_colors=p.colors;result.color=color;
    const auto stride=static_cast<std::size_t>(p.color_count)+1;
    ElasticIndexRange* ranges[]={&result.nodes,&result.triangles,&result.solids,&result.hinges};
    for(std::size_t kind=0;kind<4;++kind) {
        const auto begin=p.host_offsets[kind*stride+color],end=p.host_offsets[kind*stride+color+1];
        *ranges[kind]={p.lists[kind].indices.data?p.lists[kind].indices.data+begin:nullptr,end-begin};
    }
    return result;
}

void begin_deformable_substep(const DeformableScene& s,const DeformableState& x,double dt,cudaStream_t stream) {
    if(!(dt>0.0) || !std::isfinite(dt)) throw std::invalid_argument("Resident substep requires positive finite dt");
    if(s.nodes) begin<<<blocks(s.nodes),width,0,stream>>>(s,x,dt);
    checked_launch();
}
void assemble_elastic_color(const DeformableScene& s,const DeformableState& x,const ElasticWorkspace& w,
    const int* colors,int color,const Parameters& p,const SolidParameters& sp,cudaStream_t stream,ElasticColorWorklist worklist) {
    ResidentProfileScope profile("elastic",stream);
    validate_worklist(worklist,colors,color);
    const auto nodes=worklist.enabled?worklist.nodes:ElasticIndexRange{nullptr,s.nodes};
    const auto triangles=worklist.enabled?worklist.triangles:ElasticIndexRange{nullptr,s.triangle_count};
    const auto solids=worklist.enabled?worklist.solids:ElasticIndexRange{nullptr,s.solid_count};
    const auto hinges=worklist.enabled?worklist.hinges:ElasticIndexRange{nullptr,s.hinge_count};
    const unsigned nt=worklist.enabled?worklist.node_threads:width,et=worklist.enabled?worklist.element_threads:width;
    {
    ResidentProfileScope gather_profile("gather",stream);
    if(nodes.count) gather_points<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,x,w,colors,color,nodes);
    if(triangles.count) gather_elements<TriangleIncident,Triangle,3><<<list_blocks(triangles.count,et),et,0,stream>>>(s.triangles,s.triangle_count,w.triangles,s,x,colors,color,triangles);
    if(solids.count) gather_elements<SolidIncident,Solid,4><<<list_blocks(solids.count,et),et,0,stream>>>(s.solids,s.solid_count,w.solids,s,x,colors,color,solids);
    if(hinges.count && p.bending_stiffness>0.0) gather_elements<HingeIncident,Hinge,4><<<list_blocks(hinges.count,et),et,0,stream>>>(s.hinges,s.hinge_count,w.hinges,s,x,colors,color,hinges);
    checked_launch();
    }
    {
    ResidentProfileScope compute_profile("compute",stream);
    if(nodes.count) compute_points<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,w,colors,color,p,nodes);
    if(triangles.count) compute_elements<<<list_blocks(triangles.count,et),et,0,stream>>>(s.triangles,s.triangle_count,w.triangles,w.triangle_outputs,s,colors,color,p,w.error,triangles);
    if(solids.count) compute_elements<<<list_blocks(solids.count,et),et,0,stream>>>(s.solids,s.solid_count,w.solids,w.solid_outputs,s,colors,color,sp,w.error,solids);
    if(hinges.count && p.bending_stiffness>0.0) compute_elements<<<list_blocks(hinges.count,et),et,0,stream>>>(s.hinges,s.hinge_count,w.hinges,w.hinge_outputs,s,colors,color,p,w.error,hinges);
    checked_launch();
    }
    ResidentProfileScope reduce_profile("reduce",stream);
    if(nodes.count) reduce<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,w,colors,color,p.dt2,p.bending_stiffness>0.0,nodes);
    checked_launch();
}
void propose_deformable_color(const DeformableScene& s,const DeformableState& x,const ElasticWorkspace& w,
    const int* colors,int color,double damping,cudaStream_t stream,ElasticColorWorklist worklist) {
    if(!std::isfinite(damping)) throw std::invalid_argument("Nonfinite resident damping");
    validate_worklist(worklist,colors,color);
    const auto nodes=worklist.enabled?worklist.nodes:ElasticIndexRange{nullptr,s.nodes};
    const unsigned nt=worklist.enabled?worklist.node_threads:width;
    if(nodes.count) propose<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,x,w,colors,color,damping,nodes);
    checked_launch();
}
void commit_deformable_color(const DeformableScene& s,const DeformableState& x,const ElasticWorkspace& w,
    const int* colors,int color,const double* fractions,cudaStream_t stream,ElasticColorWorklist worklist) {
    validate_worklist(worklist,colors,color);
    const auto nodes=worklist.enabled?worklist.nodes:ElasticIndexRange{nullptr,s.nodes};
    const unsigned nt=worklist.enabled?worklist.node_threads:width;
    if(nodes.count) {
        validate_fractions<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,w,colors,color,fractions,nodes);
        checked_launch();
        commit<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,x,w,colors,color,fractions,nodes);
    }
    checked_launch();
}
void finish_deformable_substep(const DeformableScene& s,const DeformableState& x,double dt,cudaStream_t stream) {
    if(!(dt>0.0) || !std::isfinite(dt)) throw std::invalid_argument("Resident substep requires positive finite dt");
    if(s.nodes) finish<<<blocks(s.nodes),width,0,stream>>>(s,x,dt);
    checked_launch();
}
void commit_deformable_endpoints(const DeformableScene& s,const DeformableState& x,
    const ElasticWorkspace& w,const int* colors,int color,const double (*accepted)[3],cudaStream_t stream,ElasticColorWorklist worklist) {
    validate_worklist(worklist,colors,color);
    const auto nodes=worklist.enabled?worklist.nodes:ElasticIndexRange{nullptr,s.nodes};
    const unsigned nt=worklist.enabled?worklist.node_threads:width;
    if(nodes.count) {
        validate_endpoints<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,w,colors,color,accepted,nodes);
        checked_launch();
        commit_endpoints<<<list_blocks(nodes.count,nt),nt,0,stream>>>(s,x,w,colors,color,accepted,nodes);
    }
    checked_launch();
}
} // namespace ipc_gpu::resident

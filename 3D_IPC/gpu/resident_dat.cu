#include "resident_dat.h"
#include "resident_dat_math.cuh"
#include "resident_solver.h"
#include "resident_memory.h"
#include "resident_sdf.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <climits>
#include <cstdint>

namespace ipc_gpu::resident {
namespace {
unsigned blocks(std::size_t n){return static_cast<unsigned>((n+127)/128);}
struct Contribution {Derivatives roles[4];};
__global__ void point_boxes(std::size_t n,const double (*x)[3],BpAabb* boxes) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)for(int k=0;k<3;++k)boxes[i].lo[k]=boxes[i].hi[k]=x[i][k];
}
__global__ void initialize_pairs(BroadPhaseView broad,const double (*reference)[3],double rq,dat::Geometry* geometry,std::uint64_t* keys) {
    const auto pair=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    const auto contact=broad.contacts[pair];double x[4][3];
    for(int r=0;r<4;++r){for(int k=0;k<3;++k)x[r][k]=reference[contact.vertices[r]][k];keys[4*pair+r]=(std::uint64_t(contact.vertices[r])<<32)|std::uint64_t(4*pair+r);}
    geometry[pair]=dat::geometry(x,contact.kind==0,rq);
}
__global__ void node_offsets(std::size_t n,const std::uint64_t* keys,std::size_t count,std::size_t* offsets) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(node>n)return;
    const std::uint64_t key=std::uint64_t(node)<<32;std::size_t lo=0,hi=count;
    while(lo<hi){const auto mid=lo+(hi-lo)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}offsets[node]=lo;
}
__global__ void contact_derivatives(BroadPhaseView broad,DeformableState state,ContactParameters p,Contribution* output,int* error) {
    const auto pair=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(pair>=broad.contact_count||*error)return;
    const auto c=broad.contacts[pair];ContactInput in{};in.kind=c.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=p.d_hat;in.k_barrier=p.k_barrier;in.dt=p.dt;in.friction=p.friction;in.eps_v=p.eps_v;in.eps=p.eps;in.hessian_blocks=0x8421u;
    for(int r=0;r<4;++r)for(int k=0;k<3;++k){in.positions[r][k]=state.positions[c.vertices[r]][k];in.previous[r][k]=state.previous[c.vertices[r]][k];}
    Contribution value{};double squared_gap=0;const int split=c.kind==0?1:2;
    for(int k=0;k<3;++k) {
        double lo[2]={INFINITY,INFINITY},hi[2]={-INFINITY,-INFINITY};
        for(int r=0;r<4;++r){const int side=r<split?0:1;lo[side]=fmin(lo[side],in.positions[r][k]);hi[side]=fmax(hi[side],in.positions[r][k]);}
        const double gap=fmax(0.0,fmax(lo[0]-hi[1],lo[1]-hi[0]));squared_gap+=gap*gap;
    }
    if(squared_gap>p.d_hat*p.d_hat){output[pair]=value;return;}
    const auto result=dat::evaluate(in);
    if(result.status!=ContactStatus::Ok) {
        if(result.status==ContactStatus::InvalidInput||result.status==ContactStatus::ExactCapacityExceeded||result.status==ContactStatus::ZeroDistance)atomicCAS(error,0,60+static_cast<int>(result.status));
        // CPU DAT retains a vertex's accepted displacement when its local
        // Newton system is nonfinite. Mark incident systems for that policy.
        for(int r=0;r<4;++r)value.roles[r].gradient[0]=NAN;
        output[pair]=value;return;
    }
    if(result.active)for(int r=0;r<4;++r) {
        auto& d=value.roles[r];const double scale=p.dt*p.dt*p.k_barrier,w=result.weights[r];
        for(int k=0;k<3;++k)d.gradient[k]=scale*result.barrier_gradient[r][k]+w*result.friction_relative_gradient[k];
        for(int k=0;k<9;++k)d.hessian[k]=scale*result.barrier_hessian[r][r][k]+w*w*result.friction_relative_hessian[k];
    }
    output[pair]=value;
}
__global__ void assemble_contacts(DeformableScene scene,DeformableState state,ElasticWorkspace elastic,const int* colors,int color,const std::uint64_t* keys,const std::size_t* offsets,const Contribution* contacts,SdfScene sdf) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||colors[scene.node_blocks[node]]!=color||*elastic.error)return;
    auto& system=elastic.systems[node];
    for(int k=0;k<3;++k)if(!isfinite(system.gradient[k]))return;
    for(int k=0;k<9;++k)if(!isfinite(system.hessian[k]))return;
    const auto obstacle=evaluate_sdf_contact(sdf.colliders,sdf.count,state.positions[node],state.previous[node],sdf.parameters);
    if(obstacle.status!=SdfStatus::Ok){atomicCAS(elastic.error,0,70);return;}
    for(int k=0;k<3;++k)system.gradient[k]+=sdf.parameters.dt*sdf.parameters.dt*obstacle.gradient[k]+obstacle.friction_gradient[k];
    for(int k=0;k<9;++k)system.hessian[k]+=sdf.parameters.dt*sdf.parameters.dt*obstacle.hessian[k]+obstacle.friction_hessian[k];
    for(auto i=offsets[node];i<offsets[node+1];++i) {
        const auto encoded=static_cast<unsigned>(keys[i]);const auto& value=contacts[encoded/4].roles[encoded%4];
        for(int k=0;k<3;++k)system.gradient[k]+=value.gradient[k];
        for(int k=0;k<9;++k)system.hessian[k]+=value.hessian[k];
    }
    // Nonfinite sums are left for propose() to reject locally, as on CPU.
}
__global__ void propose(DeformableScene scene,ElasticWorkspace elastic,const int* colors,int color,double damping,const double (*displacement)[3],double (*trial)[3]) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(node>=scene.nodes||*elastic.error)return;
    for(int k=0;k<3;++k)trial[node][k]=displacement[node][k];
    if(colors[scene.node_blocks[node]]!=color)return;
    double delta[3],q[3];const auto& system=elastic.systems[node];if(!solve3(system.hessian,system.gradient,delta))return;
    for(int k=0;k<3;++k){const double increment=-damping*delta[k];q[k]=trial[node][k]+increment;if(!isfinite(increment)||!isfinite(q[k]))return;}
    for(int k=0;k<3;++k)trial[node][k]=q[k];
}
__global__ void divide(BroadPhaseView broad,const dat::Geometry* geometry,const double (*trial)[3],double gamma,double* planes,double (*scales)[4],const int* error) {
    const auto pair=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(pair>=broad.contact_count||*error)return;
    double q[4][3];for(int r=0;r<4;++r)for(int k=0;k<3;++k)q[r][k]=trial[broad.contacts[pair].vertices[r]][k];
    planes[pair]=dat::plane(geometry[pair],q);
    for(int r=0;r<4;++r)scales[pair][r]=dat::pair_scale(geometry[pair],r,q[r],planes[pair],gamma);
}
__global__ void truncate(DeformableScene scene,DeformableState state,const double (*reference)[3],double (*displacement)[3],const double (*trial)[3],const std::uint64_t* keys,const std::size_t* offsets,const dat::Geometry* geometry,const double* planes,const double (*scales)[4],double radius,double gamma,int* error) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(node>=scene.nodes||*error)return;
    double scale=1;
    for(auto i=offsets[node];i<offsets[node+1];++i){const auto encoded=static_cast<unsigned>(keys[i]);scale=fmin(scale,scales[encoded/4][encoded%4]);}
    const double length=norm3(trial[node],0);if(length>radius)scale=fmin(scale,(radius/length)*(1-16*DBL_EPSILON));
    double accepted[3];for(int k=0;k<3;++k)accepted[k]=reference[node][k];
    for(int attempt=0;attempt<64;++attempt) {
        double candidate[3];bool safe=true;
        for(int k=0;k<3;++k){candidate[k]=reference[node][k]+scale*trial[node][k];safe=safe&&isfinite(candidate[k]);}
        safe=safe&&dat::within_radius(reference[node],candidate,radius);
        for(auto i=offsets[node];safe&&i<offsets[node+1];++i){const auto encoded=static_cast<unsigned>(keys[i]);safe=dat::accepts(geometry[encoded/4],encoded%4,planes[encoded/4],reference[node],candidate,gamma);}
        if(safe){for(int k=0;k<3;++k)accepted[k]=candidate[k];break;}
        scale*=attempt<8?1-1e-8:.5;
    }
    for(int k=0;k<3;++k){state.positions[node][k]=accepted[k];displacement[node][k]=accepted[k]-reference[node][k];}
}
}
struct ResidentDAT::Impl {
    DeviceArray<Vector3> reference,displacement,trial;
    DeviceArray<BpAabb> boxes;
    DeviceArray<unsigned char> no_queries;
    DeviceArray<int> elastic_status;
    DeviceArray<dat::Geometry> geometry;
    DeviceArray<Contribution> contributions;
    DeviceArray<std::uint64_t> keys;
    DeviceArray<std::size_t> offsets;
    DeviceArray<double> planes;
    DeviceArray<std::array<double,4>> scales;
    ResidentBroadPhase elastic_graph,detection;
    ResidentElasticWorklists worklists;
    BroadPhaseView graph{};
    bool graph_ready=false;
};
ResidentDAT::ResidentDAT():impl_(std::make_unique<Impl>()){}
ResidentDAT::~ResidentDAT()=default;
void ResidentDAT::solve(const DeformableScene& scene,const DeformableState& state,const ElasticWorkspace& elastic,const BroadPhaseTopology& topology,const SdfScene& sdf,const StepParameters& p) {
    auto& w=*impl_;const auto n=scene.nodes;if(!n)return;
    w.reference.resize(n);w.displacement.resize(n);w.trial.resize(n);w.boxes.resize(n);w.no_queries.resize(n);w.no_queries.clear();w.offsets.resize(n+1);
    auto v3=[](DeviceArray<Vector3>& a){return reinterpret_cast<double(*)[3]>(a.data());};
    if(!w.graph_ready) {
        point_boxes<<<blocks(n),128>>>(n,state.positions,w.boxes.data());
        auto graph_topology=topology;graph_topology.edges=nullptr;graph_topology.edge_count=0;graph_topology.surface_query=w.no_queries.data();graph_topology.protect_linear_gap=false;graph_topology.include_triangle_dependencies=scene.triangle_count>0;
        w.graph=w.elastic_graph.rebuild(graph_topology,w.boxes.data(),0);
        w.worklists.rebuild(scene,w.graph.block_colors,topology.block_count,w.graph.color_count);w.graph_ready=true;
    }
    const auto graph=w.graph;
    auto detection_topology=topology;detection_topology.protect_linear_gap=false;
    BroadPhaseView pairs{};const double rq=1.5*p.contact.d_hat,radius=.5*p.dat_relaxation*rq;
    const auto detect=[&]() {
        cuda_checked(cudaMemcpy(w.reference.data(),state.positions,n*sizeof(Vector3),cudaMemcpyDeviceToDevice));w.displacement.clear();
        point_boxes<<<blocks(n),128>>>(n,state.positions,w.boxes.data());
        pairs=w.detection.rebuild(detection_topology,w.boxes.data(),rq,BroadPhaseMode::CandidatesOnly);
        if(pairs.contact_count>std::size_t(INT_MAX/4))throw std::overflow_error("DAT contact incidence capacity");
        const auto count=pairs.contact_count;w.geometry.resize(count);w.contributions.resize(count);w.keys.resize(4*count);w.planes.resize(count);w.scales.resize(count);
        if(count){initialize_pairs<<<blocks(count),128>>>(pairs,v3(w.reference),rq,w.geometry.data(),w.keys.data());thrust::sort(thrust::device_pointer_cast(w.keys.data()),thrust::device_pointer_cast(w.keys.data()+4*count));}
        node_offsets<<<blocks(n+1),128>>>(n,w.keys.data(),4*count,w.offsets.data());
    };
    detect();int since_detection=0;w.elastic_status.resize(1);
    ElasticWorkspace dat_elastic=elastic;dat_elastic.error=w.elastic_status.data();
    for(int iteration=0;iteration<p.iterations;++iteration)for(int color=0;color<graph.color_count;++color) {
        if(since_detection>=p.dat_detection_interval){detect();since_detection=0;}
        const auto worklist=w.worklists.color(color);
        // Shared elasticity reports nonfinite local derivatives globally.
        // DAT's CPU policy instead skips only that vertex's Newton proposal.
        w.elastic_status.clear();
        assemble_elastic_color(scene,state,dat_elastic,graph.block_colors,color,p.elastic,p.solid,nullptr,worklist);
        if(pairs.contact_count)contact_derivatives<<<blocks(pairs.contact_count),128>>>(pairs,state,p.contact,w.contributions.data(),elastic.error);
        assemble_contacts<<<blocks(n),128>>>(scene,state,elastic,graph.block_colors,color,w.keys.data(),w.offsets.data(),w.contributions.data(),sdf);
        propose<<<blocks(n),128>>>(scene,elastic,graph.block_colors,color,p.damping,v3(w.displacement),v3(w.trial));
        if(pairs.contact_count)divide<<<blocks(pairs.contact_count),128>>>(pairs,w.geometry.data(),v3(w.trial),p.dat_relaxation,w.planes.data(),reinterpret_cast<double(*)[4]>(w.scales.data()),elastic.error);
        truncate<<<blocks(n),128>>>(scene,state,v3(w.reference),v3(w.displacement),v3(w.trial),w.keys.data(),w.offsets.data(),w.geometry.data(),w.planes.data(),reinterpret_cast<double(*)[4]>(w.scales.data()),radius,p.dat_relaxation,elastic.error);
        ++since_detection;
    }
    cuda_checked(cudaGetLastError());
}
} // namespace ipc_gpu::resident

#include "contact_pipeline_test.h"
#include "contact_pipeline.h"
#include "original_contact.cuh"
#include "directed_rounding.cuh"
#include "rigid_dat.cuh"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string("pipeline CUDA test: ")+cudaGetErrorString(error));}
template<class T>struct Memory {
    T* data=nullptr;std::size_t count;
    explicit Memory(std::size_t n,const T* host=nullptr):count(n){check(cudaMalloc(&data,n*sizeof(T)));if(host)check(cudaMemcpy(data,host,n*sizeof(T),cudaMemcpyHostToDevice));else check(cudaMemset(data,0,n*sizeof(T)));}
    ~Memory(){cudaFree(data);}
    void download(T* host){check(cudaMemcpy(host,data,count*sizeof(T),cudaMemcpyDeviceToHost));}
};
__global__ void shift_rigids(ipc_gpu::solver::DeformableState state,const int* body,const double* shift) {
    int v=threadIdx.x;if(v<8&&body[v]>=0)for(int k=0;k<3;++k)state.positions[v][k]+=shift[3*body[v]+k];
}
}
void gpu_contact_pipeline_test_gpu(const GpuPipelineTestInput& input,GpuPipelineTestOutput& output) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    const int vertex_blocks[8]={0,8,8,8,9,9,6,7},body_map[8]={-1,0,0,0,1,1,-1,-1},body_blocks[2]={8,9};
    const int colors[10]={0,-1,-1,-1,-1,-1,1,2,1,0},body_nodes[5]={1,2,3,4,5};
    const std::size_t body_offsets[3]={0,3,5};
    const unsigned char deformable[8]={1,0,0,0,0,0,1,1},solid[8]={0,0,0,0,0,0,1,1},surface[8]={1,1,1,1,1,1,1,0},enabled[2]={1,1};
    BpContact contacts[2]={{{0,1,2,3},0},{{4,5,6,7},1}};
    Memory<double> positions(24,&input.positions[0][0]),previous(24,&input.previous[0][0]),references(24,&input.references[0][0]),shift(6,&input.shift[0][0]);
    Memory<int> d_blocks(8,vertex_blocks),d_body_map(8,body_map),d_body_blocks(2,body_blocks),d_colors(10,colors),d_body_nodes(5,body_nodes);
    Memory<std::size_t> d_body_offsets(3,body_offsets);
    Memory<unsigned char> d_deformable(8,deformable),d_solid(8,solid),d_surface(8,surface),d_enabled(2,enabled);
    Memory<BpContact> d_contacts(2,contacts);
    Memory<double> orientation(8,&input.previous_orientation[0][0]),omega(6,&input.omega[0][0]),com(6),velocity(6),moments(18);
    Memory<Derivatives> systems(8),rigid_systems(2);
    Memory<int> failed(1,&input.initial_error);
    Memory<SdfCollider> collider(1,&input.collider);
    DeformableScene scene{};scene.nodes=8;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(positions.data);state.previous=reinterpret_cast<double(*)[3]>(previous.data);
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=2;rigid.body_blocks=d_body_blocks.data;rigid.node_body=d_body_map.data;rigid.reference=reinterpret_cast<const double(*)[3]>(references.data);
    rigid.node_offsets=d_body_offsets.data;rigid.nodes=d_body_nodes.data;rigid.translate=d_enabled.data;rigid.rotate=d_enabled.data;
    RigidState rigid_state{};rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(orientation.data);rigid_state.orientation=rigid_state.previous_orientation;
    rigid_state.omega=reinterpret_cast<double(*)[3]>(omega.data);rigid_state.com=reinterpret_cast<double(*)[3]>(com.data);
    RigidWorkspace rigid_workspace{};rigid_workspace.systems=rigid_systems.data;
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=2;broad.block_colors=d_colors.data;
    SdfScene sdf{collider.data,1,input.sdf_parameters};
    ContactParameters parameters{.8,370,.031,.2,.1,1e-12};
    GpuContactPipeline pipeline;
    pipeline.rebuild(broad,10,d_blocks.data,d_surface.data);
    std::size_t offsets[11];check(cudaMemcpy(offsets,pipeline.incidence().block_offsets,sizeof(offsets),cudaMemcpyDeviceToHost));
    for(int b=0;b<10;++b)output.incident_counts[b]=int(offsets[b+1]-offsets[b]);
    for(int color=0;color<3;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::Deformable);
    systems.download(output.deformable);
    for(int color=0;color<3;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidTranslation);
    rigid_systems.download(output.rigid_com);
    check(cudaMemset(rigid_systems.data,0,2*sizeof(Derivatives)));
    shift_rigids<<<1,8>>>(state,d_body_map.data,shift.data);check(cudaGetLastError());
    for(int color=0;color<3;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidOrientation);
    rigid_systems.download(output.rigid_omega);
    // Same pipeline, body/reference identities and pair topology, but a new
    // orientation state. Pullbacks must be rebuilt rather than reused stale.
    check(cudaMemset(rigid_systems.data,0,2*sizeof(Derivatives)));
    check(cudaMemcpy(omega.data,&input.omega_after[0][0],6*sizeof(double),cudaMemcpyHostToDevice));
    for(int color=0;color<3;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidOrientation);
    rigid_systems.download(output.rigid_omega_after);failed.download(&output.error);
}

namespace {
__global__ void original_status(const ipc_gpu::solver::ContactInput* input,int* status) {
    *status=int(ipc_gpu::solver::evaluate_contact_original(*input).status);
}
__global__ void directed_rounding_check(const std::uint64_t* patterns,int count,int* failure) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    namespace rounding=ipc_gpu::directed_rounding;
    const double x=rounding::detail::from_bits(patterns[i]);
    if(rounding::detail::bits(rounding::next_up(x))!=rounding::detail::bits(::nextafter(x,double(INFINITY)))
        ||rounding::detail::bits(rounding::next_down(x))!=rounding::detail::bits(::nextafter(x,-double(INFINITY)))
        ||rounding::detail::bits(rounding::toward_positive_zero(x))!=rounding::detail::bits(::nextafter(x,0.)))
        atomicCAS(failure,0,1);
}
__global__ void capped_radius_check(int* failure) {
    namespace rd=ipc_gpu::solver::rigid_dat_detail;
    const int i=threadIdx.x;if(i>=18)return;
    rd::Trajectory path;path.reference[0]=1;path.delta_omega[2]=6.28318530717958647692;
    const double query[3]={1,0,0};const double caps[6]={0.,-0.,.01,.1,.5,1.};
    const int depth=i>=6&&i<12?1:40,budget=i<12?4096:1;
    const auto full=rd::radius_step(path,query,.4,depth,budget);
    rd::PreparedTrajectory prepared;const auto status=rd::prepare_trajectory(path,prepared);
    const double cap=caps[i%6];
    const auto actual=rd::radius_step_capped(path,query,.4,cap,depth,budget);
    const auto cached=rd::radius_step_capped(path,query,.4,cap,depth,budget,&prepared);
    // Retain signed zero from the requested cap as the production minimum does.
    const double expected=cap<full.factor?cap:full.factor;
    if(status!=rd::Status::Ok||rd::is_fatal(full.status)||rd::is_fatal(actual.status)
        ||ipc_gpu::directed_rounding::detail::bits(actual.factor)!=ipc_gpu::directed_rounding::detail::bits(expected)
        ||ipc_gpu::directed_rounding::detail::bits(actual.factor)!=ipc_gpu::directed_rounding::detail::bits(cached.factor)
        ||actual.status!=cached.status
        ||(actual.status==rd::Status::Ok&&actual.factor!=cap)
        ||((cap==1||cap>full.factor)&&actual.status!=full.status))atomicCAS(failure,0,2);
    const double invalid_query[3]={INFINITY,0,0};
    if(rd::radius_step_capped(path,invalid_query,.4,0).status!=rd::Status::InvalidInput)
        atomicCAS(failure,0,3);
}
}

void gpu_dat_numerical_helpers_test_gpu() {
    std::vector<std::uint64_t> patterns{0,UINT64_C(0x8000000000000000),1,
        UINT64_C(0x000fffffffffffff),UINT64_C(0x0010000000000000),
        UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff0000000000000),
        UINT64_C(0x7ff8000000000001),UINT64_C(0xfff0000000000000)};
    for(std::uint64_t exponent=1;exponent<2047;++exponent)for(auto sign:{UINT64_C(0),UINT64_C(0x8000000000000000)})
        for(std::uint64_t offset:{UINT64_C(0),UINT64_C(1),UINT64_C(0x000fffffffffffff)})
            patterns.push_back(sign|(exponent<<52)|offset);
    std::uint64_t random=UINT64_C(0x7891a543f2330abc);
    for(int i=0;i<4096;++i) {
        random^=random<<13;random^=random>>7;random^=random<<17;patterns.push_back(random);
    }
    Memory<std::uint64_t> inputs(patterns.size(),patterns.data());Memory<int> failed(1);
    directed_rounding_check<<<unsigned((patterns.size()+127)/128),128>>>(inputs.data,int(patterns.size()),failed.data);
    check(cudaGetLastError());capped_radius_check<<<1,32>>>(failed.data);check(cudaGetLastError());
    int failure=0;failed.download(&failure);
    if(failure)throw std::runtime_error("device DAT rounding/capped-radius helper mismatch "+std::to_string(failure));
}

void gpu_contact_original_frozen_test_gpu(const double (*positions)[3],const double (*previous)[3],
    bool frozen,ipc_gpu::Derivatives* output,int& error,int& fast_status) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    const int blocks[4]={0,1,2,3},colors[4]={0,0,0,0};
    const unsigned char enabled[4]={1,1,1,1},solid[4]{};const BpContact pair{{0,1,2,3},1};
    Memory<double> x(12,&positions[0][0]),history(12,&previous[0][0]);
    Memory<int> vertex_blocks(4,blocks),block_colors(4,colors),failed(1);
    Memory<unsigned char> deformable(4,enabled),d_solid(4,solid);Memory<BpContact> contact(1,&pair);
    Memory<Derivatives> systems(4);
    ContactInput input;input.kind=ContactKind::EdgeEdge;input.d_hat=.005;input.k_barrier=37;
    input.dt=.031;input.friction=.2;input.eps_v=.1;
    std::memcpy(input.positions,positions,sizeof(input.positions));
    std::memcpy(input.previous,previous,sizeof(input.previous));
    Memory<ContactInput> fast_input(1,&input);Memory<int> fast_result(1);
    original_status<<<1,1>>>(fast_input.data,fast_result.data);check(cudaGetLastError());
    fast_result.download(&fast_status);
    DeformableScene scene{};scene.nodes=4;scene.node_blocks=vertex_blocks.data;
    scene.deformable=deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);
    state.previous=reinterpret_cast<double(*)[3]>(history.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contact.data;broad.contact_count=1;broad.block_colors=block_colors.data;
    GpuContactPipeline pipeline;pipeline.rebuild(broad,4,vertex_blocks.data);
    const ContactParameters parameters{input.d_hat,input.k_barrier,input.dt,input.friction,input.eps_v,input.eps};
    pipeline.assemble_color(scene,state,work,{},{},{},{},parameters,0,ContactAssemblyPhase::Deformable,
        frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},ContactGeometryPolicy::Original);
    systems.download(output);failed.download(&error);
    double unchanged[12];x.download(unchanged);
    if(std::memcmp(unchanged,positions,sizeof(unchanged))!=0)throw std::runtime_error("contact retry modified frozen positions");
    history.download(unchanged);
    if(std::memcmp(unchanged,previous,sizeof(unchanged))!=0)throw std::runtime_error("contact retry modified previous positions");
}

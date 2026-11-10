#include "contact_pipeline_test.h"
#include "contact_pipeline.h"
#include "original_contact.cuh"
#include "sdf.cuh"
#include "contact_exact.cuh"
#include "contact_compact.cuh"
#include "directed_rounding.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstring>
#include <cstdlib>
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
namespace {
struct MixedOptions {
    std::size_t capacity=65536;
    bool original=false,frozen=false,pins=false,seeded=false,interleaved=false;
    bool empty_contacts=false,reverse_contacts=false,swap_owners=false;
    bool disable_translation=false,disable_rotation=false;
    std::size_t repeat_contacts=1;
    double d_hat=-1.;
    bool point_only=false,mixed_activity=false,missing_owner=false,no_sdf=false;
    bool local_colors=false;
};
void require(bool value,const std::string& why) {if(!value)throw std::runtime_error(why);}
class RigidCacheSetting {
    bool had_value=false;
    std::string previous;
public:
    explicit RigidCacheSetting(bool enabled) {
        const char* value=std::getenv("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE");
        had_value=value!=nullptr;if(value)previous=value;
        if(setenv("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE",enabled?"1":"0",1)!=0)
            throw std::runtime_error("cannot select rigid contact cache test route");
    }
    ~RigidCacheSetting() {
        if(had_value)setenv("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE",previous.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE");
    }
};
class RigidCompactSetting {
    bool had_value=false;std::string previous;
public:
    explicit RigidCompactSetting(bool enabled) {
        const char* value=std::getenv("IPC_GPU_RIGID_COMPACT_REDUCE");had_value=value!=nullptr;if(value)previous=value;
        if(setenv("IPC_GPU_RIGID_COMPACT_REDUCE",enabled?"1":"0",1)!=0)
            throw std::runtime_error("cannot select rigid compact reduction test route");
    }
    ~RigidCompactSetting() {
        if(had_value)setenv("IPC_GPU_RIGID_COMPACT_REDUCE",previous.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_COMPACT_REDUCE");
    }
};
class ContactOptionSetting {
    const char* name;
    bool had_value=false;
    std::string previous;
public:
    ContactOptionSetting(const char* option,const char* selected):name(option) {
        const char* value=std::getenv(name);had_value=value!=nullptr;if(value)previous=value;
        if((selected?setenv(name,selected,1):unsetenv(name))!=0)
            throw std::runtime_error("cannot select contact execution test option");
    }
    ~ContactOptionSetting(){if(had_value)setenv(name,previous.c_str(),1);else unsetenv(name);}
};
class ColorLocalSetting:public ContactOptionSetting {
public: explicit ColorLocalSetting(bool enabled):ContactOptionSetting("IPC_GPU_COLOR_LOCAL_PACKETS",enabled?"1":"0"){}
};
ipc_gpu::Derivatives initial_system(int index) {
    ipc_gpu::Derivatives out{};
    for(int k=0;k<3;++k) {
        out.gradient[k]=(k==0?1e8:k==1?-1e-7:.125)*(index+1);
        for(int j=0;j<3;++j)out.hessian[3*k+j]=k==j?2.0*(index+1)*(k+1):.03125*(k+j+1);
    }
    return out;
}
template<class T>void same_bytes(const T* actual,const T* expected,std::size_t count,const char* why) {
    require(!count||std::memcmp(actual,expected,count*sizeof(T))==0,why);
}
void run_mixed(const GpuPipelineTestInput& input,GpuPipelineTestOutput& output,
    MixedOptions option,ipc_gpu::solver::ContactStorageBytes* storage=nullptr,
    ipc_gpu::solver::GpuContactPipeline* existing_pipeline=nullptr,
    std::vector<std::size_t>* color_counts=nullptr) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    const int vertex_blocks[8]={0,8,8,8,9,9,6,7};
    int body_map[8]={-1,0,0,0,1,1,-1,-1},body_blocks[2]={8,9};
    int colors[10]={0,-1,-1,-1,-1,-1,1,2,1,0};
    if(option.local_colors)colors[0]=2;
    if(option.frozen)for(int block:{0,6,7,8,9})colors[block]=0;
    int body_nodes[5]={1,2,3,4,5};
    std::size_t body_offsets[3]={0,3,5};
    if(option.swap_owners) {
        for(int& body:body_map)if(body>=0)body=1-body;
        std::swap(body_blocks[0],body_blocks[1]);
        const int regrouped[5]={4,5,1,2,3};
        std::copy(regrouped,regrouped+5,body_nodes);body_offsets[1]=2;
    }
    if(option.missing_owner) {
        // The PT incidence still belongs to block/body 0, while all its rigid
        // roles name body 1. This intentionally supported direct-API mismatch
        // must retain accumulate_rigid_contact's absent-owner fallback.
        for(int vertex=1;vertex<=3;++vertex)body_map[vertex]=1;
    }
    const unsigned char deformable[8]={1,0,0,0,0,0,1,1},solid[8]={0,0,0,0,0,0,1,1},surface[8]={1,1,1,1,1,1,1,0};
    const unsigned char translate[2]={static_cast<unsigned char>(!option.disable_translation),static_cast<unsigned char>(!option.disable_translation)};
    const unsigned char rotate[2]={static_cast<unsigned char>(!option.disable_rotation),static_cast<unsigned char>(!option.disable_rotation)};
    std::vector<BpContact> contacts{{{0,1,2,3},0},{{4,5,6,7},1}};
    if(option.interleaved)contacts={{{0,5,6,7},1},{{0,1,2,3},0},{{2,4,6,7},1},
        {{4,1,2,3},0},{{6,1,2,3},0},{{4,5,6,7},1},{{0,1,2,3},0}};
    if(option.point_only) {
        contacts={{{0,1,2,3},0}};
        if(option.mixed_activity)contacts.push_back({{6,1,2,3},0});
    }
    if(option.repeat_contacts!=1) {
        const auto pattern=contacts;contacts.clear();
        for(std::size_t repeat=0;repeat<option.repeat_contacts;++repeat)
            contacts.insert(contacts.end(),pattern.begin(),pattern.end());
    }
    if(option.reverse_contacts)std::reverse(contacts.begin(),contacts.end());
    if(option.empty_contacts)contacts.clear();
    Memory<double> positions(24,&input.positions[0][0]),previous(24,&input.previous[0][0]),references(24,&input.references[0][0]),shift(6,&input.shift[0][0]);
    Memory<int> d_blocks(8,vertex_blocks),d_body_map(8,body_map),d_body_blocks(2,body_blocks),d_colors(10,colors),d_body_nodes(5,body_nodes);
    Memory<std::size_t> d_body_offsets(3,body_offsets);
    Memory<unsigned char> d_deformable(8,deformable),d_solid(8,solid),d_surface(8,surface),d_translate(2,translate),d_rotate(2,rotate);
    Memory<BpContact> d_contacts(std::max<std::size_t>(1,contacts.size()),contacts.empty()?nullptr:contacts.data());
    Memory<double> orientation(8,&input.previous_orientation[0][0]),omega(6,&input.omega[0][0]),com(6),velocity(6),moments(18);
    Derivatives node_seed[8]{},body_seed[2]{};
    if(option.seeded) {
        for(int n=0;n<8;++n)node_seed[n]=initial_system(n);
        for(int b=0;b<2;++b)body_seed[b]=initial_system(b+8);
    }
    Memory<Derivatives> systems(8,node_seed),rigid_systems(2,body_seed);
    Memory<int> failed(1,&input.initial_error);
    Memory<SdfCollider> collider(1,&input.collider);
    DeformableScene scene{};scene.nodes=8;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(positions.data);state.previous=reinterpret_cast<double(*)[3]>(previous.data);
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=2;rigid.body_blocks=d_body_blocks.data;rigid.node_body=d_body_map.data;rigid.reference=reinterpret_cast<const double(*)[3]>(references.data);
    rigid.node_offsets=d_body_offsets.data;rigid.nodes=d_body_nodes.data;rigid.translate=d_translate.data;rigid.rotate=d_rotate.data;
    RigidState rigid_state{};rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(orientation.data);rigid_state.orientation=rigid_state.previous_orientation;
    rigid_state.omega=reinterpret_cast<double(*)[3]>(omega.data);rigid_state.com=reinterpret_cast<double(*)[3]>(com.data);
    RigidWorkspace rigid_workspace{};rigid_workspace.systems=rigid_systems.data;
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();broad.block_colors=d_colors.data;broad.color_count=option.frozen?1:3;
    SdfScene sdf{collider.data,1,input.sdf_parameters};
    if(option.no_sdf)sdf={};
    ContactParameters parameters{option.d_hat>=0.?option.d_hat:option.interleaved?4.:.8,370,.031,.2,.1,1e-12};
    RigidPinConstraint pin[2];const std::size_t pin_offsets[3]={0,1,2};
    for(int body=0;body<2;++body) {
        pin[body].body=body;pin[body].local_position[0]=.13*(body+1);pin[body].local_position[1]=-.07;
        pin[body].initial_target[0]=.4;pin[body].initial_target[1]=-.2;pin[body].initial_target[2]=.3;
        pin[body].rotation_center[0]=.2;pin[body].angular_velocity_z=body?.7:-.9;
    }
    Memory<RigidPinConstraint> d_pins(2,pin);Memory<std::size_t> d_pin_offsets(3,pin_offsets);
    RigidPinAssembly pins;
    if(option.pins){pins.scene={2,2,d_pins.data,d_pin_offsets.data};pins.stiffness=83.;pins.time=.17;}
    std::unique_ptr<GpuContactPipeline> owned_pipeline;
    if(!existing_pipeline)owned_pipeline=std::make_unique<GpuContactPipeline>(option.capacity);
    GpuContactPipeline& pipeline=existing_pipeline?*existing_pipeline:*owned_pipeline;
    pipeline.rebuild(broad,10,d_blocks.data,d_surface.data);
    if(color_counts) {
        const auto list=pipeline.incidence().color_pairs;
        require(list.offsets!=nullptr,"mixed color-local test needs indexed color queues");
        std::vector<std::size_t> host(std::size_t(broad.color_count)+1);
        check(cudaMemcpy(host.data(),list.offsets,host.size()*sizeof(std::size_t),cudaMemcpyDeviceToHost));
        color_counts->resize(broad.color_count);
        for(int color=0;color<broad.color_count;++color)(*color_counts)[color]=host[color+1]-host[color];
    }
    std::size_t offsets[11];check(cudaMemcpy(offsets,pipeline.incidence().block_offsets,sizeof(offsets),cudaMemcpyDeviceToHost));
    for(int b=0;b<10;++b)output.incident_counts[b]=int(offsets[b+1]-offsets[b]);
    if(option.original&&!option.frozen&&!existing_pipeline)require(pipeline.storage_bytes().exact_bytes==0,"Original GS rebuild allocated unused exact scratch");
    const auto mode=option.frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep;
    const auto policy=option.original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust;
    for(int color=0;color<broad.color_count;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::Deformable,mode,pins,policy);
    systems.download(output.deformable);
    for(int color=0;color<broad.color_count;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidTranslation,mode,pins,policy);
    rigid_systems.download(output.rigid_com);
    check(cudaMemcpy(rigid_systems.data,body_seed,sizeof(body_seed),cudaMemcpyHostToDevice));
    shift_rigids<<<1,8>>>(state,d_body_map.data,shift.data);check(cudaGetLastError());
    for(int color=0;color<broad.color_count;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidOrientation,mode,pins,policy);
    rigid_systems.download(output.rigid_omega);
    // Same pipeline, body/reference identities and pair topology, but a new
    // orientation state. Pullbacks must be rebuilt rather than reused stale.
    check(cudaMemcpy(rigid_systems.data,body_seed,sizeof(body_seed),cudaMemcpyHostToDevice));
    check(cudaMemcpy(omega.data,&input.omega_after[0][0],6*sizeof(double),cudaMemcpyHostToDevice));
    for(int color=0;color<broad.color_count;++color)pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,parameters,color,ContactAssemblyPhase::RigidOrientation,mode,pins,policy);
    rigid_systems.download(output.rigid_omega_after);failed.download(&output.error);
    double unchanged[24],expected[24];positions.download(unchanged);
    for(int n=0;n<8;++n)for(int k=0;k<3;++k)
        expected[3*n+k]=input.positions[n][k]+(body_map[n]>=0?input.shift[body_map[n]][k]:0.);
    same_bytes(unchanged,expected,24,"contact assembly modified positions beyond explicit test shift");
    previous.download(unchanged);same_bytes(unchanged,&input.previous[0][0],24,"contact assembly modified frozen history");
    if(storage)*storage=pipeline.storage_bytes();
    if(option.original&&!option.frozen)require(pipeline.storage_bytes().exact_bytes==0,"Original GS allocated unused exact scratch");
}
}
void gpu_contact_pipeline_test_gpu(const GpuPipelineTestInput& input,GpuPipelineTestOutput& output) {
    RigidCompactSetting compact(false);
    ColorLocalSetting exact(false); // Preserve all existing bitwise gates.
    {RigidCacheSetting setting(false);run_mixed(input,output,{});}
    GpuPipelineTestOutput cached;
    {RigidCacheSetting setting(true);run_mixed(input,cached,{});}
    same_bytes(cached.deformable,output.deformable,8,"rigid cache changed deformable derivatives");
    same_bytes(cached.rigid_com,output.rigid_com,2,"rigid cache changed COM derivatives/error guard");
    same_bytes(cached.rigid_omega,output.rigid_omega,2,"rigid cache changed angular derivatives/error guard");
    same_bytes(cached.rigid_omega_after,output.rigid_omega_after,2,"rigid cache reused stale angular pullbacks");
    same_bytes(cached.incident_counts,output.incident_counts,10,"rigid cache changed incidence counts");
    require(cached.error==output.error,"rigid cache changed zero-gap/preexisting device error");
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

}

void gpu_directed_rounding_helpers_test_gpu() {
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
    check(cudaGetLastError());
    int failure=0;failed.download(&failure);
    if(failure)throw std::runtime_error("device directed-rounding helper mismatch "+std::to_string(failure));
}

void gpu_contact_original_frozen_test_gpu(const double (*positions)[3],const double (*previous)[3],
    bool frozen,ipc_gpu::Derivatives* output,int& error,int& fast_status) {
    ColorLocalSetting exact(false);
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

namespace {
std::size_t packet_bound(std::size_t capacity,bool rigid_cache=false) {
    using namespace ipc_gpu::solver;
    return capacity*(sizeof(ContactInput)+sizeof(ContactOutput)+sizeof(CompactDeformableContact)
        +2*sizeof(int)+2*sizeof(unsigned char)
        +(rigid_cache?8*sizeof(ipc_gpu::Derivatives):0))+256;
}
void compare_mixed_bytes(const GpuPipelineTestOutput& a,const GpuPipelineTestOutput& b) {
    same_bytes(a.deformable,b.deformable,8,"tiled deformable accumulation changed bytes");
    same_bytes(a.rigid_com,b.rigid_com,2,"tiled rigid translation changed bytes");
    same_bytes(a.rigid_omega,b.rigid_omega,2,"tiled rigid orientation changed bytes");
    same_bytes(a.rigid_omega_after,b.rigid_omega_after,2,"tiled refreshed pullbacks changed bytes");
    same_bytes(a.incident_counts,b.incident_counts,10,"tiling changed candidate incidence");
    require(a.error==b.error,"tiling changed device error");
}

// Direct API callers and frozen DAT can select more than two rigid owners
// in one pair. Cache by distinct owner, rather than assuming one active side.
void rigid_cache_many_owners() {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    constexpr std::size_t count=4;
    const double positions[count][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0}};
    double previous[count][3],reference[count][3];
    for(std::size_t n=0;n<count;++n)for(int k=0;k<3;++k) {
        previous[n][k]=positions[n][k]-.001*(n+1)*(k+1);
        reference[n][k]=positions[n][k]+.03*(n+1)*(k+1);
    }
    const int owners[count]={0,1,2,3},colors[count]={0,0,0,0},body_nodes[count]={0,1,2,3};
    const std::size_t offsets[count+1]={0,1,2,3,4};
    const unsigned char deformable[count]{},solid[count]{};
    const unsigned char translate[count]={1,1,0,1},rotate[count]={1,0,1,1};
    const double q[count][4]={{1,0,0,0},{1,0,0,0},{1,0,0,0},{1,0,0,0}};
    const double omega[count][3]={{.13,-.21,.08},{.26,-.42,.16},{-.11,.27,.06},{.31,.03,-.17}};
    const BpContact pairs[3]={{{0,1,2,3},1},{{0,1,2,3},0},{{0,1,2,3},1}};
    Derivatives seed[count];for(std::size_t b=0;b<count;++b)seed[b]=initial_system(int(b)+4);
    Memory<double> x(3*count,&positions[0][0]),old(3*count,&previous[0][0]),refs(3*count,&reference[0][0]);
    Memory<int> d_owners(count,owners),d_colors(count,colors),d_nodes(count,body_nodes),failed(1);
    Memory<std::size_t> d_offsets(count+1,offsets);
    Memory<unsigned char> d_deformable(count,deformable),d_solid(count,solid),d_translate(count,translate),d_rotate(count,rotate);
    Memory<BpContact> contacts(3,pairs);Memory<Derivatives> node_systems(count,seed),systems(count,seed);
    Memory<double> d_q(4*count,&q[0][0]),d_omega(3*count,&omega[0][0]),com(3*count);
    DeformableScene scene{};scene.nodes=count;scene.node_blocks=d_owners.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace elastic{};elastic.systems=node_systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=count;rigid.body_blocks=d_owners.data;rigid.node_body=d_owners.data;
    rigid.reference=reinterpret_cast<const double(*)[3]>(refs.data);rigid.node_offsets=d_offsets.data;rigid.nodes=d_nodes.data;
    rigid.translate=d_translate.data;rigid.rotate=d_rotate.data;
    RigidState rigid_state{};rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(d_q.data);
    rigid_state.orientation=rigid_state.previous_orientation;rigid_state.omega=reinterpret_cast<double(*)[3]>(d_omega.data);rigid_state.com=reinterpret_cast<double(*)[3]>(com.data);
    RigidWorkspace workspace{};workspace.systems=systems.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=3;broad.block_colors=d_colors.data;broad.color_count=1;
    const ContactParameters params{.8,37,.031,.2,.1,1e-12};
    for(bool original:{false,true})for(bool frozen:{false,true}) {
        std::array<std::array<Derivatives,count>,2> expected{};
        for(std::size_t capacity:{std::size_t(32),std::size_t(1)})for(bool cache:{false,true}) {
            RigidCacheSetting setting(cache);GpuContactPipeline pipeline(capacity);
            pipeline.rebuild(broad,int(count),d_owners.data);
            for(int phase=0;phase<2;++phase) {
                check(cudaMemcpy(systems.data,seed,sizeof(seed),cudaMemcpyHostToDevice));
                check(cudaMemset(failed.data,0,sizeof(int)));
                pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,workspace,{},params,0,
                    phase?ContactAssemblyPhase::RigidOrientation:ContactAssemblyPhase::RigidTranslation,
                    frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},
                    original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust);
                int status=0;failed.download(&status);require(status==0,"many-owner rigid cache fixture failed");
                std::array<Derivatives,count> actual;systems.download(actual.data());
                if(capacity==32&&!cache)expected[phase]=actual;
                else same_bytes(actual.data(),expected[phase].data(),count,"rigid owner-slot cache changed ordered derivatives");
                const int disabled=phase?1:2;
                same_bytes(actual.data()+disabled,seed+disabled,1,"rigid cache touched a disabled body phase");
            }
            require(pipeline.storage_bytes().packet_bytes<=packet_bound(capacity,cache),"many-owner cache exceeded packet bound");
        }
    }
    double unchanged[3*count];x.download(unchanged);same_bytes(unchanged,&positions[0][0],3*count,"rigid cache modified positions");
    old.download(unchanged);same_bytes(unchanged,&previous[0][0],3*count,"rigid cache modified history");
}

// Keep both PImpls alive across rebuilds. Each run owns fresh device inputs,
// so this also catches retained pointers and allocator-address reuse.
void rigid_cache_reuse_transitions(const GpuPipelineTestInput& input) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    constexpr std::size_t capacity=3;
    std::unique_ptr<GpuContactPipeline> uncached,cached;
    {RigidCacheSetting setting(false);uncached=std::make_unique<GpuContactPipeline>(capacity);}
    {RigidCacheSetting setting(true);cached=std::make_unique<GpuContactPipeline>(capacity);}
    struct Step {bool tiled,empty,reverse,swapped,translation_off,rotation_off,original,frozen;};
    const Step steps[]={
        {false,false,false,false,false,false,true,false},
        {true,false,true,true,true,false,false,true},
        {false,true,false,true,true,true,true,false},
        {false,false,true,true,false,true,true,true},
        {true,false,false,false,false,false,false,false},
        {false,false,false,true,false,false,true,false},
    };
    Derivatives seed[2];for(int body=0;body<2;++body)seed[body]=initial_system(body+8);
    std::size_t high_water_pairs=0;
    for(const auto step:steps) {
        MixedOptions option;option.capacity=capacity;option.seeded=true;option.pins=true;
        option.interleaved=step.tiled;option.empty_contacts=step.empty;
        option.reverse_contacts=step.reverse;option.swap_owners=step.swapped;
        option.disable_translation=step.translation_off;option.disable_rotation=step.rotation_off;
        option.original=step.original;option.frozen=step.frozen;
        GpuPipelineTestOutput expected,actual;ContactStorageBytes a,b;
        run_mixed(input,expected,option,&a,uncached.get());
        run_mixed(input,actual,option,&b,cached.get());
        require(expected.error==0&&actual.error==0,"rigid cache lifecycle fixture failed");
        compare_mixed_bytes(actual,expected);
        if(step.translation_off)
            same_bytes(actual.rigid_com,seed,2,"disabled translation changed seeded systems during cache reuse");
        if(step.rotation_off) {
            same_bytes(actual.rigid_omega,seed,2,"disabled rotation changed seeded systems during cache reuse");
            same_bytes(actual.rigid_omega_after,seed,2,"disabled repeated rotation changed systems during cache reuse");
        }
        high_water_pairs=std::max(high_water_pairs,std::min(capacity,step.empty?std::size_t(0):step.tiled?std::size_t(7):std::size_t(2)));
        require(a.packet_bytes<=packet_bound(capacity)&&b.packet_bytes<=packet_bound(capacity,true),
            "rigid cache lifecycle exceeded configured packet storage");
        require(b.packet_bytes==a.packet_bytes+high_water_pairs*8*sizeof(Derivatives),
            "rigid cache lifecycle did not account for retained owner-slot storage");
    }
    std::cout<<"GPU rigid cache lifecycle: same PImpl small/tiled/empty/reordered-owner/disabled-active and geometry-mode reuse PASS\n";
}

void late_tile_case(ipc_gpu::solver::ContactAssemblyPhase phase,bool original,bool frozen,
    int failure,std::size_t capacity,std::size_t repeat_prefix=1,
    ipc_gpu::solver::GpuContactPipeline* existing_pipeline=nullptr) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    constexpr int nodes=12;
    double positions[nodes][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0},
        {2,0,0},{3,0,0},{2.5,-.5,.17},{2.5,.5,.17},
        {4,0,0},{5,0,0},{4.5,-1,.17},{4.5,1,.17}};
    if(failure==1)positions[10][2]=positions[11][2]=0;
    double previous[nodes][3],references[nodes][3];
    const int owners[nodes]={-1,0,0,0,0,0,1,1,0,0,1,1},body_blocks[2]={12,13};
    const int body_nodes[11]={1,2,3,4,5,8,9,6,7,10,11};const std::size_t body_offsets[3]={0,7,11};
    int blocks[nodes],colors[14];std::fill(colors,colors+14,0);
    unsigned char deformable[nodes],solid[nodes]{},enabled[2]={1,1};
    const bool bodies=phase!=ContactAssemblyPhase::Deformable;
    for(int n=0;n<nodes;++n) {
        blocks[n]=bodies&&owners[n]>=0?12+owners[n]:n;
        deformable[n]=!bodies||owners[n]<0;
        for(int k=0;k<3;++k) {
            previous[n][k]=positions[n][k]-.001*(n+1)*(k+1);
            references[n][k]=positions[n][k]-(owners[n]==0?(k==0?.3:k==1?.3:0):(k==0?2.5:0));
        }
    }
    if(failure==2)previous[11][2]=std::numeric_limits<double>::quiet_NaN();
    const BpContact pairs[3]={{{0,1,2,3},0},{{4,5,6,7},1},{{8,9,10,11},1}};
    std::vector<BpContact> pair_values;
    for(std::size_t repeat=0;repeat<repeat_prefix;++repeat){pair_values.push_back(pairs[0]);pair_values.push_back(pairs[1]);}
    if(failure)pair_values.push_back(pairs[2]);
    const std::size_t count=pair_values.size();
    Derivatives seed[nodes],body_seed[2];for(int n=0;n<nodes;++n)seed[n]=initial_system(n);
    for(int b=0;b<2;++b)body_seed[b]=initial_system(b+12);
    Memory<double> x(3*nodes,&positions[0][0]),old(3*nodes,&previous[0][0]),reference(3*nodes,&references[0][0]);
    Memory<int> node_blocks(nodes,blocks),block_colors(14,colors),body_map(nodes,owners),d_body_blocks(2,body_blocks),d_body_nodes(11,body_nodes);
    Memory<std::size_t> d_body_offsets(3,body_offsets);
    Memory<unsigned char> d_deformable(nodes,deformable),d_solid(nodes,solid),d_enabled(2,enabled);
    Memory<BpContact> contacts(count,pair_values.data());Memory<Derivatives> systems(nodes,seed),body_systems(2,body_seed);Memory<int> failed(1);
    const double q[2][4]={{1,0,0,0},{1,0,0,0}},w[2][3]={{.13,-.21,.08},{.26,-.42,.16}};
    Memory<double> orientation(8,&q[0][0]),omega(6,&w[0][0]),com(6);
    DeformableScene scene{};scene.nodes=nodes;scene.node_blocks=node_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    RigidScene rigid{};RigidState rigid_state{};RigidWorkspace rigid_work{};
    if(bodies) {
        rigid.bodies=2;rigid.body_blocks=d_body_blocks.data;rigid.node_body=body_map.data;
        rigid.reference=reinterpret_cast<const double(*)[3]>(reference.data);rigid.node_offsets=d_body_offsets.data;rigid.nodes=d_body_nodes.data;
        rigid.translate=d_enabled.data;rigid.rotate=d_enabled.data;
        rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(orientation.data);rigid_state.orientation=rigid_state.previous_orientation;
        rigid_state.omega=reinterpret_cast<double(*)[3]>(omega.data);rigid_state.com=reinterpret_cast<double(*)[3]>(com.data);
        rigid_work.systems=body_systems.data;
    }
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=count;broad.block_colors=block_colors.data;broad.color_count=1;
    std::unique_ptr<GpuContactPipeline> owned_pipeline;
    if(!existing_pipeline)owned_pipeline=std::make_unique<GpuContactPipeline>(capacity);
    auto& pipeline=existing_pipeline?*existing_pipeline:*owned_pipeline;
    pipeline.rebuild(broad,bodies?14:nodes,node_blocks.data);
    const ContactParameters params{.8,37,.031,.2,.1,1e-12};
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.08;
    Memory<SdfCollider> d_collider(1,&collider);SdfScene sdf;
    // A NaN history is reserved for the last contact's validation, not SDF input.
    if(failure!=2) {
        SdfParameters p;p.stiffness=170;p.epsilon=.2;p.dt=.031;p.friction=.2;p.eps_v=.1;
        sdf={d_collider.data,1,p};
    }
    RigidPinConstraint pin[2];const std::size_t pin_offsets[3]={0,1,2};
    for(int body=0;body<2;++body) {
        pin[body].body=body;pin[body].local_position[0]=.13;pin[body].local_position[1]=-.07;
        pin[body].initial_target[0]=.4;pin[body].initial_target[1]=-.2;pin[body].initial_target[2]=.3;
    }
    Memory<RigidPinConstraint>d_pins(2,pin);Memory<std::size_t>d_pin_offsets(3,pin_offsets);RigidPinAssembly pins;
    if(bodies){pins.scene={2,2,d_pins.data,d_pin_offsets.data};pins.stiffness=83.;pins.time=.17;}
    pipeline.assemble_color(scene,state,work,rigid,rigid_state,rigid_work,sdf,params,0,phase,
        frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,pins,
        original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust);
    int status=0;failed.download(&status);Derivatives actual[nodes],actual_body[2];systems.download(actual);body_systems.download(actual_body);
    if(failure) {
        require(status==(failure==1?21:20),"late-tile failure status changed");
        same_bytes(actual,seed,nodes,"late-tile failure partially published node derivatives");
        same_bytes(actual_body,body_seed,2,"late-tile failure partially published rigid derivatives");
    } else {
        require(status==0,"positive late-tile control failed");
        require(bodies?std::memcmp(actual_body,body_seed,sizeof(body_seed))!=0:std::memcmp(actual,seed,sizeof(seed))!=0,
            "earlier active tiles did not contribute in the positive control");
    }
    double unchanged[3*nodes];x.download(unchanged);same_bytes(unchanged,&positions[0][0],3*nodes,"late-tile test modified positions");
    old.download(unchanged);same_bytes(unchanged,&previous[0][0],3*nodes,"late-tile test modified history");
}

__global__ void full_width_premise(ipc_gpu::solver::ContactInput input,int* diagnostics,
    std::uint32_t* words,int* slots) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    ipc_gpu::exact::DevicePool pool;ipc_gpu::exact::initialize_pool(pool,words,slots,1600,512);
    diagnostics[0]=evaluate_contact(input).status==ContactStatus::NeedsExactGeometry;
    diagnostics[1]=evaluate_contact_exact<256>(input).status==ContactStatus::ExactCapacityExceeded;
    diagnostics[2]=evaluate_contact_exact<1600>(input).status==ContactStatus::Ok;
    diagnostics[3]=!pool.failed&&pool.available==pool.slot_count;
    ipc_gpu::exact::release_pool();
}
void full_width_tile_case() {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    ContactInput in;in.kind=ContactKind::PointTriangle;in.d_hat=.8;in.k_barrier=37;in.dt=.031;in.friction=0;in.eps_v=.1;in.hessian_blocks=1;
    const double tiny[4][3]={{.25,.25,std::ldexp(1.,-500)},{0,0,0},{1,0,0},{0,1,0}};
    std::memcpy(in.positions,tiny,sizeof(tiny));std::memcpy(in.previous,tiny,sizeof(tiny));
    Memory<std::uint32_t> words(1600*512);Memory<int> slots(512),diagnostics(4);
    full_width_premise<<<1,1>>>(in,diagnostics.data,words.data,slots.data);check(cudaGetLastError());
    int premise[4];diagnostics.download(premise);for(int i=0;i<4;++i)require(premise[i]==1,"tiny PT did not exercise a healthy full-width exact retry");
    double positions[12][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0},
        {2,0,0},{3,0,0},{2.5,-.5,.17},{2.5,.5,.17}};
    std::memcpy(positions+8,tiny,sizeof(tiny));
    int blocks[12],colors[12]{};unsigned char deformable[12],solid[12]{};
    for(int n=0;n<12;++n){blocks[n]=n;deformable[n]=1;}
    const BpContact pairs[3]={{{4,5,6,7},1},{{0,1,2,3},0},{{8,9,10,11},0}};
    Memory<double> x(36,&positions[0][0]),old(36,&positions[0][0]);Memory<int> d_blocks(12,blocks),d_colors(12,colors);
    Memory<unsigned char> d_deformable(12,deformable),d_solid(12,solid);Memory<BpContact> contacts(3,pairs);Memory<int> failed(1);
    Memory<Derivatives> systems(12);DeformableScene scene{};scene.nodes=12;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=3;broad.block_colors=d_colors.data;broad.color_count=1;
    const ContactParameters params{.8,37,.031,0,.1,1e-12};Derivatives reference[12];
    for(std::size_t capacity:{std::size_t(3),std::size_t(1),std::size_t(2)}) {
        check(cudaMemset(systems.data,0,12*sizeof(Derivatives)));check(cudaMemset(failed.data,0,sizeof(int)));
        GpuContactPipeline pipeline(capacity);pipeline.rebuild(broad,12,d_blocks.data);
        pipeline.assemble_color(scene,state,work,{},{},{},{},params,0,ContactAssemblyPhase::Deformable,ContactAssemblyMode::FrozenSnapshot);
        int status=0;failed.download(&status);require(status==0,"tiled full-width exact retry failed");
        Derivatives actual[12];systems.download(actual);
        if(capacity==3)std::memcpy(reference,actual,sizeof(reference));
        else same_bytes(actual,reference,12,"tiling changed full-width exact results");
        require(pipeline.storage_bytes().packet_bytes<=packet_bound(capacity),"full-width packets exceeded tile storage bound");
    }
}

void reuse_transitions() {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    const double positions[4][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0}};
    const int blocks[4]={0,1,2,3},colors[4]={0,0,0,0};const unsigned char enabled[4]={1,1,1,1},solid[4]{};
    const BpContact pairs[7]={{{0,1,2,3},0},{{0,1,2,3},0},{{0,1,2,3},0},
        {{0,1,2,3},0},{{0,1,2,3},0},{{0,1,2,3},0},{{0,1,2,3},0}};
    Derivatives seed[4];for(int n=0;n<4;++n)seed[n]=initial_system(n);
    Memory<double>x(12,&positions[0][0]),old(12,&positions[0][0]);Memory<int>d_blocks(4,blocks),d_colors(4,colors),failed(1);
    Memory<unsigned char>d_deformable(4,enabled),d_solid(4,solid);Memory<BpContact>contacts(7,pairs);Memory<Derivatives>systems(4,seed);
    DeformableScene scene{};scene.nodes=4;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.block_colors=d_colors.data;broad.color_count=1;
    const ContactParameters params{.8,37,.031,.2,.1,1e-12};GpuContactPipeline reused(3);
    struct Step{std::size_t count;bool original,frozen;};
    const Step steps[]={{2,true,false},{7,false,true},{2,true,false},{7,true,true},{2,false,false},{7,true,false}};
    for(const auto step:steps) {
        broad.contact_count=step.count;
        Derivatives expected[4];
        for(bool fresh:{true,false}) {
            check(cudaMemcpy(systems.data,seed,sizeof(seed),cudaMemcpyHostToDevice));check(cudaMemset(failed.data,0,sizeof(int)));
            GpuContactPipeline reference(32);auto& pipeline=fresh?reference:reused;
            pipeline.rebuild(broad,4,d_blocks.data);
            pipeline.assemble_color(scene,state,work,{},{},{},{},params,0,ContactAssemblyPhase::Deformable,
                step.frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},
                step.original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust);
            int status=0;failed.download(&status);require(status==0,"contact memory reuse transition failed");
            Derivatives actual[4];systems.download(actual);
            if(fresh)std::memcpy(expected,actual,sizeof(expected));
            else {
                same_bytes(actual,expected,4,"small/tiled or geometry-policy reuse changed derivatives");
                require(pipeline.storage_bytes().packet_bytes<=packet_bound(3),"reused packet storage exceeded capacity");
                if(step.original&&!step.frozen)require(pipeline.storage_bytes().exact_bytes==0,"Original GS retained unused exact arenas after robust assembly");
            }
        }
    }
}

void packet_memory_scaling() {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    constexpr std::size_t pairs_count=32777,tile_capacity=127;
    const double positions[4][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0}};
    const int blocks[4]={0,1,2,3},colors[4]={0,1,1,1};const unsigned char deformable[4]={1,1,1,1},solid[4]{};
    std::vector<BpContact> pairs(pairs_count,BpContact{{0,1,2,3},0});
    Memory<double> x(12,&positions[0][0]),old(12,&positions[0][0]);Memory<int>d_blocks(4,blocks),d_colors(4,colors);
    Memory<unsigned char>d_deformable(4,deformable),d_solid(4,solid);Memory<BpContact>contacts(pairs.size(),pairs.data());
    Memory<Derivatives>systems(4);Memory<int>failed(1);
    DeformableScene scene{};scene.nodes=4;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=pairs.size();broad.block_colors=d_colors.data;broad.color_count=2;
    const ContactParameters params{.8,37,.031,0,.1,1e-12};Derivatives tiled_result[4];
    std::size_t small_packet=0,big_packet=0,small_device=0,big_device=0;
    for(std::size_t capacity:{tile_capacity,pairs_count}) {
        std::size_t before=0,total=0,after=0;check(cudaMemGetInfo(&before,&total));
        {
            GpuContactPipeline pipeline(capacity);pipeline.rebuild(broad,4,d_blocks.data);
            require(pipeline.storage_bytes().exact_bytes==0,"Original GS rebuild eagerly allocated exact arenas");
            check(cudaMemset(systems.data,0,4*sizeof(Derivatives)));check(cudaMemset(failed.data,0,sizeof(int)));
            pipeline.assemble_color(scene,state,work,{},{},{},{},params,0,ContactAssemblyPhase::Deformable,
                ContactAssemblyMode::ColorSweep,{},ContactGeometryPolicy::Original);
            int status=0;failed.download(&status);require(status==0,"memory-scaling contact assembly failed");
            Derivatives actual[4];systems.download(actual);
            const auto storage=pipeline.storage_bytes();
            require(storage.exact_bytes==0,"Original GS allocated unused exact scratch");
            require(storage.packet_bytes<=packet_bound(capacity),"contact packet allocation grew with total candidate count");
            check(cudaMemGetInfo(&after,&total));
            if(capacity==tile_capacity){small_packet=storage.packet_bytes;small_device=before>after?before-after:0;std::memcpy(tiled_result,actual,sizeof(actual));}
            else{big_packet=storage.packet_bytes;big_device=before>after?before-after:0;same_bytes(actual,tiled_result,4,"bounded packet storage dropped or reordered candidates");}
        }
    }
    require(big_packet>small_packet*100,"packet storage did not materially scale with tile budget");
    require(big_device>small_device+16*1024*1024,"observed GPU allocation did not fall with bounded packets");
    std::cout<<"GPU contact storage: "<<pairs_count<<" retained candidates; packet bytes "<<big_packet<<" -> "<<small_packet
        <<", observed device bytes "<<big_device<<" -> "<<small_device<<"; ordered outputs identical\n";
}
}

void gpu_contact_packet_memory_test_gpu(const GpuPipelineTestInput* inputs,std::size_t count) {
    RigidCompactSetting compact(false);
    ColorLocalSetting exact(false); // Preserve existing allocation and equality gates.
    using namespace ipc_gpu::solver;
    bool rejected=false;try{GpuContactPipeline invalid(0);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"zero contact packet budget was accepted");
    std::size_t checked=0;
    for(std::size_t i=0;i<count;++i)for(bool original:{false,true})for(bool frozen:{false,true})for(bool pins:{false,true}) {
        MixedOptions option;option.interleaved=true;option.original=original;option.frozen=frozen;option.pins=pins;option.seeded=true;option.capacity=32;
        GpuPipelineTestOutput reference;
        {RigidCacheSetting setting(false);run_mixed(inputs[i],reference,option);}
        require(reference.error==0,"untiled memory-acceptance fixture failed");
        for(std::size_t capacity:{std::size_t(1),std::size_t(2),std::size_t(3),std::size_t(5),std::size_t(32)}) {
            std::size_t uncached_bytes=0;
            for(bool cache:{false,true}) {
                RigidCacheSetting setting(cache);
                option.capacity=capacity;GpuPipelineTestOutput actual;ContactStorageBytes storage;
                run_mixed(inputs[i],actual,option,&storage);compare_mixed_bytes(actual,reference);
                require(storage.packet_bytes<=packet_bound(capacity,cache),"mixed packets exceeded configured tile capacity");
                if(!cache)uncached_bytes=storage.packet_bytes;
                else require(storage.packet_bytes==uncached_bytes+std::min(capacity,std::size_t(7))*8*sizeof(ipc_gpu::Derivatives),
                    "rigid owner-slot cache storage was not fully accounted");
                ++checked;
            }
        }
    }
    for(bool cache:{false,true}) {
        RigidCacheSetting setting(cache);
        for(auto phase:{ContactAssemblyPhase::Deformable,ContactAssemblyPhase::RigidTranslation,ContactAssemblyPhase::RigidOrientation})
            for(bool original:{false,true})for(bool frozen:{false,true}) {
                late_tile_case(phase,original,frozen,0,1);
                for(int failure:{1,2})for(std::size_t capacity:{std::size_t(1),std::size_t(3)})
                    late_tile_case(phase,original,frozen,failure,capacity);
            }
    }
    full_width_tile_case();reuse_transitions();packet_memory_scaling();rigid_cache_many_owners();
    require(count>0,"rigid cache lifecycle needs a valid mixed fixture");
    rigid_cache_reuse_transitions(inputs[count-1]);
    std::cout<<"GPU bounded contacts: "<<checked<<" mixed bitwise comparisons; late-tile failure publication, exact-capacity retry, SDF/friction/pins and immutable snapshots PASS\n";
}

namespace {
struct LocalPacketOptions {
    bool original=false,frozen=false,collapse_colors=false,pruned=false;
    bool tiny=false,reverse=false,swap_colors=false,empty=false,only_first=false;
    bool missing_map=false,two_faults=false,combined_sdf_fault=false;
    int fault=0;
    std::size_t capacity=4,decoys=5;
};
struct LocalPacketRun {
    std::vector<ipc_gpu::Derivatives> snapshots;
    std::vector<int> statuses;
    std::vector<std::size_t> counts;
    ipc_gpu::solver::ContactStorageBytes storage;
    std::size_t global_count=0;
};
LocalPacketRun run_local_packet_case(LocalPacketOptions option,bool enabled,
    ipc_gpu::solver::GpuContactPipeline* existing=nullptr) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    ColorLocalSetting selected(enabled);
    constexpr int nodes=12;
    double positions[nodes][3]={{.15,.10,.13},{.25,.12,.15},{.35,.14,.17},{.4,.04,.19},
        {0,0,0},{1,0,0},{0,1,0},{4.2,.3,4},{4,0,0},{5,0,0},{4,1,0},{2,.04,2}};
    if(option.tiny){positions[2][0]=.25;positions[2][1]=.25;positions[2][2]=std::ldexp(1.,-500);}
    if(option.fault==1)positions[2][2]=0;
    if(option.two_faults)positions[0][2]=0;
    double previous[nodes][3];
    for(int n=0;n<nodes;++n)for(int k=0;k<3;++k)previous[n][k]=positions[n][k]-.001*(n+1)*(k+1);
    if(option.fault==2)previous[2][0]=std::numeric_limits<double>::quiet_NaN();
    if(option.two_faults)previous[1][0]=std::numeric_limits<double>::quiet_NaN();
    int blocks[nodes],colors[nodes]={0,1,2,1,-1,-1,-1,-1,-1,-1,-1,3};
    unsigned char deformable[nodes]{},solid[nodes]{};
    for(int n=0;n<nodes;++n){blocks[n]=n;deformable[n]=n<4||n==11||option.pruned;}
    if(option.pruned)for(int n=4;n<=10;++n)colors[n]=n;
    if(option.swap_colors)std::swap(colors[0],colors[2]);
    if(option.collapse_colors)for(int n=0;n<nodes;++n)if(deformable[n])colors[n]=0;
    const int color_count=option.pruned?11:option.collapse_colors?1:4;
    std::vector<int> color_blocks,color_offsets;
    for(int color=0;color<color_count;++color) {
        color_offsets.push_back(int(color_blocks.size()));
        for(int block=0;block<nodes;++block)if(colors[block]==color)color_blocks.push_back(block);
    }
    color_offsets.push_back(int(color_blocks.size()));
    std::vector<BpContact> pairs;
    for(int i=0;i<3;++i)pairs.push_back({{0,4,5,6},0});
    for(int i=0;i<6;++i)pairs.push_back({{i%2?3:1,4,5,6},0});
    for(int i=0;i<2;++i)pairs.push_back({{2,4,5,6},0});
    for(std::size_t i=0;i<option.decoys;++i)pairs.push_back({{7,8,9,10},0});
    if(option.two_faults)pairs.push_back({{0,1,4,5},1});
    if(option.fault==3)pairs[1].kind=7;
    if(option.fault==4)pairs.back().kind=7;
    if(option.reverse)std::reverse(pairs.begin(),pairs.end());
    if(option.empty)pairs.clear();
    Derivatives seeds[nodes];for(int n=0;n<nodes;++n)seeds[n]=initial_system(n);
    std::vector<BpAabb> boxes(nodes);
    for(int n=0;n<nodes;++n)for(int k=0;k<3;++k){boxes[n].lo[k]=positions[n][k]-.001;boxes[n].hi[k]=positions[n][k]+.001;}
    Memory<double>x(3*nodes,&positions[0][0]),old(3*nodes,&previous[0][0]);
    Memory<int>d_blocks(nodes,blocks),d_colors(nodes,colors),d_color_blocks(std::max<std::size_t>(1,color_blocks.size()),color_blocks.empty()?nullptr:color_blocks.data()),
        d_color_offsets(color_offsets.size(),color_offsets.data()),failed(1);
    Memory<unsigned char>d_deformable(nodes,deformable),d_solid(nodes,solid);
    Memory<BpContact>contacts(std::max<std::size_t>(1,pairs.size()),pairs.empty()?nullptr:pairs.data());
    Memory<BpAabb>d_boxes(nodes,boxes.data());Memory<Derivatives>systems(nodes,seeds);
    DeformableScene scene{};scene.nodes=nodes;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=pairs.size();broad.block_colors=d_colors.data;
    broad.color_count=color_count;broad.color_blocks=d_color_blocks.data;broad.color_offsets=d_color_offsets.data;
    const ContactParameters parameters{.8,37,.031,0,.1,1e-12};
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.08;
    Memory<SdfCollider>d_collider(1,&collider);SdfParameters sdf_parameters;
    sdf_parameters.stiffness=170;sdf_parameters.epsilon=.2;sdf_parameters.dt=.031;sdf_parameters.friction=option.fault==2&&!option.combined_sdf_fault?0.0:0.1;sdf_parameters.eps_v=.1;
    SdfScene sdf{d_collider.data,1,sdf_parameters};
    std::unique_ptr<GpuContactPipeline> owned;
    if(!existing)owned=std::make_unique<GpuContactPipeline>(option.capacity);
    auto& pipeline=existing?*existing:*owned;
    ContactBoxPruning pruning;
    if(option.pruned)pruning={d_boxes.data,d_deformable.data,reinterpret_cast<const double(*)[3]>(x.data),
        reinterpret_cast<const double(*)[3]>(old.data),parameters,option.original};
    pipeline.rebuild(broad,nodes,d_blocks.data,nullptr,pruning);
    LocalPacketRun result;result.global_count=pairs.size();
    const auto incidence=pipeline.incidence();
    require(incidence.color_pairs.offsets!=nullptr,"color-local acceptance needs indexed worklists");
    std::vector<std::size_t> offsets(std::size_t(color_count)+1);
    check(cudaMemcpy(offsets.data(),incidence.color_pairs.offsets,offsets.size()*sizeof(std::size_t),cudaMemcpyDeviceToHost));
    for(int color=0;color<color_count;++color)result.counts.push_back(offsets[color+1]-offsets[color]);
    if(option.missing_map) {
        require(!option.reverse&&!option.pruned&&!option.collapse_colors&&result.counts[0]==3,"invalid missing-map premise");
        const int foreign=11;
        check(cudaMemcpy(const_cast<int*>(incidence.color_pairs.pair_indices)+offsets[1]-1,&foreign,sizeof(foreign),cudaMemcpyHostToDevice));
    }
    std::vector<Derivatives> before(seeds,seeds+nodes),after(nodes);
    for(int color=0;color<(option.only_first?1:color_count);++color) {
        pipeline.assemble_color(scene,state,work,{},{},{},sdf,parameters,color,ContactAssemblyPhase::Deformable,
            option.frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},
            option.original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust);
        int status=0;failed.download(&status);systems.download(after.data());
        result.statuses.push_back(status);result.snapshots.insert(result.snapshots.end(),after.begin(),after.end());
        if(status)same_bytes(after.data(),before.data(),nodes,"color-local failure published partial systems");
        else before=after;
    }
    result.storage=pipeline.storage_bytes();
    double unchanged[3*nodes];x.download(unchanged);same_bytes(unchanged,&positions[0][0],3*nodes,"color-local packets changed positions");
    old.download(unchanged);same_bytes(unchanged,&previous[0][0],3*nodes,"color-local packets changed previous positions");
    return result;
}
void compare_local_packet_runs(const LocalPacketRun& actual,const LocalPacketRun& expected) {
    require(actual.statuses==expected.statuses,"color-local packets changed single-fault statuses");
    require(actual.counts==expected.counts&&actual.global_count==expected.global_count,"color-local packets changed candidate/worklist membership");
    require(actual.snapshots.size()==expected.snapshots.size(),"color-local snapshot count changed");
    same_bytes(actual.snapshots.data(),expected.snapshots.data(),actual.snapshots.size(),"color-local packets changed ordered systems");
}
}

namespace {
void assert_color_local_fault_premise() {
    using namespace ipc_gpu::solver;
    const double point[4][3]={{.35,.14,.17},{0,0,0},{1,0,0},{0,1,0}};
    ContactInput input;input.kind=ContactKind::PointTriangle;
    std::memcpy(input.positions,point,sizeof(point));std::memcpy(input.previous,point,sizeof(point));
    input.previous[0][0]=std::numeric_limits<double>::quiet_NaN();
    input.d_hat=.8;input.k_barrier=37;input.dt=.031;input.friction=0;
    input.eps_v=.1;input.eps=1e-12;input.hessian_blocks=1;
    require(evaluate_contact(input).status==ContactStatus::InvalidInput,
        "late-fault premise lost Robust mesh InvalidInput");
    require(evaluate_contact_original(input).status==ContactStatus::InvalidInput,
        "late-fault premise lost Original mesh InvalidInput");
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.08;
    SdfParameters sdf;sdf.stiffness=170;sdf.epsilon=.2;sdf.dt=.031;sdf.friction=.1;sdf.eps_v=.1;
    require(evaluate_sdf_contact(&collider,1,point[0],input.previous[0],sdf).status==SdfStatus::InvalidInput,
        "combined late-fault premise lost SDF invalid history");
    sdf.friction=0;
    require(evaluate_sdf_contact(&collider,1,point[0],input.previous[0],sdf).status==SdfStatus::Ok,
        "mesh-only late-fault premise still has an invalid SDF");
}
}

void gpu_contact_color_local_packets_test_gpu(const GpuPipelineTestInput* inputs,std::size_t count) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    RigidCompactSetting compact_isolation(false);
    assert_color_local_fault_premise();
    ContactOptionSetting queues("IPC_GPU_COLOR_PAIR_QUEUES","1"),parallel("IPC_GPU_RIGID_PARALLEL_REDUCE","0"),
        pruning("IPC_GPU_BOX_CONTACT_PRUNING","1"),warp("IPC_GPU_WARP_CONTACT_REDUCE","0"),
        compact_warp("IPC_GPU_COMPACT_WARP_REDUCE","0"),prefetch("IPC_GPU_CONTACT_PREFETCH_REDUCE","0"),
        component("IPC_GPU_CONTACT_COMPONENT_REDUCE","0");
    std::size_t checks=0;
    for(bool compact:{false,true})for(bool cooperative:{false,true}) {
        ContactOptionSetting compact_flag("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0"),
            selected("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",compact?"1":"0"),
            cooperative_flag("IPC_GPU_COOPERATIVE_EXACT_CONTACT",cooperative?"1":"0"),
            activity("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",compact?"1":"0"),
            deferred("IPC_GPU_DEFER_CONTACT_INPUTS",cooperative?"1":"0");
        for(std::size_t i=0;i<count;++i)for(bool original:{false,true})for(bool frozen:{false,true})for(bool pins:{false,true})for(bool cache:{false,true}) {
            RigidCacheSetting cache_flag(cache);MixedOptions option;
            option.capacity=5;option.interleaved=true;option.local_colors=true;option.original=original;
            option.frozen=frozen;option.pins=pins;option.seeded=true;
            GpuPipelineTestOutput expected,actual;ContactStorageBytes storage;std::vector<std::size_t> colors;
            {ColorLocalSetting disabled(false);run_mixed(inputs[i],expected,option);}
            {ColorLocalSetting enabled(true);run_mixed(inputs[i],actual,option,&storage,nullptr,&colors);}
            compare_mixed_bytes(actual,expected);
            if(!frozen)require(colors==std::vector<std::size_t>({4,7,5}),"mixed fixture did not exercise small/big/small color counts");
            require(storage.packet_bytes<=packet_bound(option.capacity,cache)+7*sizeof(int),"color-local mixed map exceeded packet accounting");
            ++checks;
        }
        for(bool original:{false,true})for(bool frozen:{false,true})for(bool retired:{false,true}) {
            LocalPacketOptions option;option.original=original;option.frozen=frozen;option.pruned=retired;
            const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true);
            compare_local_packet_runs(actual,expected);
            require(actual.global_count>option.capacity&&actual.counts[0]<=option.capacity&&actual.counts[1]>option.capacity
                &&actual.counts[2]<=option.capacity,"deformable fixture lost its local/fallback/local premise");
            if(retired)for(int color=7;color<=10;++color)require(actual.counts[color]==0,"far pairs were not pruned from retained color queues");
            require(actual.storage.packet_bytes<=packet_bound(option.capacity)+actual.global_count*sizeof(int),"color-local deformable map exceeded packet accounting");
            ++checks;
        }
        for(bool frozen:{false,true}) {
            LocalPacketOptions exact;exact.tiny=true;exact.frozen=frozen;
            compare_local_packet_runs(run_local_packet_case(exact,true),run_local_packet_case(exact,false));++checks;
            LocalPacketOptions collapsed;collapsed.frozen=true;collapsed.original=frozen;collapsed.collapse_colors=true;collapsed.capacity=12;
            const auto expected=run_local_packet_case(collapsed,false),actual=run_local_packet_case(collapsed,true);
            compare_local_packet_runs(actual,expected);
            require(actual.global_count>collapsed.capacity&&actual.counts[0]==11,"frozen single-color fixture did not fit its local packet");++checks;
        }
    }
    // Exact full-width premise for the tiny retained PT in color2.
    {ContactInput in;in.kind=ContactKind::PointTriangle;in.d_hat=.8;in.k_barrier=37;in.dt=.031;in.friction=0;in.eps_v=.1;in.hessian_blocks=1;
        const double tiny[4][3]={{.25,.25,std::ldexp(1.,-500)},{0,0,0},{1,0,0},{0,1,0}};
        std::memcpy(in.positions,tiny,sizeof(tiny));std::memcpy(in.previous,tiny,sizeof(tiny));
        Memory<std::uint32_t> words(1600*512);Memory<int> slots(512),diagnostics(4);
        full_width_premise<<<1,1>>>(in,diagnostics.data,words.data,slots.data);check(cudaGetLastError());
        int premise[4];diagnostics.download(premise);for(int x:premise)require(x==1,"color-local exact fixture did not require healthy full-width retry");}
    // Rebuild/reuse across local, all-global fallback, pruned, empty, and
    // reordered queues. Fresh input allocations also exercise pointer changes.
    for(bool compact:{false,true}) {
        ContactOptionSetting compact_flag("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0"),
            selected("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",compact?"1":"0");
        std::unique_ptr<GpuContactPipeline> reused;
        {ColorLocalSetting enabled(true);reused=std::make_unique<GpuContactPipeline>(4);}
        for(int step=0;step<8;++step) {
            LocalPacketOptions option;option.reverse=step&1;option.swap_colors=step%3==1;option.pruned=step==3||step==6;
            option.frozen=step==2||step==5;option.collapse_colors=step==2;option.original=step%3==0;option.empty=step==4;
            const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true,reused.get());
            compare_local_packet_runs(actual,expected);
            require(actual.storage.packet_bytes<=packet_bound(4)+16*sizeof(int),"reused color-local map exceeded high-water packet accounting");++checks;
        }
    }
    for(bool original:{false,true})for(bool frozen:{false,true})for(int fault:{1,2}) {
        LocalPacketOptions option;option.original=original;option.frozen=frozen;option.fault=fault;
        const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true);
        compare_local_packet_runs(actual,expected);
        require(actual.statuses.size()>2&&actual.statuses[2]==(fault==1?21:20),"color-local mesh-only late fault did not preserve its code");++checks;
    }
    // Preserve the original combined invalid mesh/SDF history as an explicit
    // priority test: Frozen seeds SDF first; colored GS encounters mesh first.
    for(bool original:{false,true})for(bool frozen:{false,true}) {
        LocalPacketOptions option;option.original=original;option.frozen=frozen;
        option.fault=2;option.combined_sdf_fault=true;
        const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true);
        compare_local_packet_runs(actual,expected);
        const int expected_code=frozen?25:20;
        require(actual.statuses.size()>2&&actual.statuses[0]==0&&actual.statuses[1]==0
            &&actual.statuses[2]==expected_code,
            "combined late mesh/SDF fault changed its exact seed/finish priority");++checks;
    }
    for(bool original:{false,true})for(int fault:{3,4}) {
        LocalPacketOptions option;option.fault=fault;option.original=original;option.only_first=true;
        const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true);
        compare_local_packet_runs(actual,expected);
        require(actual.statuses[0]==20,"mapped malformed contact kind bypassed invalid-input error");++checks;
    }
    // Concurrent force failures can arbitrate differently after regrouping;
    // this does not relax the separate CCD first-unsafe ordering contract.
    {LocalPacketOptions option;option.two_faults=true;option.only_first=true;
        const auto ordered=run_local_packet_case(option,false),local=run_local_packet_case(option,true);
        require(ordered.statuses[0]==21,"ordered two-fault premise lost its first PT failure");
        require(local.statuses[0]==20||local.statuses[0]==21,"color-local two-fault result was not a known failure");++checks;}
    for(int reducer=0;reducer<6;++reducer) {
        ContactOptionSetting compact_flag("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",reducer==2||reducer==5?"1":"0"),
            warp_flag("IPC_GPU_WARP_CONTACT_REDUCE",reducer==1?"1":"0"),
            selected("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",reducer==2||reducer==5?"1":"0"),
            compact_warp_flag("IPC_GPU_COMPACT_WARP_REDUCE",reducer==5?"1":"0"),
            compact_cutoff("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE","1"),
            component_flag("IPC_GPU_CONTACT_COMPONENT_REDUCE",reducer==3?"1":"0"),
            prefetch_flag("IPC_GPU_CONTACT_PREFETCH_REDUCE",reducer==4?"1":"0"),
            grain("IPC_GPU_CONTACT_COMPONENT_GRAIN","1");
        LocalPacketOptions healthy;
        const auto expected=run_local_packet_case(healthy,false),actual=run_local_packet_case(healthy,true);
        compare_local_packet_runs(actual,expected);++checks;
        LocalPacketOptions option;option.missing_map=true;option.only_first=true;
        const auto result=run_local_packet_case(option,true);
        require(result.statuses[0]==20,"missing local slot did not fail closed for a reducer");++checks;
    }
    // The retained inverse map is part of packet accounting. Make it larger
    // than the whole old packet budget so omitting it cannot pass this check.
    {LocalPacketOptions option;option.decoys=4086;option.only_first=true;
        ContactOptionSetting compact_flag("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS","0"),
            cooperative_flag("IPC_GPU_COOPERATIVE_EXACT_CONTACT","0"),selected("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE","0");
        const auto expected=run_local_packet_case(option,false),actual=run_local_packet_case(option,true);
        compare_local_packet_runs(actual,expected);
        require(actual.global_count*sizeof(int)>packet_bound(4),"map accounting fixture is too small");
        require(actual.storage.packet_bytes>=actual.global_count*sizeof(int)
            &&actual.storage.packet_bytes<=packet_bound(4)+actual.global_count*sizeof(int),"global local-slot map is absent from packet accounting");++checks;}
    std::cout<<"GPU color-local packets: "<<checks<<" ordered mixed/local/fallback, frozen/pruned, exact/compact/selected, reuse, missing-map and error-publication cases PASS\n";
}

void gpu_contact_original_frozen_tiled_test_gpu(const double (*captured)[3],const double (*history)[3],
    const ipc_gpu::Derivatives* expected_captured) {
    ColorLocalSetting exact(false);
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    double positions[8][3]{},previous[8][3]{};
    std::memcpy(positions,captured,12*sizeof(double));std::memcpy(previous,history,12*sizeof(double));
    const double background[4][3]={{.002,.003,.0017},{0,0,0},{.01,0,0},{0,.01,0}};
    for(int role=0;role<4;++role)for(int k=0;k<3;++k) {
        positions[role+4][k]=background[role][k];previous[role+4][k]=background[role][k]-.0001*(role+1)*(k+1);
    }
    int blocks[8],colors[8]{};unsigned char deformable[8],solid[8]{};
    for(int n=0;n<8;++n){blocks[n]=n;deformable[n]=1;}
    const BpContact pairs[2]={{{4,5,6,7},0},{{0,1,2,3},1}};
    Memory<double>x(24,&positions[0][0]),old(24,&previous[0][0]);Memory<int>d_blocks(8,blocks),d_colors(8,colors),failed(1);
    Memory<unsigned char>d_deformable(8,deformable),d_solid(8,solid);Memory<BpContact>contacts(2,pairs);Memory<Derivatives>systems(8);
    DeformableScene scene{};scene.nodes=8;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=2;broad.block_colors=d_colors.data;broad.color_count=1;
    const ContactParameters params{.005,37,.031,.2,.1,1e-12};Derivatives untiled[8];
    for(std::size_t capacity:{std::size_t(2),std::size_t(1)}) {
        check(cudaMemset(systems.data,0,8*sizeof(Derivatives)));check(cudaMemset(failed.data,0,sizeof(int)));
        GpuContactPipeline pipeline(capacity);pipeline.rebuild(broad,8,d_blocks.data);
        pipeline.assemble_color(scene,state,work,{},{},{},{},params,0,ContactAssemblyPhase::Deformable,
            ContactAssemblyMode::FrozenSnapshot,{},ContactGeometryPolicy::Original);
        int status=0;failed.download(&status);require(status==0,"Original frozen EE retry failed across a tile boundary");
        Derivatives actual[8];systems.download(actual);same_bytes(actual,expected_captured,4,"tiled Original exact retry changed CPU-qualified captured derivatives");
        if(capacity==2)std::memcpy(untiled,actual,sizeof(actual));
        else same_bytes(actual,untiled,8,"Original exact retry changed other tile contributions");
    }
    double unchanged[24];x.download(unchanged);same_bytes(unchanged,&positions[0][0],24,"tiled Original retry changed positions");
    old.download(unchanged);same_bytes(unchanged,&previous[0][0],24,"tiled Original retry changed history");
}

namespace {
std::size_t rigid_compact_bytes(std::size_t bodies,std::size_t capacity) {
    return bodies*((capacity+127)/128)*(128+1)*sizeof(std::uint32_t);
}
__global__ void compact_activity_premise(GpuPipelineTestInput input,int* results) {
    using namespace ipc_gpu::solver;
    const int point[2][4]={{0,1,2,3},{6,1,2,3}};
    for(int fixture=0;fixture<2;++fixture)for(int original=0;original<2;++original) {
        ContactInput in;in.kind=ContactKind::PointTriangle;in.d_hat=.2;in.k_barrier=370;
        in.dt=.031;in.friction=.2;in.eps_v=.1;in.hessian_blocks=1;
        for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
            in.positions[role][axis]=input.positions[point[fixture][role]][axis];
            in.previous[role][axis]=input.previous[point[fixture][role]][axis];
        }
        const auto out=original?evaluate_contact_original(in):evaluate_contact(in);
        results[2*fixture+original]=out.status==ContactStatus::Ok&&out.active==(fixture==0);
    }
}
struct ManyCompactOutcome {
    std::vector<ipc_gpu::Derivatives> translation,orientation;
    ipc_gpu::solver::ContactStorageBytes storage{};
};
ManyCompactOutcome compact_many_run(ipc_gpu::solver::GpuContactPipeline& pipeline,std::size_t bodies,
    std::size_t capacity,bool empty=false,bool disabled=false) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    constexpr std::size_t nodes=4;
    const double positions[nodes][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0}};
    double previous[nodes][3],reference[nodes][3];
    for(std::size_t node=0;node<nodes;++node)for(int axis=0;axis<3;++axis) {
        previous[node][axis]=positions[node][axis]-.001*(node+1)*(axis+1);
        reference[node][axis]=positions[node][axis]+.03*(node+1)*(axis+1);
    }
    const int owners[nodes]={0,1,2,3},body_nodes[nodes]={0,1,2,3};
    const unsigned char deformable[nodes]{},solid[nodes]{};
    std::vector<int> colors(bodies,0),body_blocks(bodies);
    std::vector<std::size_t> offsets(bodies+1);
    std::vector<unsigned char> enabled(bodies,0);
    std::vector<double> q(4*bodies,0.),omega(3*bodies,0.);
    std::vector<Derivatives> seeds(bodies);
    for(std::size_t body=0;body<bodies;++body) {
        body_blocks[body]=int(body);offsets[body]=std::min(body,nodes);q[4*body]=1.;
        enabled[body]=static_cast<unsigned char>(body<nodes&&!disabled);seeds[body]=initial_system(int(body)+4);
        for(int axis=0;axis<3;++axis)omega[3*body+axis]=body<nodes?.07*(body+1)*(axis-1):0.;
    }
    offsets[bodies]=nodes;
    std::vector<BpContact> pair_values(empty?0:capacity+2,BpContact{{0,1,2,3},0});
    Memory<double> x(3*nodes,&positions[0][0]),old(3*nodes,&previous[0][0]),refs(3*nodes,&reference[0][0]);
    Memory<int> d_owners(nodes,owners),d_nodes(nodes,body_nodes),d_blocks(bodies,body_blocks.data()),d_colors(bodies,colors.data()),failed(1);
    Memory<std::size_t> d_offsets(offsets.size(),offsets.data());
    Memory<unsigned char> d_deformable(nodes,deformable),d_solid(nodes,solid),d_enabled(bodies,enabled.data());
    Memory<BpContact> contacts(std::max<std::size_t>(1,pair_values.size()),pair_values.empty()?nullptr:pair_values.data());
    Derivatives node_seeds[nodes];for(std::size_t node=0;node<nodes;++node)node_seeds[node]=initial_system(int(node));
    Memory<Derivatives> node_systems(nodes,node_seeds),systems(bodies,seeds.data());
    Memory<double> d_q(q.size(),q.data()),d_omega(omega.size(),omega.data()),com(3*bodies);
    DeformableScene scene{};scene.nodes=nodes;scene.node_blocks=d_owners.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace elastic{};elastic.systems=node_systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=bodies;rigid.body_blocks=d_blocks.data;rigid.node_body=d_owners.data;
    rigid.reference=reinterpret_cast<const double(*)[3]>(refs.data);rigid.node_offsets=d_offsets.data;rigid.nodes=d_nodes.data;
    rigid.translate=d_enabled.data;rigid.rotate=d_enabled.data;
    RigidState rigid_state{};rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(d_q.data);rigid_state.orientation=rigid_state.previous_orientation;
    rigid_state.omega=reinterpret_cast<double(*)[3]>(d_omega.data);rigid_state.com=reinterpret_cast<double(*)[3]>(com.data);
    RigidWorkspace workspace{};workspace.systems=systems.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.contact_count=pair_values.size();broad.block_colors=d_colors.data;broad.color_count=1;
    pipeline.rebuild(broad,int(bodies),d_owners.data);
    const ContactParameters params{.8,37,.031,.2,.1,1e-12};ManyCompactOutcome result;
    for(int phase=0;phase<2;++phase) {
        check(cudaMemcpy(systems.data,seeds.data(),bodies*sizeof(Derivatives),cudaMemcpyHostToDevice));check(cudaMemset(failed.data,0,sizeof(int)));
        pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,workspace,{},params,0,
            phase?ContactAssemblyPhase::RigidOrientation:ContactAssemblyPhase::RigidTranslation,ContactAssemblyMode::FrozenSnapshot);
        int status=0;failed.download(&status);require(status==0,"many-body rigid compact fixture failed");
        auto& values=phase?result.orientation:result.translation;values.resize(bodies);systems.download(values.data());
        for(std::size_t body=0;body<bodies;++body)if(body>=nodes||disabled)
            same_bytes(values.data()+body,seeds.data()+body,1,"rigid compact touched a disabled body phase");
    }
    result.storage=pipeline.storage_bytes();
    double unchanged[3*nodes];x.download(unchanged);same_bytes(unchanged,&positions[0][0],3*nodes,"rigid compact modified positions");
    old.download(unchanged);same_bytes(unchanged,&previous[0][0],3*nodes,"rigid compact modified history");
    return result;
}
void compare_many_compact(const ManyCompactOutcome& actual,const ManyCompactOutcome& expected) {
    require(actual.translation.size()==expected.translation.size()&&actual.orientation.size()==expected.orientation.size(),"rigid compact body count changed");
    same_bytes(actual.translation.data(),expected.translation.data(),actual.translation.size(),"rigid compact many-body translation bits changed");
    same_bytes(actual.orientation.data(),expected.orientation.data(),actual.orientation.size(),"rigid compact many-body orientation bits changed");
    require(actual.storage.exact_bytes==expected.storage.exact_bytes,"rigid compact changed exact arena storage");
}
void compact_many_body_boundaries() {
    using namespace ipc_gpu::solver;
    constexpr std::size_t capacity=129;
    for(bool cache:{false,true}) {
        RigidCacheSetting cache_setting(cache);
        for(std::size_t bodies:{std::size_t(64),std::size_t(65)}) {
            std::unique_ptr<GpuContactPipeline> off,on;
            {RigidCompactSetting setting(false);off=std::make_unique<GpuContactPipeline>(capacity);}
            {RigidCompactSetting setting(true);on=std::make_unique<GpuContactPipeline>(capacity);}
            const auto expected=compact_many_run(*off,bodies,capacity),actual=compact_many_run(*on,bodies,capacity);
            compare_many_compact(actual,expected);
            const auto additional=bodies<=64?rigid_compact_bytes(bodies,capacity):0;
            require(actual.storage.packet_bytes==expected.storage.packet_bytes+additional,
                "rigid compact 64-body maximum/65-body fallback storage delta changed");
        }
        std::unique_ptr<GpuContactPipeline> off,on;
        {RigidCompactSetting setting(false);off=std::make_unique<GpuContactPipeline>(capacity);}
        {RigidCompactSetting setting(true);on=std::make_unique<GpuContactPipeline>(capacity);}
        // Retain the maximum allocation through unsupported, empty, disabled,
        // and smaller-body rebuilds using the same pair of PImpls.
        for(int step=0;step<5;++step) {
            const std::size_t bodies=step==1?65:step==4?4:64;
            const auto expected=compact_many_run(*off,bodies,capacity,step==2,step==3);
            const auto actual=compact_many_run(*on,bodies,capacity,step==2,step==3);compare_many_compact(actual,expected);
            require(actual.storage.packet_bytes==expected.storage.packet_bytes+rigid_compact_bytes(64,capacity),
                "rigid compact high-water scratch was not accounted across body-count/empty/disabled reuse");
        }
    }
}
}

void gpu_rigid_compact_reduce_test_gpu(const GpuPipelineTestInput* inputs,std::size_t count) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    ColorLocalSetting local_isolation(false);
    require(count>0,"rigid compact suite needs a valid mixed input");const auto& input=inputs[count-1];
    Memory<int> premise(4);compact_activity_premise<<<1,1>>>(input,premise.data);check(cudaGetLastError());
    int active[4];premise.download(active);for(int valid:active)require(valid==1,"rigid compact active/inactive premise failed");
    std::size_t comparisons=0;
    auto check_option=[&](MixedOptions option,bool cache) {
        RigidCacheSetting cache_setting(cache);GpuPipelineTestOutput expected,actual;ContactStorageBytes a,b;
        {RigidCompactSetting setting(false);run_mixed(input,expected,option,&a);}
        {RigidCompactSetting setting(true);run_mixed(input,actual,option,&b);}
        require(expected.error==0&&actual.error==0,"rigid compact mixed fixture failed");compare_mixed_bytes(actual,expected);
        if(option.missing_owner) {
            const auto seed=initial_system(8);
            require(std::memcmp(expected.rigid_com,&seed,sizeof(seed))!=0,
                "absent-owner cache fallback has no observable PT translation term");
        }
        const auto pattern=option.point_only?(option.mixed_activity?2u:1u):(option.interleaved?7u:2u);
        const auto candidates=option.empty_contacts?0:std::size_t(pattern)*option.repeat_contacts;
        const auto additional=candidates>option.capacity?rigid_compact_bytes(2,option.capacity):0;
        require(b.packet_bytes==a.packet_bytes+additional,"rigid compact mixed scratch accounting changed");
        require(b.exact_bytes==a.exact_bytes,"rigid compact changed numerical exact scratch");++comparisons;
    };
    for(bool cache:{false,true})for(bool original:{false,true})for(bool frozen:{false,true}) {
        for(std::size_t capacity:{std::size_t(1),std::size_t(3),std::size_t(5)}) {
            MixedOptions option;option.capacity=capacity;option.original=original;option.frozen=frozen;
            option.interleaved=true;option.repeat_contacts=3;option.d_hat=.2;option.seeded=true;option.pins=true;
            check_option(option,cache);
        }
        for(std::size_t capacity:{std::size_t(128),std::size_t(129),std::size_t(257)})for(bool mixed:{false,true}) {
            MixedOptions option;option.capacity=capacity;option.original=original;option.frozen=frozen;
            option.point_only=true;option.mixed_activity=mixed;option.repeat_contacts=2*capacity+1;
            option.d_hat=.2;option.seeded=true;check_option(option,cache);
        }
        for(std::size_t capacity:{std::size_t(128),std::size_t(129),std::size_t(257)}) {
            MixedOptions inactive;inactive.capacity=capacity;inactive.original=original;inactive.frozen=frozen;
            inactive.point_only=true;inactive.repeat_contacts=2*capacity+1;inactive.d_hat=.001;inactive.seeded=true;
            // Nonempty, enabled CSR chunks with no active payloads must never
            // consume uninitialized kept-index slots when every count is zero.
            check_option(inactive,cache);
        }
        MixedOptions missing;missing.capacity=3;missing.original=original;missing.frozen=frozen;
        missing.point_only=true;missing.repeat_contacts=5;missing.d_hat=.2;missing.seeded=true;
        missing.missing_owner=true;missing.no_sdf=true;check_option(missing,cache);
        // Untiled assembly must be the old route, including zero compact bytes.
        MixedOptions untiled;untiled.capacity=32;untiled.original=original;untiled.frozen=frozen;untiled.seeded=true;
        check_option(untiled,cache);

        std::unique_ptr<GpuContactPipeline> off,on;
        {RigidCompactSetting setting(false);RigidCacheSetting c(cache);off=std::make_unique<GpuContactPipeline>(3);}
        {RigidCompactSetting setting(true);RigidCacheSetting c(cache);on=std::make_unique<GpuContactPipeline>(3);}
        std::size_t high_water=0;
        for(int step=0;step<6;++step) {
            MixedOptions option;option.capacity=3;option.original=original;option.frozen=frozen;option.seeded=true;
            option.point_only=step==1||step==4;option.mixed_activity=step==1;
            option.repeat_contacts=step==1?130:step==4?5:1;option.d_hat=.2;
            option.empty_contacts=step==2;option.disable_translation=step==3;option.disable_rotation=step==3;
            option.missing_owner=step==4;option.no_sdf=step==4;
            GpuPipelineTestOutput expected,actual;ContactStorageBytes a,b;
            run_mixed(input,expected,option,&a,off.get());run_mixed(input,actual,option,&b,on.get());
            require(expected.error==0&&actual.error==0,"rigid compact lifecycle fixture failed");compare_mixed_bytes(actual,expected);
            if(step==1)high_water=rigid_compact_bytes(2,3);
            require(b.packet_bytes==a.packet_bytes+high_water,"rigid compact retained scratch lifecycle delta changed");
            ++comparisons;
        }
        for(bool compact:{false,true}) {
            RigidCompactSetting compact_setting(compact);RigidCacheSetting cache_setting(cache);
            for(auto phase:{ContactAssemblyPhase::RigidTranslation,ContactAssemblyPhase::RigidOrientation}) {
                late_tile_case(phase,original,frozen,0,1);
                for(int failure:{1,2}) {
                    late_tile_case(phase,original,frozen,failure,1);
                    late_tile_case(phase,original,frozen,failure,257,129);
                }
            }
        }
    }
    compact_many_body_boundaries();
    std::cout<<"GPU rigid compact reduction: "<<comparisons<<" OFF/ON bitwise mixed comparisons, active/inactive chunks/tails,"
        <<" cache/frozen/original modes, absent-owner fallback, late failures, exact retained scratch and 64/65-body gates PASS\n";
}

// Joint color-local and compact-storage regression coverage.
namespace {
std::unique_ptr<ipc_gpu::solver::GpuContactPipeline> joint_pipeline(
    std::size_t capacity,bool enabled,bool cache) {
    ColorLocalSetting color(enabled);RigidCompactSetting compact(enabled);RigidCacheSetting cached(cache);
    return std::make_unique<ipc_gpu::solver::GpuContactPipeline>(capacity);
}
void joint_seeded_failure(const GpuPipelineTestOutput& out,int expected) {
    require(out.error==expected,"joint late fault changed exact device status");
    for(int node=0;node<8;++node) {
        const auto seed=initial_system(node);
        same_bytes(out.deformable+node,&seed,1,"joint late fault published deformable systems");
    }
    for(int body=0;body<2;++body) {
        const auto seed=initial_system(body+8);
        same_bytes(out.rigid_com+body,&seed,1,"joint late fault published translation systems");
        same_bytes(out.rigid_omega+body,&seed,1,"joint late fault published rotation systems");
        same_bytes(out.rigid_omega_after+body,&seed,1,"joint late fault published refreshed rotation systems");
    }
}
void joint_storage(const ipc_gpu::solver::ContactStorageBytes& on,
    const ipc_gpu::solver::ContactStorageBytes& off,std::size_t map_capacity,
    std::size_t bodies,std::size_t packet_capacity,const char* why,
    std::size_t compact_record_slots=0) {
    require(on.packet_bytes==off.packet_bytes+map_capacity*sizeof(int)
        +rigid_compact_bytes(bodies,packet_capacity)
        +compact_record_slots*sizeof(ipc_gpu::solver::CompactDeformableContact),why);
}
}

void gpu_color_compact_joint_test_gpu(const GpuPipelineTestInput* inputs,std::size_t count) {
    using namespace ipc_gpu;using namespace ipc_gpu::solver;
    require(count>0,"joint suite needs a healthy mixed input");
    const auto& input=inputs[0];std::size_t comparisons=0;
    ContactOptionSetting queues("IPC_GPU_COLOR_PAIR_QUEUES","1"),
        parallel("IPC_GPU_RIGID_PARALLEL_REDUCE","0"),
        component("IPC_GPU_CONTACT_COMPONENT_REDUCE","0"),
        compact_warp("IPC_GPU_COMPACT_WARP_REDUCE","0");
    // Distinct allocation/alias routes, including active==output_activity and
    // compact transient tags, prefetch CSR reads, exact local queue IDs, and
    // deferred input publication. Parent isolated gates remain unchanged.
    for(int lane=0;lane<4;++lane) {
        ContactOptionSetting compact_def("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",lane==1||lane==2?"1":"0"),
            activity("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",lane?"1":"0"),
            selected("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",lane==1?"1":"0"),
            prefetch("IPC_GPU_CONTACT_PREFETCH_REDUCE",lane==2?"1":"0"),
            cooperative("IPC_GPU_COOPERATIVE_EXACT_CONTACT",lane==2?"1":"0"),
            deferred("IPC_GPU_DEFER_CONTACT_INPUTS",lane==2?"1":"0"),
            warp("IPC_GPU_WARP_CONTACT_REDUCE",lane==3?"1":"0");
        for(bool cache:{false,true})for(bool original:{false,true}) {
            constexpr std::size_t capacity=5;
            auto off=joint_pipeline(capacity,false,cache),on=joint_pipeline(capacity,true,cache);
            for(int epoch=0;epoch<11;++epoch) {
                MixedOptions option;option.capacity=capacity;option.local_colors=true;
                option.interleaved=true;option.seeded=true;option.pins=true;option.original=original;
                option.repeat_contacts=epoch==1||epoch==3?3:1;
                option.empty_contacts=epoch==2;option.disable_translation=epoch==3;option.disable_rotation=epoch==3;
                option.reverse_contacts=epoch==4;option.swap_owners=epoch==4;
                option.frozen=epoch==5;
                if(epoch==7) {
                    option.interleaved=false;option.point_only=true;option.repeat_contacts=5;
                    option.missing_owner=true;option.no_sdf=true;option.d_hat=.2;
                }
                auto sample=input;
                if(epoch==9) {
                    sample.positions[0][2]=0;
                    // Fail in the FIRST deformable color, before a different
                    // healthy color is allowed to publish its independent row.
                    option.local_colors=false;
                }
                GpuPipelineTestOutput expected,actual;ContactStorageBytes a,b;
                std::vector<std::size_t> colors;
                run_mixed(sample,expected,option,&a,off.get());
                run_mixed(sample,actual,option,&b,on.get(),&colors);
                compare_mixed_bytes(actual,expected);
                if(epoch==0||epoch==4||epoch==6||epoch==8||epoch==10)
                    require(colors==std::vector<std::size_t>({4,7,5}),"joint local/tiled/local cardinalities changed");
                if(epoch==9)require(colors==std::vector<std::size_t>({6,7,3}),
                    "joint first-color error premise changed");
                if(epoch==1||epoch==3)
                    require(colors==std::vector<std::size_t>({12,21,15}),"joint all-tiled cardinalities changed");
                if(epoch==2)require(colors==std::vector<std::size_t>({0,0,0}),"joint empty epoch retained contacts");
                // Map7 remains allocated while fallback must use nullptr and
                // explicit pair-first slots. Compact128+count remains through
                // empty/disabled/reordered/untiled and failed epochs.
                joint_storage(b,a,7,2,capacity,"joint retained map+kept/count storage delta changed");
                if(epoch==9)joint_seeded_failure(actual,21);
                else require(actual.error==0,"joint valid lifecycle returned a device error");
                ++comparisons;
            }
        }
    }
    // The same PImpl first owns mapped records, then64-body tiled chunks,
    // returns to mapped records, exceeds the body bound, overwrites every
    // inactive count, visits empty state, then reactivates both mechanisms.
    for(bool cache:{false,true}) {
        ContactOptionSetting compact_def("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS","1"),
            activity("IPC_GPU_CONTACT_OUTPUT_ACTIVITY","1"),
            prefetch("IPC_GPU_CONTACT_PREFETCH_REDUCE","1"),
            cooperative("IPC_GPU_COOPERATIVE_EXACT_CONTACT","0"),
            warp("IPC_GPU_WARP_CONTACT_REDUCE","0");
        constexpr std::size_t capacity=129,map_size=141;
        auto off=joint_pipeline(capacity,false,cache),on=joint_pipeline(capacity,true,cache);
        LocalPacketOptions local;local.capacity=capacity;local.decoys=130;
        auto expected=run_local_packet_case(local,false,off.get());
        auto actual=run_local_packet_case(local,true,on.get());compare_local_packet_runs(actual,expected);
        require(actual.global_count==map_size&&actual.counts==std::vector<std::size_t>({3,6,2,0}),
            "joint64 map-prime premise changed");
        require(actual.storage.packet_bytes==expected.storage.packet_bytes+map_size*sizeof(int)
            +capacity*sizeof(CompactDeformableContact),
            "joint map-only prime allocated compact scratch");++comparisons;
        // Independent public-API premise: the SAME mapped owner releases
        // compact records on a GS->Frozen switch, before any rigid allocation.
        const auto before_frozen_off=expected.storage,before_frozen_on=actual.storage;
        LocalPacketOptions frozen_prime=local;frozen_prime.frozen=true;
        const auto frozen_off=run_local_packet_case(frozen_prime,false,off.get());
        const auto frozen_on=run_local_packet_case(frozen_prime,true,on.get());
        compare_local_packet_runs(frozen_on,frozen_off);
        require(frozen_on.storage.packet_bytes+capacity*sizeof(CompactDeformableContact)
            ==before_frozen_on.packet_bytes,"Frozen did not release exactly the compact packet records");
        require(frozen_off.storage.packet_bytes==before_frozen_off.packet_bytes,
            "Frozen changed unrelated fallback packet storage");
        require(frozen_on.storage.packet_bytes==frozen_off.storage.packet_bytes+map_size*sizeof(int),
            "Frozen map-only premise retained an unexpected packet allocation");++comparisons;
        for(int epoch=0;epoch<7;++epoch) {
            const std::size_t bodies=epoch==2?65:64;
            const auto a=compact_many_run(*off,bodies,capacity,epoch==3,epoch==4);
            const auto b=compact_many_run(*on,bodies,capacity,epoch==3,epoch==4);
            compare_many_compact(b,a);
            joint_storage(b.storage,a.storage,map_size,64,capacity,
                "joint64 map/kept/count high-water accounting changed");++comparisons;
            if(epoch==0||epoch==1||epoch==5) {
                local.reverse=epoch&1;local.frozen=epoch==1;local.pruned=epoch==1;
                local.original=epoch==5;
                expected=run_local_packet_case(local,false,off.get());
                actual=run_local_packet_case(local,true,on.get());compare_local_packet_runs(actual,expected);
                // Frozen explicitly releases compact records; only map and
                // rigid kept/count buffers retain their high-water capacities.
                joint_storage(actual.storage,expected.storage,map_size,64,capacity,
                    "joint64 local return consumed or lost tiled scratch",local.frozen?0:capacity);++comparisons;
                if(local.pruned)for(int color=7;color<=10;++color)
                    require(actual.counts[color]==0,"joint64 pruned decoys remained in mapped CSR rows");
            }
        }
        // Retained map and64-body kept/count buffers must not affect a later
        // two-body, multi-tile failure. Every parent exact status and zero-
        // publication assertion remains active in this reused-owner path.
        for(bool original:{false,true})for(bool frozen:{false,true})
            for(auto phase:{ContactAssemblyPhase::RigidTranslation,ContactAssemblyPhase::RigidOrientation})
                for(int fault:{1,2}) {
                    late_tile_case(phase,original,frozen,fault,capacity,257,off.get());
                    late_tile_case(phase,original,frozen,fault,capacity,257,on.get());
                    ++comparisons;
                }
        // Deliberately corrupt one current-color queue entry. Existing
        // fail-closed map checks must win even with retained compact buffers.
        LocalPacketOptions missing;missing.capacity=capacity;missing.decoys=130;missing.missing_map=true;
        const auto failure=run_local_packet_case(missing,true,on.get());
        require(!failure.statuses.empty()&&failure.statuses[0]==20,
            "joint missing map did not fail closed with exact InvalidInput");++comparisons;
        // The subsequent valid active tile must overwrite every count; stale
        // missing-map/error/private packet state must not poison the rebuild.
        const auto a=compact_many_run(*off,64,capacity),b=compact_many_run(*on,64,capacity);
        compare_many_compact(b,a);joint_storage(b.storage,a.storage,map_size,64,capacity,
            "joint64 post-error scratch storage changed");++comparisons;
    }
    std::cout<<"GPU joint color-local/rigid-compact: "<<comparisons
        <<" checks (exact bothOFF/bothON systems/status); local/tiled/local, cache/aliases,"
        <<" null-map fallback, zero kept counts,64/65 bodies, missing maps and error publication PASS\n";
}

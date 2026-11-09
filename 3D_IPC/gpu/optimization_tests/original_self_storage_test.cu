#include "qualified_evaluator_oracle.h"
#include "contact_pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ipc_gpu;
using namespace ipc_gpu::solver;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
template<class T>struct Memory {
    T* data=nullptr;
    explicit Memory(std::size_t n,const T* source=nullptr) {
        check(cudaMalloc(reinterpret_cast<void**>(&data),std::max<std::size_t>(n,1)*sizeof(T)));
        if(n&&source)check(cudaMemcpy(data,source,n*sizeof(T),cudaMemcpyHostToDevice));
        else check(cudaMemset(data,0,std::max<std::size_t>(n,1)*sizeof(T)));
    }
    ~Memory(){cudaFree(data);}
    Memory(const Memory&)=delete;
    Memory& operator=(const Memory&)=delete;
};
class Setting {
    const char* name;bool present;std::string previous;
public:
    Setting(const char* key,const char* value):name(key),present(std::getenv(key)!=nullptr) {
        if(present)previous=std::getenv(key);
        if(value?setenv(key,value,1):unsetenv(key))throw std::runtime_error("cannot set test flag");
    }
    ~Setting(){if(present)setenv(name,previous.c_str(),1);else unsetenv(name);}
};
template<class T>__device__ bool equal_bytes(const T& a,const T& b) {
    const auto* x=reinterpret_cast<const unsigned char*>(&a);
    const auto* y=reinterpret_cast<const unsigned char*>(&b);
    for(std::size_t k=0;k<sizeof(T);++k)if(x[k]!=y[k])return false;
    return true;
}
struct PairCheck {unsigned failure=0;};
__global__ void differential(const ContactInput* inputs,std::size_t n,PairCheck* results) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=n)return;
    const auto& in=inputs[index];
    const auto old=evaluate_contact_original_qualified_oracle(in);
    const auto full=evaluate_contact_original(in);
    unsigned failure=equal_bytes(old,full)?0u:1u;
    for(int role=-1;role<=4;++role) {
        const bool eligible=role>=0&&role<4&&in.hessian_blocks==(1u<<(5*role))&&in.friction==0.0;
        const auto thin=evaluate_contact_original_self_zero(in,role);
        if(!eligible) {
            if(thin.status!=ContactStatus::InvalidInput||thin.active)failure|=2u;
            continue;
        }
        if(thin.status!=old.status||thin.active!=old.active)failure|=4u;
        if(!equal_bytes(thin.distance,old.distance)||!equal_bytes(thin.weights,old.weights)
            ||!equal_bytes(thin.normal,old.normal)||!equal_bytes(thin.barrier_gradient,old.barrier_gradient)
            ||!equal_bytes(thin.selected_hessian,old.barrier_hessian[role][role]))failure|=8u;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(i!=role||j!=role)
            for(int k=0;k<9;++k)if(__double_as_longlong(old.barrier_hessian[i][j][k])!=0)failure|=16u;
        for(int k=0;k<3;++k)if(__double_as_longlong(old.friction_relative_gradient[k])!=0)failure|=16u;
        for(int k=0;k<9;++k)if(__double_as_longlong(old.friction_relative_hessian[k])!=0)failure|=16u;
    }
    results[index].failure=failure;
}
struct PipelineCase {
    std::size_t capacity=257,repeats=1;
    bool frozen=false,original=true,empty=false,collapse=false,repeat_vertex=false;
    bool inactive=false,zero_gap=false,invalid_previous=false,retry_ee=false;
    int initial_error=0;
    double friction=0.0,dt=.031,eps_v=.1;
};
struct PipelineResult {
    std::vector<Derivatives> snapshots;
    std::vector<int> errors;
    ContactStorageBytes storage;
};
PipelineResult pipeline_case(const PipelineCase& option,GpuContactPipeline& pipeline) {
    constexpr int nodes=8;
    double x[nodes][3]={{.2,.2,.04},{0,0,0},{1,0,0},{0,1,0},
                        {-.5,0,.06},{.5,0,.06},{0,-.5,0},{0,.5,0}};
    if(option.inactive)for(int n:{0,4,5})x[n][2]=4.;
    if(option.zero_gap)x[0][2]=0.;
    if(option.retry_ee) {
        x[0][2]=4.;
        // Existing qualified DAT rounded-zero fixture from contact_pipeline_test.cpp.
        const double captured[4][3]={{-1.2508422453585308,.30142330872441742,-.3441007037557508},
            {-1.249271199389888,.43384953344448823,-.36383141911477945},
            {-1.252873745189554,.35850525592266047,-.35936719358941788},
            {-1.2435860863993213,.36377791656553377,-.33712447408743129}};
        std::memcpy(x+4,captured,sizeof(captured));
    }
    double previous[nodes][3];std::memcpy(previous,x,sizeof(x));
    if(option.invalid_previous)previous[3][0]=std::numeric_limits<double>::quiet_NaN();
    int blocks[nodes],colors[nodes];unsigned char deformable[nodes],solid[nodes]{};
    for(int n=0;n<nodes;++n){blocks[n]=n;colors[n]=option.collapse?0:n;deformable[n]=1;}
    std::vector<int> color_blocks,color_offsets;
    const int count=option.collapse?1:nodes;
    for(int color=0;color<count;++color){color_offsets.push_back(int(color_blocks.size()));
        for(int n=0;n<nodes;++n)if(colors[n]==color)color_blocks.push_back(n);}
    color_offsets.push_back(int(color_blocks.size()));
    std::vector<BpContact> contacts;
    for(std::size_t i=0;i<option.repeats;++i) {
        contacts.push_back({{0,1,2,3},0});contacts.push_back({{4,5,6,7},1});
        if(option.repeat_vertex)contacts.push_back({{0,1,1,3},0});
    }
    if(option.empty)contacts.clear();
    Derivatives seeds[nodes]{};
    for(int n=0;n<nodes;++n){seeds[n].gradient[0]=.03125*(n+1);
        for(int k=0;k<9;++k)seeds[n].hessian[k]=.015625*(k+n+1);}
    Memory<double> positions(3*nodes,&x[0][0]),old(3*nodes,&previous[0][0]);
    Memory<int> d_blocks(nodes,blocks),d_colors(nodes,colors),d_color_blocks(color_blocks.size(),color_blocks.data()),
        d_color_offsets(color_offsets.size(),color_offsets.data()),failed(1,&option.initial_error);
    Memory<unsigned char> d_deformable(nodes,deformable),d_solid(nodes,solid);
    Memory<BpContact> pairs(contacts.size(),contacts.empty()?nullptr:contacts.data());
    Memory<Derivatives> systems(nodes,seeds);
    DeformableScene scene{};scene.nodes=nodes;scene.node_blocks=d_blocks.data;
    scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(positions.data);
    state.previous=reinterpret_cast<double(*)[3]>(old.data);
    ElasticWorkspace work{};work.systems=systems.data;work.error=failed.data;
    BroadPhaseView broad{};broad.contacts=pairs.data;broad.contact_count=contacts.size();
    broad.block_colors=d_colors.data;broad.color_count=count;
    broad.color_blocks=d_color_blocks.data;broad.color_offsets=d_color_offsets.data;
    pipeline.rebuild(broad,nodes,d_blocks.data,nullptr);
    const ContactParameters parameters{.8,37.,option.dt,option.friction,option.eps_v,1e-12};
    PipelineResult result;std::vector<Derivatives> before(seeds,seeds+nodes),after(nodes);
    for(int color=0;color<count;++color) {
        pipeline.assemble_color(scene,state,work,{},{},{},{},parameters,color,ContactAssemblyPhase::Deformable,
            option.frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},
            option.original?ContactGeometryPolicy::Original:ContactGeometryPolicy::Robust);
        int status=0;check(cudaMemcpy(&status,failed.data,sizeof(status),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(after.data(),systems.data,nodes*sizeof(Derivatives),cudaMemcpyDeviceToHost));
        result.errors.push_back(status);result.snapshots.insert(result.snapshots.end(),after.begin(),after.end());
        if(status)require(std::memcmp(before.data(),after.data(),nodes*sizeof(Derivatives))==0,
                          "failed self-storage gather published derivatives");
        else before=after;
    }
    double unchanged[3*nodes];check(cudaMemcpy(unchanged,positions.data,sizeof(x),cudaMemcpyDeviceToHost));
    require(std::memcmp(unchanged,x,sizeof(x))==0,"self storage modified current positions");
    check(cudaMemcpy(unchanged,old.data,sizeof(previous),cudaMemcpyDeviceToHost));
    require(std::memcmp(unchanged,previous,sizeof(previous))==0,"self storage modified previous positions");
    result.storage=pipeline.storage_bytes();
    if(option.retry_ee)require(std::all_of(result.errors.begin(),result.errors.end(),[](int code){return code==0;}),
                              "frozen DAT rounded-zero EE emergency retry failed");
    if(!option.empty&&!option.inactive&&!option.zero_gap&&!option.invalid_previous&&!option.initial_error
        &&std::all_of(result.errors.begin(),result.errors.end(),[](int code){return code==0;})
        &&std::isfinite(option.dt*option.eps_v)&&option.dt*option.eps_v>0&&std::isfinite(option.dt*option.dt))
        require(std::memcmp(seeds,before.data(),sizeof(seeds))!=0,"active-force premise is vacuous");
    return result;
}
void compare_pipeline(const PipelineResult& a,const PipelineResult& b) {
    require(a.errors==b.errors,"self storage changed single-pair error statuses");
    require(a.snapshots.size()==b.snapshots.size()&&std::memcmp(a.snapshots.data(),b.snapshots.data(),
        a.snapshots.size()*sizeof(Derivatives))==0,"self storage changed assembled system bytes");
    require(a.storage.packet_bytes==b.storage.packet_bytes&&a.storage.exact_bytes==b.storage.exact_bytes,
            "self storage changed persistent allocation");
}
void pipeline_gates() {
    std::size_t comparisons=0;
    for(bool compact:{false,true})for(bool activity:{false,true})for(bool deferred:{false,true}) {
        Setting compact_flag("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0");
        Setting activity_flag("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",activity?"1":"0");
        Setting defer_flag("IPC_GPU_DEFER_CONTACT_INPUTS",deferred?"1":"0");
        for(std::size_t capacity:{1u,3u,5u,128u,129u,257u}) {
            std::unique_ptr<GpuContactPipeline> off,on;
            {Setting flag("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER","0");off=std::make_unique<GpuContactPipeline>(capacity);}
            {Setting flag("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER","1");on=std::make_unique<GpuContactPipeline>(capacity);}
            for(int kind=0;kind<16;++kind) {
                PipelineCase option;option.capacity=capacity;option.repeats=kind==0?129:kind==1?257:1;
                option.original=kind!=10;option.frozen=kind==2||kind==13;option.friction=kind==3?.1:kind==4?-0.:0.;
                option.collapse=kind==5;option.repeat_vertex=kind==6;option.empty=kind==7;
                option.inactive=kind==8;option.initial_error=kind==9?73:0;option.invalid_previous=kind==11;
                option.zero_gap=kind==12;option.retry_ee=kind==13;
                if(kind==14){option.dt=std::numeric_limits<double>::denorm_min();option.eps_v=.001;}
                if(kind==15)option.dt=std::numeric_limits<double>::max();
                Setting captured("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER","invalid");
                const auto expected=pipeline_case(option,*off);const auto actual=pipeline_case(option,*on);
                compare_pipeline(actual,expected);++comparisons;
            }
        }
    }
    // Fresh constructor must reject malformed flags; previously captured owners
    // above ignore later environment changes. This is the ordinary option contract.
    {Setting invalid("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER","invalid");bool rejected=false;
        try{GpuContactPipeline pipeline;}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"malformed self-storage option was accepted");}
    std::cout<<"Original self storage public pipeline PASS: comparisons="<<comparisons
             <<"; compact/activity/deferred, tails, fallback, active force, errors, reuse\n";
}
} // namespace

void gpu_original_self_storage_check(const ipc_gpu::solver::ContactInput* inputs,std::size_t count) {
    constexpr std::size_t chunk=257;
    Memory<ContactInput> device(chunk);Memory<PairCheck> results(chunk);
    for(std::size_t first=0;first<count;first+=chunk) {
        const auto n=std::min(chunk,count-first);
        check(cudaMemcpy(device.data,inputs+first,n*sizeof(ContactInput),cudaMemcpyHostToDevice));
        differential<<<static_cast<unsigned>((n+127)/128),128>>>(device.data,n,results.data);check(cudaGetLastError());
        std::vector<PairCheck> host(n);check(cudaMemcpy(host.data(),results.data,n*sizeof(PairCheck),cudaMemcpyDeviceToHost));
        for(std::size_t k=0;k<n;++k)if(host[k].failure)
            throw std::runtime_error("Original self storage device fields differ at"+std::to_string(first+k)
                +" failure bits"+std::to_string(host[k].failure));
    }
    pipeline_gates();
    std::cout<<"Original self storage device evaluator PASS: inputs="<<count<<"\n";
}

#include "contact_pipeline.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ipc_gpu;
using namespace ipc_gpu::solver;
// TEST-ONLY friend authority. This binary must link ContactPipeline and real
// CUDA, but NO production Solver object/symbols (mandatory link-map audit).
namespace ipc_gpu::solver {
class GpuSolver {
public: static void bind(GpuContactPipeline& pipeline,std::size_t n) {
    pipeline.bind_validated_two_rigid_blocks(n);
}
};
}
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer {
    T* data=nullptr;std::size_t n;
    Buffer(std::size_t count,const T* host=nullptr):n(count) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data),(n?n:1)*sizeof(T)));
        if(host&&n)checked(cudaMemcpy(data,host,n*sizeof(T),cudaMemcpyHostToDevice));
        else checked(cudaMemset(data,0,(n?n:1)*sizeof(T)));
    }
    ~Buffer(){cudaFree(data);}
    void load(const T* p){checked(cudaMemcpy(data,p,n*sizeof(T),cudaMemcpyHostToDevice));}
    void get(T* p){checked(cudaMemcpy(p,data,n*sizeof(T),cudaMemcpyDeviceToHost));}
};
class Setting {
    const char* name;bool had;std::string prior;
public:
    Setting(const char* key,const char* value):name(key),had(std::getenv(key)!=nullptr) {
        if(had)prior=std::getenv(key);
        if(value?setenv(key,value,1):unsetenv(key))throw std::runtime_error("cannot set fixture option");
    }
    ~Setting(){if(had)setenv(name,prior.c_str(),1);else unsetenv(name);}
};
struct Result {std::vector<Derivatives> values;std::vector<int> codes;std::vector<ContactStorageBytes> storage;};
Result run(bool enabled,bool trusted,int graph_case,bool frozen,int fault,bool orientation,
    std::size_t maximum=4,std::size_t capacity=2,bool alternating=false,bool large_seed=false,
    const Derivatives* seed_override=nullptr,const ContactParameters* parameters_override=nullptr,
    int kinds=0,bool far_geometry=false,bool cache_enabled=true) {
    Setting feature("IPC_GPU_RIGID_ONE_SLOT_TERMS",enabled?"1":"0");
    Setting dense("IPC_GPU_TWO_RIGID_DENSE_CSR","1"),cache("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE",cache_enabled?"1":"0");
    Setting compact_reduce("IPC_GPU_RIGID_COMPACT_REDUCE","1");
    Setting pruning("IPC_GPU_BOX_CONTACT_PRUNING","0"),local("IPC_GPU_COLOR_LOCAL_PACKETS","0");
    double x[4][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0}};
    if(far_geometry)x[0][2]=3.;
    if(fault==1)x[0][2]=0.;
    const int mapping[4]={0,1,1,1},body_blocks[2]={0,1},nodes[4]={0,1,2,3};
    const std::size_t offsets[3]={0,1,4};const unsigned char no[4]={},yes[2]={1,1};
    int colors[2]={0,1},color_offsets[3]={0,1,2},color_blocks[2]={0,1};
    if(graph_case==1)colors[1]=0;
    if(graph_case==2)colors[1]=2;
    if(graph_case==3)color_offsets[1]=0;
    if(graph_case==4)color_blocks[1]=0;
    if(graph_case==5){colors[0]=1;colors[1]=0;color_blocks[0]=1;color_blocks[1]=0;}
    const double q[2][4]={{1,0,0,0},{1,0,0,0}};
    std::vector<BpContact> pair(maximum,BpContact{{0,1,2,3},0});
    if(alternating)for(std::size_t i=1;i<maximum;i+=2)pair[i]=BpContact{{1,0,2,3},0};
    for(std::size_t i=0;i<maximum;++i)pair[i].kind=kinds==1?1:kinds==2&&i>=maximum/2?1:0;
    Buffer<double> positions(12,&x[0][0]),previous(12,&x[0][0]),refs(12,&x[0][0]),quat(8,&q[0][0]),omega(6),com(6);
    Buffer<int> block(4,mapping),owners(4,mapping),bblocks(2,body_blocks),d_nodes(4,nodes),dcolors(2,colors),
        dcolor_offsets(3,color_offsets),dcolor_blocks(2,color_blocks),failed(1);
    Buffer<std::size_t> d_offsets(3,offsets);Buffer<unsigned char> deformable(4,no),solid(4,no),enabled_bodies(2,yes);
    Buffer<BpContact> contacts(maximum,pair.data());Buffer<Derivatives> node_systems(4),body_systems(2);
    DeformableScene scene{};scene.nodes=4;scene.node_blocks=block.data;scene.deformable=deformable.data;scene.solid=solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(positions.data);state.previous=reinterpret_cast<double(*)[3]>(previous.data);
    ElasticWorkspace elastic{};elastic.systems=node_systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=2;rigid.body_blocks=bblocks.data;rigid.node_body=owners.data;
    rigid.reference=reinterpret_cast<const double(*)[3]>(refs.data);rigid.nodes=d_nodes.data;rigid.node_offsets=d_offsets.data;
    rigid.translate=enabled_bodies.data;rigid.rotate=enabled_bodies.data;
    RigidState rs{};rs.previous_orientation=reinterpret_cast<double(*)[4]>(quat.data);rs.orientation=rs.previous_orientation;
    rs.omega=reinterpret_cast<double(*)[3]>(omega.data);rs.com=reinterpret_cast<double(*)[3]>(com.data);
    RigidWorkspace rw{};rw.systems=body_systems.data;
    BroadPhaseView broad{};broad.contacts=contacts.data;broad.block_colors=dcolors.data;broad.color_count=2;
    broad.color_offsets=dcolor_offsets.data;broad.color_blocks=dcolor_blocks.data;
    ContactParameters params{.2,37,.031,0,.1,1e-12};
    if(parameters_override)params=*parameters_override;
    GpuContactPipeline pipeline(capacity);
    if(trusted)GpuSolver::bind(pipeline,4);
    Result result;
    for(std::size_t count:{maximum,std::size_t(1),std::size_t(0),maximum}) {
        broad.contact_count=count;pipeline.rebuild(broad,2,block.data);
        Derivatives seed[2]{};
        for(int b=0;b<2;++b)for(int k=0;k<3;++k){seed[b].gradient[k]=(large_seed?1e8:.125)*(b+k+1);seed[b].hessian[3*k+k]=b+k+1.;}
        if(seed_override)std::memcpy(seed,seed_override,sizeof(seed));
        body_systems.load(seed);int status=fault==2?123:0;failed.load(&status);
        for(int color=0;color<2;++color)pipeline.assemble_color(scene,state,elastic,rigid,rs,rw,{},params,color,
            orientation?ContactAssemblyPhase::RigidOrientation:ContactAssemblyPhase::RigidTranslation,
            frozen?ContactAssemblyMode::FrozenSnapshot:ContactAssemblyMode::ColorSweep,{},ContactGeometryPolicy::Original);
        Derivatives values[2];body_systems.get(values);failed.get(&status);
        result.values.insert(result.values.end(),values,values+2);result.codes.push_back(status);
        result.storage.push_back(pipeline.storage_bytes());
        if(fault==2)need(status==123,"upstream error changed");
        if(fault==1)need(status==(count?21:0),"zero-gap/nonempty or empty positive control status changed");
        if(fault&&status)need(std::memcmp(values,seed,sizeof(seed))==0,"error published partial body derivatives");
    }
    double unchanged[12];positions.get(unchanged);need(std::memcmp(unchanged,x,sizeof(x))==0,"force assembly changed positions");
    return result;
}
}
int main(){try{
    std::size_t comparisons=0;
    for(bool trusted:{false,true})for(int graph=0;graph<6;++graph)for(bool frozen:{false,true})
        for(int fault=0;fault<3;++fault)for(bool orientation:{false,true}) {
            const auto a=run(false,trusted,graph,frozen,fault,orientation),b=run(true,trusted,graph,frozen,fault,orientation);
            need(a.codes==b.codes,"private graph fallback changed status");
            need(a.values.size()==b.values.size()&&std::memcmp(a.values.data(),b.values.data(),a.values.size()*sizeof(Derivatives))==0,
                "private graph fallback changed ordered FP bytes");
            // First/last N4 epochs are tiled atcap2. A trusted normal proper
            // map must use1slot (saving6records*192 minus8B proof); every
            // malformed/frozen map must keep4slots (proof8B only).
            for(std::size_t epoch:{std::size_t(0),std::size_t(3)}) {
                const bool active_layout=trusted&&!frozen&&(graph==0||graph==5);
                if(active_layout)need(a.storage[epoch].packet_bytes==b.storage[epoch].packet_bytes+1144,
                    "proper private graph did not activate one-slot storage");
                else need(b.storage[epoch].packet_bytes==a.storage[epoch].packet_bytes+(trusted?8:0),
                    "untrusted/malformed/frozen graph failed four-slot fallback storage witness");
                need(a.storage[epoch].exact_bytes==b.storage[epoch].exact_bytes,"one-slot graph changed exact arenas");
            }
            ++comparisons;
        }
    std::cout<<"One-slot private graph facade PASS: comparisons="<<comparisons
        <<"; NO production Solver allowed in link map; no activation/speed claim\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

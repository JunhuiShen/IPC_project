// Link ipc_resident_solver and its normal CUDA dependencies. This compares
// post-color pruning ON/OFF, not CPU/GPU trajectories. Original contact IDs,
// incidence order, coloring, numerical outputs and accepted endpoints matter.
#include "resident_contact_pipeline.h"
#include "resident_linear_pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::resident;
namespace {
using V=std::array<double,3>;
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void checked(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
template<class T>struct Memory {
    T* data=nullptr;std::size_t size;
    explicit Memory(const std::vector<T>& value):size(value.size()) {
        if(size){checked(cudaMalloc(reinterpret_cast<void**>(&data),size*sizeof(T)));write(value);}
    }
    ~Memory(){cudaFree(data);}
    void write(const std::vector<T>& value){require(value.size()==size,"buffer length changed");if(size)checked(cudaMemcpy(data,value.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> v(size);if(size)checked(cudaMemcpy(v.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Environment {
    struct Entry{std::string key,value;bool present;};std::vector<Entry> entries;
    void set(const char* key,const char* value){const auto old=std::getenv(key);entries.push_back({key,old?old:"",old!=nullptr});if(value)setenv(key,value,1);else unsetenv(key);}
    ~Environment(){for(auto i=entries.rbegin();i!=entries.rend();++i)if(i->present)setenv(i->key.c_str(),i->value.c_str(),1);else unsetenv(i->key.c_str());}
};
double(*xyz(V* p))[3]{return reinterpret_cast<double(*)[3]>(p);}
template<class T>void require_bytes_equal(const std::vector<T>& a,const std::vector<T>& b,const std::string& message){require(a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0),message);}
template<class T>std::vector<T> copy(const T* p,std::size_t count){std::vector<T> out(count);if(count)checked(cudaMemcpy(out.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return out;}
std::vector<BpAabb> boxes(const std::vector<V>& positions,double radius){
    std::vector<BpAabb> result(positions.size());
    for(std::size_t i=0;i<positions.size();++i)for(int k=0;k<3;++k){result[i].lo[k]=positions[i][k]-radius;result[i].hi[k]=positions[i][k]+radius;}
    return result;
}

void check_incidence(const ContactIncidence& incidence,const std::vector<BpContact>& contacts,
    const std::vector<int>& colors,const std::vector<unsigned char>& expected,bool queues,bool mask_expected) {
    const auto offsets=copy(incidence.block_offsets,colors.size()+1);
    require(offsets.front()==0,"incidence must start at zero");
    const auto ids=copy(incidence.pair_indices,offsets.back());
    for(std::size_t node=0;node<colors.size();++node) {
        std::vector<int> want;
        for(std::size_t i=0;i<contacts.size();++i)if(!expected[i])
            for(int role=0;role<4;++role)if(contacts[i].vertices[role]==int(node)){want.push_back(int(i));break;}
        std::stable_sort(want.begin(),want.end(),[&](int a,int b){return contacts[a].kind!=contacts[b].kind?contacts[a].kind<contacts[b].kind:a<b;});
        require(offsets[node]<=offsets[node+1]&&offsets[node+1]<=ids.size(),"bad incident offsets");
        require_bytes_equal(want,std::vector<int>(ids.begin()+offsets[node],ids.begin()+offsets[node+1]),"retained incident ID/order changed");
    }
    const auto& work=incidence.color_pairs;
    require(bool(work.box_separated)==mask_expected,"wrong certificate lifecycle/null mask");
    if(mask_expected)require_bytes_equal(expected,copy(work.box_separated,contacts.size()),"incorrect prune flags");
    if(!queues){require(!work.offsets&&!work.pair_indices&&!work.max_color_pairs,"queues-off exposed index arrays");return;}
    const auto color_offsets=copy(work.offsets,5);
    require(color_offsets.front()==0,"color offsets must start at zero");
    const auto pairs=copy(work.pair_indices,color_offsets.back());std::size_t maximum=0;
    for(int color=0;color<4;++color) {
        std::vector<int> want;
        for(std::size_t i=0;i<contacts.size();++i)if(!expected[i]) {
            bool selected=false;for(int role=0;role<4;++role)selected=selected||colors[contacts[i].vertices[role]]==color;
            if(selected)want.push_back(int(i));
        }
        require_bytes_equal(want,std::vector<int>(pairs.begin()+color_offsets[color],pairs.begin()+color_offsets[color+1]),"retained color worklist changed order/cardinality");
        maximum=std::max(maximum,want.size());
    }
    require(maximum==work.max_color_pairs,"retained maximum color count incorrect");
}

struct Results{std::vector<Derivatives> systems;std::vector<V> accepted;std::vector<int> errors;};
Results run(bool prune,bool queues,bool compact,bool cooperative) {
    Environment env;env.set("IPC_GPU_BOX_CONTACT_PRUNING",prune?"1":"0");
    env.set("IPC_GPU_COLOR_PAIR_QUEUES",queues?"1":"0");env.set("IPC_GPU_WARP_CONTACT_REDUCE","0");
    env.set("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0");
    env.set("IPC_GPU_COOPERATIVE_EXACT_CONTACT",cooperative?"1":"0");
    env.set("IPC_GPU_PROFILE_CSV",nullptr);env.set("IPC_GPU_CONTACT_QUEUE_CSV",nullptr);
    std::vector<V> positions{{.25,.25,.05},{0,0,0},{1,0,0},{0,1,0},
                            {2.25,.25,.5},{2,0,0},{3,0,0},{2,1,0},
                            {4,0,0},{5,0,0},{4,.5,0},{5,.5,0},
                            {6,0,.05},{7,0,.05},{6.5,-.5,0},{6.5,.5,0}};
    const auto original=positions;auto previous=positions;
    for(auto& p:previous)p[0]-=.001;
    auto targets=positions;
    std::vector<int> node_blocks(16),colors(16),color_blocks,color_offsets;
    std::iota(node_blocks.begin(),node_blocks.end(),0);
    for(int i=0;i<16;++i)colors[i]=i%4;
    for(int color=0;color<4;++color){color_offsets.push_back(int(color_blocks.size()));for(int i=0;i<16;++i)if(colors[i]==color)color_blocks.push_back(i);}
    color_offsets.push_back(int(color_blocks.size()));
    std::vector<unsigned char> deformable(16,1),solid(16,0);
    const BpContact choices[]={{{0,1,2,3},0},{{4,5,6,7},0},{{8,9,10,11},1},{{12,13,14,15},1}};
    std::vector<BpContact> contacts;for(int i=0;i<33;++i)contacts.push_back(choices[i%4]);
    Memory<V> x(positions),old(previous),proposed(targets),accepted(std::vector<V>(16));
    Memory<int> d_blocks(node_blocks),d_colors(colors),d_color_blocks(color_blocks),d_color_offsets(color_offsets),error(std::vector<int>{0});
    Memory<unsigned char> d_deformable(deformable),d_solid(solid);
    Memory<BpContact> d_contacts(contacts);Memory<BpAabb> bounds(boxes(positions,.005));
    std::vector<Derivatives> initial(16);
    for(int n=0;n<16;++n){for(int k=0;k<3;++k)initial[n].gradient[k]=(n+1)*.125/(k+1);for(int k=0;k<9;++k)initial[n].hessian[k]=(n+1)*.5/(k+1);}
    Memory<Derivatives> systems(initial);
    DeformableScene scene{};scene.nodes=16;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=xyz(x.data);state.previous=xyz(old.data);
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.proposed=xyz(proposed.data);elastic.error=error.data;
    RigidScene rigid{};RigidState rigid_state{};RigidWorkspace rigid_work{};SdfScene sdf{};
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();broad.block_colors=d_colors.data;
    broad.color_count=4;broad.color_blocks=d_color_blocks.data;broad.color_offsets=d_color_offsets.data;
    ContactParameters params{.1,100,.02,.1,.01,1e-12};
    ResidentContactPipeline pipeline;ResidentLinearClipper clipper;Results result;
    for(int pass=0;pass<9;++pass) {
        positions=original;deformable.assign(16,1);params.dt=.02;
        if(pass==1)for(auto& p:positions)p[0]+=.001; // Reuse unchanged boxes/certificates.
        if(pass==4)deformable[7]=0; // Any non-deformable endpoint must veto pruning.
        previous=original;for(auto& p:previous)p[0]-=.001;
        if(pass==5)previous[4][1]=NAN; // Invalid old input must NOT be hidden.
        if(pass==6)params.dt=0; // Invalid parameters must disable certification.
        x.write(positions);old.write(previous);d_deformable.write(deformable);
        auto blue=boxes(original,pass==2?.75:.005);bounds.write(blue);
        broad.contact_count=pass==7?0:contacts.size();
        ContactBoxPruning pruning{bounds.data,d_deformable.data,xyz(x.data),xyz(old.data),params};
        if(pass!=1)pipeline.rebuild(broad,16,d_blocks.data,nullptr,pass==8?ContactBoxPruning{}:pruning);
        const auto incidence=pipeline.incidence();
        const bool mask_expected=prune&&pass!=6&&pass!=8;
        std::vector<unsigned char> expected(broad.contact_count,0);
        if(mask_expected&&pass!=2)for(std::size_t i=0;i<expected.size();++i)
            expected[i]=i%4==2||(i%4==1&&pass!=4&&pass!=5);
        const std::vector<BpContact> current_contacts(contacts.begin(),contacts.begin()+broad.contact_count);
        check_incidence(incidence,current_contacts,colors,expected,queues,mask_expected);
        require_bytes_equal(colors,d_colors.read(),"pruning changed graph colors");
        require_bytes_equal(color_blocks,d_color_blocks.read(),"pruning changed color group order");
        systems.write(initial);error.write({0});accepted.write(std::vector<V>(16,V{{-99,-99,-99}}));
        targets=positions;for(int n=0;n<16;++n)if(deformable[n])targets[n][0]+=.002;
        if(pass==2)targets[0][2]=-.05; // Retained active pair forces backtracking.
        proposed.write(targets);
        for(int color=0;color<4;++color) {
            pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_work,sdf,params,color,ContactAssemblyPhase::Deformable);
            const auto values=systems.read();result.systems.insert(result.systems.end(),values.begin(),values.end());
            result.errors.push_back(error.read()[0]);
            // Error fixture buffers can be partial in a nondeterministic way;
            // status parity matters, but no failed assembly is committed.
            if(pass==5||pass==6)break;
            require(result.errors.back()==0,"unexpected assembly failure");
            clipper.clip_deformable_color(scene,state,elastic,broad,incidence.block_offsets,incidence.pair_indices,
                bounds.data,d_colors.data,color,xyz(accepted.data),error.data,true,nullptr,incidence.color_pairs);
            const auto endpoints=accepted.read();result.accepted.insert(result.accepted.end(),endpoints.begin(),endpoints.end());
            result.errors.push_back(error.read()[0]);require(result.errors.back()==0,"unexpected CCD failure");
        }
        if(pass==5||pass==6) {
            require(result.errors.back()==20,"invalid-input error hidden by pruning");
            // Discard partial unordered error-state output from byte comparison.
            result.systems.resize(result.systems.size()-16);
        }
        if(pass==2)require(accepted.read()[0][2]>0&&accepted.read()[0][2]<positions[0][2],"active pair failed to limit crossing");
        require_bytes_equal(positions,x.read(),"test clipper unexpectedly committed positions");
    }
    return result;
}
}

int main(){
    try {
        int combinations=0;
        for(bool queues:{false,true})for(bool compact:{false,true})for(bool cooperative:{false,true}) {
            const auto reference=run(false,queues,compact,cooperative);
            const auto candidate=run(true,queues,compact,cooperative);
            require_bytes_equal(reference.systems,candidate.systems,"pruned assembly differs bytewise");
            require_bytes_equal(reference.accepted,candidate.accepted,"pruned CCD endpoint differs bytewise");
            require_bytes_equal(reference.errors,candidate.errors,"pruned error statuses differ");++combinations;
        }
        std::cout<<"PASS: movement-box pruning, "<<combinations<<" queue/compact/cooperative combinations, nine rebuild/reuse states; byte-identical derivatives and CCD endpoints\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}

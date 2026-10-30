// Integration test: link ipc_gpu_solver (and its normal dependencies).
// All comparisons are byte-for-byte; no tolerance masks a scheduling change.
#include "contact_pipeline.h"
#include "linear_pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::solver;
namespace {
using V=std::array<double,3>;
void checked(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void require(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
template<class T> struct Memory {
    T* data=nullptr;std::size_t size;
    explicit Memory(const std::vector<T>& values):size(values.size()) {
        if(size){checked(cudaMalloc(reinterpret_cast<void**>(&data),size*sizeof(T)));write(values);}
    }
    ~Memory(){cudaFree(data);}
    void write(const std::vector<T>& values) {
        require(values.size()==size,"test buffer length mismatch");
        if(size)checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> read()const {
        std::vector<T> values(size);if(size)checked(cudaMemcpy(values.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return values;
    }
};
struct Environment {
    struct Entry {std::string key,value;bool present;};std::vector<Entry> entries;
    void set(const char* key,const char* value) {
        const char* prior=std::getenv(key);entries.push_back({key,prior?prior:"",prior!=nullptr});
        if(value)setenv(key,value,1);else unsetenv(key);
    }
    ~Environment(){for(auto i=entries.rbegin();i!=entries.rend();++i)if(i->present)setenv(i->key.c_str(),i->value.c_str(),1);else unsetenv(i->key.c_str());}
};
double(*xyz(V* values))[3]{return reinterpret_cast<double(*)[3]>(values);}
template<class T>void same(const std::vector<T>& left,const std::vector<T>& right,const std::string& context) {
    require(left.size()==right.size(),context+": size mismatch");
    require(left.empty()||std::memcmp(left.data(),right.data(),left.size()*sizeof(T))==0,context+": byte mismatch");
}

void check_worklist(ColorContactWorklist work,const std::vector<BpContact>& contacts,
    const std::vector<int>& vertex_blocks,const std::vector<int>& colors,int color_count,bool enabled) {
    if(!enabled) {
        require(!work.offsets&&!work.pair_indices&&!work.max_color_pairs,"disabled worklist exposed storage");return;
    }
    require(work.offsets!=nullptr,"enabled worklist lacks offsets");
    std::vector<std::size_t> offsets(color_count+1);
    checked(cudaMemcpy(offsets.data(),work.offsets,offsets.size()*sizeof(std::size_t),cudaMemcpyDeviceToHost));
    std::vector<int> pairs(offsets.back());
    if(!pairs.empty())checked(cudaMemcpy(pairs.data(),work.pair_indices,pairs.size()*sizeof(int),cudaMemcpyDeviceToHost));
    require(offsets.front()==0,"color offsets do not start at zero");
    std::size_t maximum=0;
    for(int color=0;color<color_count;++color) {
        std::vector<int> expected;
        for(std::size_t pair=0;pair<contacts.size();++pair) {
            bool selected=false;
            for(int role=0;role<4;++role)selected=selected||colors[vertex_blocks[contacts[pair].vertices[role]]]==color;
            if(selected)expected.push_back(int(pair));
        }
        require(offsets[color]<=offsets[color+1]&&offsets[color+1]<=pairs.size(),"nonmonotone color offsets");
        const std::vector<int> actual(pairs.begin()+offsets[color],pairs.begin()+offsets[color+1]);
        same(expected,actual,"stable unique candidate IDs for color "+std::to_string(color));
        maximum=std::max(maximum,expected.size());
    }
    require(work.max_color_pairs==maximum,"incorrect maximum per-color candidate count");
}

struct AssemblyResult {std::vector<Derivatives> snapshots;std::vector<int> statuses;};
AssemblyResult assembly_fixture(int fixture,bool queues,bool warp,bool compact=false,bool cooperative=false,
    bool selected=false,bool compact_warp=false,const char* cutoff="32",bool color_lists=true,const char* selected_threads="8",double contact_friction=-1.) {
    Environment env;
    env.set("IPC_GPU_COLOR_PAIR_QUEUES",queues?"1":"0");
    env.set("IPC_GPU_WARP_CONTACT_REDUCE",warp?"1":"0");
    env.set("IPC_GPU_COOPERATIVE_EXACT_CONTACT",cooperative?"1":"0");
    env.set("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",selected?"1":"0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_THREADS",selected_threads);
    env.set("IPC_GPU_COMPACT_WARP_REDUCE",compact_warp?"1":"0");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE",cutoff);
    env.set("IPC_GPU_PROFILE_CSV",nullptr);env.set("IPC_GPU_CONTACT_QUEUE_CSV",nullptr);
    const std::vector<int> vertex_blocks{0,8,8,8,9,9,6,7,1,2,3};
    const std::vector<int> colors{0,-1,-1,-1,-1,-1,fixture==7?0:1,2,1,0};
    const std::vector<int> color_blocks=fixture==7?std::vector<int>{0,6,9,8,7}:std::vector<int>{0,9,6,8,7};
    const std::vector<int> color_offsets=fixture==7?std::vector<int>{0,3,4,5}:std::vector<int>{0,2,4,5};
    const std::vector<int> body_map{-1,0,0,0,1,1,-1,-1,-1,-1,-1},body_blocks{8,9},body_nodes{1,2,3,4,5};
    const std::vector<std::size_t> body_offsets{0,3,5};
    const std::vector<unsigned char> deformable{1,0,0,0,0,0,1,1,0,0,0},
        solid{0,0,0,0,0,0,1,1,0,0,0},surface{1,1,1,1,1,1,1,0,0,0,0},enabled{1,1};
    std::vector<V> positions{{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0},
                             {2,0,0},{3,0,0},{2.5,-.5,.17},{2.5,.5,.17},
                             {0,0,0},{1,0,0},{0,1,0}};
    if(fixture==3)positions[0]={{.25,.25,std::ldexp(1.,-500)}};
    if(fixture==5)positions[0][2]=1.17; // A candidate remains present but inactive.
    std::vector<V> previous=positions,references(positions.size());
    for(std::size_t node=0;node<positions.size();++node)for(int k=0;k<3;++k) {
        previous[node][k]-=.001*(node+1)*(k+1);
        if(body_map[node]>=0)references[node][k]=positions[node][k]-(body_map[node]==0?V{{.3,.3,0}}:V{{2.5,0,0}})[k];
    }
    if(fixture==9)previous[0][1]=std::numeric_limits<double>::quiet_NaN();
    std::vector<BpContact> contacts;
    for(int pair=0;pair<(fixture==0?0:fixture>=6?132:20);++pair) {
        // Three roles on one rigid block, three fixed (-1 color) roles,
        // duplicate geometry with different original pair IDs, and an
        // entirely fixed candidate must all retain their intended identities.
        const BpContact alternatives[]={{{0,1,2,3},0},{{4,5,6,7},1},{{0,8,9,10},0},{{8,8,9,10},0}};
        contacts.push_back(alternatives[pair%4]);
        // Two selected deformable roles in a valid PT require the full-output
        // fallback even while other pairs use compact records in this color.
        if(fixture==7&&pair%4==0)contacts.back()={{0,6,2,3},0};
        // Repeated selected triangle IDs require the original role1/role2
        // full fallback, even in a zero-friction compact assembly.
        if(fixture==8&&pair%4==0)contacts.back()={{8,0,0,3},0};
    }
    Memory<V> x(positions),previous_x(previous),reference_x(references);
    Memory<int> blocks(vertex_blocks),d_colors(colors),d_color_blocks(color_blocks),d_color_offsets(color_offsets),
        d_bodies(body_map),d_body_blocks(body_blocks),d_body_nodes(body_nodes);
    Memory<std::size_t> d_body_offsets(body_offsets);
    Memory<unsigned char> d_deformable(deformable),d_solid(solid),d_surface(surface),d_enabled(enabled);
    Memory<BpContact> d_contacts(contacts);
    Memory<Derivatives> systems(std::vector<Derivatives>(positions.size())),rigid_systems(std::vector<Derivatives>(2));
    Memory<int> failed(std::vector<int>{fixture==4?123:0});
    std::vector<std::array<double,4>> orientation{{{.97,.03,-.05,.08}},{{.97,.06,-.05,.08}}};
    for(auto& q:orientation){double length=0;for(double v:q)length+=v*v;length=std::sqrt(length);for(double& v:q)v/=length;}
    Memory<std::array<double,4>> d_orientation(orientation);
    Memory<V> omega(std::vector<V>{{.13,-.21,.08},{.26,-.42,.16}}),com(std::vector<V>(2));
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.08;collider.current_pose.translation[0]=.003;
    Memory<SdfCollider> d_collider(std::vector<SdfCollider>{collider});
    SdfParameters sdf_parameters;sdf_parameters.stiffness=170;sdf_parameters.epsilon=.2;sdf_parameters.dt=.031;
    sdf_parameters.friction=fixture==2?0:.2;sdf_parameters.eps_v=.1;
    DeformableScene scene{};scene.nodes=positions.size();scene.node_blocks=blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=xyz(x.data);state.previous=xyz(previous_x.data);
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.error=failed.data;
    RigidScene rigid{};rigid.bodies=2;rigid.body_blocks=d_body_blocks.data;rigid.node_body=d_bodies.data;
    rigid.reference=reinterpret_cast<const double(*)[3]>(reference_x.data);rigid.node_offsets=d_body_offsets.data;rigid.nodes=d_body_nodes.data;
    rigid.translate=d_enabled.data;rigid.rotate=d_enabled.data;
    RigidState rigid_state{};rigid_state.previous_orientation=reinterpret_cast<double(*)[4]>(d_orientation.data);
    rigid_state.orientation=rigid_state.previous_orientation;rigid_state.omega=xyz(omega.data);rigid_state.com=xyz(com.data);
    RigidWorkspace rigid_workspace{};rigid_workspace.systems=rigid_systems.data;
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();broad.block_colors=d_colors.data;
    broad.color_count=3;broad.color_blocks=color_lists?d_color_blocks.data:nullptr;
    broad.color_offsets=color_lists?d_color_offsets.data:nullptr;
    SdfScene sdf{d_collider.data,fixture==6?0:1,sdf_parameters};
    ContactParameters params{.8,370,.031,contact_friction>=0?contact_friction:(fixture==2?0.:.2),.1,1e-12};
    GpuContactPipeline pipeline;pipeline.rebuild(broad,int(colors.size()),blocks.data,d_surface.data);
    check_worklist(pipeline.incidence().color_pairs,contacts,vertex_blocks,colors,3,queues);
    AssemblyResult result;
    for(int pass=0;pass<4;++pass) {
        const auto phase=pass==0?ContactAssemblyPhase::Deformable:pass==1?ContactAssemblyPhase::RigidTranslation:ContactAssemblyPhase::RigidOrientation;
        if(pass>=2)rigid_systems.write(std::vector<Derivatives>(2));
        if(pass==2) {
            for(std::size_t node=0;node<positions.size();++node)if(body_map[node]>=0)
                for(int k=0;k<3;++k)positions[node][k]+=(body_map[node]+1)*V{{.001,-.003,.004}}[k];
            x.write(positions);
        }
        if(pass==3)omega.write(std::vector<V>{{-.22,.09,-.05},{-.44,.18,-.10}});
        for(int color=0;color<3;++color) {
            pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,params,color,phase);
            const auto derivatives=pass==0?systems.read():rigid_systems.read();
            result.snapshots.insert(result.snapshots.end(),derivatives.begin(),derivatives.end());
            result.statuses.push_back(failed.read()[0]);
            require(result.statuses.back()==(fixture==4?123:fixture==9?20:0),"unexpected contact fixture error");
        }
    }
    return result;
}

struct CcdResult {std::vector<V> accepted;std::vector<int> errors;};

AssemblyResult rebuild_lifecycle_fixture(bool queues,bool warp,bool compact=false,
    bool selected=false,bool compact_warp=false,const char* cutoff="32",bool cooperative=false,
    bool mixed_friction=false) {
    Environment env;env.set("IPC_GPU_COLOR_PAIR_QUEUES",queues?"1":"0");
    env.set("IPC_GPU_WARP_CONTACT_REDUCE",warp?"1":"0");
    env.set("IPC_GPU_COOPERATIVE_EXACT_CONTACT",cooperative?"1":"0");
    env.set("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",compact?"1":"0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",selected?"1":"0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_THREADS","8");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE",compact_warp?"1":"0");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE",cutoff);
    env.set("IPC_GPU_PROFILE_CSV",nullptr);env.set("IPC_GPU_CONTACT_QUEUE_CSV",nullptr);
    const std::vector<V> positions{{.15,.10,.13},{.25,.12,.15},{.35,.14,.17},{.4,.04,.19},
                                 {0,0,0},{1,0,0},{0,1,0},{2,2,2}};
    auto previous=positions;
    for(int node=0;node<4;++node){previous[node][0]-=.003;previous[node][1]-=.001;}
    Memory<V> x(positions),previous_x(previous);
    Memory<int> blocks(std::vector<int>(8)),d_colors(std::vector<int>(8)),
        d_color_blocks(std::vector<int>(4)),d_color_offsets(std::vector<int>(5));
    Memory<unsigned char> deformable(std::vector<unsigned char>{1,1,1,1,0,0,0,0}),
        solid(std::vector<unsigned char>{0,1,1,0,0,0,0,0}),
        surface(std::vector<unsigned char>{1,1,0,1,0,0,0,0});
    std::vector<Derivatives> initial(8);
    for(int node=0;node<8;++node) {
        for(int k=0;k<3;++k)initial[node].gradient[k]=.125*(node+1)/(k+1);
        for(int k=0;k<9;++k)initial[node].hessian[k]=.375*(node+1)/(k+1);
    }
    Memory<Derivatives> systems(initial);Memory<int> failed(std::vector<int>{0});
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.1;
    Memory<SdfCollider> d_collider(std::vector<SdfCollider>{collider});
    SdfParameters sdf_parameters;sdf_parameters.stiffness=170;sdf_parameters.epsilon=.2;
    sdf_parameters.dt=.031;sdf_parameters.friction=.1;sdf_parameters.eps_v=.1;
    DeformableScene scene{};scene.nodes=8;scene.node_blocks=blocks.data;scene.deformable=deformable.data;scene.solid=solid.data;
    DeformableState state{};state.positions=xyz(x.data);state.previous=xyz(previous_x.data);
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.error=failed.data;
    RigidScene rigid{};RigidState rigid_state{};RigidWorkspace rigid_workspace{};
    SdfScene sdf{d_collider.data,1,sdf_parameters};
    GpuContactPipeline pipeline; // Deliberately reused across all rebuilds.
    AssemblyResult result;
    const int counts[]={17,3,0,41,9,33,0,65};
    const int assignments[][4]={{0,1,2,3},{2,0,1,3},{1,1,0,3},{0,1,2,3},
                                {2,1,0,3},{1,0,2,3},{0,2,1,3},{2,0,1,3}};
    for(int rebuild=0;rebuild<8;++rebuild) {
        std::vector<int> vertex_blocks(8),colors(8,-1),color_blocks,color_offsets;
        for(int node=0;node<8;++node)vertex_blocks[node]=node<4?(node+rebuild)%4:node;
        for(int node=0;node<4;++node)colors[vertex_blocks[node]]=assignments[rebuild][node];
        for(int color=0;color<4;++color) {
            color_offsets.push_back(int(color_blocks.size()));
            for(int block=0;block<8;++block)if(colors[block]==color)color_blocks.push_back(block);
        }
        color_offsets.push_back(int(color_blocks.size()));
        blocks.write(vertex_blocks);d_colors.write(colors);d_color_blocks.write(color_blocks);d_color_offsets.write(color_offsets);
        std::vector<BpContact> contacts;
        for(int pair=0;pair<counts[rebuild];++pair)contacts.push_back({{pair%3,4,5,6},0});
        Memory<BpContact> d_contacts(contacts);
        BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();
        broad.block_colors=d_colors.data;broad.color_count=4;broad.color_blocks=d_color_blocks.data;broad.color_offsets=d_color_offsets.data;
        pipeline.rebuild(broad,8,blocks.data,surface.data);
        check_worklist(pipeline.incidence().color_pairs,contacts,vertex_blocks,colors,4,queues);
        // A zero barrier with a NONEMPTY candidate list follows a previously
        // active assembly, then the next rebuild enables it again. Stale pair
        // activity or old launch metadata must never survive these changes.
        ContactParameters params{.8,rebuild==4?0.:370.,.031,.1,.1,1e-12};
        auto assemble=[&](int color) {
            const double friction=params.friction;
            const double sequence[]={0.,.2,-0.,.1,0.};
            for(int pass=0;pass<(mixed_friction?5:1);++pass) {
                params.friction=mixed_friction?sequence[pass]:friction;
                pipeline.assemble_color(scene,state,elastic,rigid,rigid_state,rigid_workspace,sdf,params,color,ContactAssemblyPhase::Deformable);
                if(mixed_friction) {
                    const auto intermediate=systems.read();
                    result.snapshots.insert(result.snapshots.end(),intermediate.begin(),intermediate.end());
                    result.statuses.push_back(failed.read()[0]);
                    require(result.statuses.back()==0,"mixed-friction intermediate assembly failed");
                }
            }
            params.friction=friction;
        };
        systems.write(initial);failed.write(std::vector<int>{0});
        for(int color=0;color<4;++color) {
            assemble(color);
            const auto output=systems.read();result.snapshots.insert(result.snapshots.end(),output.begin(),output.end());
            result.statuses.push_back(failed.read()[0]);require(result.statuses.back()==0,"rebuild lifecycle assembly failed");
            if(color==2)require(std::memcmp(&output[3],&initial[3],sizeof(Derivatives))==0,"contactless-color node updated before its color");
            if(color==3) {
                // Node3 has no pair in ANY rebuild, but its SDF contribution
                // must still be processed even with an empty color worklist.
                require(std::memcmp(&output[3],&initial[3],sizeof(Derivatives))!=0,"contactless color incorrectly skipped SDF assembly");
            }
        }
        // Replace either scene pointer without rebuilding the pipeline. A
        // changed mask removes a formerly mapped block; later restoring the
        // original pointer must remap it, even with identical node counts.
        Memory<int> alternate_blocks(vertex_blocks);
        Memory<unsigned char> alternate_mask(std::vector<unsigned char>{1,0,1,1,0,0,0,0});
        for(int pointer_pass=0;pointer_pass<3;++pointer_pass) {
            scene.node_blocks=pointer_pass==0?alternate_blocks.data:blocks.data;
            scene.deformable=pointer_pass==1?alternate_mask.data:deformable.data;
            systems.write(initial);failed.write(std::vector<int>{0});
            for(int color=0;color<4;++color) {
                assemble(color);
                const auto output=systems.read();result.snapshots.insert(result.snapshots.end(),output.begin(),output.end());
                result.statuses.push_back(failed.read()[0]);require(result.statuses.back()==0,"pointer lifecycle assembly failed");
                if(pointer_pass==1)require(std::memcmp(&output[1],&initial[1],sizeof(Derivatives))==0,"stale deformable mask mapping");
            }
        }
    }
    return result;
}

CcdResult ccd_fixture(int fixture,bool queues) {
    Environment env;env.set("IPC_GPU_COLOR_PAIR_QUEUES",queues?"1":"0");
    env.set("IPC_GPU_WARP_CONTACT_REDUCE","0");env.set("IPC_GPU_COOPERATIVE_EXACT_CONTACT","0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE","0");env.set("IPC_GPU_COMPACT_WARP_REDUCE","0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_THREADS","8");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE","32");
    const bool cache=fixture==1,tiny=fixture==2;
    const std::vector<V> positions=cache
        ?std::vector<V>{{-1,0,1e-9},{-.1,-1,0},{.1,-1,0},{0,1,0},{.5,-1,-1},{.5,1,-1},{.5,0,1}}
        :std::vector<V>{{.25,.25,tiny?1e-12:1.},{0,0,0},{1,0,0},{0,1,0},{0,0,.75},{1,0,.75},{0,1,.75}};
    std::vector<BpContact> contacts;
    for(int p=0;p<(cache?2:129);++p)contacts.push_back(p%2?BpContact{{0,4,5,6},0}:BpContact{{0,1,2,3},0});
    std::vector<int> blocks{0,1,2,3,4,5,6},colors=blocks,color_offsets{0,1,2,3,4,5,6,7};
    Memory<V> x(positions),proposed(positions),accepted(std::vector<V>(7,V{{-99,-99,-99}}));
    Memory<int> d_blocks(blocks),d_colors(colors),d_color_offsets(color_offsets),error(std::vector<int>{0});
    Memory<unsigned char> mask(std::vector<unsigned char>{1,0,0,0,0,0,0});
    Memory<BpContact> d_contacts(contacts);
    Memory<BpAabb> boxes(std::vector<BpAabb>(7,BpAabb{{-2,-2,-2},{2,2,2}}));
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();broad.block_colors=d_colors.data;
    broad.color_count=7;broad.color_blocks=d_blocks.data;broad.color_offsets=d_color_offsets.data;
    GpuContactPipeline pipeline;pipeline.rebuild(broad,7,d_blocks.data);
    const auto incidence=pipeline.incidence();check_worklist(incidence.color_pairs,contacts,blocks,colors,7,queues);
    DeformableScene scene{};scene.nodes=7;scene.deformable=mask.data;scene.node_blocks=d_blocks.data;
    DeformableState state{};state.positions=xyz(x.data);
    ElasticWorkspace elastic{};elastic.proposed=xyz(proposed.data);elastic.error=error.data;
    GpuLinearClipper clipper;CcdResult result;
    for(int sweep=0;sweep<3;++sweep) {
        auto targets=positions;
        if(cache)targets[0][0]=sweep==1?-1.5:1.;
        else targets[0][2]=tiny?(sweep==1?1e-12:2e-12):(sweep==1?1.5:-1.);
        proposed.write(targets);
        clipper.clip_deformable_color(scene,state,elastic,broad,incidence.block_offsets,incidence.pair_indices,
            boxes.data,d_colors.data,0,xyz(accepted.data),error.data,true,nullptr,incidence.color_pairs);
        const auto output=accepted.read();result.accepted.insert(result.accepted.end(),output.begin(),output.end());
        result.errors.push_back(error.read()[0]);require(result.errors.back()==0,"CCD equivalence fixture failed");
        same(x.read(),positions,"CCD committed positions");
    }
    // Empty-motion color after a populated color exercises reuse of queues.
    clipper.clip_deformable_color(scene,state,elastic,broad,incidence.block_offsets,incidence.pair_indices,
        boxes.data,d_colors.data,6,xyz(accepted.data),error.data,true,nullptr,incidence.color_pairs);
    const auto untouched=accepted.read();result.accepted.insert(result.accepted.end(),untouched.begin(),untouched.end());
    result.errors.push_back(error.read()[0]);
    return result;
}

AssemblyResult reduction_failure_fixture(int failure,bool selected,bool compact_warp) {
    Environment env;
    env.set("IPC_GPU_COLOR_PAIR_QUEUES","1");env.set("IPC_GPU_WARP_CONTACT_REDUCE","0");
    env.set("IPC_GPU_COOPERATIVE_EXACT_CONTACT","0");env.set("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS","1");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",selected?"1":"0");
    env.set("IPC_GPU_CONTACT_SELECTED_NODE_THREADS","8");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE",compact_warp?"1":"0");
    env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE","1");
    env.set("IPC_GPU_PROFILE_CSV",nullptr);env.set("IPC_GPU_CONTACT_QUEUE_CSV",nullptr);
    Memory<V> positions(std::vector<V>{{.2,.2,failure==4?0.:.1},{0,0,0},{1,0,0},{0,1,0}});
    Memory<int> blocks(std::vector<int>{0,1,2,3}),colors(std::vector<int>{0,-1,-1,-1}),
        color_blocks(std::vector<int>{0}),color_offsets(std::vector<int>{0,1});
    Memory<unsigned char> deformable(std::vector<unsigned char>{1,0,0,0}),solid(std::vector<unsigned char>(4));
    Memory<BpContact> contacts(std::vector<BpContact>{{{0,1,2,3},0}});
    std::vector<Derivatives> initial(4);
    if(failure==1)initial[0].hessian[8]=std::numeric_limits<double>::infinity();
    Memory<Derivatives> systems(initial);Memory<int> failed(std::vector<int>{failure==3?123:0});
    SdfCollider collider;collider.kind=SdfKind::Plane;collider.point[1]=.05;
    Memory<SdfCollider> colliders(std::vector<SdfCollider>{collider});
    SdfParameters sdf_parameters;sdf_parameters.stiffness=170;sdf_parameters.epsilon=.2;
    sdf_parameters.dt=failure==2?0.:.031;sdf_parameters.friction=.1;sdf_parameters.eps_v=.1;
    DeformableScene scene{};scene.nodes=4;scene.node_blocks=blocks.data;scene.deformable=deformable.data;scene.solid=solid.data;
    DeformableState state{};state.positions=xyz(positions.data);state.previous=state.positions;
    ElasticWorkspace elastic{};elastic.systems=systems.data;elastic.error=failed.data;
    BroadPhaseView broad{};broad.contact_count=1;broad.contacts=contacts.data;broad.block_colors=colors.data;
    broad.color_count=1;broad.color_blocks=color_blocks.data;broad.color_offsets=color_offsets.data;
    GpuContactPipeline pipeline;pipeline.rebuild(broad,4,blocks.data);
    pipeline.assemble_color(scene,state,elastic,RigidScene{},RigidState{},RigidWorkspace{},
        SdfScene{colliders.data,1,sdf_parameters},ContactParameters{.8,370,.031,.1,.1,1e-12},0,ContactAssemblyPhase::Deformable);
    AssemblyResult result{systems.read(),failed.read()};
    const int expected=failure==1?26:failure==2?25:failure==3?123:21;
    require(result.statuses[0]==expected,"selected/compact reduction error contract mismatch");
    same(initial,result.snapshots,"failed node committed partial derivatives");
    return result;
}
} // namespace

#ifndef IPC_CONTACT_WORKLIST_FIXTURE_ONLY
int main() {
    try {
        for(int fixture=0;fixture<8;++fixture) {
            const auto baseline=assembly_fixture(fixture,false,false);
            for(const auto flags: {std::array<bool,2>{{true,false}},std::array<bool,2>{{false,true}},std::array<bool,2>{{true,true}}}) {
                const auto trial=assembly_fixture(fixture,flags[0],flags[1]);
                same(baseline.snapshots,trial.snapshots,"contact phase/color fixture "+std::to_string(fixture));
                same(baseline.statuses,trial.statuses,"contact phase/color error state");
            }
            for(bool cooperative:{false,true}) {
                const auto trial=assembly_fixture(fixture,true,false,true,cooperative);
                same(baseline.snapshots,trial.snapshots,"compact contact phase/color fixture "+std::to_string(fixture));
                same(baseline.statuses,trial.statuses,"compact contact error state");
                for(bool queues:{false,true}) {
                    const auto scalar=assembly_fixture(fixture,queues,false,true,cooperative,true);
                    same(baseline.snapshots,scalar.snapshots,"selected compact scalar fixture "+std::to_string(fixture));
                    same(baseline.statuses,scalar.statuses,"selected compact scalar statuses");
                    // 1 forces every nonempty node through the compact warp;
                    // 6 splits the ordinary fixture,32 exercises the default
                    // heavy-node cutoff, and INT_MAX keeps every node scalar.
                    for(const char* cutoff:{"1","6","32","2147483647"}) {
                        const auto warp_trial=assembly_fixture(fixture,queues,false,true,cooperative,false,true,cutoff);
                        same(baseline.snapshots,warp_trial.snapshots,"compact component warp fixture "+std::to_string(fixture)+" cutoff "+cutoff);
                        same(baseline.statuses,warp_trial.statuses,"compact component warp statuses");
                    }
                }
            }
            const auto selected_full=assembly_fixture(fixture,true,false,false,false,true);
            same(baseline.snapshots,selected_full.snapshots,"selected full-output scalar fixture");
            same(baseline.statuses,selected_full.statuses,"selected full-output scalar statuses");
            const auto missing_lists=assembly_fixture(fixture,true,false,true,false,true,true,"1",false);
            same(baseline.snapshots,missing_lists.snapshots,"missing color-list scalar fallback");
            same(baseline.statuses,missing_lists.statuses,"missing color-list scalar statuses");
            if(fixture==7)for(const char* selected_threads:{"1","128"}) {
                const auto trial=assembly_fixture(fixture,true,false,true,true,true,true,"32",true,selected_threads);
                same(baseline.snapshots,trial.snapshots,"selected launch-width derivative bytes");
                same(baseline.statuses,trial.statuses,"selected launch-width statuses");
            }
        }
        for(int fixture=0;fixture<3;++fixture) {
            const auto baseline=ccd_fixture(fixture,false),trial=ccd_fixture(fixture,true);
            same(baseline.accepted,trial.accepted,"CCD endpoints fixture "+std::to_string(fixture));
            same(baseline.errors,trial.errors,"CCD error state");
        }
        const auto lifecycle_baseline=rebuild_lifecycle_fixture(false,false);
        const auto compact_lifecycle=rebuild_lifecycle_fixture(true,false,true);
        same(lifecycle_baseline.snapshots,compact_lifecycle.snapshots,"compact rebuild lifecycle");
        same(lifecycle_baseline.statuses,compact_lifecycle.statuses,"compact rebuild statuses");
        for(const auto flags: {std::array<bool,2>{{true,false}},std::array<bool,2>{{false,true}},std::array<bool,2>{{true,true}}}) {
            const auto trial=rebuild_lifecycle_fixture(flags[0],flags[1]);
            same(lifecycle_baseline.snapshots,trial.snapshots,"same-pipeline rebuild lifecycle");
            same(lifecycle_baseline.statuses,trial.statuses,"rebuild lifecycle error state");
        }
        for(bool queues:{false,true})for(bool cooperative:{false,true}) {
            const auto selected=rebuild_lifecycle_fixture(queues,false,true,true,false,"32",cooperative);
            same(lifecycle_baseline.snapshots,selected.snapshots,"selected scalar rebuild/pointer lifecycle");
            same(lifecycle_baseline.statuses,selected.statuses,"selected scalar rebuild/pointer statuses");
            for(const char* cutoff:{"1","6","32"}) {
                const auto trial=rebuild_lifecycle_fixture(queues,false,true,true,true,cutoff,cooperative);
                same(lifecycle_baseline.snapshots,trial.snapshots,"compact warp rebuild/pointer lifecycle");
                same(lifecycle_baseline.statuses,trial.statuses,"compact warp rebuild/pointer statuses");
            }
        }
        for(int failure=1;failure<=4;++failure) {
            const auto baseline=reduction_failure_fixture(failure,false,false);
            for(bool compact_warp:{false,true}) {
                const auto trial=reduction_failure_fixture(failure,true,compact_warp);
                same(baseline.snapshots,trial.snapshots,"reduction failure derivatives");
                same(baseline.statuses,trial.statuses,"reduction failure statuses");
            }
        }
        for(const char* invalid:{"","0","-1","1.5","32x","2147483648"}) {
            Environment env;env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE",invalid);
            bool rejected=false;
            try{GpuContactPipeline pipeline;}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"invalid compact warp degree cutoff accepted");
        }
        for(const char* invalid:{"","0","-1","3","128x","129"}) {
            Environment env;env.set("IPC_GPU_CONTACT_SELECTED_NODE_THREADS",invalid);
            env.set("IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE","32");
            bool rejected=false;
            try{GpuContactPipeline pipeline;}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"invalid selected node launch width accepted");
        }
        std::cout<<"Color worklists: mixed cloth/solid/rigid per-phase bytes, stable unique pair IDs, fixed negative colors, selected scalar/compact component warp cutoffs, compact/full fallback, cooperative/queue combinations, CCD cache/backtrack/tiny-gap, rebuild growth/shrink/zero/recolor/remap/pointer changes, zero barrier, contactless-color SDF equivalence PASS\n";
    } catch(const std::exception& failure) {
        std::cerr<<failure.what()<<'\n';return 1;
    }
}

#endif // IPC_CONTACT_WORKLIST_FIXTURE_ONLY

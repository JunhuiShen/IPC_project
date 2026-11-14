// Maintained public-API functional parity test. Derived from the qualified
// ScopeV2 fixture; transient launch observation/fault hooks are excluded.
#include "solver.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace ipc_gpu::solver;
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
class Flag {
    bool existed;std::string previous;
public:
    Flag():existed(std::getenv("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL")!=nullptr) {
        if(existed)previous=std::getenv("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL");
    }
    ~Flag(){if(existed)setenv("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL",previous.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL");}
    void set(const char* value){require(setenv("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL",value,1)==0,"cannot select solver proxy flag");}
};
template<class T>bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
void equal(const StateSnapshot& a,const StateSnapshot& b) {
    require(same(a.positions,b.positions)&&same(a.velocities,b.velocities)&&same(a.com,b.com)
        &&same(a.com_velocity,b.com_velocity)&&same(a.omega,b.omega)&&same(a.orientation,b.orientation),
        "actual GpuSolver proxy OFF/ON state bytes differ");
}
void finite(const StateSnapshot& state) {
    const auto values=[](const auto& vector) {
        for(const auto& value:vector)for(double x:value)if(!std::isfinite(x))return false;
        return true;
    };
    require(values(state.positions)&&values(state.velocities)&&values(state.com)&&values(state.com_velocity)
        &&values(state.omega)&&values(state.orientation),"actual solver returned nonfinite state");
}
SceneDescription scene(std::size_t moving_nodes,bool fixed=false,bool empty_contacts=false) {
    require(moving_nodes>=3,"public fixture requires triangle proxies");
    SceneDescription h;
    h.positions={Vector3{-1,-1,0},Vector3{1,-1,0},Vector3{0,1,0},
                 Vector3{-.2,-.2,.003},Vector3{.2,-.2,.003},Vector3{0,.2,.003}};
    while(h.positions.size()<3+moving_nodes)h.positions.push_back(Vector3{10.+double(h.positions.size()),0,.003});
    const auto n=h.positions.size();
    h.velocities.resize(n);h.pin_targets=h.positions;h.body_reference.resize(n);
    h.deformable.assign(n,0);h.solid.assign(n,0);h.pinned.assign(n,0);h.surface.assign(n,0);
    for(int node=0;node<6;++node)h.surface[node]=1;
    h.mass.assign(n,1.);h.node_blocks.resize(n);h.node_body.resize(n);
    h.block_count=2;h.body_blocks={0,1};h.body_node_offsets={0,3,n};
    h.body_mass={1.,1.};h.body_second_moment.assign(2,Matrix3{1,0,0,0,1,0,0,0,1});
    h.body_translate={0,static_cast<unsigned char>(!fixed)};h.body_rotate={0,0};
    h.com={Vector3{0,0,0},Vector3{0,0,.003}};
    h.com_velocity={Vector3{0,0,0},Vector3{0,0,fixed?0.:-.2}};
    h.omega.assign(2,Vector3{});h.orientation.assign(2,Quaternion{1,0,0,0});
    for(std::size_t node=0;node<n;++node) {
        const int body=node<3?0:1;h.node_blocks[node]=h.node_body[node]=body;
        h.body_nodes.push_back(static_cast<int>(node));
        for(int axis=0;axis<3;++axis)h.body_reference[node][axis]=h.positions[node][axis]-h.com[body][axis];
        h.velocities[node]=h.com_velocity[body];
    }
    h.triangle_offsets.assign(n+1,0);h.solid_offsets.assign(n+1,0);h.hinge_offsets.assign(n+1,0);
    if(!empty_contacts) {
        h.surface_triangles={BpTriangle{{0,1,2}},BpTriangle{{3,4,5}}};
        h.edges={BpEdge{{0,1}},BpEdge{{1,2}},BpEdge{{2,0}},BpEdge{{3,4}},BpEdge{{4,5}},BpEdge{{5,3}}};
    }
    return h;
}
StepParameters parameters(int iterations) {
    StepParameters p;p.exact_computation_fallback=false;p.dt=.01;p.elastic.dt2=p.dt*p.dt;
    p.contact.dt=p.sdf.dt=p.dt;p.contact.d_hat=.01;p.contact.k_barrier=10000;
    p.contact.eps=1e-12;p.contact.eps_v=.001;p.contact.friction=0;
    p.elastic.gravity[0]=p.elastic.gravity[1]=p.elastic.gravity[2]=0;
    p.damping=1.;p.iterations=iterations;p.rebuild_interval=1;p.box_radius_min=.002;p.box_radius=.02;
    p.rotation_radius=.01;return p;
}
void fixed_and_gap(const StateSnapshot& snapshot,const SceneDescription& original) {
    for(int node=0;node<3;++node)require(snapshot.positions[node]==original.positions[node],"fixed triangle moved");
    require(snapshot.com[0]==original.com[0]&&snapshot.orientation[0]==original.orientation[0],"fixed pose changed");
    for(int node=3;node<6;++node)require(snapshot.positions[node][2]>0.,"moving triangle crossed fixed plane");
}

}
int main(){try{
    Flag flag;std::size_t comparisons=0,requested_sweeps=0,force_witnesses=0;
    for(std::size_t moving:{3u,129u,257u})for(int iterations:{1,2}) {
        const auto h=scene(moving);auto p=parameters(iterations);
        flag.set("0");GpuSolver off(h);flag.set("1");GpuSolver on(h);
        for(int step=0;step<4;++step) {
            // A malformed postconstruction flag must not change captured owners.
            flag.set("invalid");off.advance_substep(p,h.pin_targets,{});
            const auto expected=off.snapshot();on.advance_substep(p,h.pin_targets,{});
            const auto actual=on.snapshot();
            equal(actual,expected);finite(actual);fixed_and_gap(actual,h);++comparisons;requested_sweeps+=iterations;
            require(actual.com[1]!=h.com[1],"owned solver motion fixture is vacuous");
        }
        // An additional ordinary step keeps captured proxy ownership valid.
        p.iterations=1;
        off.advance_substep(p,h.pin_targets,{});
        on.advance_substep(p,h.pin_targets,{});
        equal(on.snapshot(),off.snapshot());finite(on.snapshot());fixed_and_gap(on.snapshot(),h);++comparisons;++requested_sweeps;
    }
    // Positive barrier response must change the solution relative to an
    // otherwise identical free-inertia control, beyond merely having pairs.
    {
        const auto h=scene(3);auto p=parameters(1);flag.set("0");GpuSolver force(h),free(h);
        force.advance_substep(p,h.pin_targets,{});p.contact.k_barrier=0;free.advance_substep(p,h.pin_targets,{});
        const auto a=force.snapshot(),b=free.snapshot();finite(a);finite(b);
        require(std::abs(a.com[1][2]-b.com[1][2])>1e-10,"barrier-OFF control shows no active force");++force_witnesses;
    }
    for(bool fixed:{false,true}) {
        const auto h=scene(3,fixed,true);auto p=parameters(2);flag.set("0");GpuSolver off(h);flag.set("1");GpuSolver on(h);
        off.advance_substep(p,h.pin_targets,{});
        on.advance_substep(p,h.pin_targets,{});
        equal(on.snapshot(),off.snapshot());finite(on.snapshot());++comparisons;
    }
    // Immutable ownership defects are rejected at construction.
    for(int kind=0;kind<4;++kind) {
        auto bad=scene(3);if(kind==0)bad.body_nodes[1]=bad.body_nodes[0];
        if(kind==1)bad.body_nodes[1]=int(bad.positions.size());
        if(kind==2)bad.node_body[0]=1;
        if(kind==3)bad.node_blocks[0]=1;
        flag.set("1");bool rejected=false;
        try{GpuSolver invalid(bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid owned CSR accepted");
    }
    {
        const auto h=scene(3);flag.set("invalid");bool rejected=false;
        try{GpuSolver invalid(h);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"fresh malformed proxy flag was accepted");
    }
    require(requested_sweeps>0&&force_witnesses>0,"public solver functional acceptance has no requested sweeps/force");
    std::cout<<"Rigid proxy public GpuSolver functional parity PASS: state_comparisons="<<comparisons
        <<" requested_translation_sweeps="<<requested_sweeps<<" active_force_controls="<<force_witnesses
        <<"; functional parity only; route activation is qualified by the separate native observer tests\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}

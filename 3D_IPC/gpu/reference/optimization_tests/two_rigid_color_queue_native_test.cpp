#include "solver.h"
#include "contact_pipeline.h"
#include "device_memory.h"
#include "two_rigid_color_queue_policy.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::solver;
namespace {
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
struct Environment {
    std::string key,value;bool present;
    explicit Environment(const char* name):key(name),present(std::getenv(name)!=nullptr) {
        if(present)value=std::getenv(name);
    }
    ~Environment(){if(present)setenv(key.c_str(),value.c_str(),1);else unsetenv(key.c_str());}
    void set(const char* value_) {require((value_?setenv(key.c_str(),value_,1):unsetenv(key.c_str()))==0,"environment failure");}
};
struct AllocationEvent {bool allocation;std::size_t bytes;};
std::mutex allocation_mutex;
std::map<void*,std::size_t> allocations;
std::vector<AllocationEvent> events;
bool tracking=false,tracking_failed=false;
std::size_t resident_bytes() {
    std::lock_guard<std::mutex> lock(allocation_mutex);std::size_t total=0;
    for(const auto& p:allocations)total+=p.second;return total;
}
template<class T>bool equal_bytes(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
void equal(const StateSnapshot& a,const StateSnapshot& b,const std::string& label) {
    require(equal_bytes(a.positions,b.positions),label+" positions");
    require(equal_bytes(a.velocities,b.velocities),label+" velocities");
    require(equal_bytes(a.com,b.com),label+" COM");
    require(equal_bytes(a.com_velocity,b.com_velocity),label+" COM velocity");
    require(equal_bytes(a.orientation,b.orientation),label+" quaternion");
    require(equal_bytes(a.omega,b.omega),label+" omega");
}
template<class T>void finite(const std::vector<T>& a) {
    for(const auto& v:a)for(double x:v)require(std::isfinite(x),"nonfinite state");
}
void finite(const StateSnapshot& a) {
    finite(a.positions);finite(a.velocities);finite(a.com);finite(a.com_velocity);finite(a.orientation);finite(a.omega);
    for(const auto& q:a.orientation) {
        double squared=0.;for(double x:q)squared+=x*x;
        require(std::abs(std::sqrt(squared)-1.)<=1e-10,"quaternion norm failure");
    }
}
SceneDescription fixture(double first_z=.1) {
    // Public-API rigid contact fixture for the ordinary colored solver:
    // a four-proxy translating/rotating body above one fixed rigid triangle.
    SceneDescription h;
    h.positions={Vector3{.1,0,first_z},Vector3{.2,0,.15},Vector3{.1,.1,.15},Vector3{0,-.1,.15},
        Vector3{-2,-2,0},Vector3{4,-2,0},Vector3{-2,4,0}};
    const std::size_t nodes=h.positions.size();
    h.velocities.assign(nodes,Vector3{});h.pin_targets=h.positions;h.body_reference.resize(nodes);
    h.deformable.assign(nodes,0);h.solid.assign(nodes,0);h.pinned.assign(nodes,0);h.surface.assign(nodes,1);
    h.mass.assign(nodes,0.0);h.node_body={0,0,0,0,1,1,1};h.node_blocks=h.node_body;
    h.block_count=2;h.body_blocks={0,1};h.body_nodes={0,1,2,3,4,5,6};h.body_node_offsets={0,4,7};
    h.body_mass={4.,3.};h.body_translate={1,0};h.body_rotate={1,0};
    h.com.assign(2,Vector3{});h.body_second_moment.assign(2,Matrix3{});
    h.com_velocity={Vector3{.15,0,-.01},Vector3{0,0,0}};
    h.omega={Vector3{0,0,.1},Vector3{0,0,0}};
    h.orientation={Quaternion{1,0,0,0},Quaternion{1,0,0,0}};
    h.triangle_offsets.assign(nodes+1,0);h.solid_offsets.assign(nodes+1,0);h.hinge_offsets.assign(nodes+1,0);
    h.surface_triangles={BpTriangle{{4,5,6}}};
    h.edges={BpEdge{{4,5}},BpEdge{{5,6}},BpEdge{{6,4}}};
    for(std::size_t body=0;body<2;++body) {
        const auto first=h.body_node_offsets[body],last=h.body_node_offsets[body+1];
        const double nodal_mass=h.body_mass[body]/double(last-first);
        for(auto i=first;i<last;++i)for(int axis=0;axis<3;++axis)
            h.com[body][axis]+=nodal_mass*h.positions[h.body_nodes[i]][axis];
        for(double& coordinate:h.com[body])coordinate/=h.body_mass[body];
        for(auto i=first;i<last;++i) {
            const int node=h.body_nodes[i];
            auto& local=h.body_reference[node];
            for(int axis=0;axis<3;++axis)local[axis]=h.positions[node][axis]-h.com[body][axis];
            for(int row=0;row<3;++row)for(int col=0;col<3;++col)
                h.body_second_moment[body][3*row+col]+=nodal_mass*local[row]*local[col];
            const auto& v=h.com_velocity[body];const auto& w=h.omega[body];
            h.velocities[node]=Vector3{v[0]+w[1]*local[2]-w[2]*local[1],
                v[1]+w[2]*local[0]-w[0]*local[2],v[2]+w[0]*local[1]-w[1]*local[0]};
        }
    }
    return h;
}

StepParameters parameters(bool exact) {
    StepParameters p;p.dt=.025;p.elastic.dt2=p.dt*p.dt;p.contact.dt=p.dt;p.sdf.dt=p.dt;
    p.iterations=2;p.rebuild_interval=1;p.damping=.02;p.box_radius_min=.001;p.box_radius=.001;
    p.rotation_radius=.01;p.contact.d_hat=.2;p.contact.k_barrier=20.;p.contact.friction=.15;
    p.contact.eps_v=.001;p.contact.eps=1e-12;p.exact_computation_fallback=exact;
    p.global_ccd_guess=false;p.colored_guess_sweeps=0;return p;
}
struct Run {std::vector<StateSnapshot> states;std::vector<std::size_t> resident;};
Run run(const char* flag,const char* pairs,bool exact,bool contact=true) {
    Environment feature("IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION"),queues("IPC_GPU_COLOR_PAIR_QUEUES"),cap("IPC_GPU_CONTACT_PACKET_CAPACITY");
    feature.set(flag);queues.set(pairs);cap.set("3");const auto h=fixture();auto p=parameters(exact);
    require(two_rigid_color_queue::eligible_validated_scene(h),"valid fixture not eligible");
    const auto before=resident_bytes();Run result;
    {
        GpuSolver solver(h);
        // Same owner: above-cap four-PT, below-cap one-PT, above-cap four-PT.
        // The geometry gap separates the .1 point from the three .15 points.
        for(double activation:{.2,.11,.2}) {
            p.contact.d_hat=activation;if(!contact){p.contact.k_barrier=0.;p.contact.friction=0.;}
            solver.advance_substep(p,h.pin_targets,{});auto state=solver.snapshot();finite(state);
            result.states.push_back(std::move(state));result.resident.push_back(resident_bytes()-before);
        }
    }
    require(resident_bytes()==before,"solver destruction retained owned allocation");
    require(!tracking_failed,"allocation ledger lost an event");return result;
}
void pairs_and_solver_gates() {
    Environment independent_dense("IPC_GPU_TWO_RIGID_DENSE_CSR");independent_dense.set("0");
    for(bool exact:{false,true}) {
        const auto off=run("0",nullptr,exact),unset=run(nullptr,nullptr,exact),on=run("1",nullptr,exact);
        const auto forced=run("1","1",exact),disabled=run("1","0",exact);
        for(std::size_t epoch=0;epoch<off.states.size();++epoch) {
            const auto label="exact="+std::to_string(exact)+" epoch="+std::to_string(epoch);
            equal(off.states[epoch],unset.states[epoch],label+" defaultON");
            equal(off.states[epoch],on.states[epoch],label+" candidate");
            equal(off.states[epoch],forced.states[epoch],label+" explicitPAIR1");
            equal(off.states[epoch],disabled.states[epoch],label+" explicitPAIR0");
            require(off.resident[epoch]==forced.resident[epoch],label+" PAIR1 failed to retain old storage");
            require(unset.resident[epoch]==on.resident[epoch],label+" unset queue flag did not select defaultON storage");
        }
        // At N=4, both color rows are4>cap3. Exactly one four-N int color
        // array and three size_t offsets remain in OFF; ON releases them.
        const std::size_t large_difference=4*4*sizeof(int)+3*sizeof(std::size_t);
        const auto storage_delta=[&](std::size_t epoch,std::size_t expected,const char* label) {
            require(off.resident[epoch]==on.resident[epoch]+expected,
                std::string(label)+" OFF="+std::to_string(off.resident[epoch])
                +" ON="+std::to_string(on.resident[epoch])+" expected_difference="+std::to_string(expected));
        };
        storage_delta(0,large_difference,"first large trusted omission");
        // The nonempty small epoch has one PT. OFF keeps16 large color-pair
        // slots while ON creates4 small slots; both retain3 offsets and the
        // same small sort/incident arrays: (16-4)*sizeof(int)==48 bytes.
        storage_delta(1,(16-4)*sizeof(int),"one-PT small queue construction");
        storage_delta(2,large_difference,"large return releases small color buffers");
        require(on.resident[0]==disabled.resident[0]&&on.resident[2]==disabled.resident[2],"trusted large route differs from existing PAIR0 storage");
        const auto no_contact=run("0",nullptr,exact,false);
        require(!equal_bytes(off.states[0].com,no_contact.states[0].com)
            ||!equal_bytes(off.states[0].omega,no_contact.states[0].omega),"active contact force witness is vacuous");
    }
}
std::string force_error(const char* flag,bool exact) {
    Environment feature("IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION"),queues("IPC_GPU_COLOR_PAIR_QUEUES"),cap("IPC_GPU_CONTACT_PACKET_CAPACITY");
    feature.set(flag);queues.set(nullptr);cap.set("3");const auto h=fixture(0.);const auto p=parameters(exact);
    const auto before=resident_bytes();std::string message;
    {
        GpuSolver solver(h);const auto initial=solver.snapshot();finite(initial);
        try{solver.advance_substep(p,h.pin_targets,{});}catch(const std::runtime_error& e){message=e.what();}
        require(message.find("device status 21;")!=std::string::npos,"single zero-gap force did not retain code21");
        bool snapshot_rejected=false,retry_rejected=false;
        try{(void)solver.snapshot();}catch(const std::runtime_error&){snapshot_rejected=true;}
        try{solver.advance_substep(p,h.pin_targets,{});}catch(const std::runtime_error&){retry_rejected=true;}
        require(snapshot_rejected&&retry_rejected,"failed force left owner reusable/publishable");
    }
    require(resident_bytes()==before,"force-error owner leaked an allocation");return message;
}
void raw_pipeline_gates() {
    Environment feature("IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION"),queues("IPC_GPU_COLOR_PAIR_QUEUES");
    feature.set("1");queues.set(nullptr);const auto h=fixture();
    DeviceArray<int> owners(h.node_blocks);std::vector<BpContact> records;
    for(int node=0;node<4;++node)records.push_back(BpContact{{node,4,5,6},0});
    DeviceArray<BpContact> contacts(records);DeviceArray<int> colors(std::vector<int>{0,1});
    DeviceArray<int> color_blocks(std::vector<int>{0,1}),color_offsets(std::vector<int>{0,1,2});
    BroadPhaseView view;view.contacts=contacts.data();view.block_colors=colors.data();view.color_count=2;
    view.color_blocks=color_blocks.data();view.color_offsets=color_offsets.data();
    GpuContactPipeline raw(3);
    for(std::size_t count:{4u,1u,0u,4u}) {
        view.contact_count=count;raw.rebuild(view,2,owners.data());const auto incidence=raw.incidence();
        require(incidence.color_pairs.offsets!=nullptr,"raw pipeline incorrectly accepted private hint");
        std::size_t offsets[3]{};
        cuda_checked(cudaMemcpy(offsets,incidence.color_pairs.offsets,sizeof(offsets),cudaMemcpyDeviceToHost));
        require(offsets[0]==0&&offsets[1]==count&&offsets[2]==2*count,"raw worklist lost exact full rows");
        require(incidence.color_pairs.max_color_pairs==count,"raw max row metadata stale");
    }
    auto bad=h;bad.node_blocks[0]=1;bool rejected=false;
    try{GpuSolver invalid(bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"invalid owning-block scene was hinted instead of rejected");
}
} // namespace

extern "C" cudaError_t __real_cudaMalloc(void**,std::size_t);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer,std::size_t bytes) {
    const auto status=__real_cudaMalloc(pointer,bytes);
    if(tracking&&status==cudaSuccess&&bytes)try{
        std::lock_guard<std::mutex> lock(allocation_mutex);allocations[*pointer]=bytes;events.push_back({true,bytes});
    }catch(...){tracking_failed=true;}
    return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
    const auto status=__real_cudaFree(pointer);
    if(tracking&&status==cudaSuccess&&pointer)try{
        std::lock_guard<std::mutex> lock(allocation_mutex);const auto found=allocations.find(pointer);
        if(found!=allocations.end()){events.push_back({false,found->second});allocations.erase(found);}
    }catch(...){tracking_failed=true;}
    return status;
}
int main() {
    try {
        tracking=true;pairs_and_solver_gates();raw_pipeline_gates();
        for(bool exact:{false,true})require(force_error("0",exact)==force_error("1",exact),"force error diagnostic changed");
        require(resident_bytes()==0&&!tracking_failed,"owned allocation ledger not empty");tracking=false;
        std::cout<<"Two-rigid queue elision: Original/Robust six-array parity, forced/default/raw rollback, active forces, same-owner large-small-large exact storage PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"two-rigid queue test: "<<e.what()<<'\n';return 1;}
}

// Standalone CUDA equivalence test. No production pipeline/link dependency.
// nvcc -std=c++17 --fmad=false -lineinfo -I gpu \
//   gpu/resident_contact_warp_reduce_test.cu -o /tmp/contact_warp_test
#include "resident_scene.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::resident;
namespace {
struct Incident {int block,contact;};
struct Work {
    ContactOutput* outputs;
    unsigned char* active;
    const Incident* incidents;
    const std::size_t* offsets;
    const unsigned char* surface_query;
};
__device__ void error(int* destination,int value){atomicCAS(destination,0,value);}
__device__ int code(ContactStatus value) {
    if(value==ContactStatus::InvalidInput)return 20;
    if(value==ContactStatus::ZeroDistance)return 21;
    if(value==ContactStatus::ExactCapacityExceeded)return 23;
    return 22;
}
#include "resident_contact_warp_reduce.cuh"

// Frozen original reducer: keep the loop nesting and expression ordering
// independent of the lane helper so this is not a self-comparison.
__global__ void serial_reference(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,int color) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    Derivatives system=elastic.systems[node];
    const int block=scene.node_blocks[node];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
        const int pair=work.incidents[incident].contact;
        if(!work.active[pair])continue;
        const auto& out=work.outputs[pair];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(!out.active)continue;
        for(int role=0;role<4;++role)if(broad.contacts[pair].vertices[role]==int(node)) {
            const double weight=out.weights[role];
            for(int k=0;k<3;++k) {
                system.gradient[k]+=normal_scale*out.barrier_gradient[role][k];
                system.gradient[k]+=weight*out.friction_relative_gradient[k];
                for(int l=0;l<3;++l) {
                    system.hessian[3*k+l]+=normal_scale*out.barrier_hessian[role][role][3*k+l];
                    system.hessian[3*k+l]+=(weight*out.friction_relative_hessian[3*k+l])*weight;
                }
            }
        }
    }
    if(!scene.solid[node]||!work.surface_query||work.surface_query[node]) {
        const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,state.positions[node],state.previous[node],sdf.parameters);
        if(out.status!=SdfStatus::Ok){error(elastic.error,25);return;}
        const double dt2=params.dt*params.dt;
        for(int k=0;k<3;++k) {
            system.gradient[k]+=dt2*out.gradient[k];system.gradient[k]+=out.friction_gradient[k];
            for(int l=0;l<3;++l){system.hessian[3*k+l]+=dt2*out.hessian[3*k+l];system.hessian[3*k+l]+=out.friction_hessian[3*k+l];}
        }
    }
    bool finite=true;
    for(int k=0;k<3;++k)finite=finite&&contact_detail::finite(system.gradient[k]);
    for(int k=0;k<9;++k)finite=finite&&contact_detail::finite(system.hessian[k]);
    if(!finite){error(elastic.error,26);return;}
    elastic.systems[node]=system;
}

void checked(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Device {
    T* data=nullptr;std::size_t size=0;
    explicit Device(const std::vector<T>& input):size(input.size()) {
        if(size){checked(cudaMalloc(reinterpret_cast<void**>(&data),size*sizeof(T)));checked(cudaMemcpy(data,input.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    }
    ~Device(){cudaFree(data);}
    std::vector<T> read()const {
        std::vector<T> result(size);if(size)checked(cudaMemcpy(result.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return result;
    }
};
struct Position {double v[3];};
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
void run_fixture(int mode) {
    constexpr int deformable_count=19,nodes=21,blocks=20;
    const bool friction=mode==1||mode==2;
    const bool has_sdf=mode==1||mode==2||mode==6;
    std::vector<int> node_blocks(nodes),colors(blocks),mapping(blocks,-1),color_blocks,color_offsets;
    std::vector<unsigned char> deformable(nodes),solid(nodes),surface(nodes);
    std::vector<Position> positions(nodes),previous(nodes);
    std::vector<Derivatives> initial(nodes);
    for(int node=0;node<nodes;++node) {
        node_blocks[node]=node<deformable_count?node:blocks-1;
        deformable[node]=node<deformable_count;solid[node]=node%3==0;surface[node]=node%2;
        positions[node]={{.03*node,-.04+.004*node,.11}};
        previous[node]={{positions[node].v[0]-.002,positions[node].v[1]-.001,.109}};
        for(int k=0;k<3;++k)initial[node].gradient[k]=(node+1)*.111/(k+1);
        for(int k=0;k<9;++k)initial[node].hessian[k]=(node+1)*.217/(k+1);
        if(node<deformable_count)mapping[node]=node;
    }
    for(int block=0;block<blocks;++block)colors[block]=block%3;
    for(int color=0;color<3;++color) {
        color_offsets.push_back(int(color_blocks.size()));
        for(int block=0;block<blocks;++block)if(colors[block]==color)color_blocks.push_back(block);
    }
    color_offsets.push_back(int(color_blocks.size()));
    std::vector<BpContact> contacts;
    std::vector<ContactOutput> outputs;
    std::vector<unsigned char> active;
    std::vector<Incident> incidents;
    std::vector<std::size_t> offsets;
    for(int block=0;block<blocks;++block) {
        offsets.push_back(incidents.size());
        if(block==blocks-1)continue;
        const int count=block==0?273:1+block*7;
        for(int j=0;j<count;++j) {
            const int pair=int(contacts.size());
            BpContact contact{{19,20,19,20},j%2};contact.vertices[j%4]=block;
            if(j%11==0)contact.vertices[(j+1)%4]=block; // Exercise repeated-role order.
            ContactOutput output{};output.active=(j%7!=3);
            for(int role=0;role<4;++role) {
                output.weights[role]=(role%2?-1.:1.)*.231*(role+1);
                for(int k=0;k<3;++k)output.barrier_gradient[role][k]=(j%2?-1.:1.)*.123456789*(pair+1)/(k+role+1);
                for(int k=0;k<9;++k)output.barrier_hessian[role][role][k]=(j%3?-1.:1.)*.987654321*(pair+1)/(k+role+1);
            }
            for(int k=0;k<3;++k)output.friction_relative_gradient[k]=friction?.123*(j+1)/(k+1):0.;
            for(int k=0;k<9;++k)output.friction_relative_hessian[k]=friction?.321*(j+1)/(k+1):0.;
            contacts.push_back(contact);outputs.push_back(output);active.push_back(j%13!=5);incidents.push_back({block,pair});
        }
    }
    offsets.push_back(incidents.size());
    if(mode==4)outputs[0].status=ContactStatus::ZeroDistance;
    if(mode==5)outputs[0].barrier_gradient[0][0]=std::numeric_limits<double>::infinity();
    std::vector<SdfCollider> colliders(2);
    colliders[0].kind=SdfKind::Plane;
    colliders[1].kind=SdfKind::Sphere;colliders[1].point[1]=-.45;colliders[1].radius=.1;
    if(mode==2) { // Select the curved collider, not just its geometry branch.
        colliders[0].point[1]=-1.;colliders[1].point[1]=-.1;colliders[1].radius=.2;
    }
    SdfParameters sdf_parameters;sdf_parameters.dt=.031;sdf_parameters.epsilon=.2;
    sdf_parameters.stiffness=170;sdf_parameters.friction=friction?.2:0.;sdf_parameters.eps_v=.1;
    if(mode==6)sdf_parameters.dt=0.;
    Device<int> d_blocks(node_blocks),d_colors(colors),d_mapping(mapping),d_color_blocks(color_blocks),d_color_offsets(color_offsets);
    Device<unsigned char> d_deformable(deformable),d_solid(solid),d_surface(surface),d_active(active);
    Device<Position> d_positions(positions),d_previous(previous);
    Device<BpContact> d_contacts(contacts);
    Device<ContactOutput> d_outputs(outputs);
    Device<Incident> d_incidents(incidents);
    Device<std::size_t> d_offsets(offsets);
    Device<SdfCollider> d_colliders(colliders);
    Device<Derivatives> d_serial(initial),d_warp(initial);
    Device<int> d_error_serial(std::vector<int>{mode==3?123:0}),d_error_warp(std::vector<int>{mode==3?123:0});
    BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();
    broad.block_colors=d_colors.data;broad.color_blocks=d_color_blocks.data;broad.color_offsets=d_color_offsets.data;broad.color_count=3;
    DeformableScene scene{};scene.nodes=nodes;scene.node_blocks=d_blocks.data;scene.deformable=d_deformable.data;scene.solid=d_solid.data;
    DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(d_positions.data);state.previous=reinterpret_cast<double(*)[3]>(d_previous.data);
    ElasticWorkspace serial{},warp{};serial.systems=d_serial.data;serial.error=d_error_serial.data;warp.systems=d_warp.data;warp.error=d_error_warp.data;
    Work work{d_outputs.data,d_active.data,d_incidents.data,d_offsets.data,mode==2?nullptr:d_surface.data};
    SdfScene sdf{d_colliders.data,has_sdf?2:0,sdf_parameters};
    ContactParameters params{.005,17.,.031,friction?.2:0.,.1,1e-12};
    for(int color=0;color<3;++color) {
        serial_reference<<<1,128>>>(broad,scene,state,serial,sdf,params,work,color);
        // Two warps force grid-stride reuse of shared SDF storage.
        reduce_deformable_warp<<<2,32>>>(broad,scene,state,warp,sdf,params,work,color,d_mapping.data);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
    }
    const auto expected=d_serial.read(),actual=d_warp.read();
    const int expected_error=d_error_serial.read()[0],actual_error=d_error_warp.read()[0];
    const int required_error=mode==3?123:mode==4?21:mode==5?26:mode==6?25:0;
    require(expected_error==required_error&&actual_error==required_error,"error mismatch for fixture "+std::to_string(mode));
    if(required_error==0||mode==3)
        require(std::memcmp(expected.data(),actual.data(),nodes*sizeof(Derivatives))==0,"nonidentical derivative bits for fixture "+std::to_string(mode));
    if(mode==4||mode==5)
        require(std::memcmp(&actual[0],&initial[0],sizeof(Derivatives))==0,"failed node was partially committed");
    if(mode==6) // Node1 is cloth and therefore must fail invalid SDF validation.
        require(std::memcmp(&actual[1],&initial[1],sizeof(Derivatives))==0,"invalid SDF node was partially committed");
    for(int node=deformable_count;node<nodes;++node)
        require(std::memcmp(&actual[node],&initial[node],sizeof(Derivatives))==0,"rigid node was changed");
}
} // namespace

int main() {
    try {
        for(int mode=0;mode<7;++mode)run_fixture(mode);
        std::cout<<"Warp contact reduction: seven scalar/GPU byte-equivalence and fail-closed fixtures PASS\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}

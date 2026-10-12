#include "resident_contact_pipeline_test.h"
#include "resident_contact_pipeline.h"
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
__global__ void shift_rigids(ipc_gpu::resident::DeformableState state,const int* body,const double* shift) {
    int v=threadIdx.x;if(v<8&&body[v]>=0)for(int k=0;k<3;++k)state.positions[v][k]+=shift[3*body[v]+k];
}
}
void resident_contact_pipeline_test_gpu(const ResidentPipelineTestInput& input,ResidentPipelineTestOutput& output) {
    using namespace ipc_gpu;using namespace ipc_gpu::resident;
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
    ResidentContactPipeline pipeline;
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

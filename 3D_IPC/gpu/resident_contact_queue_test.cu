// Production-pipeline regression for parallel exact queues and ordered joins.
#include "resident_contact_pipeline.h"
#include "resident_contact_exact.cuh"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::resident;
namespace {
constexpr int per_color=1031, colors_count=3, pair_count=per_color*colors_count;
constexpr int node_count=4*pair_count;
static_assert(sizeof(Derivatives)==12*sizeof(double),"bytewise derivative comparison excludes padding");
void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool okay,const char* message) {if(!okay)throw std::runtime_error(message);}
template<class T>struct Device {
    T* data=nullptr;std::size_t count;
    explicit Device(std::size_t n):count(n) {
        if(n)check(cudaMalloc(&data,n*sizeof(T)));
    }
    explicit Device(const std::vector<T>& values):Device(values.size()){upload(values);}
    ~Device(){cudaFree(data);}
    void upload(const std::vector<T>& values) {
        require(values.size()==count,"test upload size");
        if(count)check(cudaMemcpy(data,values.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> download()const {
        std::vector<T> result(count);
        if(count)check(cudaMemcpy(result.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));
        return result;
    }
};
ContactInput fixture(int kind) {
    ContactInput in;in.d_hat=.8;in.k_barrier=370;in.dt=.031;in.friction=.2;in.eps_v=.1;
    in.hessian_blocks=1u; // Only the selected vertex's H00 is consumed.
    in.kind=kind?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    const double x[2][4][3]={{{0,0,0},{1,0,0},{.25,.125,0},{.75,.125,0}},
        {{.25,.25,std::ldexp(1.0,-500)},{0,0,0},{1,0,0},{0,1,0}}};
    for(int i=0;i<4;++i)for(int a=0;a<3;++a) {
        in.positions[i][a]=x[kind][i][a];
        // Tiny PT uses no slip to avoid making a huge physical friction force
        // obscure the allocator/bit-capacity regression being tested.
        in.previous[i][a]=x[kind][i][a]-(kind?0:.001*(i+1)*(a+1));
    }
    return in;
}
__global__ void serial_reference(const ContactInput* inputs,ContactOutput* outputs,
                                std::uint32_t* words,int* slots,int* diagnostics) {
    resident_exact::DevicePool pool;
    resident_exact::initialize_pool(pool,words,slots,1600,512);
    for(int kind=0;kind<2;++kind) {
        if(evaluate_contact(inputs[kind]).status!=ContactStatus::NeedsExactGeometry)diagnostics[0]=1;
        auto out=evaluate_contact_exact<256>(inputs[kind]);
        if(out.status==ContactStatus::ExactCapacityExceeded&&!pool.failed) {
            ++diagnostics[1];out=evaluate_contact_exact<1600>(inputs[kind]);
        }
        if(out.status!=ContactStatus::Ok||!out.active||pool.failed||pool.available!=512)diagnostics[0]=2;
        outputs[kind]=out;
    }
    resident_exact::release_pool();
}
__device__ Derivatives seed(int node) {
    Derivatives out{};
    for(int i=0;i<3;++i) {
        out.gradient[i]=.0625*(node+1)*(i+1);
        for(int j=0;j<3;++j)out.hessian[3*i+j]=.125*(i+1)*(j+1);
    }
    return out;
}
__global__ void reset_systems(Derivatives* systems) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node<node_count)systems[node]=seed(node);
}
__global__ void expected_systems(Derivatives* expected,const ContactOutput* reference,
                                 unsigned visited_colors,bool contacts_active,ContactParameters params) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=node_count)return;
    Derivatives out=seed(node);
    const int pair=node/4,color=pair/per_color,local=pair%per_color;
    if(node%4==0&&(visited_colors&(1u<<color))&&contacts_active) {
        const auto& contact=reference[local==0?1:0];
        const double normal_scale=params.dt*params.dt*params.k_barrier;
        // The first two owners have three identical candidate records. The
        // production CSR orders by kind then pair ID; additions stay serial.
        for(int incident=0;incident<(local<2?3:1);++incident) {
            const double weight=contact.weights[0];
            for(int k=0;k<3;++k) {
                out.gradient[k]+=normal_scale*contact.barrier_gradient[0][k];
                out.gradient[k]+=weight*contact.friction_relative_gradient[k];
                for(int l=0;l<3;++l) {
                    out.hessian[3*k+l]+=normal_scale*contact.barrier_hessian[0][0][3*k+l];
                    out.hessian[3*k+l]+=(weight*contact.friction_relative_hessian[3*k+l])*weight;
                }
            }
        }
    }
    expected[node]=out;
}
void compare(const Device<Derivatives>& actual,Device<Derivatives>& expected,
             const Device<ContactOutput>& reference,unsigned visited,bool active,ContactParameters params) {
    expected_systems<<<(node_count+127)/128,128>>>(expected.data,reference.data,visited,active,params);
    check(cudaGetLastError());
    const auto a=actual.download(),b=expected.download();
    for(int node=0;node<node_count;++node)
        if(std::memcmp(&a[node],&b[node],sizeof(Derivatives)))
            throw std::runtime_error("parallel exact queue differs from serial ordered reference at node "+std::to_string(node));
}
}

int main() {
    try {
        const std::vector<ContactInput> fixtures{fixture(0),fixture(1)};
        Device<ContactInput> ref_input(fixtures);Device<ContactOutput> ref_output(2);
        Device<std::uint32_t> ref_words(1600*512);Device<int> ref_slots(512),diagnostics(std::vector<int>(2,0));
        serial_reference<<<1,1>>>(ref_input.data,ref_output.data,ref_words.data,ref_slots.data,diagnostics.data);
        check(cudaGetLastError());
        const auto diagnostic=diagnostics.download();
        require(diagnostic[0]==0&&diagnostic[1]==1,"serial fixtures must exercise active exact geometry and exactly one full-width retry");

        using V=std::array<double,3>;
        std::vector<V> positions(node_count),previous(node_count),far_positions(node_count);
        std::vector<int> blocks(node_count),colors(node_count,colors_count);
        std::vector<unsigned char> deformable(node_count,0),solid(node_count,0);
        std::vector<BpContact> contacts;
        for(int pair=0;pair<pair_count;++pair) {
            const int variant=pair%per_color==0?1:0;
            const auto& in=fixtures[variant];
            for(int role=0;role<4;++role) {
                const int node=4*pair+role;blocks[node]=node;
                for(int axis=0;axis<3;++axis) {
                    positions[node][axis]=in.positions[role][axis];
                    previous[node][axis]=in.previous[role][axis];
                }
            }
            deformable[4*pair]=1;colors[4*pair]=pair/per_color;
            contacts.push_back({{4*pair,4*pair+1,4*pair+2,4*pair+3},variant?0:1});
        }
        // Repeated records intentionally stress ordered per-node reduction;
        // every independent owner still has only one selected deformable node.
        for(int color=0;color<colors_count;++color)for(int local=0;local<2;++local)
            for(int duplicate=0;duplicate<2;++duplicate)contacts.push_back(contacts[color*per_color+local]);
        far_positions=positions;
        for(int pair=0;pair<pair_count;++pair) {
            if(pair%per_color==0)far_positions[4*pair][2]=2;
            else {far_positions[4*pair+2][1]=2;far_positions[4*pair+3][1]=2;}
        }
        Device<V> x(positions),old(previous);Device<int> d_blocks(blocks),d_colors(colors),error(std::vector<int>{0});
        Device<unsigned char> d_deformable(deformable),d_solid(solid);Device<BpContact> d_contacts(contacts);
        Device<Derivatives> systems(node_count),expected(node_count);
        DeformableScene scene{};scene.nodes=node_count;scene.node_blocks=d_blocks.data;
        scene.deformable=d_deformable.data;scene.solid=d_solid.data;
        DeformableState state{};state.positions=reinterpret_cast<double(*)[3]>(x.data);
        state.previous=reinterpret_cast<double(*)[3]>(old.data);
        ElasticWorkspace work{};work.systems=systems.data;work.error=error.data;
        BroadPhaseView broad{};broad.contacts=d_contacts.data;broad.contact_count=contacts.size();broad.block_colors=d_colors.data;
        const ContactParameters params{.8,370,.031,.2,.1,1e-12};
        const RigidScene rigid{};const RigidState rigid_state{};const RigidWorkspace rigid_work{};const SdfScene sdf{};
        ResidentContactPipeline pipeline;pipeline.rebuild(broad,node_count,d_blocks.data);
        auto sweep=[&](bool active) {
            reset_systems<<<(node_count+127)/128,128>>>(systems.data);check(cudaGetLastError());
            error.upload({0});
            for(int color=0;color<colors_count;++color) {
                pipeline.assemble_color(scene,state,work,rigid,rigid_state,rigid_work,sdf,params,color,ContactAssemblyPhase::Deformable);
                require(error.download()[0]==0,"production exact queue failed");
                compare(systems,expected,ref_output,(1u<<(color+1))-1,active,params);
            }
        };
        sweep(true); // >1024 exact queries per color, multiple worker blocks.
        x.upload(far_positions);sweep(false); // Empty exact queue, stale outputs ignored.
        x.upload(positions);sweep(true); // Reuse all arenas and output slots.
        reset_systems<<<(node_count+127)/128,128>>>(systems.data);error.upload({123});
        pipeline.assemble_color(scene,state,work,rigid,rigid_state,rigid_work,sdf,params,0,ContactAssemblyPhase::Deformable);
        require(error.download()[0]==123,"preexisting queue error overwritten");
        compare(systems,expected,ref_output,0,false,params);
        // Rebuild empty then populated, retaining old capacity but no old work.
        error.upload({0});broad.contact_count=0;pipeline.rebuild(broad,node_count,d_blocks.data);sweep(false);
        broad.contact_count=contacts.size();pipeline.rebuild(broad,node_count,d_blocks.data);sweep(true);
        std::cout<<"Resident exact queue PASS: "<<contacts.size()<<" mixed records, >1024 exact contacts/color, "
                   "256-to-1600 retry, byte-identical ordered derivatives, empty/reused queues and error guards\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

#include "contact_pipeline.h"
#include "profile.h"
#include "contact_schedule.h"
#include "contact_color_launch.h"
#include "contact_exact.cuh"
#include "contact_cooperative.h"
#include "contact_compact_point.h"
#include "contact_selected_ordinary.cuh"
#include "cooperative_compact_point_study.h"
#include "contact_execution.h"
#include "two_rigid_color_queue_policy.h"
#include "two_rigid_dense_csr_policy.h"
#include "contact_input_publication.cuh"
#include "original_contact.cuh"
#include "original_guess.cuh"
#include "box_certificate.cuh"
#include "rigid.cuh"
#include "rigid_pins.cuh"
#include "rigid_compact_index.cuh"
#include "rigid_one_slot_policy.h"
#include "original_sparse_gather_policy.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>

namespace ipc_gpu::solver {
namespace {
constexpr int threads=128;
constexpr std::size_t rigid_compact_body_limit=64;
static_assert(sizeof(unsigned)==sizeof(std::uint32_t),"rigid compact count ABI");
// Independent bounded arenas, spread over many SMs rather than one serial
// device worker. The larger-capacity retry is a second ordered kernel so its
// scratch can reuse the first pass's allocation (800 MiB default maximum).
constexpr int exact_slots=512;
constexpr int exact_small_limbs=256;
constexpr int exact_large_limbs=1600;
void checked(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string("GPU contact ")+operation+": "+cudaGetErrorString(error));
}
unsigned blocks(std::size_t count){return static_cast<unsigned>((count+threads-1)/threads);}
int selected_reduce_threads() {
    const int value=ExactContactSchedule::environment_integer("IPC_GPU_CONTACT_SELECTED_NODE_THREADS",8,128);
    if((value&(value-1))!=0)
        throw std::invalid_argument("IPC_GPU_CONTACT_SELECTED_NODE_THREADS must be one of 1,2,4,8,16,32,64,128");
    return value;
}
int component_reduce_groups() {
    const int groups=ExactContactSchedule::environment_integer("IPC_GPU_CONTACT_COMPONENT_GROUPS",12,12);
    if(groups!=1&&groups!=2&&groups!=3&&groups!=4&&groups!=6&&groups!=12)
        throw std::invalid_argument("IPC_GPU_CONTACT_COMPONENT_GROUPS must be one of 1,2,3,4,6,12");
    return groups;
}
template<class T> class Buffer {
public:
    T* data=nullptr;std::size_t capacity=0;
    ~Buffer(){cudaFree(data);}
    void release() {
        if(data)checked(cudaFree(data),"release buffer");data=nullptr;capacity=0;
    }
    void reserve(std::size_t count) {
        if(count<=capacity)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("GPU contact allocation overflow");
        T* replacement=nullptr;checked(cudaMalloc(&replacement,count*sizeof(T)),"allocate");
        checked(cudaFree(data),"release old buffer");data=replacement;capacity=count;
    }
};
struct Incident {int block,contact;};
struct IncidentOrder {
    const BpContact* contacts;
    __host__ __device__ bool operator()(const Incident& a,const Incident& b)const {
        if(a.block!=b.block)return a.block<b.block;
        if(a.block==INT_MAX)return a.contact<b.contact;
        const int ka=contacts[a.contact].kind,kb=contacts[b.contact].kind;
        return ka!=kb?ka<kb:a.contact<b.contact;
    }
};
struct NodePullback {
    double jacobian[9];
    double curvature[3][9];
};
// Per-packet, per-distinct-owner terms. Keep barrier unscaled and friction
// separate so the serial CSR consumer retains both original additions.
struct RigidContactTerms { Derivatives barrier,friction; };
static_assert(sizeof(RigidContactTerms)==24*sizeof(double));
struct Work {
    ContactInput* inputs;
    ContactOutput* outputs;
    CompactDeformableContact* compact;
    unsigned char* active;
    unsigned char* output_activity;
    int* exact_queue;
    unsigned* exact_count;
    int* exact_retry_queue;
    unsigned* exact_retry_count;
    unsigned* exact_next;
    std::uint32_t* exact_words;
    int* exact_free_slots;
    const Incident* incidents;
    const int* pair_indices;
    const std::size_t* offsets;
    const int* vertex_block;
    const unsigned char* surface_query;
    const NodePullback* node_pullbacks;
    RigidContactTerms* rigid_terms;
    const int* record_slots;
    bool pending_contacts;
    bool one_slot_terms;
    __host__ __device__ int record_slot(int pair)const {
        return record_slots?record_slots[pair]:pair;
    }
    __host__ __device__ int incident_pair(std::size_t incident)const {
        return incidents?incidents[incident].contact:pair_indices[incident];
    }
};
#include "contact_record_index.cuh"
__device__ void error(int* destination,int code){atomicCAS(destination,0,code);}
__device__ int code(ContactStatus status) {
    if(status==ContactStatus::InvalidInput)return 20;
    if(status==ContactStatus::ZeroDistance)return 21;
    if(status==ContactStatus::ExactCapacityExceeded)return 23;
    return 22;
}
__global__ void certify_boxes(BroadPhaseView broad,const int* vertex_block,ContactBoxPruning input,unsigned char* separated) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    separated[pair]=0;
    if(broad.contacts[pair].kind!=0&&broad.contacts[pair].kind!=1)return;
    BpAabb boxes[4];double positions[4][3];
    for(int role=0;role<4;++role) {
        const int node=broad.contacts[pair].vertices[role];
        if(!input.deformable[node])return;
        // Retain malformed colorings so the original CCD unsupported-motion
        // guard still sees them; a proof must not hide an input error.
        for(int prior=0;prior<role;++prior) {
            const int other=broad.contacts[pair].vertices[prior];
            if(other!=node&&broad.block_colors[vertex_block[other]]==broad.block_colors[vertex_block[node]])return;
        }
        boxes[role]=input.boxes[node];
        for(int axis=0;axis<3;++axis) {
            positions[role][axis]=input.positions[node][axis];
            // Keep the original input-error contract even with zero friction.
            if(!contact_detail::finite(input.previous[node][axis]))return;
        }
    }
    separated[pair]=certify_movement_box_contact(boxes,positions,broad.contacts[pair].kind==0,input.parameters.d_hat);
}
// Original CPU cache policy: complete four-box projection from the original
// geometry, never a current-point plane shortcut. Keep input/error visibility.
__global__ void certify_original_boxes(BroadPhaseView broad,const int* vertex_block,
    ContactBoxPruning input,unsigned char* separated) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    separated[pair]=0;
    const auto& contact=broad.contacts[pair];
    if(contact.kind!=0&&contact.kind!=1)return;
    BpAabb boxes[4];double positions[4][3];
    for(int role=0;role<4;++role) {
        const int node=contact.vertices[role];
        if(!input.deformable[node])return;
        for(int prior=0;prior<role;++prior) {
            const int other=contact.vertices[prior];
            if(other!=node&&broad.block_colors[vertex_block[other]]
                ==broad.block_colors[vertex_block[node]])return;
        }
        boxes[role]=input.boxes[node];
        for(int axis=0;axis<3;++axis) {
            positions[role][axis]=input.positions[node][axis];
            if(!contact_detail::finite(input.previous[node][axis]))return;
        }
    }
    separated[pair]=original_guess::boxes_separated_original(
        positions,boxes,contact.kind!=0,input.parameters.d_hat);
}
__global__ void count_raw_incidents(BroadPhaseView broad,const int* vertex_block,
    int block_count,std::size_t* counts) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    static_assert(sizeof(std::size_t)==sizeof(unsigned long long),"64-bit incidence counters required");
    for(int role=0;role<4;++role) {
        const int block=vertex_block[broad.contacts[pair].vertices[role]];
        bool duplicate=false;
        for(int prior=0;prior<role;++prior)
            duplicate=duplicate||block==vertex_block[broad.contacts[pair].vertices[prior]];
        if(!duplicate&&block>=0&&block<block_count)
            atomicAdd(reinterpret_cast<unsigned long long*>(counts+block),1ULL);
    }
}
__global__ void emit_incidents(BroadPhaseView broad,const int* vertex_block,Incident* incidents,const unsigned char* separated) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    for(int role=0;role<4;++role) {
        const int block=vertex_block[broad.contacts[pair].vertices[role]];
        bool duplicate=false;
        for(int prior=0;prior<role;++prior)
            duplicate=duplicate||block==vertex_block[broad.contacts[pair].vertices[prior]];
        incidents[4*pair+role]={duplicate||(separated&&separated[pair])?INT_MAX:block,static_cast<int>(pair)};
    }
}
__global__ void incident_offsets(const Incident* incidents,std::size_t count,int block_count,std::size_t* offsets) {
    const int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>block_count)return;
    std::size_t begin=0,end=count;
    while(begin<end){const auto mid=begin+(end-begin)/2;if(incidents[mid].block<block)begin=mid+1;else end=mid;}
    offsets[block]=begin;
}
__global__ void incident_indices(const Incident* incidents,std::size_t count,int* pairs) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)pairs[i]=incidents[i].contact;
}
template<bool KindPackets>
__global__ void inspect_two_rigid_dense_contacts(BroadPhaseView broad,
    const int* vertex_block,std::size_t nodes,unsigned* unsupported) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    const int previous=pair?broad.contacts[pair-1].kind:-1;
    const auto reason=two_rigid_dense_csr::inspect(broad.contacts[pair],previous,
        vertex_block,nodes);
    if(reason)atomicOr(unsupported,reason);
    if constexpr(KindPackets) {
        // One transition in a certified kind-ordered array. Atomic publication
        // also keeps arbitrary unsupported layouts race-free before fallback.
        // N<=INT_MAX is part of the existing dense eligibility gate.
        const int kind=broad.contacts[pair].kind;
        if(kind==1&&previous==0)
            atomicMax(unsupported+1,static_cast<unsigned>(pair));
        else if(kind==0&&pair+1==broad.contact_count)
            atomicMax(unsupported+1,static_cast<unsigned>(pair+1));
    }
}
__global__ void emit_two_rigid_dense_rows(std::size_t count,int* pairs,
    std::size_t* offsets) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair==0){offsets[0]=0;offsets[1]=count;offsets[2]=2*count;}
    if(pair<count){pairs[pair]=static_cast<int>(pair);pairs[count+pair]=static_cast<int>(pair);}
}
__global__ void emit_color_pairs(BroadPhaseView broad,const int* vertex_block,std::uint64_t* keys,const unsigned char* separated) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=broad.contact_count)return;
    for(int role=0;role<4;++role) {
        const int color=broad.block_colors[vertex_block[broad.contacts[pair].vertices[role]]];
        bool duplicate=false;
        for(int prior=0;prior<role;++prior)
            duplicate=duplicate||color==broad.block_colors[vertex_block[broad.contacts[pair].vertices[prior]]];
        keys[4*pair+role]=duplicate||color<0||(separated&&separated[pair])?UINT64_MAX:
            (std::uint64_t(static_cast<unsigned>(color))<<32)|static_cast<unsigned>(pair);
    }
}
__global__ void color_pair_offsets(const std::uint64_t* keys,std::size_t count,int colors,std::size_t* offsets) {
    const int color=blockIdx.x*blockDim.x+threadIdx.x;
    if(color>colors)return;
    const std::uint64_t key=std::uint64_t(static_cast<unsigned>(color))<<32;
    std::size_t begin=0,end=count;
    while(begin<end){const auto mid=begin+(end-begin)/2;if(keys[mid]<key)begin=mid+1;else end=mid;}
    offsets[color]=begin;
}
__global__ void color_pair_indices(const std::uint64_t* keys,std::size_t count,int* pairs) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)pairs[i]=static_cast<int>(keys[i]&UINT32_MAX);
}
__global__ void bind_color_record_slots(BroadPhaseView broad,ColorContactWorklist colors,
    int color,std::size_t capacity,int* slots,int* failed) {
    const auto local=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=colors.offsets[color],end=colors.offsets[color+1];
    if(local>=end-begin||atomicAdd(failed,0))return;
    const int pair=colors.pair_indices[begin+local];
    if(local>=capacity||pair<0||static_cast<std::size_t>(pair)>=broad.contact_count
        ||(local&&colors.pair_indices[begin+local-1]>=pair)) {
        error(failed,20);return;
    }
    slots[pair]=static_cast<int>(local);
}
__global__ void validate_color_local_contact_kinds(BroadPhaseView broad,
    ContactParameters params,int* failed) {
    if(params.k_barrier==0||params.d_hat<=0)return;
    for(auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        pair<broad.contact_count;pair+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if(atomicAdd(failed,0))return;
        const int kind=broad.contacts[pair].kind;
        if(kind!=0&&kind!=1){error(failed,20);return;}
    }
}
__global__ void map_deformable_blocks(DeformableScene scene,int block_count,int* mapping,int* failed) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node])return;
    const int block=scene.node_blocks[node];
    if(block<0||block>=block_count||node>INT_MAX){error(failed,27);return;}
    const int prior=atomicCAS(mapping+block,-1,static_cast<int>(node));
    if(prior!=-1&&prior!=static_cast<int>(node))error(failed,27);
}
__device__ bool selected_node(const DeformableScene& scene,const RigidScene& rigid,
    const BroadPhaseView& broad,int vertex,int color,ContactAssemblyPhase phase) {
    if(broad.block_colors[scene.node_blocks[vertex]]!=color)return false;
    if(phase==ContactAssemblyPhase::Deformable)return scene.deformable[vertex]!=0;
    const int body=rigid.node_body?rigid.node_body[vertex]:-1;
    if(body<0)return false;
    return phase==ContactAssemblyPhase::RigidTranslation?rigid.translate[body]!=0:rigid.rotate[body]!=0;
}
template<bool DeferredInput>
__global__ void gather_compute(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    if(work.record_slots&&contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    ContactInputPublication<DeferredInput> publication(work.inputs?work.inputs+slot:nullptr);
    auto& in=publication.input;
    in.kind=contact.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=0;
    if(phase==ContactAssemblyPhase::RigidTranslation&&contact.kind==0) {
        // Either rigid side of PT uses the point's H00 COM shortcut.
        in.hessian_blocks=1u;
    } else {
        for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j]) {
            if(phase==ContactAssemblyPhase::Deformable&&i!=j)continue;
            in.hessian_blocks|=1u<<(4*i+j);
        }
    }
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    const auto result=evaluate_contact(in);
    store_contact_output(in,result,work.outputs+slot,work.compact?work.compact+slot:nullptr,
        work.output_activity?work.output_activity+slot:nullptr);
    const auto status=result.status;
    if(publication.publish_if_queued(status)) {
        const unsigned index=atomicAdd(work.exact_count,1u);work.exact_queue[index]=slot;
    } else if(status!=ContactStatus::Ok)error(failed,code(status));
}

// ORIGINAL (exact_computation_fallback == false) ordinary gather. It uses the
// qualified evaluate_contact_original geometry and the generic gathered/masked
// ContactOutput path. Only a frozen DAT EE packet's rounded zero distance may
// enter the emergency exact queue. Ordinary Original GS failures still abort.
template<bool DeferredInput,bool ClassifySelfZero=false>
__global__ void gather_original(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs,bool frozen_snapshot) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    if(work.record_slots&&contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    ContactInputPublication<DeferredInput> publication(work.inputs?work.inputs+slot:nullptr);
    auto& in=publication.input;
    in.kind=contact.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=0;
    if(phase==ContactAssemblyPhase::RigidTranslation&&contact.kind==0) {
        in.hessian_blocks=1u;
    } else {
        for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j]) {
            if(phase==ContactAssemblyPhase::Deformable&&i!=j)continue;
            in.hessian_blocks|=1u<<(4*i+j);
        }
    }
    if constexpr(ClassifySelfZero) {
        const int role=compact_contact_role(in.hessian_blocks);
        if(!frozen_snapshot&&phase==ContactAssemblyPhase::Deformable&&work.compact
            &&params.friction==0.0&&role>=0) {
            // Private same-stream partition tag, consumed/reset by the thin pass.
            work.active[slot]=static_cast<unsigned char>(2+role);return;
        }
    }
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    const auto result=evaluate_contact_original(in);
    store_contact_output(in,result,work.outputs+slot,work.compact?work.compact+slot:nullptr,
        work.output_activity?work.output_activity+slot:nullptr);
    if(original_dat_zero_distance_retry(true,frozen_snapshot,in.kind,result.status)
        &&publication.publish_if_queued(result.status)) {
        const unsigned index=atomicAdd(work.exact_count,1u);work.exact_queue[index]=slot;
    } else if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
}

template<bool Original>
__global__ void gather_tile(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,
    int* failed,const unsigned char* separated,std::size_t first,std::size_t last,int kind,
    bool frozen_snapshot) {
    const auto slot=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=last-first)return;
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    const auto pair=first+slot;
    const auto& contact=broad.contacts[pair];
    if(contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    if(contact.kind!=kind||(separated&&separated[pair]))return;
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    ContactInputPublication<true> publication(work.inputs?work.inputs+slot:nullptr);
    auto& in=publication.input;
    in.kind=kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=0;
    if(phase==ContactAssemblyPhase::RigidTranslation&&kind==0)in.hessian_blocks=1u;
    else for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j]) {
        if(phase==ContactAssemblyPhase::Deformable&&i!=j)continue;
        in.hessian_blocks|=1u<<(4*i+j);
    }
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    ContactOutput result;
    if constexpr(Original)result=evaluate_contact_original(in);
    else result=evaluate_contact(in);
    store_contact_output(in,result,work.outputs+slot,nullptr,work.output_activity?work.output_activity+slot:nullptr);
    const bool retry=!Original||original_dat_zero_distance_retry(true,frozen_snapshot,in.kind,result.status);
    if(retry&&publication.publish_if_queued(result.status)) {
        work.exact_queue[atomicAdd(work.exact_count,1u)]=static_cast<int>(slot);
    } else if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
}

// Full global packet publication from the unchanged 240-byte value evaluator.
// An inactive activity-gated slot retains its prior full payload exactly.
__device__ void store_original_pt_translation_zero(
    const original_contact_storage::SelfZeroOutput& out,ContactOutput* full,
    unsigned char* output_activity) {
    if(output_activity) {
        *output_activity=static_cast<unsigned char>(out.status!=ContactStatus::Ok||out.active!=0);
        if(!*output_activity)return;
    }
    full->distance=out.distance;
    for(int i=0;i<4;++i)full->weights[i]=out.weights[i];
    for(int k=0;k<3;++k)full->normal[k]=out.normal[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        full->barrier_gradient[i][k]=out.barrier_gradient[i][k];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<9;++k)
        full->barrier_hessian[i][j][k]=(i==0&&j==0)?out.selected_hessian[k]:0.0;
    for(int k=0;k<3;++k)full->friction_relative_gradient[k]=0.0;
    for(int k=0;k<9;++k)full->friction_relative_hessian[k]=0.0;
    full->status=out.status;full->active=out.active;
}

__global__ void gather_tile_original_pt_translation_zero(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,
    int* failed,const unsigned char* separated,std::size_t first,std::size_t last,int kind,
    bool frozen_snapshot) {
    const auto slot=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=last-first)return;
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(phase!=ContactAssemblyPhase::RigidTranslation||kind!=0||frozen_snapshot||params.friction!=0.0) {
        error(failed,code(ContactStatus::InvalidInput));return;
    }
    const auto pair=first+slot;
    const auto& contact=broad.contacts[pair];
    if(contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    if(contact.kind!=kind||(separated&&separated[pair]))return;
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,ContactAssemblyPhase::RigidTranslation);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    ContactInputPublication<true> publication(work.inputs?work.inputs+slot:nullptr);
    auto& in=publication.input;
    in.kind=ContactKind::PointTriangle;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=1u;
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    const auto result=evaluate_contact_original_self_zero(in,0);
    store_original_pt_translation_zero(result,work.outputs+slot,
        work.output_activity?work.output_activity+slot:nullptr);
    const bool retry=original_dat_zero_distance_retry(true,frozen_snapshot,in.kind,result.status);
    if(retry&&publication.publish_if_queued(result.status)) {
        work.exact_queue[atomicAdd(work.exact_count,1u)]=static_cast<int>(slot);
    } else if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
}

#include "original_sparse_gather_kernels.cuh"

// Paired opt-in gather: general resets all; thin writes only its eligible partition.
template<class SelectedOutput>
__device__ void store_selected_ordinary_result(const SelectedOutput& out,int role,
    CompactDeformableContact* compact,unsigned char* output_activity) {
    // Same publication order/early returns as original store_contact_output.
    if(output_activity) {
        *output_activity=static_cast<unsigned char>(out.status!=ContactStatus::Ok||out.active!=0);
        if(!*output_activity)return;
    }
    compact->status=out.status;compact->active=out.active;compact->role=role;
    if(out.status!=ContactStatus::Ok||!out.active)return;
    compact->weight=out.weights[role];
    for(int k=0;k<3;++k) {
        compact->gradient[k]=out.barrier_gradient[role][k];
        compact->friction_gradient[k]=0.0;
    }
    for(int k=0;k<9;++k) {
        compact->hessian[k]=out.selected_hessian[k];
        compact->friction_hessian[k]=0.0;
    }
    return;
}
// Original sole-self storage path. The preceding Original complement retains
// every unsupported packet; no Robust geometry is used by this kernel.
__global__ void gather_original_self_zero(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    const unsigned partition=work.active[slot];
    if(partition<2||partition>5)return;
    // Clear every private tag, including after a general-path failure.
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(phase!=ContactAssemblyPhase::Deformable||!work.compact||params.friction!=0.0) {
        error(failed,20);return;
    }
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    if(work.record_slots&&contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    const int role=static_cast<int>(partition)-2;
    ContactInputPublication<true> publication(work.inputs?work.inputs+slot:nullptr);
    auto& in=publication.input;
    in.kind=contact.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=1u<<(5*role);
    for(int vertex=0;vertex<4;++vertex)for(int axis=0;axis<3;++axis) {
        in.positions[vertex][axis]=state.positions[contact.vertices[vertex]][axis];
        in.previous[vertex][axis]=state.previous[contact.vertices[vertex]][axis];
    }
    work.active[slot]=1;
    const auto result=evaluate_contact_original_self_zero(in,role);
    store_selected_ordinary_result(result,role,work.compact+slot,
        work.output_activity?work.output_activity+slot:nullptr);
    // Ordinary Original failures abort. Frozen DAT never enters this route.
    if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
}
template<bool DeferredInput>
__global__ void gather_general_complement(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    if(work.record_slots&&contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    unsigned hessian_blocks=0;
    if(phase==ContactAssemblyPhase::RigidTranslation&&contact.kind==0) {
        // Either rigid side of PT uses the point's H00 COM shortcut.
        hessian_blocks=1u;
    } else {
        for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j]) {
            if(phase==ContactAssemblyPhase::Deformable&&i!=j)continue;
            hessian_blocks|=1u<<(4*i+j);
        }
    }
    const int selected_role=compact_contact_role(hessian_blocks);
    const bool selected_eligible=phase==ContactAssemblyPhase::Deformable
        &&work.compact&&params.friction==0.0&&selected_role>=0;
    // Transient partition marker, consumed/reset by the immediately following
    // thin kernel on the same stream. No consumer observes these private tags.
    if(selected_eligible){work.active[slot]=static_cast<unsigned char>(2+selected_role);return;}
    ContactInputPublication<DeferredInput> publication(work.inputs+slot);
    auto& in=publication.input;
    in.kind=contact.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=hessian_blocks;
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    const auto result=evaluate_contact(in);
    store_contact_output(in,result,work.outputs+slot,work.compact?work.compact+slot:nullptr,
        work.output_activity?work.output_activity+slot:nullptr);
    const auto status=result.status;
    if(publication.publish_if_queued(status)) {
        const unsigned index=atomicAdd(work.exact_count,1u);work.exact_queue[index]=slot;
    } else if(status!=ContactStatus::Ok)error(failed,code(status));
}
template<bool DeferredInput>
__global__ void gather_selected_ordinary(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    // Generic classification just established the current live sole-self role.
    // Unsupported generic active1 and pruned/unselected active0 remain untouched.
    const unsigned partition=work.active[slot];
    if(partition<2||partition>5)return;
    // Clear the transient tag even when a generic error stopped the assembly.
    // This also clears output_activity when it aliases active in the pipeline.
    work.active[slot]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    if(work.record_slots&&contact.kind!=0&&contact.kind!=1){error(failed,20);return;}
    const int selected_role=static_cast<int>(partition)-2;
    const unsigned hessian_blocks=1u<<(5*selected_role);
    ContactInputPublication<DeferredInput> publication(work.inputs+slot);
    auto& in=publication.input;
    in.kind=contact.kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=hessian_blocks;
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    work.active[slot]=1;
    const auto result=selected_ordinary::evaluate_eligible(in,selected_role);
    store_selected_ordinary_result(result,selected_role,work.compact+slot,
        work.output_activity?work.output_activity+slot:nullptr);
    const auto status=result.status;
    if(publication.publish_if_queued(status)) {
        const unsigned index=atomicAdd(work.exact_count,1u);work.exact_queue[index]=slot;
    } else if(status!=ContactStatus::Ok)error(failed,code(status));
}
template<int Limbs,bool Retry>
__global__ void evaluate_exact_queue(Work work,int* failed) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned count=Retry?*work.exact_retry_count:*work.exact_count;
    if(worker>=count||atomicAdd(failed,0))return;
    const int* queue=Retry?work.exact_retry_queue:work.exact_queue;
    exact::DevicePool pool;
    exact::initialize_pool(pool,
        work.exact_words+std::size_t(worker)*Limbs*exact_slots,
        work.exact_free_slots+std::size_t(worker)*exact_slots,Limbs,exact_slots);
    // A queue cursor balances heterogeneous exact queries. Each pair writes
    // only its own output; later per-node reductions retain their old order.
    for(;;) {
        if(atomicAdd(failed,0))break;
        const unsigned index=atomicAdd(work.exact_next,1u);
        if(index>=count)break;
        const int slot=queue[index];
        const auto result=evaluate_contact_exact<Limbs>(work.inputs[slot]);
        if(pool.failed||pool.available!=pool.slot_count){error(failed,24);break;}
        if constexpr(!Retry) {
            if(result.status==ContactStatus::ExactCapacityExceeded) {
                const unsigned retry=atomicAdd(work.exact_retry_count,1u);
                work.exact_retry_queue[retry]=slot;
                continue;
            }
        }
        if(result.status!=ContactStatus::Ok){error(failed,code(result.status));break;}
        store_contact_output(work.inputs[slot],result,work.outputs+slot,work.compact?work.compact+slot:nullptr,
            work.output_activity?work.output_activity+slot:nullptr);
    }
    exact::release_pool();
}
__device__ bool finite_system(const Derivatives& system) {
    for(int i=0;i<3;++i)if(!contact_detail::finite(system.gradient[i]))return false;
    for(int i=0;i<9;++i)if(!contact_detail::finite(system.hessian[i]))return false;
    return true;
}
__device__ void add_point_contact(Derivatives& system,const ContactOutput& out,int role,double normal_scale) {
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
__device__ void add_compact_contact(Derivatives& system,const CompactDeformableContact& out,double normal_scale) {
    // Keep the original scalar operation/accumulation order, including the
    // friction additions when they are zero. Do not fuse or pre-scale terms.
    const double weight=out.weight;
    for(int k=0;k<3;++k) {
        system.gradient[k]+=normal_scale*out.gradient[k];
        system.gradient[k]+=weight*out.friction_gradient[k];
        for(int l=0;l<3;++l) {
            system.hessian[3*k+l]+=normal_scale*out.hessian[3*k+l];
            system.hessian[3*k+l]+=(weight*out.friction_hessian[3*k+l])*weight;
        }
    }
}
__device__ bool add_deformable_sdf(Derivatives& system,const DeformableScene& scene,
    const DeformableState& state,const ElasticWorkspace& elastic,const SdfScene& sdf,
    const ContactParameters& params,const Work& work,std::size_t node) {
    if(!scene.solid[node]||!work.surface_query||work.surface_query[node]) {
        const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,state.positions[node],state.previous[node],sdf.parameters);
        if(out.status!=SdfStatus::Ok){error(elastic.error,25);return false;}
        const double dt2=params.dt*params.dt;
        for(int k=0;k<3;++k) {
            system.gradient[k]+=dt2*out.gradient[k];system.gradient[k]+=out.friction_gradient[k];
            for(int l=0;l<3;++l){system.hessian[3*k+l]+=dt2*out.hessian[3*k+l];system.hessian[3*k+l]+=out.friction_hessian[3*k+l];}
        }
    }
    return true;
}
__device__ void reduce_deformable_node(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,std::size_t node,
    bool frozen_snapshot=false) {
    Derivatives system=elastic.systems[node];
    const int block=scene.node_blocks[node];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    // CPU Jacobi assembles its noncontact/SDF system before mesh contact.
    // Keep the existing colored path's contact-then-SDF addition order.
    if(!contact_pending(work)&&frozen_snapshot&&!add_deformable_sdf(system,scene,state,elastic,sdf,params,work,node))return;
    for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
        const int pair=contact_incident_pair(work,incident);
        const int slot=contact_record_slot(work,pair);
        if(slot<0){error(elastic.error,20);return;}
        if(!work.active[slot])continue;
        if(work.compact) {
            const auto& compact=work.compact[slot];
            if(compact.status!=ContactStatus::Ok){error(elastic.error,code(compact.status));return;}
            if(!compact.active)continue;
            if(compact.role>=0) {
                if(broad.contacts[pair].vertices[compact.role]==static_cast<int>(node))
                    add_compact_contact(system,compact,normal_scale);
                continue;
            }
        }
        const auto& out=work.outputs[slot];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(!out.active)continue;
        for(int role=0;role<4;++role)if(broad.contacts[pair].vertices[role]==static_cast<int>(node))
            add_point_contact(system,out,role,normal_scale);
    }
    if(!contact_pending(work)&&!frozen_snapshot&&!add_deformable_sdf(system,scene,state,elastic,sdf,params,work,node))return;
    if(!contact_pending(work)&&!finite_system(system)){error(elastic.error,26);return;}
    elastic.systems[node]=system;
}

__global__ void reduce_deformable(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,int color,bool frozen_snapshot) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    reduce_deformable_node(broad,scene,state,elastic,sdf,params,work,node,frozen_snapshot);
}

// Launch only this color's existing blocks. Each selected node still owns its
// complete scalar incident loop, including the original SDF/finite checks.
// A nonzero cutoff delegates only high-degree nodes to the compact warp pass.
__global__ void reduce_deformable_selected(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,int color,
    const int* block_to_deformable_node,int block_count,std::size_t warp_cutoff) {
    const auto index=static_cast<std::size_t>(broad.color_offsets[color])
        +static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=static_cast<std::size_t>(broad.color_offsets[color+1])||atomicAdd(elastic.error,0))return;
    const int block=broad.color_blocks[index];
    if(block<0||block>=block_count){error(elastic.error,20);return;}
    const int node=block_to_deformable_node[block];
    if(node<0)return;
    if(std::size_t(node)>=scene.nodes||!scene.deformable[node]
        ||scene.node_blocks[node]!=block||broad.block_colors[block]!=color) {
        error(elastic.error,20);return;
    }
    if(warp_cutoff&&work.offsets[block+1]-work.offsets[block]>=warp_cutoff)return;
    reduce_deformable_node(broad,scene,state,elastic,sdf,params,work,std::size_t(node));
}

#include "contact_warp_reduce.cuh"
#include "contact_component_reduce.cuh"
#include "contact_prefetch_reduce.cuh"

__device__ RigidResult evaluate_rigid_contact(const ContactOutput& out,
    const BpContact& contact,int body,RigidScene rigid,const NodePullback* pullbacks,
    bool orientation) {
    bool selected[4]{};
    RigidPreparedContact prepared{};
    prepared.segment_segment=contact.kind!=0;
    prepared.flags=orientation?RigidOrientationGradient|RigidOrientationHessian|RigidUpperCrossBlocks
        :RigidTranslationGradient|RigidTranslationHessian;
    prepared.flags|=RigidFriction;
    prepared.update_mask=orientation?2u:1u;
    for(int i=0;i<4;++i) {
        const int vertex=contact.vertices[i];selected[i]=rigid.node_body[vertex]==body;
        if(!selected[i])continue;
        prepared.friction_weights[i]=out.weights[i];
        for(int k=0;k<3;++k)prepared.gradients[i][k]=out.barrier_gradient[i][k];
        if(orientation) {
            const auto& node=pullbacks[vertex];
            for(int k=0;k<9;++k)prepared.jacobians[i][k]=node.jacobian[k];
            for(int a=0;a<3;++a)for(int k=0;k<9;++k)prepared.curvature[i][a][k]=node.curvature[a][k];
        }
    }
    for(int k=0;k<3;++k)prepared.friction_gradient[k]=out.friction_relative_gradient[k];
    for(int k=0;k<9;++k)prepared.friction_hessian[k]=out.friction_relative_hessian[k];
    if(!orientation&&!contact.kind) {
        // Production TranslationHessian deliberately uses the point self
        // block for either side of PT, avoiding cancellation in a 3x3 sum of
        // triangle-vertex blocks. This is more than just omitting cross terms.
        for(int k=0;k<9;++k)prepared.translation_hessian[k]=out.barrier_hessian[0][0][k];
    } else {
        for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j])for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
            prepared.hessians[i][j][3*k+l]=out.barrier_hessian[i][j][3*k+l];
            if(!orientation&&i!=j)prepared.hessians[j][i][3*l+k]=out.barrier_hessian[i][j][3*k+l];
        }
    }
    return evaluate_rigid(prepared);
}
__device__ void add_rigid_contact(Derivatives& system,const ContactOutput& out,
    const BpContact& contact,int body,RigidScene rigid,const NodePullback* pullbacks,
    double scale,bool orientation) {
    const auto result=evaluate_rigid_contact(out,contact,body,rigid,pullbacks,orientation);
    const double* gradient=orientation?result.barrier.orientation_gradient:result.barrier.translation_gradient;
    const double* H=orientation?result.barrier.orientation_orientation_hessian:result.barrier.translation_translation_hessian;
    const double* friction_g=orientation?result.friction.orientation_gradient:result.friction.translation_gradient;
    const double* friction_H=orientation?result.friction.orientation_orientation_hessian:result.friction.translation_translation_hessian;
    for(int k=0;k<3;++k) {
        system.gradient[k]+=scale*gradient[k];
        system.gradient[k]+=friction_g[k];
        for(int l=0;l<3;++l) {
            system.hessian[3*k+l]+=scale*H[3*k+l];
            system.hessian[3*k+l]+=friction_H[3*k+l];
        }
    }
}
// Slot i belongs to the first contact role owned by a rigid body. Four slots
// cover all existing direct-call and frozen/DAT ownership patterns, including
// multiple simultaneously selected bodies. Other slots are never consumed.
__global__ void prepare_rigid_contact_terms(BroadPhaseView broad,RigidScene scene,
    Work work,int color,bool orientation,int* failed,std::size_t first,
    std::size_t last,int kind,ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:first;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:last;
    if(index>=end-begin||atomicAdd(failed,0))return;
    const auto pair=color_pairs.offsets
        ?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):first+index;
    const int slot=work.record_slots?contact_record_slot(work,static_cast<int>(pair)):static_cast<int>(pair-first);
    if(slot<0){error(failed,20);return;}
    if(!work.active[slot])return;
    const auto& out=work.outputs[slot];
    if(out.status!=ContactStatus::Ok||!out.active)return;
    const auto& contact=broad.contacts[pair];
    if(kind>=0&&contact.kind!=kind)return;
    for(int role=0;role<4;++role) {
        const int body=scene.node_body[contact.vertices[role]];
        if(body<0||static_cast<std::size_t>(body)>=scene.bodies
            ||broad.block_colors[scene.body_blocks[body]]!=color
            ||(orientation?!scene.rotate[body]:!scene.translate[body]))continue;
        bool duplicate=false;
        for(int prior=0;prior<role;++prior)
            duplicate=duplicate||scene.node_body[contact.vertices[prior]]==body;
        if(duplicate)continue;
        // Match the original consumer set even for direct callers whose
        // node_body and vertex_block mappings disagree. Avoid evaluating a
        // body with no incidence row for this contact (and no marked J).
        bool incident=false;
        for(int i=0;i<4;++i)
            incident=incident||work.vertex_block[contact.vertices[i]]==scene.body_blocks[body];
        if(!incident)continue;
        const auto result=evaluate_rigid_contact(out,contact,body,scene,
            work.node_pullbacks,orientation);
        const double* g=orientation?result.barrier.orientation_gradient:result.barrier.translation_gradient;
        const double* H=orientation?result.barrier.orientation_orientation_hessian:result.barrier.translation_translation_hessian;
        const double* fg=orientation?result.friction.orientation_gradient:result.friction.translation_gradient;
        const double* fH=orientation?result.friction.orientation_orientation_hessian:result.friction.translation_translation_hessian;
        auto& terms=work.rigid_terms[rigid_one_slot::term_index(work.one_slot_terms,static_cast<std::size_t>(slot),static_cast<unsigned>(role))];
        for(int k=0;k<3;++k) {
            terms.barrier.gradient[k]=g[k];terms.friction.gradient[k]=fg[k];
            for(int l=0;l<3;++l) {
                terms.barrier.hessian[3*k+l]=H[3*k+l];
                terms.friction.hessian[3*k+l]=fH[3*k+l];
            }
        }
    }
}
__device__ void accumulate_rigid_contact(Derivatives& system,const ContactOutput& out,
    const BpContact& contact,int body,RigidScene scene,Work work,std::size_t slot,
    double scale,bool orientation) {
    int role=0;
    if(work.rigid_terms)
        while(role<4&&scene.node_body[contact.vertices[role]]!=body)++role;
    // Preserve the original direct-API behavior even for an incidence row
    // whose vertex_block mapping disagrees with node_body ownership.
    if(!work.rigid_terms||role==4) {
        add_rigid_contact(system,out,contact,body,scene,work.node_pullbacks,scale,orientation);
        return;
    }
    const auto& terms=work.rigid_terms[rigid_one_slot::term_index(work.one_slot_terms,slot,static_cast<unsigned>(role))];
    for(int k=0;k<3;++k) {
        system.gradient[k]+=scale*terms.barrier.gradient[k];
        system.gradient[k]+=terms.friction.gradient[k];
        for(int l=0;l<3;++l) {
            system.hessian[3*k+l]+=scale*terms.barrier.hessian[3*k+l];
            system.hessian[3*k+l]+=terms.friction.hessian[3*k+l];
        }
    }
}
// Integer-only proper-color capability, never inferred from omitted pair queues.
__global__ void validate_one_slot_colors(BroadPhaseView broad,int blocks,int* result) {
    if(blockIdx.x||threadIdx.x)return;
    result[0]=-1;result[1]=-1;
    if(!rigid_one_slot::proper_colors(broad.block_colors,broad.color_offsets,
        broad.color_blocks,blocks,broad.color_count))return;
    result[0]=broad.block_colors[0];result[1]=broad.block_colors[1];
}
__global__ void prepare_body_kinematics(BroadPhaseView broad,RigidScene scene,RigidState state,
    RigidKinematics* kinematics,double dt,int color,int* failed) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||!scene.rotate[body]||broad.block_colors[scene.body_blocks[body]]!=color||atomicAdd(failed,0))return;
    kinematics[body]=rigid_kinematics(state.previous_orientation[body],state.omega[body],dt);
}
__global__ void mark_pullback_nodes(BroadPhaseView broad,DeformableScene nodes,RigidScene scene,
    Work work,unsigned* needed,int color,int* failed,ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    if(atomicAdd(failed,0))return;
    const int slot=contact_record_slot(work,static_cast<int>(pair));
    if(slot<0){error(failed,20);return;}
    if(!work.active[slot]||!work.outputs[slot].active)return;
    for(int role=0;role<4;++role) {
        const int vertex=broad.contacts[pair].vertices[role];
        if(selected_node(nodes,scene,broad,vertex,color,ContactAssemblyPhase::RigidOrientation))
            atomicExch(needed+vertex,1u);
    }
}
__global__ void prepare_node_pullbacks(BroadPhaseView broad,DeformableScene nodes,RigidScene scene,
    const RigidKinematics* kinematics,const unsigned* needed,bool has_sdf,NodePullback* output,int color,int* failed) {
    const auto vertex=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(vertex>=nodes.nodes||atomicAdd(failed,0))return;
    const int body=scene.node_body[vertex];
    if(body<0||!scene.rotate[body]||broad.block_colors[scene.body_blocks[body]]!=color)return;
    if(!has_sdf&&!needed[vertex])return;
    auto& node=output[vertex];
    rigid_reference_derivatives(scene.reference[vertex],kinematics[body],node.jacobian,node.curvature);
    for(int k=0;k<9;++k) {
        if(!contact_detail::finite(node.jacobian[k]))error(failed,26);
        for(int a=0;a<3;++a)if(!contact_detail::finite(node.curvature[a][k]))error(failed,26);
    }
}
__device__ bool add_rigid_sdf(Derivatives& system,const DeformableState& deformable,
    const RigidScene& scene,const ElasticWorkspace& elastic,const SdfScene& sdf,
    const ContactParameters& params,const Work& work,std::size_t body,bool orientation) {
    if(sdf.parameters.stiffness>0)for(auto offset=scene.node_offsets[body];offset<scene.node_offsets[body+1];++offset) {
        const int vertex=scene.nodes[offset];
        const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,deformable.positions[vertex],deformable.previous[vertex],sdf.parameters);
        if(out.status!=SdfStatus::Ok){error(elastic.error,25);return false;}
        double J[9]{};
        if(orientation)for(int k=0;k<9;++k)J[k]=work.node_pullbacks[vertex].jacobian[k];
        const auto pulled=sdf_rigid_pullback(out,J,params.dt*params.dt);
        const double* g=orientation?pulled.orientation_gradient:pulled.translation_gradient;
        const double* H=orientation?pulled.orientation_orientation_hessian:pulled.translation_translation_hessian;
        for(int k=0;k<3;++k){system.gradient[k]+=g[k];for(int l=0;l<3;++l)system.hessian[3*k+l]+=H[3*k+l];}
    }
    return true;
}
__global__ void reduce_rigid(BroadPhaseView broad,DeformableState deformable,RigidScene scene,
    RigidState state,RigidWorkspace workspace,ElasticWorkspace elastic,SdfScene sdf,
    ContactParameters params,Work work,int color,bool orientation,RigidPinAssembly pins,
    bool frozen_snapshot) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color||atomicAdd(elastic.error,0))return;
    if(orientation?!scene.rotate[body]:!scene.translate[body])return;
    Derivatives system=workspace.systems[body];
    // New pin forces follow the CPU noncontact order: inertia, SDF, then one
    // stable per-body pin sum. Preserve the old no-pin contact/SDF order.
    if(!contact_pending(work)&&(pins.scene.count||frozen_snapshot))
        if(!add_rigid_sdf(system,deformable,scene,elastic,sdf,params,work,body,orientation))return;
    if(!contact_pending(work)&&pins.scene.count) {
        const auto kin=rigid_kinematics(state.previous_orientation[body],state.omega[body],params.dt);
        RigidDerivatives contribution{};
        const int status=evaluate_rigid_body_pins(pins.scene,body,state.com[body],kin,
            params.dt,pins.stiffness,pins.time,orientation,contribution);
        if(status){error(elastic.error,status);return;}
        const auto* g=orientation?contribution.orientation_gradient:contribution.translation_gradient;
        const auto* H=orientation?contribution.orientation_orientation_hessian:contribution.translation_translation_hessian;
        for(int k=0;k<3;++k) {
            system.gradient[k]+=g[k];
            for(int l=0;l<3;++l)system.hessian[3*k+l]+=H[3*k+l];
        }
    }
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    const int block=scene.body_blocks[body];
    for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
        const int pair=contact_incident_pair(work,incident);
        const int slot=contact_record_slot(work,pair);
        if(slot<0){error(elastic.error,20);return;}
        if(!work.active[slot])continue;
        const auto& out=work.outputs[slot];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(out.active)accumulate_rigid_contact(system,out,broad.contacts[pair],static_cast<int>(body),
            scene,work,static_cast<std::size_t>(slot),normal_scale,orientation);
    }
    if(!contact_pending(work)&&!pins.scene.count&&!frozen_snapshot
        &&!add_rigid_sdf(system,deformable,scene,elastic,sdf,params,work,body,orientation))return;
    if(!contact_pending(work)&&!finite_system(system)){error(elastic.error,26);return;}
    workspace.systems[body]=system;
}

__device__ std::size_t tile_incident_bound(BroadPhaseView broad,const Work& work,
    std::size_t begin,std::size_t end,int kind,std::size_t pair) {
    while(begin<end) {
        const auto mid=begin+(end-begin)/2;
        const int candidate=work.pair_indices[mid],candidate_kind=broad.contacts[candidate].kind;
        if(candidate_kind<kind||(candidate_kind==kind&&std::size_t(candidate)<pair))begin=mid+1;
        else end=mid;
    }
    return begin;
}
__global__ void seed_pending_deformable(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,
    Work work,int color,bool frozen,Derivatives* pending) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]
        ||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    Derivatives system=elastic.systems[node];
    if(frozen&&!add_deformable_sdf(system,scene,state,elastic,sdf,params,work,node))return;
    pending[node]=system;
}
__global__ void seed_pending_rigid(BroadPhaseView broad,DeformableState deformable,
    RigidScene scene,RigidState state,RigidWorkspace workspace,ElasticWorkspace elastic,
    SdfScene sdf,ContactParameters params,Work work,int color,bool orientation,
    RigidPinAssembly pins,bool frozen,Derivatives* pending) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color
        ||(orientation?!scene.rotate[body]:!scene.translate[body])||atomicAdd(elastic.error,0))return;
    Derivatives system=workspace.systems[body];
    if(pins.scene.count||frozen)
        if(!add_rigid_sdf(system,deformable,scene,elastic,sdf,params,work,body,orientation))return;
    if(pins.scene.count) {
        const auto kin=rigid_kinematics(state.previous_orientation[body],state.omega[body],params.dt);
        RigidDerivatives contribution{};
        const int status=evaluate_rigid_body_pins(pins.scene,body,state.com[body],kin,
            params.dt,pins.stiffness,pins.time,orientation,contribution);
        if(status){error(elastic.error,status);return;}
        const auto* g=orientation?contribution.orientation_gradient:contribution.translation_gradient;
        const auto* H=orientation?contribution.orientation_orientation_hessian:contribution.translation_translation_hessian;
        for(int k=0;k<3;++k) {
            system.gradient[k]+=g[k];
            for(int l=0;l<3;++l)system.hessian[3*k+l]+=H[3*k+l];
        }
    }
    pending[body]=system;
}
__global__ void reduce_deformable_tile(BroadPhaseView broad,DeformableScene scene,
    ElasticWorkspace elastic,ContactParameters params,Work work,int color,
    std::size_t first,std::size_t last,int kind,Derivatives* pending) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]
        ||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    const int block=scene.node_blocks[node];
    const auto row_begin=work.offsets[block],row_end=work.offsets[block+1];
    const auto begin=tile_incident_bound(broad,work,row_begin,row_end,kind,first);
    const auto end=tile_incident_bound(broad,work,begin,row_end,kind,last);
    if(begin==end)return;
    Derivatives system=pending[node];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    for(auto incident=begin;incident<end;++incident) {
        const auto pair=static_cast<std::size_t>(work.pair_indices[incident]),slot=pair-first;
        if(!work.active[slot])continue;
        const auto& out=work.outputs[slot];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(out.active)for(int role=0;role<4;++role)
            if(broad.contacts[pair].vertices[role]==static_cast<int>(node))
                add_point_contact(system,out,role,normal_scale);
    }
    pending[node]=system;
}
__global__ void mark_tile_pullbacks(BroadPhaseView broad,DeformableScene nodes,RigidScene scene,
    Work work,unsigned* needed,int color,int* failed,std::size_t first,std::size_t last,int kind) {
    const auto slot=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=last-first||atomicAdd(failed,0)||!work.active[slot]||!work.outputs[slot].active)return;
    const auto& pair=broad.contacts[first+slot];
    if(pair.kind!=kind)return;
    for(int role=0;role<4;++role) {
        const int vertex=pair.vertices[role];
        if(selected_node(nodes,scene,broad,vertex,color,ContactAssemblyPhase::RigidOrientation))
            atomicExch(needed+vertex,1u);
    }
}
__global__ void reduce_rigid_tile(BroadPhaseView broad,RigidScene scene,
    ElasticWorkspace elastic,ContactParameters params,Work work,int color,bool orientation,
    std::size_t first,std::size_t last,int kind,Derivatives* pending) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color
        ||(orientation?!scene.rotate[body]:!scene.translate[body])||atomicAdd(elastic.error,0))return;
    const int block=scene.body_blocks[body];
    const auto row_begin=work.offsets[block],row_end=work.offsets[block+1];
    const auto begin=tile_incident_bound(broad,work,row_begin,row_end,kind,first);
    const auto end=tile_incident_bound(broad,work,begin,row_end,kind,last);
    if(begin==end)return;
    Derivatives system=pending[body];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    for(auto incident=begin;incident<end;++incident) {
        const auto pair=static_cast<std::size_t>(work.pair_indices[incident]),slot=pair-first;
        if(!work.active[slot])continue;
        const auto& out=work.outputs[slot];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(out.active)accumulate_rigid_contact(system,out,broad.contacts[pair],static_cast<int>(body),
            scene,work,slot,normal_scale,orientation);
    }
    pending[body]=system;
}
// One CTA owns a fixed CSR chunk for one body. Stable warp/CTA integer ranks
// only select records; every floating-point addition remains in the original
// body consumer. Counts are overwritten even for inactive/empty chunks.
__global__ void classify_rigid_tile_chunks(BroadPhaseView broad,RigidScene scene,
    Work work,int color,bool orientation,std::size_t first,std::size_t last,int kind,
    std::size_t chunk_stride,std::size_t tile_chunks,std::uint32_t* kept,
    unsigned* counts,int* failed) {
    const auto group=static_cast<std::size_t>(blockIdx.x);
    const auto body=group/tile_chunks,chunk=group%tile_chunks;
    const unsigned lane=threadIdx.x,warp=lane/32,warp_lane=lane%32;
    __shared__ std::size_t begin,end;
    __shared__ int enabled;
    __shared__ unsigned warp_counts[4],warp_offsets[4];
    if(lane==0) {
        enabled=0;
        if(body<scene.bodies&&chunk<chunk_stride) {
            counts[body*chunk_stride+chunk]=0;
            if(broad.block_colors[scene.body_blocks[body]]==color
                &&(orientation?scene.rotate[body]:scene.translate[body])&&!atomicAdd(failed,0)) {
                const int block=scene.body_blocks[body];
                const auto row_begin=work.offsets[block],row_end=work.offsets[block+1];
                begin=tile_incident_bound(broad,work,row_begin,row_end,kind,first);
                end=tile_incident_bound(broad,work,begin,row_end,kind,last);
                // The public CSR uses size_t. Preserve wide direct-API rows
                // through the old consumer rather than truncating absolute IDs.
                if(rigid_compact::absolute_ids_fit(end)&&end-begin<=last-first
                    &&chunk<=(end-begin)/threads)
                    enabled=begin+chunk*threads<end;
            }
        }
    }
    __syncthreads();
    if(!enabled)return;
    const auto incident=begin+chunk*threads+lane;
    bool keep=false;
    if(incident<end) {
        const auto pair=static_cast<std::size_t>(work.pair_indices[incident]),slot=pair-first;
        if(work.active[slot]) {
            const auto& out=work.outputs[slot];
            keep=out.status!=ContactStatus::Ok||out.active;
        }
    }
    const unsigned mask=__ballot_sync(0xffffffffu,keep);
    if(warp_lane==0)warp_counts[warp]=__popc(mask);
    __syncthreads();
    if(lane==0) {
        unsigned total=0;
        for(unsigned w=0;w<4;++w){warp_offsets[w]=total;total+=warp_counts[w];}
        counts[body*chunk_stride+chunk]=total;
    }
    __syncthreads();
    if(keep) {
        const unsigned prior=__popc(mask&((1u<<warp_lane)-1u));
        kept[(body*chunk_stride+chunk)*threads+warp_offsets[warp]+prior]=static_cast<std::uint32_t>(incident);
    }
}
__global__ void reduce_rigid_tile_compact(BroadPhaseView broad,RigidScene scene,
    ElasticWorkspace elastic,ContactParameters params,Work work,int color,bool orientation,
    std::size_t first,std::size_t last,int kind,const std::uint32_t* kept,
    const unsigned* counts,std::size_t chunk_stride,Derivatives* pending) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color
        ||(orientation?!scene.rotate[body]:!scene.translate[body])||atomicAdd(elastic.error,0))return;
    const int block=scene.body_blocks[body];
    const auto row_begin=work.offsets[block],row_end=work.offsets[block+1];
    const auto begin=tile_incident_bound(broad,work,row_begin,row_end,kind,first);
    const auto end=tile_incident_bound(broad,work,begin,row_end,kind,last);
    if(begin==end)return;
    Derivatives system=pending[body];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    if(!rigid_compact::absolute_ids_fit(end)||end-begin>last-first) {
        // Identical ordered legacy loop for absolute CSR IDs wider than the
        // compact index, or a noncanonical row exceeding one incidence per
        // packet pair. No record is dropped or narrowed to uint32.
        for(auto incident=begin;incident<end;++incident) {
            const auto pair=static_cast<std::size_t>(work.pair_indices[incident]),slot=pair-first;
            if(!work.active[slot])continue;
            const auto& out=work.outputs[slot];
            if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
            if(out.active)accumulate_rigid_contact(system,out,broad.contacts[pair],static_cast<int>(body),
                scene,work,slot,normal_scale,orientation);
        }
    } else {
        const auto chunks=(last-first)/threads+((last-first)%threads!=0);
        if(chunks>chunk_stride){error(elastic.error,20);return;}
        for(std::size_t chunk=0;chunk<chunks;++chunk) {
            const auto count=counts[body*chunk_stride+chunk];
            if(count>threads){error(elastic.error,20);return;}
            for(unsigned entry=0;entry<count;++entry) {
                const auto incident=static_cast<std::size_t>(kept[(body*chunk_stride+chunk)*threads+entry]);
                if(incident<begin||incident>=end){error(elastic.error,20);return;}
                const auto pair=static_cast<std::size_t>(work.pair_indices[incident]),slot=pair-first;
                if(pair<first||pair>=last||broad.contacts[pair].kind!=kind){error(elastic.error,20);return;}
                const auto& out=work.outputs[slot];
                if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
                if(out.active)accumulate_rigid_contact(system,out,broad.contacts[pair],static_cast<int>(body),
                    scene,work,slot,normal_scale,orientation);
            }
        }
    }
    pending[body]=system;
}
__global__ void finish_pending_deformable(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,
    Work work,int color,bool frozen,Derivatives* pending) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]
        ||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    Derivatives system=pending[node];
    if(!frozen&&!add_deformable_sdf(system,scene,state,elastic,sdf,params,work,node))return;
    if(!finite_system(system)){error(elastic.error,26);return;}
    pending[node]=system;
}
__global__ void finish_pending_rigid(BroadPhaseView broad,DeformableState deformable,
    RigidScene scene,ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,
    Work work,int color,bool orientation,bool has_pins,bool frozen,Derivatives* pending) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color
        ||(orientation?!scene.rotate[body]:!scene.translate[body])||atomicAdd(elastic.error,0))return;
    Derivatives system=pending[body];
    if(!has_pins&&!frozen
        &&!add_rigid_sdf(system,deformable,scene,elastic,sdf,params,work,body,orientation))return;
    if(!finite_system(system)){error(elastic.error,26);return;}
    pending[body]=system;
}
__global__ void publish_pending(BroadPhaseView broad,DeformableScene nodes,RigidScene rigid,
    ElasticWorkspace elastic,RigidWorkspace workspace,int color,ContactAssemblyPhase phase,
    const Derivatives* pending) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(atomicAdd(elastic.error,0))return;
    if(phase==ContactAssemblyPhase::Deformable) {
        if(index<nodes.nodes&&nodes.deformable[index]
            &&broad.block_colors[nodes.node_blocks[index]]==color)elastic.systems[index]=pending[index];
    } else if(index<rigid.bodies&&broad.block_colors[rigid.body_blocks[index]]==color
        &&(phase==ContactAssemblyPhase::RigidOrientation?rigid.rotate[index]:rigid.translate[index]))
        workspace.systems[index]=pending[index];
}
}

struct GpuContactPipeline::Impl {
    const std::size_t packet_capacity;
    const ExactContactSchedule schedule=ExactContactSchedule::from_environment();
    const bool use_current_color_ccd_launch=contact_execution_option("IPC_GPU_CCD_CURRENT_COLOR_LAUNCH",false);
    ContactColorLaunchBinding color_launch_binding;
    const bool use_color_pairs=contact_execution_option("IPC_GPU_COLOR_PAIR_QUEUES");
    const bool color_pairs_explicitly_forced=[] {
        const char* value=std::getenv("IPC_GPU_COLOR_PAIR_QUEUES");
        return value&&std::strcmp(value,"1")==0;
    }();
    bool validated_two_rigid_blocks=false;
    bool omit_two_rigid_color_queues=false;
    bool use_two_rigid_dense_csr=false;
    const bool use_two_rigid_kind_packets=contact_execution_option(
        "IPC_GPU_TWO_RIGID_KIND_PACKETS",true);
    std::size_t dense_point_triangle_end=0;
    std::size_t validated_two_rigid_nodes=0;
    bool pair_indices_dense=false;
    Buffer<unsigned> dense_unsupported;
    const bool use_color_local_packets=contact_execution_option("IPC_GPU_COLOR_LOCAL_PACKETS",true);
    ContactColorLaunchBinding packet_color_binding;
    Buffer<int> record_slots;
    const int* active_record_slots=nullptr;
    bool pending_contacts=false;
    const bool defer_inputs=contact_execution_option("IPC_GPU_DEFER_CONTACT_INPUTS",false);
    const bool use_warp_reduce=contact_execution_option("IPC_GPU_WARP_CONTACT_REDUCE");
    const bool use_rigid_pullback_cache=contact_execution_option("IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE",true);
    const bool use_one_slot_terms=contact_execution_option("IPC_GPU_RIGID_ONE_SLOT_TERMS",true);
    const bool use_original_sparse_gather=contact_execution_option("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER",true);
    bool one_slot_graph_proven=false,rigid_terms_one_slot_layout=false;
    Buffer<int> one_slot_color_metadata;
    const bool use_rigid_compact_reduce=contact_execution_option("IPC_GPU_RIGID_COMPACT_REDUCE",true);
    const bool use_selected_reduce=contact_execution_option("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",false);
    const bool use_component_reduce=contact_execution_option("IPC_GPU_CONTACT_COMPONENT_REDUCE",false);
    const bool use_prefetch_reduce=contact_execution_option("IPC_GPU_CONTACT_PREFETCH_REDUCE",true);
    const int component_groups=component_reduce_groups();
    const int component_grain=ExactContactSchedule::environment_integer("IPC_GPU_CONTACT_COMPONENT_GRAIN",64,INT_MAX);
    const int component_max_color_blocks=ExactContactSchedule::environment_integer("IPC_GPU_CONTACT_COMPONENT_MAX_COLOR_BLOCKS",128,INT_MAX);
    const int selected_threads=selected_reduce_threads();
    const bool use_compact_warp_reduce=contact_execution_option("IPC_GPU_COMPACT_WARP_REDUCE",false);
    const int compact_warp_degree_cutoff=ExactContactSchedule::environment_integer(
        "IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE",32,INT_MAX);
    const bool use_cooperative_exact=contact_execution_option("IPC_GPU_COOPERATIVE_EXACT_CONTACT");
    const bool use_compact_point=contact_execution_option("IPC_GPU_COMPACT_POINT_FIRST_TIER",false);
    const bool use_selected_ordinary=contact_execution_option("IPC_GPU_SELECTED_ORDINARY_GATHER",true);
    const bool use_original_self_zero=contact_execution_option("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER",true);
    const bool use_original_pt_translation_zero=contact_execution_option("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER",true);
    const bool use_cooperative_compact_point=contact_execution_option("IPC_GPU_COMPACT_POINT_COOPERATIVE",false);
    // Preserve the original scalar/disabled paths, including flag parsing.
    const bool compact_point_ratio=use_compact_point&&use_cooperative_exact
        ?contact_execution_option("IPC_GPU_CONTACT_INPLACE_RATIO",false):false;
    const bool use_compact=contact_execution_option("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",true);
    const bool use_output_activity=contact_execution_option("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",true);
    const bool use_box_pruning=contact_execution_option("IPC_GPU_BOX_CONTACT_PRUNING",false);
    std::ofstream queue_log;
    unsigned long long rebuild_index=0,assembly_index=0;
    BroadPhaseView broad;
    int block_count=0;
    const int* vertex_block=nullptr;
    const unsigned char* surface_query=nullptr;
    Buffer<Incident> incidents;
    Buffer<int> pair_indices;
    Buffer<std::size_t> offsets,raw_block_counts;
    Buffer<std::uint64_t> color_keys;
    Buffer<int> color_pairs,block_to_deformable_node;
    Buffer<std::size_t> color_offsets;
    std::vector<std::size_t> host_color_offsets;
    std::vector<int> host_selected_offsets;
    std::size_t max_color_pairs=0;
    bool mapping_dirty=true;
    std::size_t mapped_nodes=0;
    const int* mapped_node_blocks=nullptr;
    const unsigned char* mapped_deformable=nullptr;
    Buffer<Derivatives> component_systems;
    Buffer<ContactInput> inputs;
    Buffer<ContactOutput> outputs;
    Buffer<CompactDeformableContact> compact;
    Buffer<unsigned char> active;
    Buffer<unsigned char> box_separated;
    bool pruning_ready=false,original_pruning_ready=false;
    double pruning_d_hat=0;
    Buffer<int> exact_queue,exact_retry_queue,exact_free_slots;
    Buffer<unsigned> exact_count,exact_retry_count,exact_next;
    Buffer<std::uint32_t> exact_words;
    Buffer<std::uint32_t> cooperative_words;
    Buffer<int> cooperative_slots;
    unsigned cooperative_workers=0;
    Buffer<RigidKinematics> body_kinematics;
    Buffer<NodePullback> node_pullbacks;
    // The cache is bounded by packet capacity: four generic owner slots,
    // or one slot for a proven selected rigid owner. At the default 262,144
    // pairs this is 192 MiB or 48 MiB, with no body*scene-pair allocation.
    Buffer<RigidContactTerms> rigid_terms;
    Buffer<unsigned> pullback_needed;
    Buffer<Derivatives> pending_systems;
    Buffer<std::uint32_t> rigid_kept_incidents;
    Buffer<unsigned> rigid_chunk_counts;
    Buffer<unsigned> sparse_gather_queue,sparse_gather_count;
    unsigned sparse_gather_worker_blocks=0;
    explicit Impl(std::size_t capacity):packet_capacity(capacity) {
        if(!capacity||capacity>std::size_t(INT_MAX))
            throw std::invalid_argument("GPU contact packet capacity must be in [1,INT_MAX]");
        // This explicitly separate diagnostic mode performs synchronous queue
        // count readbacks. Never enable it for clean runtime measurements.
        const char* path=std::getenv("IPC_GPU_CONTACT_QUEUE_CSV");
        if(path&&*path) {
            std::ifstream existing(path);
            if(existing.good())throw std::runtime_error(std::string("contact queue diagnostic output already exists: ")+path);
            queue_log.open(path);
            if(!queue_log)throw std::runtime_error(std::string("cannot create contact queue diagnostic output: ")+path);
            queue_log<<"rebuild,assembly,color,phase,candidates,exact_queries,retry_queries,device_status,small_workers,small_threads,large_workers,large_threads\n";
        }
    }
    void record_queues(int color,ContactAssemblyPhase phase,const int* failed,
        bool compact_queue_swap=false,unsigned original_exact_count=0) {
        if(!queue_log.is_open())return;
        unsigned count=original_exact_count,retries=0;int status=0;
        // CSV exact_queries remains original M; retry_queries remains actual
        // scalar256-to1600 W. B's compact-unresolved R is never mislabeled.
        if(!compact_queue_swap)
            checked(cudaMemcpy(&count,exact_count.data,sizeof(count),cudaMemcpyDeviceToHost),"diagnostic exact count");
        if(broad.contact_count)
            checked(cudaMemcpy(&retries,compact_queue_swap?exact_count.data:exact_retry_count.data,sizeof(retries),cudaMemcpyDeviceToHost),"diagnostic retry count");
        checked(cudaMemcpy(&status,failed,sizeof(status),cudaMemcpyDeviceToHost),"diagnostic device status");
        const char* phase_name=phase==ContactAssemblyPhase::Deformable?"deformable":
            phase==ContactAssemblyPhase::RigidTranslation?"rigid_translation":"rigid_orientation";
        queue_log<<rebuild_index<<','<<assembly_index<<','<<color<<','<<phase_name<<','<<broad.contact_count<<','
            <<count<<','<<retries<<','<<status<<','<<schedule.small_workers<<','<<schedule.small_threads<<','
            <<schedule.large_workers<<','<<schedule.large_threads<<'\n';
        queue_log.flush();
        if(!queue_log)throw std::runtime_error("failed writing contact queue diagnostic output");
    }
    Work work(bool compact_phase=false,bool one_slot_terms=false)const{return {inputs.data,outputs.data,compact_phase?compact.data:nullptr,active.data,use_output_activity?active.data:nullptr,exact_queue.data,exact_count.data,
        exact_retry_queue.data,exact_retry_count.data,exact_next.data,
        exact_words.data,exact_free_slots.data,incidents.data,pair_indices.data,offsets.data,vertex_block,surface_query,node_pullbacks.data,rigid_terms.data,active_record_slots,pending_contacts,one_slot_terms};}
    void prepare_packet_storage(bool need_exact,bool cooperative,bool compact_phase) {
        const auto packets=std::min(broad.contact_count,packet_capacity);
        if(compact_phase)compact.reserve(packets);else compact.release();
        if(need_exact&&packets) {
            inputs.reserve(packets);exact_queue.reserve(packets);exact_retry_queue.reserve(packets);
            const auto arena=schedule.arena_sizes(packets,exact_small_limbs,exact_large_limbs,exact_slots);
            exact_words.reserve(arena.words);exact_free_slots.reserve(arena.free_slots);
        } else {
            inputs.release();exact_queue.release();exact_retry_queue.release();
            exact_words.release();exact_free_slots.release();
        }
        if(cooperative&&packets) {
            cooperative_workers=static_cast<unsigned>(std::min<std::size_t>(packets,256));
            const auto arena=cooperative_exact_arena_sizes(cooperative_workers);
            cooperative_words.reserve(arena.words);cooperative_slots.reserve(arena.free_slots);
        } else {
            cooperative_workers=0;cooperative_words.release();cooperative_slots.release();
        }
    }
    std::size_t prepare_rigid_compact_storage(std::size_t bodies) {
        const auto pairs=std::min(broad.contact_count,packet_capacity);
        const auto chunks=pairs/threads+(pairs%threads!=0);
        if(!bodies||bodies>rigid_compact_body_limit
            ||chunks>std::numeric_limits<std::size_t>::max()/bodies)
            throw std::overflow_error("rigid compact chunk extent overflow");
        const auto groups=bodies*chunks;
        if(groups>std::numeric_limits<std::size_t>::max()/threads
            ||groups>std::size_t(INT_MAX))
            throw std::overflow_error("rigid compact chunk launch/storage overflow");
        rigid_kept_incidents.reserve(groups*threads);
        rigid_chunk_counts.reserve(groups);
        return chunks;
    }
    unsigned sparse_gather_blocks(std::size_t pairs) {
        if(!sparse_gather_worker_blocks) {
            int device=0,sms=0;checked(cudaGetDevice(&device),"sparse gather current device");
            checked(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"sparse gather SM count");
            if(sms<=0||static_cast<unsigned>(sms)>UINT_MAX/2)
                throw std::runtime_error("invalid sparse gather SM count");
            sparse_gather_worker_blocks=2*static_cast<unsigned>(sms);
        }
        return std::min(blocks(pairs),sparse_gather_worker_blocks);
    }
    ColorContactWorklist color_worklist()const {
        ColorContactWorklist result=!host_color_offsets.empty()?ColorContactWorklist{color_offsets.data,color_pairs.data,max_color_pairs}:ColorContactWorklist{};
        result.box_separated=pruning_ready?box_separated.data:nullptr;
        return result;
    }
};
GpuContactPipeline::GpuContactPipeline():GpuContactPipeline(65536){}
GpuContactPipeline::GpuContactPipeline(std::size_t packet_capacity)
    :impl_(std::make_unique<Impl>(packet_capacity)){}
GpuContactPipeline::~GpuContactPipeline()=default;
void GpuContactPipeline::bind_validated_two_rigid_blocks(std::size_t validated_nodes) {
    auto& p=*impl_;
    if(p.block_count||p.broad.contacts||p.broad.contact_count)
        throw std::logic_error("two-rigid contact hint must precede rebuild");
    p.validated_two_rigid_blocks=true;
    p.omit_two_rigid_color_queues=contact_execution_option(
        "IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION",true);
    p.validated_two_rigid_nodes=validated_nodes;
    p.use_two_rigid_dense_csr=contact_execution_option(
        "IPC_GPU_TWO_RIGID_DENSE_CSR",true);
}
ContactStorageBytes GpuContactPipeline::storage_bytes()const noexcept {
    const auto& p=*impl_;
    return {p.inputs.capacity*sizeof(ContactInput)+p.outputs.capacity*sizeof(ContactOutput)
            +p.compact.capacity*sizeof(CompactDeformableContact)+p.active.capacity
            +p.rigid_terms.capacity*sizeof(RigidContactTerms)
            +p.one_slot_color_metadata.capacity*sizeof(int)
            +p.rigid_kept_incidents.capacity*sizeof(std::uint32_t)
            +p.rigid_chunk_counts.capacity*sizeof(unsigned)
            +p.sparse_gather_queue.capacity*sizeof(unsigned)+p.sparse_gather_count.capacity*sizeof(unsigned)
            +p.record_slots.capacity*sizeof(int)
            +(p.exact_queue.capacity+p.exact_retry_queue.capacity)*sizeof(int)
            +(p.exact_count.capacity+p.exact_retry_count.capacity+p.exact_next.capacity)*sizeof(unsigned),
        (p.exact_words.capacity+p.cooperative_words.capacity)*sizeof(std::uint32_t)
            +(p.exact_free_slots.capacity+p.cooperative_slots.capacity)*sizeof(int)};
}
void GpuContactPipeline::rebuild(const BroadPhaseView& broad,int block_count,const int* vertex_block,const unsigned char* surface_query,const ContactBoxPruning& pruning) {
    auto& p=*impl_;
    p.color_launch_binding.begin_rebuild();
    p.packet_color_binding.begin_rebuild();
    p.active_record_slots=nullptr;p.pending_contacts=false;
    p.dense_point_triangle_end=0;
    if(block_count<0||(!vertex_block&&block_count)||broad.contact_count>std::size_t(INT_MAX))
        throw std::invalid_argument("GPU contact topology/count unsupported");
    p.broad=broad;p.block_count=block_count;p.vertex_block=vertex_block;p.surface_query=surface_query;
    p.one_slot_graph_proven=false;
    if(p.use_one_slot_terms&&p.validated_two_rigid_blocks&&block_count==2
        &&broad.color_count==2&&broad.block_colors&&broad.color_offsets&&broad.color_blocks) {
        p.one_slot_color_metadata.reserve(2);
        validate_one_slot_colors<<<1,1>>>(broad,block_count,p.one_slot_color_metadata.data);
        checked(cudaGetLastError(),"one-slot integer color capability");
        int colors[2]={-1,-1};
        checked(cudaMemcpy(colors,p.one_slot_color_metadata.data,sizeof(colors),cudaMemcpyDeviceToHost),
            "one-slot two-color metadata");
        p.one_slot_graph_proven=colors[0]>=0&&colors[0]<2&&colors[1]>=0&&colors[1]<2&&colors[0]!=colors[1];
    }

    p.mapping_dirty=true;
    p.host_selected_offsets.clear();
    if((p.use_selected_reduce||p.use_compact_warp_reduce||p.use_component_reduce||p.use_prefetch_reduce)&&broad.color_count>0&&broad.color_blocks&&broad.color_offsets) {
        p.host_selected_offsets.resize(static_cast<std::size_t>(broad.color_count)+1);
        checked(cudaMemcpy(p.host_selected_offsets.data(),broad.color_offsets,
            p.host_selected_offsets.size()*sizeof(int),cudaMemcpyDeviceToHost),"selected node launch metadata");
        if(p.host_selected_offsets.front()!=0||p.host_selected_offsets.back()>block_count
            ||!std::is_sorted(p.host_selected_offsets.begin(),p.host_selected_offsets.end()))
            throw std::invalid_argument("GPU contact selected color offsets are invalid");
    }
    if(p.use_component_reduce&&!p.host_selected_offsets.empty()) {
        std::size_t maximum=0;
        for(int color=0;color<broad.color_count;++color)
            maximum=std::max(maximum,static_cast<std::size_t>(p.host_selected_offsets[color+1]-p.host_selected_offsets[color]));
        p.component_systems.reserve(std::min(maximum,static_cast<std::size_t>(p.component_max_color_blocks)));
    }
    const auto& q=pruning.parameters;
    p.pruning_ready=(p.use_box_pruning||pruning.original_geometry)&&broad.block_colors&&pruning.boxes&&pruning.deformable&&pruning.positions&&pruning.previous
        &&std::isfinite(q.d_hat)&&q.d_hat>0&&std::isfinite(q.k_barrier)&&q.k_barrier>=0
        &&std::isfinite(q.dt)&&q.dt>0&&std::isfinite(q.friction)&&q.friction>=0
        &&std::isfinite(q.eps_v)&&q.eps_v>0&&std::isfinite(q.eps)&&q.eps>0;
    // Original evaluation validates these scales even for zero friction. A
    // cache must not hide their original error path on an otherwise far pair.
    if(pruning.original_geometry)
        p.pruning_ready=p.pruning_ready&&q.d_hat>1e-8
            &&std::isfinite(q.dt*q.dt)&&std::isfinite((q.dt*q.dt)*q.k_barrier)
            &&std::isfinite(q.dt*q.eps_v)&&q.dt*q.eps_v>0.0;
    p.original_pruning_ready=p.pruning_ready&&pruning.original_geometry;
    if(p.pruning_ready&&broad.contact_count) {
        p.box_separated.reserve(broad.contact_count);
        p.pruning_d_hat=q.d_hat;
        if(p.original_pruning_ready)
            certify_original_boxes<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,pruning,p.box_separated.data);
        else
            certify_boxes<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,pruning,p.box_separated.data);
    }
    if(p.original_pruning_ready) {
        p.raw_block_counts.reserve(static_cast<std::size_t>(block_count));
        if(block_count)checked(cudaMemsetAsync(p.raw_block_counts.data,0,
            static_cast<std::size_t>(block_count)*sizeof(std::size_t)),"clear raw reference incidence counts");
        if(broad.contact_count)
            count_raw_incidents<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,block_count,p.raw_block_counts.data);
    }
    if(p.queue_log.is_open())++p.rebuild_index;
    bool dense=false;
    if(two_rigid_dense_csr::eligible(p.validated_two_rigid_blocks,
        p.use_two_rigid_dense_csr,p.pruning_ready,broad.contact_count,
        p.packet_capacity,block_count)) {
        const auto metadata_words=p.use_two_rigid_kind_packets?2u:1u;
        p.dense_unsupported.reserve(metadata_words);
        checked(cudaMemsetAsync(p.dense_unsupported.data,0,metadata_words*sizeof(unsigned)),"clear dense CSR preflight");
        if(p.use_two_rigid_kind_packets)
            inspect_two_rigid_dense_contacts<true><<<blocks(broad.contact_count),threads>>>(
                broad,vertex_block,p.validated_two_rigid_nodes,p.dense_unsupported.data);
        else
            inspect_two_rigid_dense_contacts<false><<<blocks(broad.contact_count),threads>>>(
                broad,vertex_block,p.validated_two_rigid_nodes,p.dense_unsupported.data);
        checked(cudaGetLastError(),"dense CSR integer preflight");
        unsigned metadata[2]{};
        checked(cudaMemcpy(metadata,p.dense_unsupported.data,metadata_words*sizeof(unsigned),
            cudaMemcpyDeviceToHost),"dense CSR preflight metadata");
        dense=metadata[0]==0;
        if(dense&&p.use_two_rigid_kind_packets)
            p.dense_point_triangle_end=metadata[1];
    }
    if(dense) {
        // Generic rows expose only the first2N entries; duplicate-owner
        // sentinels in their unconsumed4N tail are not part of this CSR.
        const auto count=2*broad.contact_count;
        p.incidents.release();
        if(two_rigid_dense_csr::release_generic_capacity(p.pair_indices_dense,
            p.pair_indices.capacity,count))p.pair_indices.release();
        p.pair_indices.reserve(count);p.offsets.reserve(3);
    } else {
        p.incidents.reserve(4*broad.contact_count);p.pair_indices.reserve(4*broad.contact_count);p.offsets.reserve(static_cast<std::size_t>(block_count)+1);
    }
    p.pair_indices_dense=dense;
    const auto packets=std::min(broad.contact_count,p.packet_capacity);
    p.outputs.reserve(packets);p.active.reserve(packets);
    p.exact_count.reserve(1);p.exact_retry_count.reserve(1);p.exact_next.reserve(1);
    if(dense) {
        emit_two_rigid_dense_rows<<<blocks(broad.contact_count),threads>>>(
            broad.contact_count,p.pair_indices.data,p.offsets.data);
    } else {
    if(broad.contact_count) {
        emit_incidents<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,p.incidents.data,p.pruning_ready?p.box_separated.data:nullptr);
        thrust::sort(thrust::device_pointer_cast(p.incidents.data),thrust::device_pointer_cast(p.incidents.data+4*broad.contact_count),IncidentOrder{broad.contacts});
        incident_indices<<<blocks(4*broad.contact_count),threads>>>(p.incidents.data,4*broad.contact_count,p.pair_indices.data);
    }
    incident_offsets<<<blocks(static_cast<std::size_t>(block_count)+1),threads>>>(p.incidents.data,4*broad.contact_count,block_count,p.offsets.data);
    }
    checked(cudaGetLastError(),"incident CSR kernels");
    p.host_color_offsets.clear();
    p.max_color_pairs=0;
    const bool omit_color_pairs=p.use_color_pairs&&two_rigid_color_queue::omit_color_pairs(
        p.validated_two_rigid_blocks,p.omit_two_rigid_color_queues,
        p.color_pairs_explicitly_forced,broad.contact_count,p.packet_capacity,
        block_count,broad.color_count);
    if(omit_color_pairs) {
        // Every candidate in this validated topology spans both body blocks.
        // Each proper color row therefore contains all N>capacity candidates;
        // no color-local packet path can use these redundant queues. Clear
        // retained small-epoch buffers before publishing this empty binding.
        p.color_keys.release();p.color_pairs.release();p.color_offsets.release();
    }
    if(p.use_color_pairs&&!omit_color_pairs&&broad.color_count>0&&broad.block_colors) {
        const auto count=4*broad.contact_count;
        p.color_keys.reserve(count);p.color_pairs.reserve(count);
        p.color_offsets.reserve(static_cast<std::size_t>(broad.color_count)+1);
        if(count) {
            emit_color_pairs<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,p.color_keys.data,p.pruning_ready?p.box_separated.data:nullptr);
            thrust::sort(thrust::device_pointer_cast(p.color_keys.data),thrust::device_pointer_cast(p.color_keys.data+count));
            color_pair_indices<<<blocks(count),threads>>>(p.color_keys.data,count,p.color_pairs.data);
        }
        color_pair_offsets<<<blocks(static_cast<std::size_t>(broad.color_count)+1),threads>>>(p.color_keys.data,count,broad.color_count,p.color_offsets.data);
        checked(cudaGetLastError(),"color candidate worklist kernels");
        p.host_color_offsets.resize(static_cast<std::size_t>(broad.color_count)+1);
        checked(cudaMemcpy(p.host_color_offsets.data(),p.color_offsets.data,p.host_color_offsets.size()*sizeof(std::size_t),cudaMemcpyDeviceToHost),"color worklist launch metadata");
        p.max_color_pairs=0;
        for(int color=0;color<broad.color_count;++color)
            p.max_color_pairs=std::max(p.max_color_pairs,p.host_color_offsets[color+1]-p.host_color_offsets[color]);
    }
    p.color_launch_binding.publish(p.broad,p.color_worklist(),p.host_color_offsets,
        p.use_current_color_ccd_launch);
    p.packet_color_binding.publish(p.broad,p.color_worklist(),p.host_color_offsets,
        p.use_color_local_packets);
    if(broad.contact_count>p.packet_capacity) {
        p.incidents.release();p.color_keys.release();
    }
}
ContactIncidence GpuContactPipeline::incidence()const {
    return {impl_->offsets.data,impl_->pair_indices.data,impl_->color_worklist(),
        impl_->original_pruning_ready?impl_->raw_block_counts.data:nullptr};
}
ContactIncidence GpuContactPipeline::incidence(int color)const {
    auto result=incidence();
    result.color_pairs=impl_->color_launch_binding.for_color(impl_->broad,result.color_pairs,color);
    return result;
}
void GpuContactPipeline::assemble_color(const DeformableScene& scene,const DeformableState& state,
    const ElasticWorkspace& elastic,const RigidScene& rigid,const RigidState& rigid_state,
    const RigidWorkspace& rigid_workspace,const SdfScene& sdf,const ContactParameters& parameters,
    int color,ContactAssemblyPhase phase,ContactAssemblyMode mode,const RigidPinAssembly& pins,
    ContactGeometryPolicy geometry) {
    GpuProfileScope profile("contact");
    auto& p=*impl_;
    p.active_record_slots=nullptr;p.pending_contacts=false;
    const bool original=geometry==ContactGeometryPolicy::Original;
    // A robust rebuild may have filtered incidence/color-pair lists with the
    // robust force certificate. Original geometry must never consume that
    // stale certificate or the already-omitted incidence; the caller must
    // rebuild the SAME graph with pruning disabled first.
    if(p.pruning_ready&&original!=p.original_pruning_ready)
        throw std::invalid_argument(
            "GPU contact geometry requires a rebuild with the matching cache policy");
    if(p.queue_log.is_open())++p.assembly_index;
    if(!p.offsets.data||!elastic.error)throw std::logic_error("GPU contact assembly requires rebuild and device error pointer");
    if(color<0||(p.broad.color_count>0&&color>=p.broad.color_count))throw std::invalid_argument("GPU contact color out of range");
    const bool frozen_snapshot=mode==ContactAssemblyMode::FrozenSnapshot;
    if(p.pruning_ready&&p.broad.contact_count) {
        const auto& q=parameters;
        if(!std::isfinite(q.d_hat)||q.d_hat>p.pruning_d_hat||q.d_hat<=0
            ||!std::isfinite(q.k_barrier)||q.k_barrier<0||!std::isfinite(q.dt)||q.dt<=0
            ||!std::isfinite(q.friction)||q.friction<0||!std::isfinite(q.eps_v)||q.eps_v<=0
            ||!std::isfinite(q.eps)||q.eps<=0)
            throw std::invalid_argument("contact box certificates require valid parameters and rebuild before increasing d_hat");
        if(p.original_pruning_ready) {
            const double dt2=q.dt*q.dt,eps_u=q.dt*q.eps_v;
            if(!std::isfinite(dt2)||!std::isfinite(dt2*q.k_barrier)
                ||!std::isfinite(eps_u)||!(eps_u>0.0))
                throw std::invalid_argument("original contact cache requires finite force and friction scales");
        }
    }
    // A synchronous Jacobi pair contributes to every selected vertex. The
    // one-active-role compact packet belongs only to independent GS colors.
    bool color_local=false;
    std::size_t current_color_pairs=0;
    if(p.use_color_local_packets&&p.broad.contact_count>p.packet_capacity
        &&!p.host_color_offsets.empty()) {
        const auto binding=p.packet_color_binding.for_color(p.broad,p.color_worklist(),color);
        current_color_pairs=binding.max_color_pairs;
        color_local=binding.offsets&&current_color_pairs<=p.packet_capacity;
    }
    const bool tiled=p.broad.contact_count>p.packet_capacity&&!color_local;
    const bool compact_phase=!tiled&&!frozen_snapshot&&p.use_compact&&phase==ContactAssemblyPhase::Deformable;
    const bool need_exact=!original||frozen_snapshot;
    p.prepare_packet_storage(need_exact,!tiled&&!original&&p.use_cooperative_exact,compact_phase);
    if(color_local) {
        p.record_slots.reserve(p.broad.contact_count);
        checked(cudaMemsetAsync(p.record_slots.data,0xff,p.broad.contact_count*sizeof(int)),"invalidate color-local record slots");
        if(current_color_pairs)bind_color_record_slots<<<blocks(current_color_pairs),threads>>>(
            p.broad,p.color_worklist(),color,p.packet_capacity,p.record_slots.data,elastic.error);
        p.active_record_slots=p.record_slots.data;p.pending_contacts=true;
        checked(cudaGetLastError(),"bind color-local record slots");
    }
    const bool cached_rigid=p.use_rigid_pullback_cache
        &&phase!=ContactAssemblyPhase::Deformable&&rigid.bodies;
    const bool one_slot_terms=rigid_one_slot::eligible(p.validated_two_rigid_blocks,
        p.use_one_slot_terms,p.pair_indices_dense,p.pruning_ready,p.one_slot_graph_proven,
        original,frozen_snapshot,tiled,cached_rigid,phase!=ContactAssemblyPhase::Deformable,
        rigid.bodies,scene.nodes,p.validated_two_rigid_nodes,scene.node_blocks==p.vertex_block);
    if(cached_rigid) {
        const auto packets=std::min(p.broad.contact_count,p.packet_capacity);
        const auto required=packets*(one_slot_terms?1u:4u);
        if(rigid_one_slot::release_on_transition(p.rigid_terms_one_slot_layout,one_slot_terms,
            p.rigid_terms.capacity,required))p.rigid_terms.release();
        p.rigid_terms.reserve(required);p.rigid_terms_one_slot_layout=one_slot_terms;
    }
    if(tiled) {
        GpuProfileScope tiled_profile("bounded_packets");
        const bool orientation=phase==ContactAssemblyPhase::RigidOrientation;
        const auto system_count=phase==ContactAssemblyPhase::Deformable?scene.nodes:rigid.bodies;
        const bool compact_rigid=p.use_rigid_compact_reduce&&phase!=ContactAssemblyPhase::Deformable
            &&rigid.bodies>0&&rigid.bodies<=rigid_compact_body_limit;
        const auto chunk_stride=compact_rigid?p.prepare_rigid_compact_storage(rigid.bodies):0;
        const bool sparse_original_gather=p.use_original_sparse_gather&&one_slot_terms
            &&p.use_output_activity&&original&&!frozen_snapshot
            &&phase!=ContactAssemblyPhase::Deformable&&parameters.friction==0.0;
        if(sparse_original_gather) {
            p.sparse_gather_queue.reserve(std::min(p.broad.contact_count,p.packet_capacity));
            p.sparse_gather_count.reserve(1);
        }
        p.pending_systems.reserve(system_count);
        if(orientation&&rigid.bodies) {
            p.body_kinematics.reserve(rigid.bodies);p.node_pullbacks.reserve(scene.nodes);
            p.pullback_needed.reserve(scene.nodes);
            if(scene.nodes)checked(cudaMemsetAsync(p.pullback_needed.data,0,
                scene.nodes*sizeof(unsigned)),"clear tiled pullback marks");
            prepare_body_kinematics<<<blocks(rigid.bodies),threads>>>(p.broad,rigid,rigid_state,
                p.body_kinematics.data,parameters.dt,color,elastic.error);
            if(scene.nodes&&sdf.parameters.stiffness>0)
                prepare_node_pullbacks<<<blocks(scene.nodes),threads>>>(p.broad,scene,rigid,
                    p.body_kinematics.data,p.pullback_needed.data,true,p.node_pullbacks.data,color,elastic.error);
        }
        const Work work=p.work(false,one_slot_terms);
        if(system_count) {
            if(phase==ContactAssemblyPhase::Deformable)
                seed_pending_deformable<<<blocks(system_count),threads>>>(p.broad,scene,state,elastic,
                    sdf,parameters,work,color,frozen_snapshot,p.pending_systems.data);
            else seed_pending_rigid<<<blocks(system_count),threads>>>(p.broad,state,rigid,rigid_state,
                rigid_workspace,elastic,sdf,parameters,work,color,orientation,pins,frozen_snapshot,p.pending_systems.data);
        }
        unsigned long long exact_total=0,retry_total=0;
        // CSR rows are ordered by contact kind, then global pair index.
        // Packet reuse never changes that order or publishes a partial assembly.
        for(int kind=0;kind<2;++kind) {
        const auto span=two_rigid_dense_csr::kind_packet_span(
            original&&!frozen_snapshot&&p.pair_indices_dense&&p.use_two_rigid_kind_packets,
            p.broad.contact_count,p.packet_capacity,p.dense_point_triangle_end,kind);
        for(std::size_t first=span.first;first<span.last;) {
            const auto last=first+std::min(p.packet_capacity,span.last-first);
            checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned)),"clear tiled exact queue");
            checked(cudaMemsetAsync(p.exact_retry_count.data,0,sizeof(unsigned)),"clear tiled retry queue");
            checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear tiled exact cursor");
            const auto* separated=p.pruning_ready?p.box_separated.data:nullptr;
            if(sparse_original_gather) {
                checked(cudaMemsetAsync(p.sparse_gather_count.data,0,sizeof(unsigned)),"clear sparse gather count");
                gather_tile_original_sparse_probe<<<blocks(last-first),threads>>>(p.broad,scene,state,rigid,
                    parameters,work,color,phase,elastic.error,separated,first,last,kind,
                    p.sparse_gather_queue.data,p.sparse_gather_count.data);
                const auto worker_blocks=p.sparse_gather_blocks(last-first);
                if(p.use_original_pt_translation_zero&&phase==ContactAssemblyPhase::RigidTranslation&&kind==0)
                    gather_tile_original_sparse_full<true><<<worker_blocks,threads>>>(p.broad,scene,state,rigid,
                        parameters,work,color,phase,elastic.error,separated,first,last,kind,
                        p.sparse_gather_queue.data,p.sparse_gather_count.data);
                else gather_tile_original_sparse_full<false><<<worker_blocks,threads>>>(p.broad,scene,state,rigid,
                    parameters,work,color,phase,elastic.error,separated,first,last,kind,
                    p.sparse_gather_queue.data,p.sparse_gather_count.data);
            } else if(original&&p.use_original_pt_translation_zero&&!frozen_snapshot
                &&phase==ContactAssemblyPhase::RigidTranslation&&kind==0&&parameters.friction==0.0)
                gather_tile_original_pt_translation_zero<<<blocks(last-first),threads>>>(p.broad,scene,state,rigid,parameters,
                    work,color,phase,elastic.error,separated,first,last,kind,frozen_snapshot);
            else if(original)
                gather_tile<true><<<blocks(last-first),threads>>>(p.broad,scene,state,rigid,parameters,
                    work,color,phase,elastic.error,separated,first,last,kind,frozen_snapshot);
            else gather_tile<false><<<blocks(last-first),threads>>>(p.broad,scene,state,rigid,parameters,
                    work,color,phase,elastic.error,separated,first,last,kind,frozen_snapshot);
            if(need_exact) {
                evaluate_exact_queue<exact_small_limbs,false><<<p.schedule.small_workers/p.schedule.small_threads,p.schedule.small_threads>>>(work,elastic.error);
                checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear tiled retry cursor");
                evaluate_exact_queue<exact_large_limbs,true><<<p.schedule.large_workers/p.schedule.large_threads,p.schedule.large_threads>>>(work,elastic.error);
            }
            if(p.queue_log.is_open()) {
                unsigned count=0,retries=0;
                checked(cudaMemcpy(&count,p.exact_count.data,sizeof(count),cudaMemcpyDeviceToHost),"tiled diagnostic exact count");
                checked(cudaMemcpy(&retries,p.exact_retry_count.data,sizeof(retries),cudaMemcpyDeviceToHost),"tiled diagnostic retry count");
                exact_total+=count;retry_total+=retries;
            }
            if(orientation&&rigid.bodies&&scene.nodes) {
                mark_tile_pullbacks<<<blocks(last-first),threads>>>(p.broad,scene,rigid,work,
                    p.pullback_needed.data,color,elastic.error,first,last,kind);
                prepare_node_pullbacks<<<blocks(scene.nodes),threads>>>(p.broad,scene,rigid,
                    p.body_kinematics.data,p.pullback_needed.data,false,p.node_pullbacks.data,color,elastic.error);
            }
            if(cached_rigid) {
                prepare_rigid_contact_terms<<<blocks(last-first),threads>>>(p.broad,rigid,
                    work,color,orientation,elastic.error,first,last,kind,{});
            }
            if(system_count) {
                if(phase==ContactAssemblyPhase::Deformable)
                    reduce_deformable_tile<<<blocks(system_count),threads>>>(p.broad,scene,elastic,parameters,
                        work,color,first,last,kind,p.pending_systems.data);
                else if(compact_rigid) {
                    const auto tile_chunks=(last-first)/threads+((last-first)%threads!=0);
                    const auto groups=rigid.bodies*tile_chunks;
                    classify_rigid_tile_chunks<<<static_cast<unsigned>(groups),threads>>>(p.broad,rigid,
                        work,color,orientation,first,last,kind,chunk_stride,tile_chunks,
                        p.rigid_kept_incidents.data,p.rigid_chunk_counts.data,elastic.error);
                    reduce_rigid_tile_compact<<<blocks(system_count),threads>>>(p.broad,rigid,elastic,parameters,
                        work,color,orientation,first,last,kind,p.rigid_kept_incidents.data,
                        p.rigid_chunk_counts.data,chunk_stride,p.pending_systems.data);
                } else reduce_rigid_tile<<<blocks(system_count),threads>>>(p.broad,rigid,elastic,parameters,
                    work,color,orientation,first,last,kind,p.pending_systems.data);
            }
            first=last;
        }
        }
        if(system_count) {
            if(phase==ContactAssemblyPhase::Deformable)
                finish_pending_deformable<<<blocks(system_count),threads>>>(p.broad,scene,state,elastic,
                    sdf,parameters,work,color,frozen_snapshot,p.pending_systems.data);
            else finish_pending_rigid<<<blocks(system_count),threads>>>(p.broad,state,rigid,elastic,sdf,
                parameters,work,color,orientation,pins.scene.count!=0,frozen_snapshot,p.pending_systems.data);
            publish_pending<<<blocks(system_count),threads>>>(p.broad,scene,rigid,elastic,rigid_workspace,
                color,phase,p.pending_systems.data);
        }
        checked(cudaGetLastError(),"tiled assembly kernels");
        if(p.queue_log.is_open()) {
            int status=0;checked(cudaMemcpy(&status,elastic.error,sizeof(status),cudaMemcpyDeviceToHost),"tiled diagnostic status");
            const char* phase_name=phase==ContactAssemblyPhase::Deformable?"deformable":
                phase==ContactAssemblyPhase::RigidTranslation?"rigid_translation":"rigid_orientation";
            p.queue_log<<p.rebuild_index<<','<<p.assembly_index<<','<<color<<','<<phase_name<<','<<p.broad.contact_count<<','
                <<exact_total<<','<<retry_total<<','<<status<<','<<p.schedule.small_workers<<','<<p.schedule.small_threads<<','
                <<p.schedule.large_workers<<','<<p.schedule.large_threads<<'\n';
            p.queue_log.flush();
            if(!p.queue_log)throw std::runtime_error("failed writing tiled contact diagnostic");
        }
        return;
    }
    // Above-cap colors retain the original packet path's seed/contact/finish/
    // publish transactions. Only the contact kernels use the recovered routes.
    if(color_local) {
        const bool orientation=phase==ContactAssemblyPhase::RigidOrientation;
        const auto count=phase==ContactAssemblyPhase::Deformable?scene.nodes:rigid.bodies;
        p.pending_systems.reserve(count);
        if(orientation&&rigid.bodies) {
            p.body_kinematics.reserve(rigid.bodies);p.node_pullbacks.reserve(scene.nodes);
            p.pullback_needed.reserve(scene.nodes);
            if(scene.nodes)checked(cudaMemsetAsync(p.pullback_needed.data,0,scene.nodes*sizeof(unsigned)),"clear color-local pullback marks");
            prepare_body_kinematics<<<blocks(rigid.bodies),threads>>>(p.broad,rigid,rigid_state,
                p.body_kinematics.data,parameters.dt,color,elastic.error);
            if(scene.nodes&&sdf.parameters.stiffness>0)
                prepare_node_pullbacks<<<blocks(scene.nodes),threads>>>(p.broad,scene,rigid,
                    p.body_kinematics.data,p.pullback_needed.data,true,p.node_pullbacks.data,color,elastic.error);
        }
        if(count) {
            if(phase==ContactAssemblyPhase::Deformable)
                seed_pending_deformable<<<blocks(count),threads>>>(p.broad,scene,state,elastic,sdf,
                    parameters,p.work(),color,frozen_snapshot,p.pending_systems.data);
            else seed_pending_rigid<<<blocks(count),threads>>>(p.broad,state,rigid,rigid_state,
                rigid_workspace,elastic,sdf,parameters,p.work(),color,orientation,pins,frozen_snapshot,p.pending_systems.data);
        }
        // The original tiled gather rejects malformed kinds even on global
        // pairs absent from this color's worklist. Keep that direct-API check
        // after the original seed stage, before any contact arithmetic.
        if(p.broad.contact_count)validate_color_local_contact_kinds<<<blocks(p.broad.contact_count),threads>>>(
            p.broad,parameters,elastic.error);
    }
    const bool warp_ready=(!color_local||phase==ContactAssemblyPhase::Deformable)
        &&!frozen_snapshot&&p.use_warp_reduce&&!compact_phase&&p.broad.color_blocks&&p.broad.color_offsets;
    const bool selected_ready=!frozen_snapshot&&phase==ContactAssemblyPhase::Deformable&&!p.host_selected_offsets.empty()
        &&(p.use_selected_reduce||(compact_phase&&p.use_compact_warp_reduce));
    const bool compact_warp_ready=selected_ready&&compact_phase&&p.use_compact_warp_reduce&&p.broad.contact_count;
    const auto component_selected_count=p.host_selected_offsets.empty()?0:
        static_cast<std::size_t>(p.host_selected_offsets[color+1]-p.host_selected_offsets[color]);
    const bool component_ready=!frozen_snapshot&&p.use_component_reduce&&phase==ContactAssemblyPhase::Deformable
        &&p.broad.contact_count&&component_selected_count
        &&component_selected_count<=static_cast<std::size_t>(p.component_max_color_blocks);
    const bool prefetch_ready=!frozen_snapshot&&p.use_prefetch_reduce&&phase==ContactAssemblyPhase::Deformable
        &&!p.host_selected_offsets.empty();
    if((warp_ready||selected_ready||component_ready||prefetch_ready)&&(p.mapping_dirty||p.mapped_nodes!=scene.nodes
        ||p.mapped_node_blocks!=scene.node_blocks||p.mapped_deformable!=scene.deformable)) {
        p.block_to_deformable_node.reserve(p.block_count);
        if(p.block_count)checked(cudaMemsetAsync(p.block_to_deformable_node.data,0xff,p.block_count*sizeof(int)),"clear deformable block map");
        if(scene.nodes)map_deformable_blocks<<<blocks(scene.nodes),threads>>>(scene,p.block_count,p.block_to_deformable_node.data,elastic.error);
        p.mapping_dirty=false;
        p.mapped_nodes=scene.nodes;p.mapped_node_blocks=scene.node_blocks;p.mapped_deformable=scene.deformable;
    }
    {
    GpuProfileScope reset_profile("queue_reset");
    checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned)),"clear exact queue");
    }
    const auto gather_count=p.color_worklist().offsets?p.host_color_offsets[color+1]-p.host_color_offsets[color]:p.broad.contact_count;
    bool compact_queue_swap=false;
    unsigned original_exact_count_for_diagnostic=0;
    if(p.broad.contact_count) {
        {
        GpuProfileScope fast_profile("fused_gather_fast_compute");
        checked(cudaMemsetAsync(p.exact_retry_count.data,0,sizeof(unsigned)),"clear exact retry queue");
        checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear exact cursor");
        if(gather_count) {
            if(original) {
                if(p.use_original_self_zero&&compact_phase&&!frozen_snapshot&&parameters.friction==0.0) {
                    gather_original<true,true><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist(),false);
                    gather_original_self_zero<<<blocks(gather_count),threads>>>(p.broad,scene,state,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                } else if(p.defer_inputs||!frozen_snapshot)
                    gather_original<true><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist(),frozen_snapshot);
                else
                    gather_original<false><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist(),frozen_snapshot);
            } else if(p.use_selected_ordinary&&compact_phase&&parameters.friction==0.0) {
                if(p.defer_inputs) {
                    gather_general_complement<true><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                    gather_selected_ordinary<true><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                } else {
                    gather_general_complement<false><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                    gather_selected_ordinary<false><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                }
            } else {
                if(p.defer_inputs)
                    gather_compute<true><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
                else
                    gather_compute<false><<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
            }
        }
        }
        if(!original || frozen_snapshot) {
        Work exact_work=p.work(compact_phase);
        if(!original&&p.use_compact_point&&p.use_cooperative_exact) {
            GpuProfileScope compact_profile("exact_compact_point_first");
            // Both original arena reservations remain. Reuse cooperative
            // scratch only across whole ordered default-stream launches.
            if(p.use_cooperative_compact_point&&p.compact_point_ratio) {
                launch_cooperative_compact_point_inplace_study(p.inputs.data,p.outputs.data,p.exact_queue.data,p.exact_count.data,
                    p.exact_retry_queue.data,p.exact_retry_count.data,p.exact_next.data,
                    p.cooperative_words.data,p.cooperative_slots.data,p.cooperative_workers,elastic.error,nullptr,
                    compact_phase?p.compact.data:nullptr,p.use_output_activity?p.active.data:nullptr);
            } else {
                launch_compact_point_contacts(p.inputs.data,p.outputs.data,p.exact_queue.data,p.exact_count.data,
                    p.exact_retry_queue.data,p.exact_retry_count.data,p.exact_next.data,
                    p.exact_words.data,p.exact_free_slots.data,p.schedule.small_workers,p.schedule.small_threads,
                    elastic.error,nullptr,compact_phase?p.compact.data:nullptr,
                    p.use_output_activity?p.active.data:nullptr,p.compact_point_ratio);
            }
            checked(cudaGetLastError(),"compact point first-tier kernel");
            // This optional read belongs ONLY to the existing explicit queue
            // diagnostic mode. Production does not read any per-color count.
            if(p.queue_log.is_open())
                checked(cudaMemcpy(&original_exact_count_for_diagnostic,p.exact_count.data,sizeof(unsigned),cudaMemcpyDeviceToHost),"diagnostic original compact-tier exact count");
            // A is dead after this ordered launch. B contains unresolved R;
            // cooperative256 consumes B and appends true wide retries to A.
            checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned)),"clear compact-tier wide retry queue");
            checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear compact-tier fallback cursor");
            std::swap(exact_work.exact_queue,exact_work.exact_retry_queue);
            std::swap(exact_work.exact_count,exact_work.exact_retry_count);
            compact_queue_swap=true;
        }
        {
        GpuProfileScope exact_profile("exact_small");
        if(!original&&p.use_cooperative_exact)
            launch_cooperative_exact_contacts(p.inputs.data,p.outputs.data,exact_work.exact_queue,exact_work.exact_count,
                exact_work.exact_retry_queue,exact_work.exact_retry_count,p.exact_next.data,
                p.cooperative_words.data,p.cooperative_slots.data,p.cooperative_workers,elastic.error,nullptr,
                compact_phase?p.compact.data:nullptr,cooperative_geometry_conversions_enabled(),
                p.use_output_activity?p.active.data:nullptr);
        else evaluate_exact_queue<exact_small_limbs,false><<<p.schedule.small_workers/p.schedule.small_threads,p.schedule.small_threads>>>(exact_work,elastic.error);
        }
        GpuProfileScope retry_profile("exact_large_retry");
        checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear exact retry cursor");
        evaluate_exact_queue<exact_large_limbs,true><<<p.schedule.large_workers/p.schedule.large_threads,p.schedule.large_threads>>>(exact_work,elastic.error);
        }
    }
    if(p.queue_log.is_open()) {
        GpuProfileScope diagnostic_profile("queue_diagnostic_readback");
        p.record_queues(color,phase,elastic.error,compact_queue_swap,original_exact_count_for_diagnostic);
    }
    if(phase==ContactAssemblyPhase::RigidOrientation&&rigid.bodies) {
        GpuProfileScope pullback_profile("rigid_pullback");
        // Scope reuse to this immutable orientation assembly. Recompute after
        // every COM/orientation update; never reuse another body's/time's J.
        p.body_kinematics.reserve(rigid.bodies);p.node_pullbacks.reserve(scene.nodes);p.pullback_needed.reserve(scene.nodes);
        if(scene.nodes)checked(cudaMemsetAsync(p.pullback_needed.data,0,scene.nodes*sizeof(unsigned)),"clear pullback marks");
        if(gather_count)mark_pullback_nodes<<<blocks(gather_count),threads>>>(p.broad,scene,rigid,p.work(),p.pullback_needed.data,color,elastic.error,p.color_worklist());
        prepare_body_kinematics<<<blocks(rigid.bodies),threads>>>(p.broad,rigid,rigid_state,
            p.body_kinematics.data,parameters.dt,color,elastic.error);
        if(scene.nodes)prepare_node_pullbacks<<<blocks(scene.nodes),threads>>>(p.broad,scene,rigid,
            p.body_kinematics.data,p.pullback_needed.data,sdf.parameters.stiffness>0,p.node_pullbacks.data,color,elastic.error);
    }
    if(cached_rigid&&gather_count) {
        prepare_rigid_contact_terms<<<blocks(gather_count),threads>>>(p.broad,rigid,
            p.work(),color,phase==ContactAssemblyPhase::RigidOrientation,elastic.error,
            0,p.broad.contact_count,-1,p.color_worklist());
    }
    ElasticWorkspace contact_elastic=elastic;
    RigidWorkspace contact_rigid=rigid_workspace;
    if(color_local) {
        if(phase==ContactAssemblyPhase::Deformable)contact_elastic.systems=p.pending_systems.data;
        else contact_rigid.systems=p.pending_systems.data;
    }
    GpuProfileScope reduce_profile("reduce_sdf");
    if(phase==ContactAssemblyPhase::Deformable) {
        if(scene.nodes) {
            if(prefetch_ready) {
                // Prefetch is an alternative to the component-split candidate;
                // when explicitly enabled it takes precedence for this phase.
                if(component_selected_count)
                    reduce_deformable_prefetch_warp<<<static_cast<unsigned>(std::min<std::size_t>(component_selected_count,256)),32>>>(
                        p.broad,scene,state,contact_elastic,sdf,parameters,p.work(compact_phase),color,
                        p.block_to_deformable_node.data,p.block_count);
            } else if(component_ready) {
                // More blocks only for heavier nodes in colors with limited
                // parallel work. Each block owns complete scalar chains.
                const auto tasks=component_selected_count*static_cast<std::size_t>(p.component_groups);
                const auto chain_blocks=static_cast<unsigned>(std::min<std::size_t>(tasks,65535));
                reduce_deformable_component_chains<<<chain_blocks,32>>>(p.broad,scene,contact_elastic,parameters,
                    p.work(compact_phase),color,p.block_to_deformable_node.data,p.block_count,
                    static_cast<std::size_t>(p.component_grain),static_cast<unsigned>(p.component_groups),p.component_systems.data);
                // Ordered default-stream completion publishes every pending
                // component before this once-per-node SDF/check/commit stage.
                const auto final_blocks=static_cast<unsigned>((component_selected_count+p.selected_threads-1)/p.selected_threads);
                finalize_deformable_component_chains<<<final_blocks,p.selected_threads>>>(p.broad,scene,state,contact_elastic,sdf,
                    parameters,p.work(compact_phase),color,p.block_to_deformable_node.data,p.block_count,
                    static_cast<std::size_t>(p.component_grain),static_cast<unsigned>(p.component_groups),p.component_systems.data);
            } else if(selected_ready) {
                const auto selected_count=static_cast<std::size_t>(p.host_selected_offsets[color+1]-p.host_selected_offsets[color]);
                if(selected_count) {
                    // Small blocks spread narrow colors across SMs; this
                    // opt-in launch knob does not change node arithmetic.
                    const auto selected_blocks=static_cast<unsigned>((selected_count+p.selected_threads-1)/p.selected_threads);
                    reduce_deformable_selected<<<selected_blocks,p.selected_threads>>>(p.broad,scene,state,contact_elastic,sdf,
                        parameters,p.work(compact_phase),color,p.block_to_deformable_node.data,p.block_count,
                        compact_warp_ready?static_cast<std::size_t>(p.compact_warp_degree_cutoff):0);
                    if(compact_warp_ready)
                        reduce_deformable_compact_warp<<<static_cast<unsigned>(std::min<std::size_t>(selected_count,256)),32>>>(
                            p.broad,scene,state,contact_elastic,sdf,parameters,p.work(true),color,p.block_to_deformable_node.data,
                            p.block_count,static_cast<std::size_t>(p.compact_warp_degree_cutoff));
                }
            } else if(warp_ready)
                reduce_deformable_warp<<<std::min(p.block_count,256),32>>>(p.broad,scene,state,contact_elastic,sdf,parameters,p.work(),color,p.block_to_deformable_node.data);
            else reduce_deformable<<<blocks(scene.nodes),threads>>>(p.broad,scene,state,contact_elastic,sdf,parameters,p.work(compact_phase),color,frozen_snapshot);
        }
    } else if(rigid.bodies)reduce_rigid<<<blocks(rigid.bodies),threads>>>(p.broad,state,rigid,rigid_state,contact_rigid,contact_elastic,sdf,parameters,p.work(),color,phase==ContactAssemblyPhase::RigidOrientation,pins,frozen_snapshot);
    if(color_local) {
        const auto count=phase==ContactAssemblyPhase::Deformable?scene.nodes:rigid.bodies;
        if(count) {
            if(phase==ContactAssemblyPhase::Deformable)
                finish_pending_deformable<<<blocks(count),threads>>>(p.broad,scene,state,elastic,
                    sdf,parameters,p.work(),color,frozen_snapshot,p.pending_systems.data);
            else finish_pending_rigid<<<blocks(count),threads>>>(p.broad,state,rigid,elastic,sdf,
                parameters,p.work(),color,phase==ContactAssemblyPhase::RigidOrientation,
                pins.scene.count!=0,frozen_snapshot,p.pending_systems.data);
            publish_pending<<<blocks(count),threads>>>(p.broad,scene,rigid,elastic,rigid_workspace,
                color,phase,p.pending_systems.data);
        }
    }
    checked(cudaGetLastError(),"assembly kernels");
}

} // namespace ipc_gpu::solver

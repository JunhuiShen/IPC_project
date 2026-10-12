#include "resident_contact_pipeline.h"
#include "resident_profile.h"
#include "resident_contact_schedule.h"
#include "resident_contact_exact.cuh"
#include "resident_contact_cooperative.h"
#include "resident_contact_execution.h"
#include "resident_box_certificate.cuh"
#include "resident_rigid.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>

namespace ipc_gpu::resident {
namespace {
constexpr int threads=128;
// Independent bounded arenas, spread over many SMs rather than one serial
// device worker. The larger-capacity retry is a second ordered kernel so its
// scratch can reuse the first pass's allocation (800 MiB default maximum).
constexpr int exact_slots=512;
constexpr int exact_small_limbs=256;
constexpr int exact_large_limbs=1600;
void checked(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string("Resident contact ")+operation+": "+cudaGetErrorString(error));
}
unsigned blocks(std::size_t count){return static_cast<unsigned>((count+threads-1)/threads);}
int selected_reduce_threads() {
    const int value=ExactContactSchedule::environment_integer("IPC_GPU_CONTACT_SELECTED_NODE_THREADS",8,128);
    if((value&(value-1))!=0)
        throw std::invalid_argument("IPC_GPU_CONTACT_SELECTED_NODE_THREADS must be one of 1,2,4,8,16,32,64,128");
    return value;
}
template<class T> class Buffer {
public:
    T* data=nullptr;std::size_t capacity=0;
    ~Buffer(){cudaFree(data);}
    void reserve(std::size_t count) {
        if(count<=capacity)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("resident contact allocation overflow");
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
struct Work {
    ContactInput* inputs;
    ContactOutput* outputs;
    CompactDeformableContact* compact;
    unsigned char* active;
    int* exact_queue;
    unsigned* exact_count;
    int* exact_retry_queue;
    unsigned* exact_retry_count;
    unsigned* exact_next;
    std::uint32_t* exact_words;
    int* exact_free_slots;
    const Incident* incidents;
    const std::size_t* offsets;
    const int* vertex_block;
    const unsigned char* surface_query;
    const NodePullback* node_pullbacks;
};
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
__global__ void gather_compute(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    RigidScene rigid,ContactParameters params,Work work,int color,ContactAssemblyPhase phase,int* failed,
    ColorContactWorklist color_pairs) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:broad.contact_count;
    if(index>=end-begin)return;
    const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
    work.active[pair]=0;
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(color_pairs.box_separated&&color_pairs.box_separated[pair])return;
    const auto& contact=broad.contacts[pair];
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return;
    auto& in=work.inputs[pair];
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
    work.active[pair]=1;
    const auto result=evaluate_contact(in);
    store_contact_output(in,result,work.outputs+pair,work.compact?work.compact+pair:nullptr);
    const auto status=result.status;
    if(status==ContactStatus::NeedsExactGeometry||status==ContactStatus::ZeroDistance||status==ContactStatus::NonFiniteDerivative) {
        const unsigned index=atomicAdd(work.exact_count,1u);work.exact_queue[index]=static_cast<int>(pair);
    } else if(status!=ContactStatus::Ok)error(failed,code(status));
}
template<int Limbs,bool Retry>
__global__ void evaluate_exact_queue(Work work,int* failed) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned count=Retry?*work.exact_retry_count:*work.exact_count;
    if(worker>=count||atomicAdd(failed,0))return;
    const int* queue=Retry?work.exact_retry_queue:work.exact_queue;
    resident_exact::DevicePool pool;
    resident_exact::initialize_pool(pool,
        work.exact_words+std::size_t(worker)*Limbs*exact_slots,
        work.exact_free_slots+std::size_t(worker)*exact_slots,Limbs,exact_slots);
    // A queue cursor balances heterogeneous exact queries. Each pair writes
    // only its own output; later per-node reductions retain their old order.
    for(;;) {
        if(atomicAdd(failed,0))break;
        const unsigned index=atomicAdd(work.exact_next,1u);
        if(index>=count)break;
        const int pair=queue[index];
        const auto result=evaluate_contact_exact<Limbs>(work.inputs[pair]);
        if(pool.failed||pool.available!=pool.slot_count){error(failed,24);break;}
        if constexpr(!Retry) {
            if(result.status==ContactStatus::ExactCapacityExceeded) {
                const unsigned retry=atomicAdd(work.exact_retry_count,1u);
                work.exact_retry_queue[retry]=pair;
                continue;
            }
        }
        if(result.status!=ContactStatus::Ok){error(failed,code(result.status));break;}
        store_contact_output(work.inputs[pair],result,work.outputs+pair,work.compact?work.compact+pair:nullptr);
    }
    resident_exact::release_pool();
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
__device__ void reduce_deformable_node(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,std::size_t node) {
    Derivatives system=elastic.systems[node];
    const int block=scene.node_blocks[node];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
        const int pair=work.incidents[incident].contact;
        if(!work.active[pair])continue;
        if(work.compact) {
            const auto& compact=work.compact[pair];
            if(compact.status!=ContactStatus::Ok){error(elastic.error,code(compact.status));return;}
            if(!compact.active)continue;
            if(compact.role>=0) {
                if(broad.contacts[pair].vertices[compact.role]==static_cast<int>(node))
                    add_compact_contact(system,compact,normal_scale);
                continue;
            }
        }
        const auto& out=work.outputs[pair];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(!out.active)continue;
        for(int role=0;role<4;++role)if(broad.contacts[pair].vertices[role]==static_cast<int>(node))
            add_point_contact(system,out,role,normal_scale);
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
    if(!finite_system(system)){error(elastic.error,26);return;}
    elastic.systems[node]=system;
}

__global__ void reduce_deformable(BroadPhaseView broad,DeformableScene scene,DeformableState state,
    ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,int color) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes||!scene.deformable[node]||broad.block_colors[scene.node_blocks[node]]!=color||atomicAdd(elastic.error,0))return;
    reduce_deformable_node(broad,scene,state,elastic,sdf,params,work,node);
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

#include "resident_contact_warp_reduce.cuh"

__device__ void add_rigid_contact(Derivatives& system,const ContactOutput& out,
    const BpContact& contact,int body,RigidScene rigid,const NodePullback* pullbacks,
    double scale,bool orientation) {
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
    const auto result=evaluate_rigid(prepared);
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
    if(atomicAdd(failed,0)||!work.active[pair]||!work.outputs[pair].active)return;
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
__global__ void reduce_rigid(BroadPhaseView broad,DeformableState deformable,RigidScene scene,
    RigidWorkspace workspace,ElasticWorkspace elastic,SdfScene sdf,ContactParameters params,Work work,
    int color,bool orientation) {
    const auto body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=scene.bodies||broad.block_colors[scene.body_blocks[body]]!=color||atomicAdd(elastic.error,0))return;
    if(orientation?!scene.rotate[body]:!scene.translate[body])return;
    Derivatives system=workspace.systems[body];
    const double normal_scale=params.dt*params.dt*params.k_barrier;
    const int block=scene.body_blocks[body];
    for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
        const int pair=work.incidents[incident].contact;
        if(!work.active[pair])continue;
        const auto& out=work.outputs[pair];
        if(out.status!=ContactStatus::Ok){error(elastic.error,code(out.status));return;}
        if(out.active)add_rigid_contact(system,out,broad.contacts[pair],static_cast<int>(body),scene,work.node_pullbacks,normal_scale,orientation);
    }
    if(sdf.parameters.stiffness>0)for(auto offset=scene.node_offsets[body];offset<scene.node_offsets[body+1];++offset) {
        const int vertex=scene.nodes[offset];
        const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,deformable.positions[vertex],deformable.previous[vertex],sdf.parameters);
        if(out.status!=SdfStatus::Ok){error(elastic.error,25);return;}
        double J[9]{};
        if(orientation)for(int k=0;k<9;++k)J[k]=work.node_pullbacks[vertex].jacobian[k];
        const auto pulled=sdf_rigid_pullback(out,J,params.dt*params.dt);
        const double* g=orientation?pulled.orientation_gradient:pulled.translation_gradient;
        const double* H=orientation?pulled.orientation_orientation_hessian:pulled.translation_translation_hessian;
        for(int k=0;k<3;++k){system.gradient[k]+=g[k];for(int l=0;l<3;++l)system.hessian[3*k+l]+=H[3*k+l];}
    }
    if(!finite_system(system)){error(elastic.error,26);return;}
    workspace.systems[body]=system;
}
}

struct ResidentContactPipeline::Impl {
    const ExactContactSchedule schedule=ExactContactSchedule::from_environment();
    const bool use_color_pairs=contact_execution_option("IPC_GPU_COLOR_PAIR_QUEUES");
    const bool use_warp_reduce=contact_execution_option("IPC_GPU_WARP_CONTACT_REDUCE");
    const bool use_selected_reduce=contact_execution_option("IPC_GPU_CONTACT_SELECTED_NODE_REDUCE",false);
    const int selected_threads=selected_reduce_threads();
    const bool use_compact_warp_reduce=contact_execution_option("IPC_GPU_COMPACT_WARP_REDUCE",false);
    const int compact_warp_degree_cutoff=ExactContactSchedule::environment_integer(
        "IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE",32,INT_MAX);
    const bool use_cooperative_exact=contact_execution_option("IPC_GPU_COOPERATIVE_EXACT_CONTACT");
    const bool use_compact=contact_execution_option("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS",false);
    const bool use_box_pruning=contact_execution_option("IPC_GPU_BOX_CONTACT_PRUNING",false);
    std::ofstream queue_log;
    unsigned long long rebuild_index=0,assembly_index=0;
    BroadPhaseView broad;
    int block_count=0;
    const int* vertex_block=nullptr;
    const unsigned char* surface_query=nullptr;
    Buffer<Incident> incidents;
    Buffer<int> pair_indices;
    Buffer<std::size_t> offsets;
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
    Buffer<ContactInput> inputs;
    Buffer<ContactOutput> outputs;
    Buffer<CompactDeformableContact> compact;
    Buffer<unsigned char> active;
    Buffer<unsigned char> box_separated;
    bool pruning_ready=false;
    double pruning_d_hat=0;
    Buffer<int> exact_queue,exact_retry_queue,exact_free_slots;
    Buffer<unsigned> exact_count,exact_retry_count,exact_next;
    Buffer<std::uint32_t> exact_words;
    Buffer<std::uint32_t> cooperative_words;
    Buffer<int> cooperative_slots;
    unsigned cooperative_workers=0;
    Buffer<RigidKinematics> body_kinematics;
    Buffer<NodePullback> node_pullbacks;
    Buffer<unsigned> pullback_needed;
    Impl() {
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
    void record_queues(int color,ContactAssemblyPhase phase,const int* failed) {
        if(!queue_log.is_open())return;
        unsigned count=0,retries=0;int status=0;
        checked(cudaMemcpy(&count,exact_count.data,sizeof(count),cudaMemcpyDeviceToHost),"diagnostic exact count");
        if(broad.contact_count)
            checked(cudaMemcpy(&retries,exact_retry_count.data,sizeof(retries),cudaMemcpyDeviceToHost),"diagnostic retry count");
        checked(cudaMemcpy(&status,failed,sizeof(status),cudaMemcpyDeviceToHost),"diagnostic device status");
        const char* phase_name=phase==ContactAssemblyPhase::Deformable?"deformable":
            phase==ContactAssemblyPhase::RigidTranslation?"rigid_translation":"rigid_orientation";
        queue_log<<rebuild_index<<','<<assembly_index<<','<<color<<','<<phase_name<<','<<broad.contact_count<<','
            <<count<<','<<retries<<','<<status<<','<<schedule.small_workers<<','<<schedule.small_threads<<','
            <<schedule.large_workers<<','<<schedule.large_threads<<'\n';
        queue_log.flush();
        if(!queue_log)throw std::runtime_error("failed writing contact queue diagnostic output");
    }
    Work work(bool compact_phase=false)const{return {inputs.data,outputs.data,compact_phase?compact.data:nullptr,active.data,exact_queue.data,exact_count.data,
        exact_retry_queue.data,exact_retry_count.data,exact_next.data,
        exact_words.data,exact_free_slots.data,incidents.data,offsets.data,vertex_block,surface_query,node_pullbacks.data};}
    ColorContactWorklist color_worklist()const {
        ColorContactWorklist result=!host_color_offsets.empty()?ColorContactWorklist{color_offsets.data,color_pairs.data,max_color_pairs}:ColorContactWorklist{};
        result.box_separated=pruning_ready?box_separated.data:nullptr;
        return result;
    }
};
ResidentContactPipeline::ResidentContactPipeline():impl_(std::make_unique<Impl>()){}
ResidentContactPipeline::~ResidentContactPipeline()=default;
void ResidentContactPipeline::rebuild(const BroadPhaseView& broad,int block_count,const int* vertex_block,const unsigned char* surface_query,const ContactBoxPruning& pruning) {
    if(block_count<0||(!vertex_block&&block_count)||broad.contact_count>std::size_t(INT_MAX))
        throw std::invalid_argument("resident contact topology/count unsupported");
    auto& p=*impl_;p.broad=broad;p.block_count=block_count;p.vertex_block=vertex_block;p.surface_query=surface_query;
    p.mapping_dirty=true;
    p.host_selected_offsets.clear();
    if((p.use_selected_reduce||p.use_compact_warp_reduce)&&broad.color_count>0&&broad.color_blocks&&broad.color_offsets) {
        p.host_selected_offsets.resize(static_cast<std::size_t>(broad.color_count)+1);
        checked(cudaMemcpy(p.host_selected_offsets.data(),broad.color_offsets,
            p.host_selected_offsets.size()*sizeof(int),cudaMemcpyDeviceToHost),"selected node launch metadata");
        if(p.host_selected_offsets.front()!=0||p.host_selected_offsets.back()>block_count
            ||!std::is_sorted(p.host_selected_offsets.begin(),p.host_selected_offsets.end()))
            throw std::invalid_argument("resident contact selected color offsets are invalid");
    }
    const auto& q=pruning.parameters;
    p.pruning_ready=p.use_box_pruning&&broad.block_colors&&pruning.boxes&&pruning.deformable&&pruning.positions&&pruning.previous
        &&std::isfinite(q.d_hat)&&q.d_hat>0&&std::isfinite(q.k_barrier)&&q.k_barrier>=0
        &&std::isfinite(q.dt)&&q.dt>0&&std::isfinite(q.friction)&&q.friction>=0
        &&std::isfinite(q.eps_v)&&q.eps_v>0&&std::isfinite(q.eps)&&q.eps>0;
    if(p.pruning_ready&&broad.contact_count) {
        p.box_separated.reserve(broad.contact_count);
        p.pruning_d_hat=q.d_hat;
        certify_boxes<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,pruning,p.box_separated.data);
    }
    if(p.queue_log.is_open())++p.rebuild_index;
    p.incidents.reserve(4*broad.contact_count);p.pair_indices.reserve(4*broad.contact_count);p.offsets.reserve(static_cast<std::size_t>(block_count)+1);
    p.inputs.reserve(broad.contact_count);p.outputs.reserve(broad.contact_count);p.active.reserve(broad.contact_count);p.exact_queue.reserve(broad.contact_count);
    if(p.use_compact)p.compact.reserve(broad.contact_count);
    p.exact_count.reserve(1);p.exact_retry_count.reserve(1);p.exact_next.reserve(1);
    p.exact_retry_queue.reserve(broad.contact_count);
    if(broad.contact_count) {
        const auto arena=p.schedule.arena_sizes(broad.contact_count,exact_small_limbs,exact_large_limbs,exact_slots);
        p.exact_words.reserve(arena.words);
        p.exact_free_slots.reserve(arena.free_slots);
        if(p.use_cooperative_exact) {
            p.cooperative_workers=static_cast<unsigned>(std::min<std::size_t>(broad.contact_count,256));
            const auto cooperative=cooperative_exact_arena_sizes(p.cooperative_workers);
            p.cooperative_words.reserve(cooperative.words);p.cooperative_slots.reserve(cooperative.free_slots);
        }
    }
    if(broad.contact_count) {
        emit_incidents<<<blocks(broad.contact_count),threads>>>(broad,vertex_block,p.incidents.data,p.pruning_ready?p.box_separated.data:nullptr);
        thrust::sort(thrust::device_pointer_cast(p.incidents.data),thrust::device_pointer_cast(p.incidents.data+4*broad.contact_count),IncidentOrder{broad.contacts});
        incident_indices<<<blocks(4*broad.contact_count),threads>>>(p.incidents.data,4*broad.contact_count,p.pair_indices.data);
    }
    incident_offsets<<<blocks(static_cast<std::size_t>(block_count)+1),threads>>>(p.incidents.data,4*broad.contact_count,block_count,p.offsets.data);
    checked(cudaGetLastError(),"incident CSR kernels");
    p.host_color_offsets.clear();
    if(p.use_color_pairs&&broad.color_count>0&&broad.block_colors) {
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
}
ContactIncidence ResidentContactPipeline::incidence()const {
    return {impl_->offsets.data,impl_->pair_indices.data,impl_->color_worklist()};
}
void ResidentContactPipeline::assemble_color(const DeformableScene& scene,const DeformableState& state,
    const ElasticWorkspace& elastic,const RigidScene& rigid,const RigidState& rigid_state,
    const RigidWorkspace& rigid_workspace,const SdfScene& sdf,const ContactParameters& parameters,
    int color,ContactAssemblyPhase phase) {
    ResidentProfileScope profile("contact");
    auto& p=*impl_;
    if(p.queue_log.is_open())++p.assembly_index;
    if(!p.offsets.data||!elastic.error)throw std::logic_error("resident contact assembly requires rebuild and device error pointer");
    if(color<0||(p.broad.color_count>0&&color>=p.broad.color_count))throw std::invalid_argument("resident contact color out of range");
    if(p.pruning_ready&&p.broad.contact_count) {
        const auto& q=parameters;
        if(!std::isfinite(q.d_hat)||q.d_hat>p.pruning_d_hat||q.d_hat<=0
            ||!std::isfinite(q.k_barrier)||q.k_barrier<0||!std::isfinite(q.dt)||q.dt<=0
            ||!std::isfinite(q.friction)||q.friction<0||!std::isfinite(q.eps_v)||q.eps_v<=0
            ||!std::isfinite(q.eps)||q.eps<=0)
            throw std::invalid_argument("contact box certificates require valid parameters and rebuild before increasing d_hat");
    }
    const bool compact_phase=p.use_compact&&phase==ContactAssemblyPhase::Deformable;
    const bool warp_ready=p.use_warp_reduce&&!compact_phase&&p.broad.color_blocks&&p.broad.color_offsets;
    const bool selected_ready=phase==ContactAssemblyPhase::Deformable&&!p.host_selected_offsets.empty()
        &&(p.use_selected_reduce||(compact_phase&&p.use_compact_warp_reduce));
    const bool compact_warp_ready=selected_ready&&compact_phase&&p.use_compact_warp_reduce&&p.broad.contact_count;
    if((warp_ready||selected_ready)&&(p.mapping_dirty||p.mapped_nodes!=scene.nodes
        ||p.mapped_node_blocks!=scene.node_blocks||p.mapped_deformable!=scene.deformable)) {
        p.block_to_deformable_node.reserve(p.block_count);
        if(p.block_count)checked(cudaMemsetAsync(p.block_to_deformable_node.data,0xff,p.block_count*sizeof(int)),"clear deformable block map");
        if(scene.nodes)map_deformable_blocks<<<blocks(scene.nodes),threads>>>(scene,p.block_count,p.block_to_deformable_node.data,elastic.error);
        p.mapping_dirty=false;
        p.mapped_nodes=scene.nodes;p.mapped_node_blocks=scene.node_blocks;p.mapped_deformable=scene.deformable;
    }
    {
    ResidentProfileScope reset_profile("queue_reset");
    checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned)),"clear exact queue");
    }
    const auto gather_count=p.color_worklist().offsets?p.host_color_offsets[color+1]-p.host_color_offsets[color]:p.broad.contact_count;
    if(p.broad.contact_count) {
        {
        ResidentProfileScope fast_profile("fused_gather_fast_compute");
        checked(cudaMemsetAsync(p.exact_retry_count.data,0,sizeof(unsigned)),"clear exact retry queue");
        checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear exact cursor");
        if(gather_count)gather_compute<<<blocks(gather_count),threads>>>(p.broad,scene,state,rigid,parameters,p.work(compact_phase),color,phase,elastic.error,p.color_worklist());
        }
        {
        ResidentProfileScope exact_profile("exact_small");
        if(p.use_cooperative_exact)
            launch_cooperative_exact_contacts(p.inputs.data,p.outputs.data,p.exact_queue.data,p.exact_count.data,
                p.exact_retry_queue.data,p.exact_retry_count.data,p.exact_next.data,
                p.cooperative_words.data,p.cooperative_slots.data,p.cooperative_workers,elastic.error,nullptr,
                compact_phase?p.compact.data:nullptr);
        else evaluate_exact_queue<exact_small_limbs,false><<<p.schedule.small_workers/p.schedule.small_threads,p.schedule.small_threads>>>(p.work(compact_phase),elastic.error);
        }
        ResidentProfileScope retry_profile("exact_large_retry");
        checked(cudaMemsetAsync(p.exact_next.data,0,sizeof(unsigned)),"clear exact retry cursor");
        evaluate_exact_queue<exact_large_limbs,true><<<p.schedule.large_workers/p.schedule.large_threads,p.schedule.large_threads>>>(p.work(compact_phase),elastic.error);
    }
    if(p.queue_log.is_open()) {
        ResidentProfileScope diagnostic_profile("queue_diagnostic_readback");
        p.record_queues(color,phase,elastic.error);
    }
    if(phase==ContactAssemblyPhase::RigidOrientation&&rigid.bodies) {
        ResidentProfileScope pullback_profile("rigid_pullback");
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
    ResidentProfileScope reduce_profile("reduce_sdf");
    if(phase==ContactAssemblyPhase::Deformable) {
        if(scene.nodes) {
            if(selected_ready) {
                const auto selected_count=static_cast<std::size_t>(p.host_selected_offsets[color+1]-p.host_selected_offsets[color]);
                if(selected_count) {
                    // Small blocks spread narrow colors across SMs; this
                    // opt-in launch knob does not change node arithmetic.
                    const auto selected_blocks=static_cast<unsigned>((selected_count+p.selected_threads-1)/p.selected_threads);
                    reduce_deformable_selected<<<selected_blocks,p.selected_threads>>>(p.broad,scene,state,elastic,sdf,
                        parameters,p.work(compact_phase),color,p.block_to_deformable_node.data,p.block_count,
                        compact_warp_ready?static_cast<std::size_t>(p.compact_warp_degree_cutoff):0);
                    if(compact_warp_ready)
                        reduce_deformable_compact_warp<<<static_cast<unsigned>(std::min<std::size_t>(selected_count,256)),32>>>(
                            p.broad,scene,state,elastic,sdf,parameters,p.work(true),color,p.block_to_deformable_node.data,
                            p.block_count,static_cast<std::size_t>(p.compact_warp_degree_cutoff));
                }
            } else if(warp_ready)
                reduce_deformable_warp<<<std::min(p.block_count,256),32>>>(p.broad,scene,state,elastic,sdf,parameters,p.work(),color,p.block_to_deformable_node.data);
            else reduce_deformable<<<blocks(scene.nodes),threads>>>(p.broad,scene,state,elastic,sdf,parameters,p.work(compact_phase),color);
        }
    } else if(rigid.bodies)reduce_rigid<<<blocks(rigid.bodies),threads>>>(p.broad,state,rigid,rigid_workspace,elastic,sdf,parameters,p.work(),color,phase==ContactAssemblyPhase::RigidOrientation);
    checked(cudaGetLastError(),"assembly kernels");
}

} // namespace ipc_gpu::resident

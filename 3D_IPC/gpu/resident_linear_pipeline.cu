#define IPC_GPU_TU_FINITE_CCD 1
#include "resident_linear_pipeline.h"
#include "resident_profile.h"
#include "resident_ccd.cuh"
#include "resident_exact.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace ipc_gpu::resident {
namespace {
namespace cert=ipc_gpu::resident_ccd;
namespace exact=ipc_gpu::resident_exact;
constexpr int arena_width=2048,arena_slots=512;

void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
__device__ void fail(int*error,LinearClipError code) { atomicCAS(error,0,int(code)); }
__device__ bool finite(double v) { return v-v==0.0; }
__device__ bool inside(const double*p,const BpAabb&box) {
    for(int k=0;k<3;++k)
        if(!finite(p[k])||!finite(box.lo[k])||!finite(box.hi[k])||p[k]<box.lo[k]||p[k]>box.hi[k])return false;
    return true;
}
__device__ double pair_floor(const double p[4][3]) {
    double scale=0.;for(int i=0;i<4;++i)for(int k=0;k<3;++k)scale=fmax(scale,fabs(p[i][k]));
    return fmax(1e-8,128.0*DBL_EPSILON*scale);
}
__device__ bool box_fraction(const double*position,const double*delta,
    const BpAabb&box,double&fraction,int*error) {
    if(!inside(position,box)) { fail(error,LinearClipError::OutsideBlueBox);return false; }
    for(int k=0;k<3;++k) {
        if(!finite(delta[k])) { fail(error,LinearClipError::InvalidInput);return false; }
        if(delta[k]>0.)fraction=fmin(fraction,(box.hi[k]-position[k])/delta[k]);
        if(delta[k]<0.)fraction=fmin(fraction,(box.lo[k]-position[k])/delta[k]);
    }
    fraction=fmax(0.,fmin(1.,fraction));return true;
}
__device__ bool map_failure(exact::Status status,int*error) {
    if(status==exact::Status::Safe||status==exact::Status::Unsafe)return false;
    fail(error,status==exact::Status::InitialContact?LinearClipError::InitialContact
        :status==exact::Status::CapacityExceeded?LinearClipError::ExactResources:LinearClipError::InvalidInput);
    return true;
}
__device__ exact::Status exact_endpoint(const cert::LinearQuery&q) {
    auto status=exact::validate_step<128>(q.start,q.end,q.vertex_face,q.gap_floor);
    if(status==exact::Status::CapacityExceeded)
        status=exact::validate_step<2048>(q.start,q.end,q.vertex_face,q.gap_floor);
    return status;
}
__device__ cert::LinearQuery node_query(const BpContact&contact,const double(*positions)[3],
    int node,const double*endpoint) {
    cert::LinearQuery q;q.vertex_face=contact.kind==0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        q.start[i][k]=positions[contact.vertices[i]][k];
        q.end[i][k]=contact.vertices[i]==node?endpoint[k]:q.start[i][k];
    }
    q.gap_floor=pair_floor(q.start);return q;
}

__global__ void propose_nodes(DeformableScene scene,DeformableState state,ElasticWorkspace work,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],unsigned int*pending,int*error) {
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes)return;
    pending[node]=0;
    // Extra slot is a device-only cold-queue count. The following kernel
    // cannot observe it until this whole proposal kernel has joined.
    if(node==0)pending[scene.nodes]=0;
    // A failed assembly/proposal may leave work.proposed unwritten. Do not
    // inspect it once an earlier stage (or another node) has reported failure.
    if(atomicAdd(error,0))return;
    if(!scene.deformable[node]||colors[scene.node_blocks[node]]!=color)return;
    for(int k=0;k<3;++k)accepted[node][k]=state.positions[node][k];
    double delta[3];for(int k=0;k<3;++k)delta[k]=work.proposed[node][k]-state.positions[node][k];
    double fraction=1.;if(!box_fraction(state.positions[node],delta,boxes[node],fraction,error))return;
    bool boxed=false;
    for(int attempt=0;attempt<64;++attempt) {
        for(int k=0;k<3;++k)accepted[node][k]=fraction==1.?work.proposed[node][k]
            :fma(fraction,delta[k],state.positions[node][k]);
        if(inside(accepted[node],boxes[node])) { boxed=true;break; }
        fraction*=.5;
    }
    if(!boxed) { fail(error,LinearClipError::OutsideBlueBox);return; }
}

// The color graph guarantees at most one deformable update block per pair.
// Endpoint predicates are independent while these positions/proposals remain
// immutable, so contacts of a heavy node can occupy many GPU workers instead
// of being checked serially by that node's proposal thread. Like the CPU's
// joined cooperative predicate, this kernel completes before any cold check
// or position commit. Only explicit Safe certificates are reusable.
__global__ void certify_node_contacts(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const int*colors,int color,const double(*accepted)[3],
    unsigned int*pending,unsigned char*initial_safe,int*error,ColorContactWorklist color_pairs) {
    const std::size_t stride=std::size_t(gridDim.x)*blockDim.x;
    const auto begin=color_pairs.offsets?color_pairs.offsets[color]:0;
    const auto end=color_pairs.offsets?color_pairs.offsets[color+1]:bp.contact_count;
    for(std::size_t index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
        index<end-begin;index+=stride) {
        const auto pair=color_pairs.offsets?static_cast<std::size_t>(color_pairs.pair_indices[begin+index]):index;
        initial_safe[pair]=0;
        if(atomicAdd(error,0))continue;
        if(color_pairs.box_separated&&color_pairs.box_separated[pair]) {
            initial_safe[pair]=1;continue;
        }
        const auto& contact=bp.contacts[pair];
        int active_node=-1;
        for(int i=0;i<4;++i) {
            const int node=contact.vertices[i];
            if(!scene.deformable[node]||colors[scene.node_blocks[node]]!=color)continue;
            if(active_node>=0&&active_node!=node) {
                fail(error,LinearClipError::UnsupportedMotion);active_node=-1;break;
            }
            active_node=node;
        }
        if(active_node<0)continue;
        const auto q=node_query(contact,state.positions,active_node,accepted[active_node]);
        const auto status=cert::validate_linear(q);
        if(status==cert::Status::Safe)initial_safe[pair]=1;
        else if(status==cert::Status::Invalid)fail(error,LinearClipError::InvalidInput);
        else if(atomicOr(pending+active_node,1U)==0)
            atomicAdd(pending+scene.nodes,1U);
    }
}

__global__ void resolve_nodes(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots) {
    if(blockIdx.x||threadIdx.x||*error||!pending[scene.nodes])return;
    exact::DevicePool pool;exact::initialize_pool(pool,words,slots,arena_width,arena_slots);
    for(std::size_t node=0;node<scene.nodes&&!*error;++node)if(pending[node]) {
        const int block=scene.node_blocks[node];double target[3],delta[3];
        for(int k=0;k<3;++k) { target[k]=accepted[node][k];delta[k]=target[k]-state.positions[node][k]; }
        double fraction=1.;bool complete=false;
        for(int attempt=0;attempt<=64;++attempt) {
            // The production solver ultimately permits no motion. This last
            // candidate is NOT an uncertified freeze: every incident contact
            // must still prove the current represented state strictly valid.
            double trial[3];for(int k=0;k<3;++k)trial[k]=attempt==64?state.positions[node][k]
                :attempt?fma(fraction,delta[k],state.positions[node][k]):target[k];
            bool safe=inside(trial,boxes[node]);
            if(safe)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const int pair_index=indices[pair];
                // Cache is scoped to this immutable color's original proposed
                // endpoint. Every changed backtracking trial must check again.
                if(attempt==0&&initial_safe[pair_index])continue;
                const auto q=node_query(bp.contacts[pair_index],state.positions,int(node),trial);
                // The parallel pass already found the remaining original
                // predicates inconclusive; do not run its interval work twice.
                const auto quick=attempt?cert::validate_linear(q):cert::Status::Uncertain;
                if(quick==cert::Status::Safe)continue;
                const auto status=quick==cert::Status::Invalid?exact::Status::Invalid:exact_endpoint(q);
                if(map_failure(status,error)) { safe=false;break; }
                if(status==exact::Status::Unsafe) { safe=false;break; }
            }
            if(*error)break;
            if(safe) {
                for(int k=0;k<3;++k)accepted[node][k]=trial[k];complete=true;break;
            }
            fraction*=.5;
        }
        if(!complete&&!*error)fail(error,LinearClipError::Unresolved);
        if(pool.failed||pool.available!=pool.slot_count)fail(error,LinearClipError::ExactResources);
    }
    exact::release_pool();
}

__device__ bool body_trial(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const BpAabb*boxes,int body,double fraction,
    double(*accepted)[3],double*delta,int*error) {
    for(int k=0;k<3;++k) {
        const double direction=work.proposed_com[body][k]-state.com[body][k];
        const double com=fraction==0.?state.com[body][k]
            :fraction==1.?work.proposed_com[body][k]:fma(fraction,direction,state.com[body][k]);
        if(!finite(com)) { fail(error,LinearClipError::InvalidInput);return false; }
        work.accepted_com[body][k]=com;delta[k]=com-state.com[body][k];
    }
    bool boxed=true;
    for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i) {
        const int node=scene.nodes[i];
        for(int k=0;k<3;++k)accepted[node][k]=vertices.positions[node][k]+delta[k];
        boxed=boxed&&inside(accepted[node],boxes[node]);
    }
    return boxed;
}
__device__ cert::LinearQuery body_query(const BpContact&contact,RigidScene scene,
    DeformableState vertices,int body,const double(*accepted)[3]) {
    cert::LinearQuery q;q.vertex_face=contact.kind==0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const int node=contact.vertices[i];q.start[i][k]=vertices.positions[node][k];
        q.end[i][k]=scene.node_body[node]==body?accepted[node][k]:q.start[i][k];
    }
    q.gap_floor=pair_floor(q.start);return q;
}

__global__ void resolve_bodies(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd) {
    if(blockIdx.x||threadIdx.x||*error)return;
    exact::DevicePool pool;exact::initialize_pool(pool,words,slots,arena_width,arena_slots);
    for(std::size_t body=0;body<scene.bodies&&!*error;++body) {
        const int block=scene.body_blocks[body];if(colors[block]!=color)continue;
        for(int k=0;k<3;++k)work.accepted_com[body][k]=state.com[body][k];
        for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i)
            for(int k=0;k<3;++k)accepted[scene.nodes[i]][k]=vertices.positions[scene.nodes[i]][k];
        if(!scene.translate[body])continue;
        double proposed_delta[3];for(int k=0;k<3;++k)proposed_delta[k]=work.proposed_com[body][k]-state.com[body][k];
        double fraction=1.;bool valid=true;
        for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i) {
            const int node=scene.nodes[i];
            if(!box_fraction(vertices.positions[node],proposed_delta,boxes[node],fraction,error)) { valid=false;break; }
        }
        if(!valid)break;
        bool complete=false;
        for(int attempt=0;attempt<=64;++attempt) {
            if(attempt==64)fraction=0.;
            double delta[3];bool safe=body_trial(scene,state,work,vertices,boxes,int(body),fraction,accepted,delta,error);
            if(*error)break;
            if(safe&&use_ccd)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const auto&contact=bp.contacts[indices[pair]];
                const auto q=body_query(contact,scene,vertices,int(body),accepted);
                const auto quick=cert::validate_linear(q);
                if(quick==cert::Status::Safe)continue;
                double motion[4][3];
                for(int i=0;i<4;++i)for(int k=0;k<3;++k)
                    motion[i][k]=scene.node_body[contact.vertices[i]]==int(body)?delta[k]:0.;
                // Match the CPU rigid linear-motion contract. Its path is
                // original positions + t*one common represented delta. The
                // actual rounded endpoint gap is checked independently below.
                auto status=exact::validate_motion<128>(q.start,motion,q.vertex_face,q.gap_floor);
                if(status==exact::Status::CapacityExceeded)
                    status=exact::validate_motion<2048>(q.start,motion,q.vertex_face,q.gap_floor);
                if(status==exact::Status::Safe) {
                    status=exact::validate_endpoint_gap<128>(q.start,q.end,q.vertex_face,q.gap_floor);
                    if(status==exact::Status::CapacityExceeded)
                        status=exact::validate_endpoint_gap<2048>(q.start,q.end,q.vertex_face,q.gap_floor);
                }
                if(map_failure(status,error)) { safe=false;break; }
                if(status==exact::Status::Unsafe) { safe=false;break; }
            }
            if(*error)break;
            if(safe) { complete=true;break; }
            fraction*=.5;
        }
        if(!complete&&!*error)fail(error,LinearClipError::Unresolved);
        if(pool.failed||pool.available!=pool.slot_count)fail(error,LinearClipError::ExactResources);
    }
    exact::release_pool();
}
} // namespace

struct ResidentLinearClipper::Impl {
    std::uint32_t* words=nullptr;int* slots=nullptr;
    unsigned int* pending=nullptr;std::size_t pending_capacity=0;
    unsigned char* initial_safe=nullptr;std::size_t contact_capacity=0;
    Impl() {
        check(cudaMalloc(&words,std::size_t(arena_width)*arena_slots*sizeof(*words)));
        try { check(cudaMalloc(&slots,arena_slots*sizeof(*slots))); }
        catch(...) { cudaFree(words);throw; }
    }
    ~Impl() { if(initial_safe)cudaFree(initial_safe);if(pending)cudaFree(pending);if(slots)cudaFree(slots);if(words)cudaFree(words); }
    void reserve(std::size_t nodes,std::size_t contacts) {
        if(nodes>pending_capacity) {
            unsigned int* replacement=nullptr;check(cudaMalloc(&replacement,(nodes+1)*sizeof(*replacement)));
            if(pending)check(cudaFree(pending));pending=replacement;pending_capacity=nodes;
        }
        if(contacts>contact_capacity) {
            unsigned char* replacement=nullptr;check(cudaMalloc(&replacement,contacts));
            if(initial_safe)check(cudaFree(initial_safe));initial_safe=replacement;contact_capacity=contacts;
        }
    }
};
ResidentLinearClipper::ResidentLinearClipper():impl_(new Impl) {}
ResidentLinearClipper::~ResidentLinearClipper()=default;

void ResidentLinearClipper::clip_deformable_color(const DeformableScene&scene,
    const DeformableState&state,const ElasticWorkspace&work,const BroadPhaseView&bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const int*colors,int color,
    double(*accepted)[3],int*error,bool use_ccd,cudaStream_t stream,ColorContactWorklist color_pairs) {
    {
    ResidentProfileScope prepare_profile("prepare",stream);
    impl_->reserve(scene.nodes,use_ccd?bp.contact_count:0);if(!scene.nodes)return;
    propose_nodes<<<unsigned((scene.nodes+127)/128),128,0,stream>>>(scene,state,work,
        boxes,colors,color,accepted,impl_->pending,error);check(cudaGetLastError());
    }
    if(!use_ccd||!bp.contact_count)return;
    if(bp.contact_count) {
        ResidentProfileScope certify_profile("parallel_certify",stream);
        const auto count=color_pairs.offsets?color_pairs.max_color_pairs:bp.contact_count;
        const unsigned blocks=unsigned(std::max<std::size_t>(1,std::min<std::size_t>((count+127)/128,65535)));
        certify_node_contacts<<<blocks,128,0,stream>>>(scene,state,bp,colors,color,accepted,
            impl_->pending,impl_->initial_safe,error,color_pairs);check(cudaGetLastError());
    }
    ResidentProfileScope resolve_profile("serial_resolve",stream);
    resolve_nodes<<<1,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,impl_->pending,impl_->initial_safe,accepted,error,
        impl_->words,impl_->slots);check(cudaGetLastError());
}

void ResidentLinearClipper::clip_rigid_translation_color(const RigidScene&scene,const RigidState&state,
    const RigidWorkspace&work,const DeformableState&vertices,const BroadPhaseView&bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const int*colors,int color,
    double(*accepted)[3],int*error,bool use_ccd,cudaStream_t stream) {
    ResidentProfileScope resolve_profile("serial_resolve",stream);
    if(!scene.bodies)return;
    resolve_bodies<<<1,1,0,stream>>>(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
        accepted,error,impl_->words,impl_->slots,use_ccd);check(cudaGetLastError());
}

} // namespace ipc_gpu::resident

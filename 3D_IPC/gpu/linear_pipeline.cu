#define IPC_GPU_TU_FINITE_CCD 1
#include "linear_pipeline_internal.cuh"
#include "contact_schedule.h"
#include "contact_execution.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ipc_gpu::solver {
namespace rigid_proxy_owner {thread_local Scope* current_scope=nullptr;}
using namespace linear_detail;
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
int rigid_translation_pair_threads() {
    const char* value=std::getenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS");
    if(!value||!*value||std::strcmp(value,"1")==0)return 1;
    if(std::strcmp(value,"8")==0)return 8;
    if(std::strcmp(value,"16")==0)return 16;
    if(std::strcmp(value,"32")==0)return 32;
    if(std::strcmp(value,"64")==0)return 64;
    if(std::strcmp(value,"128")==0)return 128;
    throw std::invalid_argument("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS must be 1,8,16,32,64,128");
}

__global__ void propose_nodes(DeformableScene scene,DeformableState state,ElasticWorkspace work,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],double(*raw_delta)[3],
    unsigned int*pending,int*error) {
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
    const double*start=state.positions[node];const BpAabb&box=boxes[node];
    if(!inside(start,box)) { fail(error,LinearClipError::OutsideBlueBox);return; }
    // Match per_vertex_safe_step exactly: every raw proposal is clamped
    // COORDINATEWISE to this node's blue box with the same fixed 1e-10 inset.
    // This is not the uniform sweeping fraction used for rigid translation;
    // an exiting axis must not shrink motion on axes that stay inside.
    constexpr double inset=1e-10;
    double clipped[3];
    for(int k=0;k<3;++k) {
        const double raw=work.proposed[node][k];
        if(!finite(raw)) { fail(error,LinearClipError::InvalidInput);return; }
        const double lo=__dadd_rn(box.lo[k],inset),hi=__dsub_rn(box.hi[k],inset);
        // An empty or nonfinite inner interval cannot authorize an update.
        if(!finite(lo)||!finite(hi)||lo>hi) { fail(error,LinearClipError::OutsideBlueBox);return; }
        double value=raw;
        if(value<lo)value=lo;
        if(value>hi)value=hi;
        clipped[k]=value;
    }
    // The represented endpoint is the CPU's unfused before + (clipped-before),
    // not the clipped value itself: the two can differ by one rounding.
    double endpoint[3];
    for(int k=0;k<3;++k) {
        const double delta=__dsub_rn(clipped[k],start[k]);
        // Stored raw displacement. Original mode must commit from this value,
        // never from a rounded unit endpoint minus `before`.
        if(raw_delta)raw_delta[node][k]=delta;
        endpoint[k]=__dadd_rn(start[k],delta);
        // Overflow of the unfused source displacement is a nonfinite input
        // regime, not a box violation; it must never authorize an update.
        if(!finite(delta)||!finite(endpoint[k])) {
            fail(error,LinearClipError::InvalidInput);return;
        }
    }
    // Never publish a rounded endpoint outside the actual blue box.
    if(!inside(endpoint,box)) { fail(error,LinearClipError::OutsideBlueBox);return; }
    for(int k=0;k<3;++k)accepted[node][k]=endpoint[k];
}

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

__global__ void validate_original_mask(std::size_t nodes,const unsigned char*selected,
    int*error) {
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=nodes)return;
    if(selected[node]>1)fail(error,LinearClipError::InvalidInput);
}

__global__ void apply_selected_small_step_cutoff(DeformableScene scene,DeformableState state,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],
    const unsigned char*selected,int*error) {
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes)return;
    if(atomicAdd(error,0))return;
    if(!selected||!scene.deformable[node])return;
    if(colors[scene.node_blocks[node]]!=color)return;
    if(!selected[node])return;
    double squared=0.0;
    for(int k=0;k<3;++k)squared+=raw_delta[node][k]*raw_delta[node][k];
    if(squared<1e-28)
        for(int k=0;k<3;++k)accepted[node][k]=state.positions[node][k];
}
} // namespace

struct GpuLinearClipper::Impl {
    // Qualified CSR execution redistribution; capture the rollback switch once per clipper.
    const bool node_csr_partition=contact_execution_option("IPC_GPU_NODE_CSR_PARTITION",true);
    // Qualified controller defaults on; resource checks retain the fixed CSR rollback.
    const bool node_cooperative=contact_execution_option("IPC_GPU_NODE_CCD_COOPERATIVE",true);
    const bool parallel_rigid_proxy=contact_execution_option("IPC_GPU_RIGID_TRANSLATION_PROXY_PARALLEL",true);
    std::uint32_t* words=nullptr;int* slots=nullptr;
    // Bounded ORIGINAL-mode worker pool: `original_workers` live workers, each
    // with its own disjoint arena slice. Grown lazily, never per scene node.
    std::uint32_t* original_words=nullptr;int* original_slots=nullptr;
    std::size_t original_worker_capacity=0;
    RigidCcdBatchContext* rigid_batch_context=nullptr;
    unsigned* rigid_batch_queue=nullptr;
    std::size_t rigid_batch_capacity=0;
    OriginalNodeCcdControl* node_batch_control=nullptr;
    OriginalNodeCcdContext* node_batch_contexts=nullptr;
    unsigned* node_batch_active=nullptr;
    unsigned long long* node_batch_queue=nullptr;
    std::size_t node_batch_nodes_capacity=0,node_batch_queue_capacity=0;
    // Capture this host-only concurrency limit once per clipper. Each worker
    // retains its own 4 MiB limb arena; 256 workers use at most 1 GiB of limbs
    // (plus 2 KiB of allocator slots per worker). Numerical budgets are intact.
    const int original_worker_limit=ExactContactSchedule::environment_integer(
        "IPC_GPU_ORIGINAL_CCD_WORKERS",original_worker_cap,256);
    unsigned int* pending=nullptr;std::size_t pending_capacity=0;
    unsigned char* initial_safe=nullptr;std::size_t contact_capacity=0;
    double (*clip_raw)[3]=nullptr;std::size_t clip_raw_capacity=0;
    Impl() {
        check(cudaMalloc(&words,std::size_t(arena_width)*arena_slots*sizeof(*words)));
        try { check(cudaMalloc(&slots,arena_slots*sizeof(*slots))); }
        catch(...) { cudaFree(words);throw; }
    }
    ~Impl() { if(node_batch_queue)cudaFree(node_batch_queue);if(node_batch_active)cudaFree(node_batch_active);
        if(node_batch_contexts)cudaFree(node_batch_contexts);if(node_batch_control)cudaFree(node_batch_control);
        if(rigid_batch_queue)cudaFree(rigid_batch_queue);
        if(rigid_batch_context)cudaFree(rigid_batch_context);
        if(clip_raw)cudaFree(clip_raw);if(initial_safe)cudaFree(initial_safe);if(pending)cudaFree(pending);
        if(original_slots)cudaFree(original_slots);if(original_words)cudaFree(original_words);
        if(slots)cudaFree(slots);if(words)cudaFree(words); }
    void reserve(std::size_t nodes,std::size_t contacts) {
        if(nodes>pending_capacity) {
            unsigned int* replacement=nullptr;check(cudaMalloc(&replacement,(nodes+1)*sizeof(*replacement)));
            if(pending)check(cudaFree(pending));pending=replacement;pending_capacity=nodes;
        }
        if(nodes>clip_raw_capacity) {
            double (*replacement)[3]=nullptr;check(cudaMalloc(&replacement,nodes*sizeof(*replacement)));
            if(clip_raw)check(cudaFree(clip_raw));clip_raw=replacement;clip_raw_capacity=nodes;
        }
        if(contacts>contact_capacity) {
            unsigned char* replacement=nullptr;check(cudaMalloc(&replacement,contacts));
            if(initial_safe)check(cudaFree(initial_safe));initial_safe=replacement;contact_capacity=contacts;
        }
    }
    // Bounded worker count for the original resolver: one disjoint arena per
    // worker, capped independently of the scene size.
    void reserve_original_workers(std::size_t wanted) {
        if(wanted>original_worker_capacity) {
            std::uint32_t* new_words=nullptr;int* new_slots=nullptr;
            check(cudaMalloc(&new_words,wanted*std::size_t(original_arena_width)
                *std::size_t(original_arena_slots)*sizeof(*new_words)));
            try { check(cudaMalloc(&new_slots,wanted*std::size_t(original_arena_slots)*sizeof(*new_slots))); }
            catch(...) { cudaFree(new_words);throw; }
            if(original_slots)cudaFree(original_slots);
            if(original_words)cudaFree(original_words);
            original_words=new_words;original_slots=new_slots;original_worker_capacity=wanted;
        }
    }
    int original_workers(std::size_t nodes) {
        const std::size_t wanted=std::min<std::size_t>(nodes,static_cast<std::size_t>(original_worker_limit));
        reserve_original_workers(wanted);
        return static_cast<int>(wanted);
    }
    void reserve_rigid_batch(std::size_t count) {
        if(!rigid_batch_context)check(cudaMalloc(&rigid_batch_context,sizeof(*rigid_batch_context)));
        if(count<=rigid_batch_capacity)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(unsigned))
            throw std::overflow_error("rigid CCD batch storage overflow");
        unsigned* queue=nullptr;check(cudaMalloc(&queue,count*sizeof(*queue)));
        if(rigid_batch_queue)cudaFree(rigid_batch_queue);
        rigid_batch_queue=queue;rigid_batch_capacity=count;
    }
    void reserve_node_batch(std::size_t nodes,std::size_t queue_capacity) {
        if(!node_batch_control)check(cudaMalloc(&node_batch_control,sizeof(*node_batch_control)));
        if(nodes>node_batch_nodes_capacity) {
            if(nodes>std::numeric_limits<std::size_t>::max()/sizeof(OriginalNodeCcdContext)
                ||nodes>std::numeric_limits<std::size_t>::max()/sizeof(unsigned))
                throw std::overflow_error("original node CCD context storage overflow");
            OriginalNodeCcdContext* contexts=nullptr;unsigned* active=nullptr;
            check(cudaMalloc(&contexts,nodes*sizeof(*contexts)));
            try {check(cudaMalloc(&active,nodes*sizeof(*active)));}
            catch(...) {cudaFree(contexts);throw;}
            if(node_batch_contexts)cudaFree(node_batch_contexts);if(node_batch_active)cudaFree(node_batch_active);
            node_batch_contexts=contexts;node_batch_active=active;node_batch_nodes_capacity=nodes;
        }
        if(queue_capacity>node_batch_queue_capacity) {
            if(queue_capacity>std::numeric_limits<std::size_t>::max()/sizeof(unsigned long long))
                throw std::overflow_error("original node CCD queue storage overflow");
            unsigned long long* queue=nullptr;check(cudaMalloc(&queue,queue_capacity*sizeof(*queue)));
            if(node_batch_queue)cudaFree(node_batch_queue);
            node_batch_queue=queue;node_batch_queue_capacity=queue_capacity;
        }
    }
};
GpuLinearClipper::GpuLinearClipper():impl_(new Impl) {}
GpuLinearClipper::~GpuLinearClipper()=default;

void GpuLinearClipper::clip_deformable_color(const DeformableScene&scene,
    const DeformableState&state,const ElasticWorkspace&work,const BroadPhaseView&bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const int*colors,int color,
    double(*accepted)[3],int*error,bool use_ccd,cudaStream_t stream,ColorContactWorklist color_pairs,
    bool exact_step_bound,LinearClipMode mode,OriginalUpdateConvention convention,
    const unsigned char*original_cooperative_selected) {
    {
    impl_->reserve(scene.nodes,use_ccd?bp.contact_count:0);if(!scene.nodes)return;
    propose_nodes<<<unsigned((scene.nodes+127)/128),128,0,stream>>>(scene,state,work,
        boxes,colors,color,accepted,impl_->clip_raw,impl_->pending,error);check(cudaGetLastError());
    }
    if(mode!=LinearClipMode::Robust) {
        // Original mode never uses the robust exact queue/certificates. It
        // validates the solver-owned cooperative mask ONCE (when applicable)
        // and applies the CPU raw-displacement commit rule on the device with a
        // bounded pool of independent whole-node workers.
        if(convention==OriginalUpdateConvention::CooperativeSquaredNorm
            &&original_cooperative_selected)
            validate_original_mask<<<unsigned((scene.nodes+127)/128),128,0,stream>>>(
                scene.nodes,original_cooperative_selected,error);
        if(contact_execution_option("IPC_GPU_ORIGINAL_NODE_CCD_BATCH",true)
            &&scene.nodes<=std::numeric_limits<unsigned>::max()) {
            const auto capacity=static_cast<std::size_t>(ExactContactSchedule::environment_integer(
                "IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY",1<<20,1<<26));
            impl_->reserve_node_batch(scene.nodes,capacity);
            impl_->reserve_original_workers(static_cast<std::size_t>(impl_->original_worker_limit));
            const int fallback_workers=static_cast<int>(std::min<std::size_t>(scene.nodes,impl_->original_worker_limit));
            launch_resolve_nodes_original_batch(scene,state,bp,offsets,indices,boxes,colors,color,
                accepted,impl_->clip_raw,error,impl_->original_words,impl_->original_slots,
                impl_->original_worker_limit,fallback_workers,use_ccd,mode,convention,
                original_cooperative_selected,impl_->node_batch_contexts,impl_->node_batch_control,
                impl_->node_batch_active,impl_->node_batch_queue,capacity,impl_->node_csr_partition,impl_->node_cooperative,stream);
        } else {
        const int workers=impl_->original_workers(scene.nodes);
        if(workers>0)
            launch_resolve_nodes_original(scene,state,bp,
                offsets,indices,boxes,colors,color,accepted,impl_->clip_raw,error,
                impl_->original_words,impl_->original_slots,workers,
                use_ccd,mode,convention,original_cooperative_selected,stream);
        }
        check(cudaGetLastError());
        return;
    }
    if(!use_ccd) {
        // CPU ColoredContactSweep dispatch when validate_cooperative_step is
        // false (here: exact + CCD off). Selected cooperative vertices discard
        // a stored move with squared norm < 1e-28; the proposal kernel already
        // performed the finite and componentwise-box checks. The solver only
        // supplies the mask for this exact CPU rule, so an absent mask keeps
        // the historical no-filter behaviour.
        if(original_cooperative_selected) {
            validate_original_mask<<<unsigned((scene.nodes+127)/128),128,0,stream>>>(
                scene.nodes,original_cooperative_selected,error);
            apply_selected_small_step_cutoff<<<unsigned((scene.nodes+127)/128),128,0,stream>>>(
                scene,state,colors,color,accepted,impl_->clip_raw,
                original_cooperative_selected,error);
        }
        check(cudaGetLastError());
        return;
    }
    if(!bp.contact_count)return;
    if(bp.contact_count) {
        const auto count=color_pairs.offsets?color_pairs.max_color_pairs:bp.contact_count;
        const unsigned blocks=unsigned(std::max<std::size_t>(1,std::min<std::size_t>((count+127)/128,65535)));
        certify_node_contacts<<<blocks,128,0,stream>>>(scene,state,bp,colors,color,accepted,
            impl_->pending,impl_->initial_safe,error,color_pairs);check(cudaGetLastError());
    }
    if(exact_step_bound)
        launch_resolve_nodes_direct_bound(scene,state,bp,offsets,indices,boxes,impl_->pending,impl_->initial_safe,accepted,error,
            impl_->words,impl_->slots,stream);
    else
    launch_resolve_nodes(scene,state,bp,offsets,indices,boxes,impl_->pending,impl_->initial_safe,accepted,error,
        impl_->words,impl_->slots,stream);
    check(cudaGetLastError());
}

void GpuLinearClipper::clip_rigid_translation_color(const RigidScene&scene,const RigidState&state,
    const RigidWorkspace&work,const DeformableState&vertices,const BroadPhaseView&bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const int*colors,int color,
    double(*accepted)[3],int*error,bool use_ccd,cudaStream_t stream,LinearClipMode mode) {
    rigid_proxy_owner::Permit proxy_permit;
    if(const auto* bound=rigid_proxy_owner::Scope::binding(this)) {
        auto current=*bound;
        current.bodies=scene.bodies;current.positions=vertices.positions;
        current.com=state.com;current.proposed_com=work.proposed_com;
        current.accepted_com=work.accepted_com;current.accepted=accepted;
        current.boxes=boxes;current.colors=colors;current.error=error;
        current.node_offsets=scene.node_offsets;current.nodes=scene.nodes;
        current.node_body=scene.node_body;current.body_blocks=scene.body_blocks;
        current.translate=scene.translate;current.rotate=scene.rotate;
        proxy_permit=rigid_proxy_owner::Scope::consume(this,current,
            impl_->parallel_rigid_proxy&&mode==LinearClipMode::OriginalSolver&&use_ccd&&stream==nullptr);
    }
    if(!scene.bodies)return;
    if(mode==LinearClipMode::OriginalColoredGuess)
        throw std::invalid_argument("Rigid translation is not an original deformable predictor");
    if(mode==LinearClipMode::OriginalSolver) {
        const int threads=rigid_translation_pair_threads();
        if(contact_execution_option("IPC_GPU_RIGID_CCD_BATCH",true)) {
            const auto limit=static_cast<std::size_t>(ExactContactSchedule::environment_integer(
                "IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS",1<<26,1<<26));
            const auto queue_limit=static_cast<std::size_t>(ExactContactSchedule::environment_integer(
                "IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY",1<<20,1<<26));
            const auto capacity=std::min(bp.contact_count,queue_limit);
            impl_->reserve_rigid_batch(capacity);
            const int workers=impl_->original_worker_limit;
            impl_->reserve_original_workers(static_cast<std::size_t>(std::max(workers,threads)));
            launch_resolve_bodies_original_batch(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
                accepted,error,impl_->words,impl_->slots,impl_->original_words,impl_->original_slots,
                workers,threads,use_ccd,impl_->rigid_batch_context,
                impl_->rigid_batch_queue,limit,capacity,proxy_permit,stream);
        } else if(threads==1)
            launch_resolve_bodies_original(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
                accepted,error,impl_->words,impl_->slots,use_ccd,stream);
        else {
            impl_->reserve_original_workers(static_cast<std::size_t>(threads));
            launch_resolve_bodies_original_parallel(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
                accepted,error,impl_->original_words,impl_->original_slots,threads,use_ccd,stream);
        }
        check(cudaGetLastError());
        return;
    }
    launch_resolve_bodies(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
        accepted,error,impl_->words,impl_->slots,use_ccd,stream);check(cudaGetLastError());
}

} // namespace ipc_gpu::solver

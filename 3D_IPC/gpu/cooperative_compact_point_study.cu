#define IPC_GPU_TU_ACTIVE_POINT 1
#define IPC_GPU_TU_FINITE_CCD 1
#include "cooperative_compact_point_study.h"
#include "compact_point_prefix.cuh"
#include "direct_native_point.cuh"
#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ipc_gpu::solver {
namespace {
namespace cp=contact_cooperative_detail;
namespace pc=point_vertex_compact_prefix;
namespace ex=exact;
static_assert(cp::capacity==256&&cp::owner_slots==512&&cp::helper_slots==128);
static_assert(cp::active_lanes==9&&cp::slots_per_worker==1536&&pc::compact_capacity==8);
struct BlockState {
    cp::SharedGeometry geometry;
    ContactOutput* output;
    int pair,role;
    unsigned work,eligible,prepared,replay,corrupt;
};
static_assert(std::is_trivial<BlockState>::value);

__global__ void cooperative_compact_point_inplace_kernel(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* words,int* free_slots,int* failed,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity) {
    const unsigned lane=threadIdx.x,worker=blockIdx.x;
    // count is immutable for this stage, so this return is block-uniform and
    // occurs BEFORE arena access. Never add a per-lane failed-latch return:
    // another block can change that latch between lanes reaching a barrier.
    if(worker>=*count)return;
    __shared__ BlockState state;
    __shared__ __align__(16) unsigned char output_storage[sizeof(ContactOutput)];
    if(lane==0)state.output=new(output_storage)ContactOutput{};
    ex::DevicePool pool;
    const int private_slots=lane==0?cp::owner_slots:cp::helper_slots;
    const std::size_t offset=std::size_t(worker)*cp::slots_per_worker
        +(lane==0?0:cp::owner_slots+(lane-1)*cp::helper_slots);
    if(lane<cp::active_lanes)
        ex::initialize_pool(pool,words+offset*cp::capacity,free_slots+offset,cp::capacity,private_slots);
    using Owner=cp::GeometryOwner<pc::compact_capacity>;
    alignas(Owner) unsigned char owner_storage[sizeof(Owner)];
    Owner* owner=nullptr;
    pc::Gate gate; // Lane0 only; no Integer construction/current_pool lookup.
    __syncthreads(); // B0: initialized shared output and private pool contexts.
    for(;;) {
        if(lane==0) {
            state.work=0;state.eligible=0;state.prepared=0;state.replay=0;state.corrupt=0;
            if(!atomicAdd(failed,0)) {
                const unsigned index=atomicAdd(next,1u);
                if(index<*count) {
                    state.work=1;state.pair=queue[index];
                    state.role=cp::self_block(inputs[state.pair].hessian_blocks);
                    gate=pc::classify(inputs[state.pair],state.role);
                    state.eligible=gate.eligible();state.replay=!state.eligible;
                }
            }
        }
        __syncthreads(); // B1: block-uniform claim/gate/failure decision.
        if(!state.work)break;
        if(state.eligible&&lane<cp::active_lanes&&!pc::pool_admits(&pool,lane==0))
            atomicOr(&state.replay,1u);
        __syncthreads(); // B2: ALL owner/helper admission votes precede skipping original work.
        if(lane==0&&!state.replay) {
            // B2 has joined the ORIGINAL owner/helper admission vote. A true
            // native result is fully finalized and needs no exact owner/views.
            // False leaves shared output untouched and enters the old path.
            const bool native_complete=direct_native_point_study::try_evaluate(
                inputs[state.pair],state.role,gate,state.eligible&&!state.replay,*state.output);
            if(!native_complete) {
            owner=new(owner_storage)Owner();
            state.output->status=ContactStatus::NeedsExactGeometry;
            // Same serial preparation as the frozen scalar C8 hot path.
            state.prepared=pc::prepare<false,true>(inputs[state.pair],state.role,gate,
                *owner,*state.output,state.geometry);
            if(pool.failed||state.output->status!=ContactStatus::Ok) {
                state.replay=1;state.prepared=0;
            }
            }
        }
        __syncthreads(); // B3: publishes immutable owner limbs and prepared POD views.
        if(state.prepared&&lane<cp::active_lanes) {
            const auto result=cp::compute_entry<pc::compact_capacity,true>(state.geometry,int(lane));
            state.output->barrier_hessian[state.role][state.role][lane]=result.value;
            if(!result.ok||pool.failed)atomicOr(&state.replay,1u);
        }
        __syncthreads(); // B4: every borrowed adapter/entry temporary has died.
        if(lane==0) {
            if(state.prepared&&!state.replay) {
                cp::finish_output(inputs[state.pair],*owner,*state.output);
                if(state.output->status!=ContactStatus::Ok)state.replay=1;
            }
            if(owner){owner->~Owner();owner=nullptr;}
        }
        __syncthreads(); // B5: owner destruction complete; no borrowed pointer can be used.
        if(lane<cp::active_lanes) {
            // Detect lost slots BEFORE reset, exactly as original cooperative recovery.
            if(pool.available!=private_slots)atomicOr(&state.corrupt,1u);
            if(pool.failed) {
                atomicOr(&state.replay,1u);
                ex::initialize_pool(pool,words+offset*cp::capacity,free_slots+offset,cp::capacity,private_slots);
            }
            if(pool.failed||pool.available!=private_slots)atomicOr(&state.corrupt,1u);
        }
        __syncthreads(); // B6: every active private pool has been checked/recovered.
        if(lane==0) {
            if(state.corrupt)atomicCAS(failed,0,24);
            else if(state.replay||state.output->status!=ContactStatus::Ok) {
                const unsigned index=atomicAdd(fallback_count,1u);
                fallback_queue[index]=state.pair;
            } else {
                // prepare=false/Ok is the identical original inactive output.
                store_contact_output(inputs[state.pair],*state.output,outputs+state.pair,
                    compact_outputs?compact_outputs+state.pair:nullptr,
                    output_activity?output_activity+state.pair:nullptr);
            }
        }
        __syncthreads(); // B7: publication complete before the next query resets shared state.
    }
    if(lane<cp::active_lanes)ex::release_pool();
}
} // namespace

void launch_cooperative_compact_point_inplace_study(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* cooperative_words,int* cooperative_slots,unsigned workers,
    int* failed,cudaStream_t stream,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity) {
    if(!workers)return;
    if(workers>256)throw std::invalid_argument("cooperative C8 study requires original cooperative worker bound");
    if(!inputs||!outputs||!queue||!count||!fallback_queue||!fallback_count||!next||!cooperative_words||!cooperative_slots||!failed)
        throw std::invalid_argument("cooperative C8 study requires all buffers");
    cooperative_compact_point_inplace_kernel<<<workers,32,0,stream>>>(inputs,outputs,
        queue,count,fallback_queue,fallback_count,next,cooperative_words,cooperative_slots,failed,compact_outputs,output_activity);
}
} // namespace ipc_gpu::solver

#include "resident_contact_compact_point.h"
#include "compact_point_prefix.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace ipc_gpu::resident {
namespace {
namespace cp=contact_cooperative_detail;
namespace pc=point_vertex_compact_prefix;
namespace ex=resident_exact;
constexpr int arena_width=256,arena_slots=512;
static_assert(cp::capacity==arena_width&&cp::owner_slots==arena_slots);
static_assert(pc::compact_capacity==8);

template<bool InplaceRatio>
__device__ bool evaluate_compact_point(const ContactInput& in,ContactOutput& out,
    ex::DevicePool& pool) {
    const int role=cp::self_block(in.hessian_blocks);
    const auto gate=pc::classify(in,role);
    if(!gate.eligible()||!pc::pool_admits(&pool,true))return false;
    bool okay=true;
    {
        // Admission precedes construction. The fresh production pool has512
        // slots; the conservative gate permits at least256 before this41-slot
        // owner. No owner/view survives the scope or is copied to another lane.
        cp::GeometryOwner<pc::compact_capacity> owner;
        cp::SharedGeometry shared{};
        // A defensive gate/precondition rejection must not look like an
        // inactive success. Valid prepare resets the complete output itself.
        out.status=ContactStatus::NeedsExactGeometry;
        const bool prepared=pc::prepare<false,InplaceRatio>(in,role,gate,owner,out,shared);
        if(prepared) {
            for(int entry=0;entry<9;++entry) {
                const auto result=cp::compute_entry<pc::compact_capacity,InplaceRatio>(shared,entry);
                out.barrier_hessian[role][role][entry]=result.value;
                if(!result.ok||pool.failed){okay=false;break;}
            }
            if(okay&&!pool.failed)cp::finish_output(in,owner,out);
        }
        // prepare=false,statusOk is the original successful inactive result.
        // Any failed/nonfinite preparation/entry/finish is replayed wholesale.
        okay=okay&&!pool.failed&&out.status==ContactStatus::Ok;
    }
    return okay; // All41 owner Integers and every temporary have died.
}

template<bool InplaceRatio>
__global__ void compact_point_contacts(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* queue_count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* words,int* free_slots,int* failed,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned count=*queue_count;
    if(worker>=count||atomicAdd(failed,0))return;
    ex::DevicePool pool;
    auto* private_words=words+std::size_t(worker)*arena_width*arena_slots;
    auto* private_slots=free_slots+std::size_t(worker)*arena_slots;
    ex::initialize_pool(pool,private_words,private_slots,arena_width,arena_slots);
    for(;;) {
        if(atomicAdd(failed,0))break;
        const unsigned index=atomicAdd(next,1u);
        if(index>=count)break;
        const int pair=queue[index];
        ContactOutput result{};
        bool publish=evaluate_compact_point<InplaceRatio>(inputs[pair],result,pool);
        // Match cooperative recovery: never hide lost ownership; recover an
        // exhausted but completely released pool before original-path replay.
        if(pool.available!=pool.slot_count) {atomicCAS(failed,0,24);break;}
        if(pool.failed) {
            ex::initialize_pool(pool,private_words,private_slots,arena_width,arena_slots);
            publish=false;
        }
        if(pool.failed||pool.available!=arena_slots) {atomicCAS(failed,0,24);break;}
        if(!publish) {
            const unsigned destination=atomicAdd(fallback_count,1u);
            fallback_queue[destination]=pair;
        } else {
            store_contact_output(inputs[pair],result,outputs+pair,
                compact_outputs?compact_outputs+pair:nullptr,
                output_activity?output_activity+pair:nullptr);
        }
    }
    ex::release_pool();
}
} // namespace

void launch_compact_point_contacts(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* words,int* free_slots,unsigned workers,unsigned threads,
    int* failed,cudaStream_t stream,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity,
    bool inplace_ratio) {
    if(!workers)return;
    if(workers>4096||!threads||threads>32||(threads&(threads-1))||workers%threads)
        throw std::invalid_argument("compact point first-tier schedule must preserve original scalar worker bounds");
    if(!inputs||!outputs||!queue||!count||!fallback_queue||!fallback_count||!next||!words||!free_slots||!failed)
        throw std::invalid_argument("compact point first-tier launch requires all buffers");
    if(inplace_ratio)
        compact_point_contacts<true><<<workers/threads,threads,0,stream>>>(inputs,outputs,
            queue,count,fallback_queue,fallback_count,next,words,free_slots,failed,compact_outputs,output_activity);
    else
        compact_point_contacts<false><<<workers/threads,threads,0,stream>>>(inputs,outputs,
            queue,count,fallback_queue,fallback_count,next,words,free_slots,failed,compact_outputs,output_activity);
}
} // namespace ipc_gpu::resident

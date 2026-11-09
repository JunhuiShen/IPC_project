#pragma once
#include "linear_pipeline.h"
#include "node_cooperative_controller.h"
#include "rigid_translation_proxy_owner.h"
#include <cstdint>

namespace ipc_gpu::solver::linear_detail {
#if defined(__CUDACC__)
#define IPC_RIGID_BATCH_HD __host__ __device__
#else
#define IPC_RIGID_BATCH_HD
#endif
struct RigidCcdBatchVerdict {
    int error;bool collision,safe;unsigned char reserved[2];double toi;
    IPC_RIGID_BATCH_HD RigidCcdBatchVerdict(int e=0,bool hit=false,bool accepted=true,double t=1.0)
        :error(e),collision(hit),safe(accepted),reserved{0,0},toi(t){}
};
#undef IPC_RIGID_BATCH_HD
static_assert(sizeof(RigidCcdBatchVerdict)==16,"bounded rigid CCD batch result ABI");
// Device-only stage state. Normal dispatch never downloads these fields.
struct RigidCcdBatchContext {
    int mode=0,done=0,body=-1,zero_step=0,safe=0,do_pairs=0,phase=0;
    int attempt=0,internal_error=0;
    std::size_t first=0,count=0;
    unsigned queue_count=0,queue_next=0,queue_overflow=0,hit=0;
    unsigned long long first_bad=~0ull;
    double raw[3]{},delta[3]{},toi=1.0,step=1.0;
};
struct OriginalNodeCcdContext {
    int active=0,done=0,phase=0,error=0,boxed=1;unsigned hit=0;
    std::size_t first=0,count=0;
    unsigned long long first_bad=~0ull;
    double start[3]{},raw[3]{},trial[3]{},initial_accepted[3]{};
    double toi=1.0,step=1.0;
};
struct OriginalNodeCcdControl {
    int mode=1,done=0,internal_error=0;
    unsigned active_count=0,pending=0,queue_overflow=0,reserved=0;
    unsigned long long queue_count=0,queue_next=0;
};
static_assert(sizeof(OriginalNodeCcdContext)==160,"original node context layout");
static_assert(sizeof(OriginalNodeCcdControl)==48,"original node control layout");
void launch_resolve_nodes(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots,cudaStream_t stream);

void launch_resolve_nodes_direct_bound(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots,cudaStream_t stream);

void launch_resolve_nodes_original(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t*offsets,const int*indices,const BpAabb*boxes,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int*error,
    std::uint32_t*words,int*slots,int workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char*selected,cudaStream_t stream);

// Resource/capability metadata only; no numerical state is downloaded.
node_controller::Plan query_node_cooperative_plan(std::size_t nodes,int workers);


void launch_resolve_nodes_original_batch(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t*offsets,const int*indices,const BpAabb*boxes,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int*error,
    std::uint32_t*words,int*slots,int workers,int fallback_workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char*selected,OriginalNodeCcdContext*contexts,
    OriginalNodeCcdControl*control,unsigned*active_nodes,unsigned long long*queue,
    std::size_t queue_capacity,bool node_csr_partition,bool cooperative_controller,cudaStream_t stream);

void launch_resolve_bodies(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd,cudaStream_t stream);

void launch_resolve_bodies_original(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd,cudaStream_t stream);

void launch_resolve_bodies_original_parallel(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,int threads,bool use_ccd,cudaStream_t stream);

void launch_resolve_bodies_original_batch(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*serial_words,int*serial_slots,std::uint32_t*worker_words,int*worker_slots,
    int workers,int fallback_threads,bool use_ccd,RigidCcdBatchContext*context,
    unsigned*queue,std::size_t row_limit,std::size_t queue_capacity,
    rigid_proxy_owner::Permit proxy_permit,cudaStream_t stream);

} // namespace ipc_gpu::solver::linear_detail

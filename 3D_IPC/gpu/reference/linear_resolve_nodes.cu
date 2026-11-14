#define IPC_GPU_TU_FINITE_CCD 1
#include "exact.cuh"
#include "linear_pipeline_internal.cuh"

namespace ipc_gpu::solver {
namespace {
namespace exact=ipc_gpu::exact;

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
} // namespace

namespace linear_detail {
void launch_resolve_nodes(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots,cudaStream_t stream) {
    resolve_nodes<<<1,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,pending,initial_safe,accepted,error,words,slots);
}
} // namespace linear_detail
} // namespace ipc_gpu::solver

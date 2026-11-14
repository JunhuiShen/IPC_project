#define IPC_GPU_TU_FINITE_CCD 1
#include "exact.cuh"
#include "direct_bound.cuh"
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

__host__ __device__ inline bool useful_direct_proposal(double fraction,
    const double* start,const double* target,const double* trial) {
    if(!(fraction>0.&&fraction<1.))return false;
    bool moved=false,different_target=false;
    for(int k=0;k<3;++k) {
        moved=moved||trial[k]!=start[k];
        different_target=different_target||trial[k]!=target[k];
    }
    return moved&&different_target;
}

__device__ bool checked_direct_endpoint(DeformableState state,BroadPhaseView bp,
    const std::size_t* offsets,const int* indices,int block,int node,
    const double* trial,const BpAabb& box,int* error) {
    if(!inside(trial,box))return false;
    for(auto pair=offsets[block];pair<offsets[block+1];++pair) {
        const auto q=node_query(bp.contacts[indices[pair]],state.positions,node,trial);
        const auto quick=cert::validate_linear(q);
        if(quick==cert::Status::Safe)continue;
        const auto status=quick==cert::Status::Invalid?exact::Status::Invalid:exact_endpoint(q);
        if(map_failure(status,error)||status==exact::Status::Unsafe)return false;
    }
    return true;
}

__global__ void resolve_nodes_direct_bound(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots) {
    if(blockIdx.x||threadIdx.x||*error||!pending[scene.nodes])return;
    namespace direct=ipc_gpu::direct_bound;
    exact::DevicePool pool;exact::initialize_pool(pool,words,slots,arena_width,arena_slots);
    for(std::size_t node=0;node<scene.nodes&&!*error;++node)if(pending[node]) {
        const int block=scene.node_blocks[node];double target[3],delta[3];
        for(int k=0;k<3;++k) { target[k]=accepted[node][k];delta[k]=target[k]-state.positions[node][k]; }
        bool rejected=false,hint_available=true,complete=false;
        double hint=1.;
        // The original represented endpoint is checked before requesting a
        // proposal. Only contacts which reject it contribute direct bounds.
        for(auto pair=offsets[block];pair<offsets[block+1];++pair) {
            const int index=indices[pair];
            if(initial_safe[index])continue;
            const auto q=node_query(bp.contacts[index],state.positions,int(node),target);
            const auto status=exact_endpoint(q);
            if(map_failure(status,error))break;
            if(status!=exact::Status::Unsafe)continue;
            rejected=true;
            const auto bound=direct::separation_step_bound<arena_width>(q.start,q.end,q.vertex_face,q.gap_floor);
            if(bound.status==direct::BoundStatus::InvalidResourceContext||pool.failed
                ||pool.available!=pool.slot_count) {
                fail(error,LinearClipError::ExactResources);break;
            }
            // Zero, malformed or resource-inconclusive hints retain the old
            // backtracking schedule. In particular, zero cannot lock a node.
            if(bound.status!=direct::BoundStatus::Computed||!finite(bound.fraction)
                ||bound.fraction<0.||bound.fraction>1.)hint_available=false;
            else hint=fmin(hint,bound.fraction);
        }
        if(*error)break;
        if(!rejected)complete=true;
        if(rejected&&hint_available&&hint>0.&&hint<1.) {
            const double fraction=nextafter(hint,0.);
            double trial[3];for(int k=0;k<3;++k)trial[k]=fma(fraction,delta[k],state.positions[node][k]);
            // Every incident contact is rechecked at the actual rounded
            // endpoint; even original Safe certificates are invalid here.
            if(useful_direct_proposal(fraction,state.positions[node],target,trial)
                &&checked_direct_endpoint(state,bp,offsets,indices,block,int(node),trial,boxes[node],error)) {
                for(int k=0;k<3;++k)accepted[node][k]=trial[k];complete=true;
            }
        }
        // An unsuccessful proposal does not narrow or replace the original
        // interval. Preserve its 64 halvings, final strict start-state check,
        // FMA endpoints, error mapping and original-proposal certificates.
        double fraction=1.;
        for(int attempt=0;!complete&&!*error&&attempt<=64;++attempt) {
            double trial[3];for(int k=0;k<3;++k)trial[k]=attempt==64?state.positions[node][k]
                :attempt?fma(fraction,delta[k],state.positions[node][k]):target[k];
            bool safe=inside(trial,boxes[node]);
            if(safe)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const int pair_index=indices[pair];
                if(attempt==0&&initial_safe[pair_index])continue;
                const auto q=node_query(bp.contacts[pair_index],state.positions,int(node),trial);
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
void launch_resolve_nodes_direct_bound(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const unsigned int*pending,
    const unsigned char*initial_safe,
    double(*accepted)[3],int*error,std::uint32_t*words,int*slots,cudaStream_t stream) {
    resolve_nodes_direct_bound<<<1,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,pending,initial_safe,accepted,error,words,slots);
}
} // namespace linear_detail
} // namespace ipc_gpu::solver

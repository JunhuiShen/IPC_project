#define IPC_GPU_TU_FINITE_CCD 1
#include "exact.cuh"
#include "original_ccd.cuh"
#include "original_guess.cuh"
#include "linear_pipeline_internal.cuh"
#include "contact_execution.h"
#include "node_cooperative_grid.h"
#include "node_csr_partition.h"
#include "node_cooperative_controller.h"
#include <cooperative_groups.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ipc_gpu::solver {
namespace {
namespace exact=ipc_gpu::exact;
namespace occd=ipc_gpu::original_ccd;

__device__ occd::Result original_fast_exact(const occd::Query& query,
    exact::DevicePool& pool) {
    occd::Result result=occd::evaluate_fast(query);
    if(result.status!=occd::Status::NeedsExact)return result;
    // Bounded 256 -> 2048 retry only with a healthy pool ownership state.
    result=occd::evaluate_exact<256>(query);
    if(result.status==occd::Status::CapacityExceeded
        &&!pool.failed&&pool.available==pool.slot_count)
        result=occd::evaluate_exact<2048>(query);
    return result;
}

__device__ bool original_single_active_vertex(const BpContact& contact,
    DeformableScene scene,const int* colors,int color,int node,int* error) {
    for(int role=0;role<4;++role) {
        const int vertex=contact.vertices[role];
        if(vertex!=node&&scene.deformable[vertex]
            &&colors[scene.node_blocks[vertex]]==color) {
            fail(error,LinearClipError::UnsupportedMotion);return false;
        }
    }
    return true;
}

__device__ exact::Status original_represented_path(const double start[4][3],
    const double end[4][3],bool face,exact::DevicePool& pool) {
    cert::LinearQuery q;q.vertex_face=face;
    bool finite_inputs=true;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        q.start[i][k]=start[i][k];q.end[i][k]=end[i][k];
        finite_inputs=finite_inputs&&finite(start[i][k])&&finite(end[i][k]);
    }
    if(finite_inputs&&cert::detail::separated(q,0.0,1.0,0.0))
        return exact::Status::Safe;
    // One moving vertex. Bounded 256 -> 2048 retry only with a healthy arena;
    // an arithmetic/resource/ownership failure is propagated explicitly and is
    // never turned into a geometric no-op or an accepted endpoint.
    auto status=exact::validate_collision_step<256>(start,end,face);
    if(status==exact::Status::CapacityExceeded
        &&!pool.failed&&pool.available==pool.slot_count)
        status=exact::validate_collision_step<2048>(start,end,face);
    return status;
}

__global__ void resolve_nodes_original(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t*offsets,const int*indices,const BpAabb*boxes,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int*error,
    std::uint32_t*words,int*slots,int workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char*selected,const OriginalNodeCcdControl* gate) {
    if(gate&&gate->mode!=0)return;
    const int worker=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(worker>=workers||!scene.nodes)return;
    if(atomicAdd(error,0))return;
    const bool cooperative=convention==OriginalUpdateConvention::CooperativeSquaredNorm;
    if(!colors) { fail(error,LinearClipError::InvalidInput);return; }
    // The {0,1} mask contract was already checked once per call by
    // validate_original_mask; workers never rescan it.
    if(cooperative&&!selected) { fail(error,LinearClipError::InvalidInput);return; }
    exact::DevicePool pool;
    exact::initialize_pool(pool,
        words+std::size_t(worker)*original_arena_width*original_arena_slots,
        slots+std::size_t(worker)*original_arena_slots,
        original_arena_width,original_arena_slots);
    const auto map_status=[&](exact::Status status) -> bool {
        if(status==exact::Status::Safe||status==exact::Status::Unsafe)return false;
        if(status==exact::Status::InitialContact)fail(error,LinearClipError::InitialContact);
        else if(status==exact::Status::CapacityExceeded)fail(error,LinearClipError::ExactResources);
        else fail(error,LinearClipError::InvalidInput);
        return true;
    };
    for(std::size_t node=std::size_t(worker);node<scene.nodes;
        node+=std::size_t(workers)) {
        // Atomic read: another worker's failure must stop this worker promptly,
        // and an ordinary load could keep a stale L1 line.
        if(atomicAdd(error,0))break;
        const int block=scene.node_blocks[node];
        // Only this color's deformable nodes were proposed; leave every other
        // node's accepted coordinate untouched.
        if(!scene.deformable[node]||colors[block]!=color)continue;
        double start[3],raw[3];
        for(int k=0;k<3;++k) {
            start[k]=state.positions[node][k];
            raw[k]=raw_delta[node][k];
            if(!finite(start[k])||!finite(raw[k])) {
                fail(error,LinearClipError::InvalidInput);break;
            }
        }
        if(*error)break;
        bool moved=false;
        for(int k=0;k<3;++k)moved=moved||raw[k]!=0.0;
        if(!moved) {
            // Keep the CPU's no-op TOI shortcut, but retain the additional
            // represented-state health check when CCD was requested. A true
            // initial intersection must not be accepted merely because this
            // node has no stored motion; healthy positive gaps remain valid.
            if(use_ccd)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const auto& contact=bp.contacts[indices[pair]];
                if(!original_single_active_vertex(contact,scene,colors,color,int(node),error))break;
                double unchanged[4][3];
                for(int i=0;i<4;++i)for(int k=0;k<3;++k)
                    unchanged[i][k]=state.positions[contact.vertices[i]][k];
                const auto status=original_represented_path(
                    unchanged,unchanged,contact.kind==0,pool);
                if(status==exact::Status::Safe)continue;
                if(map_status(status))break;
                fail(error,LinearClipError::InvalidInput);break;
            }
            if(atomicAdd(error,0))break;
            if(pool.failed||pool.available!=pool.slot_count) {
                fail(error,LinearClipError::ExactResources);break;
            }
            for(int k=0;k<3;++k)accepted[node][k]=start[k];
            continue;
        }
        const bool selected_cutoff=
            mode==LinearClipMode::OriginalSolver&&cooperative&&selected[node]!=0;
        if(selected_cutoff) {
            double squared=0.0;
            for(int k=0;k<3;++k)squared+=raw[k]*raw[k];
            if(squared<1e-28) {
                // CPU selected cooperative false vertex: stored move < 1e-28
                // squared is a zero step. With CCD disabled this keeps the
                // requested finite/box-only semantics. With CCD enabled a
                // no-op must not silently bless a true initial intersection,
                // so certify the unchanged start geometry (InitialContact is
                // then the explicit initial-contact error).
                if(use_ccd)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                    const auto&contact=bp.contacts[indices[pair]];
                    if(!original_single_active_vertex(contact,scene,colors,color,int(node),error))break;
                    double s[4][3];
                    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
                        s[i][k]=state.positions[contact.vertices[i]][k];
                    const auto status=original_represented_path(s,s,contact.kind==0,pool);
                    if(status==exact::Status::Safe)continue;
                    if(map_status(status))break;
                    fail(error,LinearClipError::InvalidInput);break;
                }
                if(*error)break;
                for(int k=0;k<3;++k)accepted[node][k]=start[k];
                continue;
            }
        }
        bool hit=false;
        double toi=1.0;
        if(use_ccd)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
            const auto&contact=bp.contacts[indices[pair]];
            if(!original_single_active_vertex(contact,scene,colors,color,int(node),error))break;
            occd::Query query;
            query.vertex_face=contact.kind==0;
            query.eps=1e-12;
            for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
                query.positions[i][k]=state.positions[contact.vertices[i]][k];
                query.motion[i][k]=contact.vertices[i]==int(node)?raw[k]:0.0;
            }
            const auto result=original_fast_exact(query,pool);
            if(result.status==occd::Status::InvalidInput) {
                fail(error,LinearClipError::InvalidInput);break;
            }
            if(result.status!=occd::Status::Ok) {
                // NeedsExact after a completed bounded retry or any capacity/
                // ownership failure is a resource error, never a miss.
                fail(error,LinearClipError::ExactResources);break;
            }
            if(result.collision) { hit=true;toi=fmin(toi,result.toi); }
        }
        if(*error)break;
        double step=hit?0.9*toi:1.0;
        // Normalize signed zero so a zero TOI cannot introduce -0.0 motion.
        if(step==0.0)step=0.0;
        if(!use_ccd) {
            // Requested no-CCD: commit the stored raw displacement unchanged,
            // with finite/box checks only.
            double trial[3];
            for(int k=0;k<3;++k)
                trial[k]=__dadd_rn(start[k],raw[k]);
            if(!inside(trial,boxes[node])) { fail(error,LinearClipError::OutsideBlueBox);break; }
            for(int k=0;k<3;++k)accepted[node][k]=trial[k];
            continue;
        }
        // CPU original colored predictor: after each positive original step,
        // preserve each pair's own min(startgap, 1e-8) using the corrected CPU
        // point-triangle and false edge-edge distances. This floor is NOT
        // applied to ordinary OriginalSolver steps.
        const auto apply_predictor_floor=[&](const double* s,double* trial) {
            bool far=true;
            for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const auto&contact=bp.contacts[indices[pair]];
                const double d=original_guess::incident_distance(
                    contact,int(node),state.positions,trial);
                if(!finite(d)||!(d>=original_guess::kMinimumSeparation)) {
                    far=false;break;
                }
            }
            if(far)return;
            const auto acceptable=[&](const double* cand) {
                for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                    const auto&contact=bp.contacts[indices[pair]];
                    const double d0=original_guess::incident_distance(
                        contact,int(node),state.positions,s);
                    if(!finite(d0))return false;
                    const double required=d0<original_guess::kMinimumSeparation
                        ?d0:original_guess::kMinimumSeparation;
                    const double d=original_guess::incident_distance(
                        contact,int(node),state.positions,cand);
                    if(!finite(d)||!(d>=required))return false;
                }
                return true;
            };
            // Full endpoint first; then a tested-endpoint refinement whose
            // bracket is fixed at 48 attempts and 1e-14 * stableNorm(dx).
            if(acceptable(trial))return;
            double displacement[3];
            for(int k=0;k<3;++k)displacement[k]=trial[k]-s[k];
            const double length=level_predictor::stable_norm3(displacement);
            double lower=0.0,upper=1.0,best[3];
            for(int k=0;k<3;++k)best[k]=s[k];
            for(int attempt=0;attempt<48;++attempt) {
                if(!((upper-lower)*length>1e-14))break;
                const double weight=0.5*(lower+upper);
                double candidate[3];
                for(int k=0;k<3;++k)candidate[k]=s[k]+weight*displacement[k];
                if(acceptable(candidate)) {
                    lower=weight;
                    for(int k=0;k<3;++k)best[k]=candidate[k];
                } else upper=weight;
            }
            for(int k=0;k<3;++k)trial[k]=best[k];
        };
        bool complete=false;
        // Existing fixed safety budget: 64 halvings plus the unchanged start
        // candidate. Only a collision-only Safe represented path is accepted.
        for(int attempt=0;attempt<=64&&!*error;++attempt) {
            if(attempt==64)step=0.0;
            double trial[3];
            for(int k=0;k<3;++k)
                trial[k]=__dadd_rn(start[k],__dmul_rn(step,raw[k]));
            if(mode==LinearClipMode::OriginalColoredGuess&&step>0.0)
                apply_predictor_floor(start,trial);
            bool ok=inside(trial,boxes[node]);
            if(ok)for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const auto&contact=bp.contacts[indices[pair]];
                double endpoint_start[4][3],endpoint_end[4][3];
                for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
                    const int vertex=contact.vertices[i];
                    endpoint_start[i][k]=state.positions[vertex][k];
                    endpoint_end[i][k]=vertex==int(node)?trial[k]:endpoint_start[i][k];
                }
                // The fast zero-gap certificate may accept only a PROVED
                // collision-free represented path; an inconclusive result goes
                // to the unchanged exact collision-only predicate.
                const auto status=original_represented_path(
                    endpoint_start,endpoint_end,contact.kind==0,pool);
                if(map_status(status)) { ok=false;break; }
                if(status==exact::Status::Unsafe) { ok=false;break; }
            }
            if(*error)break;
            if(ok) {
                for(int k=0;k<3;++k)accepted[node][k]=trial[k];
                complete=true;break;
            }
            step*=0.5;
        }
        if(!complete&&!*error)fail(error,LinearClipError::Unresolved);
        if(pool.failed||pool.available!=pool.slot_count)
            fail(error,LinearClipError::ExactResources);
    }
    exact::release_pool();
}
enum NodeBatchPhase : int {NodeRaw=0,NodeHealth=1,NodeDirect=2,NodeTrial=3,NodeError=4};
using NodeVerdict=RigidCcdBatchVerdict;
__device__ int node_ownership_error(const BpContact& contact,DeformableScene scene,
    const int* colors,int color,int node) {
    for(int role=0;role<4;++role) {
        const int vertex=contact.vertices[role];
        if(vertex!=node&&scene.deformable[vertex]&&colors[scene.node_blocks[vertex]]==color)
            return int(LinearClipError::UnsupportedMotion);
    }
    return 0;
}
__device__ int node_status_error(exact::Status status) {
    if(status==exact::Status::Safe||status==exact::Status::Unsafe)return 0;
    return int(status==exact::Status::InitialContact?LinearClipError::InitialContact
        :status==exact::Status::CapacityExceeded?LinearClipError::ExactResources:LinearClipError::InvalidInput);
}
struct NodeBatchPool {
    exact::DevicePool pool;
    __device__ NodeBatchPool(std::uint32_t* words,int* slots,unsigned worker) {
        exact::initialize_pool(pool,words+std::size_t(worker)*original_arena_width*original_arena_slots,
            slots+std::size_t(worker)*original_arena_slots,original_arena_width,original_arena_slots);
    }
    __device__ NodeVerdict finish(NodeVerdict verdict) {
        if(!verdict.error&&(pool.failed||pool.available!=pool.slot_count))verdict.error=int(LinearClipError::ExactResources);
        return verdict;
    }
    __device__ ~NodeBatchPool(){exact::release_pool();}
};
__device__ occd::Query node_original_query(const BpContact& contact,DeformableState state,
    unsigned node,const double* raw) {
    occd::Query query;query.vertex_face=contact.kind==0;query.eps=1e-12;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        query.positions[i][k]=state.positions[contact.vertices[i]][k];
        query.motion[i][k]=contact.vertices[i]==int(node)?raw[k]:0.0;
    }
    return query;
}
__device__ void node_endpoint_query(const BpContact& contact,DeformableState state,
    unsigned node,const double* trial,cert::LinearQuery& q) {
    q.vertex_face=contact.kind==0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const int vertex=contact.vertices[i];q.start[i][k]=state.positions[vertex][k];
        q.end[i][k]=vertex==int(node)?trial[k]:q.start[i][k];
    }
}
__device__ NodeVerdict node_fast_pair(const BpContact& contact,DeformableScene scene,
    DeformableState state,const int* colors,int color,unsigned node,const OriginalNodeCcdContext& ctx,
    bool& unresolved) {
    NodeVerdict verdict;unresolved=false;
    if(ctx.phase==NodeRaw||ctx.phase==NodeHealth) {
        verdict.error=node_ownership_error(contact,scene,colors,color,int(node));
        if(verdict.error)return verdict;
    }
    if(ctx.phase==NodeRaw) {
        const auto result=occd::evaluate_fast(node_original_query(contact,state,node,ctx.raw));
        if(result.status==occd::Status::NeedsExact){unresolved=true;return verdict;}
        if(result.status!=occd::Status::Ok)verdict.error=int(result.status==occd::Status::InvalidInput
            ?LinearClipError::InvalidInput:LinearClipError::ExactResources);
        verdict.collision=result.collision;verdict.toi=result.toi;return verdict;
    }
    cert::LinearQuery q;node_endpoint_query(contact,state,node,ctx.trial,q);
    bool finite_inputs=true;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        finite_inputs=finite_inputs&&finite(q.start[i][k])&&finite(q.end[i][k]);
    if(finite_inputs&&cert::detail::separated(q,0.0,1.0,0.0))return verdict;
    unresolved=true;return verdict;
}
__device__ NodeVerdict node_exact_pair(const BpContact& contact,DeformableState state,
    unsigned node,const OriginalNodeCcdContext& ctx,std::uint32_t* words,int* slots,unsigned worker) {
    NodeVerdict verdict;
    if(ctx.phase==NodeRaw) {
        const auto query=node_original_query(contact,state,node,ctx.raw);
        auto result=occd::evaluate_fast(query);
        if(result.status==occd::Status::NeedsExact) {
            NodeBatchPool arena(words,slots,worker);
            result=occd::evaluate_exact<256>(query);
            if(result.status==occd::Status::CapacityExceeded&&!arena.pool.failed
                &&arena.pool.available==arena.pool.slot_count)result=occd::evaluate_exact<2048>(query);
            if(result.status!=occd::Status::Ok)verdict.error=int(result.status==occd::Status::InvalidInput
                ?LinearClipError::InvalidInput:LinearClipError::ExactResources);
            verdict.collision=result.collision;verdict.toi=result.toi;return arena.finish(verdict);
        }
        if(result.status!=occd::Status::Ok)verdict.error=int(result.status==occd::Status::InvalidInput
            ?LinearClipError::InvalidInput:LinearClipError::ExactResources);
        verdict.collision=result.collision;verdict.toi=result.toi;return verdict;
    }
    cert::LinearQuery q;node_endpoint_query(contact,state,node,ctx.trial,q);
    NodeBatchPool arena(words,slots,worker);
    const auto status=original_represented_path(q.start,q.end,q.vertex_face,arena.pool);
    verdict.error=node_status_error(status);verdict.safe=status==exact::Status::Safe;
    if(ctx.phase==NodeHealth&&status==exact::Status::Unsafe)verdict.error=int(LinearClipError::InvalidInput);
    return arena.finish(verdict);
}
__device__ void node_atomic_min(double* destination,double value) {
    auto* bits=reinterpret_cast<unsigned long long*>(destination);
    auto observed=atomicCAS(bits,0ull,0ull);
    while(true) {
        const auto desired=static_cast<unsigned long long>(__double_as_longlong(
            fmin(__longlong_as_double(static_cast<long long>(observed)),value)));
        if(desired==observed)return;
        const auto previous=atomicCAS(bits,observed,desired);
        if(previous==observed)return;observed=previous;
    }
}
__device__ void node_publish(unsigned rank,const NodeVerdict& verdict,OriginalNodeCcdContext& ctx) {
    if(verdict.error||(ctx.phase!=NodeRaw&&!verdict.safe)) {
        atomicMin(&ctx.first_bad,(static_cast<unsigned long long>(rank)<<32)|static_cast<unsigned>(verdict.error));
    } else if(ctx.phase==NodeRaw&&verdict.collision) {
        atomicOr(&ctx.hit,1u);node_atomic_min(&ctx.toi,verdict.toi);
    }
}

__device__ void node_predictor_floor(DeformableState state,BroadPhaseView bp,
    const std::size_t* offsets,const int* indices,int block,int node,const double* s,double* trial) {
    bool far=true;
    for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
        const auto&contact=bp.contacts[indices[pair]];
        const double d=original_guess::incident_distance(
            contact,int(node),state.positions,trial);
        if(!finite(d)||!(d>=original_guess::kMinimumSeparation)) {
            far=false;break;
        }
    }
    if(far)return;
    const auto acceptable=[&](const double* cand) {
        for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
            const auto&contact=bp.contacts[indices[pair]];
            const double d0=original_guess::incident_distance(
                contact,int(node),state.positions,s);
            if(!finite(d0))return false;
            const double required=d0<original_guess::kMinimumSeparation
                ?d0:original_guess::kMinimumSeparation;
            const double d=original_guess::incident_distance(
                contact,int(node),state.positions,cand);
            if(!finite(d)||!(d>=required))return false;
        }
        return true;
    };
    // Full endpoint first; then a tested-endpoint refinement whose
    // bracket is fixed at 48 attempts and 1e-14 * stableNorm(dx).
    if(acceptable(trial))return;
    double displacement[3];
    for(int k=0;k<3;++k)displacement[k]=trial[k]-s[k];
    const double length=level_predictor::stable_norm3(displacement);
    double lower=0.0,upper=1.0,best[3];
    for(int k=0;k<3;++k)best[k]=s[k];
    for(int attempt=0;attempt<48;++attempt) {
        if(!((upper-lower)*length>1e-14))break;
        const double weight=0.5*(lower+upper);
        double candidate[3];
        for(int k=0;k<3;++k)candidate[k]=s[k]+weight*displacement[k];
        if(acceptable(candidate)) {
            lower=weight;
            for(int k=0;k<3;++k)best[k]=candidate[k];
        } else upper=weight;
    }
    for(int k=0;k<3;++k)trial[k]=best[k];
}


__global__ void node_batch_begin(const int* colors,OriginalUpdateConvention convention,
    const unsigned char* selected,int* error,OriginalNodeCcdControl* control) {
    if(blockIdx.x||threadIdx.x)return;*control=OriginalNodeCcdControl{};
    if(*error){control->done=1;return;}
    if(!colors||(convention==OriginalUpdateConvention::CooperativeSquaredNorm&&!selected)) {
        fail(error,LinearClipError::InvalidInput);control->done=1;
    }
}
__global__ void node_batch_prepare(DeformableScene scene,DeformableState state,
    const std::size_t* offsets,const BpAabb* boxes,const int* colors,int color,
    const double(*raw_delta)[3],const double(*accepted)[3],bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char* selected,OriginalNodeCcdContext* contexts,
    OriginalNodeCcdControl* control,unsigned* active_nodes) {
    if(control->done)return;
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(node>=scene.nodes)return;
    auto& ctx=contexts[node];ctx=OriginalNodeCcdContext{};
    if(!scene.deformable[node]||colors[scene.node_blocks[node]]!=color)return;
    ctx.active=1;const int block=scene.node_blocks[node];
    const auto first=offsets[block],last=offsets[block+1];ctx.first=first;
    if(last<first||last-first>0xffffffffull) {
        atomicExch(&control->mode,0);
    } else ctx.count=last-first;
    bool moved=false;
    for(int k=0;k<3;++k) {
        ctx.start[k]=state.positions[node][k];ctx.raw[k]=raw_delta[node][k];
        ctx.initial_accepted[k]=accepted[node][k];ctx.trial[k]=ctx.start[k];
        if(!finite(ctx.start[k])||!finite(ctx.raw[k]))ctx.error=int(LinearClipError::InvalidInput);
        moved=moved||ctx.raw[k]!=0.0;
    }
    bool noop=!moved;
    if(!ctx.error&&moved&&mode==LinearClipMode::OriginalSolver
        &&convention==OriginalUpdateConvention::CooperativeSquaredNorm&&selected[node]) {
        double squared=0.0;for(int k=0;k<3;++k)squared+=ctx.raw[k]*ctx.raw[k];
        if(squared<1e-28)noop=true;
    }
    if(ctx.error)ctx.phase=NodeError;
    else if(noop)ctx.phase=use_ccd?NodeHealth:NodeDirect;
    else if(!use_ccd) {
        ctx.phase=NodeDirect;
        for(int k=0;k<3;++k)ctx.trial[k]=__dadd_rn(ctx.start[k],ctx.raw[k]);
        if(!inside(ctx.trial,boxes[node]))ctx.error=int(LinearClipError::OutsideBlueBox);
    } else ctx.phase=NodeRaw;
    const unsigned index=atomicAdd(&control->active_count,1u);active_nodes[index]=static_cast<unsigned>(node);
    atomicAdd(&control->pending,1u);
}
__global__ void node_batch_prepare_trials(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const std::size_t* offsets,const int* indices,const BpAabb* boxes,LinearClipMode mode,
    OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,int attempt) {
    if(control->mode!=1||control->done)return;
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node==0){control->queue_count=control->queue_next=0;control->queue_overflow=0;}
    if(node>=scene.nodes)return;
    auto& ctx=contexts[node];if(!ctx.active||ctx.done||ctx.phase!=NodeTrial)return;
    ctx.first_bad=~0ull;if(attempt==64)ctx.step=0.0;
    for(int k=0;k<3;++k)ctx.trial[k]=__dadd_rn(ctx.start[k],__dmul_rn(ctx.step,ctx.raw[k]));
    if(mode==LinearClipMode::OriginalColoredGuess&&ctx.step>0.0)
        node_predictor_floor(state,bp,offsets,indices,scene.node_blocks[node],int(node),ctx.start,ctx.trial);
    ctx.boxed=inside(ctx.trial,boxes[node]);
}
__global__ void node_batch_fast(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const int* indices,const int* colors,int color,OriginalNodeCcdContext* contexts,
    OriginalNodeCcdControl* control,const unsigned* active_nodes,unsigned long long* queue,std::size_t capacity) {
    if(control->mode!=1||control->done)return;
    for(std::size_t index=blockIdx.x;index<control->active_count;index+=gridDim.x) {
        const unsigned node=active_nodes[index];auto& ctx=contexts[node];
        if(ctx.done||ctx.error||ctx.phase==NodeDirect||ctx.phase==NodeError
            ||(ctx.phase==NodeTrial&&!ctx.boxed))continue;
        for(std::size_t rank=threadIdx.x;rank<ctx.count;rank+=blockDim.x) {
            bool unresolved=false;
            const auto verdict=node_fast_pair(bp.contacts[indices[ctx.first+rank]],scene,state,
                colors,color,node,ctx,unresolved);
            if(unresolved) {
                const auto job=atomicAdd(&control->queue_count,1ull);
                if(job<capacity)queue[job]=(static_cast<unsigned long long>(node)<<32)|static_cast<unsigned>(rank);
                else atomicExch(&control->queue_overflow,1u);
            } else node_publish(static_cast<unsigned>(rank),verdict,ctx);
        }
    }
}
__global__ void node_batch_exact(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const int* indices,OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,
    const unsigned long long* queue,std::size_t capacity,std::uint32_t* words,int* slots,int workers) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=static_cast<unsigned>(workers)||control->mode!=1||control->done||control->queue_overflow)return;
    const auto count=control->queue_count;
    for(auto job=atomicAdd(&control->queue_next,1ull);job<count;job=atomicAdd(&control->queue_next,1ull)) {
        if(job>=capacity){atomicCAS(&control->internal_error,0,int(LinearClipError::ExactResources));continue;}
        const auto encoded=queue[job];const unsigned node=static_cast<unsigned>(encoded>>32),rank=static_cast<unsigned>(encoded);
        if(node>=scene.nodes){atomicCAS(&control->internal_error,0,int(LinearClipError::ExactResources));continue;}
        auto& ctx=contexts[node];
        if(!ctx.active||ctx.done||rank>=ctx.count||(ctx.phase!=NodeRaw&&ctx.phase!=NodeHealth&&ctx.phase!=NodeTrial)) {
            atomicCAS(&control->internal_error,0,int(LinearClipError::ExactResources));continue;
        }
        const auto verdict=node_exact_pair(bp.contacts[indices[ctx.first+rank]],state,node,ctx,words,slots,worker);
        node_publish(rank,verdict,ctx);
    }
}
// Optional launch redistribution only. The qualified kernels above remain
// untouched for rollback. Every verdict still comes from the SAME query and
// packed-rank publisher; no collision/TOI/queue arithmetic is changed.
__global__ void node_batch_fast_partitioned(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const int* indices,const int* colors,int color,
    OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,
    const unsigned* active_nodes,unsigned long long* queue,std::size_t capacity) {
    if(control->mode!=1||control->done)return;
    const auto active=static_cast<std::size_t>(control->active_count);
    if(!active)return;
    const auto plan=node_csr_partition::make_plan(active,gridDim.x,blockIdx.x,
        blockDim.x,threadIdx.x,true);
    for(auto index=plan.first_row;index<active;
        index=node_csr_partition::next_row(index,active,plan)) {
        const unsigned node=active_nodes[index];auto& ctx=contexts[node];
        if(ctx.done||ctx.error||ctx.phase==NodeDirect||ctx.phase==NodeError
            ||(ctx.phase==NodeTrial&&!ctx.boxed))continue;
        for(auto rank=plan.first_rank;rank<ctx.count;
            rank=node_csr_partition::next_rank(rank,ctx.count,plan)) {
            bool unresolved=false;
            const auto verdict=node_fast_pair(bp.contacts[indices[ctx.first+rank]],scene,state,
                colors,color,node,ctx,unresolved);
            if(unresolved) {
                const auto job=atomicAdd(&control->queue_count,1ull);
                if(job<capacity)queue[job]=(static_cast<unsigned long long>(node)<<32)|static_cast<unsigned>(rank);
                else atomicExch(&control->queue_overflow,1u);
            } else node_publish(static_cast<unsigned>(rank),verdict,ctx);
        }
    }
}
__global__ void node_batch_stage_gate(int* error,OriginalNodeCcdControl* control) {
    if(blockIdx.x||threadIdx.x||control->mode!=1||control->done)return;
    if(control->internal_error){atomicCAS(error,0,control->internal_error);control->done=1;return;}
    if(control->queue_overflow){control->mode=0;return;}
    if(!control->pending)control->done=1;
}
__device__ void node_batch_error(int* error,OriginalNodeCcdControl* control,int code) {
    atomicCAS(error,0,code);atomicExch(&control->done,1);
}
__device__ void node_batch_finish(unsigned node,OriginalNodeCcdContext& ctx,
    double(*accepted)[3],OriginalNodeCcdControl* control) {
    for(int k=0;k<3;++k)accepted[node][k]=ctx.trial[k];ctx.done=1;
    if(atomicSub(&control->pending,1u)==1u)atomicExch(&control->done,1);
}
__global__ void node_batch_consume(OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,
    const unsigned* active_nodes,double(*accepted)[3],int* error,int attempt) {
    if(control->mode!=1||atomicAdd(&control->done,0))return;
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=control->active_count)return;
    const unsigned node=active_nodes[index];auto& ctx=contexts[node];if(ctx.done)return;
    if(ctx.error){node_batch_error(error,control,ctx.error);return;}
    if(ctx.first_bad!=~0ull) {
        const unsigned code=static_cast<unsigned>(ctx.first_bad);
        if(code){node_batch_error(error,control,static_cast<int>(code));return;}
        if(ctx.phase!=NodeTrial){node_batch_error(error,control,int(LinearClipError::InvalidInput));return;}
        ctx.boxed=0;
    }
    if(ctx.phase==NodeRaw) {
        ctx.step=ctx.hit?0.9*ctx.toi:1.0;if(ctx.step==0.0)ctx.step=0.0;
        ctx.phase=NodeTrial;return;
    }
    if(ctx.phase==NodeDirect||ctx.phase==NodeHealth){node_batch_finish(node,ctx,accepted,control);return;}
    if(ctx.phase==NodeTrial) {
        if(ctx.boxed){node_batch_finish(node,ctx,accepted,control);return;}
        if(attempt==64){node_batch_error(error,control,int(LinearClipError::Unresolved));return;}
        ctx.step*=0.5;
    }
}
__global__ void node_batch_restore(DeformableScene scene,OriginalNodeCcdContext* contexts,
    const OriginalNodeCcdControl* control,double(*accepted)[3]) {
    if(control->mode!=0)return;
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(node>=scene.nodes)return;
    const auto& ctx=contexts[node];if(ctx.active)
        for(int k=0;k<3;++k)accepted[node][k]=ctx.initial_accepted[k];
}

// Current literal device stage copies; no query culling or arithmetic changes.
#include "node_cooperative_stages.cuh"
struct NodeCooperativeArguments {
    DeformableScene scene;DeformableState state;BroadPhaseView bp;
    const std::size_t* offsets;const int* indices;const BpAabb* boxes;
    const int* colors;int color;double(*accepted)[3];const double(*raw_delta)[3];
    int* error;std::uint32_t* words;int* slots;int workers;bool use_ccd;
    LinearClipMode mode;OriginalUpdateConvention convention;const unsigned char* selected;
    OriginalNodeCcdContext* contexts;OriginalNodeCcdControl* control;
    unsigned* active_nodes;unsigned long long* queue;std::size_t queue_capacity;
    bool partition_rows;
};
static_assert(std::is_trivially_copyable<NodeCooperativeArguments>::value,
    "cooperative arguments must remain plain copied device views");
struct NodeCooperativeStages {
    NodeCooperativeArguments a;
    cooperative_groups::grid_group grid;
    __device__ void begin() {cooperative_begin_stage(a.colors,a.convention,a.selected,a.error,a.control);}
    __device__ void prepare() {
        cooperative_prepare_stage(a.scene,a.state,a.offsets,a.boxes,a.colors,a.color,
            a.raw_delta,a.accepted,a.use_ccd,a.mode,a.convention,a.selected,a.contexts,a.control,a.active_nodes);
    }
    __device__ void fast() {
        if(a.partition_rows)
            cooperative_fast_partitioned_stage(a.scene,a.state,a.bp,a.indices,a.colors,a.color,
                a.contexts,a.control,a.active_nodes,a.queue,a.queue_capacity);
        else
            cooperative_fast_stage(a.scene,a.state,a.bp,a.indices,a.colors,a.color,
                a.contexts,a.control,a.active_nodes,a.queue,a.queue_capacity);
    }
    __device__ void exact() {
        cooperative_exact_stage(a.scene,a.state,a.bp,a.indices,a.contexts,a.control,a.queue,
            a.queue_capacity,a.words,a.slots,a.workers);
    }
    __device__ void gate() {cooperative_gate_stage(a.error,a.control);}
    __device__ void consume(int attempt) {
        cooperative_consume_stage(a.contexts,a.control,a.active_nodes,a.accepted,a.error,attempt);
    }
    __device__ void prepare_trial(int attempt) {
        cooperative_trials_stage(a.scene,a.state,a.bp,a.offsets,a.indices,a.boxes,a.mode,
            a.contexts,a.control,attempt);
    }
    __device__ void join() {grid.sync();}
};
__global__ void node_batch_cooperative(NodeCooperativeArguments arguments) {
    // Every thread participates. Device-stage returns never exit this kernel.
    NodeCooperativeStages stage{arguments,cooperative_groups::this_grid()};
    node_controller::execute(stage,arguments.control);
}
void cooperative_status(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)
        throw std::runtime_error(std::string("GPU node cooperative ")+operation+": "+cudaGetErrorString(status));
}
node_controller::Plan cooperative_plan_impl(std::size_t nodes,int workers,int* observed_sms=nullptr) {
    if(observed_sms)*observed_sms=0;
    // Never clear/reinterpret a prior asynchronous fault as ineligibility.
    cooperative_status(cudaPeekAtLastError(),"pending launch status");
    int device=0,supported=0,sms=0,blocks_per_sm=0;
    cooperative_status(cudaGetDevice(&device),"current device");
    cooperative_status(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device),"capability");
    if(!supported)return node_controller::make_plan(nodes,workers,false,0,0);
    cooperative_status(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"SM count");
    if(observed_sms)*observed_sms=sms;
    cudaFuncAttributes attributes{};
    cooperative_status(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(node_batch_cooperative)),"function attributes");
    if(attributes.maxThreadsPerBlock<static_cast<int>(node_controller::width))
        return node_controller::make_plan(nodes,workers,true,sms,0);
    cooperative_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
        reinterpret_cast<const void*>(node_batch_cooperative),node_controller::width,0),"actual combined occupancy");
    return node_controller::make_plan(nodes,workers,true,sms,blocks_per_sm);
}
bool try_node_cooperative(NodeCooperativeArguments arguments,cudaStream_t stream) {
    static_assert(node_cooperative_grid::block_width==node_controller::width,
        "grid coverage must use the actual cooperative block width");
    int sms=0;
    const auto plan=cooperative_plan_impl(arguments.scene.nodes,arguments.workers,&sms);
    if(!plan.eligible())return false;
    const bool one_wave=contact_execution_option("IPC_GPU_NODE_CCD_ONE_WAVE",true);
    const auto chosen=node_cooperative_grid::choose(plan.blocks,arguments.scene.nodes,
        arguments.workers,one_wave?static_cast<unsigned>(sms):plan.blocks);
    if(!chosen.valid)return false;
    void* parameters[]={&arguments};
    // Every launch error propagates. No retry over possibly executed state.
    cooperative_status(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(node_batch_cooperative),
        dim3(chosen.grid),dim3(node_controller::width),parameters,0,stream),"launch");
    return true;
}

} // namespace

namespace linear_detail {
node_controller::Plan query_node_cooperative_plan(std::size_t nodes,int workers) {
    return cooperative_plan_impl(nodes,workers);
}

void launch_resolve_nodes_original(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t*offsets,const int*indices,const BpAabb*boxes,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int*error,
    std::uint32_t*words,int*slots,int workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char*selected,cudaStream_t stream) {
    resolve_nodes_original<<<unsigned(workers),1,0,stream>>>(scene,state,bp,offsets,indices,boxes,colors,color,accepted,raw_delta,error,words,slots,workers,use_ccd,mode,convention,selected,nullptr);
}
void launch_resolve_nodes_original_batch(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t* offsets,const int* indices,const BpAabb* boxes,
    const int* colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int* error,
    std::uint32_t* words,int* slots,int workers,int fallback_workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char* selected,OriginalNodeCcdContext* contexts,
    OriginalNodeCcdControl* control,unsigned* active_nodes,unsigned long long* queue,
    std::size_t queue_capacity,bool partition_rows,bool cooperative_controller,cudaStream_t stream) {
    if(!scene.nodes)return;
    const unsigned node_grid=static_cast<unsigned>((scene.nodes+127)/128);
    const unsigned pair_grid=static_cast<unsigned>(scene.nodes<1024?scene.nodes:1024);
    if(cooperative_controller) {
        NodeCooperativeArguments arguments{scene,state,bp,offsets,indices,boxes,colors,color,
            accepted,raw_delta,error,words,slots,workers,use_ccd,mode,convention,selected,
            contexts,control,active_nodes,queue,queue_capacity,partition_rows};
        if(try_node_cooperative(arguments,stream)) {
            node_batch_restore<<<node_grid,128,0,stream>>>(scene,contexts,control,accepted);
            resolve_nodes_original<<<fallback_workers,1,0,stream>>>(scene,state,bp,offsets,indices,
                boxes,colors,color,accepted,raw_delta,error,words,slots,fallback_workers,
                use_ccd,mode,convention,selected,control);
            return;
        }
    }
    node_batch_begin<<<1,1,0,stream>>>(colors,convention,selected,error,control);
    node_batch_prepare<<<node_grid,128,0,stream>>>(scene,state,offsets,boxes,colors,color,raw_delta,
        accepted,use_ccd,mode,convention,selected,contexts,control,active_nodes);
    if(partition_rows)
        node_batch_fast_partitioned<<<pair_grid,128,0,stream>>>(scene,state,bp,indices,colors,color,contexts,control,active_nodes,queue,queue_capacity);
    else
        node_batch_fast<<<pair_grid,128,0,stream>>>(scene,state,bp,indices,colors,color,contexts,control,active_nodes,queue,queue_capacity);
    node_batch_exact<<<workers,1,0,stream>>>(scene,state,bp,indices,contexts,control,queue,queue_capacity,words,slots,workers);
    node_batch_stage_gate<<<1,1,0,stream>>>(error,control);
    node_batch_consume<<<node_grid,128,0,stream>>>(contexts,control,active_nodes,accepted,error,-1);
    for(int attempt=0;attempt<=64;++attempt) {
        node_batch_prepare_trials<<<node_grid,128,0,stream>>>(scene,state,bp,offsets,indices,boxes,mode,contexts,control,attempt);
        if(partition_rows)
            node_batch_fast_partitioned<<<pair_grid,128,0,stream>>>(scene,state,bp,indices,colors,color,contexts,control,active_nodes,queue,queue_capacity);
        else
            node_batch_fast<<<pair_grid,128,0,stream>>>(scene,state,bp,indices,colors,color,contexts,control,active_nodes,queue,queue_capacity);
        node_batch_exact<<<workers,1,0,stream>>>(scene,state,bp,indices,contexts,control,queue,queue_capacity,words,slots,workers);
        node_batch_stage_gate<<<1,1,0,stream>>>(error,control);
        node_batch_consume<<<node_grid,128,0,stream>>>(contexts,control,active_nodes,accepted,error,attempt);
    }
    node_batch_restore<<<node_grid,128,0,stream>>>(scene,contexts,control,accepted);
    resolve_nodes_original<<<fallback_workers,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,colors,color,
        accepted,raw_delta,error,words,slots,fallback_workers,use_ccd,mode,convention,selected,control);
}
} // namespace linear_detail
} // namespace ipc_gpu::solver

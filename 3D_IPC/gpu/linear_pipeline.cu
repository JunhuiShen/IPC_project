#define IPC_GPU_TU_FINITE_CCD 1
#include "linear_pipeline.h"
#include "profile.h"
#include "ccd.cuh"
#include "exact.cuh"
#include "direct_bound.cuh"
#include "original_ccd.cuh"
#include "original_guess.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace ipc_gpu::solver {
namespace {
namespace cert=ipc_gpu::ccd;
namespace exact = ipc_gpu::exact;
namespace occd = ipc_gpu::original_ccd;
constexpr int arena_width=2048,arena_slots=512;
// Bounded reusable ORIGINAL-mode worker pool. Each live worker owns a DISJOINT
// arena slice of the same shape as the robust resolver's arena, so the arena
// count is bounded by the worker cap instead of by the scene node count.
constexpr int original_worker_cap=16;
constexpr int original_arena_width=2048,original_arena_slots=512;

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

// The optional direct-bound mode gets a separate kernel so its exact-arithmetic
// liveness cannot enlarge the legacy resolver's registers or stack. A bound
// proposes one represented endpoint; it never accepts a trajectory itself.
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

// The solver-owned cooperative mask is validated ONCE per call, not once per
// worker and not per node inside the resolvers. A value outside {0,1} fails
// closed before any publish.
__global__ void validate_original_mask(std::size_t nodes,const unsigned char*selected,
    int*error) {
    const std::size_t node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=nodes)return;
    if(selected[node]>1)fail(error,LinearClipError::InvalidInput);
}

// Match the robust clipper's caller contract: a contact can contain at most
// one active deformable vertex in this color. Checking independent frozen
// one-vertex paths cannot authorize a simultaneous conflicting update.
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

// The ADDITIONAL represented-path safety check for original-mode commits.
// `ccd::detail::separated` at exact ZERO gap is a proved complete-trajectory
// separation: TRUE is a sufficient collision-free certificate, FALSE is
// inconclusive only and must reach the unchanged exact collision-only
// predicate. It introduces no positive separation floor, never passes floor
// zero to the positive-floor validate_step/validate_linear contract, and never
// replaces the CPU ORIGINAL TOI (whose fixed 1e-10 event band may
// intentionally report t=0 for a healthy positive gap).
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

// ORIGINAL (exact_computation_fallback == false) ordinary commit. The CPU
// original solver uses one moving vertex per pair, so every incident pair is
// queried with the active vertex's stored RAW displacement. A hit commits
// 0.9 * earliest TOI (TOI 1 included); a miss commits weight 1. The stored
// endpoint and its represented path are then certified by the collision-only
// exact check, which allows an unchanged healthy positive-gap endpoint and
// fails a truly zero initial gap. Only cooperative-selected vertices apply
// the CPU 1e-28 squared-cutoff; baseline/batch vertices keep small moves.
// `colors` is the caller's explicit current-color block map. The protected
// direct caller supplies it while BroadPhaseView may only contain contacts, so
// the resolver must not read bp.block_colors here.
//
// BOUNDED PARALLEL EXECUTION. `workers` device workers walk fixed strides of
// the scene nodes; each owns a DISJOINT reusable exact arena slice and writes
// only its own `accepted` entries, so independent same-color vertices commit
// concurrently while the current positions stay immutable. The whole kernel
// joins before any physical commit or the next color. There is no shared
// arena between concurrent workers and no per-scene-node arena.
__global__ void resolve_nodes_original(DeformableScene scene,DeformableState state,
    BroadPhaseView bp,const std::size_t*offsets,const int*indices,const BpAabb*boxes,
    const int*colors,int color,double(*accepted)[3],const double(*raw_delta)[3],int*error,
    std::uint32_t*words,int*slots,int workers,bool use_ccd,LinearClipMode mode,
    OriginalUpdateConvention convention,const unsigned char*selected) {
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

// CPU ColoredContactSweep selected-vertex dispatch when the exact
// `validate_cooperative_step` rule is FALSE (original mode, or robust exact
// mode with CCD disabled):
//     nonzero_step = !(steps.squaredNorm() < 1e-28)
// Only SELECTED cooperative vertices discard such a stored move; scalar /
// batch-fallback vertices keep representable small updates. The move itself is
// already the CPU componentwise-clipped raw displacement written by
// propose_nodes, and the endpoint was finite/box checked there.
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

struct GpuLinearClipper::Impl {
    std::uint32_t* words=nullptr;int* slots=nullptr;
    // Bounded ORIGINAL-mode worker pool: `original_workers` live workers, each
    // with its own disjoint arena slice. Grown lazily, never per scene node.
    std::uint32_t* original_words=nullptr;int* original_slots=nullptr;
    std::size_t original_worker_capacity=0;
    unsigned int* pending=nullptr;std::size_t pending_capacity=0;
    unsigned char* initial_safe=nullptr;std::size_t contact_capacity=0;
    double (*clip_raw)[3]=nullptr;std::size_t clip_raw_capacity=0;
    Impl() {
        check(cudaMalloc(&words,std::size_t(arena_width)*arena_slots*sizeof(*words)));
        try { check(cudaMalloc(&slots,arena_slots*sizeof(*slots))); }
        catch(...) { cudaFree(words);throw; }
    }
    ~Impl() { if(clip_raw)cudaFree(clip_raw);if(initial_safe)cudaFree(initial_safe);if(pending)cudaFree(pending);
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
    int original_workers(std::size_t nodes) {
        const std::size_t wanted=std::min<std::size_t>(
            nodes==0?0:nodes,static_cast<std::size_t>(original_worker_cap));
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
        return static_cast<int>(wanted);
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
    GpuProfileScope prepare_profile("prepare",stream);
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
        const int workers=impl_->original_workers(scene.nodes);
        if(workers>0)
            resolve_nodes_original<<<unsigned(workers),1,0,stream>>>(scene,state,bp,
                offsets,indices,boxes,colors,color,accepted,impl_->clip_raw,error,
                impl_->original_words,impl_->original_slots,workers,
                use_ccd,mode,convention,original_cooperative_selected);
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
        GpuProfileScope certify_profile("parallel_certify",stream);
        const auto count=color_pairs.offsets?color_pairs.max_color_pairs:bp.contact_count;
        const unsigned blocks=unsigned(std::max<std::size_t>(1,std::min<std::size_t>((count+127)/128,65535)));
        certify_node_contacts<<<blocks,128,0,stream>>>(scene,state,bp,colors,color,accepted,
            impl_->pending,impl_->initial_safe,error,color_pairs);check(cudaGetLastError());
    }
    GpuProfileScope resolve_profile("serial_resolve",stream);
    if(exact_step_bound)
        resolve_nodes_direct_bound<<<1,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,impl_->pending,impl_->initial_safe,accepted,error,
            impl_->words,impl_->slots);
    else
    resolve_nodes<<<1,1,0,stream>>>(scene,state,bp,offsets,indices,boxes,impl_->pending,impl_->initial_safe,accepted,error,
        impl_->words,impl_->slots);
    check(cudaGetLastError());
}

void GpuLinearClipper::clip_rigid_translation_color(const RigidScene&scene,const RigidState&state,
    const RigidWorkspace&work,const DeformableState&vertices,const BroadPhaseView&bp,
    const std::size_t*offsets,const int*indices,const BpAabb*boxes,const int*colors,int color,
    double(*accepted)[3],int*error,bool use_ccd,cudaStream_t stream) {
    GpuProfileScope resolve_profile("serial_resolve",stream);
    if(!scene.bodies)return;
    resolve_bodies<<<1,1,0,stream>>>(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,
        accepted,error,impl_->words,impl_->slots,use_ccd);check(cudaGetLastError());
}

} // namespace ipc_gpu::solver

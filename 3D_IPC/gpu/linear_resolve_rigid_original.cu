#define IPC_GPU_TU_FINITE_CCD 1
#include "exact.cuh"
#include "original_ccd.cuh"
#include "original_rigid.cuh"
#include "linear_pipeline_internal.cuh"

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

__device__ bool original_body_side(const BpContact& contact,RigidScene scene,
    const int* colors,int color,int body,bool& first,bool& involved,int* error) {
    int owner[4];
    for(int i=0;i<4;++i)owner[i]=scene.node_body[contact.vertices[i]];
    const int second=contact.kind==0?1:2;
    if((contact.kind==0&&(owner[1]!=owner[2]||owner[1]!=owner[3]))
        ||(contact.kind!=0&&(owner[0]!=owner[1]||owner[2]!=owner[3]))) {
        fail(error,LinearClipError::UnsupportedMotion);return false;
    }
    first=owner[0]==body;
    const bool other=owner[second]==body;
    involved=first!=other;
    if(!involved)return true;
    const int fixed=first?owner[second]:owner[0];
    if(fixed>=0&&(std::size_t(fixed)>=scene.bodies
        ||(colors[scene.body_blocks[fixed]]==color
            &&(scene.translate[fixed]||scene.rotate[fixed])))) {
        fail(error,LinearClipError::UnsupportedMotion);return false;
    }
    return true;
}

__device__ bool original_body_trial(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const BpAabb* boxes,int body,const double* raw,double step,
    double(*accepted)[3],double* delta,int* error) {
    for(int k=0;k<3;++k) {
        // CPU COM and every proxy add the SAME stored step * raw. Rebuilding
        // delta from a rounded accepted COM, or using FMA, changes this path.
        delta[k]=step==0.0?0.0:__dmul_rn(step,raw[k]);
        work.accepted_com[body][k]=step==0.0?state.com[body][k]
            :__dadd_rn(state.com[body][k],delta[k]);
        if(!finite(delta[k])||!finite(work.accepted_com[body][k])) {
            fail(error,LinearClipError::InvalidInput);return false;
        }
    }
    bool boxed=true;
    for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i) {
        const int node=scene.nodes[i];
        for(int k=0;k<3;++k)accepted[node][k]=step==0.0?vertices.positions[node][k]
            :__dadd_rn(vertices.positions[node][k],delta[k]);
        boxed=boxed&&inside(vertices.positions[node],boxes[node])&&inside(accepted[node],boxes[node]);
    }
    return boxed;
}

__device__ bool original_result_failed(const occd::Result& result,int* error) {
    if(result.status==occd::Status::Ok)return false;
    fail(error,result.status==occd::Status::InvalidInput?LinearClipError::InvalidInput
        :LinearClipError::ExactResources);
    return true;
}

__global__ void resolve_bodies_original(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd,const RigidCcdBatchContext* gate) {
    if(gate&&gate->mode!=0)return;
    if(blockIdx.x||threadIdx.x||*error)return;
    exact::DevicePool pool;exact::initialize_pool(pool,words,slots,arena_width,arena_slots);
    for(std::size_t body=0;body<scene.bodies&&!*error;++body) {
        const int block=scene.body_blocks[body];if(colors[block]!=color)continue;
        for(int k=0;k<3;++k)work.accepted_com[body][k]=state.com[body][k];
        for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i)
            for(int k=0;k<3;++k)accepted[scene.nodes[i]][k]=vertices.positions[scene.nodes[i]][k];
        if(!scene.translate[body])continue;
        double raw[3];
        for(int k=0;k<3;++k) {
            raw[k]=__dsub_rn(work.proposed_com[body][k],state.com[body][k]);
            if(!finite(raw[k])||!finite(state.com[body][k])) {
                fail(error,LinearClipError::InvalidInput);break;
            }
        }
        if(*error)break;
        double toi=1.0;bool hit=false;
        const bool cpu_zero_step=ipc_gpu::original_rigid::skip_translation_toi(raw);
        if(use_ccd&&!cpu_zero_step)
            for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                const auto& contact=bp.contacts[indices[pair]];
                bool first=false,involved=false;
                if(!original_body_side(contact,scene,colors,color,int(body),first,involved,error))break;
                if(!involved)continue;
                double positions[4][3];
                for(int i=0;i<4;++i)for(int k=0;k<3;++k)
                    positions[i][k]=vertices.positions[contact.vertices[i]][k];
                if(!ipc_gpu::original_rigid::translation_boxes_overlap(positions,contact.kind==0,first,raw))continue;
                const auto result=original_fast_exact(
                    ipc_gpu::original_rigid::translation_query(positions,contact.kind==0,first,raw),pool);
                if(original_result_failed(result,error))break;
                if(result.collision) { hit=true;toi=fmin(toi,result.toi); }
            }
        if(*error)break;
        double step=hit?0.9*toi:1.0;
        bool complete=false;
        for(int attempt=0;attempt<=64&&!*error;++attempt) {
            if(attempt==64)step=0.0;
            double delta[3];
            bool safe=original_body_trial(scene,state,work,vertices,boxes,int(body),raw,step,accepted,delta,error);
            if(*error)break;
            if(safe&&use_ccd)
                for(std::size_t pair=offsets[block];pair<offsets[block+1];++pair) {
                    const auto& contact=bp.contacts[indices[pair]];
                    bool first=false,involved=false;
                    if(!original_body_side(contact,scene,colors,color,int(body),first,involved,error)) { safe=false;break; }
                    if(!involved)continue;
                    const auto q=body_query(contact,scene,vertices,int(body),accepted);
                    // Independently recheck the represented common delta. Its
                    // multiplication rounding must not inherit a certificate
                    // for the unscaled raw proposal. A band hit is conservative.
                    if(!cpu_zero_step&&(delta[0]!=0.0||delta[1]!=0.0||delta[2]!=0.0)
                        &&ipc_gpu::original_rigid::translation_boxes_overlap(q.start,q.vertex_face,first,delta)) {
                        const auto result=original_fast_exact(
                            ipc_gpu::original_rigid::translation_query(q.start,q.vertex_face,first,delta),pool);
                        if(original_result_failed(result,error)) { safe=false;break; }
                        if(result.collision) { safe=false;break; }
                    }
                    using Status=ipc_gpu::original_rigid::RepresentationStatus;
                    auto status=ipc_gpu::original_rigid::represented_path<256>(q.start,q.end,q.vertex_face);
                    if(status==Status::CapacityExceeded&&!pool.failed&&pool.available==pool.slot_count)
                        status=ipc_gpu::original_rigid::represented_path<2048>(q.start,q.end,q.vertex_face);
                    if(status==Status::InitialContact)fail(error,LinearClipError::InitialContact);
                    else if(status==Status::CapacityExceeded)fail(error,LinearClipError::ExactResources);
                    else if(status==Status::Invalid)fail(error,LinearClipError::InvalidInput);
                    if(status!=Status::Safe) { safe=false;break; }
                }
            if(*error)break;
            if(safe) { complete=true;break; }
            step*=0.5;
        }
        if(!complete&&!*error)fail(error,LinearClipError::Unresolved);
        if(pool.failed||pool.available!=pool.slot_count)fail(error,LinearClipError::ExactResources);
    }
    exact::release_pool();
}
// Prototype only: preserve the scalar kernel above as the direct oracle.
// Speculative lanes publish no global error. Lane zero consumes pair verdicts
// in original CSR order, including the first unsafe/error stopping point.
using PairVerdict=RigidCcdBatchVerdict;

__device__ int original_body_side_local(const BpContact& contact,RigidScene scene,
    const int* colors,int color,int body,bool& first,bool& involved) {
    int owner[4];
    for(int i=0;i<4;++i)owner[i]=scene.node_body[contact.vertices[i]];
    const int second=contact.kind==0?1:2;
    if((contact.kind==0&&(owner[1]!=owner[2]||owner[1]!=owner[3]))
        ||(contact.kind!=0&&(owner[0]!=owner[1]||owner[2]!=owner[3])))
        return int(LinearClipError::UnsupportedMotion);
    first=owner[0]==body;
    involved=first!=(owner[second]==body);
    if(!involved)return 0;
    const int fixed=first?owner[second]:owner[0];
    if(fixed>=0&&(std::size_t(fixed)>=scene.bodies
        ||(colors[scene.body_blocks[fixed]]==color
            &&(scene.translate[fixed]||scene.rotate[fixed]))))
        return int(LinearClipError::UnsupportedMotion);
    return 0;
}

// Every speculative exact query owns a fresh pool in its lane's disjoint slice.
// All Integer temporaries are destroyed before finish/release. A discarded
// later verdict therefore cannot poison a subsequent logically executed query.
struct PairPool {
    exact::DevicePool pool;
    __device__ PairPool(std::uint32_t* words,int* slots,unsigned lane) {
        exact::initialize_pool(pool,words+std::size_t(lane)*original_arena_width*original_arena_slots,
            slots+std::size_t(lane)*original_arena_slots,original_arena_width,original_arena_slots);
    }
    __device__ PairVerdict finish(PairVerdict verdict) {
        if(!verdict.error&&(pool.failed||pool.available!=pool.slot_count))
            verdict.error=int(LinearClipError::ExactResources);
        return verdict;
    }
    __device__ ~PairPool() {exact::release_pool();}
};
__device__ int original_result_error(const occd::Result& result) {
    return result.status==occd::Status::Ok?0
        :int(result.status==occd::Status::InvalidInput?LinearClipError::InvalidInput
            :LinearClipError::ExactResources);
}

__device__ PairVerdict original_query_local(const occd::Query& query,
    std::uint32_t* words,int* slots,unsigned lane) {
    PairVerdict verdict{0,false,true,1.0};
    auto result=occd::evaluate_fast(query);
    if(result.status!=occd::Status::NeedsExact) {
        verdict.error=original_result_error(result);
        verdict.collision=result.collision;verdict.toi=result.toi;
        return verdict;
    }
    // The allocation-free fast route never constructs an Integer. Install a
    // fresh arena only for the exact branch it requested, with unchanged retry.
    PairPool arena(words,slots,lane);
    result=occd::evaluate_exact<256>(query);
    if(result.status==occd::Status::CapacityExceeded
        &&!arena.pool.failed&&arena.pool.available==arena.pool.slot_count)
        result=occd::evaluate_exact<2048>(query);
    verdict.error=original_result_error(result);
    verdict.collision=result.collision;verdict.toi=result.toi;
    return arena.finish(verdict);
}

__device__ PairVerdict original_toi_pair(const BpContact& contact,RigidScene scene,
    DeformableState vertices,const int* colors,int color,int body,const double* raw,
    std::uint32_t* words,int* slots,unsigned lane) {
    PairVerdict verdict{0,false,true,1.0};bool first=false,involved=false;
    verdict.error=original_body_side_local(contact,scene,colors,color,body,first,involved);
    if(verdict.error||!involved)return verdict;
    double positions[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        positions[i][k]=vertices.positions[contact.vertices[i]][k];
    if(!ipc_gpu::original_rigid::translation_boxes_overlap(positions,contact.kind==0,first,raw))return verdict;
    return original_query_local(
        ipc_gpu::original_rigid::translation_query(positions,contact.kind==0,first,raw),words,slots,lane);
}

__device__ PairVerdict original_trial_pair(const BpContact& contact,RigidScene scene,
    DeformableState vertices,const int* colors,int color,int body,const double* delta,
    bool cpu_zero_step,const double(*accepted)[3],std::uint32_t* words,int* slots,unsigned lane) {
    PairVerdict verdict{0,false,true,1.0};bool first=false,involved=false;
    verdict.error=original_body_side_local(contact,scene,colors,color,body,first,involved);
    if(verdict.error||!involved)return verdict;
    const auto q=body_query(contact,scene,vertices,body,accepted);
    if(!cpu_zero_step&&(delta[0]!=0.0||delta[1]!=0.0||delta[2]!=0.0)
        &&ipc_gpu::original_rigid::translation_boxes_overlap(q.start,q.vertex_face,first,delta)) {
        const auto toi=original_query_local(
            ipc_gpu::original_rigid::translation_query(q.start,q.vertex_face,first,delta),words,slots,lane);
        if(toi.error||toi.collision) {
            verdict.error=toi.error;verdict.safe=false;return verdict;
        }
    }
    // Literal allocation-free prefix of original_rigid::represented_path:
    // finite inputs precede the SAME zero-gap whole-trajectory certificate.
    // An uncertain no-op still reaches the unchanged exact InitialContact test.
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        if(!cert::detail::finite(q.start[i][k])||!cert::detail::finite(q.end[i][k])) {
            verdict.error=int(LinearClipError::InvalidInput);verdict.safe=false;return verdict;
        }
    if(cert::detail::separated(q,0.0,1.0,0.0))return verdict;
    PairPool arena(words,slots,lane);
    using Status=ipc_gpu::original_rigid::RepresentationStatus;
    auto status=ipc_gpu::original_rigid::represented_path<256>(q.start,q.end,q.vertex_face);
    if(status==Status::CapacityExceeded&&!arena.pool.failed&&arena.pool.available==arena.pool.slot_count)
        status=ipc_gpu::original_rigid::represented_path<2048>(q.start,q.end,q.vertex_face);
    if(status==Status::InitialContact)verdict.error=int(LinearClipError::InitialContact);
    else if(status==Status::CapacityExceeded)verdict.error=int(LinearClipError::ExactResources);
    else if(status==Status::Invalid)verdict.error=int(LinearClipError::InvalidInput);
    verdict.safe=status==Status::Safe;
    return arena.finish(verdict);
}

__global__ void resolve_bodies_original_parallel(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t* offsets,const int* indices,
    const BpAabb* boxes,const int* colors,int color,double(*accepted)[3],int* error,
    std::uint32_t* words,int* slots,bool use_ccd,const RigidCcdBatchContext* gate) {
    if(gate&&gate->mode!=0)return;
    const unsigned lane=threadIdx.x;
    __shared__ int running,active_body,zero_step,hit,safe,complete,scan_done;
    __shared__ double raw[3],delta[3],toi,step;
    __shared__ int pair_errors[128],pair_hits[128],pair_safe[128];
    __shared__ double pair_times[128];
    if(lane==0)running=!*error;
    __syncthreads();if(!running)return;
    // One block intentionally retains the ORIGINAL ascending body order.
    for(std::size_t body=0;body<scene.bodies;++body) {
        // Complete every previous body's shared-control read (including its
        // inactive-body continue) before lane zero publishes the next body.
        __syncthreads();
        const int block=scene.body_blocks[body];
        if(lane==0) {
            active_body=colors[block]==color;
            if(active_body) {
                for(int k=0;k<3;++k)work.accepted_com[body][k]=state.com[body][k];
                for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i)
                    for(int k=0;k<3;++k)accepted[scene.nodes[i]][k]=vertices.positions[scene.nodes[i]][k];
                if(!scene.translate[body])active_body=0;
            }
            if(active_body) {
                for(int k=0;k<3;++k) {
                    raw[k]=__dsub_rn(work.proposed_com[body][k],state.com[body][k]);
                    if(!finite(raw[k])||!finite(state.com[body][k])) {
                        fail(error,LinearClipError::InvalidInput);running=0;break;
                    }
                }
                zero_step=running?ipc_gpu::original_rigid::skip_translation_toi(raw):0;
                toi=1.0;hit=0;
            }
        }
        __syncthreads();if(!running)return;if(!active_body)continue;
        if(use_ccd&&!zero_step)for(std::size_t first=offsets[block];first<offsets[block+1];first+=blockDim.x) {
            const std::size_t incident=first+lane;
            PairVerdict verdict{0,false,true,1.0};
            if(incident<offsets[block+1])verdict=original_toi_pair(bp.contacts[indices[incident]],
                scene,vertices,colors,color,int(body),raw,words,slots,lane);
            pair_errors[lane]=verdict.error;pair_hits[lane]=verdict.collision;pair_times[lane]=verdict.toi;
            __syncthreads();
            if(lane==0)for(unsigned item=0;item<blockDim.x&&first+item<offsets[block+1];++item) {
                if(pair_errors[item]) {atomicCAS(error,0,pair_errors[item]);running=0;break;}
                if(pair_hits[item]) {hit=1;toi=fmin(toi,pair_times[item]);}
            }
            __syncthreads();if(!running)return;
        }
        if(lane==0){step=hit?0.9*toi:1.0;complete=0;}
        __syncthreads();
        for(int attempt=0;attempt<=64;++attempt) {
            if(lane==0) {
                if(attempt==64)step=0.0;
                safe=original_body_trial(scene,state,work,vertices,boxes,int(body),raw,step,accepted,delta,error);
                if(*error)running=0;
                scan_done=0;
            }
            __syncthreads();if(!running)return;
            if(safe&&use_ccd)for(std::size_t first=offsets[block];first<offsets[block+1];first+=blockDim.x) {
                const std::size_t incident=first+lane;
                PairVerdict verdict{0,false,true,1.0};
                if(incident<offsets[block+1])verdict=original_trial_pair(bp.contacts[indices[incident]],
                    scene,vertices,colors,color,int(body),delta,zero_step,accepted,words,slots,lane);
                pair_errors[lane]=verdict.error;pair_safe[lane]=verdict.safe;
                __syncthreads();
                if(lane==0)for(unsigned item=0;item<blockDim.x&&first+item<offsets[block+1];++item) {
                    if(pair_errors[item]) {atomicCAS(error,0,pair_errors[item]);running=0;safe=0;scan_done=1;break;}
                    if(!pair_safe[item]) {safe=0;scan_done=1;break;}
                }
                __syncthreads();if(!running)return;if(scan_done)break;
            }
            if(lane==0) {if(safe)complete=1;else step*=0.5;}
            __syncthreads();if(complete)break;
        }
        if(lane==0&&!complete) {fail(error,LinearClipError::Unresolved);running=0;}
        __syncthreads();if(!running)return;
    }
}
// Multi-SM prototype. The context mode is published before any numerical
// mutation: 0 selects the gated scalar/CTA fallback; 1 owns all staged work.
// Only exact/uncertain queries enter the bounded cold queue. Final ordinary
// and exact verdicts publish deterministic row-relative aggregates directly.
__device__ void batch_initialize_body(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,std::size_t body,double(*accepted)[3]) {
    for(int k=0;k<3;++k)work.accepted_com[body][k]=state.com[body][k];
    for(std::size_t i=scene.node_offsets[body];i<scene.node_offsets[body+1];++i)
        for(int k=0;k<3;++k)accepted[scene.nodes[i]][k]=vertices.positions[scene.nodes[i]][k];
}
__global__ void batch_prepare(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const std::size_t* offsets,const int* colors,int color,
    double(*accepted)[3],int* error,bool use_ccd,std::size_t row_limit,RigidCcdBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x)return;
    *ctx=RigidCcdBatchContext{};
    if(*error) {ctx->mode=1;ctx->done=1;return;}
    if(!use_ccd) {return;}
    int selected=-1;
    for(std::size_t body=0;body<scene.bodies;++body)
        if(colors[scene.body_blocks[body]]==color&&scene.translate[body]) {
            if(selected>=0) {return;}
            selected=static_cast<int>(body);
        }
    if(selected<0) {return;}
    const int block=scene.body_blocks[selected];
    const auto first=offsets[block],last=offsets[block+1];
    if(last<first||last-first>row_limit) {return;}
    ctx->mode=1;ctx->body=selected;ctx->first=first;ctx->count=last-first;
    ctx->phase=0;ctx->do_pairs=1;
    // Match scalar partial publication: earlier matching fixed bodies and the
    // selected body initialize now; later matching fixed bodies wait for success.
    for(int body=0;body<=selected;++body)
        if(colors[scene.body_blocks[body]]==color)
            batch_initialize_body(scene,state,work,vertices,body,accepted);
    for(int k=0;k<3;++k) {
        ctx->raw[k]=__dsub_rn(work.proposed_com[selected][k],state.com[selected][k]);
        if(!finite(ctx->raw[k])||!finite(state.com[selected][k])) {
            fail(error,LinearClipError::InvalidInput);ctx->done=1;return;
        }
    }
    ctx->zero_step=ipc_gpu::original_rigid::skip_translation_toi(ctx->raw);
    if(ctx->zero_step)ctx->do_pairs=0;
}

// Solver-owned unique CSR/disjoint-storage path. All legacy stage bodies
// remain intact for direct/untrusted calls and explicit rollback.
__global__ void batch_prepare_proxy_header(RigidScene scene,
    const std::size_t* offsets,const int* colors,int color,int* error,
    bool use_ccd,std::size_t row_limit,RigidCcdBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x)return;
    *ctx=RigidCcdBatchContext{};
    if(*error) {ctx->mode=1;ctx->done=1;return;}
    if(!use_ccd) {return;}
    int selected=-1;
    for(std::size_t body=0;body<scene.bodies;++body)
        if(colors[scene.body_blocks[body]]==color&&scene.translate[body]) {
            if(selected>=0) {return;}
            selected=static_cast<int>(body);
        }
    if(selected<0) {return;}
    const int block=scene.body_blocks[selected];
    const auto first=offsets[block],last=offsets[block+1];
    if(last<first||last-first>row_limit) {return;}
    ctx->mode=1;ctx->body=selected;ctx->first=first;ctx->count=last-first;
    ctx->phase=0;ctx->do_pairs=1;
}
__global__ void batch_initialize_proxy_prefix(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const int* colors,int color,double(*accepted)[3],
    const RigidCcdBatchContext* ctx) {
    if(ctx->mode!=1||ctx->done)return;
    const auto worker=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=std::size_t(gridDim.x)*blockDim.x;
    // Bodies/nodes are disjoint by the solver-only permit. Every selected
    // prefix body finishes before the following scalar raw validation.
    for(int body=0;body<=ctx->body;++body)if(colors[scene.body_blocks[body]]==color) {
        if(worker==0)for(int k=0;k<3;++k)work.accepted_com[body][k]=state.com[body][k];
        const auto begin=scene.node_offsets[body],end=scene.node_offsets[body+1];
        for(std::size_t index=worker;index<end-begin;index+=stride) {
            const int node=scene.nodes[begin+index];
            for(int k=0;k<3;++k)accepted[node][k]=vertices.positions[node][k];
        }
    }
}
__global__ void batch_prepare_proxy_finish(RigidState state,RigidWorkspace work,
    int* error,RigidCcdBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    const int selected=ctx->body;
    for(int k=0;k<3;++k) {
        ctx->raw[k]=__dsub_rn(work.proposed_com[selected][k],state.com[selected][k]);
        if(!finite(ctx->raw[k])||!finite(state.com[selected][k])) {
            fail(error,LinearClipError::InvalidInput);ctx->done=1;return;
        }
    }
    ctx->zero_step=ipc_gpu::original_rigid::skip_translation_toi(ctx->raw);
    if(ctx->zero_step)ctx->do_pairs=0;
}
__global__ void batch_prepare_proxy_trial_header(RigidState state,RigidWorkspace work,
    int* error,RigidCcdBatchContext* ctx,int attempt) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    ctx->attempt=attempt;ctx->phase=1;ctx->queue_count=ctx->queue_next=0;
    ctx->first_bad=~0ull;ctx->queue_overflow=0;
    if(attempt==64)ctx->step=0.0;
    const int body=ctx->body;
    const double* raw=ctx->raw;const double step=ctx->step;
    double* delta=ctx->delta;
    for(int k=0;k<3;++k) {
        // Literal original_body_trial COM/delta arithmetic and axis-error prefix.
        delta[k]=step==0.0?0.0:__dmul_rn(step,raw[k]);
        work.accepted_com[body][k]=step==0.0?state.com[body][k]
            :__dadd_rn(state.com[body][k],delta[k]);
        if(!finite(delta[k])||!finite(work.accepted_com[body][k])) {
            fail(error,LinearClipError::InvalidInput);ctx->safe=0;ctx->done=1;ctx->do_pairs=0;return;
        }
    }
    ctx->safe=1;
}
__global__ void batch_write_proxy_trial(RigidScene scene,DeformableState vertices,
    const BpAabb* boxes,double(*accepted)[3],RigidCcdBatchContext* ctx) {
    if(ctx->mode!=1||ctx->done)return;
    const auto worker=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=std::size_t(gridDim.x)*blockDim.x;
    const int body=ctx->body;
    const auto begin=scene.node_offsets[body],end=scene.node_offsets[body+1];
    const double step=ctx->step;const double* delta=ctx->delta;
    for(std::size_t index=worker;index<end-begin;index+=stride) {
        const int node=scene.nodes[begin+index];
        for(int k=0;k<3;++k)accepted[node][k]=step==0.0?vertices.positions[node][k]
            :__dadd_rn(vertices.positions[node][k],delta[k]);
        // The old scalar loop writes EVERY endpoint even after boxed=false.
        // inside has no error/publication side effects; integer AND is exact.
        const bool boxed=inside(vertices.positions[node],boxes[node])&&inside(accepted[node],boxes[node]);
        if(!boxed)atomicExch(&ctx->safe,0);
    }
}
__global__ void batch_prepare_proxy_trial_finish(int* error,RigidCcdBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    if(*error) {ctx->done=1;ctx->do_pairs=0;return;}
    ctx->do_pairs=ctx->safe;
}

__device__ void batch_enqueue(unsigned slot,unsigned* queue,std::size_t capacity,
    RigidCcdBatchContext* ctx) {
    const unsigned job=atomicAdd(&ctx->queue_count,1u);
    if(static_cast<std::size_t>(job)<capacity)queue[job]=slot;
    else atomicExch(&ctx->queue_overflow,1u);
}
__device__ void batch_min_toi(double* destination,double value) {
    auto* bits=reinterpret_cast<unsigned long long*>(destination);
    auto observed=atomicCAS(bits,0ull,0ull);
    while(true) {
        const double current=__longlong_as_double(static_cast<long long>(observed));
        // Apply CUDA fmin to binary64 operands, including its NaN and signed
        // zero semantics. Never order floating values by their integer bits.
        const auto desired=static_cast<unsigned long long>(__double_as_longlong(fmin(current,value)));
        if(desired==observed)return;
        const auto previous=atomicCAS(bits,observed,desired);
        if(previous==observed)return;
        observed=previous;
    }
}
__device__ void batch_publish(unsigned slot,const PairVerdict& verdict,RigidCcdBatchContext* ctx) {
    if(verdict.error||(ctx->phase!=0&&!verdict.safe)) {
        const auto key=(static_cast<unsigned long long>(slot)<<32)
            |static_cast<unsigned>(verdict.error);
        atomicMin(&ctx->first_bad,key);
        return;
    }
    if(ctx->phase==0&&verdict.collision) {
        atomicOr(&ctx->hit,1u);batch_min_toi(&ctx->toi,verdict.toi);
    }
}
// This function and the fast-stage kernel contain NO Integer/pool calls.
// Their arithmetic is the exact allocation-free prefix already used by the
// original lazy per-pair helpers, including TOI-before-represented validation.
__device__ PairVerdict batch_fast_pair(const BpContact& contact,RigidScene scene,
    DeformableState vertices,const int* colors,int color,int body,const double* motion,
    bool trial,bool zero_step,const double(*accepted)[3],bool& unresolved) {
    PairVerdict verdict{0,false,true,1.0};unresolved=false;
    bool first=false,involved=false;
    verdict.error=original_body_side_local(contact,scene,colors,color,body,first,involved);
    if(verdict.error||!involved)return verdict;
    double positions[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        positions[i][k]=vertices.positions[contact.vertices[i]][k];
    if((!trial||(!zero_step&&(motion[0]!=0.0||motion[1]!=0.0||motion[2]!=0.0)))
        &&ipc_gpu::original_rigid::translation_boxes_overlap(positions,contact.kind==0,first,motion)) {
        const auto result=occd::evaluate_fast(
            ipc_gpu::original_rigid::translation_query(positions,contact.kind==0,first,motion));
        if(result.status==occd::Status::NeedsExact) {unresolved=true;return verdict;}
        verdict.error=original_result_error(result);verdict.collision=result.collision;verdict.toi=result.toi;
        if(verdict.error) {verdict.safe=false;return verdict;}
        if(!trial)return verdict;
        if(result.collision) {verdict.safe=false;return verdict;}
    }
    if(!trial)return verdict;
    const auto q=body_query(contact,scene,vertices,body,accepted);
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        if(!cert::detail::finite(q.start[i][k])||!cert::detail::finite(q.end[i][k])) {
            verdict.error=int(LinearClipError::InvalidInput);verdict.safe=false;return verdict;
        }
    if(cert::detail::separated(q,0.0,1.0,0.0))return verdict;
    unresolved=true;return verdict;
}
__global__ void batch_fast(RigidScene scene,DeformableState vertices,BroadPhaseView bp,
    const int* indices,const int* colors,int color,const double(*accepted)[3],
    RigidCcdBatchContext* ctx,unsigned* queue,std::size_t capacity) {
    if(ctx->mode!=1||ctx->done||!ctx->do_pairs)return;
    const auto count=ctx->count;
    for(std::size_t slot=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;slot<count;
        slot+=std::size_t(gridDim.x)*blockDim.x) {
        bool unresolved=false;
        auto verdict=batch_fast_pair(bp.contacts[indices[ctx->first+slot]],scene,vertices,
            colors,color,ctx->body,ctx->phase==0?ctx->raw:ctx->delta,ctx->phase!=0,
            ctx->zero_step,accepted,unresolved);
        if(unresolved)batch_enqueue(static_cast<unsigned>(slot),queue,capacity,ctx);
        else batch_publish(static_cast<unsigned>(slot),verdict,ctx);
    }
}
__global__ void batch_exact(RigidScene scene,DeformableState vertices,BroadPhaseView bp,
    const int* indices,const int* colors,int color,const double(*accepted)[3],
    RigidCcdBatchContext* ctx,const unsigned* queue,std::size_t capacity,
    std::uint32_t* words,int* slots,int workers) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=static_cast<unsigned>(workers)||ctx->mode!=1||ctx->done||!ctx->do_pairs||ctx->queue_overflow)return;
    const unsigned count=ctx->queue_count;
    for(unsigned job=atomicAdd(&ctx->queue_next,1u);job<count;job=atomicAdd(&ctx->queue_next,1u)) {
        if(static_cast<std::size_t>(job)>=capacity) {
            atomicCAS(&ctx->internal_error,0,int(LinearClipError::ExactResources));continue;
        }
        const unsigned slot=queue[job];
        if(static_cast<std::size_t>(slot)>=ctx->count) {
            atomicCAS(&ctx->internal_error,0,int(LinearClipError::ExactResources));continue;
        }
        const auto& contact=bp.contacts[indices[ctx->first+slot]];
        const auto verdict=ctx->phase==0
            ?original_toi_pair(contact,scene,vertices,colors,color,ctx->body,ctx->raw,words,slots,worker)
            :original_trial_pair(contact,scene,vertices,colors,color,ctx->body,ctx->delta,
                ctx->zero_step,accepted,words,slots,worker);
        batch_publish(slot,verdict,ctx);
    }
}
__global__ void batch_consume_raw(RigidCcdBatchContext* ctx,int* error) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    if(ctx->internal_error) {atomicCAS(error,0,ctx->internal_error);ctx->done=1;return;}
    if(ctx->queue_overflow) {
        // Every producer was speculative. State/proposals are unchanged; the
        // final gated scalar/CTA route recomputes the COMPLETE original call.
        ctx->mode=0;return;
    }
    if(ctx->first_bad!=~0ull) {
        const unsigned code=static_cast<unsigned>(ctx->first_bad);
        atomicCAS(error,0,code?static_cast<int>(code):int(LinearClipError::ExactResources));
        ctx->done=1;return;
    }
    ctx->step=ctx->hit?0.9*ctx->toi:1.0;
}
__global__ void batch_prepare_trial(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const BpAabb* boxes,double(*accepted)[3],int* error,
    RigidCcdBatchContext* ctx,int attempt) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    ctx->attempt=attempt;ctx->phase=1;ctx->queue_count=ctx->queue_next=0;
    ctx->first_bad=~0ull;ctx->queue_overflow=0;
    if(attempt==64)ctx->step=0.0;
    ctx->safe=original_body_trial(scene,state,work,vertices,boxes,ctx->body,ctx->raw,
        ctx->step,accepted,ctx->delta,error);
    if(*error) {ctx->done=1;ctx->do_pairs=0;return;}
    ctx->do_pairs=ctx->safe;
}
__global__ void batch_consume_trial(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,const int* colors,int color,double(*accepted)[3],int* error,
    RigidCcdBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    if(ctx->internal_error) {atomicCAS(error,0,ctx->internal_error);ctx->done=1;return;}
    if(ctx->queue_overflow) {ctx->mode=0;return;}
    if(ctx->safe&&ctx->first_bad!=~0ull) {
        const unsigned code=static_cast<unsigned>(ctx->first_bad);
        if(code) {atomicCAS(error,0,static_cast<int>(code));ctx->done=1;return;}
        ctx->safe=0;
    }
    if(ctx->safe) {
        for(std::size_t body=std::size_t(ctx->body)+1;body<scene.bodies;++body)
            if(colors[scene.body_blocks[body]]==color)
                batch_initialize_body(scene,state,work,vertices,body,accepted);
        ctx->done=1;return;
    }
    ctx->step*=0.5;
    if(ctx->attempt==64) {fail(error,LinearClipError::Unresolved);ctx->done=1;}
}
} // namespace

namespace linear_detail {
void launch_resolve_bodies_original(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd,cudaStream_t stream) {
    resolve_bodies_original<<<1,1,0,stream>>>(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,accepted,error,words,slots,use_ccd,nullptr);
}
void launch_resolve_bodies_original_parallel(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t* offsets,const int* indices,
    const BpAabb* boxes,const int* colors,int color,double(*accepted)[3],int* error,
    std::uint32_t* words,int* slots,int threads,bool use_ccd,cudaStream_t stream) {
    resolve_bodies_original_parallel<<<1,threads,0,stream>>>(scene,state,work,vertices,bp,
        offsets,indices,boxes,colors,color,accepted,error,words,slots,use_ccd,nullptr);
}
void launch_resolve_bodies_original_batch(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t* offsets,const int* indices,
    const BpAabb* boxes,const int* colors,int color,double(*accepted)[3],int* error,
    std::uint32_t* serial_words,int* serial_slots,std::uint32_t* worker_words,int* worker_slots,
    int workers,int fallback_threads,bool use_ccd,RigidCcdBatchContext* context,
    unsigned* queue,std::size_t row_limit,std::size_t queue_capacity,
    rigid_proxy_owner::Permit proxy_permit,cudaStream_t stream) {
    const auto wanted=(bp.contact_count+127)/128;
    const unsigned grid=static_cast<unsigned>(wanted==0?1:wanted>65535?65535:wanted);
    const auto owned_nodes=proxy_permit.proxy_nodes();
    const auto proxy_wanted=(owned_nodes+127)/128;
    const unsigned proxy_grid=static_cast<unsigned>(proxy_wanted==0?1:proxy_wanted>65535?65535:proxy_wanted);
    if(owned_nodes) {
        batch_prepare_proxy_header<<<1,1,0,stream>>>(scene,offsets,colors,color,error,use_ccd,row_limit,context);
        batch_initialize_proxy_prefix<<<proxy_grid,128,0,stream>>>(scene,state,work,vertices,colors,color,accepted,context);
        batch_prepare_proxy_finish<<<1,1,0,stream>>>(state,work,error,context);
    } else {
    batch_prepare<<<1,1,0,stream>>>(scene,state,work,vertices,offsets,colors,color,
        accepted,error,use_ccd,row_limit,context);
    }
    batch_fast<<<grid,128,0,stream>>>(scene,vertices,bp,indices,colors,color,accepted,context,queue,queue_capacity);
    batch_exact<<<workers,1,0,stream>>>(scene,vertices,bp,indices,colors,color,accepted,
        context,queue,queue_capacity,worker_words,worker_slots,workers);
    batch_consume_raw<<<1,1,0,stream>>>(context,error);
    for(int attempt=0;attempt<=64;++attempt) {
        if(owned_nodes) {
            batch_prepare_proxy_trial_header<<<1,1,0,stream>>>(state,work,error,context,attempt);
            batch_write_proxy_trial<<<proxy_grid,128,0,stream>>>(scene,vertices,boxes,accepted,context);
            batch_prepare_proxy_trial_finish<<<1,1,0,stream>>>(error,context);
        } else {
        batch_prepare_trial<<<1,1,0,stream>>>(scene,state,work,vertices,boxes,accepted,error,context,attempt);
        }
        batch_fast<<<grid,128,0,stream>>>(scene,vertices,bp,indices,colors,color,accepted,context,queue,queue_capacity);
        batch_exact<<<workers,1,0,stream>>>(scene,vertices,bp,indices,colors,color,accepted,
            context,queue,queue_capacity,worker_words,worker_slots,workers);
        batch_consume_trial<<<1,1,0,stream>>>(scene,state,work,vertices,colors,color,accepted,error,context);
    }
    if(fallback_threads==1)
        resolve_bodies_original<<<1,1,0,stream>>>(scene,state,work,vertices,bp,offsets,indices,
            boxes,colors,color,accepted,error,serial_words,serial_slots,use_ccd,context);
    else resolve_bodies_original_parallel<<<1,fallback_threads,0,stream>>>(scene,state,work,vertices,bp,
        offsets,indices,boxes,colors,color,accepted,error,worker_words,worker_slots,use_ccd,context);
}
} // namespace linear_detail
} // namespace ipc_gpu::solver

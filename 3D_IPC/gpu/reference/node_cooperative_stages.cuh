#pragma once
// Generated from CURRENT qualified CSR stage bodies. Only declared exact-worker
// adaptation differs: lane0 of block=w retains the old exact worker and arena.

__device__ void cooperative_begin_stage(const int* colors,OriginalUpdateConvention convention,
    const unsigned char* selected,int* error,OriginalNodeCcdControl* control) {
    if(blockIdx.x||threadIdx.x)return;*control=OriginalNodeCcdControl{};
    if(*error){control->done=1;return;}
    if(!colors||(convention==OriginalUpdateConvention::CooperativeSquaredNorm&&!selected)) {
        fail(error,LinearClipError::InvalidInput);control->done=1;
    }
}

__device__ void cooperative_prepare_stage(DeformableScene scene,DeformableState state,
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

__device__ void cooperative_trials_stage(DeformableScene scene,DeformableState state,BroadPhaseView bp,
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

__device__ void cooperative_fast_stage(DeformableScene scene,DeformableState state,BroadPhaseView bp,
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

__device__ void cooperative_fast_partitioned_stage(DeformableScene scene,DeformableState state,
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

__device__ void cooperative_exact_stage(DeformableScene scene,DeformableState state,BroadPhaseView bp,
    const int* indices,OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,
    const unsigned long long* queue,std::size_t capacity,std::uint32_t* words,int* slots,int workers) {
    const unsigned worker=blockIdx.x;
    if(threadIdx.x!=0||worker>=static_cast<unsigned>(workers)||control->mode!=1||control->done||control->queue_overflow)return;
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

__device__ void cooperative_gate_stage(int* error,OriginalNodeCcdControl* control) {
    if(blockIdx.x||threadIdx.x||control->mode!=1||control->done)return;
    if(control->internal_error){atomicCAS(error,0,control->internal_error);control->done=1;return;}
    if(control->queue_overflow){control->mode=0;return;}
    if(!control->pending)control->done=1;
}

__device__ void cooperative_consume_stage(OriginalNodeCcdContext* contexts,OriginalNodeCcdControl* control,
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

// Private CU include after the unchanged qualified tile gathers/publisher.
__device__ bool original_sparse_input(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,RigidScene rigid,ContactParameters params,int color,
    ContactAssemblyPhase phase,int* failed,const unsigned char* separated,
    std::size_t pair,int kind,ContactInput& in) {
    const auto& contact=broad.contacts[pair];
    if(contact.kind!=0&&contact.kind!=1){error(failed,20);return false;}
    if(contact.kind!=kind||(separated&&separated[pair]))return false;
    bool selected[4]{},any_selected=false;
    for(int role=0;role<4;++role) {
        selected[role]=selected_node(scene,rigid,broad,contact.vertices[role],color,phase);
        any_selected=any_selected||selected[role];
    }
    if(!any_selected)return false;
    in.kind=kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=0;
    if(phase==ContactAssemblyPhase::RigidTranslation&&kind==0)in.hessian_blocks=1u;
    else for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j]) {
        if(phase==ContactAssemblyPhase::Deformable&&i!=j)continue;
        in.hessian_blocks|=1u<<(4*i+j);
    }
    for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis) {
        in.positions[role][axis]=state.positions[contact.vertices[role]][axis];
        in.previous[role][axis]=state.previous[contact.vertices[role]][axis];
    }
    return true;
}
__global__ void gather_tile_original_sparse_probe(BroadPhaseView broad,
    DeformableScene scene,DeformableState state,RigidScene rigid,ContactParameters params,
    Work work,int color,ContactAssemblyPhase phase,int* failed,const unsigned char* separated,
    std::size_t first,std::size_t last,int kind,unsigned* queue,unsigned* count) {
    const auto slot=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=last-first)return;
    work.active[slot]=0;
    // Literal old gather early guards, including no validation for k==0.
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(!work.one_slot_terms||!work.output_activity||work.output_activity!=work.active
        ||params.friction!=0.0||phase==ContactAssemblyPhase::Deformable) {
        error(failed,20);return;
    }
    ContactInput actual;
    if(!original_sparse_input(broad,scene,state,rigid,params,color,phase,failed,separated,
        first+slot,kind,actual))return;
    auto probe=actual;
    probe.k_barrier=0.0;probe.hessian_blocks=1u;probe.friction=0.0;
    const auto out=evaluate_contact_original_self_zero(probe,0);
    if(original_sparse_gather::discard_force(actual,out))return;
    const auto rank=atomicAdd(count,1u);
    // At most one append per slot; the exact packet capacity proves this cannot
    // overflow. Still fail closed if an internal count is ever corrupted.
    if(rank>=last-first){error(failed,20);return;}
    queue[rank]=static_cast<unsigned>(slot);
}
template<bool ThinPointTranslation>
__global__ void gather_tile_original_sparse_full(BroadPhaseView broad,
    DeformableScene scene,DeformableState state,RigidScene rigid,ContactParameters params,
    Work work,int color,ContactAssemblyPhase phase,int* failed,const unsigned char* separated,
    std::size_t first,std::size_t last,int kind,const unsigned* queue,const unsigned* count) {
    if(atomicAdd(failed,0)||params.k_barrier==0||params.d_hat<=0)return;
    if(!work.one_slot_terms||!work.output_activity||work.output_activity!=work.active
        ||params.friction!=0.0||phase==ContactAssemblyPhase::Deformable) {
        error(failed,20);return;
    }
    if constexpr(ThinPointTranslation)
        if(phase!=ContactAssemblyPhase::RigidTranslation||kind!=0){error(failed,20);return;}
    const auto queued=*count;
    if(queued>last-first){error(failed,20);return;}
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto rank=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        rank<queued;rank+=stride) {
        if(atomicAdd(failed,0))return;
        const auto slot=static_cast<std::size_t>(queue[rank]);
        if(slot>=last-first){error(failed,20);return;}
        ContactInput in;
        if(!original_sparse_input(broad,scene,state,rigid,params,color,phase,failed,separated,
            first+slot,kind,in))continue;
        work.active[slot]=1;
        if constexpr(ThinPointTranslation) {
            const auto result=evaluate_contact_original_self_zero(in,0);
            store_original_pt_translation_zero(result,work.outputs+slot,work.output_activity+slot);
            if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
        } else {
            const auto result=evaluate_contact_original(in);
            store_contact_output(in,result,work.outputs+slot,nullptr,work.output_activity+slot);
            if(result.status!=ContactStatus::Ok)error(failed,code(result.status));
        }
    }
}

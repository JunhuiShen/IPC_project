#pragma once

// Include after resident_contact_warp_reduce.cuh inside the pipeline's
// anonymous namespace. All numerical expressions reuse its existing helpers.
#if defined(__CUDACC__)
#define IPC_CONTACT_PREFETCH_DEVICE __device__
#else
#define IPC_CONTACT_PREFETCH_DEVICE
#endif
template<class WorkType>
IPC_CONTACT_PREFETCH_DEVICE inline int contact_prefetch_roles(const BroadPhaseView& broad,WorkType work,
    int pair,int node,int& failure) {
    failure=0;
    if(!work.active[pair])return 0;
    if(work.compact) {
        const auto& compact=work.compact[pair];
        if(compact.status!=ContactStatus::Ok){failure=code(compact.status);return 0;}
        if(!compact.active)return 0;
        if(compact.role>=0)
            return broad.contacts[pair].vertices[compact.role]==node?16|(1<<compact.role):0;
    }
    const auto& out=work.outputs[pair];
    if(out.status!=ContactStatus::Ok){failure=code(out.status);return 0;}
    if(!out.active)return 0;
    int roles=0;
    for(int role=0;role<4;++role)if(broad.contacts[pair].vertices[role]==node)roles|=1<<role;
    return roles;
}
// Private Phase2 candidate: producer friction arrays are literal positive zero
// when the unchanged per-assembly friction parameter compares equal to zero.
// Keep both original friction additions and weight multiplication order, so
// signed zero and exceptional weight behavior are preserved. The complete
// compact216/full1416 layouts and producer publication remain unchanged.
template<class CompactOutput>
IPC_CONTACT_PREFETCH_DEVICE inline double contact_prefetch_add_compact_component(
    double value,const CompactOutput& out,int component,double normal_scale,
    bool synthesize_zero_friction) {
    if(!synthesize_zero_friction)
        return contact_warp_add_compact_component(value,out,component,normal_scale);
    const double weight=out.weight;
    if(component<3) {
        value+=normal_scale*out.gradient[component];
        value+=weight*0.0;
    } else {
        const int entry=component-3;
        value+=normal_scale*out.hessian[entry];
        value+=(weight*0.0)*weight;
    }
    return value;
}
#undef IPC_CONTACT_PREFETCH_DEVICE

#if defined(__CUDACC__)
// All 32 lanes fetch consecutive incident metadata. Surviving lanes are then
// visited in increasing lane order, preserving the original incident order.
// Lanes 0..11 still own complete, unchanged scalar accumulation chains.
template<class WorkType>
__global__ void reduce_deformable_prefetch_warp(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,ElasticWorkspace elastic,SdfScene sdf,
    ContactParameters params,WorkType work,int color,const int* block_to_deformable_node,
    int block_count) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned lane=threadIdx.x;
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||!block_to_deformable_node) {
        if(lane==0)error(elastic.error,20);
        return;
    }
    __shared__ double sdf_components[24];
    __shared__ int sdf_status;
    const int begin=broad.color_offsets[color],end=broad.color_offsets[color+1];
    for(std::size_t index=std::size_t(begin)+blockIdx.x;index<std::size_t(end);index+=gridDim.x) {
        int failed=lane==0?atomicAdd(elastic.error,0):0;
        if(__shfl_sync(mask,failed,0))return;
        const int block=broad.color_blocks[index];
        if(block<0||block>=block_count) {
            if(lane==0)error(elastic.error,20);
            return;
        }
        const int node=block_to_deformable_node[block];
        if(node<0)continue;
        if(std::size_t(node)>=scene.nodes||!scene.deformable[node]
            ||scene.node_blocks[node]!=block||broad.block_colors[block]!=color) {
            if(lane==0)error(elastic.error,20);
            return;
        }
        double value=0;
        if(lane<3)value=elastic.systems[node].gradient[lane];
        else if(lane<12)value=elastic.systems[node].hessian[lane-3];
        const double normal_scale=params.dt*params.dt*params.k_barrier;
        const auto incident_end=work.offsets[block+1];
        for(auto chunk=work.offsets[block];chunk<incident_end;chunk+=32) {
            const auto incident=chunk+lane;
            int local_pair=0,local_roles=0,local_failure=0;
            if(incident<incident_end) {
                local_pair=work.incidents[incident].contact;
                local_roles=contact_prefetch_roles(broad,work,local_pair,node,local_failure);
            }
            const unsigned failures=__ballot_sync(mask,local_failure!=0);
            if(failures) {
                // Preserve the first failing incident's status within a node.
                // Earlier arithmetic in this chunk need not run: the original
                // reducer also commits nothing when it reaches this failure.
                const int first=__ffs(failures)-1;
                const int failure=__shfl_sync(mask,local_failure,first);
                if(lane==0)error(elastic.error,failure);
                return;
            }
            unsigned survivors=__ballot_sync(mask,local_roles!=0);
            while(survivors) {
                const int source=__ffs(survivors)-1;
                const int pair=__shfl_sync(mask,local_pair,source);
                const int selected=__shfl_sync(mask,local_roles,source);
                if(lane<12) {
                    if(selected&16)
                        value=contact_prefetch_add_compact_component(value,work.compact[pair],int(lane),normal_scale,
                            params.friction==0.0);
                    else for(int role=0;role<4;++role)if(selected&(1<<role))
                        value=contact_warp_add_component(value,work.outputs[pair],role,int(lane),normal_scale);
                }
                survivors&=survivors-1;
            }
        }
        if(!scene.solid[node]||!work.surface_query||work.surface_query[node]) {
            if(lane==0) {
                const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,
                    state.positions[node],state.previous[node],sdf.parameters);
                sdf_status=out.status==SdfStatus::Ok?0:25;
                for(int k=0;k<3;++k) {
                    sdf_components[k]=out.gradient[k];
                    sdf_components[12+k]=out.friction_gradient[k];
                }
                for(int k=0;k<9;++k) {
                    sdf_components[3+k]=out.hessian[k];
                    sdf_components[15+k]=out.friction_hessian[k];
                }
            }
            __syncwarp(mask);
            if(sdf_status) {
                if(lane==0)error(elastic.error,sdf_status);
                return;
            }
            if(lane<12) {
                const double dt2=params.dt*params.dt;
                value+=dt2*sdf_components[lane];
                value+=sdf_components[12+lane];
            }
        }
        if(__ballot_sync(mask,lane<12&&!contact_detail::finite(value))) {
            if(lane==0)error(elastic.error,26);
            return;
        }
        if(lane<3)elastic.systems[node].gradient[lane]=value;
        else if(lane<12)elastic.systems[node].hessian[lane-3]=value;
        // Prevent the next grid-stride node from overwriting shared SDF data
        // before every lane consumes the current node's values.
        __syncwarp(mask);
    }
}
#endif

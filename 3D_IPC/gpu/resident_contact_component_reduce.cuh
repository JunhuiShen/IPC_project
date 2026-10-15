#pragma once

// Include inside the contact pipeline's anonymous namespace, after the scalar
// node reducer and resident_contact_warp_reduce.cuh. Reuse their arithmetic.
#if defined(__CUDACC__)
#define IPC_CONTACT_COMPONENT_SCHEDULE_HD __host__ __device__
#else
#define IPC_CONTACT_COMPONENT_SCHEDULE_HD
#endif

// Incidence degree is rebuild-stable and measures metadata traversal work even
// for currently inactive contacts. Values below grain retain the scalar node
// routine. Heavier nodes get more separately schedulable component groups;
// each permitted group count divides the twelve ordered scalar chains.
IPC_CONTACT_COMPONENT_SCHEDULE_HD inline unsigned contact_component_groups_for_degree(
    std::size_t degree,std::size_t grain,unsigned limit) {
    if(degree<grain)return 0;
    const std::size_t needed=degree/grain+(degree%grain!=0);
    const unsigned groups=needed<=1?1:needed<=2?2:needed<=3?3:needed<=4?4:needed<=6?6:12;
    return groups<limit?groups:limit;
}

#undef IPC_CONTACT_COMPONENT_SCHEDULE_HD

#if defined(__CUDACC__)
// Distinct blocks own independent scalar chains, never pieces of a sum. Their
// task order may vary; each chain visits every incident and matching role in
// the original order. There is no reduction tree or intermediate partial sum.
// Dense launch slots avoid a per-color scan/readback; cheap unused group slots
// return before traversing contacts. Sparse-color gating bounds these slots.
template<class WorkType>
__global__ void reduce_deformable_component_chains(BroadPhaseView broad,
    DeformableScene scene,ElasticWorkspace elastic,ContactParameters params,
    WorkType work,int color,const int* block_to_deformable_node,int block_count,
    std::size_t grain,unsigned max_groups,Derivatives* pending) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned lane=threadIdx.x;
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||!block_to_deformable_node||!pending) {
        if(lane==0)error(elastic.error,20);
        return;
    }
    const int begin=broad.color_offsets[color],end=broad.color_offsets[color+1];
    const std::size_t task_count=std::size_t(end-begin)*max_groups;
    for(std::size_t task=blockIdx.x;task<task_count;task+=gridDim.x) {
        int failed=lane==0?atomicAdd(elastic.error,0):0;
        if(__shfl_sync(mask,failed,0))return;
        const std::size_t selected=task/max_groups;
        const unsigned group=static_cast<unsigned>(task%max_groups);
        const int block=broad.color_blocks[std::size_t(begin)+selected];
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
        const auto degree=work.offsets[block+1]-work.offsets[block];
        const unsigned groups=contact_component_groups_for_degree(degree,grain,max_groups);
        if(group>=groups)continue;
        const unsigned component=group+lane*groups;
        double value=0;
        if(component<3)value=elastic.systems[node].gradient[component];
        else if(component<12)value=elastic.systems[node].hessian[component-3];
        const double normal_scale=params.dt*params.dt*params.k_barrier;
        for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
            const int pair=work.incidents[incident].contact;
            // Bits 0..3 retain role order; bit 4 selects the compact payload.
            int selected_roles=0;
            if(lane==0&&work.active[pair]) {
                bool full_payload=true;
                if(work.compact) {
                    const auto& compact=work.compact[pair];
                    if(compact.status!=ContactStatus::Ok) {
                        error(elastic.error,code(compact.status));selected_roles=-1;full_payload=false;
                    } else if(!compact.active)full_payload=false;
                    else if(compact.role>=0) {
                        full_payload=false;
                        if(broad.contacts[pair].vertices[compact.role]==node)
                            selected_roles=16|(1<<compact.role);
                    }
                }
                if(full_payload) {
                    const auto& out=work.outputs[pair];
                    if(out.status!=ContactStatus::Ok) {
                        error(elastic.error,code(out.status));selected_roles=-1;
                    } else if(out.active) {
                        for(int role=0;role<4;++role)
                            if(broad.contacts[pair].vertices[role]==node)selected_roles|=1<<role;
                    }
                }
            }
            selected_roles=__shfl_sync(mask,selected_roles,0);
            if(selected_roles<0)return;
            if(component<12) {
                if(selected_roles&16)
                    value=contact_warp_add_compact_component(value,work.compact[pair],int(component),normal_scale);
                else for(int role=0;role<4;++role)if(selected_roles&(1<<role))
                    value=contact_warp_add_component(value,work.outputs[pair],role,int(component),normal_scale);
            }
        }
        // Disjoint double slots: each component has exactly one group/lane
        // owner. The following finalize kernel reads only after completion.
        // Even a nonfinite value remains private until the whole-node check.
        if(component<3)pending[selected].gradient[component]=value;
        else if(component<12)pending[selected].hessian[component-3]=value;
        __syncwarp(mask);
    }
}

template<class WorkType>
__global__ void finalize_deformable_component_chains(BroadPhaseView broad,
    DeformableScene scene,DeformableState state,ElasticWorkspace elastic,SdfScene sdf,
    ContactParameters params,WorkType work,int color,const int* block_to_deformable_node,
    int block_count,std::size_t grain,unsigned max_groups,const Derivatives* pending) {
    const std::size_t selected=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const std::size_t begin=std::size_t(broad.color_offsets[color]);
    if(selected>=std::size_t(broad.color_offsets[color+1])-begin||atomicAdd(elastic.error,0))return;
    const int block=broad.color_blocks[begin+selected];
    if(block<0||block>=block_count){error(elastic.error,20);return;}
    const int node=block_to_deformable_node[block];
    if(node<0)return;
    if(std::size_t(node)>=scene.nodes||!scene.deformable[node]
        ||scene.node_blocks[node]!=block||broad.block_colors[block]!=color) {
        error(elastic.error,20);return;
    }
    const auto degree=work.offsets[block+1]-work.offsets[block];
    if(!contact_component_groups_for_degree(degree,grain,max_groups)) {
        reduce_deformable_node(broad,scene,state,elastic,sdf,params,work,std::size_t(node));
        return;
    }
    Derivatives system=pending[selected];
    if(!scene.solid[node]||!work.surface_query||work.surface_query[node]) {
        // Exactly one unchanged SDF evaluation per selected node. Match the
        // original normal then friction addition order for every component.
        const auto out=evaluate_sdf_contact(sdf.colliders,sdf.count,state.positions[node],state.previous[node],sdf.parameters);
        if(out.status!=SdfStatus::Ok){error(elastic.error,25);return;}
        const double dt2=params.dt*params.dt;
        for(int k=0;k<3;++k) {
            system.gradient[k]+=dt2*out.gradient[k];system.gradient[k]+=out.friction_gradient[k];
            for(int l=0;l<3;++l){system.hessian[3*k+l]+=dt2*out.hessian[3*k+l];system.hessian[3*k+l]+=out.friction_hessian[3*k+l];}
        }
    }
    for(int k=0;k<3;++k)if(!contact_detail::finite(system.gradient[k])){error(elastic.error,26);return;}
    for(int k=0;k<9;++k)if(!contact_detail::finite(system.hessian[k])){error(elastic.error,26);return;}
    // One writer publishes all twelve components after all checks. Failed
    // nodes never publish partial components to the subsequent local solve.
    elastic.systems[node]=system;
}
#endif

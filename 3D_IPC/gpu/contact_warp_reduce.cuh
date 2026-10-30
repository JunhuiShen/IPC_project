#pragma once

// Include INSIDE contact_pipeline.cu's anonymous namespace, after
// Work, error(), code() and the serial reducer helpers have been declared.
// This header introduces no storage, allocations, synchronization with the
// host, contact pruning, or change to contact/role accumulation order.
#if defined(__CUDACC__)
#define IPC_CONTACT_COMPONENT_HD __host__ __device__
#else
#define IPC_CONTACT_COMPONENT_HD
#endif

// One lane owns one scalar throughout the complete original incident list.
// These are deliberately two additions, not a combined contribution or a
// warp/tree reduction. Compile with the existing --fmad=false policy.
IPC_CONTACT_COMPONENT_HD inline double contact_warp_add_component(
    double value,const ContactOutput& out,int role,int component,double normal_scale) {
    const double weight=out.weights[role];
    if(component<3) {
        value+=normal_scale*out.barrier_gradient[role][component];
        value+=weight*out.friction_relative_gradient[component];
    } else {
        const int entry=component-3;
        value+=normal_scale*out.barrier_hessian[role][role][entry];
        value+=(weight*out.friction_relative_hessian[entry])*weight;
    }
    return value;
}

template<class CompactOutput>
IPC_CONTACT_COMPONENT_HD inline double contact_warp_add_compact_component(
    double value,const CompactOutput& out,int component,double normal_scale) {
    const double weight=out.weight;
    if(component<3) {
        value+=normal_scale*out.gradient[component];
        value+=weight*out.friction_gradient[component];
    } else {
        const int entry=component-3;
        value+=normal_scale*out.hessian[entry];
        value+=(weight*out.friction_hessian[entry])*weight;
    }
    return value;
}

#undef IPC_CONTACT_COMPONENT_HD

#if defined(__CUDACC__)
// Exactly one 32-thread block per worker warp. Grid-stride over the already
// existing color-block list; a bounded grid (e.g. min(block_count,256)) avoids
// launching one mostly empty block per scene node. block_to_deformable_node
// has one valid node for each deformable block, and -1 for every rigid block.
// Lanes0..2 own gradient entries, lanes3..11 own row-major Hessian entries.
// One warp/block spreads small colors over SMs instead of packing their few
// selected nodes into a single 128-thread block.
__global__ void reduce_deformable_warp(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,ElasticWorkspace elastic,SdfScene sdf,
    ContactParameters params,Work work,int color,const int* block_to_deformable_node) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned lane=threadIdx.x;
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||!block_to_deformable_node) {
        if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)error(elastic.error,20);
        return;
    }
    __shared__ double sdf_components[24];
    __shared__ int sdf_status;
    const int begin=broad.color_offsets[color],end=broad.color_offsets[color+1];
    for(std::size_t index=std::size_t(begin)+blockIdx.x;index<std::size_t(end);index+=gridDim.x) {
        const int block=broad.color_blocks[index];
        const int node=block_to_deformable_node[block];
        if(node<0)continue;
        if(std::size_t(node)>=scene.nodes||!scene.deformable[node]
            ||scene.node_blocks[node]!=block||broad.block_colors[block]!=color) {
            if(lane==0)error(elastic.error,20);
            return;
        }
        int failed=lane==0?atomicAdd(elastic.error,0):0;
        if(__shfl_sync(mask,failed,0))return;
        double value=0;
        if(lane<3)value=elastic.systems[node].gradient[lane];
        else if(lane<12)value=elastic.systems[node].hessian[lane-3];
        const double normal_scale=params.dt*params.dt*params.k_barrier;
        for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
            const int pair=work.incidents[incident].contact;
            int roles=0;
            if(lane==0&&work.active[pair]) {
                const auto& out=work.outputs[pair];
                if(out.status!=ContactStatus::Ok) {
                    error(elastic.error,code(out.status));roles=-1;
                } else if(out.active) {
                    for(int role=0;role<4;++role)
                        if(broad.contacts[pair].vertices[role]==node)roles|=1<<role;
                }
            }
            roles=__shfl_sync(mask,roles,0);
            if(roles<0)return;
            if(lane<12) {
                // Repeated occurrences of the same node, if present, retain
                // the scalar reducer's role0,role1,role2,role3 order.
                for(int role=0;role<4;++role)if(roles&(1<<role))
                    value=contact_warp_add_component(value,work.outputs[pair],role,int(lane),normal_scale);
            }
        }
        if(!scene.solid[node]||!work.surface_query||work.surface_query[node]) {
            if(lane==0) {
                // The existing SDF routine runs once, with unchanged inputs
                // and arithmetic. Only its final component additions spread
                // over lanes. Shared POD arrays avoid a shared constructor.
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
        // Commit nothing for this node if ANY of its12 components is invalid.
        if(__ballot_sync(mask,lane<12&&!contact_detail::finite(value))) {
            if(lane==0)error(elastic.error,26);
            return;
        }
        if(lane<3)elastic.systems[node].gradient[lane]=value;
        else if(lane<12)elastic.systems[node].hessian[lane-3]=value;
        // A later grid-stride node must not overwrite shared SDF data while
        // another lane is still consuming the current node's values.
        __syncwarp(mask);
    }
}

// Independent compact experiment: do not revive the full-output warp path
// when compact records are enabled. Low-degree nodes keep the scalar loop;
// this pass reads the 216-byte compact stream for each high-degree node.
// WorkType is dependent so standalone tests of the older full-output kernel
// may still include this header with their smaller, full-output-only Work.
template<class WorkType>
__global__ void reduce_deformable_compact_warp(BroadPhaseView broad,DeformableScene scene,
    DeformableState state,ElasticWorkspace elastic,SdfScene sdf,
    ContactParameters params,WorkType work,int color,const int* block_to_deformable_node,
    int block_count,std::size_t degree_cutoff) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned lane=threadIdx.x;
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||!block_to_deformable_node||!work.compact) {
        if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)error(elastic.error,20);
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
        if(work.offsets[block+1]-work.offsets[block]<degree_cutoff)continue;
        double value=0;
        if(lane<3)value=elastic.systems[node].gradient[lane];
        else if(lane<12)value=elastic.systems[node].hessian[lane-3];
        const double normal_scale=params.dt*params.dt*params.k_barrier;
        for(auto incident=work.offsets[block];incident<work.offsets[block+1];++incident) {
            const int pair=work.incidents[incident].contact;
            // Bits0..3 retain role order; bit4 selects the compact payload.
            int selected=0;
            if(lane==0&&work.active[pair]) {
                const auto& compact=work.compact[pair];
                if(compact.status!=ContactStatus::Ok) {
                    error(elastic.error,code(compact.status));selected=-1;
                } else if(compact.active) {
                    if(compact.role>=0) {
                        if(broad.contacts[pair].vertices[compact.role]==node)selected=16|(1<<compact.role);
                    } else {
                        const auto& out=work.outputs[pair];
                        if(out.status!=ContactStatus::Ok) {
                            error(elastic.error,code(out.status));selected=-1;
                        } else if(out.active) {
                            for(int role=0;role<4;++role)
                                if(broad.contacts[pair].vertices[role]==node)selected|=1<<role;
                        }
                    }
                }
            }
            selected=__shfl_sync(mask,selected,0);
            if(selected<0)return;
            if(lane<12) {
                if(selected&16)
                    value=contact_warp_add_compact_component(value,work.compact[pair],int(lane),normal_scale);
                else for(int role=0;role<4;++role)if(selected&(1<<role))
                    value=contact_warp_add_component(value,work.outputs[pair],role,int(lane),normal_scale);
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
        // All12 components must pass before any component is committed.
        if(__ballot_sync(mask,lane<12&&!contact_detail::finite(value))) {
            if(lane==0)error(elastic.error,26);
            return;
        }
        if(lane<3)elastic.systems[node].gradient[lane]=value;
        else if(lane<12)elastic.systems[node].hessian[lane-3]=value;
        __syncwarp(mask);
    }
}
#endif

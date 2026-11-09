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

namespace linear_detail {
void launch_resolve_bodies(RigidScene scene,RigidState state,RigidWorkspace work,
    DeformableState vertices,BroadPhaseView bp,const std::size_t*offsets,const int*indices,
    const BpAabb*boxes,const int*colors,int color,double(*accepted)[3],int*error,
    std::uint32_t*words,int*slots,bool use_ccd,cudaStream_t stream) {
    resolve_bodies<<<1,1,0,stream>>>(scene,state,work,vertices,bp,offsets,indices,boxes,colors,color,accepted,error,words,slots,use_ccd);
}
} // namespace linear_detail
} // namespace ipc_gpu::solver

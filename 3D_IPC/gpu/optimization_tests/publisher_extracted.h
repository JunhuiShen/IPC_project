#pragma once
#include "original_contact.cuh"
#ifdef __CUDACC__
#define IPC_F4_PUBLISH_HD __host__ __device__
#else
#define IPC_F4_PUBLISH_HD
#endif
namespace ipc_gpu::solver {
// Full global packet publication from the unchanged 240-byte value evaluator.
// An inactive activity-gated slot retains its prior full payload exactly.
IPC_F4_PUBLISH_HD inline void store_original_pt_translation_zero(
    const original_contact_storage::SelfZeroOutput& out,ContactOutput* full,
    unsigned char* output_activity) {
    if(output_activity) {
        *output_activity=static_cast<unsigned char>(out.status!=ContactStatus::Ok||out.active!=0);
        if(!*output_activity)return;
    }
    full->distance=out.distance;
    for(int i=0;i<4;++i)full->weights[i]=out.weights[i];
    for(int k=0;k<3;++k)full->normal[k]=out.normal[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        full->barrier_gradient[i][k]=out.barrier_gradient[i][k];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<9;++k)
        full->barrier_hessian[i][j][k]=(i==0&&j==0)?out.selected_hessian[k]:0.0;
    for(int k=0;k<3;++k)full->friction_relative_gradient[k]=0.0;
    for(int k=0;k<9;++k)full->friction_relative_hessian[k]=0.0;
    full->status=out.status;full->active=out.active;
}

}

#undef IPC_F4_PUBLISH_HD

#pragma once
#include "contact.cuh"
#ifdef __CUDACC__
#define IPC_COMPACT_HD __host__ __device__
#else
#define IPC_COMPACT_HD
#endif

namespace ipc_gpu::solver {

// A deformable color normally requests one self-Hessian. Keep only that
// role's derivatives in the hot reduction stream; rigid/multiple-role masks
// still use the complete ContactOutput. Packing performs no arithmetic.
struct CompactDeformableContact {
    ContactStatus status = ContactStatus::Ok;
    unsigned active = 0;
    int role = -1;
    double weight;
    double gradient[3];
    double hessian[9];
    double friction_gradient[3];
    double friction_hessian[9];
};

IPC_COMPACT_HD inline int compact_contact_role(unsigned mask) {
    for(int role=0;role<4;++role)if(mask==(1u<<(5*role)))return role;
    return -1;
}

IPC_COMPACT_HD inline void store_contact_output(const ContactInput& in,
    const ContactOutput& out,ContactOutput* full,CompactDeformableContact* compact,
    unsigned char* output_activity=nullptr) {
    // This optional byte is the pipeline's existing per-pair publication gate.
    // Only a successful, mathematically inactive result may clear it. Pending
    // exact geometry and every failure MUST keep the status/payload observable.
    // Kernel completion publishes both this byte and the payload to the next
    // ordered assembly stage; no consumer races a producer within a kernel.
    if(output_activity) {
        *output_activity=static_cast<unsigned char>(out.status!=ContactStatus::Ok||out.active!=0);
        if(!*output_activity)return;
    }
    if(!compact){*full=out;return;}
    compact->status=out.status;compact->active=out.active;
    compact->role=compact_contact_role(in.hessian_blocks);
    if(compact->role<0){*full=out;return;}
    if(out.status!=ContactStatus::Ok||!out.active)return;
    const int role=compact->role;
    compact->weight=out.weights[role];
    for(int k=0;k<3;++k) {
        compact->gradient[k]=out.barrier_gradient[role][k];
        compact->friction_gradient[k]=out.friction_relative_gradient[k];
    }
    for(int k=0;k<9;++k) {
        compact->hessian[k]=out.barrier_hessian[role][role][k];
        compact->friction_hessian[k]=out.friction_relative_hessian[k];
    }
}

static_assert(sizeof(CompactDeformableContact)<sizeof(ContactOutput)/4,
    "deformable contact stream must stay compact");
} // namespace ipc_gpu::solver
#undef IPC_COMPACT_HD

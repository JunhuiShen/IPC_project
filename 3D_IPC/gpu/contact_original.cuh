#pragma once

// GS-only storage specialization of our maintained ORIGINAL zero-friction
// contact evaluator. All four self-Hessian blocks and all twelve gradient
// components remain computed and validated. No geometry/energy formula changes.
#include "original_contact.cuh"

#if defined(__CUDACC__)
#define IPC_ORIGINAL_CONTACT_HD __host__ __device__
#else
#define IPC_ORIGINAL_CONTACT_HD
#endif

namespace ipc_gpu::solver {
namespace original_contact_storage {
struct OriginalSelfZeroOutput {
    double distance=0.0,weights[4]{},normal[3]{},barrier_gradient[4][3]{};
    double self_hessian[4][9]{};
    ContactStatus status=ContactStatus::Ok;
    unsigned active=0;
};
static_assert(sizeof(OriginalSelfZeroOutput)==456);

template<> struct Access<OriginalSelfZeroOutput> {
    static constexpr bool stores_friction=false;
    IPC_ORIGINAL_CONTACT_HD static bool valid(const ContactInput& in,int mask) {
        return mask==15&&in.friction==0.0&&in.hessian_blocks==0x8421u;
    }
    IPC_ORIGINAL_CONTACT_HD static bool put_hessian(OriginalSelfZeroOutput& out,int i,int j,int k,double value,int mask) {
        if(mask!=15||i<0||i>=4||j!=i||k<0||k>=9)return false;
        out.self_hessian[i][k]=value;return true;
    }
    IPC_ORIGINAL_CONTACT_HD static double hessian(const OriginalSelfZeroOutput& out,int i,int j,int k,int mask) {
        return mask==15&&i==j?out.self_hessian[i][k]:0.0;
    }
    IPC_ORIGINAL_CONTACT_HD static double friction_gradient(const OriginalSelfZeroOutput&,int){return 0.0;}
    IPC_ORIGINAL_CONTACT_HD static double friction_hessian(const OriginalSelfZeroOutput&,int){return 0.0;}
};
} // namespace original_contact_storage

IPC_ORIGINAL_CONTACT_HD inline original_contact_storage::OriginalSelfZeroOutput
evaluate_contact_original_self_zero(const ContactInput& in) {
    return evaluate_contact_original_storage<original_contact_storage::OriginalSelfZeroOutput>(in,15);
}
} // namespace ipc_gpu::solver

#undef IPC_ORIGINAL_CONTACT_HD

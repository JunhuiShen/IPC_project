#pragma once
#include "contact.cuh"
#include <cfloat>
#if defined(__CUDACC__)
#define IPC_ORIGINAL_SPARSE_HD __host__ __device__
#else
#define IPC_ORIGINAL_SPARSE_HD
#endif
namespace ipc_gpu::solver::original_sparse_gather {
// The probe is the existing Original evaluator on a copy with k=0, mask1,
// role0 and friction+0. Geometry, all inputs, weights and normal were validated
// before its no-force return. Only an old full OK/inactive pair may disappear
// from FORCE evaluation; contact/CSR/CCD membership is never changed.
template<class Probe>
IPC_ORIGINAL_SPARSE_HD inline bool discard_force(const ContactInput& actual,const Probe& probe) {
    return actual.k_barrier>=0.0&&actual.k_barrier<=DBL_MAX
        &&actual.friction==0.0&&(actual.hessian_blocks&~0xffffu)==0
        &&probe.status==ContactStatus::Ok&&probe.distance>=actual.d_hat;
}
}
#undef IPC_ORIGINAL_SPARSE_HD

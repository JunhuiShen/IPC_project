#pragma once

#include "rigid_ccd.cuh"

#if defined(__CUDACC__)
#define IPC_PROXY_VALUE_HD __host__ __device__
#else
#define IPC_PROXY_VALUE_HD
#endif

namespace ipc_gpu::solver::proxy_values {

// Scalar components and operation order are the original finish kernel's.
// This helper is shared by device preflight/publication and a host Eigen oracle;
// the unchanged legacy kernel remains the bitwise native differential oracle.
IPC_PROXY_VALUE_HD inline void finish_body(const double* com,const double* previous_com,
    const double* current_omega,bool translate,bool rotate,double dt,
    double* velocity,double* omega) {
    for(int k=0;k<3;++k) {
        velocity[k]=translate?(com[k]-previous_com[k])/dt:0.0;
        omega[k]=rotate?current_omega[k]:0.0;
    }
}

IPC_PROXY_VALUE_HD inline void finish_proxy(const double* reference,
    const double* orientation,const double* velocity,const double* omega,double* result) {
    const double zero[3]={0,0,0};double offset[3];
    rigid_world_position(reference,zero,orientation,offset);
    const auto tangent=rotation_detail::vec(omega).cross(rotation_detail::vec(offset));
    const double rotational[3]={tangent.x,tangent.y,tangent.z};
    for(int k=0;k<3;++k)result[k]=velocity[k]+rotational[k];
}

} // namespace ipc_gpu::solver::proxy_values

#undef IPC_PROXY_VALUE_HD

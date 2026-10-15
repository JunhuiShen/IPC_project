#pragma once
// Isolated host mathematical draft. No production caller includes this header.
#include "compact_point_prefix.cuh"
#include <type_traits>
#if defined(__CUDACC__)
#define IPC_NATIVE_UNUSED_HD __host__ __device__
#else
#define IPC_NATIVE_UNUSED_HD
#endif
namespace ipc_gpu::resident::direct_native_point_study {
namespace cp=contact_cooperative_detail;namespace pc=point_vertex_compact_prefix;
struct NativeValues {double residual[3],bp,bpp;};
static_assert(std::is_trivial<NativeValues>::value&&sizeof(NativeValues)==5*sizeof(double));

// This descriptor owns only floating values; it is not pe::EntryData and makes
// no claim that41 resident Integers exist. Original pre-owner admission is the
// caller's collective proof, checked before try_evaluate reaches this code.
struct FloatingCoefficients {double distance,bp,bpp;unsigned weight_squared;};
IPC_NATIVE_UNUSED_HD inline double mul(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a,b);
#else
    volatile double r=a*b;return r;
#endif
}
IPC_NATIVE_UNUSED_HD inline double add(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,b);
#else
    volatile double r=a+b;return r;
#endif
}
IPC_NATIVE_UNUSED_HD inline double sub(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a,b);
#else
    volatile double r=a-b;return r;
#endif
}
IPC_NATIVE_UNUSED_HD inline double div(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    volatile double r=a/b;return r;
#endif
}

IPC_NATIVE_UNUSED_HD inline bool evaluate_symmetric(const FloatingCoefficients& p,
    const double* normal,double* H) {
    if(p.weight_squared>1||!normal||!H)return false;
    if(!(p.distance>0)||!contact_detail::finite(p.distance)
        ||!contact_detail::finite(p.bp)||!contact_detail::finite(p.bpp))return false;
    if(!(p.bp<=0.0)||!(p.bpp>=0.0))return false;
    for(int k=0;k<3;++k)if(!contact_detail::finite(normal[k]))return false;
    const double c=div(p.bp,p.distance);
    if(!contact_detail::finite(c)||!(c<=0.0))return false;
    for(int k=0;k<3;++k)for(int l=k;l<3;++l) {
        const double t=p.weight_squared?mul(normal[k],normal[l]):0.0;
        const double curvature=p.weight_squared?sub(k==l?1.0:0.0,t):0.0;
        // Retain the original coefficient-times-zero expression for unused roles.
        const double h=add(mul(p.bpp,t),mul(c,curvature));
        if(!contact_detail::finite(h))return false;
        H[3*k+l]=h;H[3*l+k]=h;
    }
    return true;
}

IPC_NATIVE_UNUSED_HD inline void finish_output(
    const ContactInput& in, const NativeValues& owner, ContactOutput& out) {
    if (in.friction > 0) {
        double displacement[3]{}, P[9], u[3]{};
        for (int k = 0; k < 3; ++k) for (int i = 0; i < 4; ++i)
            displacement[k] += out.weights[i]*(in.positions[i][k]-in.previous[i][k]);
        const double normal_length = contact_exact_detail::stable_norm(owner.residual);
        for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
            P[3*k+l] = (k == l ? 1.0 : 0.0)-(owner.residual[k]/normal_length)*(owner.residual[l]/normal_length);
            u[k] += P[3*k+l]*displacement[l];
        }
        const double slip = contact_detail::norm(u), eps_u = in.dt*in.eps_v;
        const double mollifier = slip < eps_u ? 2/eps_u-slip/(eps_u*eps_u) : 1/slip;
        const double scale = in.dt*in.dt*in.friction*(-in.k_barrier*owner.bp)*mollifier;
        for (int k = 0; k < 3; ++k) {
            out.friction_relative_gradient[k] = scale*u[k];
            for (int l = 0; l < 3; ++l) out.friction_relative_hessian[3*k+l] = scale*P[3*k+l];
        }
    }
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        if (!contact_detail::finite(out.barrier_gradient[i][k]) || !contact_detail::finite(out.friction_relative_gradient[k]))
            out.status = ContactStatus::NonFiniteDerivative;
        for (int j = 0; j < 4; ++j) for (int l = 0; l < 3; ++l)
            if (!contact_detail::finite(out.barrier_hessian[i][j][3*k+l]) || !contact_detail::finite(out.friction_relative_hessian[3*k+l]))
                out.status = ContactStatus::NonFiniteDerivative;
    }
    out.active = out.status == ContactStatus::Ok;
}
// Requires an unchanged gate on this immutable record and the ORIGINAL fresh
// owner/helper pool admission (all nine lanes for cooperative execution).
// This bool is an explicit caller proof boundary, not a device-pool emulator.
// Unsupported/inactive/nonfinite results leave every caller output byte intact
// and require the original full C8 path; no partial result is published.
IPC_NATIVE_UNUSED_HD inline bool try_evaluate(const ContactInput& in,int role,
 const pc::Gate& gate,bool original_pools_admitted,ContactOutput& output,
 FloatingCoefficients* coefficient_witness=nullptr) {
 if(!original_pools_admitted||!gate.eligible()||gate.input!=&in||gate.role!=role
    ||in.kind!=ContactKind::PointTriangle||role<0||role>3
    ||gate.vertex<1||gate.vertex>3||cp::self_block(in.hessian_blocks)!=role
    ||gate.coordinate_bits<1||gate.coordinate_bits>99
    ||gate.compact_working_bits!=2*gate.coordinate_bits+58
    ||gate.original_working_bits!=10*gate.coordinate_bits+76)return false;
 NativeValues owner{};ContactOutput out{};
    // D=1: IEEE round-to-nearest subtraction is the same exact dyadic
    // rational conversion. Exact zero must retain the converter's positive zero.
    // The original gate, feature construction and resource admission still apply.
    for(int k=0;k<3;++k){
#if defined(__CUDA_ARCH__)
        const double residual=__dsub_rn(in.positions[0][k],in.positions[gate.vertex][k]);
#else
        const double residual=in.positions[0][k]-in.positions[gate.vertex][k];
#endif
        owner.residual[k]=residual==0.0?0.0:residual;
    }
    for(int i=0;i<4;++i)out.weights[i]=i==0?1.0:(i==gate.vertex?-1.0:0.0);
    out.distance = contact_exact_detail::stable_norm(owner.residual);
    if (!(out.distance > 0) || !contact_detail::finite(out.distance)) {
        out.status = ContactStatus::NonFiniteDerivative; return false;
    }
    for (int k = 0; k < 3; ++k) out.normal[k] = owner.residual[k] / out.distance;
    if (out.distance >= in.d_hat) return false;
    const double distance = out.distance, s = distance - in.d_hat;
    const double logratio = log(distance / in.d_hat), ratio = in.d_hat / distance;
    owner.bp = -2*s*logratio-s*s/distance;
    owner.bpp = ratio*ratio+2*ratio-3-2*logratio;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        out.barrier_gradient[i][k] = (owner.bp*out.weights[i])*out.normal[k];
    }

 // Six independently evaluated values; each off-diagonal is stored twice.
 const FloatingCoefficients coefficients{out.distance,owner.bp,owner.bpp,
     unsigned(role==0||role==gate.vertex)};
 if(!evaluate_symmetric(coefficients,out.normal,out.barrier_hessian[role][role]))return false;
 finish_output(in,owner,out);
 if(out.status!=ContactStatus::Ok)return false;
 if(coefficient_witness)*coefficient_witness=coefficients;
 output=out;return true;
}
} // namespace ipc_gpu::resident::direct_native_point_study
#undef IPC_NATIVE_UNUSED_HD

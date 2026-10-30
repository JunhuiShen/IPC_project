#pragma once
// Exact predicate extraction only; no ContactOutput, Hessian or friction body.
#include "contact.cuh"
#if defined(__CUDACC__)
#define IPC_GUARD103_HD __host__ __device__
#else
#define IPC_GUARD103_HD
#endif
namespace ipc_gpu::solver::point_vertex_guard103 {
using namespace contact_detail;
IPC_GUARD103_HD inline unsigned geometry_reason(const ContactInput& in) {
    const auto& x = in.positions;
    double a[3],b[3],w[3],n[3];
    sub(x[2],x[1],a); sub(x[3],x[1],b); sub(x[0],x[1],w); cross(a,b,n);
    const double nn = norm(n);
    if (!(nn > in.eps)) {
        // Degenerate triangles have a deterministic three-edge fallback in
        // production. Do not pretend a smooth face Hessian exists here.
        return 101u;
    }
    for (int k = 0; k < 3; ++k) n[k] /= nn;
    const double phi = dot(n,w);
    double projected[3],projected_w[3];
    for (int k = 0; k < 3; ++k) projected[k] = x[0][k] - phi*n[k];
    sub(projected,x[1],projected_w);
    const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(projected_w,a),E=dot(projected_w,b);
    const double det = A*C-B*B;
    if (!(det > in.eps) || det <= 64*DBL_EPSILON*A*C)
        return 102u;
    const double l2=(D*C-E*B)/det,l3=(-D*B+E*A)/det,l1=1-l2-l3;
    if (l1>=0 && l2>=0 && l3>=0) return 0u;
    if(l2<=0 && l3<=0) return 0u;
    else if(l3<=0 && l1<=0) return 0u;
    else if(l1<=0 && l2<=0) return 0u;
    else {
        const int ia=l3<0?1:(l1<0?2:3),ib=l3<0?2:(l1<0?3:1);
        const double t=edge_parameter(projected,x[ia],x[ib]);
        // At a clamped endpoint production may retain the edge feature enum
        // while evaluating its unclamped derivative formula. Do not silently
        // choose a different Hessian at that branch boundary.
        if(!(t>1e-14&&t<1-1e-14))return 103u;
        return 0u;
    }
}
IPC_GUARD103_HD inline bool eligible(const ContactInput& in) {
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(in.positions[i][k])||!finite(in.previous[i][k])) {
            return false;
        }
    }
    if(!finite(in.d_hat)||!(in.d_hat>0)||!finite(in.k_barrier)||in.k_barrier<0
        ||!finite(in.dt)||!(in.dt>0)||!finite(in.friction)||in.friction<0
        ||!finite(in.eps_v)||!(in.eps_v>0)||!finite(in.eps)||!(in.eps>0)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge)
        ||(in.hessian_blocks&~0xffffu)) {
        return false;
    }
    if(in.k_barrier==0)return false;
    if(in.kind!=ContactKind::PointTriangle)return false;
    if(geometry_reason(in)!=103u)return false;
    // Same original top-level inactive certificate, evaluated only on the
    // only geometry branch that could yield the target final103 outcome.
    return !certified_contact_inactive(in);
}
#if defined(IPC_GPU_TU_ACTIVE_POINT)
IPC_GUARD103_HD inline bool eligible_before_inactive(const ContactInput& in) {
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(in.positions[i][k])||!finite(in.previous[i][k])) {
            return false;
        }
    }
    if(!finite(in.d_hat)||!(in.d_hat>0)||!finite(in.k_barrier)||in.k_barrier<0
        ||!finite(in.dt)||!(in.dt>0)||!finite(in.friction)||in.friction<0
        ||!finite(in.eps_v)||!(in.eps_v>0)||!finite(in.eps)||!(in.eps>0)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge)
        ||(in.hessian_blocks&~0xffffu)) {
        return false;
    }
    if(in.k_barrier==0)return false;
    if(in.kind!=ContactKind::PointTriangle)return false;
    if(geometry_reason(in)!=103u)return false;
    // Same original top-level inactive certificate, evaluated only on the
    // only geometry branch that could yield the target final103 outcome.
    return true;
}
#endif
} // namespace ipc_gpu::solver::point_vertex_guard103
#undef IPC_GUARD103_HD

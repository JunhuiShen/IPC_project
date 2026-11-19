#pragma once
#ifndef IPC_GPU_CCD_PLANE_FP32_H
#define IPC_GPU_CCD_PLANE_FP32_H

#include "ccd_fp32.cuh"

#if defined(__CUDACC__)
#define IPC_PLANE_HD __host__ __device__
#else
#define IPC_PLANE_HD
#endif

// Experimental bounded-work CCD for contact-independent vertex updates.
// Axes are only proposals. Outward-rounded projection bounds certify them.
// No temporal subdivision, bisection, FP64 or exact-arithmetic fallback.
namespace ipc_gpu { namespace ccd32plane {
using ccd32::LinearQuery;
using ccd32::ClipResult;
using ccd32::Status;
namespace detail {
using namespace ccd32::detail;

IPC_PLANE_HD inline int moving_vertex(const LinearQuery& q) {
    int mover=-1;
    for(int i=0;i<4;++i) {
        bool changed=false;
        for(int k=0;k<3;++k) changed=changed || !same_value(q.start[i][k],q.end[i][k]);
        if(changed) { if(mover>=0)return -2; mover=i; }
    }
    return mover;
}

// Specialized point-axis interval product: two multiplications, not the four
// combinations needed for a general interval times interval operation.
IPC_PLANE_HD inline Interval projected_gap(const float* a,const float* b,Vec n) {
    Interval result=point(0.0f);
    for(int k=0;k<3;++k) {
        const Interval d=subtract(point(a[k]),point(b[k]));
        if(!bounded(d))return whole();
        const bool negative=(bits(n.v[k]) & 0x80000000u)!=0u;
        const float lo=multiply_bound(n.v[k],negative?d.hi:d.lo,false);
        const float hi=multiply_bound(n.v[k],negative?d.lo:d.hi,true);
        if(!finite(lo)||!finite(hi))return whole();
        result=add(result,Interval{lo,hi});
    }
    return result;
}

IPC_PLANE_HD inline float divide_down(float a,float b) {
    if(!finite(a)||!finite(b)||!(a>0.0f)||!(b>0.0f))return 0.0f;
#if defined(__CUDA_ARCH__)
    float r;
    asm("div.rm.f32 %0, %1, %2;" : "=f"(r) : "f"(a),"f"(b));
    return r;
#else
    if(subnormal(a)||subnormal(b))return 0.0f;
    volatile float r=a/b;
    if(!finite(r)||zero(r)||subnormal(r))return 0.0f;
    return maximum(0.0f,neighbor(r,false));
#endif
}

// Ten bounded candidate directions, all generated from the starting geometry.
// Missing an axis can only reject useful motion; it cannot certify a crossing.
IPC_PLANE_HD inline Vec candidate_axis(const LinearQuery& q,int index) {
    Vec n{{0.0f,0.0f,0.0f}};
    if(index<3) { n.v[index]=1.0f; return n; }
    const auto& p=q.start;
    if(q.vertex_face) {
        const Vec normal=cross(difference(p[2],p[1]),difference(p[3],p[1]));
        if(index==3)return normal;
        if(index<7) {
            const int i=index-3;
            return cross(normal,difference(p[i==3?1:i+1],p[i]));
        }
        return difference(p[0],p[index-6]);
    }
    const Vec a=difference(p[1],p[0]),b=difference(p[3],p[2]),w=difference(p[2],p[0]);
    if(index==3)return cross(a,b);
    if(index==4)return cross(cross(a,w),a);
    if(index==5)return cross(cross(b,w),b);
    const int pair=index-6;
    return difference(p[pair/2],p[2+pair%2]);
}

// The same oriented axis must separate EVERY cross-vertex gap at the start
// and endpoint. Their affine interpolation then certifies the whole path,
// including when the moving corner belongs to the triangle or an edge.
IPC_PLANE_HD inline float axis_fraction(const LinearQuery& q,Vec n,bool certificate_only) {
    float scale=0.0f;
    for(int k=0;k<3;++k) {
        if(!finite(n.v[k]))return 0.0f;
        scale=maximum(scale,absolute(n.v[k]));
    }
    if(!(scale>0.0f))return 0.0f;
    for(int k=0;k<3;++k)n.v[k]/=scale;
    const int split=q.vertex_face?1:2;
    const Interval first=projected_gap(q.start[0],q.start[split],n);
    if(!bounded(first))return 0.0f;
    if(first.hi<0.0f)for(int k=0;k<3;++k)n.v[k]=-n.v[k];
    Interval norm2=point(0.0f);
    for(int k=0;k<3;++k)norm2=add(norm2,multiply(point(n.v[k]),point(n.v[k])));
    const float threshold=multiply(point(q.gap_floor),point(sqrt_upper(norm2.hi))).hi;
    if(!finite(threshold)||!(threshold>0.0f))return 0.0f;
    float fraction=1.0f;
    for(int i=0;i<split;++i)for(int j=split;j<4;++j) {
        const Interval start=projected_gap(q.start[i],q.start[j],n);
        const Interval end=projected_gap(q.end[i],q.end[j],n);
        // This experimental path preserves the full floor on one fixed axis.
        // It deliberately blocks unresolved sub-floor starts rather than
        // borrowing clearance from a different axis at the endpoint.
        if(!bounded(start)||!bounded(end)||!(start.lo>threshold))return 0.0f;
        if(end.lo>threshold)continue;
        if(certificate_only||!(start.lo>threshold))return 0.0f;
        // The chord between lower endpoint bounds is a lower bound throughout.
        const float numerator=subtract(point(start.lo),point(threshold)).lo;
        const float denominator=subtract(point(start.lo),point(end.lo)).hi;
        const float limit=divide_down(numerator,denominator);
        // Reserve clearance before the represented endpoint is reconstructed.
        const float interior=multiply_bound(limit,0.99f,false);
        fraction=minimum(fraction,maximum(0.0f,interior));
    }
    return minimum(1.0f,fraction);
}
} // namespace detail

IPC_PLANE_HD inline Status validate_linear(const LinearQuery& q) {
    if(!ccd32::detail::valid(q))return Status::Invalid;
    const int mover=detail::moving_vertex(q);
    if(mover==-2)return Status::Invalid;
    if(mover==-1)return Status::Safe;
    for(int axis=0;axis<10;++axis)
        if(detail::axis_fraction(q,detail::candidate_axis(q,axis),true)==1.0f)return Status::Safe;
    return Status::Uncertain;
}

// Unrounded proposal only. The GPU batch must certify the stored endpoint
// after reducing all pairs' proposals for the moving vertex.
IPC_PLANE_HD inline float propose_fraction(const LinearQuery& q) {
    if(!ccd32::detail::valid(q)||detail::moving_vertex(q)==-2)return -1.0f;
    if(detail::moving_vertex(q)==-1)return 1.0f;
    float best=0.0f;
    for(int axis=0;axis<10 && best<1.0f;++axis)
        best=ccd32::detail::maximum(best,detail::axis_fraction(q,detail::candidate_axis(q,axis),false));
    return best;
}

IPC_PLANE_HD inline ClipResult clip_linear(const LinearQuery& q) {
    ClipResult result;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)result.endpoint[i][k]=q.start[i][k];
    if(!ccd32::detail::valid(q)||detail::moving_vertex(q)==-2) {
        result.status=Status::Invalid; return result;
    }
    float best=detail::moving_vertex(q)==-1?1.0f:0.0f;
    for(int axis=0;axis<10 && best<1.0f;++axis) {
        best=ccd32::detail::maximum(best,detail::axis_fraction(q,detail::candidate_axis(q,axis),false));
        ++result.checks;
    }
    LinearQuery trial=q;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        trial.end[i][k]=ccd32::detail::interpolate(q.start[i][k],q.end[i][k],best);
    // A bounded final proof accounts for FP32 endpoint reconstruction. There is
    // deliberately no iterative repair: uncertainty produces no new motion.
    if(best>0.0f && (best==1.0f || ccd32plane::validate_linear(trial)==Status::Safe)) {
        result.fraction=best;
        result.full_step=best==1.0f;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)result.endpoint[i][k]=trial.end[i][k];
    }
    result.status=Status::Safe;
    return result;
}
} } // namespace ipc_gpu::ccd32plane
#undef IPC_PLANE_HD
#endif

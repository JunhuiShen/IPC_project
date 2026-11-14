#pragma once

// Isolated GS execution candidate. Original predicates and headers are intact.
#include "original_rigid.cuh"

#if defined(__CUDACC__)
#define IPC_GS_CACHE_HD __host__ __device__
#else
#define IPC_GS_CACHE_HD
#endif

namespace ipc_gpu::gs_ccd_cache {

struct Axis {
    ccd::detail::Vec direction{};
    double norm=0.0;
    double threshold=0.0;
    bool usable=false;
};

struct PreparedPath {
    ccd::detail::Interval base[4][3]{};
    ccd::detail::Interval motion[4][3]{};
    Axis axes[17]{};
    int count=0;
    int split=0;
};

// Optional proof-only trace. A null pointer removes these stores in normal use.
struct ProjectionTrace {
    double first_lo=0.0,first_hi=0.0,second_lo=0.0,second_hi=0.0;
    double ab=0.0,ba=0.0;
};

IPC_GS_CACHE_HD inline Axis prepare_axis(ccd::detail::Vec axis) {
    using namespace ccd::detail;
    Axis result;
    result.direction=axis;
    // Identical order and operations to the original projection preamble.
    double scale=0.0;
    for(int k=0;k<3;++k)scale=maximum(scale,::fabs(axis.v[k]));
    if(!(scale>0.0)||!finite(scale))return result;
    for(int k=0;k<3;++k)axis.v[k]/=scale;
    result.direction=axis;
    Interval norm2=point(0.0);
    for(int k=0;k<3;++k)
        norm2=add(norm2,multiply(point(axis.v[k]),point(axis.v[k])));
    result.norm=sqrt_upper(norm2.hi);
    result.threshold=multiply(point(0.0),point(result.norm)).hi;
    result.usable=finite(result.threshold);
    return result;
}

IPC_GS_CACHE_HD inline void append_axis(PreparedPath& prepared,ccd::detail::Vec axis) {
    prepared.axes[prepared.count++]=prepare_axis(axis);
}

IPC_GS_CACHE_HD inline PreparedPath prepare(const ccd::LinearQuery& q) {
    using namespace ccd::detail;
    PreparedPath prepared;
    prepared.split=q.vertex_face?1:2;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        prepared.base[i][k]=subtract(point(q.start[i][k]),point(q.start[3][k]));
        prepared.motion[i][k]=subtract(point(q.end[i][k]),point(q.start[i][k]));
    }
    // Exactly the original separated() axis order. Preparing later axes early
    // has no predicate/pool side effects; all cached inputs are immutable.
    for(int k=0;k<3;++k) {
        Vec axis{{0.0,0.0,0.0}};axis.v[k]=1.0;
        append_axis(prepared,axis);
    }
    for(int endpoint=0;endpoint<2;++endpoint) {
        const double (*p)[3]=endpoint?q.end:q.start;
        if(q.vertex_face) {
            const Vec a=difference(p[2],p[1]),b=difference(p[3],p[1]);
            const Vec normal=cross(a,b);
            append_axis(prepared,normal);
            for(int i=1;i<4;++i) {
                const Vec edge=difference(p[i==3?1:i+1],p[i]);
                append_axis(prepared,cross(normal,edge));
            }
        } else {
            const Vec a=difference(p[1],p[0]),b=difference(p[3],p[2]);
            const Vec w=difference(p[2],p[0]);
            append_axis(prepared,cross(a,b));
            append_axis(prepared,cross(cross(a,w),a));
            append_axis(prepared,cross(cross(b,w),b));
        }
        const int split=q.vertex_face?1:2;
        for(int i=0;i<split;++i)for(int j=split;j<4;++j)
            append_axis(prepared,difference(p[i],p[j]));
    }
    return prepared;
}

IPC_GS_CACHE_HD inline bool projection_zero_gap(const PreparedPath& prepared,const Axis& axis,
    double time_lo,double time_hi,ProjectionTrace* trace=nullptr) {
    using namespace ccd::detail;
    if(!axis.usable)return false;
    double first_lo=INFINITY,first_hi=-INFINITY;
    double second_lo=INFINITY,second_hi=-INFINITY;
    for(int i=0;i<4;++i) {
        Interval projected=point(0.0);
        for(int k=0;k<3;++k) {
            // Do not distribute dot products over time, change interval
            // expressions, or reassociate these original operations.
            const Interval base=prepared.base[i][k];
            const Interval motion=prepared.motion[i][k];
            const Interval position=add(base,multiply({time_lo,time_hi},motion));
            projected=add(projected,multiply(point(axis.direction.v[k]),position));
        }
        if(i<prepared.split) {
            first_lo=minimum(first_lo,projected.lo);
            first_hi=maximum(first_hi,projected.hi);
        } else {
            second_lo=minimum(second_lo,projected.lo);
            second_hi=maximum(second_hi,projected.hi);
        }
    }
    const double ab=subtract(point(first_lo),point(second_hi)).lo;
    const double ba=subtract(point(second_lo),point(first_hi)).lo;
    if(trace) {
        trace->first_lo=first_lo;trace->first_hi=first_hi;
        trace->second_lo=second_lo;trace->second_hi=second_hi;
        trace->ab=ab;trace->ba=ba;
    }
    return maximum(ab,ba)>axis.threshold;
}

IPC_GS_CACHE_HD inline bool separated_zero_gap(const PreparedPath& prepared,
    double time_lo,double time_hi) {
    for(int i=0;i<prepared.count;++i)
        if(projection_zero_gap(prepared,prepared.axes[i],time_lo,time_hi))return true;
    return false;
}

// Original wrapper decisions and statuses are unchanged. Cache only after
// the original whole-step rejection and exact initial-membership check.
template<int Capacity>
IPC_GS_CACHE_HD inline original_rigid::RepresentationStatus represented_path(
    const double start[4][3],const double end[4][3],bool face) {
    using original_rigid::RepresentationStatus;
    ccd::LinearQuery q;
    q.vertex_face=face;
    int moving=0;
    for(int i=0;i<4;++i) {
        bool moved=false;
        for(int k=0;k<3;++k) {
            if(!ccd::detail::finite(start[i][k])||!ccd::detail::finite(end[i][k]))
                return RepresentationStatus::Invalid;
            q.start[i][k]=start[i][k];q.end[i][k]=end[i][k];
            moved=moved||start[i][k]!=end[i][k];
        }
        moving+=moved?1:0;
    }
    if(ccd::detail::separated(q,0.0,1.0,0.0))return RepresentationStatus::Safe;
    const auto convert=[](exact::Status status) {
        return status==exact::Status::Safe?RepresentationStatus::Safe
            :status==exact::Status::InitialContact?RepresentationStatus::InitialContact
            :status==exact::Status::CapacityExceeded?RepresentationStatus::CapacityExceeded
            :status==exact::Status::Invalid?RepresentationStatus::Invalid
            :RepresentationStatus::Uncertain;
    };
    if(moving<=1)return convert(exact::validate_collision_step<Capacity>(start,end,face));
    const auto initial=exact::validate_collision_step<Capacity>(start,start,face);
    if(initial!=exact::Status::Safe)return convert(initial);
    const PreparedPath prepared=prepare(q);
    constexpr int intervals=256;
    for(int i=0;i<intervals;++i)
        if(!separated_zero_gap(prepared,double(i)/intervals,double(i+1)/intervals))
            return RepresentationStatus::Uncertain;
    return RepresentationStatus::Safe;
}

} // namespace ipc_gpu::gs_ccd_cache

#undef IPC_GS_CACHE_HD

#pragma once
#ifndef IPC_GPU_CCD_CORE_H
#define IPC_GPU_CCD_CORE_H

// Device-only, fail-closed CCD building block. This is a CERTIFICATE, not the
// CPU exact-rational CCD port: Uncertain must never be interpreted as no hit.
// The caller must use the returned represented endpoint, not recompute it.
// No host geometry call or host exact-arithmetic fallback exists here.

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define IPC_GPU_CCD_HD __host__ __device__
#else
#define IPC_GPU_CCD_HD
#endif

namespace ipc_gpu {
namespace ccd {

enum class Status : unsigned { Safe, Uncertain, Invalid };

struct LinearQuery {
    double start[4][3]{};
    double end[4][3]{};
    bool vertex_face = true;
    double gap_floor = 1e-8;
};

struct ClipResult {
    Status status = Status::Uncertain;
    double fraction = 0.0;
    double endpoint[4][3]{};
    int checks = 0;
    // A partial certified step is deliberately distinguishable from a full
    // accepted proposal. The unaccepted remainder was NOT proven colliding.
    bool full_step = false;
};

namespace detail {

#if defined(IPC_GPU_TU_FINITE_CCD)
IPC_GPU_CCD_HD inline bool finite(double x) {
#if defined(__CUDA_ARCH__)
    const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(x));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
#else
    return x - x == 0.0;
#endif
}
#else
IPC_GPU_CCD_HD inline bool finite(double x) { return x - x == 0.0; }
#endif
IPC_GPU_CCD_HD inline double minimum(double a, double b) { return a < b ? a : b; }
IPC_GPU_CCD_HD inline double maximum(double a, double b) { return a > b ? a : b; }
IPC_GPU_CCD_HD inline double clamp01(double x) { return maximum(0.0, minimum(1.0, x)); }

struct Interval { double lo, hi; };
struct Vec { double v[3]; };

IPC_GPU_CCD_HD inline Interval whole() { return {-INFINITY, INFINITY}; }
IPC_GPU_CCD_HD inline Interval point(double x) { return {x, x}; }
IPC_GPU_CCD_HD inline double neighbor(double x,bool upward) {
#if defined(__CUDA_ARCH__)
    if(x!=x)return x;
    if(x==0.0)return __longlong_as_double(upward?1LL:static_cast<long long>(0x8000000000000001ULL));
    if((upward&&x==INFINITY)||(!upward&&x==-INFINITY))return x;
    unsigned long long bits=static_cast<unsigned long long>(__double_as_longlong(x));
    if(upward==(x>0.0))++bits;else --bits;
    return __longlong_as_double(static_cast<long long>(bits));
#else
    return ::nextafter(x,upward?INFINITY:-INFINITY);
#endif
}
IPC_GPU_CCD_HD inline Interval outward(double lo, double hi) {
    if (!finite(lo) || !finite(hi)) return whole();
    return {neighbor(lo,false),neighbor(hi,true)};
}
IPC_GPU_CCD_HD inline Interval add(Interval a, Interval b) {
#if defined(__CUDA_ARCH__)
    const double lo = __dadd_rd(a.lo, b.lo), hi = __dadd_ru(a.hi, b.hi);
    return finite(lo) && finite(hi) ? Interval{lo, hi} : whole();
#else
    return outward(a.lo + b.lo, a.hi + b.hi);
#endif
}
IPC_GPU_CCD_HD inline Interval subtract(Interval a, Interval b) {
#if defined(__CUDA_ARCH__)
    const double lo = __dsub_rd(a.lo, b.hi), hi = __dsub_ru(a.hi, b.lo);
    return finite(lo) && finite(hi) ? Interval{lo, hi} : whole();
#else
    return outward(a.lo - b.hi, a.hi - b.lo);
#endif
}
IPC_GPU_CCD_HD inline Interval multiply(Interval a, Interval b) {
    double lo = INFINITY, hi = -INFINITY;
    const double av[2] = {a.lo, a.hi}, bv[2] = {b.lo, b.hi};
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
#if defined(__CUDA_ARCH__)
        const double l = __dmul_rd(av[i], bv[j]), h = __dmul_ru(av[i], bv[j]);
#else
        const double value = av[i] * bv[j];
        const double l = ::nextafter(value, -INFINITY), h = ::nextafter(value, INFINITY);
#endif
        if (!finite(l) || !finite(h)) return whole();
        lo = minimum(lo, l); hi = maximum(hi, h);
    }
    return {lo, hi};
}
IPC_GPU_CCD_HD inline double sqrt_upper(double x) {
    if (!(x >= 0.0) || !finite(x)) return INFINITY;
#if defined(__CUDA_ARCH__)
    return __dsqrt_ru(x);
#else
    return ::nextafter(::sqrt(x), INFINITY);
#endif
}
IPC_GPU_CCD_HD inline Vec difference(const double* a, const double* b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}
IPC_GPU_CCD_HD inline Vec cross(Vec a, Vec b) {
    return {{a.v[1]*b.v[2]-a.v[2]*b.v[1], a.v[2]*b.v[0]-a.v[0]*b.v[2],
             a.v[0]*b.v[1]-a.v[1]*b.v[0]}};
}
IPC_GPU_CCD_HD inline double dot(Vec a, Vec b) {
    return a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2];
}

// Approximate axes only propose a direction. Every acceptance is made by
// outward-rounded projection bounds, so their rounding cannot weaken safety.
IPC_GPU_CCD_HD inline bool projection(const LinearQuery& q, Vec axis,
    double time_lo, double time_hi, double required_gap) {
    double scale = 0.0;
    for (int k = 0; k < 3; ++k) scale = maximum(scale, ::fabs(axis.v[k]));
    if (!(scale > 0.0) || !finite(scale)) return false;
    for (int k = 0; k < 3; ++k) axis.v[k] /= scale;
    Interval norm2 = point(0.0);
    for (int k = 0; k < 3; ++k)
        norm2 = add(norm2, multiply(point(axis.v[k]), point(axis.v[k])));
    const double norm = sqrt_upper(norm2.hi);
    const double threshold = multiply(point(required_gap), point(norm)).hi;
    if (!finite(threshold)) return false;
    const int split = q.vertex_face ? 1 : 2;
    double first_lo = INFINITY, first_hi = -INFINITY;
    double second_lo = INFINITY, second_hi = -INFINITY;
    for (int i = 0; i < 4; ++i) {
        Interval projected = point(0.0);
        for (int k = 0; k < 3; ++k) {
            // Local origin reduces cancellation; interval subtraction still
            // encloses the EXACT represented-coordinate differences.
            const Interval base = subtract(point(q.start[i][k]), point(q.start[3][k]));
            const Interval motion = subtract(point(q.end[i][k]), point(q.start[i][k]));
            const Interval position = add(base, multiply({time_lo, time_hi}, motion));
            projected = add(projected, multiply(point(axis.v[k]), position));
        }
        if (i < split) {
            first_lo = minimum(first_lo, projected.lo);
            first_hi = maximum(first_hi, projected.hi);
        } else {
            second_lo = minimum(second_lo, projected.lo);
            second_hi = maximum(second_hi, projected.hi);
        }
    }
    const double ab = subtract(point(first_lo), point(second_hi)).lo;
    const double ba = subtract(point(second_lo), point(first_hi)).lo;
    return maximum(ab, ba) > threshold;
}

IPC_GPU_CCD_HD inline bool separated(const LinearQuery& q,
    double time_lo, double time_hi, double gap) {
    for (int k = 0; k < 3; ++k) {
        Vec axis{{0.0, 0.0, 0.0}}; axis.v[k] = 1.0;
        if (projection(q, axis, time_lo, time_hi, gap)) return true;
    }
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const double (*p)[3] = endpoint ? q.end : q.start;
        if (q.vertex_face) {
            const Vec a = difference(p[2], p[1]), b = difference(p[3], p[1]);
            const Vec normal = cross(a, b);
            if (projection(q, normal, time_lo, time_hi, gap)) return true;
            for (int i = 1; i < 4; ++i) {
                const Vec edge = difference(p[i == 3 ? 1 : i+1], p[i]);
                if (projection(q, cross(normal, edge), time_lo, time_hi, gap)) return true;
            }
        } else {
            const Vec a = difference(p[1], p[0]), b = difference(p[3], p[2]);
            const Vec w = difference(p[2], p[0]);
            if (projection(q, cross(a,b), time_lo,time_hi,gap)
                || projection(q,cross(cross(a,w),a),time_lo,time_hi,gap)
                || projection(q,cross(cross(b,w),b),time_lo,time_hi,gap)) return true;
        }
        // Vertex differences handle degenerate point/edge configurations.
        const int split = q.vertex_face ? 1 : 2;
        for (int i = 0; i < split; ++i) for (int j = split; j < 4; ++j)
            if (projection(q,difference(p[i],p[j]),time_lo,time_hi,gap)) return true;
    }
    return false;
}

// Upper bound on the starting distance from ANY feasible point pair. Vertex
// pairs suffice for the common gap >= floor case; near-contact comparisons
// may remain inconclusive and are intentionally not guessed from a tolerance.
IPC_GPU_CCD_HD inline double start_gap_upper(const LinearQuery& q) {
    const int split = q.vertex_face ? 1 : 2;
    double upper = INFINITY;
    for (int i = 0; i < split; ++i) for (int j = split; j < 4; ++j) {
        Interval norm2 = point(0.0);
        for (int k = 0; k < 3; ++k) {
            const Interval d = subtract(point(q.start[i][k]),point(q.start[j][k]));
            norm2 = add(norm2,multiply(d,d));
        }
        upper = minimum(upper,sqrt_upper(norm2.hi));
    }
    return upper;
}

IPC_GPU_CCD_HD inline bool valid(const LinearQuery& q) {
    if (!(q.gap_floor > 0.0) || !finite(q.gap_floor)) return false;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        if (!finite(q.start[i][k]) || !finite(q.end[i][k])) return false;
    return true;
}

} // namespace detail

// A strict projection separation proves that every point on both primitive
// trajectories remains separated. Optional temporal subdivision improves the
// certificate but never changes Uncertain into an unsafe acceptance.
// No assumption of a single moving vertex is needed for this certificate.
IPC_GPU_CCD_HD inline Status validate_linear(const LinearQuery& q,
    int subdivision_depth = 6) {
    using namespace detail;
    if (!valid(q) || subdivision_depth < 0 || subdivision_depth > 10)
        return Status::Invalid;
    // CPU far-start future events retain a 1e-10 finite-primitive validation
    // band. A certificate must not weaken that policy even when a caller
    // chooses a numerical gap floor smaller than the default 1e-8.
    if (separated(q,0.0,1.0,maximum(q.gap_floor,1e-10))) return Status::Safe;
    bool unchanged = true;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        unchanged = unchanged && q.start[i][k] == q.end[i][k];
    if (unchanged) return separated(q,0.0,0.0,0.0) ? Status::Safe : Status::Uncertain;

    const double starting_upper = start_gap_upper(q);
    const double minimum_gap = minimum(q.gap_floor,starting_upper);
    const double validation_gap = starting_upper<=1e-10?0.0:1e-10;
    if (!separated(q,1.0,1.0,minimum_gap)) return Status::Uncertain;
    if (separated(q,0.0,1.0,validation_gap)) return Status::Safe;
    const int intervals = 1 << subdivision_depth;
    for (int i = 0; i < intervals; ++i) {
        const double a = static_cast<double>(i) / intervals;
        const double b = static_cast<double>(i+1) / intervals;
        if (!separated(q,a,b,validation_gap)) return Status::Uncertain;
    }
    return Status::Safe;
}

// Find a certified represented endpoint by conservative backtracking. This is
// NOT an earliest-impact estimator. On Uncertain, fraction remains zero and
// no update is authorized. A partial result is safe only for endpoint[][];
// reconstructing it with different contraction/rounding needs revalidation.
IPC_GPU_CCD_HD inline ClipResult clip_linear(const LinearQuery& q,
    int max_attempts = 64, int subdivision_depth = 6) {
    ClipResult result;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        result.endpoint[i][k] = q.start[i][k];
    if (!detail::valid(q) || max_attempts < 1 || max_attempts > 64) {
        result.status = Status::Invalid; return result;
    }
    LinearQuery trial = q;
    double fraction = 1.0;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (attempt) for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
            trial.end[i][k] = ::fma(fraction,q.end[i][k]-q.start[i][k],q.start[i][k]);
        ++result.checks;
        const Status status = validate_linear(trial,subdivision_depth);
        if (status == Status::Invalid) { result.status = status; return result; }
        if (status == Status::Safe) {
            result.status = status; result.fraction = fraction; result.full_step = attempt == 0;
            for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
                result.endpoint[i][k] = trial.end[i][k];
            return result;
        }
        fraction *= 0.5;
    }
    return result;
}

} // namespace ccd
} // namespace ipc_gpu

#undef IPC_GPU_CCD_HD

#endif // IPC_GPU_CCD_CORE_H

// IPC_CCD_SECTION_BEGIN cache
#ifndef IPC_GPU_CCD_CACHE_H
#define IPC_GPU_CCD_CACHE_H
// GS execution cache. Shared original predicates remain the source of truth.
#include "exact.cuh"

// original_rigid.cuh includes the core CCD header. Delay use of its status
// members until template instantiation instead of creating an include cycle.
namespace ipc_gpu::original_rigid { enum class RepresentationStatus : unsigned; }

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
template<int Capacity, class RepresentationStatus=original_rigid::RepresentationStatus>
IPC_GS_CACHE_HD inline RepresentationStatus represented_path(
    const double start[4][3],const double end[4][3],bool face) {
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
#endif // IPC_GPU_CCD_CACHE_H
// IPC_CCD_SECTION_END cache

// IPC_CCD_SECTION_BEGIN single_vertex
#ifndef IPC_GPU_CCD_SINGLE_VERTEX_H
#define IPC_GPU_CCD_SINGLE_VERTEX_H
// Sufficient represented-endpoint proof for one-moving-vertex GS only.
// Uncertain results retain the existing exact CCD path; this returns no TOI
// and never authorizes a shortened endpoint that has not itself been checked.
#include "separation.cuh"

#if defined(__CUDACC__)
#define IPC_GS_CCD_FILTER_HD __host__ __device__
#else
#define IPC_GS_CCD_FILTER_HD
#endif

namespace ipc_gpu::gs_ccd_filter {

// Enclose det(p1-p0,p2-p0,p3-p0), including every subtraction error.
// The existing interval helpers use directed CUDA FP64 operations; overflow,
// underflow uncertainty and cancellation can only erase a usable sign.
IPC_GS_CCD_FILTER_HD inline int determinant_sign(const double p[4][3]) {
    using namespace ccd::detail;
    Interval a[3], b[3], c[3];
    for(int k=0;k<3;++k) {
        a[k]=subtract(point(p[1][k]),point(p[0][k]));
        b[k]=subtract(point(p[2][k]),point(p[0][k]));
        c[k]=subtract(point(p[3][k]),point(p[0][k]));
    }
    Interval value=point(0.0);
    for(int k=0;k<3;++k) {
        const int j=(k+1)%3,l=(k+2)%3;
        const Interval cross=subtract(multiply(a[j],b[l]),multiply(a[l],b[j]));
        value=add(value,multiply(cross,c[k]));
    }
    if(!finite(value.lo)||!finite(value.hi))return 0;
    return value.lo>0.0?1:(value.hi<0.0?-1:0);
}

IPC_GS_CCD_FILTER_HD inline bool safe(const double start[4][3],const double end[4][3],
                                    bool face,double gap_floor) {
    using ccd::detail::finite;
    if(!(gap_floor>0.0)||!finite(gap_floor))return false;
    int moved=0;
    for(int i=0;i<4;++i) {
        bool changed=false;
        for(int k=0;k<3;++k) {
            if(!finite(start[i][k])||!finite(end[i][k]))return false;
            changed=changed||start[i][k]!=end[i][k];
        }
        moved+=changed?1:0;
    }
    if(moved!=1)return false;

    // This deliberately leaves all near-start cases to the original policy.
    // It excludes exact initial contact and the original finite-primitive
    // 1e-10 membership band. separation::safe's double overload additionally
    // rejects any coordinate that is not exactly represented by float32.
    const double start_gap=ccd::detail::maximum(gap_floor,1e-10);
    if(!separation::safe(start,start,face,start_gap))return false;
    const int initial=determinant_sign(start);
    if(initial==0||determinant_sign(end)!=initial)return false;

    // With one moving corner the determinant is affine along the ACTUAL
    // represented segment. Equal nonzero endpoint signs exclude all possible
    // coplanar PT/EE contact events, including initial contact. The final
    // strict distance proof gives d(end)>floor>=min(d(start),floor), exactly
    // the extra endpoint condition required by exact::validate_step.
    // Failed/degenerate/near-floor proofs return false rather than guessing.
    return separation::safe(end,end,face,gap_floor);
}

} // namespace ipc_gpu::gs_ccd_filter

#undef IPC_GS_CCD_FILTER_HD
#endif // IPC_GPU_CCD_SINGLE_VERTEX_H
// IPC_CCD_SECTION_END single_vertex

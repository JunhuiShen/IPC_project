#pragma once

// Pure FP32 enclosure of a rigid rotation. The entire normalized-quaternion
// curve is checked, including connectors to the represented endpoint meshes.
// A failed certificate only requests backtracking; it never means no contact.
#include "ccd_fp32.cuh"

#if defined(__CUDACC__)
#define IPC_RIGID_CCD32_HD __host__ __device__
#else
#define IPC_RIGID_CCD32_HD
#endif

namespace ipc_gpu::rigid_ccd32 {
using ccd32::Status;
using namespace ccd32::detail;

IPC_RIGID_CCD32_HD inline float divide(float a, float b, bool up) {
#if defined(__CUDA_ARCH__)
    float out;
    if (up) asm("div.rp.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    else asm("div.rm.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
#else
    if (zero(a)) return 0.0f;
    if (subnormal(a) || subnormal(b)) return nan();
    volatile float out = a / b;
    if (zero(out) || subnormal(out)) return nan();
    return neighbor(out, up);
#endif
}

IPC_RIGID_CCD32_HD inline float sqrt_lower(float a) {
#if defined(__CUDA_ARCH__)
    float out; asm("sqrt.rm.f32 %0, %1;" : "=f"(out) : "f"(a)); return out;
#else
    if (subnormal(a)) return 0.0f;
    return maximum(0.0f, neighbor(::sqrtf(a), false));
#endif
}

IPC_RIGID_CCD32_HD inline Interval square(Interval x) {
    const Interval a = multiply(point(x.lo), point(x.lo));
    const Interval b = multiply(point(x.hi), point(x.hi));
    return {x.lo <= 0.0f && x.hi >= 0.0f ? 0.0f : minimum(a.lo, b.lo),
            maximum(a.hi, b.hi)};
}

IPC_RIGID_CCD32_HD inline Interval divide_positive(Interval x, Interval y) {
    if (!(y.lo > 0.0f) || !finite(y.hi)) return whole();
    const float a = divide(x.lo, y.lo, false), b = divide(x.lo, y.hi, false);
    const float c = divide(x.hi, y.lo, true), d = divide(x.hi, y.hi, true);
    if (!finite(a) || !finite(b) || !finite(c) || !finite(d)) return whole();
    return {minimum(a, b), maximum(c, d)};
}

// Quaternion arrays use scalar-first order. Scaling before normalization
// avoids unnecessary overflow in both interpolation and endpoint projection.
IPC_RIGID_CCD32_HD inline bool normalize(const float in[4], float out[4]) {
    float scale = 0.0f;
    for (int k = 0; k < 4; ++k) {
        if (!finite(in[k])) return false;
        scale = maximum(scale, ::fabsf(in[k]));
    }
    if (!(scale > 0.0f)) return false;
    float squared = 0.0f;
    for (int k = 0; k < 4; ++k) { out[k] = in[k] / scale; squared += out[k] * out[k]; }
    const float norm = ::sqrtf(squared);
    if (!(norm > 0.0f) || !finite(norm)) return false;
    for (int k = 0; k < 4; ++k) out[k] /= norm;
    return true;
}

IPC_RIGID_CCD32_HD inline bool interpolate(const float current[4], const float target[4],
                                    float fraction, float out[4]) {
    if (fraction == 0.0f || fraction == 1.0f) {
        for (int k = 0; k < 4; ++k) out[k] = fraction == 0.0f ? current[k] : target[k];
        return true;
    }
    float a[4], b[4], blend[4];
    if (!normalize(current, a) || !normalize(target, b)) return false;
    float product = 0.0f;
    for (int k = 0; k < 4; ++k) product += a[k] * b[k];
    if (product < 0.0f) for (int k = 0; k < 4; ++k) b[k] = -b[k];
    for (int k = 0; k < 4; ++k) blend[k] = (1.0f - fraction) * a[k] + fraction * b[k];
    return normalize(blend, out);
}

IPC_RIGID_CCD32_HD inline bool world_point(const float reference[3], const float center[3],
                                    const float orientation[4], float out[3]) {
    float q[4];
    if (!normalize(orientation, q)) return false;
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float matrix[3][3] = {
        {w*w+x*x-y*y-z*z, 2.0f*(x*y-w*z), 2.0f*(x*z+w*y)},
        {2.0f*(x*y+w*z), w*w-x*x+y*y-z*z, 2.0f*(y*z-w*x)},
        {2.0f*(x*z-w*y), 2.0f*(y*z+w*x), w*w-x*x-y*y+z*z}};
    for (int k = 0; k < 3; ++k) {
        out[k] = center[k] + ((matrix[k][0]*reference[0] + matrix[k][1]*reference[1])
                              + matrix[k][2]*reference[2]);
        if (!finite(out[k])) return false;
    }
    return true;
}

struct Query {
    float start[4][3], end[4][3], reference[4][3], center[3];
    float current[4], target[4];
    bool selected[4], vertex_face;
    float gap_floor=1.0e-8f;
};

IPC_RIGID_CCD32_HD inline bool quaternion_bounds(const Query& query, float lo, float hi, Interval q[4]) {
    Interval norm2 = point(0.0f);
    for (int k = 0; k < 4; ++k) {
        q[k] = add(point(query.current[k]), multiply({lo, hi},
            subtract(point(query.target[k]), point(query.current[k]))));
        norm2 = add(norm2, square(q[k]));
    }
    if (!(norm2.lo > 0.0f) || !finite(norm2.hi)) return false;
    const Interval norm{sqrt_lower(norm2.lo), sqrt_upper(norm2.hi)};
    for (int k = 0; k < 4; ++k) {
        q[k] = divide_positive(q[k], norm);
        if (!finite(q[k].lo) || !finite(q[k].hi)) return false;
    }
    return true;
}

IPC_RIGID_CCD32_HD inline Interval twice(Interval x) { return multiply(point(2.0f), x); }

IPC_RIGID_CCD32_HD inline bool position_bounds(const Query& query, float lo, float hi, Interval p[4][3]) {
    Interval q[4];
    if (!quaternion_bounds(query, lo, hi, q)) return false;
    const Interval w2=square(q[0]), x2=square(q[1]), y2=square(q[2]), z2=square(q[3]);
    const Interval xy=multiply(q[1],q[2]), xz=multiply(q[1],q[3]), yz=multiply(q[2],q[3]);
    const Interval wx=multiply(q[0],q[1]), wy=multiply(q[0],q[2]), wz=multiply(q[0],q[3]);
    const Interval matrix[3][3] = {
        {subtract(add(w2,x2),add(y2,z2)), twice(subtract(xy,wz)), twice(add(xz,wy))},
        {twice(add(xy,wz)), subtract(add(w2,y2),add(x2,z2)), twice(subtract(yz,wx))},
        {twice(subtract(xz,wy)), twice(add(yz,wx)), subtract(add(w2,z2),add(x2,y2))}};
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        p[i][k] = point(query.start[i][k]);
        if (query.selected[i]) {
            Interval value = point(query.center[k]);
            for (int j = 0; j < 3; ++j) value = add(value, multiply(matrix[k][j], point(query.reference[i][j])));
            // Enclose the short connectors between the analytic rigid curve
            // and the actual stored endpoints, including FP32 projection error.
            if (lo == 0.0f) { value.lo=minimum(value.lo,query.start[i][k]); value.hi=maximum(value.hi,query.start[i][k]); }
            if (hi == 1.0f) { value.lo=minimum(value.lo,query.end[i][k]); value.hi=maximum(value.hi,query.end[i][k]); }
            p[i][k] = value;
        }
        if (!finite(p[i][k].lo) || !finite(p[i][k].hi)) return false;
    }
    return true;
}

IPC_RIGID_CCD32_HD inline bool projection(const Interval p[4][3], bool face, Vec axis, float gap) {
    float scale=0.0f;
    for (int k=0;k<3;++k) scale=maximum(scale,::fabsf(axis.v[k]));
    if (!(scale>0.0f) || !finite(scale)) return false;
    for (int k=0;k<3;++k) axis.v[k]/=scale;
    Interval norm2=point(0.0f);
    for(int k=0;k<3;++k)norm2=add(norm2,multiply(point(axis.v[k]),point(axis.v[k])));
    const float threshold=multiply(point(gap),point(sqrt_upper(norm2.hi))).hi;
    if(!finite(threshold))return false;
    const int split=face?1:2;
    float alo=INFINITY, ahi=-INFINITY, blo=INFINITY, bhi=-INFINITY;
    for (int i=0;i<4;++i) {
        Interval value=point(0.0f);
        for (int k=0;k<3;++k) value=add(value,multiply(p[i][k],point(axis.v[k])));
        if (!finite(value.lo)||!finite(value.hi)) return false;
        if(i<split){alo=minimum(alo,value.lo);ahi=maximum(ahi,value.hi);}
        else{blo=minimum(blo,value.lo);bhi=maximum(bhi,value.hi);}
    }
    const float distance=maximum(subtract(point(alo),point(bhi)).lo,subtract(point(blo),point(ahi)).lo);
    return finite(distance)&&(bits(distance)&0x80000000u)==0u&&!zero(distance)
        &&order(distance)>order(threshold);
}

IPC_RIGID_CCD32_HD inline bool separated(const Query& query, float lo, float hi, float gap) {
    Interval p[4][3];
    if (!position_bounds(query,lo,hi,p)) return false;
    for(int k=0;k<3;++k){Vec axis{{0.0f,0.0f,0.0f}};axis.v[k]=1.0f;
        if(projection(p,query.vertex_face,axis,gap))return true;}
    float sample[4][3], orientation[4];
    if(!interpolate(query.current,query.target,(lo+hi)*0.5f,orientation))return false;
    for(int i=0;i<4;++i) {
        if(query.selected[i]) {
            if(!world_point(query.reference[i],query.center,orientation,sample[i]))return false;
        }else for(int k=0;k<3;++k)sample[i][k]=query.start[i][k];
    }
    if(query.vertex_face) {
        const Vec normal=cross(difference(sample[2],sample[1]),difference(sample[3],sample[1]));
        if(projection(p,true,normal,gap))return true;
        for(int i=1;i<4;++i)if(projection(p,true,cross(normal,difference(sample[i==3?1:i+1],sample[i])),gap))return true;
    }else{
        const Vec a=difference(sample[1],sample[0]),b=difference(sample[3],sample[2]),w=difference(sample[2],sample[0]);
        if(projection(p,false,cross(a,b),gap)||projection(p,false,cross(cross(a,w),a),gap)
            ||projection(p,false,cross(cross(b,w),b),gap))return true;
    }
    const int split=query.vertex_face?1:2;
    for(int i=0;i<split;++i)for(int j=split;j<4;++j)
        if(projection(p,query.vertex_face,difference(sample[i],sample[j]),gap))return true;
    return false;
}

IPC_RIGID_CCD32_HD inline Status validate_rotation(Query query, int depth=6) {
    if(depth<0||depth>10)return Status::Invalid;
    // Normalize only to choose a stable quaternion sign. Keep every input
    // bit in the interval curve: rounded normalization is not an exact pose.
    float a[4],b[4];
    if(!normalize(query.current,a)||!normalize(query.target,b))return Status::Invalid;
    float product=0.0f;
    for(int k=0;k<4;++k)product+=a[k]*b[k];
    if(product<0.0f)for(int k=0;k<4;++k)query.target[k]=-query.target[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        if(!finite(query.start[i][k])||!finite(query.end[i][k])||!finite(query.reference[i][k])||!finite(query.center[k]))return Status::Invalid;
    ccd32::LinearQuery endpoints;
    endpoints.vertex_face=query.vertex_face;endpoints.gap_floor=query.gap_floor;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){endpoints.start[i][k]=query.start[i][k];endpoints.end[i][k]=query.end[i][k];}
    if(!valid(endpoints))return Status::Invalid;
    if(!ccd32::detail::separated(endpoints,1.0f,1.0f,query.gap_floor))return Status::Uncertain;
    const float gap=ccd32::detail::separated(endpoints,0.0f,0.0f,query.gap_floor)?query.gap_floor:0.0f;
    if(separated(query,0.0f,1.0f,gap))return Status::Safe;
    const int count=1<<depth;
    const float width=1.0f/float(count);
    for(int i=0;i<count;++i)if(!separated(query,float(i)*width,float(i+1)*width,gap))return Status::Uncertain;
    return Status::Safe;
}
} // namespace ipc_gpu::rigid_ccd32

#undef IPC_RIGID_CCD32_HD

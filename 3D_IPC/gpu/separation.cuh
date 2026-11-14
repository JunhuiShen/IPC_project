#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define IPC_SEPARATION_HD __host__ __device__
#else
#define IPC_SEPARATION_HD
#endif

namespace ipc_gpu::separation {

// A one-sided sufficient certificate. Every point of an affine primitive
// trajectory lies in the convex hull of that primitive's endpoint vertices.
// Outward projection bounds for the two endpoint hulls therefore enclose the
// complete trajectories. A strict projected gap greater than gap*||axis||
// proves the requested Euclidean gap throughout the step. Failure proves
// nothing: callers must retain their existing binary64/exact fallback.
//
// Axes are heuristics only. Their rounded construction need not enclose an
// exact normal: ANY finite nonzero represented axis gives a valid proof when
// its projections and norm bounds pass. Zero, overflow and nonfinite results
// simply make that axis inconclusive.

IPC_SEPARATION_HD inline std::uint32_t bits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return result;
#endif
}
IPC_SEPARATION_HD inline float from_bits(std::uint32_t value) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(value);
#else
    float result;std::memcpy(&result,&value,sizeof(result));return result;
#endif
}
IPC_SEPARATION_HD inline std::uint64_t bits(double value) {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
    std::uint64_t result;std::memcpy(&result,&value,sizeof(result));return result;
#endif
}
IPC_SEPARATION_HD inline double from_bits(std::uint64_t value) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(value));
#else
    double result;std::memcpy(&result,&value,sizeof(result));return result;
#endif
}
IPC_SEPARATION_HD inline bool finite(float value) {
    return (bits(value)&0x7f800000u)!=0x7f800000u;
}
IPC_SEPARATION_HD inline bool finite(double value) {
    return (bits(value)&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
IPC_SEPARATION_HD inline bool zero(float value) {
    return (bits(value)&0x7fffffffu)==0;
}
IPC_SEPARATION_HD inline double exact_double(float value) {
#if defined(__CUDA_ARCH__)
    double result;
    asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));
    return result;
#else
    // Decode the represented bits, including subnormals, independently of a
    // caller's single-precision DAZ/FTZ mode.
    const auto raw=bits(value);const unsigned exponent=(raw>>23)&255u;
    if(exponent==255u)return static_cast<double>(value);
    const unsigned mantissa=(raw&0x7fffffu)+(exponent?0x800000u:0u);
    const auto scale=from_bits(static_cast<std::uint64_t>(exponent?exponent+873u:874u)<<52);
    const double result=static_cast<double>(mantissa)*scale;
    return raw&0x80000000u?-result:result;
#endif
}
IPC_SEPARATION_HD inline float rounded_float(double value) {
#if defined(__CUDA_ARCH__)
    float result;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
    return result;
#else
    return static_cast<float>(value);
#endif
}
IPC_SEPARATION_HD inline float neighbor(float value,bool up) {
    auto raw=bits(value);
    if((raw&0x7fffffffu)==0)return from_bits(up?1u:0x80000001u);
    raw+=(up==((raw&0x80000000u)==0))?1u:std::uint32_t(-1);
    return from_bits(raw);
}
IPC_SEPARATION_HD inline double upper_neighbor(double value) {
    auto raw=bits(value);
    if((raw&UINT64_C(0x7fffffffffffffff))==0)return from_bits(static_cast<std::uint64_t>(1));
    raw+=(raw>>63)?std::uint64_t(-1):UINT64_C(1);
    return from_bits(raw);
}

// Integer ordering preserves subnormal values even in a surrounding FTZ
// module. Nonfinite coordinates are rejected before these comparisons.
IPC_SEPARATION_HD inline std::uint32_t order(float value) {
    const auto raw=bits(value);return raw&0x80000000u?~raw:raw|0x80000000u;
}
IPC_SEPARATION_HD inline float lower(float a,float b) {return order(a)<order(b)?a:b;}
IPC_SEPARATION_HD inline float upper(float a,float b) {return order(a)>order(b)?a:b;}

#if !defined(__CUDA_ARCH__)
IPC_SEPARATION_HD inline float cpu_bound(double value,bool up,bool exact_operation) {
    const float result=rounded_float(value);
    if(!finite(result)||!finite(value))return static_cast<float>(NAN);
    const double represented=exact_double(result);
    const double magnitude=value<0?-value:value;
    // A host FTZ conversion may erase many subnormal ULPs. It is not an
    // outward bound; reject rather than treating zero as a nearby rounding.
    if(zero(result)&&magnitude>0x1p-149)return static_cast<float>(NAN);
    if(exact_operation) {
        if((up&&represented<value)||(!up&&represented>value))return neighbor(result,up);
        return result;
    }
    // Double rounding can cross a F32 boundary. Two extra F32 neighbors
    // enclose it even under a directed host rounding mode; zero identities
    // and exact products are handled separately.
    return neighbor(neighbor(result,up),up);
}
#endif

// Explicit PTX omits .ftz. The certificate remains valid for subnormal
// operands/results regardless of surrounding compiler contraction/FTZ flags.
IPC_SEPARATION_HD inline float add(float a,float b,bool up) {
#if defined(__CUDA_ARCH__)
    float result;
    if(up)asm("add.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    else asm("add.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    return result;
#else
    if(zero(a))return b;
    if(zero(b))return a;
    return cpu_bound(exact_double(a)+exact_double(b),up,false);
#endif
}
IPC_SEPARATION_HD inline float subtract(float a,float b,bool up) {
#if defined(__CUDA_ARCH__)
    float result;
    if(up)asm("sub.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    else asm("sub.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    return result;
#else
    if(zero(b))return a;
    if(bits(a)==bits(b))return 0.0f;
    return cpu_bound(exact_double(a)-exact_double(b),up,false);
#endif
}
IPC_SEPARATION_HD inline float multiply(float a,float b,bool up) {
#if defined(__CUDA_ARCH__)
    float result;
    if(up)asm("mul.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    else asm("mul.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    return result;
#else
    // Two F32 significands have at most 48 product bits, so binary64
    // represents this product exactly, without overflow or underflow.
    return cpu_bound(exact_double(a)*exact_double(b),up,true);
#endif
}
IPC_SEPARATION_HD inline double add_upper(double a,double b) {
#if defined(__CUDA_ARCH__)
    double result;asm("add.rp.f64 %0, %1, %2;" : "=d"(result) : "d"(a),"d"(b));return result;
#else
    return upper_neighbor(a+b);
#endif
}
IPC_SEPARATION_HD inline double multiply_upper(double a,double b) {
#if defined(__CUDA_ARCH__)
    double result;asm("mul.rp.f64 %0, %1, %2;" : "=d"(result) : "d"(a),"d"(b));return result;
#else
    const double result=a*b;
    if(result==0&&a>0&&b>0)return DBL_MIN; // conservative underflow/FTZ enclosure
    return upper_neighbor(result);
#endif
}

struct Axis {float value[3];};
struct Interval {float lo,hi;};
IPC_SEPARATION_HD inline Axis difference(const float* a,const float* b) {
    return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};
}
IPC_SEPARATION_HD inline Axis cross(const Axis& a,const Axis& b) {
    return {{a.value[1]*b.value[2]-a.value[2]*b.value[1],
             a.value[2]*b.value[0]-a.value[0]*b.value[2],
             a.value[0]*b.value[1]-a.value[1]*b.value[0]}};
}
IPC_SEPARATION_HD inline Interval project(const float* p,const Axis& axis) {
    float lo=multiply(p[0],axis.value[0],false),hi=multiply(p[0],axis.value[0],true);
    for(int k=1;k<3;++k) {
        lo=add(lo,multiply(p[k],axis.value[k],false),false);
        hi=add(hi,multiply(p[k],axis.value[k],true),true);
    }
    return {lo,hi};
}
IPC_SEPARATION_HD inline bool gap_proves(float a_hi,float b_lo,double gap,double norm_squared_upper) {
    const float projected=subtract(b_lo,a_hi,false);
    // Directed positive overflow can saturate at maxF32 rather than infinity.
    // Keep that extremal bound inconclusive as well as nonfinite arithmetic.
    if(!finite(projected)||(bits(projected)&0x7fffffffu)==0x7f7fffffu)return false;
    const double distance=exact_double(projected);
    if(!(distance>0))return false;
    if(norm_squared_upper==1.0)return distance>gap;
    // Squared comparison avoids a square root. The left product is exact
    // (at most 48 significand bits); every right operation rounds upward.
    const double limit=multiply_upper(multiply_upper(gap,gap),norm_squared_upper);
    return finite(limit)&&distance*distance>limit;
}
IPC_SEPARATION_HD inline bool unit_axis(const float start[4][3],const float end[4][3],
    int endpoints,int split,int coordinate,double gap) {
    float alo=INFINITY,ahi=-INFINITY,blo=INFINITY,bhi=-INFINITY;
    for(int endpoint=0;endpoint<endpoints;++endpoint) {
        const auto p=endpoint?end:start;
        for(int i=0;i<4;++i) {
            const float value=p[i][coordinate];
            if(i<split){alo=lower(alo,value);ahi=upper(ahi,value);}
            else {blo=lower(blo,value);bhi=upper(bhi,value);}
        }
    }
    return gap_proves(ahi,blo,gap,1.0)||gap_proves(bhi,alo,gap,1.0);
}
IPC_SEPARATION_HD inline bool axis_proves(const float start[4][3],const float end[4][3],
    int endpoints,int split,const Axis& axis,double gap) {
    int nonzero=0;double squared[3];
    for(int k=0;k<3;++k) {
        if(!finite(axis.value[k]))return false;
        nonzero+=!zero(axis.value[k]);
        const double value=exact_double(axis.value[k]);squared[k]=value*value;
    }
    // Coordinate directions have already been tested more tightly above.
    if(nonzero<=1)return false;
    const double norm_squared_upper=add_upper(add_upper(squared[0],squared[1]),squared[2]);
    if(!(norm_squared_upper>0)||!finite(norm_squared_upper))return false;
    float alo=INFINITY,ahi=-INFINITY,blo=INFINITY,bhi=-INFINITY;
    for(int endpoint=0;endpoint<endpoints;++endpoint) {
        const auto p=endpoint?end:start;
        for(int i=0;i<4;++i) {
            const auto interval=project(p[i],axis);
            if(!finite(interval.lo)||!finite(interval.hi))return false;
            if(i<split){alo=lower(alo,interval.lo);ahi=upper(ahi,interval.hi);}
            else {blo=lower(blo,interval.lo);bhi=upper(bhi,interval.hi);}
        }
    }
    return gap_proves(ahi,blo,gap,norm_squared_upper)||gap_proves(bhi,alo,gap,norm_squared_upper);
}

IPC_SEPARATION_HD inline bool safe(const float start[4][3],const float end[4][3],bool face,double gap) {
    const auto gap_bits=bits(gap);
    if(!finite(gap)||((gap_bits>>63)&&(gap_bits&UINT64_C(0x7fffffffffffffff))))return false;
    bool unchanged=true;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(start[i][k])||!finite(end[i][k]))return false;
        unchanged=unchanged&&bits(start[i][k])==bits(end[i][k]);
    }
    const int endpoints=unchanged?1:2,split=face?1:2;
    for(int k=0;k<3;++k)if(unit_axis(start,end,endpoints,split,k,gap))return true;
    for(int endpoint=0;endpoint<endpoints;++endpoint) {
        const auto p=endpoint?end:start;
        if(face) {
            const auto a=difference(p[2],p[1]),b=difference(p[3],p[1]);
            const auto normal=cross(a,b);
            if(axis_proves(start,end,endpoints,split,normal,gap))return true;
            for(int i=1;i<4;++i) {
                const auto edge=difference(p[i==3?1:i+1],p[i]);
                if(axis_proves(start,end,endpoints,split,cross(normal,edge),gap))return true;
            }
        }else {
            const auto a=difference(p[1],p[0]),b=difference(p[3],p[2]),w=difference(p[2],p[0]);
            if(axis_proves(start,end,endpoints,split,cross(a,b),gap)
                ||axis_proves(start,end,endpoints,split,cross(cross(a,w),a),gap)
                ||axis_proves(start,end,endpoints,split,cross(cross(b,w),b),gap))return true;
        }
        for(int i=0;i<split;++i)for(int j=split;j<4;++j)
            if(axis_proves(start,end,endpoints,split,difference(p[i],p[j]),gap))return true;
    }
    return false;
}

// The fast proof's domain is represented F32 geometry. Binary64 callers may
// use this overload only when conversion loses no input bits. In particular,
// a nearby double endpoint must never be silently rounded into a safe one.
IPC_SEPARATION_HD inline bool safe(const double start[4][3],const double end[4][3],bool face,double gap) {
    float a[4][3],b[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(start[i][k])||!finite(end[i][k]))return false;
        a[i][k]=rounded_float(start[i][k]);b[i][k]=rounded_float(end[i][k]);
        if(!finite(a[i][k])||!finite(b[i][k])
            ||bits(exact_double(a[i][k]))!=bits(start[i][k])
            ||bits(exact_double(b[i][k]))!=bits(end[i][k]))return false;
    }
    return safe(a,b,face,gap);
}

} // namespace ipc_gpu::separation

#undef IPC_SEPARATION_HD

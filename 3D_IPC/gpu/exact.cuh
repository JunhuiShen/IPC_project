#pragma once

// Exact dyadic predicates for GPU cold paths. All input bits are retained.
// Arithmetic has an explicit capacity failure; it never silently truncates.
// No Boost, CPU callback, dynamic allocation, or host fallback is used.
#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_GPU_EXACT_HD __host__ __device__
#else
#define IPC_GPU_EXACT_HD
#endif

namespace ipc_gpu {
namespace exact {

enum class Status : unsigned { Safe, Unsafe, InitialContact, Invalid, CapacityExceeded };

// Each exact CUDA worker installs its own pool before constructing ANY Integer.
// Limbs live in an explicitly bounded, DISJOINT global scratch slice rather
// than a per-thread 100KB--MB automatic stack frame. The installed pointer is
// thread-private within block-local shared storage, so independent blocks and
// concurrent kernel launches do not overwrite one another's allocator context.
// No Integer may outlive its owner's pool or be transferred to another worker.
struct DevicePool {
    std::uint32_t* words=nullptr;
    int* free_slots=nullptr;
    int words_per_slot=0;
    int slot_count=0;
    int available=0;
    int peak=0;
    bool failed=false;
};
#if defined(__CUDACC__)
// CUDA supports at most 1024 threads in a block. A single non-templated accessor
// owns the shared symbol used by initialize, Integer, verdict, and release.
// Static shared memory belongs to one block/launch, NOT to the translation unit
// as device-global memory would. Each thread reads/writes only its own slot;
// no block barrier is needed, including when some workers return early.
// Shared slots start uninitialized: initialize_pool MUST precede every lookup
// or Integer construction (the same installation precondition as before).
static __device__ __forceinline__ DevicePool*& thread_pool_slot() {
    __shared__ DevicePool* slots[1024];
    const unsigned local=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    return slots[local];
}
static __device__ __forceinline__ DevicePool* current_pool() {
    return thread_pool_slot();
}
__device__ inline void initialize_pool(DevicePool& pool,std::uint32_t* words,
    int* free_slots,int words_per_slot,int slot_count) {
    // Do not inspect the previous (possibly uninitialized) shared slot.
    thread_pool_slot()=nullptr;
    pool.words=words;pool.free_slots=free_slots;pool.words_per_slot=words_per_slot;
    pool.slot_count=slot_count;pool.available=0;pool.peak=0;
    pool.failed=!words||!free_slots||words_per_slot<=0||slot_count<=0;
    if(pool.failed)return;
    pool.available=slot_count;
    for(int i=0;i<slot_count;++i) free_slots[i]=i;
    thread_pool_slot()=&pool;
}
__device__ inline void release_pool() { thread_pool_slot()=nullptr; }
#endif

template<int Capacity>
struct Integer {
    static_assert(Capacity >= 2,"at least two 32-bit limbs required");
#if defined(__CUDA_ARCH__)
    std::uint32_t* limb=nullptr;
    DevicePool* pool=nullptr;
    int slot=-1;
#else
    std::uint32_t limb[Capacity];
#endif
    int used = 0;
    int sign = 0;
    bool valid = true;
    IPC_GPU_EXACT_HD Integer() {
#if defined(__CUDA_ARCH__)
        pool=current_pool();
        if(!pool||pool->words_per_slot<Capacity||pool->available<=0) {
            valid=false;if(pool)pool->failed=true;return;
        }
        slot=pool->free_slots[--pool->available];
        limb=pool->words+std::size_t(slot)*pool->words_per_slot;
        const int used_slots=pool->slot_count-pool->available;
        if(used_slots>pool->peak)pool->peak=used_slots;
#endif
    }
    IPC_GPU_EXACT_HD ~Integer() {
#if defined(__CUDA_ARCH__)
        if(pool&&slot>=0)pool->free_slots[pool->available++]=slot;
#endif
    }
    IPC_GPU_EXACT_HD Integer(std::int64_t value):Integer() {
        if(!valid)return;
        const std::uint64_t magnitude = value < 0
            ? std::uint64_t(-(value+1))+1 : std::uint64_t(value);
        if (magnitude) {
            limb[0] = std::uint32_t(magnitude);
            limb[1] = std::uint32_t(magnitude >> 32);
            used = limb[1] ? 2 : 1; sign = value < 0 ? -1 : 1;
        }
    }
    IPC_GPU_EXACT_HD Integer(const Integer& other):Integer() { *this = other; }
    IPC_GPU_EXACT_HD Integer& operator=(const Integer& other) {
        valid=valid&&other.valid;
        if(!valid){used=0;sign=0;return *this;}
        used=other.used; sign=other.sign;
        for (int i=0;i<used;++i) limb[i]=other.limb[i];
        return *this;
    }
    IPC_GPU_EXACT_HD void normalize() {
        while (used && limb[used-1]==0) --used;
        if (!used) sign=0;
    }
};

template<int C> IPC_GPU_EXACT_HD inline Integer<C> invalid_integer() {
    Integer<C> r; r.valid=false; return r;
}
template<int C> IPC_GPU_EXACT_HD inline int compare_magnitude(const Integer<C>& a,const Integer<C>& b) {
    if(a.used!=b.used) return a.used<b.used?-1:1;
    for(int i=a.used-1;i>=0;--i) if(a.limb[i]!=b.limb[i]) return a.limb[i]<b.limb[i]?-1:1;
    return 0;
}
template<int C> IPC_GPU_EXACT_HD inline int compare(const Integer<C>& a,const Integer<C>& b) {
    if(a.sign!=b.sign) return a.sign<b.sign?-1:1;
    return a.sign*compare_magnitude(a,b);
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> negate(const Integer<C>& a) {
    Integer<C> r=a; r.sign=-r.sign; return r;
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> add(const Integer<C>& a,const Integer<C>& b) {
    if(!a.valid||!b.valid) return invalid_integer<C>();
    if(!a.sign) return b;
    if(!b.sign) return a;
    Integer<C> r;
    if(!r.valid)return r;
    if(a.sign==b.sign) {
        r.used=a.used>b.used?a.used:b.used; r.sign=a.sign;
        std::uint64_t carry=0;
        for(int i=0;i<r.used;++i) {
            const std::uint64_t value=(i<a.used?std::uint64_t(a.limb[i]):0)
                +(i<b.used?std::uint64_t(b.limb[i]):0)+carry;
            r.limb[i]=std::uint32_t(value); carry=value>>32;
        }
        if(carry) { if(r.used==C) return invalid_integer<C>(); r.limb[r.used++]=std::uint32_t(carry); }
    } else {
        const int order=compare_magnitude(a,b);
        if(!order) return r;
        const Integer<C>& large=order>0?a:b;
        const Integer<C>& small=order>0?b:a;
        r.used=large.used; r.sign=large.sign;
        std::uint64_t borrow=0;
        for(int i=0;i<r.used;++i) {
            const std::uint64_t lhs=large.limb[i];
            const std::uint64_t rhs=(i<small.used?std::uint64_t(small.limb[i]):0)+borrow;
            r.limb[i]=std::uint32_t(lhs-rhs); borrow=lhs<rhs;
        }
        r.normalize();
    }
    return r;
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> subtract(const Integer<C>& a,const Integer<C>& b) {
    return add(a,negate(b));
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> multiply(const Integer<C>& a,const Integer<C>& b) {
    if(!a.valid||!b.valid) return invalid_integer<C>();
    Integer<C> r;
    if(!r.valid)return r;
    if(!a.sign||!b.sign) return r;
    if(a.used+b.used-1>C) return invalid_integer<C>();
    r.used=a.used+b.used<C?a.used+b.used:C; r.sign=a.sign*b.sign;
    for(int i=0;i<r.used;++i) r.limb[i]=0;
    for(int i=0;i<a.used;++i) {
        std::uint64_t carry=0;
        for(int j=0;j<b.used;++j) {
            const int k=i+j;
            if(k>=C) return invalid_integer<C>();
            // (2^32-1)^2 + (2^32-1) + (2^32-1) = 2^64-1.
            const std::uint64_t value=std::uint64_t(a.limb[i])*b.limb[j]+r.limb[k]+carry;
            r.limb[k]=std::uint32_t(value); carry=value>>32;
        }
        if(carry) { const int k=i+b.used; if(k>=C) return invalid_integer<C>(); r.limb[k]=std::uint32_t(carry); }
    }
    r.normalize(); return r;
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> shift_left(const Integer<C>& a,int bits) {
    if(!a.valid||bits<0) return invalid_integer<C>();
    if(!a.sign) return a;
    const int words=bits/32,shift=bits%32;
    const bool top=shift && (a.limb[a.used-1]>>(32-shift));
    if(words>C||a.used+words+int(top)>C) return invalid_integer<C>();
    Integer<C> r; if(!r.valid)return r; r.used=a.used+words+int(top); r.sign=a.sign;
    for(int i=0;i<r.used;++i) r.limb[i]=0;
    for(int i=0;i<a.used;++i) {
        r.limb[i+words]|=a.limb[i]<<shift;
        if(shift && i+words+1<r.used) r.limb[i+words+1]|=a.limb[i]>>(32-shift);
    }
    return r;
}

template<int C> IPC_GPU_EXACT_HD inline int bit_count(const Integer<C>& a) {
    if(!a.used) return 0;
    std::uint32_t top=a.limb[a.used-1]; int bits=0;
    while(top) { top>>=1; ++bits; }
    return 32*(a.used-1)+bits;
}

// Correctly rounded binary64 rational conversion, including subnormal values,
// overflow and ties-to-even. Scaling is applied to the rational BEFORE any
// rounding, so a huge numerator/denominator ratio with a compensating exponent
// does not spuriously overflow. Needs at most 54 extra working bits; capacity
// exhaustion is explicit and must trigger a larger-device-capacity retry.
template<int C> IPC_GPU_EXACT_HD inline double ratio_to_double(
    const Integer<C>& numerator,const Integer<C>& denominator,int scale_exponent,
    bool* arithmetic_ok) {
    if(!numerator.valid||!denominator.valid||!denominator.sign) {
        if(arithmetic_ok) *arithmetic_ok=false; return NAN;
    }
    if(!numerator.sign) return 0.0;
    const int sign=numerator.sign*denominator.sign;
    Integer<C> n=numerator,d=denominator; n.sign=1; d.sign=1;
    int exponent=bit_count(n)-bit_count(d);
    Integer<C> shifted=exponent>=0?shift_left(d,exponent):shift_left(n,-exponent);
    if(!shifted.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    const int order=exponent>=0?compare_magnitude(n,shifted):compare_magnitude(shifted,d);
    if(order<0) --exponent;
    const long long final_exponent=static_cast<long long>(exponent)+scale_exponent;
    if(final_exponent>1023) return sign>0?INFINITY:-INFINITY;
    if(final_exponent < -1075) return sign>0?0.0:-0.0;
    const int output_exponent=final_exponent>=-1022?int(final_exponent)-52:-1074;
    const long long shift=static_cast<long long>(scale_exponent)-output_exponent;
    if(shift>32LL*C || shift < -32LL*C) {
        if(arithmetic_ok) *arithmetic_ok=false; return NAN;
    }
    if(shift>=0) n=shift_left(n,int(shift)); else d=shift_left(d,int(-shift));
    if(!n.valid||!d.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    std::uint64_t quotient=0;
    const int quotient_bits=bit_count(n)-bit_count(d);
    if(quotient_bits>=64) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    if(quotient_bits>=0) {
        // Materialize the largest trial once. Each subsequent trial is an
        // exact one-bit right shift: d*2^bit is divisible by two while bit>0.
        // The largest shift has the strongest limb-capacity requirement, and
        // one trial occupies the same arena slot count as the original loop.
        // Keep its lifetime inside this block, before constructing `twice`.
        auto trial=shift_left(d,quotient_bits);
        if(!trial.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
        for(int bit=quotient_bits;bit>=0;--bit) {
            if(compare_magnitude(n,trial)>=0) {
                n=subtract(n,trial); quotient|=UINT64_C(1)<<bit;
            }
            if(bit>0) {
                std::uint32_t carry=0;
                for(int word=trial.used;word-- >0;) {
                    const std::uint32_t original=trial.limb[word];
                    trial.limb[word]=(original>>1)|carry;
                    carry=original<<31;
                }
                trial.normalize();
            }
        }
    }
    const auto twice=shift_left(n,1);
    if(!twice.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    const int halfway=compare_magnitude(twice,d);
    if(halfway>0 || (halfway==0 && (quotient&1))) ++quotient;
    const double magnitude=::ldexp(static_cast<double>(quotient),output_exponent);
    return sign>0?magnitude:-magnitude;
}

struct Dyadic { std::uint64_t mantissa; int exponent; int sign; bool valid; };
IPC_GPU_EXACT_HD inline Dyadic decompose(double x) {
    std::uint64_t bits;
#if defined(__CUDA_ARCH__)
    bits=static_cast<std::uint64_t>(__double_as_longlong(x));
#else
    std::memcpy(&bits,&x,sizeof(bits));
#endif
    const unsigned exponent=unsigned((bits>>52)&2047);
    std::uint64_t mantissa=bits&UINT64_C(0xfffffffffffff);
    if(exponent==2047) return {0,0,0,false};
    if(exponent) mantissa|=UINT64_C(1)<<52;
    if(!mantissa) return {0,0,0,true};
    int unit=exponent?int(exponent)-1075:-1074;
    // Strip exact powers of two to minimize capacity without rounding.
    while(!(mantissa&1)) { mantissa>>=1; ++unit; }
    return {mantissa,unit,(bits>>63)?-1:1,true};
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> from_double(double x,int unit_exponent) {
    const Dyadic d=decompose(x);
    if(!d.valid|| (d.sign && d.exponent<unit_exponent)) return invalid_integer<C>();
    Integer<C> r(static_cast<std::int64_t>(d.mantissa));
    r.sign*=d.sign;
    return d.sign?shift_left(r,d.exponent-unit_exponent):r;
}

// Context propagates capacity failures through EVERY comparison and
// operation, even if an intermediate invalid value happens to look like zero.
template<int C> struct Arithmetic {
    using I=Integer<C>;
    bool ok=true;
    IPC_GPU_EXACT_HD I sum(const I&a,const I&b) { I r=add(a,b); ok=ok&&r.valid; return r; }
    IPC_GPU_EXACT_HD I sub(const I&a,const I&b) { I r=subtract(a,b); ok=ok&&r.valid; return r; }
    IPC_GPU_EXACT_HD I mul(const I&a,const I&b) { I r=multiply(a,b); ok=ok&&r.valid; return r; }
    IPC_GPU_EXACT_HD int cmp(const I&a,const I&b) { ok=ok&&a.valid&&b.valid; return compare(a,b); }
    IPC_GPU_EXACT_HD int sign(const I&a) { ok=ok&&a.valid; return a.sign; }
};

template<int C> IPC_GPU_EXACT_HD inline Status verdict(const Arithmetic<C>& arithmetic,Status status) {
    if(!arithmetic.ok)return Status::CapacityExceeded;
#if defined(__CUDA_ARCH__)
    const auto* pool=current_pool();
    if(!pool||pool->failed)return Status::CapacityExceeded;
#endif
    return status;
}

template<int C> struct Point { Integer<C> v[3]; };
template<int C> struct Points { Point<C> p[4]; };
template<int C> struct Distance { Integer<C> numerator,denominator; };

template<int C> IPC_GPU_EXACT_HD inline Point<C> difference(Arithmetic<C>& a,const Point<C>& x,const Point<C>& y) {
    Point<C> r; for(int i=0;i<3;++i) r.v[i]=a.sub(x.v[i],y.v[i]); return r;
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> dot(Arithmetic<C>& a,const Point<C>& x,const Point<C>& y) {
    return a.sum(a.sum(a.mul(x.v[0],y.v[0]),a.mul(x.v[1],y.v[1])),a.mul(x.v[2],y.v[2]));
}
template<int C> IPC_GPU_EXACT_HD inline Point<C> cross(Arithmetic<C>& a,const Point<C>& x,const Point<C>& y) {
    Point<C> r;
    for(int i=0;i<3;++i) { int j=(i+1)%3,k=(i+2)%3; r.v[i]=a.sub(a.mul(x.v[j],y.v[k]),a.mul(x.v[k],y.v[j])); }
    return r;
}
template<int C> IPC_GPU_EXACT_HD inline bool less(Arithmetic<C>& a,const Distance<C>& x,const Distance<C>& y) {
    return a.cmp(a.mul(x.numerator,y.denominator),a.mul(y.numerator,x.denominator))<0;
}
template<int C> IPC_GPU_EXACT_HD inline Distance<C> point_segment(Arithmetic<C>& a,
    const Point<C>& p,const Point<C>& x,const Point<C>& y) {
    const auto edge=difference(a,y,x),offset=difference(a,p,x);
    const auto length=dot(a,edge,edge),projection=dot(a,offset,edge);
    if(a.sign(length)==0||a.sign(projection)<=0) return {dot(a,offset,offset),Integer<C>(1)};
    if(a.cmp(projection,length)>=0) { const auto d=difference(a,p,y); return {dot(a,d,d),Integer<C>(1)}; }
    return {a.sub(a.mul(dot(a,offset,offset),length),a.mul(projection,projection)),length};
}
template<int C> IPC_GPU_EXACT_HD inline Distance<C> point_triangle(Arithmetic<C>& a,const Points<C>& x) {
    const auto e1=difference(a,x.p[2],x.p[1]),e2=difference(a,x.p[3],x.p[1]);
    const auto offset=difference(a,x.p[0],x.p[1]),normal=cross(a,e1,e2);
    const auto area=dot(a,normal,normal);
    if(a.sign(area)>0) {
        const auto e11=dot(a,e1,e1),e12=dot(a,e1,e2),e22=dot(a,e2,e2);
        const auto p1=dot(a,offset,e1),p2=dot(a,offset,e2);
        const auto u=a.sub(a.mul(e22,p1),a.mul(e12,p2));
        const auto v=a.sub(a.mul(e11,p2),a.mul(e12,p1));
        if(a.sign(u)>=0&&a.sign(v)>=0&&a.cmp(a.sum(u,v),area)<=0) {
            const auto h=dot(a,offset,normal); return {a.mul(h,h),area};
        }
    }
    auto best=point_segment(a,x.p[0],x.p[1],x.p[2]);
    auto candidate=point_segment(a,x.p[0],x.p[2],x.p[3]); if(less(a,candidate,best)) best=candidate;
    candidate=point_segment(a,x.p[0],x.p[3],x.p[1]); if(less(a,candidate,best)) best=candidate;
    return best;
}
template<int C> IPC_GPU_EXACT_HD inline Distance<C> edge_edge(Arithmetic<C>& a,const Points<C>& x) {
    const auto u=difference(a,x.p[1],x.p[0]),v=difference(a,x.p[3],x.p[2]);
    const auto offset=difference(a,x.p[0],x.p[2]);
    const auto uu=dot(a,u,u),uv=dot(a,u,v),vv=dot(a,v,v);
    const auto denominator=a.sub(a.mul(uu,vv),a.mul(uv,uv));
    if(a.sign(denominator)>0) {
        const auto ur=dot(a,u,offset),vr=dot(a,v,offset);
        const auto alpha=a.sub(a.mul(uv,vr),a.mul(vv,ur));
        const auto beta=a.sub(a.mul(uu,vr),a.mul(uv,ur));
        if(a.sign(alpha)>=0&&a.cmp(alpha,denominator)<=0&&a.sign(beta)>=0&&a.cmp(beta,denominator)<=0) {
            const auto h=dot(a,offset,cross(a,u,v)); return {a.mul(h,h),denominator};
        }
    }
    auto best=point_segment(a,x.p[0],x.p[2],x.p[3]);
    auto candidate=point_segment(a,x.p[1],x.p[2],x.p[3]); if(less(a,candidate,best)) best=candidate;
    candidate=point_segment(a,x.p[2],x.p[0],x.p[1]); if(less(a,candidate,best)) best=candidate;
    candidate=point_segment(a,x.p[3],x.p[0],x.p[1]); if(less(a,candidate,best)) best=candidate;
    return best;
}
template<int C> IPC_GPU_EXACT_HD inline Distance<C> distance(Arithmetic<C>& a,const Points<C>& x,bool face) {
    return face?point_triangle(a,x):edge_edge(a,x);
}
template<int C> IPC_GPU_EXACT_HD inline Integer<C> determinant(Arithmetic<C>& a,const Points<C>& x) {
    return dot(a,cross(a,difference(a,x.p[1],x.p[0]),difference(a,x.p[2],x.p[0])),difference(a,x.p[3],x.p[0]));
}

// Test one affine event t=-value/velocity without division. This is the same
// finite-primitive event membership as ccd.cpp's rational resolver.
template<int C> IPC_GPU_EXACT_HD inline bool event_hit(Arithmetic<C>& a,
    const Points<C>& start,const Points<C>& motion,bool face,bool exact_membership,
    const Integer<C>& validation_squared,Integer<C> value,Integer<C> velocity) {
    if(a.sign(velocity)==0) return false;
    Integer<C> numerator=negate(value),denominator=velocity;
    if(a.sign(denominator)<0) { numerator=negate(numerator); denominator=negate(denominator); }
    if(a.sign(numerator)<=0||a.cmp(numerator,denominator)>0) return false;
    Points<C> event;
    for(int i=0;i<4;++i) for(int k=0;k<3;++k)
        event.p[i].v[k]=a.sum(a.mul(denominator,start.p[i].v[k]),a.mul(numerator,motion.p[i].v[k]));
    const auto d=distance(a,event,face);
    if(exact_membership) return a.sign(d.numerator)==0;
    const auto threshold=a.mul(a.mul(a.mul(d.denominator,denominator),denominator),validation_squared);
    return a.cmp(d.numerator,threshold)<=0;
}

// Exact parity target: ::exact_contact_step_result in ccd.cpp, for at most
// one moved vertex. No TOI is approximated: this is a proposed-step verdict.
// Capacity 2048 (65,536 bits) bounds every finite-binary64 input supported here:
// Let B=2098 bound a represented coordinate or positive threshold in common
// units (max finite divided by min subnormal). Coordinate differences need
// B+1 bits; 3x3 determinants need 3B+5; their difference/root denominator needs
// 3B+6. Event H=Q*x+N*dx needs 4B+7 and event differences D=4B+8=8400 bits.
// A dot product of event differences needs 2D+2; cross needs 2D+1; plane height
// needs 3D+3. Thus interior distance numerator needs 6D+6, denominator 4D+4.
// Boundary point/segment distance is num<=4D+5,den<=2D+2, and boundary-distance
// comparisons need <=6D+7=50,407 bits. Interior solutions return immediately:
// they are NOT cross-multiplied against boundary distances. Event threshold
// multiplication needs <=(4D+4)+(6B+12)+2B=50,400 bits. Original start/end
// distance comparisons need only <=10(B+1)+10 bits. Coplanar event roots have
// degree <=2 instead of3, so their bounds are smaller. 65,536 bits leaves
// substantial explicit headroom; full-range resource exhaustion is separate.
// Smaller capacities may be retried on
// GPU when CapacityExceeded is returned, never treated as a safe verdict.
template<int C> IPC_GPU_EXACT_HD inline Status validate_step(
    const double start[4][3],const double end[4][3],bool face,double floor,
    bool allow_primitive_translation=false,bool input_is_motion=false) {
    using I=Integer<C>;
    const auto floor_d=decompose(floor),validation_d=decompose(1e-10);
    if(!floor_d.valid||floor_d.sign<=0) return Status::Invalid;
    int unit=floor_d.exponent<validation_d.exponent?floor_d.exponent:validation_d.exponent;
    int changed=0;
    for(int i=0;i<4;++i) {
        bool moved=false;
        for(int k=0;k<3;++k) {
            const auto s=decompose(start[i][k]),e=decompose(end[i][k]);
            if(!s.valid||!e.valid) return Status::Invalid;
            if(s.sign&&s.exponent<unit) unit=s.exponent;
            if(e.sign&&e.exponent<unit) unit=e.exponent;
            moved=moved||(input_is_motion?end[i][k]!=0.0:start[i][k]!=end[i][k]);
        }
        changed+=moved;
    }
    if(changed>1&&!allow_primitive_translation) return Status::Invalid;
    Arithmetic<C> a;
    Points<C> x,y,motion;
    for(int i=0;i<4;++i) for(int k=0;k<3;++k) {
        x.p[i].v[k]=from_double<C>(start[i][k],unit);
        if(input_is_motion) {
            motion.p[i].v[k]=from_double<C>(end[i][k],unit);
            y.p[i].v[k]=a.sum(x.p[i].v[k],motion.p[i].v[k]);
        } else {
            y.p[i].v[k]=from_double<C>(end[i][k],unit);
            motion.p[i].v[k]=a.sub(y.p[i].v[k],x.p[i].v[k]);
        }
    }
    const auto floor_i=from_double<C>(floor,unit),validation_i=from_double<C>(1e-10,unit);
    const auto floor2=a.mul(floor_i,floor_i),validation2=a.mul(validation_i,validation_i);
    if(!a.ok) return Status::CapacityExceeded;
    if(changed>1) {
        // Exact represented-endpoint translations, not approximately equal
        // double subtractions. These and one-vertex motion are precisely the
        // motions for which the determinant/incidence events remain affine.
        bool fixed[4]={true,true,true,true};
        for(int i=0;i<4;++i) for(int k=0;k<3;++k)
            fixed[i]=fixed[i]&&a.sign(motion.p[i].v[k])==0;
        bool same01=true,same23=true,same12=true;
        for(int k=0;k<3;++k) {
            same01=same01&&a.cmp(motion.p[0].v[k],motion.p[1].v[k])==0;
            same23=same23&&a.cmp(motion.p[2].v[k],motion.p[3].v[k])==0;
            same12=same12&&a.cmp(motion.p[1].v[k],motion.p[2].v[k])==0;
        }
        const bool permitted=face?(fixed[0]&&same12&&same23)
            :((fixed[2]&&fixed[3]&&same01)||(fixed[0]&&fixed[1]&&same23));
        if(!a.ok) return Status::CapacityExceeded;
        if(!permitted) return Status::Invalid;
    }
    const auto start2=distance(a,x,face);
    if(!a.ok) return Status::CapacityExceeded;
    if(a.sign(start2.numerator)==0) return verdict(a,Status::InitialContact);
    if(!changed) return verdict(a,Status::Safe);
    const auto end2=distance(a,y,face);
    const Distance<C> floor_distance{floor2,I(1)};
    const auto minimum=less(a,start2,floor_distance)?start2:floor_distance;
    const bool insufficient_gap=less(a,end2,minimum);
    if(!a.ok) return Status::CapacityExceeded;
    if(insufficient_gap) return verdict(a,Status::Unsafe);
    const bool exact_membership=a.cmp(start2.numerator,a.mul(validation2,start2.denominator))<=0;
    const auto intercept=determinant(a,x),slope=a.sub(determinant(a,y),intercept);
    if(!a.ok) return Status::CapacityExceeded;
    if(a.sign(slope)!=0) {
        const bool hit=event_hit(a,x,motion,face,exact_membership,validation2,intercept,slope);
        return verdict(a,hit?Status::Unsafe:Status::Safe);
    }
    if(a.sign(intercept)!=0) return verdict(a,Status::Safe);

    // Coplanar event enumeration. CPU sorts and deduplicates roots because it
    // returns the earliest TOI. A boolean step verdict only needs any hit, so
    // event order and duplicates do not affect semantics or exact arithmetic.
    if(event_hit(a,x,motion,face,exact_membership,validation2,I(-1),I(1)))
        return verdict(a,Status::Unsafe);
    for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) {
        const auto edge=difference(a,x.p[j],x.p[i]);
        const auto velocity=difference(a,motion.p[j],motion.p[i]);
        for(int axis=0;axis<3;++axis) {
            if(event_hit(a,x,motion,face,exact_membership,validation2,edge.v[axis],velocity.v[axis]))
                return verdict(a,Status::Unsafe);
            if(!a.ok) return Status::CapacityExceeded;
        }
        for(int k=j+1;k<4;++k) {
            const auto other=difference(a,x.p[k],x.p[i]);
            const auto other_velocity=difference(a,motion.p[k],motion.p[i]);
            const auto value=cross(a,edge,other),first=cross(a,velocity,other),second=cross(a,edge,other_velocity);
            for(int axis=0;axis<3;++axis) {
                const auto derivative=a.sum(first.v[axis],second.v[axis]);
                if(event_hit(a,x,motion,face,exact_membership,validation2,value.v[axis],derivative))
                    return verdict(a,Status::Unsafe);
                if(!a.ok) return Status::CapacityExceeded;
            }
        }
    }
    return verdict(a,Status::Safe);
}

// Rigid translation uses the production CPU contract: each moved proxy follows
// x(t)=x(0)+t*delta for one COMMON binary64 delta. Convert before addition so
// the ideal endpoints need not be representable. Final stored endpoints must
// independently pass validate_endpoint_gap; this does not claim that each
// independently rounded endpoint defines an identical affine trajectory.
// Adding two represented inputs increases B by at most one bit in the bound
// above; 65,536-bit capacity still leaves >15,000 bits of headroom.
template<int C> IPC_GPU_EXACT_HD inline Status validate_motion(
    const double start[4][3],const double motion[4][3],bool face,double floor) {
    return validate_step<C>(start,motion,face,floor,true,true);
}

// Endpoint policy only, for arbitrary represented endpoint changes. This is
// NOT a path or CCD certificate. It is paired with validate_motion for rigid
// translation so rounding cannot authorize a stored zero/insufficient gap.
template<int C> IPC_GPU_EXACT_HD inline Status validate_endpoint_gap(
    const double start[4][3],const double end[4][3],bool face,double floor) {
    const auto floor_d=decompose(floor);
    if(!floor_d.valid||floor_d.sign<=0)return Status::Invalid;
    int unit=floor_d.exponent;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const auto s=decompose(start[i][k]),e=decompose(end[i][k]);
        if(!s.valid||!e.valid)return Status::Invalid;
        if(s.sign&&s.exponent<unit)unit=s.exponent;
        if(e.sign&&e.exponent<unit)unit=e.exponent;
    }
    Arithmetic<C> a;Points<C> x,y;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        x.p[i].v[k]=from_double<C>(start[i][k],unit);
        y.p[i].v[k]=from_double<C>(end[i][k],unit);
    }
    const auto f=from_double<C>(floor,unit);
    const Distance<C> f2{a.mul(f,f),Integer<C>(1)};
    const auto start2=distance(a,x,face),end2=distance(a,y,face);
    if(a.sign(start2.numerator)==0)return verdict(a,Status::InitialContact);
    const auto minimum=less(a,start2,f2)?start2:f2;
    return verdict(a,less(a,end2,minimum)?Status::Unsafe:Status::Safe);
}

// Collision-only represented-path certificate for the ORIGINAL
// (exact_computation_fallback == false) GPU commit. Unlike validate_step this
// contract has NO positive separation floor: membership is exact ZERO
// distance, and an unchanged healthy positive-gap endpoint is Safe. A true
// finite-primitive initial intersection (including the all-zero primitive)
// returns InitialContact so the caller can use the explicit initial-contact
// device error instead of treating it as a backtrackable geometric miss.
//
// The ORIGINAL CPU TOI uses a fixed 1e-10 event-membership band and may return
// t=0 for a strictly positive gap. That is deliberately NOT reproduced here:
// this is the additional represented-path physical check, not the CPU TOI.
//
// Supported motion (the approved collision-only contract): AT MOST ONE moving
// vertex (vertex_face or edge_edge). Two moving vertices such as a translating
// edge are rejected as Invalid: rounded double end-start equality does not
// prove equal exact represented motions, so that motion is not certified here.
// Any other multi-vertex motion, degenerate event, or capacity failure is
// reported as Invalid/CapacityExceeded so the caller fails closed. Integer
// units come from real nonzero coordinates/endpoints; no fake gap floor,
// DBL_MIN or denorm_min inflation is introduced. When every coordinate is
// zero the unit is the exact integer zero and the resulting zero distance is
// correctly reported as InitialContact.
template<int C> IPC_GPU_EXACT_HD inline Status validate_collision_step(
    const double start[4][3],const double end[4][3],bool face) {
    using I=Integer<C>;
    int unit=0;
    bool have_unit=false;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const auto s=decompose(start[i][k]),e=decompose(end[i][k]);
        if(!s.valid||!e.valid)return Status::Invalid;
        if(s.sign&&(!have_unit||s.exponent<unit)){unit=s.exponent;have_unit=true;}
        if(e.sign&&(!have_unit||e.exponent<unit)){unit=e.exponent;have_unit=true;}
    }
    // No nonzero coordinate at all: every primitive point is the exact integer
    // zero. Leave unit=0 rather than inventing a floor; the exact distance is
    // then zero and the initial-intersection verdict below is correct.
    (void)have_unit;

    int moved_count=0;
    bool moved[4]={false,false,false,false};
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(end[i][k]!=start[i][k])moved[i]=true;
    }
    for(int i=0;i<4;++i)moved_count+=moved[i]?1:0;
    if(moved_count>1)return Status::Invalid;

    Arithmetic<C> a;
    Points<C> x,y,motion;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        x.p[i].v[k]=from_double<C>(start[i][k],unit);
        y.p[i].v[k]=from_double<C>(end[i][k],unit);
        motion.p[i].v[k]=a.sub(y.p[i].v[k],x.p[i].v[k]);
    }
    if(!a.ok)return Status::CapacityExceeded;

    // True initial finite-primitive intersection.
    {
        const auto d0=distance(a,x,face);
        if(!a.ok)return Status::CapacityExceeded;
        const int n=a.sign(d0.numerator),q=a.sign(d0.denominator);
        if(!a.ok)return Status::CapacityExceeded;
        if(n==0)return verdict(a,Status::InitialContact);
        // A non-positive denominator is a degenerate/unknown primitive query;
        // never certify it.
        if(q<=0)return verdict(a,Status::Unsafe);
    }
    // Actual represented endpoint membership.
    {
        const auto d1=distance(a,y,face);
        if(!a.ok)return Status::CapacityExceeded;
        const int n=a.sign(d1.numerator),q=a.sign(d1.denominator);
        if(!a.ok)return Status::CapacityExceeded;
        if(n==0)return verdict(a,Status::Unsafe);
        if(q<=0)return verdict(a,Status::Unsafe);
    }

    bool unsafe=false;
    const auto consider=[&](I value,I velocity) {
        if(!a.ok)return;
        if(a.sign(velocity)==0)return;
        I numerator=negate(value),denominator=velocity;
        if(a.sign(denominator)<0) {
            numerator=negate(numerator);
            denominator=negate(denominator);
        }
        // Membership in (0,1]; t=0 was already handled as the initial query.
        if(a.sign(numerator)<=0||a.cmp(numerator,denominator)>0)return;
        Points<C> event;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            event.p[i].v[k]=a.sum(a.mul(denominator,x.p[i].v[k]),
                                  a.mul(numerator,motion.p[i].v[k]));
        if(!a.ok)return;
        const auto d=distance(a,event,face);
        if(!a.ok)return;
        const int n=a.sign(d.numerator),q=a.sign(d.denominator);
        if(!a.ok)return;
        if(n==0||q<=0)unsafe=true;
    };

    const auto det0=determinant(a,x);
    const auto det1=determinant(a,y);
    const auto slope=a.sub(det1,det0);
    if(!a.ok)return Status::CapacityExceeded;
    if(a.sign(slope)!=0) {
        // One affine coplanarity root. For the supported single-moving-vertex
        // motion every NT/EE zero-distance event has determinant zero, so this
        // is the only interior candidate.
        consider(det0,slope);
    } else if(a.sign(det0)==0) {
        // Identically coplanar: scan the original event policy roots.
        for(int i=0;i<4;++i)for(int j=i+1;j<4;++j) {
            const auto edge=difference(a,x.p[j],x.p[i]);
            const auto velocity=difference(a,motion.p[j],motion.p[i]);
            if(!a.ok)return Status::CapacityExceeded;
            for(int axis=0;axis<3;++axis)
                consider(edge.v[axis],velocity.v[axis]);
            for(int k=j+1;k<4;++k) {
                const auto other=difference(a,x.p[k],x.p[i]);
                const auto other_velocity=difference(a,motion.p[k],motion.p[i]);
                const auto value=cross(a,edge,other);
                const auto first=cross(a,velocity,other);
                const auto second=cross(a,edge,other_velocity);
                if(!a.ok)return Status::CapacityExceeded;
                for(int axis=0;axis<3;++axis)
                    consider(value.v[axis],a.sum(first.v[axis],second.v[axis]));
            }
        }
    }
    if(!a.ok)return Status::CapacityExceeded;
    return verdict(a,unsafe?Status::Unsafe:Status::Safe);
}

struct ClipResult {
    Status status=Status::Unsafe;
    double fraction=0.0;
    double endpoint[4][3]{};
    int checks=0;
    bool full_step=false;
};

// Exact predicate backtracking; only updates represented endpoints after a
// Safe verdict. This is not the CPU analytic TOI + 90%-TOI clipping schedule,
// so it can take a more conservative step while retaining its collision and
// endpoint-gap safety policy. No capacity failure is converted into freezing.
template<int C> IPC_GPU_EXACT_HD inline ClipResult clip_step(
    const double start[4][3],const double proposed[4][3],bool face,double floor,
    int max_attempts=64) {
    ClipResult result;
    for(int i=0;i<4;++i) for(int k=0;k<3;++k) result.endpoint[i][k]=start[i][k];
    if(max_attempts<1||max_attempts>64) { result.status=Status::Invalid;return result; }
    double trial[4][3],fraction=1.0;
    for(int attempt=0;attempt<max_attempts;++attempt) {
        for(int i=0;i<4;++i) for(int k=0;k<3;++k) {
            if(!attempt) trial[i][k]=proposed[i][k];
            else {
                const double displacement=proposed[i][k]-start[i][k];
                // Ordinary arithmetic is only proposing a represented point;
                // the exact endpoint/path check below certifies that point.
                trial[i][k]=displacement-displacement==0.0
                    ? ::fma(fraction,displacement,start[i][k])
                    : ::fma(fraction,proposed[i][k],(1.0-fraction)*start[i][k]);
            }
        }
        ++result.checks;
        const Status status=validate_step<C>(start,trial,face,floor);
        if(status==Status::Safe) {
            result.status=status;result.fraction=fraction;result.full_step=!attempt;
            for(int i=0;i<4;++i) for(int k=0;k<3;++k) result.endpoint[i][k]=trial[i][k];
            return result;
        }
        if(status!=Status::Unsafe) { result.status=status;return result; }
        fraction*=0.5;
    }
    ++result.checks;
    result.status=validate_step<C>(start,start,face,floor);
    // The endpoint already equals start. Safe here explicitly certifies a
    // zero step; InitialContact/invalid/resource failures remain failures.
    return result;
}

} // namespace exact
} // namespace ipc_gpu

#undef IPC_GPU_EXACT_HD

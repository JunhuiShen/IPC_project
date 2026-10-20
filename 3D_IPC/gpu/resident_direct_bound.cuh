#pragma once

// Private proposal only. A computed zero is INCONCLUSIVE, never a lock or
// endpoint/path acceptance. Caller retains the unchanged GPU fallback.
// The existing resident_exact provider must be independently hash-bound.
#include "resident_exact.cuh"

#if defined(__CUDACC__)
#define IPC_DIRECT_BOUND_HD __host__ __device__
#else
#define IPC_DIRECT_BOUND_HD
#endif

namespace ipc_gpu::resident_direct_bound {
namespace ex = ipc_gpu::resident_exact;

enum class BoundStatus : unsigned {
    Computed, InvalidInput, CapacityExceeded, InvalidResourceContext
};
struct BoundResult { BoundStatus status=BoundStatus::InvalidInput; double fraction=0.0; };
constexpr int full_range_limbs=2048;
constexpr int required_pool_slots=256;

// Admission checks only, not an allocation-extent oracle. Caller must install
// the original initialized thread-local pool and retain its ownership contract.
IPC_DIRECT_BOUND_HD inline BoundStatus resource_policy(const ex::DevicePool* pool,int limbs) {
    if(!pool||pool->failed||!pool->words||!pool->free_slots||pool->words_per_slot<=0||pool->slot_count<=0
        ||pool->available<0||pool->available>pool->slot_count||pool->peak<0||pool->peak>pool->slot_count
        ||pool->peak<pool->slot_count-pool->available)return BoundStatus::InvalidResourceContext;
    if(pool->words_per_slot<limbs||pool->available<required_pool_slots)return BoundStatus::CapacityExceeded;
    return BoundStatus::Computed;
}

template<int C> using I=ex::Integer<C>;
template<int C> using A=ex::Arithmetic<C>;
template<int C> using P=ex::Point<C>;

template<int C> IPC_DIRECT_BOUND_HD inline I<C> left(A<C>& a,const I<C>& x,int bits) {
    I<C> r=ex::shift_left(x,bits);a.ok=a.ok&&r.valid;return r;
}
template<int C> IPC_DIRECT_BOUND_HD inline I<C> right(A<C>& a,const I<C>& x,int bits) {
    if(!x.valid||bits<0) { a.ok=false;return I<C>(); }
    if(!x.sign)return x;
    const int words=bits/32,shift=bits%32;
    I<C> r;if(!r.valid) { a.ok=false;return r; }
    if(words>=x.used)return r;
    r.used=x.used-words;r.sign=x.sign;
    for(int k=0;k<r.used;++k) {
        const int source=k+words;
        r.limb[k]=x.limb[source]>>shift;
        if(shift&&source+1<x.used)r.limb[k]|=x.limb[source+1]<<(32-shift);
    }
    r.normalize();return r;
}
template<int C> IPC_DIRECT_BOUND_HD inline int trailing(const I<C>& x) {
    if(!x.sign)return 0;
    int bits=0,k=0;
    while(k<x.used&&x.limb[k]==0) { bits+=32;++k; }
    if(k==x.used)return bits;
    std::uint32_t word=x.limb[k];while(!(word&1)) { ++bits;word>>=1; }
    return bits;
}

// Binary Stein: every unequal odd-pair iteration reduces one positive
// operand's bit length. No general integer division or modulus is introduced.
template<int C> IPC_DIRECT_BOUND_HD inline I<C> gcd(A<C>& a,I<C> x,I<C> y) {
    if(!a.ok||!x.valid||!y.valid||x.sign<0||y.sign<0) { a.ok=false;return I<C>(); }
    if(!x.sign)return y;
    if(!y.sign)return x;
    const int tx=trailing(x),ty=trailing(y),common=tx<ty?tx:ty;
    x=right(a,x,tx);y=right(a,y,ty);
    for(int attempt=0;attempt<64*C;++attempt) {
        if(!a.ok)return I<C>();
        const int order=a.cmp(x,y);
        if(!order)return left(a,x,common);
        if(order>0) { x=a.sub(x,y);x=right(a,x,trailing(x)); }
        else { y=a.sub(y,x);y=right(a,y,trailing(y)); }
    }
    a.ok=false;return I<C>();
}
template<int C> IPC_DIRECT_BOUND_HD inline int quotient_msb(A<C>& a,const I<C>& n,const I<C>& g) {
    const int k=ex::bit_count(n)-ex::bit_count(g);
    if(k<0||a.sign(g)<=0) { a.ok=false;return 0; }
    const I<C> threshold=left(a,g,k);
    return a.cmp(n,threshold)<0?k-1:k;
}

// CPU sqrt rule: normalize N/D, e=msb(N/g)-msb(D/g), s=192-e/2
// (signed integer /2 truncates toward zero). Find ceil(sqrt(A/B)) by exact
// digit trials. e's definition implies A/B<2^386, so 193 trial bits followed
// by at most one increment suffice; no division or floating sqrt is used.
template<int C> IPC_DIRECT_BOUND_HD inline bool sqrt_upper(
    A<C>& a,const I<C>& numerator,const I<C>& denominator,I<C>& root,int& scale) {
    if(!a.ok||a.sign(numerator)<=0||a.sign(denominator)<=0)return false;
    const I<C> divisor=gcd(a,numerator,denominator);
    if(!a.ok)return false;
    const int exponent=quotient_msb(a,numerator,divisor)-quotient_msb(a,denominator,divisor);
    scale=192-exponent/2;
    I<C> n=numerator,d=denominator;
    if(scale>=0)n=left(a,n,2*scale);else d=left(a,d,-2*scale);
    root=I<C>();
    for(int bit=192;bit>=0;--bit) {
        const I<C> trial=a.sum(root,left(a,I<C>(1),bit));
        const I<C> squared=a.mul(trial,trial);
        if(a.cmp(a.mul(squared,d),n)<=0)root=trial;
        if(!a.ok)return false;
    }
    if(a.cmp(a.mul(a.mul(root,root),d),n)<0)root=a.sum(root,I<C>(1));
    return a.ok;
}

template<int C> IPC_DIRECT_BOUND_HD inline P<C> scale_point(A<C>& a,const P<C>& p,const I<C>& v) {
    P<C> r;for(int k=0;k<3;++k)r.v[k]=a.mul(p.v[k],v);return r;
}
template<int C> struct Residual { P<C> n;I<C> d{1}; };
template<int C> IPC_DIRECT_BOUND_HD inline Residual<C> segment_residual(
    A<C>& a,const P<C>& p,const P<C>& x,const P<C>& y) {
    const P<C> edge=ex::difference(a,y,x),offset=ex::difference(a,p,x);
    const I<C> length=ex::dot(a,edge,edge),projection=ex::dot(a,offset,edge);
    Residual<C> r;
    if(a.sign(length)==0||a.sign(projection)<=0) { r.n=offset;return r; }
    if(a.cmp(projection,length)>=0) { r.n=ex::difference(a,p,y);return r; }
    r.n=ex::difference(a,scale_point(a,offset,length),scale_point(a,edge,projection));r.d=length;
    return r;
}
template<int C> IPC_DIRECT_BOUND_HD inline bool shorter(A<C>& a,const Residual<C>& x,const Residual<C>& y) {
    return a.cmp(a.mul(ex::dot(a,x.n,x.n),a.mul(y.d,y.d)),
                 a.mul(ex::dot(a,y.n,y.n),a.mul(x.d,x.d)))<0;
}
template<int C> IPC_DIRECT_BOUND_HD inline Residual<C> closest(A<C>& a,const ex::Points<C>& p,bool face) {
    const P<C> edge0=ex::difference(a,p.p[1],p.p[0]);
    if(face) {
        const P<C> x=ex::difference(a,p.p[2],p.p[1]),y=ex::difference(a,p.p[3],p.p[1]);
        const P<C> offset=ex::difference(a,p.p[0],p.p[1]),normal=ex::cross(a,x,y);
        const I<C> area=ex::dot(a,normal,normal);
        if(a.sign(area)!=0) {
            const I<C> aa=ex::dot(a,x,x),ab=ex::dot(a,x,y),bb=ex::dot(a,y,y);
            const I<C> ar=ex::dot(a,offset,x),br=ex::dot(a,offset,y);
            const I<C> u=a.sub(a.mul(bb,ar),a.mul(ab,br)),v=a.sub(a.mul(aa,br),a.mul(ab,ar));
            if(a.sign(u)>=0&&a.sign(v)>=0&&a.cmp(a.sum(u,v),area)<=0) {
                Residual<C> r;r.n=scale_point(a,normal,ex::dot(a,offset,normal));r.d=area;return r;
            }
        }
        Residual<C> best=segment_residual(a,p.p[0],p.p[1],p.p[2]);
        for(int edge=0;edge<2;++edge) {
            const Residual<C> candidate=edge==0?segment_residual(a,p.p[0],p.p[2],p.p[3]):segment_residual(a,p.p[0],p.p[3],p.p[1]);
            if(shorter(a,candidate,best))best=candidate;
        }
        return best;
    }
    const P<C> edge1=ex::difference(a,p.p[3],p.p[2]),offset=ex::difference(a,p.p[0],p.p[2]);
    const P<C> normal=ex::cross(a,edge0,edge1);
    const I<C> aa=ex::dot(a,edge0,edge0),ab=ex::dot(a,edge0,edge1),bb=ex::dot(a,edge1,edge1);
    const I<C> denominator=a.sub(a.mul(aa,bb),a.mul(ab,ab));
    if(a.sign(denominator)!=0) {
        const I<C> ar=ex::dot(a,edge0,offset),br=ex::dot(a,edge1,offset);
        const I<C> alpha=a.sub(a.mul(ab,br),a.mul(bb,ar)),beta=a.sub(a.mul(aa,br),a.mul(ab,ar));
        if(a.sign(alpha)>=0&&a.cmp(alpha,denominator)<=0&&a.sign(beta)>=0&&a.cmp(beta,denominator)<=0) {
            Residual<C> r;r.n=scale_point(a,normal,ex::dot(a,offset,normal));r.d=denominator;return r;
        }
    }
    Residual<C> best=segment_residual(a,p.p[0],p.p[2],p.p[3]);
    for(int index=0;index<3;++index) {
        Residual<C> candidate=index==0?segment_residual(a,p.p[1],p.p[2],p.p[3]):segment_residual(a,p.p[index+1],p.p[0],p.p[1]);
        if(index)for(int k=0;k<3;++k)candidate.n.v[k]=ex::negate(candidate.n.v[k]);
        if(shorter(a,candidate,best))best=candidate;
    }
    return best;
}

IPC_DIRECT_BOUND_HD inline double from_positive_bits(std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    double value;std::memcpy(&value,&bits,sizeof(value));return value;
#endif
}
template<int C> IPC_DIRECT_BOUND_HD inline double down_ratio(A<C>& a,const I<C>& n,const I<C>& d) {
    if(a.sign(n)<=0)return 0.0;
    std::uint64_t low=0,high=UINT64_C(0x3ff0000000000000);
    for(int attempt=0;attempt<62&&low<high;++attempt) {
        const std::uint64_t middle=low+(high-low+1)/2;
        const auto binary=ex::decompose(from_positive_bits(middle));
        I<C> left_value=a.mul(d,I<C>(static_cast<std::int64_t>(binary.mantissa))),right_value=n;
        if(binary.exponent>=0)left_value=left(a,left_value,binary.exponent);
        else right_value=left(a,right_value,-binary.exponent);
        if(a.cmp(left_value,right_value)<=0)low=middle;else high=middle-1;
        if(!a.ok)return 0.0;
    }
    if(low!=high)a.ok=false;
    return from_positive_bits(low);
}
template<int C> IPC_DIRECT_BOUND_HD inline BoundResult finish(const A<C>& a,double fraction) {
    if(!a.ok)return {BoundStatus::CapacityExceeded,0.0};
#if defined(__CUDA_ARCH__)
    const auto* pool=ex::current_pool();
    if(!pool||pool->failed)return {BoundStatus::CapacityExceeded,0.0};
#endif
    return {BoundStatus::Computed,fraction};
}

template<int C> IPC_DIRECT_BOUND_HD inline BoundResult separation_step_bound(
    const double start[4][3],const double end[4][3],bool face,double floor) {
    const auto floor_value=ex::decompose(floor);
    if(!floor_value.valid||floor_value.sign<=0)return {};
    int unit=floor_value.exponent,moving=-1;
    for(int i=0;i<4;++i) {
        bool changed=false;
        for(int k=0;k<3;++k) {
            const auto x=ex::decompose(start[i][k]),y=ex::decompose(end[i][k]);
            if(!x.valid||!y.valid)return {};
            if(x.sign&&x.exponent<unit)unit=x.exponent;
            if(y.sign&&y.exponent<unit)unit=y.exponent;
            changed=changed||start[i][k]!=end[i][k];
        }
        if(changed) { if(moving>=0)return {};moving=i; }
    }
#if defined(__CUDA_ARCH__)
    const auto* pool=ex::current_pool();
    const auto resource=resource_policy(pool,C);
    if(resource!=BoundStatus::Computed)return {resource,0.0};
#endif
    A<C> a;ex::Points<C> x;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) { x.p[i].v[k]=ex::from_double<C>(start[i][k],unit);a.ok=a.ok&&x.p[i].v[k].valid; }
    const Residual<C> normal=closest(a,x,face);
    const I<C> norm2=ex::dot(a,normal.n,normal.n),den2=a.mul(normal.d,normal.d);
    if(!a.ok)return finish(a,0.0);
    if(a.sign(norm2)==0)return finish(a,0.0);
    if(moving<0)return finish(a,1.0);
    const I<C> f=ex::from_double<C>(floor,unit);a.ok=a.ok&&f.valid;
    const I<C> floor2=a.mul(f,f);
    const bool below_floor=a.cmp(norm2,a.mul(floor2,den2))<=0;
    I<C> root;int sqrt_scale=0;
    if(!below_floor) {
        I<C> qn=a.mul(floor2,norm2),qd=den2;
        if(unit>=0)qn=left(a,qn,4*unit);else qd=left(a,qd,-4*unit);
        if(!sqrt_upper(a,qn,qd,root,sqrt_scale)) { a.ok=false;return finish(a,0.0); }
    }
    P<C> displacement;
    for(int k=0;k<3;++k) { const I<C> e=ex::from_double<C>(end[moving][k],unit);a.ok=a.ok&&e.valid;displacement.v[k]=a.sub(e,x.p[moving].v[k]); }
    const I<C> motion=ex::dot(a,displacement,normal.n);
    I<C> fraction_n(1),fraction_d(1);
    const int split=face?1:2;
    for(int i=0;i<split;++i)for(int j=split;j<4;++j) {
        I<C> projection=ex::dot(a,ex::difference(a,x.p[i],x.p[j]),normal.n);
        I<C> slope=i==moving?motion:j==moving?ex::negate(motion):I<C>();
        I<C> required;
        if(below_floor) { projection=a.mul(projection,normal.d);slope=a.mul(slope,normal.d);required=norm2; }
        else {
            const int power=2*unit+sqrt_scale;required=a.mul(root,normal.d);
            if(power>=0) { projection=left(a,projection,power);slope=left(a,slope,power); }
            else required=left(a,required,-power);
        }
        if(!a.ok)return finish(a,0.0);
        if(a.cmp(projection,required)<0)return finish(a,0.0);
        if(a.sign(slope)<0) {
            const I<C> n=a.sub(projection,required),d=ex::negate(slope);
            if(a.cmp(a.mul(n,fraction_d),a.mul(fraction_n,d))<0) { fraction_n=n;fraction_d=d; }
        }
    }
    const double fraction=down_ratio(a,fraction_n,fraction_d);
    return finish(a,fraction);
}
} // namespace ipc_gpu::resident_direct_bound

#undef IPC_DIRECT_BOUND_HD

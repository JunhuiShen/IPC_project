#pragma once

#include "dat_types.h"
#include "contact_exact.cuh"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define IPC_DAT_HD __host__ __device__
#else
#define IPC_DAT_HD
#endif

namespace ipc_gpu::solver::dat_detail {

// Geometry-only helpers shared by the device filter and differential tests.
// NeedsExact is internal queue control, never an inactive-contact verdict.
constexpr int NeedsExact = 1;
struct Vec { double v[3]{}; };
IPC_DAT_HD inline bool finite(double x) { return x <= DBL_MAX && x >= -DBL_MAX; }
IPC_DAT_HD inline bool finite(Vec x) {
    return finite(x.v[0]) && finite(x.v[1]) && finite(x.v[2]);
}
IPC_DAT_HD inline double maximum(double a,double b) { return a < b ? b : a; }
IPC_DAT_HD inline double minimum(double a,double b) { return a < b ? a : b; }
IPC_DAT_HD inline Vec load(const double* x) { return {{x[0],x[1],x[2]}}; }
IPC_DAT_HD inline Vec add(Vec a,Vec b) {
    return {{a.v[0]+b.v[0],a.v[1]+b.v[1],a.v[2]+b.v[2]}};
}
IPC_DAT_HD inline Vec subtract(Vec a,Vec b) {
    return {{a.v[0]-b.v[0],a.v[1]-b.v[1],a.v[2]-b.v[2]}};
}
IPC_DAT_HD inline Vec multiply(Vec a,double scale) {
    return {{a.v[0]*scale,a.v[1]*scale,a.v[2]*scale}};
}
IPC_DAT_HD inline Vec divide(Vec a,double scale) {
    return {{a.v[0]/scale,a.v[1]/scale,a.v[2]/scale}};
}
IPC_DAT_HD inline double dot(Vec a,Vec b) {
    return (a.v[0]*b.v[0]+a.v[1]*b.v[1])+a.v[2]*b.v[2];
}
IPC_DAT_HD inline Vec cross(Vec a,Vec b) {
    return {{a.v[1]*b.v[2]-a.v[2]*b.v[1],a.v[2]*b.v[0]-a.v[0]*b.v[2],
             a.v[0]*b.v[1]-a.v[1]*b.v[0]}};
}
IPC_DAT_HD inline double largest(Vec a) {
    return maximum(maximum(::fabs(a.v[0]),::fabs(a.v[1])),::fabs(a.v[2]));
}
IPC_DAT_HD inline double norm(Vec a) { return ::sqrt(dot(a,a)); }
IPC_DAT_HD inline double stable_norm(Vec a) {
    const double scale=largest(a);
    return scale==0 ? 0 : scale*norm(divide(a,scale));
}
IPC_DAT_HD inline double clamp(double x) { return x<0 ? 0 : x>1 ? 1 : x; }
IPC_DAT_HD inline Vec capped_update(Vec update,double limit) {
    const double scale=largest(update);
    if(scale==0)return update;
    const Vec scaled=divide(update,scale);
    const double length=norm(scaled);
    if(scale<=limit/length)return update;
    return multiply(scaled,limit/length);
}
IPC_DAT_HD inline Vec segment_closest(Vec point,Vec a,Vec b) {
    const Vec direction=subtract(b,a);
    const double length=dot(direction,direction);
    const double t=length<=0 ? 0 : clamp(dot(subtract(point,a),direction)/length);
    return add(a,multiply(direction,t));
}

// Match DAT.cpp's scale-normalized finite-triangle solve, including its
// all-three-edge search for obtuse/exterior and degenerate triangles.
IPC_DAT_HD inline int vertex_triangle(const Vec x[4],Vec& separation) {
    Vec local[3];double scale=0;
    for(int i=0;i<3;++i) {
        local[i]=subtract(x[i+1],x[0]);
        if(!finite(local[i]))return DatUnresolvedGeometry;
        scale=maximum(scale,largest(local[i]));
    }
    if(scale==0){separation={};return 0;}
    for(auto& p:local)p=divide(p,scale);
    const Vec product=cross(subtract(local[1],local[0]),subtract(local[2],local[0]));
    const double length=stable_norm(product);
    if(length>0) {
        const Vec normal=divide(product,length);
        const Vec projection=multiply(normal,dot(local[0],normal));
        bool interior=true;
        for(int edge=0;edge<3;++edge) {
            const Vec direction=subtract(local[(edge+1)%3],local[edge]);
            if(dot(cross(direction,subtract(projection,local[edge])),normal)<0)interior=false;
        }
        if(interior){separation=multiply(projection,scale);return 0;}
    }
    double best=INFINITY;Vec closest{};
    for(int edge=0;edge<3;++edge) {
        const Vec candidate=segment_closest({},local[edge],local[(edge+1)%3]);
        const double distance=stable_norm(candidate);
        if(distance<best){best=distance;closest=candidate;}
    }
    if(!finite(best))return DatUnresolvedGeometry;
    separation=multiply(closest,scale);return 0;
}

// CPU segment_segment_distance(...,default exact=true) uses its historical
// floating solve only outside the following uncertainty/gap gates. Do not
// replace world-space residuals in that accepted ordinary path with a subtly
// different local expression: the CPU's exact retry handles unresolved gaps.
IPC_DAT_HD inline int edge_edge(const Vec x[4],Vec& separation) {
    for(int i=0;i<4;++i)if(!finite(x[i]))return DatInvalidInput;
    const Vec a=subtract(x[1],x[0]),b=subtract(x[3],x[2]),c=subtract(x[0],x[2]);
    const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,c),E=dot(b,c);
    const double product=A*C,determinant=product-B*B;
    constexpr double roundoff=64*DBL_EPSILON;
    if(!finite(determinant)||determinant<=roundoff*product||determinant<=1e-24)return NeedsExact;
    double s=(B*E-C*D)/determinant,t=(A*E-B*D)/determinant;
    double distance=0;
    if(s>=0&&s<=1&&t>=0&&t<=1) {
        separation=subtract(add(x[0],multiply(a,s)),add(x[2],multiply(b,t)));
        distance=norm(separation);
    } else {
        double best=DBL_MAX,best_s=0,best_t=0;
        for(int candidate=0;candidate<4;++candidate) {
            double ss,tt;
            if(candidate<2) {
                ss=double(candidate);
                tt=clamp(dot(subtract(add(x[0],multiply(a,ss)),x[2]),b)/C);
            } else {
                tt=double(candidate-2);
                ss=clamp(dot(subtract(add(x[2],multiply(b,tt)),x[0]),a)/A);
            }
            const Vec residual=subtract(add(x[0],multiply(a,ss)),add(x[2],multiply(b,tt)));
            const double d=norm(residual);
            if(d<best){best=d;best_s=ss;best_t=tt;}
        }
        separation=subtract(add(x[0],multiply(a,best_s)),add(x[2],multiply(b,best_t)));
        distance=best;
    }
    double coordinate_scale=0;for(int i=0;i<4;++i)coordinate_scale=maximum(coordinate_scale,largest(x[i]));
    if(!finite(distance)||distance<=roundoff*coordinate_scale)return NeedsExact;
    // CPU DAT negates the segment solver's B-to-A residual.
    separation=multiply(separation,-1);return 0;
}

// Pool installation/lifetime is owned by the CUDA exact queue, as in the
// existing contact pipeline. On the host, Integer owns its bounded limbs.
template<int C> IPC_DAT_HD inline int exact_edge_edge(const Vec positions[4],Vec& separation) {
    exact::Arithmetic<C> arithmetic;
    int unit=0;bool have_unit=false;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const auto part=exact::decompose(positions[i].v[k]);
        if(!part.valid)return DatInvalidInput;
        if(part.sign&&(!have_unit||part.exponent<unit)){unit=part.exponent;have_unit=true;}
    }
    exact::Points<C> x;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        x.p[i].v[k]=exact::from_double<C>(positions[i].v[k],unit);
        arithmetic.ok=arithmetic.ok&&x.p[i].v[k].valid;
    }
    if(!arithmetic.ok)return DatExactCapacity;
    const auto feature=contact_exact_detail::edge_edge(arithmetic,x);
    if(!arithmetic.ok||!feature.have)return DatExactCapacity;
    for(int k=0;k<3;++k)
        separation.v[k]=-exact::ratio_to_double(feature.residual[k],feature.denominator,unit,&arithmetic.ok);
    return arithmetic.ok ? 0 : DatExactCapacity;
}

// Produce independent factors; the device caller atomically takes each
// vertex's minimum across pairs. Every primitive corner participates.
IPC_DAT_HD inline int plane_factors(const Vec positions[4],const Vec updates[4],
    int side_size,Vec separation,const DatParameters& p,double factors[4]) {
    for(int i=0;i<4;++i)factors[i]=1;
    const double distance=stable_norm(separation);
    if(!finite(separation)||!finite(distance))return DatUnresolvedGeometry;
    if(distance>=p.query_radius)return 0;
    if(!(distance>0))return DatZeroGap;
    const Vec normal=divide(separation,distance);
    double coordinate[4],velocity[4],left=-INFINITY,right=INFINITY,approach_a=0,approach_b=0;
    for(int i=0;i<4;++i) {
        coordinate[i]=dot(subtract(positions[i],positions[0]),normal);
        velocity[i]=dot(updates[i],normal);
        if(!finite(coordinate[i])||!finite(velocity[i]))return DatUnresolvedGeometry;
        if(i<side_size){left=maximum(left,coordinate[i]);approach_a=maximum(approach_a,velocity[i]);}
        else {right=minimum(right,coordinate[i]);approach_b=maximum(approach_b,-velocity[i]);}
    }
    const double gap=right-left;
    if(!(gap>0)||!finite(gap))return DatUnresolvedGeometry;
    const double scale=maximum(approach_a,approach_b);
    const double a=scale>0?approach_a/scale:1,b=scale>0?approach_b/scale:1;
    const double budget_a=gap*(a/(a+b)),budget_b=gap*(b/(a+b));
    for(int i=0;i<4;++i) {
        const bool side_a=i<side_size;
        const double approach=side_a?velocity[i]:-velocity[i];
        if(approach<=0)continue;
        const double margin=side_a?(left-coordinate[i])+budget_a:(coordinate[i]-right)+budget_b;
        const double allowed=p.gamma*margin;
        if(approach>allowed)factors[i]=clamp(allowed/approach);
    }
    return 0;
}

} // namespace ipc_gpu::solver::dat_detail

#undef IPC_DAT_HD

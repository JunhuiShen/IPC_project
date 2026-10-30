#pragma once

// Cold-path device contact geometry and full Cartesian derivatives, using
// factored exact integer polynomials. No host geometry/Boost/CPU fallback.
// All original coordinate bits are converted BEFORE subtraction.
// Capacity failure is explicit and requires a larger GPU workspace retry.
#include "contact.cuh"
#include "exact.cuh"

#if defined(__CUDACC__)
#define IPC_RES_CONTACT_EXACT_HD __host__ __device__
#else
#define IPC_RES_CONTACT_EXACT_HD
#endif

namespace ipc_gpu {
namespace solver {
namespace contact_exact_detail {

template<int C> struct Feature {
    using I=exact::Integer<C>;
    I denominator{1},weight[4],residual[3],squared_residual;
    exact::Point<C> tangent[2];
    int sigma[2][4]{};
    int free_parameters=0;
    bool have=false;
};

template<int C> IPC_RES_CONTACT_EXACT_HD inline exact::Integer<C> clamp(
    exact::Arithmetic<C>& a,const exact::Integer<C>& n,
    const exact::Integer<C>& d) {
    if(a.sign(n)<=0)return exact::Integer<C>(0);
    if(a.cmp(n,d)>=0)return d;
    return n;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline void finish_feature(
    exact::Arithmetic<C>& a,const exact::Points<C>& x,Feature<C>& f) {
    using I=exact::Integer<C>;
    f.squared_residual=I(0);
    for(int k=0;k<3;++k) {
        f.residual[k]=I(0);
        for(int i=0;i<4;++i)f.residual[k]=a.sum(f.residual[k],a.mul(f.weight[i],x.p[i].v[k]));
        f.squared_residual=a.sum(f.squared_residual,a.mul(f.residual[k],f.residual[k]));
    }
    f.have=a.ok;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline void consider(
    exact::Arithmetic<C>& a,const Feature<C>& candidate,Feature<C>& best) {
    if(!a.ok||!candidate.have)return;
    if(!best.have) {best=candidate;return;}
    const auto lhs=a.mul(candidate.squared_residual,a.mul(best.denominator,best.denominator));
    const auto rhs=a.mul(best.squared_residual,a.mul(candidate.denominator,candidate.denominator));
    const int order=a.cmp(lhs,rhs);
    if(a.ok&&order<0)best=candidate;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline Feature<C> point_edge(
    exact::Arithmetic<C>& a,const exact::Points<C>& x,int p,int v0,int v1) {
    using I=exact::Integer<C>;
    Feature<C> f;
    const auto e=exact::difference(a,x.p[v1],x.p[v0]);
    const auto r=exact::difference(a,x.p[p],x.p[v0]);
    const auto length=exact::dot(a,e,e),projection=exact::dot(a,e,r);
    I t;
    if(a.sign(length)>0) {f.denominator=length;t=clamp(a,projection,length);}
    f.weight[p]=f.denominator;
    f.weight[v0]=a.sub(t,f.denominator);f.weight[v1]=exact::negate(t);
    if(a.sign(t)>0&&a.cmp(t,f.denominator)<0) {
        f.free_parameters=1;f.sigma[0][v0]=1;f.sigma[0][v1]=-1;
        for(int k=0;k<3;++k)f.tangent[0].v[k]=exact::negate(e.v[k]);
    }
    finish_feature(a,x,f);return f;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline Feature<C> point_triangle(
    exact::Arithmetic<C>& a,const exact::Points<C>& x,
    int contact_feature=0) {
    Feature<C> f;
    if(contact_feature>0) {
        f.weight[0]=exact::Integer<C>(1);
        f.weight[contact_feature]=exact::Integer<C>(-1);
        finish_feature(a,x,f);return f;
    }
    if(contact_feature<0) {
        const int first=-contact_feature,second=first==3?1:first+1;
        return point_edge(a,x,0,first,second);
    }
    const auto e1=exact::difference(a,x.p[2],x.p[1]);
    const auto e2=exact::difference(a,x.p[3],x.p[1]);
    const auto r=exact::difference(a,x.p[0],x.p[1]);
    const auto A=exact::dot(a,e1,e1),B=exact::dot(a,e1,e2),C2=exact::dot(a,e2,e2);
    const auto D=exact::dot(a,r,e1),E=exact::dot(a,r,e2);
    const auto det=a.sub(a.mul(A,C2),a.mul(B,B));
    if(a.sign(det)>0) {
        const auto s=a.sub(a.mul(D,C2),a.mul(E,B));
        const auto t=a.sub(a.mul(E,A),a.mul(D,B));
        if(a.sign(s)>=0&&a.sign(t)>=0&&a.cmp(a.sum(s,t),det)<=0) {
            f.denominator=det;f.weight[0]=det;
            f.weight[1]=a.sub(a.sum(s,t),det);
            f.weight[2]=exact::negate(s);f.weight[3]=exact::negate(t);
            f.free_parameters=2;f.sigma[0][1]=1;f.sigma[0][2]=-1;f.sigma[1][1]=1;f.sigma[1][3]=-1;
            for(int k=0;k<3;++k){f.tangent[0].v[k]=exact::negate(e1.v[k]);f.tangent[1].v[k]=exact::negate(e2.v[k]);}
            finish_feature(a,x,f);return f;
        }
    }
    consider(a,point_edge(a,x,0,1,2),f);
    consider(a,point_edge(a,x,0,2,3),f);
    consider(a,point_edge(a,x,0,3,1),f);
    return f;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline Feature<C> edge_parameters(
    exact::Arithmetic<C>& a,const exact::Points<C>& x,
    const exact::Point<C>& e1,const exact::Point<C>& e2,
    const exact::Integer<C>& s,const exact::Integer<C>& t,const exact::Integer<C>& denominator) {
    Feature<C> f;
    f.denominator=denominator;
    f.weight[0]=a.sub(denominator,s);f.weight[1]=s;
    f.weight[2]=a.sub(t,denominator);f.weight[3]=exact::negate(t);
    // Strict default membership: generic DAT/CCD/geometry semantics.
    const bool free_s=a.sign(s)>0&&a.cmp(s,denominator)<0;
    const bool free_t=a.sign(t)>0&&a.cmp(t,denominator)<0;
    if(free_s) {
        const int j=f.free_parameters++;f.sigma[j][0]=-1;f.sigma[j][1]=1;f.tangent[j]=e1;
    }
    if(free_t) {
        const int j=f.free_parameters++;f.sigma[j][2]=1;f.sigma[j][3]=-1;
        for(int k=0;k<3;++k)f.tangent[j].v[k]=exact::negate(e2.v[k]);
    }
    finish_feature(a,x,f);return f;
}

template<int C> IPC_RES_CONTACT_EXACT_HD inline Feature<C> edge_edge(
    exact::Arithmetic<C>& a,const exact::Points<C>& x) {
    using I=exact::Integer<C>;
    Feature<C> best;
    const auto e1=exact::difference(a,x.p[1],x.p[0]),e2=exact::difference(a,x.p[3],x.p[2]);
    const auto r=exact::difference(a,x.p[0],x.p[2]);
    const auto A=exact::dot(a,e1,e1),B=exact::dot(a,e1,e2),C2=exact::dot(a,e2,e2);
    const auto D=exact::dot(a,e1,r),E=exact::dot(a,e2,r);
    const auto det=a.sub(a.mul(A,C2),a.mul(B,B));
    if(a.sign(det)>0) {
        const auto s=a.sub(a.mul(B,E),a.mul(C2,D));
        const auto t=a.sub(a.mul(A,E),a.mul(B,D));
        if(a.sign(s)>=0&&a.cmp(s,det)<=0&&a.sign(t)>=0&&a.cmp(t,det)<=0)
            consider(a,edge_parameters(a,x,e1,e2,s,t,det),best);
    }
    if(a.sign(C2)>0) {
        consider(a,edge_parameters(a,x,e1,e2,I(0),clamp(a,E,C2),C2),best);
        consider(a,edge_parameters(a,x,e1,e2,C2,clamp(a,a.sum(E,B),C2),C2),best);
    } else {
        consider(a,edge_parameters(a,x,e1,e2,I(0),I(0),I(1)),best);
        consider(a,edge_parameters(a,x,e1,e2,I(1),I(0),I(1)),best);
    }
    if(a.sign(A)>0) {
        consider(a,edge_parameters(a,x,e1,e2,clamp(a,exact::negate(D),A),I(0),A),best);
        consider(a,edge_parameters(a,x,e1,e2,clamp(a,a.sub(B,D),A),A,A),best);
    } else {
        consider(a,edge_parameters(a,x,e1,e2,I(0),I(0),I(1)),best);
        consider(a,edge_parameters(a,x,e1,e2,I(0),I(1),I(1)),best);
    }
    return best;
}

// CONTACT-ONLY derivative-feature override. The generic exact `edge_edge()`
// above fixes the physical closest geometry (weights, residual, denominator).
// The CPU, however, chooses its derivative free parameters from the ROUNDED
// double feature branch, which can disagree with the exact closest feature.
// Keep the generic geometry and replace only the free-parameter basis here.
//
// The CPU's inclusive Interior branch keeps both s and t free even when the
// exact (or rounded) closest point sits on a closed boundary; its OUTSIDE
// branch keeps only the non-clamped parameter free (and none at a corner).
// Those are derivative-policy quirks, not geometry, so DAT/CCD and the robust
// exact feature remain untouched (`regular == false` leaves `f` unchanged).
// The tangents are the stable exact edge directions e1 and -e2.
template<int C> IPC_RES_CONTACT_EXACT_HD inline void
apply_cpu_ordinary_contact_derivative_feature(
    exact::Arithmetic<C>& a,const exact::Points<C>& x,
    const double positions[4][3],double eps,Feature<C>& f) {
    if(!a.ok||!f.have)return;
    const auto cpu=contact_detail::segment_contact_cpu_ordinary_feature(positions,eps);
    if(!cpu.regular)return;
    // Reset the 2x4 sigma matrix and free count before refilling only the
    // selected degrees. Unused tangent slots are never read (loops use n).
    f.free_parameters=0;
    for(int j=0;j<2;++j)for(int i=0;i<4;++i)f.sigma[j][i]=0;
    if(cpu.free_s) {
        const auto e1=exact::difference(a,x.p[1],x.p[0]);
        if(!a.ok)return;
        const int j=f.free_parameters++;
        f.sigma[j][0]=-1;f.sigma[j][1]=1;
        f.tangent[j]=e1;
    }
    if(cpu.free_t) {
        const auto e2=exact::difference(a,x.p[3],x.p[2]);
        if(!a.ok)return;
        const int j=f.free_parameters++;
        f.sigma[j][2]=1;f.sigma[j][3]=-1;
        for(int k=0;k<3;++k)f.tangent[j].v[k]=exact::negate(e2.v[k]);
    }
}

// Unlike a squared norm, this retains representable subnormal gaps.
IPC_RES_CONTACT_EXACT_HD inline double stable_norm(const double* v) {
    double scale=0;
    for(int k=0;k<3;++k)if(contact_detail::absval(v[k])>scale)scale=contact_detail::absval(v[k]);
    if(scale==0)return 0;
    double sum=0;for(int k=0;k<3;++k){const double y=v[k]/scale;sum+=y*y;}
    return scale*sqrt(sum);
}
} // namespace contact_exact_detail

template<int C> IPC_RES_CONTACT_EXACT_HD inline ContactOutput evaluate_contact_exact(const ContactInput& in) {
    using namespace contact_exact_detail;
    using I=exact::Integer<C>;
    ContactOutput out;
    exact::Arithmetic<C> a;
    int unit=0;bool have_unit=false;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const auto value=exact::decompose(in.positions[i][k]);
        if(!value.valid||!contact_detail::finite(in.previous[i][k])){out.status=ContactStatus::InvalidInput;return out;}
        if(value.sign&&(!have_unit||value.exponent<unit)){unit=value.exponent;have_unit=true;}
    }
    if(!contact_detail::finite(in.d_hat)||!(in.d_hat>0)||!contact_detail::finite(in.k_barrier)||in.k_barrier<0
        ||!contact_detail::finite(in.dt)||!(in.dt>0)||!contact_detail::finite(in.friction)||in.friction<0
        ||!contact_detail::finite(in.eps_v)||!(in.eps_v>0)||!contact_detail::finite(in.eps)||!(in.eps>0)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge)
        ||(in.hessian_blocks&~0xffffu)) {
        out.status=ContactStatus::InvalidInput;return out;
    }
    if(in.k_barrier==0)return out;
    exact::Points<C> x;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){x.p[i].v[k]=exact::from_double<C>(in.positions[i][k],unit);a.ok=a.ok&&x.p[i].v[k].valid;}
    if(!a.ok){out.status=ContactStatus::ExactCapacityExceeded;return out;}
    contact_exact_detail::Feature<C> f;
    if(in.kind==ContactKind::PointTriangle) {
        f=contact_exact_detail::point_triangle(a,x,
            contact_detail::degenerate_triangle_contact_feature(in));
    } else {
        // Always build the unchanged generic exact closest geometry first.
        // Then, only for a CPU regular (non-robust) branch, override the
        // derivative free-parameter basis with the CPU's rounded feature
        // policy. Geometry (weights/residual/denominator) remains exact, so
        // DAT/CCD nearest-shape decisions are untouched.
        f=contact_exact_detail::edge_edge(a,x);
        if(!a.ok){out.status=ContactStatus::ExactCapacityExceeded;return out;}
        contact_exact_detail::apply_cpu_ordinary_contact_derivative_feature(
            a,x,in.positions,in.eps,f);
    }
    if(!a.ok||!f.have){out.status=ContactStatus::ExactCapacityExceeded;return out;}
    if(!a.sign(f.squared_residual)){out.status=ContactStatus::ZeroDistance;return out;}
    double residual[3];
    for(int k=0;k<3;++k)residual[k]=exact::ratio_to_double(f.residual[k],f.denominator,unit,&a.ok);
    for(int i=0;i<4;++i)out.weights[i]=exact::ratio_to_double(f.weight[i],f.denominator,0,&a.ok);
    if(!a.ok){out.status=ContactStatus::ExactCapacityExceeded;return out;}
    out.distance=stable_norm(residual);
    if(!(out.distance>0)||!contact_detail::finite(out.distance)){out.status=ContactStatus::NonFiniteDerivative;return out;}
    for(int k=0;k<3;++k)out.normal[k]=residual[k]/out.distance;
    if(out.distance>=in.d_hat)return out;
    const double distance=out.distance,s=distance-in.d_hat,logratio=log(distance/in.d_hat),ratio=in.d_hat/distance;
    const double bp=-2*s*logratio-s*s/distance,bpp=ratio*ratio+2*ratio-3-2*logratio;
    I gram[2][2],z(1);
    const int n=f.free_parameters;
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)gram[i][j]=exact::dot(a,f.tangent[i],f.tangent[j]);
    if(n==1)z=gram[0][0];
    if(n==2)z=a.sub(a.mul(gram[0][0],gram[1][1]),a.mul(gram[0][1],gram[1][0]));
    if(!a.ok){out.status=ContactStatus::ExactCapacityExceeded;return out;}
    if(a.sign(z)<=0){out.status=ContactStatus::NonFiniteDerivative;return out;}
    const I p2=a.mul(f.denominator,f.denominator),normal_den=a.mul(p2,f.squared_residual);
    const I curvature_den=a.mul(normal_den,z);
    // Match the CPU exact-geometry reuse: each endpoint derivative is fixed
    // for this contact. Compute it once, not anew for every Hessian entry.
    // These integers stay in the device arena only for this evaluation; no
    // cache can survive a color/body update or retain stale endpoint bits.
    I derivatives[4][3][2];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        out.barrier_gradient[i][k]=(bp*out.weights[i])*out.normal[k];
        bool needed=false;
        for(int j=0;j<4;++j)needed=needed||bool(in.hessian_blocks&((1u<<(4*i+j))|(1u<<(4*j+i))));
        if(needed)for(int v=0;v<n;++v)derivatives[i][k][v]=a.sum(
            a.mul(f.weight[i],f.tangent[v].v[k]),a.mul(I(f.sigma[v][i]),f.residual[k]));
    }
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(in.hessian_blocks&(1u<<(4*i+j))) {
        const I ww=a.mul(f.weight[i],f.weight[j]);
        for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
            const I normal_num=a.mul(a.mul(ww,f.residual[k]),f.residual[l]);
            const I* vi=derivatives[i][k];const I* vj=derivatives[j][l];
            I schur;
            if(n==1)schur=a.mul(vi[0],vj[0]);
            if(n==2)schur=a.sub(
                a.sum(a.mul(gram[1][1],a.mul(vi[0],vj[0])),a.mul(gram[0][0],a.mul(vi[1],vj[1]))),
                a.mul(gram[0][1],a.sum(a.mul(vi[0],vj[1]),a.mul(vi[1],vj[0]))));
            const I hq_num=a.sub(k==l?a.mul(ww,z):I(0),schur);
            const I curvature_num=a.sub(a.mul(hq_num,f.squared_residual),a.mul(normal_num,z));
            const double normal_term=exact::ratio_to_double(normal_num,normal_den,0,&a.ok);
            const double curvature=exact::ratio_to_double(curvature_num,curvature_den,0,&a.ok);
            out.barrier_hessian[i][j][3*k+l]=bpp*normal_term+(bp/distance)*curvature;
        }
    }
    if(!a.ok){out.status=ContactStatus::ExactCapacityExceeded;return out;}
    if(in.friction>0) {
        double displacement[3]{},P[9],u[3]{};
        for(int k=0;k<3;++k)for(int i=0;i<4;++i)displacement[k]+=out.weights[i]*(in.positions[i][k]-in.previous[i][k]);
        // Production robust EE normalizes its rounded residual with stableNorm
        // independently for friction. Use the same device-only convention.
        const double normal_length=stable_norm(residual);
        for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
            P[3*k+l]=(k==l?1.0:0.0)-(residual[k]/normal_length)*(residual[l]/normal_length);
            u[k]+=P[3*k+l]*displacement[l];
        }
        const double slip=contact_detail::norm(u),eps_u=in.dt*in.eps_v;
        const double mollifier=slip<eps_u?2/eps_u-slip/(eps_u*eps_u):1/slip;
        const double scale=in.dt*in.dt*in.friction*(-in.k_barrier*bp)*mollifier;
        for(int k=0;k<3;++k) {
            out.friction_relative_gradient[k]=scale*u[k];
            for(int l=0;l<3;++l)out.friction_relative_hessian[3*k+l]=scale*P[3*k+l];
        }
    }
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!contact_detail::finite(out.barrier_gradient[i][k])||!contact_detail::finite(out.friction_relative_gradient[k]))out.status=ContactStatus::NonFiniteDerivative;
        for(int j=0;j<4;++j)for(int l=0;l<3;++l)
            if(!contact_detail::finite(out.barrier_hessian[i][j][3*k+l])||!contact_detail::finite(out.friction_relative_hessian[3*k+l]))out.status=ContactStatus::NonFiniteDerivative;
    }
    out.active=out.status==ContactStatus::Ok;
    return out;
}

// Convenient correctness entry point. For performance, the solver should
// compact uncertain contacts after the ordinary kernel and launch this exact
// evaluator only on that cold queue, rather than inflate every fast thread's
// local-memory allocation. An ExactCapacityExceeded result must be retried on
// the GPU with more limbs; it is never an acceptable final contact result.
template<int C> IPC_RES_CONTACT_EXACT_HD inline ContactOutput evaluate_contact_gpu(const ContactInput& in) {
    const auto ordinary=evaluate_contact(in);
    if(ordinary.status==ContactStatus::NeedsExactGeometry
        ||ordinary.status==ContactStatus::ZeroDistance
        ||ordinary.status==ContactStatus::NonFiniteDerivative)
        return evaluate_contact_exact<C>(in);
    return ordinary;
}

} // namespace solver
} // namespace ipc_gpu

#undef IPC_RES_CONTACT_EXACT_HD

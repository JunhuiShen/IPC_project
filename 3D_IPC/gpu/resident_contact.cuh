#pragma once

// Device-only numerical contact building block. This is deliberately independent
// of Eigen, host geometry routines, and the hybrid GPU bridge. All outputs are
// computed from four current and previous Cartesian positions on the device.
// A non-Ok status MUST abort/roll back the attempted update. In particular,
// NeedsExactGeometry is not an inactive contact and may not be ignored.
//
// This file is the ordinary-double path. The companion
// resident_contact_exact.cuh handles uncertain geometry on the GPU using exact
// integer ratios. Its fail-closed guards request that path instead of quietly
// using an unreliable double-precision result. A complete solver must route
// those requests to the device exact queue before using any derivatives.
#include <cfloat>
#include <cmath>
#include "resident_ccd.cuh"

#ifdef __CUDACC__
#define IPC_RES_CONTACT_HD __host__ __device__
#else
#define IPC_RES_CONTACT_HD
#endif

namespace ipc_gpu {
namespace resident {

enum class ContactKind : unsigned { PointTriangle, EdgeEdge };
enum class ContactStatus : unsigned {
    Ok, InvalidInput, NeedsExactGeometry, ZeroDistance, NonFiniteDerivative,
    ExactCapacityExceeded
};

struct ContactInput {
    double positions[4][3]{};
    double previous[4][3]{};
    double d_hat = 0.005;
    double k_barrier = 100.0;
    double dt = 1.0 / 300.0;
    double friction = 0.0;
    double eps_v = 0.001;
    double eps = 1e-12;
    ContactKind kind = ContactKind::PointTriangle;
    // Bit (4*i+j) requests Cartesian Hessian block (i,j). A vertex update
    // needs only its self block; rigid phases request their live cross blocks.
    // Omitted blocks are zero, not a symmetric completion. Geometry, all four
    // gradients, and relative friction derivatives are unaffected by the mask.
    unsigned hessian_blocks = 0xffffu;
};

struct ContactOutput {
    // Barrier derivatives are unscaled: caller multiplies by dt^2*k_barrier.
    // Friction derivatives already contain dt^2*friction*k_barrier. Cartesian
    // friction role blocks are w_i*g and w_i*w_j*H. Rigid pullbacks must retain
    // these cross blocks, not just four self blocks.
    // For a certified-inactive uncertain feature, distance is the d_hat
    // lower-bound sentinel and weights/normal remain zero. No derivatives or
    // geometric diagnostics may be consumed when active == 0.
    double distance = 0.0;
    double weights[4]{};
    double normal[3]{};
    double barrier_gradient[4][3]{};
    double barrier_hessian[4][4][9]{};
    double friction_relative_gradient[3]{};
    double friction_relative_hessian[9]{};
    ContactStatus status = ContactStatus::Ok;
    unsigned active = 0;
};

namespace contact_detail {
IPC_RES_CONTACT_HD inline bool finite(double a) { return a <= DBL_MAX && a >= -DBL_MAX; }
IPC_RES_CONTACT_HD inline double absval(double a) { return a < 0 ? -a : a; }
IPC_RES_CONTACT_HD inline double clamp(double t) { return t < 0 ? 0 : (t > 1 ? 1 : t); }
IPC_RES_CONTACT_HD inline double dot(const double* a, const double* b) {
    return (a[0]*b[0] + a[1]*b[1]) + a[2]*b[2];
}
IPC_RES_CONTACT_HD inline void sub(const double* a, const double* b, double* c) {
    for (int k = 0; k < 3; ++k) c[k] = a[k] - b[k];
}
IPC_RES_CONTACT_HD inline void cross(const double* a, const double* b, double* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}
IPC_RES_CONTACT_HD inline double norm(const double* a) { return sqrt(dot(a,a)); }

// These axes only propose a separating direction. Their construction need not
// be exact: acceptance uses outward-rounded projections of the ORIGINAL input
// coordinates, never an approximate closest point or approximate distance.
IPC_RES_CONTACT_HD inline resident_ccd::detail::Vec unit_scale_axis(
    resident_ccd::detail::Vec axis) {
    double scale=0;
    for(int k=0;k<3;++k) {
        if(!finite(axis.v[k]))return {{0,0,0}};
        if(absval(axis.v[k])>scale)scale=absval(axis.v[k]);
    }
    if(!(scale>0))return {{0,0,0}};
    for(int k=0;k<3;++k)axis.v[k]/=scale;
    return axis;
}

IPC_RES_CONTACT_HD inline bool inactive_projection(
    const ContactInput& in,resident_ccd::detail::Vec direction) {
    using namespace resident_ccd::detail;
    const Vec axis=unit_scale_axis(direction);
    if(axis.v[0]==0&&axis.v[1]==0&&axis.v[2]==0)return false;
    Interval norm2=point(0);
    for(int k=0;k<3;++k)norm2=add(norm2,multiply(point(axis.v[k]),point(axis.v[k])));
    const double threshold=multiply(point(in.d_hat),point(sqrt_upper(norm2.hi))).hi;
    if(!contact_detail::finite(threshold))return false;
    const int split=in.kind==ContactKind::PointTriangle?1:2;
    double lo[2]={INFINITY,INFINITY},hi[2]={-INFINITY,-INFINITY};
    for(int i=0;i<4;++i) {
        Interval projection=point(0);
        for(int k=0;k<3;++k) {
            // Local-origin subtraction is also enclosed, including its
            // rounding. Large translations therefore do not imply safety.
            const Interval position=subtract(point(in.positions[i][k]),point(in.positions[3][k]));
            projection=add(projection,multiply(point(axis.v[k]),position));
        }
        const int side=i<split?0:1;
        lo[side]=minimum(lo[side],projection.lo);
        hi[side]=maximum(hi[side],projection.hi);
    }
    const double gap01=subtract(point(lo[0]),point(hi[1])).lo;
    const double gap10=subtract(point(lo[1]),point(hi[0])).lo;
    // Strict comparison intentionally leaves equality/rounding ambiguity to
    // the exact feature path. A finite lower bound must exceed an upper bound.
    return (contact_detail::finite(gap01)&&gap01>threshold)
        ||(contact_detail::finite(gap10)&&gap10>threshold);
}

IPC_RES_CONTACT_HD inline bool inactive_bounds(const ContactInput& in) {
    using namespace resident_ccd::detail;
    const int split=in.kind==ContactKind::PointTriangle?1:2;
    // Cheapest certificate: a coordinate gap between the primitive AABBs.
    // No squares/square root means this also works for subnormal gaps.
    for(int k=0;k<3;++k) {
        double lo[2]={INFINITY,INFINITY},hi[2]={-INFINITY,-INFINITY};
        for(int i=0;i<4;++i) {
            const int side=i<split?0:1;
            lo[side]=minimum(lo[side],in.positions[i][k]);
            hi[side]=maximum(hi[side],in.positions[i][k]);
        }
        const double gap01=subtract(point(lo[0]),point(hi[1])).lo;
        const double gap10=subtract(point(lo[1]),point(hi[0])).lo;
        if((contact_detail::finite(gap01)&&gap01>in.d_hat)
            ||(contact_detail::finite(gap10)&&gap10>in.d_hat))return true;
    }
    const auto& x=in.positions;
    if(in.kind==ContactKind::PointTriangle) {
        const Vec a=unit_scale_axis(difference(x[2],x[1]));
        const Vec b=unit_scale_axis(difference(x[3],x[1]));
        const Vec normal=unit_scale_axis(resident_ccd::detail::cross(a,b));
        if(inactive_projection(in,normal))return true;
        for(int i=1;i<4;++i) {
            const Vec edge=unit_scale_axis(difference(x[i==3?1:i+1],x[i]));
            if(inactive_projection(in,resident_ccd::detail::cross(normal,edge)))return true;
        }
    } else {
        const Vec a=unit_scale_axis(difference(x[1],x[0]));
        const Vec b=unit_scale_axis(difference(x[3],x[2]));
        const Vec w=unit_scale_axis(difference(x[2],x[0]));
        if(inactive_projection(in,resident_ccd::detail::cross(a,b))
            ||inactive_projection(in,resident_ccd::detail::cross(resident_ccd::detail::cross(a,w),a))
            ||inactive_projection(in,resident_ccd::detail::cross(resident_ccd::detail::cross(b,w),b)))return true;
    }
    // Degenerate point/edge features can still have a useful endpoint axis.
    for(int i=0;i<split;++i)for(int j=split;j<4;++j)
        if(inactive_projection(in,difference(x[i],x[j])))return true;
    return false;
}

struct Feature {
    double w[4]{};
    // r = sum_i w_i*x_i; u_a = d r / d free_parameter_a;
    // sigma[a][i] = d w_i / d free_parameter_a.
    double u[2][3]{};
    double sigma[2][4]{};
    double residual[3]{};
    double distance = 0;
    int free_parameters = 0;
};

IPC_RES_CONTACT_HD inline double edge_parameter(
    const double* x, const double* a, const double* b) {
    double e[3],w[3]; sub(b,a,e); sub(x,a,w);
    const double ee = dot(e,e);
    return ee > 0 ? clamp(dot(w,e)/ee) : 0;
}

IPC_RES_CONTACT_HD inline void point_feature(
    const double x[4][3], int p, int a, Feature& f) {
    f = Feature{}; f.w[p] = 1; f.w[a] = -1;
    sub(x[p],x[a],f.residual); f.distance = norm(f.residual);
}

IPC_RES_CONTACT_HD inline void point_edge_feature(
    const double x[4][3], int p, int a, int b, double t, Feature& f) {
    if (t <= 0) { point_feature(x,p,a,f); return; }
    if (t >= 1) { point_feature(x,p,b,f); return; }
    f = Feature{};
    f.w[p] = 1; f.w[a] = -(1-t); f.w[b] = -t;
    f.free_parameters = 1; f.sigma[0][a] = 1; f.sigma[0][b] = -1;
    for (int k = 0; k < 3; ++k) {
        f.u[0][k] = x[a][k] - x[b][k];
        // Same world-space closest-point arithmetic as the ordinary CPU path.
        f.residual[k] = x[p][k] - (x[a][k]+t*(x[b][k]-x[a][k]));
    }
    f.distance = norm(f.residual);
}

IPC_RES_CONTACT_HD inline ContactStatus point_triangle(
    const ContactInput& in, Feature& f, double* normal) {
    const auto& x = in.positions;
    double a[3],b[3],w[3],n[3];
    sub(x[2],x[1],a); sub(x[3],x[1],b); sub(x[0],x[1],w); cross(a,b,n);
    const double nn = norm(n);
    if (!(nn > in.eps)) {
        // Degenerate triangles have a deterministic three-edge fallback in
        // production. Do not pretend a smooth face Hessian exists here.
        return ContactStatus::NeedsExactGeometry;
    }
    for (int k = 0; k < 3; ++k) n[k] /= nn;
    const double phi = dot(n,w);
    double projected[3],projected_w[3];
    for (int k = 0; k < 3; ++k) projected[k] = x[0][k] - phi*n[k];
    sub(projected,x[1],projected_w);
    const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(projected_w,a),E=dot(projected_w,b);
    const double det = A*C-B*B;
    if (!(det > in.eps) || det <= 64*DBL_EPSILON*A*C)
        return ContactStatus::NeedsExactGeometry;
    const double l2=(D*C-E*B)/det,l3=(-D*B+E*A)/det,l1=1-l2-l3;
    if (l1>=0 && l2>=0 && l3>=0) {
        f.w[0]=1;f.w[1]=-l1;f.w[2]=-l2;f.w[3]=-l3;
        f.free_parameters=2;
        f.sigma[0][1]=1;f.sigma[0][2]=-1;
        f.sigma[1][1]=1;f.sigma[1][3]=-1;
        for (int k=0;k<3;++k) {
            f.u[0][k]=-a[k];f.u[1][k]=-b[k];
            f.residual[k]=phi*n[k];
            normal[k]=(phi<0 ? -1.0:1.0)*n[k];
        }
        f.distance=absval(phi);
        return ContactStatus::Ok;
    }
    if(l2<=0 && l3<=0) point_feature(x,0,1,f);
    else if(l3<=0 && l1<=0) point_feature(x,0,2,f);
    else if(l1<=0 && l2<=0) point_feature(x,0,3,f);
    else {
        const int ia=l3<0?1:(l1<0?2:3),ib=l3<0?2:(l1<0?3:1);
        const double t=edge_parameter(projected,x[ia],x[ib]);
        // At a clamped endpoint production may retain the edge feature enum
        // while evaluating its unclamped derivative formula. Do not silently
        // choose a different Hessian at that branch boundary.
        if(!(t>1e-14&&t<1-1e-14))return ContactStatus::NeedsExactGeometry;
        point_edge_feature(x,0,ia,ib,t,f);
    }
    if(f.distance>0) for(int k=0;k<3;++k) normal[k]=f.residual[k]/f.distance;
    return ContactStatus::Ok;
}

IPC_RES_CONTACT_HD inline ContactStatus edge_edge(
    const ContactInput& in, Feature& f, double* normal) {
    const auto& x=in.positions;
    double a[3],b[3],c[3];sub(x[1],x[0],a);sub(x[3],x[2],b);sub(x[0],x[2],c);
    const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,c),E=dot(b,c);
    const double determinant=A*C-B*B;
    // Precisely the numerical regime where production requests rationals.
    if(!finite(determinant)||determinant<=64*DBL_EPSILON*A*C||determinant<=in.eps*in.eps)
        return ContactStatus::NeedsExactGeometry;
    double s=(B*E-C*D)/determinant,t=(A*E-B*D)/determinant;
    if(!(s>=0&&s<=1&&t>=0&&t<=1)) {
        const double ss[4]={0,1,clamp(-D/A),clamp((B-D)/A)};
        const double tt[4]={clamp(E/C),clamp((E+B)/C),0,1};
        double best=DBL_MAX;
        for(int i=0;i<4;++i) {
            double residual[3];
            for(int k=0;k<3;++k) residual[k]=(x[0][k]+ss[i]*a[k])-(x[2][k]+tt[i]*b[k]);
            const double distance=norm(residual);
            if(distance<best){best=distance;s=ss[i];t=tt[i];}
        }
    }
    // CPU classifies parameters within 1e-14 of endpoints as endpoint features.
    // Those ambiguous transitions need an exact active-feature decision.
    if ((s>0&&s<=1e-14)||(s<1&&s>=1-1e-14)||(t>0&&t<=1e-14)||(t<1&&t>=1-1e-14))
        return ContactStatus::NeedsExactGeometry;
    f.w[0]=1-s;f.w[1]=s;f.w[2]=t-1;f.w[3]=-t;
    if(s>0&&s<1) {
        const int j=f.free_parameters++;
        f.sigma[j][0]=-1;f.sigma[j][1]=1;
        for(int k=0;k<3;++k)f.u[j][k]=a[k];
    }
    if(t>0&&t<1) {
        const int j=f.free_parameters++;
        f.sigma[j][2]=1;f.sigma[j][3]=-1;
        for(int k=0;k<3;++k)f.u[j][k]=-b[k];
    }
    for(int k=0;k<3;++k)f.residual[k]=(x[0][k]+s*a[k])-(x[2][k]+t*b[k]);
    f.distance=norm(f.residual);
    if(f.distance>0)for(int k=0;k<3;++k)normal[k]=f.residual[k]/f.distance;
    return ContactStatus::Ok;
}
} // namespace contact_detail

// True proves the finite represented primitives are farther apart than d_hat.
// False means UNKNOWN, not active contact. This never prunes the candidate
// graph or CCD lists and never substitutes an approximate distance for an
// exact active-feature decision. Only geometry inputs are relevant here;
// evaluate_contact validates all material/time inputs before calling it.
IPC_RES_CONTACT_HD inline bool certified_contact_inactive(const ContactInput& in) {
    if(!(in.d_hat>0)||!contact_detail::finite(in.d_hat)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge))return false;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        if(!contact_detail::finite(in.positions[i][k]))return false;
    return contact_detail::inactive_bounds(in);
}

// Feasible (not certified closest) primitive weights for a conservative CCD
// certificate. This helper NEVER proves a positive lower distance bound.
// Its sole purpose is to choose a pair of points on the two primitives, so
// directed interval arithmetic can obtain an UPPER bound on starting distance.
// The CCD caller must certify convexity itself after any parameter conversion.
// Unlike evaluate_contact, this helper is allowed to return a suboptimal pair
// for degenerate/parallel geometry and does not require exact arithmetic.
IPC_RES_CONTACT_HD inline bool feasible_contact_weights(
    const ContactInput& in,double weights[4]) {
    using namespace contact_detail;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        if(!finite(in.positions[i][k]))return false;
    Feature feature{};double normal[3]{};
    if(in.kind==ContactKind::PointTriangle) {
        if(point_triangle(in,feature,normal)==ContactStatus::Ok) {
            for(int i=0;i<4;++i)weights[i]=feature.w[i];
            return true;
        }
        const int aa[3]={1,2,3},bb[3]={2,3,1};
        double best=DBL_MAX;
        for(int edge=0;edge<3;++edge) {
            Feature candidate{};
            point_edge_feature(in.positions,0,aa[edge],bb[edge],
                edge_parameter(in.positions[0],in.positions[aa[edge]],in.positions[bb[edge]]),candidate);
            if(finite(candidate.distance)&&candidate.distance<best){best=candidate.distance;feature=candidate;}
        }
        if(best==DBL_MAX)return false;
    } else if(in.kind==ContactKind::EdgeEdge) {
        if(edge_edge(in,feature,normal)==ContactStatus::Ok) {
            for(int i=0;i<4;++i)weights[i]=feature.w[i];
            return true;
        }
        double best=DBL_MAX,best_s=0,best_t=0;
        for(int edge=0;edge<4;++edge) {
            const double s=edge<2?double(edge):edge_parameter(in.positions[edge],in.positions[0],in.positions[1]);
            const double t=edge<2?edge_parameter(in.positions[edge],in.positions[2],in.positions[3]):double(edge-2);
            double residual[3];
            for(int k=0;k<3;++k)residual[k]=
                (in.positions[0][k]+s*(in.positions[1][k]-in.positions[0][k]))
                -(in.positions[2][k]+t*(in.positions[3][k]-in.positions[2][k]));
            const double distance=norm(residual);
            if(finite(distance)&&distance<best){best=distance;best_s=s;best_t=t;}
        }
        if(best==DBL_MAX)return false;
        feature.w[0]=1-best_s;feature.w[1]=best_s;feature.w[2]=best_t-1;feature.w[3]=-best_t;
    } else return false;
    for(int i=0;i<4;++i)weights[i]=feature.w[i];
    return true;
}

IPC_RES_CONTACT_HD inline ContactOutput evaluate_contact(const ContactInput& in) {
    using namespace contact_detail;
    ContactOutput out{};
    double coordinate_scale=0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(in.positions[i][k])||!finite(in.previous[i][k])) {
            out.status=ContactStatus::InvalidInput;return out;
        }
        if(absval(in.positions[i][k])>coordinate_scale)coordinate_scale=absval(in.positions[i][k]);
    }
    if(!finite(in.d_hat)||!(in.d_hat>0)||!finite(in.k_barrier)||in.k_barrier<0
        ||!finite(in.dt)||!(in.dt>0)||!finite(in.friction)||in.friction<0
        ||!finite(in.eps_v)||!(in.eps_v>0)||!finite(in.eps)||!(in.eps>0)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge)
        ||(in.hessian_blocks&~0xffffu)) {
        out.status=ContactStatus::InvalidInput;return out;
    }
    if(in.k_barrier==0)return out;
    Feature feature{};
    out.status=in.kind==ContactKind::PointTriangle?point_triangle(in,feature,out.normal):edge_edge(in,feature,out.normal);
    if(out.status!=ContactStatus::Ok) {
        if(out.status==ContactStatus::NeedsExactGeometry&&certified_contact_inactive(in)) {
            ContactOutput inactive{};inactive.distance=in.d_hat;return inactive;
        }
        return out;
    }
    out.distance=feature.distance;
    for(int i=0;i<4;++i)out.weights[i]=feature.w[i];
    if(!finite(out.distance)){out.status=ContactStatus::NeedsExactGeometry;return out;}
    if(out.distance>=in.d_hat)return out;
    if(out.distance==0){out.status=ContactStatus::ZeroDistance;return out;}
    // The Schur curvature subtraction below is cancellation-sensitive as the
    // gap tends to zero. Reject it instead of dividing roundoff by a tiny gap.
    // This stricter-than-production guard is intentional and is not a gap floor.
    if(out.distance<=sqrt(DBL_EPSILON)*coordinate_scale) {
        out.status=ContactStatus::NeedsExactGeometry;return out;
    }
    const double d=out.distance,s=d-in.d_hat,logratio=log(d/in.d_hat),ratio=in.d_hat/d;
    const double bp=-2*s*logratio-s*s/d,bpp=ratio*ratio+2*ratio-3-2*logratio;
    const int n=feature.free_parameters;
    double inverse[2][2]{};
    if(n==1) inverse[0][0]=1/dot(feature.u[0],feature.u[0]);
    if(n==2) {
        const double A=dot(feature.u[0],feature.u[0]),B=dot(feature.u[0],feature.u[1]),C=dot(feature.u[1],feature.u[1]);
        const double determinant=A*C-B*B;
        if(!(determinant>64*DBL_EPSILON*A*C)) {out.status=ContactStatus::NeedsExactGeometry;return out;}
        inverse[0][0]=C/determinant;inverse[0][1]=inverse[1][0]=-B/determinant;inverse[1][1]=A/determinant;
    }
    // Differentiate q=0.5*||sum_i w_i*x_i||^2 after minimizing over the
    // active feature's free parameters. The Schur complement is the full
    // Hessian of minimized q, NOT frozen-weight/Gauss-Newton differentiation.
    double q_xt[4][3][2]{};
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)for(int a=0;a<n;++a)
        q_xt[i][k][a]=feature.w[i]*feature.u[a][k]+feature.sigma[a][i]*feature.residual[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const double gi=feature.w[i]*out.normal[k];
        out.barrier_gradient[i][k]=bp*gi;
        for(int j=0;j<4;++j)if(in.hessian_blocks&(1u<<(4*i+j)))for(int l=0;l<3;++l) {
            const double gj=feature.w[j]*out.normal[l];
            double hq=(k==l?feature.w[i]*feature.w[j]:0);
            for(int a=0;a<n;++a)for(int b=0;b<n;++b)hq-=q_xt[i][k][a]*inverse[a][b]*q_xt[j][l][b];
            out.barrier_hessian[i][j][3*k+l]=bpp*gi*gj+bp*(hq-gi*gj)/d;
        }
    }
    if(in.friction>0) {
        double displacement[3]{},P[9],u[3]{};
        for(int k=0;k<3;++k)for(int i=0;i<4;++i)
            displacement[k]+=feature.w[i]*(in.positions[i][k]-in.previous[i][k]);
        for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
            P[3*k+l]=(k==l?1.0:0.0)-out.normal[k]*out.normal[l];
            u[k]+=P[3*k+l]*displacement[l];
        }
        const double slip=norm(u),eps_u=in.dt*in.eps_v;
        const double mollifier=slip<eps_u?2/eps_u-slip/(eps_u*eps_u):1/slip;
        const double scale=in.dt*in.dt*in.friction*(-in.k_barrier*bp)*mollifier;
        for(int k=0;k<3;++k) {
            out.friction_relative_gradient[k]=scale*u[k];
            for(int l=0;l<3;++l)out.friction_relative_hessian[3*k+l]=scale*P[3*k+l];
        }
    }
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(out.barrier_gradient[i][k])||!finite(out.friction_relative_gradient[k]))out.status=ContactStatus::NonFiniteDerivative;
        for(int j=0;j<4;++j)for(int l=0;l<3;++l)
            if(!finite(out.barrier_hessian[i][j][3*k+l])||!finite(out.friction_relative_hessian[3*k+l]))out.status=ContactStatus::NonFiniteDerivative;
    }
    out.active=out.status==ContactStatus::Ok;
    return out;
}

} // namespace resident
} // namespace ipc_gpu

#undef IPC_RES_CONTACT_HD

#pragma once
#include "resident_contact.cuh"
#include "resident_solve.cuh"

#ifdef __CUDACC__
#define IPC_DAT_HD __host__ __device__
#else
#define IPC_DAT_HD
#endif

namespace ipc_gpu::resident::dat {
struct Feature : contact_detail::Feature {
    int edge_a=-1,edge_b=-1;
};
using contact_detail::dot;
using contact_detail::sub;
using contact_detail::cross;
using contact_detail::norm;
using contact_detail::clamp;

// Match the corrected OGC/DAT boundary search without changing the ordinary
// resident GS contact evaluator or depending on CPU numerical routines.
IPC_DAT_HD inline Feature contact_feature(const double x[4][3],bool vertex_face) {
    Feature f{};
    if(vertex_face) {
        double a[3],b[3],r[3],n[3];sub(x[2],x[1],a);sub(x[3],x[1],b);sub(x[0],x[1],r);cross(a,b,n);
        const double nn=norm(n);
        double projected[3];for(int k=0;k<3;++k)projected[k]=x[0][k];
        if(nn>1e-12) {
            for(int k=0;k<3;++k)n[k]/=nn;
            const double phi=dot(n,r);
            for(int k=0;k<3;++k)projected[k]-=phi*n[k];
            double w[3];sub(projected,x[1],w);
            const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(w,a),E=dot(w,b),det=A*C-B*B;
            double l1=0,l2=0,l3=0;
            if(fabs(det)>1e-12){l2=(D*C-E*B)/det;l3=(-D*B+E*A)/det;l1=1-l2-l3;}
            if(l1>=0&&l2>=0&&l3>=0) {
                f.w[0]=1;f.w[1]=-l1;f.w[2]=-l2;f.w[3]=-l3;f.free_parameters=2;
                f.sigma[0][1]=1;f.sigma[0][2]=-1;f.sigma[1][1]=1;f.sigma[1][3]=-1;
                for(int k=0;k<3;++k){f.u[0][k]=-a[k];f.u[1][k]=-b[k];f.residual[k]=phi*n[k];}
                f.distance=fabs(phi);return f;
            }
        }
        double best=INFINITY;
        for(int ia=1;ia<=3;++ia) {
            const int ib=ia==3?1:ia+1;Feature candidate{};
            contact_detail::point_edge_feature(x,0,ia,ib,contact_detail::edge_parameter(x[0],x[ia],x[ib]),candidate);
            if(nn>1e-12&&candidate.free_parameters==1){candidate.edge_a=ia;candidate.edge_b=ib;}
            if(candidate.distance<best){best=candidate.distance;f=candidate;}
        }
        return f;
    }
    double a[3],b[3],r[3];sub(x[1],x[0],a);sub(x[3],x[2],b);sub(x[0],x[2],r);
    const double A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,r),E=dot(b,r),det=A*C-B*B;
    double s=0,t=0;bool interior=false;
    if(det>1e-24){s=(B*E-C*D)/det;t=(A*E-B*D)/det;interior=s>=0&&s<=1&&t>=0&&t<=1;}
    if(!interior) {
        double best=INFINITY;
        for(int e=0;e<4;++e) {
            double cs=0,ct=0,w[3];
            if(e<2){cs=e;for(int k=0;k<3;++k)w[k]=(x[0][k]+cs*a[k])-x[2][k];ct=C>0?clamp(dot(w,b)/C):0;}
            else {ct=e-2;for(int k=0;k<3;++k)w[k]=(x[2][k]+ct*b[k])-x[0][k];cs=A>0?clamp(dot(w,a)/A):0;}
            for(int k=0;k<3;++k)w[k]=(x[0][k]+cs*a[k])-(x[2][k]+ct*b[k]);
            const double d=norm(w);if(d<best){best=d;s=cs;t=ct;}
        }
    }
    f.w[0]=1-s;f.w[1]=s;f.w[2]=t-1;f.w[3]=-t;
    if(s>1e-14&&s<1-1e-14){const int j=f.free_parameters++;f.sigma[j][0]=-1;f.sigma[j][1]=1;for(int k=0;k<3;++k)f.u[j][k]=a[k];}
    if(t>1e-14&&t<1-1e-14){const int j=f.free_parameters++;f.sigma[j][2]=1;f.sigma[j][3]=-1;for(int k=0;k<3;++k)f.u[j][k]=-b[k];}
    for(int k=0;k<3;++k)f.residual[k]=(x[0][k]+s*a[k])-(x[2][k]+t*b[k]);
    f.distance=norm(f.residual);return f;
}

struct Geometry {
    double normal[3]{},projections[4]{},negative=0,positive=0,distance=0;
    int split=0;
    bool near=false,valid=false;
};
IPC_DAT_HD inline Geometry geometry(const double x[4][3],bool vertex_face,double rq) {
    using namespace resident_ccd::detail;
    Geometry g;g.split=vertex_face?1:2;
    const auto f=contact_feature(x,vertex_face);g.distance=contact_detail::finite(f.distance)&&f.distance>0?f.distance:0;
    g.near=f.distance<=rq;
    const double length=norm3(f.residual,0);
    if(!(g.distance>0)||!(length>0)||!contact_detail::finite(length))return g;
    for(int k=0;k<3;++k)g.normal[k]=f.residual[k]/length;
    g.positive=INFINITY;g.negative=-INFINITY;double scale=0;
    Interval projections[4];
    for(int i=0;i<4;++i) {
        auto value=point(0);
        for(int k=0;k<3;++k){const auto r=subtract(point(x[i][k]),point(x[0][k]));value=add(value,multiply(point(g.normal[k]),r));scale=fmax(scale,fmax(fabs(r.lo),fabs(r.hi)));}
        projections[i]=value;
    }
    const double margin=32*DBL_EPSILON*scale;
    for(int i=0;i<4;++i) {
        const bool positive=i<g.split;
        g.projections[i]=positive?subtract(projections[i],point(margin)).lo:add(projections[i],point(margin)).hi;
        if(!contact_detail::finite(g.projections[i]))return g;
        if(positive)g.positive=fmin(g.positive,g.projections[i]);else g.negative=fmax(g.negative,g.projections[i]);
    }
    const double gap=g.positive-g.negative;g.valid=contact_detail::finite(gap)&&gap>0;return g;
}
IPC_DAT_HD inline double plane(const Geometry& g,const double q[4][3]) {
    if(!g.valid)return 0;
    double ap=0,an=0;
    for(int i=0;i<4;++i){const double p=dot(g.normal,q[i]);if(i<g.split)ap=fmax(ap,-p);else an=fmax(an,p);}
    const double largest=fmax(ap,an);double lambda=.5;
    if(largest>0){const double p=ap/largest,n=an/largest;lambda=n/(p+n);}
    // Printed Eqs. (11)-(12), using the positive primitive as plane origin.
    lambda=fmax(1e-8,fmin(1-1e-8,lambda));return g.positive-lambda*(g.positive-g.negative);
}
IPC_DAT_HD inline double pair_scale(const Geometry& g,int role,const double* q,double c,double gamma) {
    if(!g.near)return 1;
    if(!g.valid){const double length=norm3(q,0);return length>0?fmin(1.0,.5*gamma*g.distance/length):1;}
    const bool positive=role<g.split;
    const double approach=fmax(0.0,positive?-dot(g.normal,q):dot(g.normal,q));
    const double clearance=fmax(0.0,positive?g.projections[role]-c:c-g.projections[role]);
    return approach>0?fmin(1.0,gamma*clearance/approach):1;
}
IPC_DAT_HD inline bool within_radius(const double* base,const double* endpoint,double radius) {
    using namespace resident_ccd::detail;
    Interval squared=point(0);
    for(int k=0;k<3;++k){const auto d=subtract(point(endpoint[k]),point(base[k]));squared=add(squared,multiply(d,d));}
    return sqrt_upper(fmax(0.0,squared.hi))<=radius;
}
IPC_DAT_HD inline bool accepts(const Geometry& g,int role,double c,const double* base,const double* endpoint,double gamma) {
    if(!g.near)return true;
    if(!g.valid)return within_radius(base,endpoint,.5*gamma*g.distance);
    using namespace resident_ccd::detail;
    Interval shift=point(0);
    for(int k=0;k<3;++k)shift=add(shift,multiply(point(g.normal[k]),subtract(point(endpoint[k]),point(base[k]))));
    const bool positive=role<g.split;
    const double initial=positive?g.projections[role]-c:c-g.projections[role];
    const auto final=positive?add(point(initial),shift):subtract(point(initial),shift);
    return initial>0?final.lo>0:initial==0&&final.lo>=0;
}

// Same full Schur-complement barrier and frozen-friction derivatives as the
// resident contact evaluator, with OGC/DAT's corrected boundary features.
IPC_DAT_HD inline ContactOutput evaluate(const ContactInput& in) {
    using namespace contact_detail;
    ContactOutput out{};
    if(in.k_barrier==0)return out;
    const Feature feature=contact_feature(in.positions,in.kind==ContactKind::PointTriangle);
    if(feature.distance>0)for(int k=0;k<3;++k)out.normal[k]=feature.residual[k]/feature.distance;
    out.distance=feature.distance;
    for(int i=0;i<4;++i)out.weights[i]=feature.w[i];
    if(!contact_detail::finite(out.distance)){out.status=ContactStatus::NeedsExactGeometry;return out;}
    if(out.distance>=in.d_hat)return out;
    if(out.distance==0){out.status=ContactStatus::ZeroDistance;return out;}
    const double d=out.distance,s=d-in.d_hat,logratio=log(d/in.d_hat),ratio=in.d_hat/d;
    const double bp=-2*s*logratio-s*s/d,bpp=ratio*ratio+2*ratio-3-2*logratio;
    Feature hessian_feature=feature;
    double hessian_normal[3];for(int k=0;k<3;++k)hessian_normal[k]=out.normal[k];
    if(feature.edge_a>=0) {
        // Match the CPU edge Hessian's unbounded projection parameter. The
        // corrected feature search routes clamped endpoints to point features.
        const int a=feature.edge_a,b=feature.edge_b;double edge[3],r[3];sub(in.positions[b],in.positions[a],edge);sub(in.positions[0],in.positions[a],r);
        const double t=dot(r,edge)/dot(edge,edge);
        hessian_feature=Feature{};hessian_feature.w[0]=1;hessian_feature.w[a]=t-1;hessian_feature.w[b]=-t;
        hessian_feature.free_parameters=1;hessian_feature.sigma[0][a]=1;hessian_feature.sigma[0][b]=-1;
        for(int k=0;k<3;++k){hessian_feature.u[0][k]=-edge[k];hessian_feature.residual[k]=in.positions[0][k]-(in.positions[a][k]+t*edge[k]);hessian_normal[k]=hessian_feature.residual[k]/d;}
    }
    const auto& hf=hessian_feature;
    const int n=hf.free_parameters;
    double inverse[2][2]{};
    if(n==1) inverse[0][0]=1/dot(hf.u[0],hf.u[0]);
    if(n==2) {
        const double A=dot(hf.u[0],hf.u[0]),B=dot(hf.u[0],hf.u[1]),C=dot(hf.u[1],hf.u[1]);
        const double determinant=A*C-B*B;
        if(!(determinant>0)) {out.status=ContactStatus::NeedsExactGeometry;return out;}
        inverse[0][0]=C/determinant;inverse[0][1]=inverse[1][0]=-B/determinant;inverse[1][1]=A/determinant;
    }
    // Differentiate q=0.5*||sum_i w_i*x_i||^2 after minimizing over the
    // active feature's free parameters. The Schur complement is the full
    // Hessian of minimized q, NOT frozen-weight/Gauss-Newton differentiation.
    double q_xt[4][3][2]{};
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)for(int a=0;a<n;++a)
        q_xt[i][k][a]=hf.w[i]*hf.u[a][k]+hf.sigma[a][i]*hf.residual[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const double gi=hf.w[i]*hessian_normal[k];
        out.barrier_gradient[i][k]=bp*feature.w[i]*out.normal[k];
        for(int j=0;j<4;++j)if(in.hessian_blocks&(1u<<(4*i+j)))for(int l=0;l<3;++l) {
            const double gj=hf.w[j]*hessian_normal[l];
            double hq=(k==l?hf.w[i]*hf.w[j]:0);
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
        if(!contact_detail::finite(out.barrier_gradient[i][k])||!contact_detail::finite(out.friction_relative_gradient[k]))out.status=ContactStatus::NonFiniteDerivative;
        for(int j=0;j<4;++j)for(int l=0;l<3;++l)
            if(!contact_detail::finite(out.barrier_hessian[i][j][3*k+l])||!contact_detail::finite(out.friction_relative_hessian[3*k+l]))out.status=ContactStatus::NonFiniteDerivative;
    }
    out.active=out.status==ContactStatus::Ok;
    return out;
}
} // namespace ipc_gpu::resident::dat
#undef IPC_DAT_HD

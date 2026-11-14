#pragma once

// DAT-only IPC specialization.  The energy, minimized-distance Schur
// curvature and lagged-friction law are unchanged; only diagonal Cartesian
// blocks are materialized, directly into float32 force/Hessian storage.
// Status 6 requests the separately compiled ordinary-double evaluator.  No
// uncertain float feature is ever silently accepted as an inactive contact.
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_DAT_FAST_HD __host__ __device__
#else
#define IPC_DAT_FAST_HD
#endif

namespace ipc_gpu { namespace dat_contact_fast {

struct Input {
    float x[4][3]{}, previous[4][3]{};
    float d_hat, stiffness, dt, friction, eps_v;
    int kind;
};
struct Feature {
    float w[4]{}, u[2][3]{}, sigma[2][4]{}, r[3]{}, normal[3]{};
    int free=0;
};
IPC_DAT_FAST_HD inline float absval(float x) { return x<0.0f?-x:x; }
IPC_DAT_FAST_HD inline bool finite(float x) { return x<=FLT_MAX&&x>=-FLT_MAX; }
IPC_DAT_FAST_HD inline float maximum(float x,float y) { return x>y?x:y; }
IPC_DAT_FAST_HD inline float clamp(float x) { return x<0.0f?0.0f:(x>1.0f?1.0f:x); }
IPC_DAT_FAST_HD inline float dot(const float* x,const float* y) {
    return (x[0]*y[0]+x[1]*y[1])+x[2]*y[2];
}
IPC_DAT_FAST_HD inline void sub(const float* x,const float* y,float* z) {
    for(int k=0;k<3;++k)z[k]=x[k]-y[k];
}
IPC_DAT_FAST_HD inline void cross(const float* x,const float* y,float* z) {
    z[0]=x[1]*y[2]-x[2]*y[1];z[1]=x[2]*y[0]-x[0]*y[2];z[2]=x[0]*y[1]-x[1]*y[0];
}
IPC_DAT_FAST_HD inline float length(const float* x) { return sqrtf(dot(x,x)); }
// A scalar triple product determines the signed gap of a face/interior EE
// pair. Its products nearly cancel as the gap shrinks, so retain binary64
// here alone. Feature storage, normalization and all force/Hessian arithmetic
// remain float32; no double derivative packet or matrix is materialized.
IPC_DAT_FAST_HD inline float signed_gap(const float* w0,const float* w1,
    const float* a0,const float* a1,const float* b0,const float* b1,float inverse_area) {
    double w[3],a[3],b[3];
    for(int k=0;k<3;++k) {
        w[k]=static_cast<double>(w0[k])-w1[k];
        a[k]=static_cast<double>(a0[k])-a1[k];
        b[k]=static_cast<double>(b0[k])-b1[k];
    }
    const double n0=a[1]*b[2]-a[2]*b[1],n1=a[2]*b[0]-a[0]*b[2],n2=a[0]*b[1]-a[1]*b[0];
    return static_cast<float>(((w[0]*n0+w[1]*n1)+w[2]*n2)*inverse_area);
}
IPC_DAT_FAST_HD inline bool near_boundary(float t,float tolerance) {
    return absval(t)<=tolerance||absval(t-1.0f)<=tolerance;
}

// Conservative coordinate-gap rejection.  Coordinates and d_hat are represented
// float32 inputs; the extra error allowance covers the rounded subtraction.
IPC_DAT_FAST_HD inline bool inactive_box(const Input& in) {
    const int split=in.kind==0?1:2;
    for(int k=0;k<3;++k) {
        float lo0=in.x[0][k],hi0=lo0,lo1=in.x[split][k],hi1=lo1;
        for(int i=1;i<split;++i) {lo0=in.x[i][k]<lo0?in.x[i][k]:lo0;hi0=maximum(hi0,in.x[i][k]);}
        for(int i=split+1;i<4;++i) {lo1=in.x[i][k]<lo1?in.x[i][k]:lo1;hi1=maximum(hi1,in.x[i][k]);}
        const float allowance=4.0f*FLT_EPSILON*(absval(lo0)+absval(hi0)+absval(lo1)+absval(hi1)+in.d_hat)+FLT_MIN;
        if(lo0-hi1>in.d_hat+allowance||lo1-hi0>in.d_hat+allowance)return true;
    }
    return false;
}

IPC_DAT_FAST_HD inline void point_edge(const Input& in,int a,int b,float t,Feature& f) {
    f=Feature{};f.w[0]=1.0f;
    if(t<=0.0f) {f.w[a]=-1.0f;sub(in.x[0],in.x[a],f.r);return;}
    if(t>=1.0f) {f.w[b]=-1.0f;sub(in.x[0],in.x[b],f.r);return;}
    f.w[a]=t-1.0f;f.w[b]=-t;f.free=1;f.sigma[0][a]=1.0f;f.sigma[0][b]=-1.0f;
    for(int k=0;k<3;++k) {
        f.u[0][k]=in.x[a][k]-in.x[b][k];
        f.r[k]=(in.x[0][k]-in.x[a][k])+t*f.u[0][k];
    }
}

// Return 0 = feature, 1 = certified inactive, 6 = ordinary-double replay.
IPC_DAT_FAST_HD inline int point_triangle(const Input& in,Feature& f,float& scale) {
    float a[3],b[3],w[3],normal[3];
    sub(in.x[2],in.x[1],a);sub(in.x[3],in.x[1],b);sub(in.x[0],in.x[1],w);cross(a,b,normal);
    const float A=dot(a,a),C=dot(b,b),area2=dot(normal,normal);
    scale=sqrtf(maximum(maximum(A,C),dot(w,w)));
    if(!finite(area2)||!(area2>0.000244140625f*A*C)||!(area2>1.0e-24f))return 6;
    const float conditioning=A*C/area2;
    scale*=conditioning;
    const float inv_area=1.0f/sqrtf(area2);
    for(int k=0;k<3;++k)normal[k]*=inv_area;
    const float phi=signed_gap(in.x[0],in.x[1],in.x[2],in.x[1],in.x[3],in.x[1],inv_area);
    const float error=64.0f*FLT_EPSILON*scale;
    if(absval(phi)>in.d_hat+error)return 1;
    float wb[3],aw[3];cross(w,b,wb);cross(a,w,aw);
    const float l2=dot(wb,normal)*inv_area,l3=dot(aw,normal)*inv_area,l1=1.0f-l2-l3;
    const float tolerance=8192.0f*FLT_EPSILON*conditioning;
    if(!finite(l1)||!finite(l2)||!finite(l3)||absval(l1)<=tolerance||absval(l2)<=tolerance||absval(l3)<=tolerance)return 6;
    if(l1>0.0f&&l2>0.0f&&l3>0.0f) {
        f.w[0]=1.0f;f.w[1]=-l1;f.w[2]=-l2;f.w[3]=-l3;f.free=2;
        f.sigma[0][1]=1.0f;f.sigma[0][2]=-1.0f;f.sigma[1][1]=1.0f;f.sigma[1][3]=-1.0f;
        for(int k=0;k<3;++k) {f.u[0][k]=-a[k];f.u[1][k]=-b[k];f.r[k]=phi*normal[k];}
        return 0;
    }
    float best=FLT_MAX;
    for(int ia=1;ia<4;++ia) {
        const int ib=ia==3?1:ia+1;
        float edge[3],delta[3];sub(in.x[ib],in.x[ia],edge);sub(in.x[0],in.x[ia],delta);
        const float e2=dot(edge,edge);
        if(!(e2>FLT_MIN))return 6;
        const float raw=dot(delta,edge)/e2;
        if(near_boundary(raw,tolerance))return 6;
        Feature candidate;point_edge(in,ia,ib,clamp(raw),candidate);
        const float distance2=dot(candidate.r,candidate.r);
        if(!finite(distance2))return 6;
        if(best<FLT_MAX&&absval(distance2-best)<=32.0f*FLT_EPSILON*(distance2+best)) {
            bool same=true;for(int i=0;i<4;++i)same=same&&candidate.w[i]==f.w[i];
            if(!same)return 6;
        }
        if(distance2<best) {best=distance2;f=candidate;}
    }
    return 0;
}

IPC_DAT_FAST_HD inline int edge_edge(const Input& in,Feature& f,float& scale) {
    float a[3],b[3],c[3];sub(in.x[1],in.x[0],a);sub(in.x[3],in.x[2],b);sub(in.x[0],in.x[2],c);
    const float A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,c),E=dot(b,c);
    const float determinant=A*C-B*B;
    scale=sqrtf(maximum(maximum(A,C),dot(c,c)));
    if(!finite(determinant)||!(determinant>0.000244140625f*A*C)||!(determinant>1.0e-24f))return 6;
    const float conditioning=A*C/determinant;
    scale*=conditioning;
    const float inv=1.0f/determinant;
    float s=(B*E-C*D)*inv,t=(A*E-B*D)*inv;
    const float tolerance=8192.0f*FLT_EPSILON*conditioning;
    if(!finite(s)||!finite(t)||near_boundary(s,tolerance)||near_boundary(t,tolerance))return 6;
    const bool interior=s>0.0f&&s<1.0f&&t>0.0f&&t<1.0f;
    if(!interior) {
        const float ss[4]={0.0f,1.0f,clamp(-D/A),clamp((B-D)/A)};
        const float tt[4]={clamp(E/C),clamp((E+B)/C),0.0f,1.0f};
        float best=FLT_MAX;
        for(int i=0;i<4;++i) {
            float r[3];for(int k=0;k<3;++k)r[k]=(c[k]+ss[i]*a[k])-tt[i]*b[k];
            const float distance2=dot(r,r);
            if(!finite(distance2))return 6;
            if(best<FLT_MAX&&absval(distance2-best)<=32.0f*FLT_EPSILON*(distance2+best)
                &&(s!=ss[i]||t!=tt[i]))return 6;
            if(distance2<best) {best=distance2;s=ss[i];t=tt[i];}
        }
        // A selected interior boundary parameter must be separated from its
        // endpoints; exact clamped endpoints retain the original free set.
        if((s>0.0f&&s<1.0f&&near_boundary(s,tolerance))||(t>0.0f&&t<1.0f&&near_boundary(t,tolerance)))return 6;
    }
    f.w[0]=1.0f-s;f.w[1]=s;f.w[2]=t-1.0f;f.w[3]=-t;
    if(interior||(s>0.0f&&s<1.0f)) {
        const int j=f.free++;f.sigma[j][0]=-1.0f;f.sigma[j][1]=1.0f;
        for(int k=0;k<3;++k)f.u[j][k]=a[k];
    }
    if(interior||(t>0.0f&&t<1.0f)) {
        const int j=f.free++;f.sigma[j][2]=1.0f;f.sigma[j][3]=-1.0f;
        for(int k=0;k<3;++k)f.u[j][k]=-b[k];
    }
    if(interior) {
        // The residual is normal to both free edges. Constructing it as the
        // difference of rounded closest points introduces an O(edge length)
        // tangential error that dominates tiny gaps in float32. This identical
        // plane projection keeps that cancellation out of the normal/curvature.
        float normal[3];cross(a,b,normal);
        const float inv_length=1.0f/length(normal);
        for(int k=0;k<3;++k)normal[k]*=inv_length;
        const float phi=signed_gap(in.x[0],in.x[2],in.x[1],in.x[0],in.x[3],in.x[2],inv_length);
        for(int k=0;k<3;++k)f.r[k]=phi*normal[k];
    } else {
        for(int k=0;k<3;++k)f.r[k]=(c[k]+s*a[k])-t*b[k];
    }
    return 0;
}

IPC_DAT_FAST_HD inline int evaluate(const Input& in,float* packet) {
    if(inactive_box(in))return 0;
    Feature feature;float scale=0.0f;
    const int status=in.kind==0?point_triangle(in,feature,scale):edge_edge(in,feature,scale);
    if(status==1)return 0;
    if(status)return status;
    const float d=length(feature.r);
    if(!finite(d))return 6;
    // Geometry error is measured in the local primitive frame. Extreme world
    // translations and feature transitions are handled by the replay as well.
    const float guard=256.0f*FLT_EPSILON*scale+FLT_MIN;
    if(absval(d-in.d_hat)<=guard||d<=guard)return 6;
    if(d>=in.d_hat)return 0;
    for(int k=0;k<3;++k)feature.normal[k]=feature.r[k]/d;
    const float u=(in.d_hat-d)/d;
#if defined(__CUDA_ARCH__)
    const float logarithm=log1pf(u);
#else
    const float logarithm=static_cast<float>(log(1.0+static_cast<double>(u)));
#endif
    const float bp=-d*u*(2.0f*logarithm+u),bpp=u*(u+4.0f)+2.0f*logarithm;
    if(!finite(bp)||!finite(bpp)||absval(bp)<FLT_MIN
        ||absval(in.stiffness*bp)<FLT_MIN)return 6;
    float i00=0.0f,i01=0.0f,i11=0.0f;
    if(feature.free==1) {i00=1.0f/dot(feature.u[0],feature.u[0]);}
    if(feature.free==2) {
        const float A=dot(feature.u[0],feature.u[0]),B=dot(feature.u[0],feature.u[1]),C=dot(feature.u[1],feature.u[1]);
        const float determinant=A*C-B*B;
        if(!(determinant>0.000244140625f*A*C))return 6;
        i00=C/determinant;i01=-B/determinant;i11=A/determinant;
    }
    float displacement[3]{},slip_vector[3]{},friction_scale=0.0f;
    if(in.friction>0.0f) {
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            displacement[k]+=feature.w[i]*(in.x[i][k]-in.previous[i][k]);
        const float normal_displacement=dot(feature.normal,displacement);
        for(int k=0;k<3;++k)slip_vector[k]=displacement[k]-feature.normal[k]*normal_displacement;
        const float slip=length(slip_vector),eps_u=in.dt*in.eps_v;
        if(!finite(slip)||!finite(eps_u)||eps_u<FLT_MIN)return 6;
        // Equivalent smoothing law without squaring eps_u: its square can
        // overflow/underflow although the represented inputs and result fit.
        const float mollifier=slip<eps_u?(2.0f-slip/eps_u)/eps_u:1.0f/slip;
        friction_scale=in.friction*(-in.stiffness*bp)*mollifier;
        if(!finite(friction_scale)||(friction_scale==0.0f&&slip>0.0f))return 6;
    }
    // Exact expansion of the minimized-distance Schur complement.  With
    // r=d*n and q_xt_i=w_i*U+r*sigma_i^T, its role-dependent part is
    // -bp*w_i*(a_i*n^T+n*a_i^T)-bp*d*c_i*n*n^T.  The common matrix is
    // formed once, including the full normal/tangential geometric curvature.
    float normal_outer[9],common[9],dual0[3]{},dual1[3]{};
    for(int k=0;k<3;++k) {
        if(feature.free>0)dual0[k]=i00*feature.u[0][k];
        if(feature.free>1) {
            dual0[k]+=i01*feature.u[1][k];
            dual1[k]=i01*feature.u[0][k]+i11*feature.u[1][k];
        }
    }
    // I-U*(U^T*U)^-1*U^T-n*n^T is the projection onto the complement
    // of the active tangent space and the normal.  Its rank is 2-free.
    // These exact identities avoid cancellation in subtracting projectors.
    float perpendicular[3]{};
    if(feature.free==1)cross(feature.u[0],feature.normal,perpendicular);
    for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
        const int at=3*k+l;
        const float n=feature.normal[k]*feature.normal[l];
        normal_outer[at]=n;
        float geometric_projection=0.0f;
        if(feature.free==0)geometric_projection=(k==l?1.0f:0.0f)-n;
        if(feature.free==1)geometric_projection=perpendicular[k]*perpendicular[l]*i00;
        common[at]=in.stiffness*(bpp*n+(bp/d)*geometric_projection)
            +friction_scale*((k==l?1.0f:0.0f)-n);
    }
    const float scaled_bp=in.stiffness*bp;
    for(int role=0;role<4;++role) {
        const float weight=feature.w[role],weight2=weight*weight;
        float a[3]{};float c=0.0f;
        if(feature.free>0) {
            const float s0=feature.sigma[0][role];
            c=s0*s0*i00;
            for(int k=0;k<3;++k)a[k]=s0*dual0[k];
        }
        if(feature.free>1) {
            const float s0=feature.sigma[0][role],s1=feature.sigma[1][role];
            c+=2.0f*s0*s1*i01+s1*s1*i11;
            for(int k=0;k<3;++k)a[k]+=s1*dual1[k];
        }
        const float mixed_scale=scaled_bp*weight,normal_scale=scaled_bp*d*c;
        for(int k=0;k<3;++k) {
            packet[12*role+k]=-(scaled_bp*weight*feature.normal[k]+weight*friction_scale*slip_vector[k]);
            for(int l=0;l<3;++l) {
                const int at=3*k+l;
                packet[12*role+3+at]=weight2*common[at]
                    -mixed_scale*(a[k]*feature.normal[l]+feature.normal[k]*a[l])
                    -normal_scale*normal_outer[at];
            }
        }
    }
    for(int k=0;k<48;++k)if(!finite(packet[k]))return 6;
    packet[49]=1.0f;
    return 0;
}

}} // namespace ipc_gpu::dat_contact_fast
#undef IPC_DAT_FAST_HD

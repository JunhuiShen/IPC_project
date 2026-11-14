#pragma once

// Compact physical SDF derivatives. Signed distances and union selection use
// binary64; the selected collider's force and complete Hessian use float32.
// Status 6 asks the independent legacy evaluator to replay an unsafe range.
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_SDF_FAST_HD __host__ __device__
#else
#define IPC_SDF_FAST_HD
#endif

namespace ipc_gpu { namespace sdf_fast {

IPC_SDF_FAST_HD inline bool finite(double x) { return x<=DBL_MAX&&x>=-DBL_MAX; }
IPC_SDF_FAST_HD inline bool finite32(float x) { return x<=FLT_MAX&&x>=-FLT_MAX; }
IPC_SDF_FAST_HD inline float absolute(float x) {return x<0?-x:x;}
IPC_SDF_FAST_HD inline bool normal_scalar(double x,float y,bool positive=false) {
    return finite32(y)&&(!positive||y>0)&&(!(x!=0)||absolute(y)>=FLT_MIN);
}
IPC_SDF_FAST_HD inline bool same_pose(const double* c) {
    for(int k=0;k<12;++k)if(c[8+k]!=c[20+k])return false;
    return true;
}
IPC_SDF_FAST_HD inline bool valid_pose(const double* c,int offset) {
    const double* r=c+offset;
    for(int k=0;k<12;++k)if(!finite(r[k]))return false;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        const double value=(r[i]*r[j]+r[3+i]*r[3+j])+r[6+i]*r[6+j];
        if(fabs(value-(i==j?1.0:0.0))>1e-10)return false;
    }
    const double determinant=r[0]*(r[4]*r[8]-r[5]*r[7])
        -r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    return fabs(determinant-1.0)<=1e-10;
}

struct Input {
    float x[3],previous[3];
    double stiffness,epsilon,dt,friction,eps_v;
    int include_curvature;
    // Set only by the backend entry point after prepare_colliders has
    // validated/canonicalized the bank and latched any error in its context.
    int prepared_poses=0;
};

// All collider records are validated, including those that lose union-min
// selection. The strict comparison retains the first record on exact ties.
IPC_SDF_FAST_HD inline int evaluate(const Input& in,const double* bank,int stride,
                                   int count,float* packet) {
    if(count<0||!finite(in.stiffness)||in.stiffness<0||!finite(in.epsilon)
        ||!finite(in.dt)||in.dt<=0||!finite(in.friction)||in.friction<0
        ||!finite(in.eps_v)||in.eps_v<=0)return 1;
    for(int k=0;k<3;++k)if(!finite(in.x[k]))return 1;
    if(count>0&&in.friction>0)
        for(int k=0;k<3;++k)if(!finite(in.previous[k]))return 1;
    if(count==0)return 0;
    int selected=-1;
    double phi=0,radial[3]{},distance=0;
    for(int index=0;index<count;++index) {
        const double* c=bank+stride*index;
        if(!(c[0]==0||c[0]==1||c[0]==2))return 1;
        double r[3];
        for(int k=0;k<3;++k) {
            if(!finite(c[1+k])||!finite(c[4+k]))return 1;
            r[k]=static_cast<double>(in.x[k])-c[1+k];
        }
        double candidate=0,candidate_distance=0;
        if(c[0]==0)candidate=(r[0]*c[4]+r[1]*c[5])+r[2]*c[6];
        else {
            if(!finite(c[7])||c[7]<0)return 1;
            if(c[0]==1) {
                const double axial=(c[4]*r[0]+c[5]*r[1])+c[6]*r[2];
                for(int k=0;k<3;++k)r[k]-=c[4+k]*axial;
            }
            candidate_distance=sqrt((r[0]*r[0]+r[1]*r[1])+r[2]*r[2]);
            candidate=candidate_distance-c[7];
        }
        if(!finite(candidate))return 1;
        if(selected<0||candidate<phi) {
            selected=index;phi=candidate;distance=candidate_distance;
            for(int k=0;k<3;++k)radial[k]=r[k];
        }
    }
    packet[13]=static_cast<float>(selected);
    if(in.stiffness==0)return 0;
    if((in.epsilon<=0&&phi>=0)||(in.epsilon>0&&phi>=in.epsilon))return 0;
    const float k=static_cast<float>(in.stiffness),e=static_cast<float>(in.epsilon);
    const float p=static_cast<float>(phi),mu=static_cast<float>(in.friction);
    const float dt=static_cast<float>(in.dt),eps_v=static_cast<float>(in.eps_v);
    if(!normal_scalar(in.stiffness,k,true)||!normal_scalar(in.epsilon,e)
        ||!normal_scalar(phi,p)||!normal_scalar(in.dt,dt,true)
        ||!normal_scalar(in.friction,mu)||!normal_scalar(in.eps_v,eps_v,true))return 6;
    const double* c=bank+stride*selected;
    float gradient[3]{},curvature[9]{},axis[3]{};
    double gradient64[3]{};
    for(int a=0;a<3;++a) {
        axis[a]=static_cast<float>(c[4+a]);
        if(!finite32(axis[a]))return 6;
    }
    if(c[0]==0) {
        for(int a=0;a<3;++a){gradient[a]=axis[a];gradient64[a]=c[4+a];}
    } else if(distance>0) {
        const float d=static_cast<float>(distance);
        if(!normal_scalar(distance,d,true))return 6;
        // The analytic cylinder Hessian presumes a unit axis. Exotic legacy
        // metadata is preserved by replay rather than silently normalized.
        if(c[0]==1) {
            const double a2=(c[4]*c[4]+c[5]*c[5])+c[6]*c[6];
            if(fabs(a2-1.0)>2e-6)return 6;
        }
        for(int a=0;a<3;++a) {
            gradient64[a]=radial[a]/distance;
            gradient[a]=static_cast<float>(gradient64[a]);
        }
        if(in.include_curvature)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
            curvature[3*a+b]=((a==b?1.0f:0.0f)
                -(c[0]==1?axis[a]*axis[b]:0.0f)-gradient[a]*gradient[b])/d;
    }
    float first=0,second=0;
    if(in.epsilon<=0) {first=k*p;second=k;}
    else {
        const float delta=e-p;
        if(!(delta>4096.0f*FLT_EPSILON*(absolute(e)+absolute(p))))return 6;
        // Classify the piecewise law using the accurate signed distance. The
        // legacy derivative has a distinct value exactly at phi==0.
        if(phi<=0) {first=-k*delta;second=k;}
        else {first=-1.5f*k*delta*(delta/e);second=3.0f*k*(delta/e);}
    }
    // A tiny coefficient may multiply a very large geometric curvature to
    // produce a representable Hessian entry. Replay before a float32
    // underflow or subnormal rounding can erase that full-curvature term.
    if(!finite32(first)||!finite32(second)||absolute(first)<FLT_MIN
        ||absolute(second)<FLT_MIN)return 6;
    float normal[3]{},slip[3]{},friction_scale=0;
    const float g2=(gradient[0]*gradient[0]+gradient[1]*gradient[1])+gradient[2]*gradient[2];
    // The legacy friction gate is a binary64 norm > 1e-12. A float32 norm
    // can cross that threshold for tiny, non-unit plane normals.
    if(mu>0&&g2>=0.5e-24f&&g2<=2.0e-24f)return 6;
    if(mu>0&&g2>1.0e-24f) {
        const float norm_gradient=sqrtf(g2);
        for(int a=0;a<3;++a)normal[a]=gradient[a]/norm_gradient;
        const float normal_force=-first*norm_gradient;
        if(normal_force>0) {
            float displacement[3];
            if(same_pose(c)) {
                for(int a=0;a<3;++a)displacement[a]=in.x[a]-in.previous[a];
            } else {
                if(!in.prepared_poses&&(!valid_pose(c,8)||!valid_pose(c,20)))return 2;
                const double g264=(gradient64[0]*gradient64[0]+gradient64[1]*gradient64[1])
                    +gradient64[2]*gradient64[2];
                double surface[3],offset[3],material[3];
                for(int a=0;a<3;++a) {
                    surface[a]=g264>0?static_cast<double>(in.x[a])-(phi/g264)*gradient64[a]:in.x[a];
                    offset[a]=surface[a]-c[29+a];
                }
                for(int a=0;a<3;++a)
                    material[a]=(c[20+a]*offset[0]+c[23+a]*offset[1])+c[26+a]*offset[2];
                for(int a=0;a<3;++a) {
                    const double old_surface=((c[8+3*a]*material[0]+c[9+3*a]*material[1])
                        +c[10+3*a]*material[2])+c[17+a];
                    if(!finite(old_surface))return 3;
                    displacement[a]=static_cast<float>((static_cast<double>(in.x[a])-in.previous[a])
                        -(surface[a]-old_surface));
                }
            }
            const float normal_move=(normal[0]*displacement[0]+normal[1]*displacement[1])
                +normal[2]*displacement[2];
            for(int a=0;a<3;++a)slip[a]=displacement[a]-normal[a]*normal_move;
            const float length=sqrtf((slip[0]*slip[0]+slip[1]*slip[1])+slip[2]*slip[2]);
            const float eps_u=dt*eps_v;
            if(!finite32(length)||!finite32(eps_u)||eps_u<FLT_MIN)return 6;
            const float mollifier=length<eps_u?(2.0f-length/eps_u)/eps_u:1.0f/length;
            friction_scale=mu*normal_force*mollifier;
            if(!finite32(friction_scale)||(friction_scale==0&&length>0))return 6;
        }
    }
    for(int a=0;a<3;++a) {
        packet[a]=-(first*gradient[a]+friction_scale*slip[a]);
        for(int b=0;b<3;++b)packet[3+3*a+b]=second*gradient[a]*gradient[b]
            +(in.include_curvature?first*curvature[3*a+b]:0.0f)
            +friction_scale*((a==b?1.0f:0.0f)-normal[a]*normal[b]);
    }
    for(int a=0;a<12;++a)if(!finite32(packet[a]))return 6;
    return 0;
}

}} // namespace ipc_gpu::sdf_fast
#undef IPC_SDF_FAST_HD

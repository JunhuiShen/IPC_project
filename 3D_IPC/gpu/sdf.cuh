#pragma once

// GPU-GPU analytic obstacle geometry, penalty, and frozen friction.
// Collider material poses are inputs, but geometry selection and every force,
// Hessian, tangent frame, and previous-material-point calculation run here.

// Plain SDF records shared by host API and device evaluators.
namespace ipc_gpu {
namespace solver {

enum class SdfKind : unsigned { Plane, Cylinder, Sphere };
enum class SdfStatus : unsigned { Ok, InvalidInput, InvalidMaterialPose, NonFiniteDerivative };

struct SdfPose {
    double rotation[9]{1,0,0,0,1,0,0,0,1}; // row-major material->world
    double translation[3]{};
};
struct SdfCollider {
    double point[3]{}; // plane/cylinder point or sphere center
    double axis[3]{0,1,0}; // plane normal or infinite-cylinder axis (unit)
    double radius=0;
    SdfPose previous_pose,current_pose;
    SdfKind kind=SdfKind::Plane;
};
struct SdfParameters {
    double stiffness=0,epsilon=0,dt=1.0/300.0,friction=0,eps_v=.001;
    // Deformable DAT requests full curvature; legacy/rigid callers may disable it.
    unsigned include_curvature=0;
};
struct SdfOutput {
    double phi=0,grad_phi[3]{},hess_phi[9]{},surface_point[3]{};
    double gradient[3]{},hessian[9]{}; // includes stiffness, NOT dt²
    double friction_gradient[3]{},friction_hessian[9]{}; // includes dt²
    int collider=-1;
    SdfStatus status=SdfStatus::Ok;
};

} // namespace solver
} // namespace ipc_gpu

#include "contact.cuh"
#include "rigid_types.h"

#if defined(__CUDACC__)
#define IPC_RES_SDF_HD __host__ __device__
#else
#define IPC_RES_SDF_HD
#endif

namespace ipc_gpu {
namespace solver {

namespace sdf_detail {
IPC_RES_SDF_HD inline bool same_pose(const SdfPose& a,const SdfPose& b) {
    for(int i=0;i<9;++i)if(a.rotation[i]!=b.rotation[i])return false;
    for(int i=0;i<3;++i)if(a.translation[i]!=b.translation[i])return false;
    return true;
}
IPC_RES_SDF_HD inline bool valid_pose(const SdfPose& pose) {
    using contact_detail::finite;
    for(int i=0;i<9;++i)if(!finite(pose.rotation[i]))return false;
    for(int i=0;i<3;++i)if(!finite(pose.translation[i]))return false;
    const auto& R=pose.rotation;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        const double entry=(R[i]*R[j]+R[3+i]*R[3+j])+R[6+i]*R[6+j];
        if(contact_detail::absval(entry-(i==j?1.0:0.0))>1e-10)return false;
    }
    const double determinant=R[0]*(R[4]*R[8]-R[5]*R[7])
        -R[1]*(R[3]*R[8]-R[5]*R[6])+R[2]*(R[3]*R[7]-R[4]*R[6]);
    return contact_detail::absval(determinant-1)<=1e-10;
}
IPC_RES_SDF_HD inline bool geometry(const SdfCollider& collider,const double* x,SdfOutput& out) {
    using namespace contact_detail;
    double r[3];
    for(int k=0;k<3;++k) {
        if(!finite(collider.point[k])||!finite(collider.axis[k]))return false;
        r[k]=x[k]-collider.point[k];
    }
    if(collider.kind==SdfKind::Plane) {
        out.phi=dot(r,collider.axis);
        for(int k=0;k<3;++k)out.grad_phi[k]=collider.axis[k];
    } else if(collider.kind==SdfKind::Cylinder||collider.kind==SdfKind::Sphere) {
        if(!finite(collider.radius)||collider.radius<0)return false;
        if(collider.kind==SdfKind::Cylinder) {
            const double axial=dot(collider.axis,r);
            for(int k=0;k<3;++k)r[k]-=collider.axis[k]*axial;
        }
        const double distance=norm(r);
        out.phi=distance-collider.radius;
        if(distance>0) {
            for(int k=0;k<3;++k)out.grad_phi[k]=r[k]/distance;
            for(int k=0;k<3;++k)for(int l=0;l<3;++l)
                out.hess_phi[3*k+l]=((k==l?1.0:0.0)
                    -(collider.kind==SdfKind::Cylinder?collider.axis[k]*collider.axis[l]:0.0)
                    -out.grad_phi[k]*out.grad_phi[l])/distance;
        }
    } else return false;
    const double g2=dot(out.grad_phi,out.grad_phi);
    for(int k=0;k<3;++k)out.surface_point[k]=g2>0&&finite(g2)?x[k]-(out.phi/g2)*out.grad_phi[k]:x[k];
    return finite(out.phi);
}

IPC_RES_SDF_HD inline SdfStatus previous_material_point(
    const SdfCollider& collider,const double* current,double* previous) {
    if(same_pose(collider.previous_pose,collider.current_pose)) {
        for(int k=0;k<3;++k)previous[k]=current[k];
        return SdfStatus::Ok;
    }
    if(!valid_pose(collider.previous_pose)||!valid_pose(collider.current_pose))return SdfStatus::InvalidMaterialPose;
    double offset[3],material[3];
    for(int k=0;k<3;++k)offset[k]=current[k]-collider.current_pose.translation[k];
    const auto& R=collider.current_pose.rotation;
    for(int k=0;k<3;++k)material[k]=(R[k]*offset[0]+R[3+k]*offset[1])+R[6+k]*offset[2];
    const auto& P=collider.previous_pose.rotation;
    for(int k=0;k<3;++k) {
        previous[k]=((P[3*k]*material[0]+P[3*k+1]*material[1])+P[3*k+2]*material[2])+collider.previous_pose.translation[k];
        if(!contact_detail::finite(previous[k]))return SdfStatus::NonFiniteDerivative;
    }
    return SdfStatus::Ok;
}
} // namespace sdf_detail

// Preserve CPU union-min ordering: caller uploads planes, then cylinders,
// then spheres. Equal phi keeps the first collider, including its material pose.
IPC_RES_SDF_HD inline SdfOutput evaluate_sdf_contact(const SdfCollider* colliders,
    int count,const double* position,const double* previous,const SdfParameters& parameters) {
    using namespace contact_detail;
    SdfOutput out;
    if(count<0||(count>0&&!colliders)||!position||!finite(parameters.stiffness)||parameters.stiffness<0
        ||!finite(parameters.epsilon)||!finite(parameters.dt)||parameters.dt<=0
        ||!finite(parameters.friction)||parameters.friction<0||!finite(parameters.eps_v)||parameters.eps_v<=0) {
        out.status=SdfStatus::InvalidInput;return out;
    }
    for(int k=0;k<3;++k)if(!finite(position[k])){out.status=SdfStatus::InvalidInput;return out;}
    // CPU requires previous positions only if friction is enabled AND there
    // is an obstacle; a zero-obstacle scene supports a null previous pointer.
    if(count>0&&parameters.friction>0) {
        if(!previous){out.status=SdfStatus::InvalidInput;return out;}
        for(int k=0;k<3;++k)if(!finite(previous[k])){out.status=SdfStatus::InvalidInput;return out;}
    }
    for(int i=0;i<count;++i) {
        SdfOutput candidate;
        if(!sdf_detail::geometry(colliders[i],position,candidate)){out.status=SdfStatus::InvalidInput;return out;}
        if(out.collider<0||candidate.phi<out.phi){out=candidate;out.collider=i;}
    }
    if(out.collider<0||parameters.stiffness==0)return out;
    double first=0,second=0;
    if(parameters.epsilon<=0) {
        if(out.phi>=0)return out;
        first=parameters.stiffness*out.phi;second=parameters.stiffness;
    } else {
        const double H=out.phi<0?1.0:(out.phi>parameters.epsilon?0.0:(parameters.epsilon-out.phi)/parameters.epsilon);
        if(H<=0)return out;
        const double Hp=out.phi<=0||out.phi>=parameters.epsilon?0.0:-1.0/parameters.epsilon;
        const double d=parameters.epsilon-out.phi;
        first=.5*parameters.stiffness*(Hp*d*d-2*H*d);
        second=.5*parameters.stiffness*(-4*Hp*d+2*H);
    }
    for(int k=0;k<3;++k) {
        out.gradient[k]=first*out.grad_phi[k];
        for(int l=0;l<3;++l) {
            out.hessian[3*k+l]=second*(out.grad_phi[k]*out.grad_phi[l]);
            if(parameters.include_curvature)out.hessian[3*k+l]+=first*out.hess_phi[3*k+l];
        }
    }
    const double norm_gradient=norm(out.grad_phi);
    if(parameters.friction>0&&norm_gradient>1e-12) {
        double n[3],P[9],previous_surface[3],displacement[3],u[3]{};
        for(int k=0;k<3;++k)n[k]=out.grad_phi[k]/norm_gradient;
        const double normal_force=-dot(out.gradient,n);
        if(normal_force>0) {
            out.status=sdf_detail::previous_material_point(colliders[out.collider],out.surface_point,previous_surface);
            if(out.status!=SdfStatus::Ok)return out;
            for(int k=0;k<3;++k)displacement[k]=(position[k]-previous[k])-(out.surface_point[k]-previous_surface[k]);
            for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
                P[3*k+l]=(k==l?1.0:0.0)-n[k]*n[l];
                u[k]+=P[3*k+l]*displacement[l];
            }
            const double slip=norm(u),eps_u=parameters.dt*parameters.eps_v;
            const double mollifier=slip<eps_u?2/eps_u-slip/(eps_u*eps_u):1/slip;
            const double scale=parameters.dt*parameters.dt*parameters.friction*normal_force*mollifier;
            for(int k=0;k<3;++k) {
                out.friction_gradient[k]=scale*u[k];
                for(int l=0;l<3;++l)out.friction_hessian[3*k+l]=scale*P[3*k+l];
            }
        }
    }
    for(int k=0;k<3;++k) {
        if(!finite(out.gradient[k])||!finite(out.friction_gradient[k]))out.status=SdfStatus::NonFiniteDerivative;
        for(int l=0;l<3;++l)if(!finite(out.hessian[3*k+l])||!finite(out.friction_hessian[3*k+l]))out.status=SdfStatus::NonFiniteDerivative;
    }
    return out;
}

// Solver-compatible rigid pullback: no SDF curvature and no orientation
// curvature in the normal/friction terms. Jacobian must be produced by the
// GPU rigid kinematics from that body's current orientation and local point.
// No frozen host-prepared contact data is required. Output is dt²-weighted.
IPC_RES_SDF_HD inline RigidDerivatives sdf_rigid_pullback(
    const SdfOutput& sdf,const double* jacobian,double dt2) {
    RigidDerivatives out{};
    double H[9],HJ[9];
    for(int k=0;k<3;++k) {
        out.translation_gradient[k]=dt2*sdf.gradient[k]+sdf.friction_gradient[k];
        for(int l=0;l<3;++l) {
            H[3*k+l]=dt2*sdf.hessian[3*k+l]+sdf.friction_hessian[3*k+l];
            out.translation_translation_hessian[3*k+l]=H[3*k+l];
        }
    }
    for(int k=0;k<3;++k) {
        out.orientation_gradient[k]=(jacobian[k]*out.translation_gradient[0]+jacobian[3+k]*out.translation_gradient[1])+jacobian[6+k]*out.translation_gradient[2];
        for(int l=0;l<3;++l) {
            HJ[3*k+l]=(H[3*k]*jacobian[l]+H[3*k+1]*jacobian[3+l])+H[3*k+2]*jacobian[6+l];
            out.translation_orientation_hessian[3*k+l]=HJ[3*k+l];
        }
    }
    for(int k=0;k<3;++k)for(int l=0;l<3;++l)
        out.orientation_orientation_hessian[3*k+l]=(jacobian[k]*HJ[l]+jacobian[3+k]*HJ[3+l])+jacobian[6+k]*HJ[6+l];
    return out;
}

} // namespace solver
} // namespace ipc_gpu
#undef IPC_RES_SDF_HD


// BEGIN COMPACT SDF ARITHMETIC
#ifndef IPC_SDF_COMPACT_ARITHMETIC
#define IPC_SDF_COMPACT_ARITHMETIC

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

#endif // IPC_SDF_COMPACT_ARITHMETIC

#pragma once

// GPU-resident analytic obstacle geometry, penalty, and frozen friction.
// Collider material poses are inputs, but geometry selection and every force,
// Hessian, tangent frame, and previous-material-point calculation run here.
#include "resident_sdf_types.h"
#include "resident_contact.cuh"
#include "rigid_types.h"

#if defined(__CUDACC__)
#define IPC_RES_SDF_HD __host__ __device__
#else
#define IPC_RES_SDF_HD
#endif

namespace ipc_gpu {
namespace resident {

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

} // namespace resident
} // namespace ipc_gpu
#undef IPC_RES_SDF_HD

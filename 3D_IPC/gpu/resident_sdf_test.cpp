#include "resident_sdf_test.h"
#include "sdf_penalty_energy.h"
#include "friction_energy.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ipc_gpu::resident::SdfCollider;
using ipc_gpu::resident::SdfKind;
using ipc_gpu::resident::SdfStatus;

namespace {
std::size_t components=0;
double maximum_error=0;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void compare(double actual,double expected,const std::string& label) {
    require(std::isfinite(actual)&&std::isfinite(expected),label+": nonfinite");
    const double error=std::abs(actual-expected)/(1+std::abs(expected));
    maximum_error=std::max(maximum_error,error);
    require(error<3e-10,label+": mismatch");++components;
}
SDFMaterialMotion motion(const SdfCollider& c) {
    SDFMaterialMotion result;
    for(int k=0;k<3;++k) {
        result.previous.translation[k]=c.previous_pose.translation[k];
        result.current.translation[k]=c.current_pose.translation[k];
        for(int l=0;l<3;++l){result.previous.rotation(k,l)=c.previous_pose.rotation[3*k+l];result.current.rotation(k,l)=c.current_pose.rotation[3*k+l];}
    }
    return result;
}
SDFEvaluation reference_geometry(const SdfCollider& c,const Vec3& x) {
    Vec3 point(c.point[0],c.point[1],c.point[2]),axis(c.axis[0],c.axis[1],c.axis[2]);
    if(c.kind==SdfKind::Plane)return evaluate_sdf(PlaneSDF{point,axis,motion(c)},x);
    if(c.kind==SdfKind::Cylinder)return evaluate_sdf(CylinderSDF{point,axis,c.radius,motion(c)},x);
    return evaluate_sdf(SphereSDF{point,c.radius,motion(c)},x);
}
std::vector<ResidentSdfTestInput> fixtures() {
    std::vector<ResidentSdfTestInput> records;
    const Vec3 center(.3,-.4,.2),axis=Vec3(.2,.9,-.3).normalized();
    const Vec3 radial=axis.unitOrthogonal();
    const Mat33 rotation=Eigen::AngleAxisd(.12,Vec3(.1,.3,.8).normalized()).toRotationMatrix();
    const Mat33 J=dx_domega(Vec3(.2,-.3,.4),quaternion_omega_kinematics(Vec4(1,0,0,0),Vec3(.4,-.3,.2),.031));
    for(int kind=0;kind<3;++kind)for(double phi:{-.1,0.0,.01,.04,.07})for(int variant=0;variant<12;++variant) {
        ResidentSdfTestInput in;
        auto& c=in.colliders[0];c.kind=SdfKind(kind);c.radius=.4;
        auto& p=in.parameters;p.stiffness=370;p.epsilon=variant%2?.05:0;p.dt=.031;
        p.friction=variant%3?.2:0;p.eps_v=.1;p.include_curvature=variant%4==0;
        Vec3 position=center+((kind==0)?Vec3(phi*axis):Vec3((c.radius+phi)*radial));
        if(kind==1)position+=.13*axis;
        const Vec3 previous=position-Vec3(.0003,-.0007,.0004)*(variant%3==1?1e-6:1);
        for(int k=0;k<3;++k) {
            c.point[k]=center[k];c.axis[k]=axis[k];in.position[k]=position[k];in.previous[k]=previous[k];
            c.current_pose.translation[k]=variant>=6?Vec3(.001,.003,-.002)[k]:0;
            for(int l=0;l<3;++l){in.jacobian[3*k+l]=J(k,l);if(variant>=6)c.current_pose.rotation[3*k+l]=rotation(k,l);}
        }
        records.push_back(in);
    }
    // Centers/centerlines have no unique normal; production makes friction
    // inactive rather than inventing an arbitrary tangent direction.
    for(int kind=1;kind<3;++kind) {
        auto in=records[0];in.colliders[0].kind=SdfKind(kind);in.parameters.friction=.2;
        for(int k=0;k<3;++k)in.position[k]=in.previous[k]=in.colliders[0].point[k];
        records.push_back(in);
    }
    // Union selection and deterministic ties; do not add penalties from every
    // collider (the existing solver uses just the minimum-phi obstacle).
    for(int variant=0;variant<3;++variant) {
        auto in=records[2];in.count=3;
        for(int i=1;i<3;++i)in.colliders[i]=in.colliders[0];
        if(variant)for(int k=0;k<3;++k)in.colliders[variant].point[k]+=.2*in.colliders[variant].axis[k];
        records.push_back(in);
    }
    auto empty=records[0];empty.count=0;records.push_back(empty);
    return records;
}
std::vector<ResidentSdfTestOutput> evaluate(const std::vector<ResidentSdfTestInput>& inputs) {
    std::vector<ResidentSdfTestOutput> outputs(inputs.size());
#ifdef IPC_RESIDENT_SDF_CUDA_TEST
    resident_sdf_compute_gpu(inputs.data(),outputs.data(),inputs.size());
#else
    for(std::size_t i=0;i<inputs.size();++i) {
        const auto& in=inputs[i];
        outputs[i].point=ipc_gpu::resident::evaluate_sdf_contact(in.colliders,in.count,in.position,in.previous,in.parameters);
        outputs[i].rigid=ipc_gpu::resident::sdf_rigid_pullback(outputs[i].point,in.jacobian,in.parameters.dt*in.parameters.dt);
    }
#endif
    return outputs;
}
void reference_check(const ResidentSdfTestInput& in,const ResidentSdfTestOutput& out,int index) {
    const std::string label="SDF "+std::to_string(index);
    require(out.point.status==SdfStatus::Ok,label+": unexpected status");
    const Vec3 x(in.position[0],in.position[1],in.position[2]),previous(in.previous[0],in.previous[1],in.previous[2]);
    SDFEvaluation sdf;int selected=-1;
    for(int i=0;i<in.count;++i){const auto value=reference_geometry(in.colliders[i],x);if(selected<0||value.phi<sdf.phi){sdf=value;selected=i;}}
    require(out.point.collider==selected,label+": wrong union collider");
    Vec3 g=Vec3::Zero(),fg=Vec3::Zero();Mat33 H=Mat33::Zero(),fH=Mat33::Zero(),J;
    if(selected>=0) {
        compare(out.point.phi,sdf.phi,label+" phi");
        for(int k=0;k<3;++k) {
            compare(out.point.grad_phi[k],sdf.grad_phi[k],label+" normal");compare(out.point.surface_point[k],sdf.surface_point[k],label+" surface");
            for(int l=0;l<3;++l)compare(out.point.hess_phi[3*k+l],sdf.hess_phi(k,l),label+" normal Hessian");
        }
        g=sdf_penalty_gradient(sdf,in.parameters.stiffness,in.parameters.epsilon);
        H=sdf_penalty_hessian(sdf,in.parameters.stiffness,in.parameters.epsilon,in.parameters.include_curvature!=0);
        if(in.parameters.friction>0) {
            const auto frozen=make_sdf_frozen_friction_contact(x,previous,sdf,in.parameters.stiffness,in.parameters.epsilon,in.parameters.dt,in.parameters.eps_v);
            const auto friction=frozen_friction_relative_gradient_and_hessian(frozen,in.parameters.friction,in.parameters.dt*in.parameters.dt);
            fg=friction.first;fH=friction.second;
        }
    }
    for(int k=0;k<3;++k)for(int l=0;l<3;++l)J(k,l)=in.jacobian[3*k+l];
    const double dt2=in.parameters.dt*in.parameters.dt;
    const Vec3 total_gradient=dt2*g+fg,omega_gradient=J.transpose()*total_gradient;
    const Mat33 total_H=dt2*H+fH,mixed_H=total_H*J,omega_H=J.transpose()*mixed_H;
    for(int k=0;k<3;++k) {
        compare(out.point.gradient[k],g[k],label+" gradient");compare(out.point.friction_gradient[k],fg[k],label+" friction gradient");
        compare(out.rigid.translation_gradient[k],total_gradient[k],label+" rigid COM gradient");compare(out.rigid.orientation_gradient[k],omega_gradient[k],label+" rigid omega gradient");
        for(int l=0;l<3;++l) {
            compare(out.point.hessian[3*k+l],H(k,l),label+" Hessian");compare(out.point.friction_hessian[3*k+l],fH(k,l),label+" friction Hessian");
            compare(out.rigid.translation_translation_hessian[3*k+l],total_H(k,l),label+" rigid COM Hessian");
            compare(out.rigid.translation_orientation_hessian[3*k+l],mixed_H(k,l),label+" rigid mixed Hessian");
            compare(out.rigid.orientation_orientation_hessian[3*k+l],omega_H(k,l),label+" rigid omega Hessian");
        }
    }
}
}
int main() {
    try {
        const auto in=fixtures();const auto out=evaluate(in);
        for(std::size_t i=0;i<in.size();++i)reference_check(in[i],out[i],i);
        std::vector<ResidentSdfTestInput> failures(4,in[1]);
        for(auto& record:failures)record.parameters.friction=.2;
        failures[0].colliders[0].current_pose.rotation[0]=2;
        failures[1]=failures[0];failures[1].parameters.friction=0;
        failures[2]=failures[0];failures[2].colliders[0].previous_pose=failures[2].colliders[0].current_pose;
        failures[3].parameters.dt=0;
        const auto failure_out=evaluate(failures);
        require(failure_out[0].point.status==SdfStatus::InvalidMaterialPose,"moving invalid material pose must reject");
        require(failure_out[1].point.status==SdfStatus::Ok,"friction off must not inspect material pose");
        require(failure_out[2].point.status==SdfStatus::Ok,"identical material poses preserve CPU shortcut");
        require(failure_out[3].point.status==SdfStatus::InvalidInput,"invalid dt must reject");
        std::cout<<"Resident SDF: "<<in.size()<<" records, "<<components<<" reference components, max scaled error "<<maximum_error
#ifdef IPC_RESIDENT_SDF_CUDA_TEST
            <<" (CUDA device execution)\n";
#else
            <<" (host arithmetic only)\n";
#endif
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

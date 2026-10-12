#include "resident_contact_pipeline_test.h"
#include "barrier_energy.h"
#include "friction_energy.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::size_t components=0;double maximum_error=0;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void compare(double actual,double expected,const std::string& label) {
    require(std::isfinite(actual)&&std::isfinite(expected),label+": nonfinite");
    const double error=std::abs(actual-expected)/(1+std::abs(expected));maximum_error=std::max(maximum_error,error);
    require(error<3e-9,label+": mismatch "+std::to_string(actual)+" vs "+std::to_string(expected));++components;
}
Vec3 v(const double* value){return Vec3(value[0],value[1],value[2]);}
void compare_system(const ipc_gpu::Derivatives& actual,const Vec3& g,const Mat33& H,const std::string& label) {
    for(int k=0;k<3;++k){compare(actual.gradient[k],g[k],label+" gradient");for(int l=0;l<3;++l)compare(actual.hessian[3*k+l],H(k,l),label+" Hessian");}
}
ResidentPipelineTestInput fixture(int kind) {
    ResidentPipelineTestInput in;in.kind=kind;
    const Vec3 positions[8]={Vec3(.2,.3,.17),Vec3::Zero(),Vec3(1,0,0),Vec3(0,1,0),
        Vec3(2,0,0),Vec3(3,0,0),Vec3(2.5,-.5,.17),Vec3(2.5,.5,.17)};
    for(int i=0;i<8;++i)for(int k=0;k<3;++k){in.positions[i][k]=positions[i][k];in.previous[i][k]=positions[i][k]-.001*(i+1)*(k+1);}
    if(kind==1) {
        const Vec3 a(2.4,.1,.17),b(2.9,.1,.17);
        for(int k=0;k<3;++k){in.positions[6][k]=a[k];in.positions[7][k]=b[k];in.previous[6][k]=a[k]-.002;in.previous[7][k]=b[k]-.003;}
    }
    if(kind==2) {in.positions[0][0]=in.previous[0][0]=.25;in.positions[0][1]=in.previous[0][1]=.25;in.positions[0][2]=in.previous[0][2]=std::ldexp(1.0,-500);}
    if(kind==3)in.positions[0][2]=0;
    if(kind==4)in.initial_error=123;
    if(kind==5)for(int body=0;body<2;++body) {
        const Vec4 q=Vec4(.97,.03*(body+1),-.05,.08).normalized();
        for(int k=0;k<4;++k)in.previous_orientation[body][k]=q[k];
        for(int k=0;k<3;++k) {
            in.omega[body][k]=(body+1)*Vec3(.13,-.21,.08)[k];
            in.omega_after[body][k]=(body+1)*Vec3(-.22,.09,-.05)[k];
        }
    }
    for(int i=1;i<=5;++i)for(int k=0;k<3;++k)in.references[i][k]=in.positions[i][k]-(i<=3?Vec3(.3,.3,0)[k]:Vec3(2.5,0,0)[k]);
    for(int body=0;body<2;++body)for(int k=0;k<3;++k)in.shift[body][k]=kind==2?0:(body+1)*Vec3(.001,-.003,.004)[k];
    in.collider.kind=ipc_gpu::resident::SdfKind::Plane;in.collider.point[1]=.08;
    in.sdf_parameters.stiffness=170;in.sdf_parameters.epsilon=.2;in.sdf_parameters.dt=.031;in.sdf_parameters.friction=.2;in.sdf_parameters.eps_v=.1;
    in.collider.current_pose.translation[0]=.003;
    return in;
}
void add_sdf(const ResidentPipelineTestInput& in,const Vec3& x,const Vec3& previous,Vec3& g,Mat33& H,const Mat33* J=nullptr) {
    PlaneSDF plane{v(in.collider.point),v(in.collider.axis),{}};
    plane.material_motion.current.translation=v(in.collider.current_pose.translation);
    const auto sdf=evaluate_sdf(plane,x);
    const auto sg=sdf_penalty_gradient(sdf,in.sdf_parameters.stiffness,in.sdf_parameters.epsilon);
    const auto sh=sdf_penalty_hessian(sdf,in.sdf_parameters.stiffness,in.sdf_parameters.epsilon,false);
    const auto frozen=make_sdf_frozen_friction_contact(x,previous,sdf,in.sdf_parameters.stiffness,in.sdf_parameters.epsilon,.031,.1);
    const auto friction=frozen_friction_relative_gradient_and_hessian(frozen,.2,.031*.031);
    const Vec3 totalg=.031*.031*sg+friction.first;const Mat33 totalh=.031*.031*sh+friction.second;
    if(J){g+=J->transpose()*totalg;H+=J->transpose()*totalh*(*J);}else{g+=totalg;H+=totalh;}
}
void reference(const ResidentPipelineTestInput& input,const ResidentPipelineTestOutput& out) {
    const double dt=.031,scale=dt*dt*370;
    const int nodes[2][4]={{0,1,2,3},{4,5,6,7}},owners[8]={-1,0,0,0,1,1,-1,-1};
    for(int node:{0,6,7}) {
        const int pair=node==0?0:1,role=node==0?0:node-4;
        std::array<Vec3,4>x,previous;
        for(int i=0;i<4;++i){x[i]=v(input.positions[nodes[pair][i]]);previous[i]=v(input.previous[nodes[pair][i]]);}
        Vec3 g;Mat33 H;FrozenFrictionContact frozen;
        if(pair==0) {
            const auto e=make_node_triangle_contact_evaluation(x,.8,370);
            const auto normal=node_triangle_barrier_self_gradient_and_hessian(x[0],x[1],x[2],x[3],role,e);
            g=scale*normal.first;H=scale*normal.second;
            frozen=make_node_triangle_frozen_friction_contact(x,previous,e,dt,.1);
        } else {
            const auto e=make_segment_segment_contact_evaluation(x,.8,370);
            const auto normal=segment_segment_barrier_self_gradient_and_hessian(x[0],x[1],x[2],x[3],role,e);
            g=scale*normal.first;H=scale*normal.second;
            frozen=make_segment_segment_frozen_friction_contact(x,previous,e,dt,.1);
        }
        const auto friction=frozen_friction_role_gradient_and_hessian(frozen,role,.2,dt*dt);g+=friction.first;H+=friction.second;
        if(node!=7)add_sdf(input,v(input.positions[node]),v(input.previous[node]),g,H);
        compare_system(out.deformable[node],g,H,"deformable "+std::to_string(node));
    }
    for(int orientation=0;orientation<3;++orientation)for(int body=0;body<2;++body) {
        const int pair=body==0?0:1;
        std::array<Vec3,4>x,previous,X;
        for(int i=0;i<4;++i) {
            const int vertex=nodes[pair][i];x[i]=v(input.positions[vertex]);previous[i]=v(input.previous[vertex]);X[i]=v(input.references[vertex]);
            if(orientation&&owners[vertex]>=0)x[i]+=v(input.shift[owners[vertex]]);
        }
        const auto& q=input.previous_orientation[body];
        const Vec4 previous_q(q[0],q[1],q[2],q[3]);
        const Vec3 omega=v(orientation==2?input.omega_after[body]:input.omega[body]);
        const auto kinematics=quaternion_omega_kinematics(previous_q,omega,dt);
        const auto mode=orientation?RigidDerivativeMode::OrientationHessian:RigidDerivativeMode::TranslationHessian;
        const auto side=body==0?RigidBarrierSide::SecondPrimitive:RigidBarrierSide::FirstPrimitive;
        const auto derivatives=pair==0?node_triangle_barrier_rb(x[0],x[1],x[2],x[3],X,side,previous_q,omega,dt,.8,mode)
            :segment_segment_barrier_rb(x[0],x[1],x[2],x[3],X,side,previous_q,omega,dt,.8,mode);
        Vec3 g=scale*(orientation?derivatives.orientation_gradient:derivatives.translation_gradient);
        Mat33 H=scale*(orientation?derivatives.orientation_orientation_hessian:derivatives.translation_translation_hessian);
        const auto frozen=pair==0?make_node_triangle_frozen_friction_contact(x,previous,.8,370,dt,.1)
            :make_segment_segment_frozen_friction_contact(x,previous,.8,370,dt,.1);
        const auto relative=frozen_friction_relative_gradient_and_hessian(frozen,.2,dt*dt);
        double weight=0;Mat33 Jf=Mat33::Zero();
        for(int i=0;i<4;++i)if(owners[nodes[pair][i]]==body){weight+=frozen.weights[i];Jf+=frozen.weights[i]*dx_domega(X[i],kinematics);}
        if(orientation){g+=Jf.transpose()*relative.first;H+=Jf.transpose()*relative.second*Jf;}
        else{g+=weight*relative.first;H+=weight*weight*relative.second;}
        for(int vertex=0;vertex<8;++vertex)if(owners[vertex]==body) {
            Vec3 position=v(input.positions[vertex]);if(orientation)position+=v(input.shift[body]);
            const Mat33 J=dx_domega(v(input.references[vertex]),kinematics);
            add_sdf(input,position,v(input.previous[vertex]),g,H,orientation?&J:nullptr);
        }
        compare_system(orientation==2?out.rigid_omega_after[body]:orientation?out.rigid_omega[body]:out.rigid_com[body],g,H,"rigid "+std::to_string(body)+" phase "+std::to_string(orientation));
    }
}
}
int main() {
    try {
        for(int kind=0;kind<6;++kind) {
            const auto input=fixture(kind);ResidentPipelineTestOutput out;
            resident_contact_pipeline_test_gpu(input,out);
            const int expected_incidents[10]={1,0,0,0,0,0,1,1,1,1};
            for(int block=0;block<10;++block)require(out.incident_counts[block]==expected_incidents[block],"wrong deduplicated incident CSR");
            if(kind<3||kind==5){require(out.error==0,"unexpected pipeline device error "+std::to_string(out.error));reference(input,out);}
            else {
                require(out.error==(kind==3?21:123),"pipeline failed to preserve zero-gap/preexisting error");
                for(const auto& system:out.deformable)compare_system(system,Vec3::Zero(),Mat33::Zero(),"error rollback guard");
                for(const auto& system:out.rigid_com)compare_system(system,Vec3::Zero(),Mat33::Zero(),"error rigid guard");
                for(const auto& system:out.rigid_omega)compare_system(system,Vec3::Zero(),Mat33::Zero(),"error orientation guard");
                for(const auto& system:out.rigid_omega_after)compare_system(system,Vec3::Zero(),Mat33::Zero(),"error refreshed orientation guard");
            }
        }
        std::cout<<"Resident contact pipeline: 6 mixed tests, "<<components<<" reference components, max scaled error "<<maximum_error<<" (CUDA)\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

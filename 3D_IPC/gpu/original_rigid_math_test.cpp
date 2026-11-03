// CPU-oracle checks of the SAME host/device mathematics used by CUDA.
// This executable intentionally has no CUDA dependency and proves no kernel,
// allocation, launch, or GPU-runtime behavior.
#include "original_rigid.cuh"
#include "original_contact.cuh"
#include "rigid.cuh"
#include "barrier_energy.h"
#include "friction_energy.h"
#include "ccd.h"
#include "safe_step.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Positions=std::array<Vec3,4>;
using namespace ipc_gpu;
using namespace ipc_gpu::solver;
std::size_t comparisons=0;
double worst_error=0;

void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
void close(double actual,double expected,const char* label) {
    require(std::isfinite(actual)&&std::isfinite(expected),std::string(label)+": nonfinite");
    const double error=std::abs(actual-expected)/(1+std::abs(expected));
    worst_error=std::max(worst_error,error);
    require(error<=2e-9,std::string(label)+": CPU parity mismatch");
    ++comparisons;
}

void check_contact(const Positions& positions,bool face,bool first,bool orientation,int slip) {
    const Vec4 q=Vec4(-.3,.2,.7,-.5).normalized();
    const Vec3 omega(.2,-.7,1.4),com(.2,-.3,.4);
    constexpr double dt=.017,dhat=.35;
    ContactInput input{};
    input.kind=face?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    input.d_hat=dhat;input.k_barrier=1;input.dt=dt;
    input.friction=.27;input.eps_v=.003;input.eps=1e-12;
    input.hessian_blocks=0xffffu;
    Positions previous=positions,local;
    const auto cpu_kin=quaternion_omega_kinematics(q,omega,dt,true);
    double qraw[4],wraw[3];
    for(int k=0;k<4;++k)qraw[k]=q[k];
    for(int k=0;k<3;++k)wraw[k]=omega[k];
    const auto kin=rigid_kinematics(qraw,wraw,dt);
    for(int i=0;i<4;++i) {
        previous[i]-=double(slip)*1e-5*double(i+1)*Vec3(.2,-.1,.3);
        local[i]=material_space_position(positions[i],com,cpu_kin.orientation);
        for(int k=0;k<3;++k) {
            input.positions[i][k]=positions[i][k];input.previous[i][k]=previous[i][k];
        }
    }
    const auto out=evaluate_contact_original(input);
    require(out.status==ContactStatus::Ok&&out.active,"original rigid contact inactive/error");
    RigidPreparedContact packet{};
    packet.segment_segment=!face;
    packet.flags=orientation?RigidOrientationGradient|RigidOrientationHessian|RigidUpperCrossBlocks
                            :RigidTranslationGradient|RigidTranslationHessian;
    packet.flags|=RigidFriction;packet.update_mask=orientation?2u:1u;
    bool selected[4];
    for(int i=0;i<4;++i) {
        selected[i]=(i<(face?1:2))==first;
        if(!selected[i])continue;
        packet.friction_weights[i]=out.weights[i];
        for(int k=0;k<3;++k)packet.gradients[i][k]=out.barrier_gradient[i][k];
        if(orientation) {
            double reference[3];for(int k=0;k<3;++k)reference[k]=local[i][k];
            rigid_reference_derivatives(reference,kin,packet.jacobians[i],packet.curvature[i]);
        }
    }
    for(int k=0;k<3;++k)packet.friction_gradient[k]=out.friction_relative_gradient[k];
    for(int k=0;k<9;++k)packet.friction_hessian[k]=out.friction_relative_hessian[k];
    if(!orientation&&face)for(int k=0;k<9;++k)packet.translation_hessian[k]=out.barrier_hessian[0][0][k];
    else for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j])
        for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
            packet.hessians[i][j][3*k+l]=out.barrier_hessian[i][j][3*k+l];
            if(!orientation&&i!=j)packet.hessians[j][i][3*l+k]=out.barrier_hessian[i][j][3*k+l];
        }
    const auto actual=evaluate_rigid(packet);
    const auto side=first?RigidBarrierSide::FirstPrimitive:RigidBarrierSide::SecondPrimitive;
    const auto mode=orientation?RigidDerivativeMode::OrientationHessian:RigidDerivativeMode::TranslationHessian;
    RigidEnergyDerivatives expected;
    FrozenFrictionContact frozen;
    if(face) {
        const auto evaluation=make_node_triangle_contact_evaluation(positions,dhat,1,input.eps,nullptr,false);
        expected=node_triangle_barrier_rb(positions[0],positions[1],positions[2],positions[3],local,
            side,q,omega,dt,dhat,mode,input.eps,&cpu_kin,&evaluation.dr);
        frozen=make_node_triangle_frozen_friction_contact(positions,previous,evaluation,dt,input.eps_v);
    } else {
        expected=segment_segment_barrier_rb(positions[0],positions[1],positions[2],positions[3],local,
            side,q,omega,dt,dhat,mode,input.eps,&cpu_kin,nullptr,nullptr,nullptr,false);
        frozen=make_segment_segment_frozen_friction_contact(positions,previous,dhat,1,dt,input.eps_v,input.eps,nullptr,false);
    }
    // CPU frozen friction uses a weighted Jacobian and deliberately omits the
    // quaternion-curvature term. This independent oracle also checks signed
    // weights on the triangle/second-edge side.
    double weight=0;
    Mat33 J=Mat33::Zero();
    for(int i=0;i<4;++i)if(selected[i]) {
        weight+=frozen.weights[i];
        J+=frozen.weights[i]*dx_domega(local[i],cpu_kin);
    }
    const auto relative=frozen_friction_relative_gradient_and_hessian(frozen,input.friction,dt*dt);
    const Vec3 fg=orientation?Vec3(J.transpose()*relative.first):Vec3(weight*relative.first);
    const Mat33 fH=orientation?Mat33(J.transpose()*relative.second*J):Mat33(weight*relative.second*weight);
    const Vec3& gradient=orientation?expected.orientation_gradient:expected.translation_gradient;
    const Mat33& hessian=orientation?expected.orientation_orientation_hessian:expected.translation_translation_hessian;
    const double* g=orientation?actual.barrier.orientation_gradient:actual.barrier.translation_gradient;
    const double* H=orientation?actual.barrier.orientation_orientation_hessian:actual.barrier.translation_translation_hessian;
    const double* friction_g=orientation?actual.friction.orientation_gradient:actual.friction.translation_gradient;
    const double* friction_H=orientation?actual.friction.orientation_orientation_hessian:actual.friction.translation_translation_hessian;
    for(int k=0;k<3;++k) {
        close(g[k],gradient[k],"rigid original gradient");close(friction_g[k],fg[k],"rigid original friction gradient");
        for(int l=0;l<3;++l) {
            close(H[3*k+l],hessian(k,l),"rigid original Hessian");
            close(friction_H[3*k+l],fH(k,l),"rigid original friction Hessian");
        }
    }
}

void contacts() {
    const Positions triangle{Vec3(.2,.2,.1),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)};
    std::vector<Positions> point_cases;
    for(const auto& point:{Vec3(.2,.2,.1),Vec3(.4,-.1,.1),Vec3(-.1,-.1,.1),Vec3(.5,.5,.1)}) {
        auto x=triangle;x[0]=point;point_cases.push_back(x);
    }
    point_cases.push_back(Positions{Vec3(.2,.1,.1),Vec3(0,0,0),Vec3(1,0,0),Vec3(2,0,0)});
    const std::vector<Positions> edge_cases{
        {Vec3(-1,0,.1),Vec3(1,0,.1),Vec3(0,-1,0),Vec3(0,1,0)},
        {Vec3(0,0,.1),Vec3(1,0,.1),Vec3(0,0,0),Vec3(0,1,0)},
        {Vec3(-1,0,.1),Vec3(1,0,.1),Vec3(-.5,.1,0),Vec3(.5,.1,0)},
        {Vec3(0,0,.1),Vec3(1,0,.1),Vec3(1.1,.1,0),Vec3(1.1,1,0)}};
    for(bool first:{true,false})for(bool orientation:{false,true})for(int slip:{0,1,10}) {
        for(const auto& x:point_cases)check_contact(x,true,first,orientation,slip);
        for(const auto& x:edge_cases)check_contact(x,false,first,orientation,slip);
    }
}

void translation_ccd() {
    using Status=original_ccd::Status;
    for(bool face:{true,false})for(bool first:{true,false})for(double dz:{-2.0,-.25,0.0,.25}) {
        Positions x=face?Positions{Vec3(.2,.2,1),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)}
                        :Positions{Vec3(-1,0,1),Vec3(1,0,1),Vec3(0,-1,0),Vec3(0,1,0)};
        if(!first)for(auto& p:x)p.z()=1-p.z();
        double start[4][3],delta[3]={0,0,dz};
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)start[i][k]=x[i][k];
        const auto query=original_rigid::translation_query(start,face,first,delta);
        auto actual=original_ccd::evaluate_fast(query);
        if(actual.status==Status::NeedsExact)actual=original_ccd::evaluate_exact<2048>(query);
        require(actual.status==Status::Ok,"original rigid CCD resource/input error");
        CCDResult expected;
        const Vec3 d(0,0,dz),zero=Vec3::Zero();
        if(face)expected=node_triangle_only_one_node_moves(x[0],first?d:-d,x[1],zero,x[2],zero,x[3],zero,1e-12,false,true);
        else if(first)expected=segment_segment_same_displacement_linear_ccd(x[0],d,x[1],d,x[2],x[3],1e-12,true);
        else expected=segment_segment_same_displacement_linear_ccd(x[2],d,x[3],d,x[0],x[1],1e-12,true);
        require(actual.collision==expected.collision,"original rigid CCD hit differs from CPU");
        if(expected.collision)close(actual.toi,expected.t,"original rigid TOI");
    }
}

void represented_safety() {
    using Status=original_rigid::RepresentationStatus;
    double start[4][3]={{-1,0,1},{1,0,1},{0,-1,0},{0,1,0}},end[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)end[i][k]=start[i][k];
    end[0][2]=.1;end[1][2]=std::nextafter(.1,1.0);
    require(original_rigid::represented_path<2048>(start,end,false)==Status::Safe,
        "safe rounded translating edge was not certified");
    // Both ENDPOINTS are separated, yet the full represented path crosses.
    end[0][2]=-1;end[1][2]=std::nextafter(-1.0,0.0);
    require(original_rigid::represented_path<2048>(start,end,false)!=Status::Safe,
        "crossing rounded two-endpoint path was accepted");
    start[0][2]=start[1][2]=0;
    require(original_rigid::represented_path<2048>(start,end,false)==Status::InitialContact,
        "true initial intersection was hidden by conservative backtracking");
    end[0][0]=std::numeric_limits<double>::quiet_NaN();
    require(original_rigid::represented_path<2048>(start,end,false)==Status::Invalid,
        "nonfinite represented rigid endpoint was accepted");
}

void tiny_translation_cpu_rule() {
    BroadPhase::Cache cache;cache.nt_pairs.push_back({0,{1,2,3}});
    const std::vector<int> nt{0},ss;
    for(bool angled:{false,true}) {
        const std::vector<Vec3> positions=angled
            ?std::vector<Vec3>{Vec3(0,0,5e-11),Vec3(-1,-1,1),Vec3(1,-1,-1),Vec3(0,1,0)}
            :std::vector<Vec3>{Vec3(.2,.2,5e-11),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)};
        for(bool first:{true,false})for(bool toward:{false,true}) {
            RefMesh mesh;mesh.node_to_rb=first?std::vector<int>{0,-1,-1,-1}:std::vector<int>{-1,0,0,0};
            for(double magnitude:{1e-15,std::nextafter(1e-14,0.0),1e-14,std::nextafter(1e-14,INFINITY),1e-9}) {
                const double sign=(first!=toward)?1.0:-1.0;
                const Vec3 delta(0,0,sign*magnitude);
                const double raw[3]{delta[0],delta[1],delta[2]};
                const bool tiny=original_rigid::skip_translation_toi(raw);
                double start[4][3],end[4][3];
                for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
                    start[i][k]=positions[i][k];
                    end[i][k]=start[i][k]+((i==0)==first?raw[k]:0);
                }
                const bool overlapping=original_rigid::translation_boxes_overlap(start,true,first,raw);
                const double cpu=per_rigid_body_translation_safe_step(mesh,cache,nt,ss,positions,0,delta,.9,false,true);
                require(tiny==(delta.squaredNorm()<1e-28),"GPU rigid tiny predicate differs from CPU cutoff");
                const bool bypass=tiny||!overlapping;
                close(cpu,bypass?1.0:0.0,"CPU original cutoff and swept-box rejection");
                auto band=original_ccd::evaluate_fast(original_rigid::translation_query(start,true,first,raw));
                if(band.status==original_ccd::Status::NeedsExact)
                    band=original_ccd::evaluate_exact<2048>(original_rigid::translation_query(start,true,first,raw));
                require(band.collision&&band.toi==0,"regression does not exercise the original positive band");
                if(bypass)require(original_rigid::represented_path<2048>(start,end,true)==original_rigid::RepresentationStatus::Safe,
                    "CPU-approved represented translation must remain safe");
            }
        }
    }
}

} // namespace

int main() {
    try {
        contacts();translation_ccd();represented_safety();tiny_translation_cpu_rule();
        std::cout<<"HOST ARITHMETIC ONLY (not CUDA validation): original rigid contact/friction, "
            <<"translation CCD and rounded-path safety PASS; "<<comparisons
            <<" CPU components, max normalized error "<<worst_error<<'\n';
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"original rigid math test FAIL: "<<error.what()<<'\n';return 1;
    }
}

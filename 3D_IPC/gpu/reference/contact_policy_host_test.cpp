// Protected Codex acceptance for CPU ordinary EE feature precedence.
// Host arithmetic only. Select the candidate's GPU headers with
// -I<isolated-candidate>/gpu BEFORE -I<actual-project>. CPU .cpp oracles stay
// the actual project sources. No source/test/tolerance changes are permitted.
#ifndef GPU_CONTACT_HEADER
#define GPU_CONTACT_HEADER "contact_exact.cuh"
#endif
#include GPU_CONTACT_HEADER
#include "barrier_energy.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::solver;
namespace {
constexpr double tolerance=2e-9; // Existing independent contact fixture bound.
using Points=std::array<Vec3,4>;
struct Case {std::string name;Points x;double scale=1;bool boundary=true;bool robust_nonparallel=false;int extra=0;};
std::vector<Case> cases(double gap) {
    const Points orth{{Vec3(0,0,0),Vec3(0,1,0),Vec3(0,.0625,gap),Vec3(1,.0625,gap)}};
    Points skew=orth;skew[3]=Vec3(.5,1.0625,gap);
    std::vector<Case> out{{"orthogonal t=0",orth},{"reverse first edge",orth},{"reverse second edge",orth},
        {"swap primitives",orth},{"skew t=0",skew},{"skew s/t reversed",skew},
        {"signed-axis rotation plus translation",skew},{"inversion plus scale/translation",skew}};
    std::swap(out[1].x[0],out[1].x[1]);std::swap(out[2].x[2],out[2].x[3]);
    std::swap(out[3].x[0],out[3].x[2]);std::swap(out[3].x[1],out[3].x[3]);
    std::swap(out[5].x[0],out[5].x[1]);std::swap(out[5].x[2],out[5].x[3]);
    for(auto& p:out[6].x)p=Vec3(p.y(),-p.x(),p.z())+Vec3(2,-3,1);
    for(auto& p:out[7].x)p=-2*p+Vec3(.5,-.25,1);out[7].scale=2;
    auto outside=orth;outside[2].x()=.0625;outside[3].x()=1.0625;
    out.push_back({"outside-minimizer edge control",outside,1,false});
    auto parallel=orth;parallel[3]=parallel[2]+Vec3(0,1,0);
    out.push_back({"robust parallel control",parallel,1,false});
    return out;
}
std::vector<Case> robust_nonparallel_controls() {
    const double gap=.001;
    const Points orth{{Vec3(0,0,0),Vec3(0,1,0),Vec3(0,.0625,gap),Vec3(1,.0625,gap)}};
    const Points nearly_parallel{{Vec3(0,0,0),Vec3(0,1,0),Vec3(0,.0625,.125),Vec3(std::ldexp(1.,-25),1.0625,.125)}};
    std::vector<Case> out;
    for(int family=0;family<2;++family)for(int variant=0;variant<4;++variant) {
        Points x=family?nearly_parallel:orth;
        if(variant==1)std::swap(x[0],x[1]);
        if(variant==2)std::swap(x[2],x[3]);
        if(variant==3){std::swap(x[0],x[2]);std::swap(x[1],x[3]);}
        // The regular Gram family forces CPU robust=true through the WORLD
        // gap trigger: 64eps*2^40=.015625 > .001, with a strict boundary
        // exact region. The second family instead has nonzero determinant
        // 2^-50 <=64eps*A*C, at ordinary coordinates and ordinary gap.
        if(!family)for(auto& v:x)v.x()+=std::ldexp(1.,40);
        out.push_back({std::string(family?"robust nonparallel Gram boundary ":"robust translated gap boundary ")+std::to_string(variant),x,1,false,true});
    }
    return out;
}
std::vector<Case> rounded_feature_controls() {
    const double u=std::ldexp(1.,-52);
    return {
        {"rounded CPU Interior exact-outside",{{Vec3(0,0,0),Vec3(1+u,1,0),Vec3(-.5-u/2,.5+u,-.01),Vec3(.5-u/2,-.5,-.01)}},1,false,false,1},
        {"outside endpoint tolerance corner",{{Vec3(0,0,0),Vec3(1,0,0),Vec3(-.1,-5e-15,-.01),Vec3(-.1,1-5e-15,-.01)}},1,false,false,2}
    };
}
}
int main()try {
    std::cout<<std::setprecision(17);double worst=0;std::size_t components=0,failures=0,boundary_cases=0,rows=0,robust_nonparallel_rows=0,robust_nonparallel_failures=0,extra_rows=0,extra_failures=0;
    std::vector<std::pair<double,Case>> inputs;
    for(double gap:{.125,std::ldexp(1.,-30),1e-12})for(const auto& c:cases(gap))inputs.push_back({gap,c});
    for(const auto& c:robust_nonparallel_controls())inputs.push_back({c.x[2].z()-c.x[0].z(),c});
    for(const auto& c:rounded_feature_controls())inputs.push_back({.01,c});
    for(const auto& input:inputs) {
        const double gap=input.first;const auto& c=input.second;const auto failures_before=failures;
        ContactInput in;in.kind=ContactKind::EdgeEdge;in.d_hat=.25*c.scale;in.k_barrier=37;
        in.dt=1./32.;in.eps_v=.125;in.friction=0;in.eps=1e-12;in.hessian_blocks=0xffffu;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)in.positions[i][k]=in.previous[i][k]=c.x[i][k];
        const auto cpu=make_segment_segment_contact_evaluation(c.x,in.d_hat,in.k_barrier,in.eps,nullptr,true);
        if(c.extra) {
            if(cpu.dr.robust||!cpu.active)throw std::runtime_error("extra fixture lost active CPU ordinary branch: "+c.name);
            const Vec3 a=c.x[1]-c.x[0],b=c.x[3]-c.x[2],r=c.x[0]-c.x[2];
            const double A=a.dot(a),B=a.dot(b),C=b.dot(b),D=a.dot(r),E=b.dot(r),det=A*C-B*B;
            const double s=(B*E-C*D)/det,t=(A*E-B*D)/det;
            if(c.extra==1) {
                using Rat=boost::multiprecision::cpp_rational;
                std::array<Rat,3> ar,br,rr;
                for(int k=0;k<3;++k){ar[k]=Rat(c.x[1][k])-Rat(c.x[0][k]);br[k]=Rat(c.x[3][k])-Rat(c.x[2][k]);rr[k]=Rat(c.x[0][k])-Rat(c.x[2][k]);}
                const auto dot=[](const auto& x,const auto& y)->Rat{return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];};
                const Rat aq=dot(ar,ar),bq=dot(ar,br),cq=dot(br,br),dq=dot(ar,rr),eq=dot(br,rr),detq=aq*cq-bq*bq;
                if(cpu.dr.region!=SegmentSegmentRegion::Interior||cpu.dr.s!=0||D!=0||B!=0||s!=0||!(t>0&&t<1)||!(dq>0)||!(detq>0)||!(bq*eq-cq*dq<0))
                    throw std::runtime_error("rounded CPU Interior/exact-outside construction did not hold");
                std::cout<<"EXTRA_PREMISE rounded_Interior CPU_D="<<D<<" CPU_B="<<B<<" exact_D="<<dq<<" exact_s_numerator="<<bq*eq-cq*dq<<'\n';
            } else {
                if(cpu.dr.region!=SegmentSegmentRegion::Corner_s0t0||cpu.dr.s!=0||!(cpu.dr.t>0&&cpu.dr.t<=1e-14)||!(s<0)||!(t>0&&t<1))
                    throw std::runtime_error("outside endpoint-tolerance construction did not hold");
                std::cout<<"EXTRA_PREMISE outside_corner CPU_s_unc="<<s<<" CPU_t_unc="<<t<<" CPU_t="<<cpu.dr.t<<'\n';
            }
            ++extra_rows;
        } else if(c.boundary) {
            if(cpu.dr.robust||cpu.dr.region!=SegmentSegmentRegion::Interior||
                !(cpu.dr.s==0||cpu.dr.s==1||cpu.dr.t==0||cpu.dr.t==1))
                throw std::runtime_error("fixture lost CPU inclusive ordinary-interior branch: "+c.name);
            ++boundary_cases;
        } else if(c.robust_nonparallel) {
            const Vec3 a=c.x[1]-c.x[0],b=c.x[3]-c.x[2];
            const double determinant=a.squaredNorm()*b.squaredNorm()-std::pow(a.dot(b),2);
            if(!cpu.dr.robust||cpu.dr.region==SegmentSegmentRegion::Interior||
                cpu.dr.region==SegmentSegmentRegion::ParallelSegments||!cpu.active||!(determinant>0)||
                !(cpu.dr.s==0||cpu.dr.s==1||cpu.dr.t==0||cpu.dr.t==1))
                throw std::runtime_error("robust nonparallel control lost strict exact boundary: "+c.name);
            ++robust_nonparallel_rows;
        } else if(c.name=="outside-minimizer edge control") {
            if(cpu.dr.robust||cpu.dr.region==SegmentSegmentRegion::Interior||!cpu.active)
                throw std::runtime_error("outside control lost its active noninterior branch");
        } else if(!cpu.dr.robust)throw std::runtime_error("parallel control lost robust branch");
        const auto ordinary=evaluate_contact(in);
        std::cout<<"CASE gap="<<gap<<' '<<c.name<<" CPU region="<<int(cpu.dr.region)<<" robust="<<cpu.dr.robust
            <<" s="<<cpu.dr.s<<" t="<<cpu.dr.t<<" ordinary status="<<unsigned(ordinary.status)<<'\n';
        for(int route=0;route<6;++route) {
            const auto route_failures_before=failures;
            ContactInput request=in;if(route>=2)request.hessian_blocks=1u<<(5*(route-2));
            auto gpu=route==1?evaluate_contact_exact<256>(request):evaluate_contact_gpu<256>(request);
            if(gpu.status==ContactStatus::ExactCapacityExceeded)gpu=evaluate_contact_exact<1600>(request);
            if(gpu.status!=ContactStatus::Ok||bool(gpu.active)!=cpu.active){
                ++failures;if(c.extra)std::cout<<"EXTRA_ROUTE case="<<c.extra<<" route="<<route<<" failures=1 status="<<unsigned(gpu.status)<<'\n';continue;
            }
            auto compare=[&](double actual,double expected,const std::string& field) {
                ++components;const double error=std::abs(actual-expected)/(1+std::abs(expected));worst=std::max(worst,error);
                if(!std::isfinite(actual)||!std::isfinite(expected)||error>tolerance) {
                    ++failures;
                    if(failures<=30||c.extra&&failures-failures_before<=8)std::cout<<"MISMATCH gap="<<gap<<' '<<c.name<<" route="<<route<<' '<<field<<" GPU="<<actual<<" CPU="<<expected<<" relative="<<error<<'\n';
                }
            };
            compare(gpu.distance,cpu.dr.distance,"distance");
            const auto weights=segment_segment_contact_weights(c.x[0],c.x[1],c.x[2],c.x[3],in.eps,&cpu.dr,true);
            for(int role=0;role<4;++role) {
                compare(gpu.weights[role],weights[role],"weight "+std::to_string(role));
                const auto gradient=segment_segment_barrier_gradient(c.x[0],c.x[1],c.x[2],c.x[3],in.d_hat,role,in.eps,&cpu.dr,&cpu.b_prime,true);
                for(int axis=0;axis<3;++axis)compare(gpu.barrier_gradient[role][axis],gradient[axis],"gradient "+std::to_string(role)+":"+std::to_string(axis));
                for(int other=0;other<4;++other) {
                    const bool requested=request.hessian_blocks&(1u<<(4*role+other));
                    const Mat33 hessian=requested?segment_segment_barrier_cross_hessian(c.x[0],c.x[1],c.x[2],c.x[3],in.d_hat,role,other,in.eps,&cpu.dr,&cpu.b_prime,&cpu.b_double_prime,true):Mat33::Zero();
                    for(int a=0;a<3;++a)for(int b=0;b<3;++b)compare(gpu.barrier_hessian[role][other][3*a+b],hessian(a,b),
                        "H "+std::to_string(role)+":"+std::to_string(other)+":"+std::to_string(a)+":"+std::to_string(b));
                }
            }
            if(c.extra)std::cout<<"EXTRA_ROUTE case="<<c.extra<<" route="<<route<<" failures="<<failures-route_failures_before<<'\n';
        }
        if(c.extra){extra_failures+=failures-failures_before;std::cout<<"EXTRA_CASE case="<<c.extra<<" failures="<<failures-failures_before<<'\n';}
        if(c.robust_nonparallel) {
            robust_nonparallel_failures+=failures-failures_before;
            std::cout<<"ROBUST_NONPARALLEL_ROW name="<<c.name<<" failures="<<failures-failures_before<<'\n';
        }
        ++rows;
    }
    if(boundary_cases!=24||rows!=40||robust_nonparallel_rows!=8||extra_rows!=2)throw std::runtime_error("missing protected boundary/scale/control cases");
    std::cout<<"ROBUST_NONPARALLEL_CONTROLS rows="<<robust_nonparallel_rows<<" failures="<<robust_nonparallel_failures<<'\n';
    std::cout<<"ROUNDED_FEATURE_CONTROLS rows="<<extra_rows<<" failures="<<extra_failures<<'\n';
    std::cout<<"HOST ARITHMETIC ONLY: 24 ordinary boundary variants +6 prior controls +8 robust nonparallel boundary controls +2 rounded-feature controls, full dispatch/direct-exact/all4self routes, "<<components
        <<" Cartesian components, failures="<<failures<<", worst normalized error="<<worst<<'\n';
    if(failures)throw std::runtime_error("CPU ordinary/cold EE feature parity failed; preserve tolerances and CPU oracle");
    std::cout<<"CPU EE ordinary+cold boundary feature parity V4 PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}

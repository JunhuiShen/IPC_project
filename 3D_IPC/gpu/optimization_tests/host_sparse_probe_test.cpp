// Independent host-only discard oracle. No CUDA launch/runtime or solver code.
#include "original_contact.cuh"
#include "../original_sparse_gather_policy.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace gpu=ipc_gpu::solver;
namespace policy=gpu::original_sparse_gather;
namespace {
using Point=std::array<double,3>;
std::size_t checked=0,discarded=0,geometry_checks=0,discard_pt=0,discard_ee=0,invalid_kept=0;
void require(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
bool bits(double a,double b){return std::memcmp(&a,&b,sizeof(a))==0;}
gpu::ContactInput make(gpu::ContactKind kind,const std::array<Point,4>& points) {
    gpu::ContactInput in;in.kind=kind;in.d_hat=.1;in.k_barrier=17.;in.dt=.01;
    in.eps=1e-12;in.eps_v=.001;in.friction=0.;in.hessian_blocks=0xffffu;
    for(int v=0;v<4;++v)for(int k=0;k<3;++k) {
        in.positions[v][k]=points[v][k];in.previous[v][k]=points[v][k]-.001*(v+1)*(k+1);
    }
    return in;
}
bool check(const gpu::ContactInput& actual,const char* label) {
    auto input=actual;input.k_barrier=0.;input.hessian_blocks=1u;input.friction=0.;
    const auto probe=gpu::evaluate_contact_original_self_zero(input,0);
    const auto old=gpu::evaluate_contact_original(actual);
    const bool drop=policy::discard_force(actual,probe);++checked;
    if(drop) {
        require(old.status==gpu::ContactStatus::Ok&&old.active==0,
            std::string(label)+" discarded old non-OK/active status="+std::to_string(unsigned(old.status)));
        ++discarded;(actual.kind==gpu::ContactKind::PointTriangle?discard_pt:discard_ee)++;
    }
    if(old.status!=gpu::ContactStatus::Ok){require(!drop,std::string(label)+" invalid old packet discarded");++invalid_kept;}
    if(probe.status==gpu::ContactStatus::Ok&&old.status==gpu::ContactStatus::Ok) {
        require(bits(probe.distance,old.distance),std::string(label)+" geometry distance bits changed");
        for(int i=0;i<4;++i)require(bits(probe.weights[i],old.weights[i]),std::string(label)+" geometry weight bits changed");
        for(int k=0;k<3;++k)require(bits(probe.normal[k],old.normal[k]),std::string(label)+" geometry normal bits changed");
        ++geometry_checks;
    }
    return drop;
}
std::vector<gpu::ContactInput> geometry() {
    std::vector<gpu::ContactInput> out;
    const std::array<Point,4> pt{{Point{.25,.25,.2},Point{0,0,0},Point{1,0,0},Point{0,1,0}}};
    for(const auto& p:std::vector<Point>{{.25,.25,.2},{.25,.25,-.2},{.5,.5,.2},{0,0,.2},{1,0,.2},{-.2,-.2,.2},{1.2,1.2,.2},{.25,.25,0},{.25,.25,-0.0}}) {
        auto q=pt;q[0]=p;out.push_back(make(gpu::ContactKind::PointTriangle,q));
    }
    for(double height:{0.,1e-14,1e-12}) {
        auto q=pt;q[3]={2.,height,0};out.push_back(make(gpu::ContactKind::PointTriangle,q));
        q[1]=q[2]=q[3]=Point{};out.push_back(make(gpu::ContactKind::PointTriangle,q));
    }
    for(const auto& q:std::vector<std::array<Point,4>>{
        {{Point{0,0,0},Point{1,0,0},Point{.5,-.5,.2},Point{.5,.5,.2}}},
        {{Point{0,0,0},Point{1,0,0},Point{0,.2,0},Point{1,.2,0}}},
        {{Point{0,0,0},Point{1,0,0},Point{.2,0,0},Point{.8,0,0}}},
        {{Point{0,0,0},Point{0,0,0},Point{0,.2,0},Point{1,.2,0}}},
        {{Point{0,0,0},Point{0,0,0},Point{0,.2,0},Point{0,.2,0}}},
        {{Point{0,0,0},Point{1,0,0},Point{1,.2,0},Point{1,1.2,0}}}})
        out.push_back(make(gpu::ContactKind::EdgeEdge,q));
    auto signedzero=out.front();signedzero.positions[1][0]=-0.;signedzero.previous[2][1]=-0.;out.push_back(signedzero);
    return out;
}
void sweep(const gpu::ContactInput& g,const char* label) {
    auto probe_input=g;probe_input.k_barrier=0.;probe_input.hessian_blocks=1;probe_input.friction=0.;
    const auto probe=gpu::evaluate_contact_original_self_zero(probe_input,0);
    std::vector<double> dhats{std::numeric_limits<double>::denorm_min(),.001,.1,1.,10.};
    if(probe.status==gpu::ContactStatus::Ok&&probe.distance>0&&std::isfinite(probe.distance)) {
        dhats.push_back(probe.distance);dhats.push_back(std::nextafter(probe.distance,0.));
        dhats.push_back(std::nextafter(probe.distance,std::numeric_limits<double>::infinity()));
    }
    for(double dhat:dhats)for(double stiffness:{0.,-0.,std::numeric_limits<double>::denorm_min(),1.,std::numeric_limits<double>::max()})
        for(unsigned mask:{0u,1u,1u<<5,0x8ce0u,0xffffu,0x10000u}) {
            auto in=g;in.d_hat=dhat;in.k_barrier=stiffness;in.hessian_blocks=mask;check(in,label);
        }
}
void invalid_matrix(const gpu::ContactInput& g) {
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(double bad:{nan,inf,-inf})for(int v=0;v<4;++v)for(int k=0;k<3;++k)for(bool previous:{false,true}) {
        auto in=g;(previous?in.previous:in.positions)[v][k]=bad;require(!check(in,"poisoned geometry"),"poisoned geometry discarded");
    }
    for(int field=0;field<7;++field)for(double value:{-1.,-0.,0.,nan,inf,-inf,std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max()}) {
        auto in=g;
        if(field==0)in.dt=value;else if(field==1)in.eps=value;else if(field==2)in.eps_v=value;
        else if(field==3)in.d_hat=value;else if(field==4)in.k_barrier=value;
        else if(field==5)in.friction=value;else in.kind=static_cast<gpu::ContactKind>(2);
        check(in,"parameter poison/boundary");
    }
    for(const auto& pair:std::vector<std::pair<double,double>>{{1e200,.001},{1e-200,1e-200},{1e200,1e200},{1e-200,1e200},{1e150,1e-150}}) {
        auto in=g;in.dt=pair.first;in.eps_v=pair.second;check(in,"dt product under/overflow");
    }
}
}
int main(){try {
    const auto base=geometry();for(const auto& in:base){sweep(in,"deterministic geometry/tie");invalid_matrix(in);}
    std::mt19937_64 random(0x51a6d1e47aULL);
    for(int sample=0;sample<256;++sample)for(int exponent:{-500,-300,-100,-30,0,30,100,300,500,900}) {
        std::array<Point,4> points{};const double scale=std::ldexp(1.,exponent);
        for(auto& point:points)for(double& value:point)value=(int(random()%2049)-1024)/512.*scale;
        auto in=make(sample%2?gpu::ContactKind::PointTriangle:gpu::ContactKind::EdgeEdge,points);
        sweep(in,"independent random scaled geometry");
    }
    require(discarded>0&&discard_pt>0&&discard_ee>0&&invalid_kept>0&&geometry_checks>0,"discard/keep/geometry premise vacuous");
    std::cout<<"SPARSE_FORCE_DISCARD_HOST_ORACLE_PASS checked="<<checked<<" discarded="<<discarded
        <<" PT="<<discard_pt<<" EE="<<discard_ee<<" old_error_kept="<<invalid_kept
        <<" geometry_bit_checks="<<geometry_checks<<" every_discard_oldOKinactive=1 host_only=1\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"SPARSE_FORCE_DISCARD_HOST_ORACLE_FAIL "<<e.what()<<'\n';return 1;}}

#include "rigid_dat.cuh"
#include "rigid.cuh"
#include "../rigid_dat_trajectory.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace gpu=ipc_gpu::solver::rigid_dat_detail;
constexpr double pi=3.14159265358979323846;
int checks=0;
void require(bool condition,const char* message) {
    ++checks;
    if(!condition)throw std::runtime_error(message);
}
rigid_dat::Trajectory cpu_path(const gpu::Trajectory& path) {
    rigid_dat::Trajectory result;
    for(int i=0;i<3;++i) {
        result.center[i]=path.center[i];result.delta_center[i]=path.delta_center[i];
        result.omega[i]=path.omega[i];result.delta_omega[i]=path.delta_omega[i];
        result.reference[i]=path.reference[i];
    }
    for(int i=0;i<4;++i)result.q0[i]=path.q0[i];
    result.dt=path.dt;
    return result;
}
double distance(const double* a,const double* b) {
    return std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
}
void production_point(const gpu::Trajectory& path,double t,double* out) {
    double omega[3],center[3],q[4];
    for(int i=0;i<3;++i) {
        omega[i]=path.omega[i]+t*path.delta_omega[i];
        center[i]=path.center[i]+t*path.delta_center[i];
    }
    ipc_gpu::solver::rigid_orientation(path.q0,omega,path.dt,q);
    ipc_gpu::solver::rigid_world_position(path.reference,center,q,out);
}
void verify_box(const gpu::Trajectory& path,double a,double b,int samples=256) {
    gpu::Box box{};
    require(gpu::trajectory_bounds(path,a,b,box)==gpu::Status::Ok,"GPU trajectory bounds failed");
    const auto cpu=cpu_path(path);
    const auto cpu_box=rigid_dat::trajectory_bounds(cpu,a,b);
    // Both ports implement the same independent outward arithmetic contract.
    for(int j=0;j<3;++j) {
        require(box.lo[j]==cpu_box.lo[j]&&box.hi[j]==cpu_box.hi[j],"GPU/CPU interval bounds disagree");
    }
    for(int i=0;i<=samples;++i) {
        const double t=a+(b-a)*double(i)/samples;
        const auto actual=rigid_dat::evaluate(cpu,t);
        double proposed[3];
        require(gpu::evaluate(path,t,proposed)==gpu::Status::Ok,"GPU point evaluation failed");
        for(int j=0;j<3;++j) {
            require(actual[j]>=box.lo[j]&&actual[j]<=box.hi[j],"CPU trajectory escaped GPU box");
            require(proposed[j]==actual[j],"GPU/CPU point proposals disagree");
        }
    }
}
void verify_plane_prefix(const gpu::Trajectory& path,const double* n,const double* p,
    const gpu::Result& result) {
    gpu::PreparedTrajectory prepared;
    require(gpu::prepare_trajectory(path,prepared)==gpu::Status::Ok,"Plane preparation failed");
    const auto cached=gpu::plane_step(path,n,p,.9,40,4096,&prepared);
    require(cached.status==result.status && cached.factor==result.factor,"Prepared plane certificate changed");
    require(!gpu::is_fatal(result.status),"Plane query failed");
    require(result.factor>0&&result.factor<=1,"Plane factor outside positive unit interval");
    const auto cpu=cpu_path(path);
    for(int i=0;i<=512;++i) {
        const auto x=rigid_dat::evaluate(cpu,result.factor*double(i)/512);
        const long double margin=static_cast<long double>(n[0])*(x[0]-p[0])+
            static_cast<long double>(n[1])*(x[1]-p[1])+
            static_cast<long double>(n[2])*(x[2]-p[2]);
        require(margin>0,"Accepted curved prefix crossed intended plane");
    }
}
void verify_radius_prefix(const gpu::Trajectory& path,const double* query,double radius,
    const gpu::Result& result) {
    gpu::PreparedTrajectory prepared;
    require(gpu::prepare_trajectory(path,prepared)==gpu::Status::Ok,"Radius preparation failed");
    const auto cached=gpu::radius_step(path,query,radius,40,4096,&prepared);
    require(cached.status==result.status && cached.factor==result.factor,"Prepared radius certificate changed");
    require(!gpu::is_fatal(result.status),"Radius query failed");
    require(result.factor>0&&result.factor<=1,"Radius factor outside positive unit interval");
    const auto cpu=cpu_path(path);
    for(int i=0;i<=512;++i) {
        const auto x=rigid_dat::evaluate(cpu,result.factor*double(i)/512);
        require(distance(x.data(),query)<=radius,"Accepted curved prefix escaped query sphere");
    }
}
std::uint64_t random_state=0x1234bcdef9876543ULL;
double random_signed() {
    random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;
    return double(random_state>>11)*0x1p-52-1;
}
}

int main() {
    try {
        const double nx[3]={1,0,0},px[3]={.5,0,0},origin[3]={0,0,0};
        gpu::Trajectory spin;
        spin.reference[0]=1;spin.delta_omega[2]=2*pi;
        verify_box(spin,0,1);verify_box(spin,.1,.3);
        double quarter[3];
        require(gpu::evaluate(spin,.25,quarter)==gpu::Status::Ok,"Quarter turn failed");
        require(std::fabs(quarter[0])<1e-14&&std::fabs(quarter[1]-1)<1e-14,"Rotation convention changed");
        // Start and endpoint coincide: endpoint/chord tests would accept the
        // complete rotation although the interior crosses x=.5.
        auto step=gpu::plane_step(spin,nx,px,.9);
        require(step.factor>.14&&step.factor<.151,"Missed hidden full-turn crossing or applied gamma twice");
        require(std::fabs(step.factor-rigid_dat::plane_step(cpu_path(spin),{1,0,0},{.5,0,0},.9))<1e-12,
            "GPU plane limit differs from CPU certified path");
        verify_plane_prefix(spin,nx,px,step);
        const double start[3]={1,0,0};
        auto radial=gpu::radius_step(spin,start,.25);
        require(radial.factor>.039&&radial.factor<.040,"Full-turn radius used endpoint displacement");
        require(std::fabs(radial.factor-rigid_dat::radius_step(cpu_path(spin),.25))<1e-10,
            "GPU/CPU radius limits disagree");
        verify_radius_prefix(spin,start,.25,radial);

        // Eight complete turns hide every crossing from all eight proposals.
        auto turns=spin;turns.delta_omega[2]=16*pi;
        const auto hidden=gpu::plane_step(turns,nx,px,.9);
        require(hidden.factor>.018&&hidden.factor<.019,"Sample-only certification accepted hidden turns");
        verify_plane_prefix(turns,nx,px,hidden);
        const auto hidden_radius=gpu::radius_step(turns,start,.25);
        require(hidden_radius.factor>.0048&&hidden_radius.factor<.0051,"Hidden turns escaped radius cap");
        verify_radius_prefix(turns,start,.25,hidden_radius);

        auto away=spin;away.reference[0]=0;away.reference[1]=1;
        const double negative_x[3]={-1,0,0};
        const auto returning=gpu::plane_step(away,negative_x,px,.9);
        require(returning.factor>.52&&returning.factor<.526,"Initially-away shortcut missed returning arc");
        verify_plane_prefix(away,negative_x,px,returning);
        require(gpu::plane_step(away,nx,px,.9).status==gpu::Status::InitialSideUnproved,
            "Plane query silently reversed the intended side");

        gpu::Trajectory translated;
        translated.reference[0]=.1;translated.reference[1]=.2;translated.reference[2]=.3;
        translated.delta_center[2]=2;
        const double negative_z[3]={0,0,-1},plane_z[3]={0,0,1.3};
        const auto linear=gpu::plane_step(translated,negative_z,plane_z,.9);
        require(std::fabs(linear.factor-.45)<1e-9,"Linear plane factor changed");
        verify_plane_prefix(translated,negative_z,plane_z,linear);
        const auto linear_radius=gpu::radius_step(translated,translated.reference,.5);
        require(std::fabs(linear_radius.factor-.25)<1e-9,"Linear radius applies gamma again");
        verify_radius_prefix(translated,translated.reference,.5,linear_radius);

        // A query-centered radius must include the stored-point/pose offset.
        const double offset_query[3]={.1,.2,.2};
        const auto offset=gpu::radius_step(translated,offset_query,.5);
        require(offset.factor>.19999999&&offset.factor<=.2,"Query anchor offset was omitted");
        verify_radius_prefix(translated,offset_query,.5,offset);
        require(gpu::radius_step(translated,offset_query,.05).status==gpu::Status::InitialRadiusUnproved,
            "Initial represented pose outside query radius was accepted");

        gpu::Trajectory stationary;
        stationary.reference[0]=1;stationary.reference[1]=2;stationary.reference[2]=3;
        require(gpu::plane_step(stationary,nx,origin,.9).factor==1,"Stationary body was damped");
        require(gpu::radius_step(stationary,stationary.reference,.1).factor==1,"Stationary radius was damped");
        require(gpu::radius_step(gpu::Trajectory{},origin,0).factor==1,
            "Exact stationary origin did not fit a zero-radius sphere");
        auto safe=translated;safe.delta_center[2]=.01;
        require(gpu::plane_step(safe,negative_z,plane_z,.9).factor==1,"No-contact update was damped");
        require(gpu::radius_step(safe,safe.reference,.5).factor==1,"Unrestricted radius update was damped");
        // The plane has a small represented gap at large world coordinates.
        // Subtract coordinates before the dot product so cancellation of two
        // large projections does not unnecessarily erase that strict gap.
        gpu::Trajectory large_world;
        large_world.center[0]=large_world.center[1]=1e16;
        const double cancelling_normal[3]={1,-1,0};
        const double cancelling_point[3]={1e16-2,1e16,0};
        require(gpu::plane_step(large_world,cancelling_normal,cancelling_point,.9).factor==1,
            "Large cancelling projections erased a certified local plane gap");

        gpu::Trajectory general;
        general.q0[0]=std::cos(.2);general.q0[1]=std::sin(.2);
        general.omega[0]=1.2;general.omega[1]=-.7;general.omega[2]=.9;
        general.delta_omega[0]=-2.1;general.delta_omega[1]=.8;general.delta_omega[2]=.3;
        general.center[0]=1e4;general.center[1]=-2e4;general.center[2]=1e4;
        general.delta_center[0]=.1;general.delta_center[1]=-.2;general.delta_center[2]=.3;
        general.reference[0]=.8;general.reference[1]=-.3;general.reference[2]=1.2;general.dt=.3;
        verify_box(general,0,1);verify_box(general,.437,.438);

        // Compare point boxes to the production quaternion path, including the
        // non-unit homogeneous map and its small-angle exponential branch.
        for(double norm_error:{-7e-9,0.,7e-9})for(double angular_scale:{0.,1e-6,.7,7.}) {
            auto varied=general;
            for(int i=0;i<4;++i)varied.q0[i]*=1+norm_error;
            for(int i=0;i<3;++i){varied.omega[i]*=angular_scale;varied.delta_omega[i]*=angular_scale;}
            verify_box(varied,.1,.2,32);
            for(int i=0;i<=32;++i) {
                const double t=double(i)/32;
                gpu::Box point{};double represented[3];
                require(gpu::trajectory_bounds(varied,t,t,point)==gpu::Status::Ok,"Point bounds failed");
                production_point(varied,t,represented);
                for(int j=0;j<3;++j)require(represented[j]>=point.lo[j]&&represented[j]<=point.hi[j],
                    "Production quaternion pose escaped point box");
            }
        }
        gpu::Trajectory scaled;
        scaled.q0[0]=1+7e-9;scaled.reference[0]=2;
        double scaled_point[3];
        require(gpu::evaluate(scaled,0,scaled_point)==gpu::Status::Ok,"Allowed quaternion norm error rejected");
        require(scaled_point[0]>2+2e-8,"Unit-only quaternion formula discarded allowed norm error");
        gpu::Box initial{};
        require(gpu::trajectory_bounds(scaled,0,0,initial)==gpu::Status::Ok,"Scaled initial bounds failed");
        const double narrow_plane[3]={2+1e-8,0,0};
        require(gpu::plane_step(scaled,negative_x,narrow_plane,.9).status==gpu::Status::InitialSideUnproved,
            "Initial pose mismatch escaped intended-side check");

        // Random deterministic changing-axis trajectories cross-check the
        // independently compiled CPU certificate as well as dense point paths.
        for(int trial=0;trial<32;++trial) {
            gpu::Trajectory varied;
            const double angle=.5*random_signed(),s=std::sin(angle);
            varied.q0[0]=std::cos(angle);varied.q0[1]=s;
            for(int i=0;i<3;++i) {
                varied.center[i]=100*random_signed();varied.delta_center[i]=random_signed();
                varied.omega[i]=3*random_signed();varied.delta_omega[i]=3*random_signed();
                varied.reference[i]=random_signed();
            }
            varied.dt=.4;
            verify_box(varied,.2,.7,32);
        }

        require(gpu::plane_step(spin,nx,px,.9,1).status==gpu::Status::DepthExhausted,
            "Depth exhaustion was not surfaced");
        const auto budget=gpu::plane_step(spin,nx,px,.9,40,1);
        require(budget.status==gpu::Status::BudgetExhausted&&budget.factor==0,
            "Unproved budget-exhausted prefix made progress");
        require(!gpu::is_fatal(budget.status),"Safe limiting status classified fatal");
        require(gpu::is_fatal(gpu::Status::InitialSideUnproved),"Initial side failure classified safe");

        auto invalid=spin;invalid.dt=std::numeric_limits<double>::quiet_NaN();
        require(gpu::plane_step(invalid,nx,px,.9).status==gpu::Status::InvalidInput,"NaN trajectory accepted");
        require(gpu::radius_step(spin,start,INFINITY).status==gpu::Status::InvalidInput,"Infinite radius accepted");
        require(gpu::plane_step(spin,origin,px,.9).status==gpu::Status::InvalidInput,"Zero normal accepted");
        require(gpu::plane_step(spin,nx,px,1.1).status==gpu::Status::InvalidInput,"Invalid gamma accepted");
        require(gpu::plane_step(spin,nx,px,.9,65).status==gpu::Status::InvalidInput,"Oversized stack depth accepted");
        invalid=spin;invalid.q0[0]=1.1;
        require(gpu::validate(invalid)==gpu::Status::InvalidInput,"Nonunit quaternion accepted");
        invalid=spin;invalid.omega[2]=1e100;
        require(gpu::trajectory_bounds(invalid,0,1,initial)==gpu::Status::AngleRange,"Unsupported angle accepted");
        invalid=spin;invalid.reference[0]=DBL_MAX;invalid.q0[0]=1+7e-9;
        require(gpu::trajectory_bounds(invalid,0,0,initial)==gpu::Status::Nonfinite,"Overflowed bounds accepted");
        require(gpu::round_down(INFINITY)==DBL_MAX&&gpu::round_up(-INFINITY)==-DBL_MAX,
            "Overflow rounding lost outward finite endpoint");
        require(gpu::round_up(INFINITY)==INFINITY&&gpu::round_down(-INFINITY)==-INFINITY,
            "Overflow rounding lost unbounded endpoint");
        std::cout<<"GPU rigid DAT host mathematical checks passed ("<<checks<<")\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}

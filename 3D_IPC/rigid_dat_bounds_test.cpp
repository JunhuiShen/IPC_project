#include "rigid_dat_trajectory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
double distance(rigid_dat::Vector a, rigid_dat::Vector b) {
    return std::hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}
void verify_box(const rigid_dat::Trajectory& path, double a, double b) {
    const auto box = rigid_dat::trajectory_bounds(path, a, b);
    for (int i=0;i<=256;++i) {
        const auto x=rigid_dat::evaluate(path,a+(b-a)*i/256.0);
        for (int j=0;j<3;++j)
            require(x[j]>=box.lo[j] && x[j]<=box.hi[j], "trajectory escaped interval box");
    }
}
bool same_bits(double a,double b) {
    return std::memcmp(&a,&b,sizeof(double))==0;
}
void verify_capped_radius(const rigid_dat::Trajectory& path,double radius) {
    const rigid_dat::PreparedTrajectory prepared(path);
    for(int iterations:{12,40}) {
        const double original=rigid_dat::radius_step(prepared,radius,iterations);
        require(same_bits(rigid_dat::radius_step_capped(prepared,radius,1,iterations),original),
            "cap one changed the original radius bound");
        // Include non-dyadic limits and both neighbors of the original result:
        // stopping the scan at cap must preserve the same completed minimum.
        const std::array<double,8> caps{{0,.01,.5,1,original,
            std::nextafter(original,0.),std::nextafter(original,1.),-0.0}};
        for(double cap:caps) {
            const double actual=rigid_dat::radius_step_capped(prepared,radius,cap,iterations);
            require(same_bits(actual,std::min(original,cap)),
                "capped radius did not preserve the bitwise minimum");
        }
    }
}
}
int main() {
    using namespace rigid_dat;
    constexpr double pi=3.14159265358979323846;
    Trajectory spin;
    spin.reference={1,0,0}; spin.delta_omega={0,0,2*pi};
    verify_box(spin,0,1); verify_box(spin,.1,.3);
    require(distance(evaluate(spin,.25),{0,1,0})<1e-14,"wrong rotation convention");
    // Every one of these two sample endpoints is safe, but the interior arc
    // crosses x=.5. Certification must catch the missed first crossing.
    const double step=plane_step(spin,{1,0,0},{.5,0,0},.9,1);
    require(step>0 && step<1.0/6,"missed between-sample crossing or froze");
    for(int i=0;i<=1000;++i)
        require(evaluate(spin,step*i/1000.0)[0]>.5,"accepted arc crossed plane");
    // Initial velocity points away from x=.5, then reverses into it.
    Trajectory away=spin; away.reference={0,1,0};
    const double turn=plane_step(away,{1,0,0},{.5,0,0},.9,1);
    require(turn>0 && turn<7.0/12,"initial-away shortcut missed returning arc");
    for(int i=0;i<=1000;++i)
        require(evaluate(away,turn*i/1000.0)[0]<.5,"return arc crossed plane");
    // A complete turn has zero endpoint displacement. Radius is a path bound.
    const double radial=radius_step(spin,.25);
    require(radial>0 && radial<.05,"radius cap used only endpoints");
    verify_capped_radius(spin,.25);
    // For this unit-radius, constant-axis turn, displacement is the chord
    // 2*sin(pi*t). Its first radius exit has an independent analytic solution.
    const double analytic_radial=std::asin(.25/2)/pi;
    require(radial<=analytic_radial && analytic_radial-radial<5e-10,
        "constant-axis radius bound disagreed with the analytic chord limit");
    const PreparedTrajectory prepared_spin(spin);
    const double capped_radial=radius_step_capped(prepared_spin,.25,.017);
    require(same_bits(capped_radial,.017),"safe non-dyadic radius cap changed");
    for(int i=0;i<=1000;++i)
        require(distance(evaluate(spin,radial*i/1000.0),evaluate(spin,0))<=.25,"radius violated");
    for(int i=0;i<=1000;++i)
        require(distance(evaluate(spin,capped_radial*i/1000.0),evaluate(spin,0))<=.25,
            "capped accepted arc violated the radius");
    require(same_bits(radius_step_capped(prepared_spin,.25,0),0),"moving zero cap changed");
    Trajectory translated; translated.reference={.1,.2,.3};
    translated.delta_center={0,0,2};
    const double linear=plane_step(translated,{0,0,1},{0,0,1.3},.9);
    require(std::abs(linear-.45)<1e-7,"linear plane step inconsistent");
    require(std::abs(radius_step(translated,.5)-.25)<1e-7,"linear radius bound inconsistent");
    verify_capped_radius(translated,.5);
    Trajectory stationary; stationary.reference={1,2,3};
    require(plane_step(stationary,{1,0,0},{0,0,0},.9)==1,"safe stationary path damped");
    require(radius_step(stationary,.1)==1,"stationary radius cap");
    const PreparedTrajectory prepared_stationary(stationary);
    for(double cap:{0.,.37,1.})
        require(same_bits(radius_step_capped(prepared_stationary,0,cap),cap),
            "stationary path did not retain its caller cap");
    for(double invalid_cap:{-std::numeric_limits<double>::denorm_min(),-1.,
            std::nextafter(1.,2.),std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected_cap=false;
        try { (void)radius_step_capped(prepared_spin,.25,invalid_cap); }
        catch(const std::invalid_argument&) { rejected_cap=true; }
        require(rejected_cap,"invalid radius cap accepted");
    }
    // The exponential-coordinate direction changes: this is not SLERP.
    Trajectory general;
    general.q0={std::cos(.2),std::sin(.2),0,0};
    general.omega={1.2,-.7,.9}; general.delta_omega={-2.1,.8,.3};
    general.center={1e4,-2e4,1e4}; general.delta_center={.1,-.2,.3};
    general.reference={.8,-.3,1.2}; general.dt=.3;
    verify_box(general,0,1); verify_box(general,.437,.438);
    verify_capped_radius(general,.2);
    Trajectory zero=general; zero.omega={0,0,0}; zero.delta_omega={0,0,0};
    verify_box(zero,0,1);
    bool rejected=false;
    try { auto invalid=spin; invalid.dt=std::numeric_limits<double>::quiet_NaN(); (void)plane_step(invalid,{1,0,0},{0,0,0},.9); }
    catch(const std::exception&) { rejected=true; }
    require(rejected,"nonfinite trajectory accepted");
    rejected=false;
    try { (void)plane_step(spin,{0,0,0},{0,0,0},.9); }
    catch(const std::exception&) { rejected=true; }
    require(rejected,"zero plane normal accepted");
    std::cout<<"Rigid DAT independent trajectory acceptance passed\n";
}

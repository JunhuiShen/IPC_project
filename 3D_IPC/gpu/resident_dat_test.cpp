#include "resident_solver.h"
#ifdef IPC_DAT_COMPONENT
#include "resident_dat.h"
#include "resident_memory.h"
#endif
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef IPC_DAT_CPU_REFERENCE
#include "resident_bridge.h"
#include "ipc_args.h"
#include "solver.h"
#endif

using namespace ipc_gpu::resident;
namespace {
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
void close(double x,double y,double eps=2e-9){if(!(std::abs(x-y)<=eps))throw std::runtime_error("DAT mismatch: "+std::to_string(x)+" versus "+std::to_string(y));}
#ifdef IPC_DAT_COMPONENT
using TestSolver=ResidentDATSolver;
#else
using TestSolver=ResidentSolver;
#endif
SceneDescription scene(std::vector<Vector3> x,bool triangle=true) {
    SceneDescription h;const auto n=x.size();h.positions=x;h.velocities.resize(n);h.pin_targets=x;h.body_reference.resize(n);
    h.deformable.assign(n,1);h.solid.assign(n,0);h.pinned.assign(n,0);h.surface.assign(n,1);h.mass.assign(n,1);h.node_body.assign(n,-1);
    for(std::size_t i=0;i<n;++i)h.node_blocks.push_back(static_cast<int>(i));h.block_count=n;
    h.triangle_offsets.assign(n+1,0);h.solid_offsets.assign(n+1,0);h.hinge_offsets.assign(n+1,0);h.body_node_offsets={0};
    if(triangle){h.surface_triangles={BpTriangle{{1,2,3}}};h.edges={BpEdge{{1,2}},BpEdge{{2,3}},BpEdge{{1,3}}};}return h;
}
SceneDescription point_triangle(double gap=.1){return scene({Vector3{0,0,gap},Vector3{-1,-1,0},Vector3{2,-1,0},Vector3{-1,2,0}});}
StepParameters parameters() {
    StepParameters p;p.planar_dat=true;p.dt=.01;p.elastic.dt2=p.dt*p.dt;p.contact={1,0,p.dt,0,.001,1e-12};p.sdf.dt=p.dt;p.sdf.eps_v=.001;
    p.iterations=1;p.dat_detection_interval=100;return p;
}
std::vector<Vector3> solve(SceneDescription h,StepParameters p) {
    TestSolver solver(h);solver.advance_substep(p,h.pin_targets,{});const auto out=solver.snapshot();
    for(const auto& x:out.positions)for(double v:x)require(std::isfinite(v),"DAT nonfinite output");
#ifdef IPC_DAT_CPU_REFERENCE
    RefMesh mesh;mesh.num_positions=h.positions.size();mesh.mass=h.mass;mesh.node_to_rb.assign(h.positions.size(),-1);
    for(const auto& t:h.surface_triangles)for(int v:t.vertices)mesh.tris.push_back(v);
    std::vector<Vec3> x,target,velocity;
    for(std::size_t i=0;i<h.positions.size();++i){x.emplace_back(h.positions[i][0],h.positions[i][1],h.positions[i][2]);velocity.emplace_back(h.velocities[i][0],h.velocities[i][1],h.velocities[i][2]);target.push_back(x.back()+p.dt*velocity.back());}
    auto cpu=SimParams::zeros();cpu.fps=1/p.dt;cpu.fixed_iters=true;cpu.max_global_iters=p.iterations;cpu.damping=p.damping;
    cpu.d_hat=p.contact.d_hat;cpu.k_barrier=p.contact.k_barrier;cpu.friction_coefficient=p.contact.friction;cpu.friction_velocity_epsilon=p.contact.eps_v;
    cpu.dat_relaxation=p.dat_relaxation;cpu.dat_detection_interval=p.dat_detection_interval;cpu.node_box_min=.001;cpu.node_box_max=.01;
    const auto previous=x;const auto result=global_gauss_seidel_solver_dat(mesh,{}, {},cpu,x,target,velocity,"",&previous);
    require(result.converged,"CPU DAT reference failed");
    for(std::size_t i=0;i<x.size();++i)for(int k=0;k<3;++k)close(out.positions[i][k],x[i][k],2e-9);
#endif
    return out.positions;
}
void point_tests() {
    auto p=parameters();
    for(int axis:{0,2}){auto h=point_triangle(.01);h.velocities[0][axis]=20;const auto x=solve(h,p);close(x[0][axis],h.positions[0][axis]+.2);}
    auto h=point_triangle();h.velocities[0][2]=-50;auto x=solve(h,p);close(x[0][2],.1-8e-10,1e-13);require(x[0][2]>0,"point crossed plane");
    h=point_triangle();for(int i=1;i<4;++i)h.velocities[i][2]=50;x=solve(h,p);for(int i=1;i<4;++i)close(x[i][2],8e-10,1e-13);
    h=point_triangle();h.velocities[0][2]=-50;for(int i=1;i<4;++i)h.velocities[i][2]=50;x=solve(h,p);close(x[0][2],.06);for(int i=1;i<4;++i)close(x[i][2],.04);
    h=point_triangle();h.velocities[0][2]=-80;for(int i=1;i<4;++i)h.velocities[i][2]=20;x=solve(h,p);close(x[0][2],.084);for(int i=1;i<4;++i)close(x[i][2],.064);
    h=point_triangle(.12);p.contact.d_hat=.1;h.velocities[0][2]=-50;for(int i=1;i<4;++i)h.velocities[i][2]=50;x=solve(h,p);close(x[0][2],.072);for(int i=1;i<4;++i)close(x[i][2],.048);
    h=point_triangle(.4);h.velocities[0][2]=-80;x=solve(h,p);close(x[0][2],.34);
}
void refresh_tests() {
    auto h=scene({Vector3{0,0,0}},false);h.velocities[0][0]=100;auto p=parameters();p.contact.d_hat=.1;p.iterations=3;
    auto x=solve(h,p);close(x[0][0],.06);p.dat_detection_interval=1;x=solve(h,p);close(x[0][0],.18);
    auto overflow=scene({Vector3{0,0,0}},false);overflow.mass[0]=1e308;overflow.velocities[0][0]=400;
    const auto retained=solve(overflow,parameters());close(retained[0][0],0);
    // DAT overrides the ordinary resident CCD predictor at solver entry.
    p.global_ccd_guess=true;x=solve(h,p);close(x[0][0],.18);
    auto empty=scene({},false);TestSolver solver(empty);solver.advance_substep(p,{},{});require(solver.snapshot().positions.empty(),"empty DAT scene");
}
void distance_and_rounding_tests() {
    auto p=parameters();auto h=scene({Vector3{.5,-.01,.1},Vector3{0,0,0},Vector3{1,0,0},Vector3{-10,.1,0}});
    h.velocities[0]={0,2,-20};auto x=solve(h,p);close(x[0][1],-.01+8e-11,1e-13);close(x[0][2],.1-8e-10,1e-13);require(x[0][2]>0,"obtuse-triangle ray crossed the edge");
    h=point_triangle();const double base=1e12,next=std::nextafter(base,INFINITY);h.positions[0][2]=next;for(int i=1;i<4;++i)h.positions[i][2]=base;
    h.velocities[0][2]=-10;p.contact.d_hat=.01;x=solve(h,p);require(x[0][2]==next,"rounded endpoint consumed plane gap");
    const double gap=next-base;h.velocities[0][2]=-gap/p.dt;for(int i=1;i<4;++i)h.velocities[i][2]=4*gap/p.dt;
    x=solve(h,p);require(x[0][2]==next,"adaptive plane accepted a rounded crossing");for(int i=1;i<4;++i)require(x[i][2]==base,"rounded triangle motion crossed its plane");
}
void edge_tests() {
    auto h=scene({Vector3{-1,0,0},Vector3{1,0,0},Vector3{-1,-1,0},Vector3{0,-1,.1},Vector3{0,1,.1},Vector3{1,1,.1}},false);
    h.surface_triangles={BpTriangle{{0,1,2}},BpTriangle{{3,4,5}}};h.edges={BpEdge{{0,1}},BpEdge{{1,2}},BpEdge{{0,2}},BpEdge{{3,4}},BpEdge{{4,5}},BpEdge{{3,5}}};
    auto p=parameters();p.contact.d_hat=.2;
    for(int node:{0,1,3,4}){auto copy=h;copy.velocities[node][2]=node<3?50:-50;const auto x=solve(copy,p);close(std::abs(x[node][2]-h.positions[node][2]),8e-10,1e-13);}
}
void force_tests() {
    auto p=parameters();p.contact.d_hat=.2;p.contact.k_barrier=100;p.contact.friction=.3;
    auto h=point_triangle(.1);h.velocities[0]={2,0,-1};const auto x=solve(h,p);require(x[0][2]>0,"barrier/friction fixture crossed");
    auto clamped=scene({Vector3{1.5,-.1,.1},Vector3{0,0,0},Vector3{1,0,0},Vector3{-10,1,0}});
    p.contact.d_hat=1;p.contact.friction=0;solve(clamped,p);
}
void validation_tests() {
    auto h=point_triangle();auto p=parameters();
    for(int kind=0;kind<5;++kind){auto bad=p;if(kind==0)bad.dat_relaxation=1;if(kind==1)bad.dat_detection_interval=0;if(kind==2)bad.contact.d_hat=0;if(kind==3)bad.iterations=0;if(kind==4)bad.dat_relaxation=std::numeric_limits<double>::quiet_NaN();TestSolver solver(h);bool rejected=false;try{solver.advance_substep(bad,h.pin_targets,{});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid DAT parameters accepted");solver.advance_substep(p,h.pin_targets,{});}
    auto overlap=point_triangle(0);auto zero_step=p;zero_step.contact.k_barrier=100;TestSolver touching(overlap);bool failed=false;
    try{touching.advance_substep(zero_step,overlap.pin_targets,{});}catch(const std::runtime_error&){failed=true;}require(failed,"zero-gap contact must fail like CPU DAT");
    failed=false;try{touching.snapshot();}catch(const std::runtime_error&){failed=true;}require(failed,"failed DAT state exported");
    h.solid[0]=1;bool rejected=false;try{TestSolver solver(h);solver.advance_substep(p,h.pin_targets,{});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"solid DAT scene accepted");
#if defined(IPC_DAT_CPU_REFERENCE) && !defined(IPC_DAT_COMPONENT)
    IPCArgs3D args;args.use_dat_solver=true;args.fixed_iters=true;args.exact_computation_fallback=false;args.use_colored_ccd_guess=true;
    const auto step=describe_step(args.to_sim_params());require(step.planar_dat&&!step.global_ccd_guess&&!step.colored_guess_sweeps&&!step.colored_guess_level_based,"DAT bridge did not override predictor");
#endif
}
}
int main(){try{point_tests();std::cout<<"point fixtures passed"<<std::endl;refresh_tests();std::cout<<"refresh fixtures passed"<<std::endl;distance_and_rounding_tests();std::cout<<"distance/rounding fixtures passed"<<std::endl;edge_tests();std::cout<<"edge fixtures passed"<<std::endl;force_tests();std::cout<<"force fixtures passed"<<std::endl;
validation_tests();
std::cout<<"GPU DAT: all fixtures passed";
#ifdef IPC_DAT_CPU_REFERENCE
std::cout<<" (including CPU DAT comparisons)";
#endif
std::cout<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

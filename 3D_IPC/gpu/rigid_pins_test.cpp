#include "rigid_pins.cuh"
#include "rigid_pins_test.h"
#include "rigid_body_ipc.h"
#include "physics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::solver;

#ifdef IPC_GPU_RIGID_PINS_HOST_REFERENCE
namespace ipc_gpu::solver {
void rigid_pin_oracle_cuda(const RigidPinOracleInput* input,std::size_t count,
    RigidDerivatives* result,int* valid,double* targets) {
    for(std::size_t i=0;i<count;++i) {
        const auto& p=input[i];
        const auto kin=rigid_kinematics(p.previous_orientation,p.omega,p.dt);
        valid[i]=evaluate_rigid_pin(p.pin,p.com,kin,p.dt,p.kpin,p.time,
            p.with_orientation_hessian!=0,result[i]) && rigid_pin_target(p.pin,p.time,targets+3*i);
    }
}
} // namespace ipc_gpu::solver
#endif

namespace {
std::size_t comparisons=0;
double max_error=0;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void compare(double a,double b,const char* label) {
    require(std::isfinite(a)&&std::isfinite(b),"nonfinite rigid-pin oracle value");
    const double error=std::abs(a-b)/(1+std::abs(b));
    max_error=std::max(max_error,error);++comparisons;
    if(error>2e-10)throw std::runtime_error(std::string(label)+" differs from CPU: "+std::to_string(error));
}
::RigidPin cpu_pin(const RigidPinConstraint& pin) {
    ::RigidPin result;result.body=pin.body;
    for(int k=0;k<3;++k) {
        result.local_position[k]=pin.local_position[k];result.initial_target[k]=pin.initial_target[k];
        result.rotation_center[k]=pin.rotation_center[k];
    }
    result.angular_velocity_z=pin.angular_velocity_z;return result;
}
void compare_derivatives(const RigidDerivatives& actual,const RigidEnergyDerivatives& expected) {
    for(int r=0;r<3;++r) {
        compare(actual.translation_gradient[r],expected.translation_gradient[r],"COM gradient");
        compare(actual.orientation_gradient[r],expected.orientation_gradient[r],"omega gradient");
        for(int c=0;c<3;++c) {
            compare(actual.translation_translation_hessian[3*r+c],expected.translation_translation_hessian(r,c),"COM Hessian");
            compare(actual.translation_orientation_hessian[3*r+c],expected.translation_orientation_hessian(r,c),"mixed Hessian");
            compare(actual.orientation_orientation_hessian[3*r+c],expected.orientation_orientation_hessian(r,c),"omega Hessian");
        }
    }
}
std::vector<RigidPinOracleInput> fixtures() {
    std::vector<RigidPinOracleInput> out;
    const std::array<Vec4,3> quaternions{Vec4(1,0,0,0),Vec4(.8,.2,-.4,.4).normalized(),Vec4(-.3,.2,.7,-.5).normalized()};
    for(const auto& q:quaternions)for(double dt:{0.,1./600.,.031,1.})
        for(double stiffness:{0.,1.,1e7})for(double time:{0.,.17,1.3})
            for(double angular:{0.,.7,-1.2})for(bool second:{false,true}) {
                RigidPinOracleInput p{};p.pin.body=2;p.dt=dt;p.kpin=stiffness;p.time=time;
                p.with_orientation_hessian=second;
                const double local[3]{.2,-.7,1.3},target[3]{-.1,.4,.8},center[3]{.5,-.2,.9};
                for(int k=0;k<3;++k) {
                    p.pin.local_position[k]=local[k];p.pin.initial_target[k]=target[k];
                    p.pin.rotation_center[k]=.1*(k-1);p.com[k]=center[k];p.omega[k]=.3*(k-1);
                }
                for(int k=0;k<4;++k)p.previous_orientation[k]=q[k];
                p.pin.angular_velocity_z=angular;out.push_back(p);
            }
    // Exact zero and both sides of the quaternion Taylor-branch threshold.
    for(double magnitude:{0.,1e-14,2e-4*(1.-1e-8),2e-4,2e-4*(1.+1e-8),30.}) {
        auto p=out.back();p.dt=1.;p.kpin=3.;p.omega[0]=magnitude;p.omega[1]=p.omega[2]=0.;out.push_back(p);
    }
    auto p=out.back();p.pin.local_position[0]=p.pin.local_position[1]=p.pin.local_position[2]=0.;out.push_back(p);
    return out;
}
void oracle_test() {
    const auto inputs=fixtures();
    std::vector<RigidDerivatives> values(inputs.size());std::vector<int> valid(inputs.size());std::vector<double> targets(3*inputs.size());
    rigid_pin_oracle_cuda(inputs.data(),inputs.size(),values.data(),valid.data(),targets.data());
    for(std::size_t i=0;i<inputs.size();++i) {
        const auto& p=inputs[i];require(valid[i]!=0,"finite pin rejected by device evaluator");
        auto params=SimParams::zeros();params.kpin=p.kpin;params.rigid_pin_time=p.time;
        params.rigid_pins={cpu_pin(p.pin)};
        const Vec3 com(p.com[0],p.com[1],p.com[2]),omega(p.omega[0],p.omega[1],p.omega[2]);
        const Vec4 q(p.previous_orientation[0],p.previous_orientation[1],p.previous_orientation[2],p.previous_orientation[3]);
        const auto expected=rigid_pin_derivatives(2,com,q,omega,params,p.dt,nullptr,p.with_orientation_hessian!=0);
        compare_derivatives(values[i],expected);
        const auto target=rigid_pin_target_position(params.rigid_pins[0],p.time);
        for(int k=0;k<3;++k)compare(targets[3*i+k],target[k],"rotating pin target");
    }
}
RigidPinAssemblyFixture assembly_fixture(double time) {
    RigidPinAssemblyFixture f;f.dt=.031;f.kpin=83.;f.time=time;
    for(std::size_t b=0;b<f.bodies;++b) {
        const Vec4 q=Vec4(.8,.1*(b+1),-.2,.4).normalized();
        for(int k=0;k<4;++k)f.previous_orientation[b][k]=q[k];
        for(int k=0;k<3;++k) {
            f.com[b][k]=.3*(k+1)+.2*b;f.omega[b][k]=.4*(k-1)+.1*b;
            f.committed_com[b][k]=f.com[b][k]+(f.translate[b]?.07*(k-1):0.);
            f.initial[b].gradient[k]=.11*(k+1);
            for(int c=0;c<3;++c)f.initial[b].hessian[3*k+c]=k==c?3.+k:.01*(k+c+1);
        }
        for(auto i=f.offsets[b];i<f.offsets[b+1];++i) {
            auto& p=f.pins[i];p.body=static_cast<int>(b);
            for(int k=0;k<3;++k) {
                p.local_position[k]=.15*(k+1)+.03*i;p.initial_target[k]=.2*(k-1)-.1*i;
                p.rotation_center[k]=.05*k;
            }
            p.angular_velocity_z=i%2?.7:-1.1;
        }
        auto params=SimParams::zeros();params.kpin=f.kpin;params.rigid_pin_time=time;
        for(std::size_t i=0;i<f.count;++i)params.rigid_pins.push_back(cpu_pin(f.pins[i]));
        const Vec3 omega(f.omega[b][0],f.omega[b][1],f.omega[b][2]);
        const Vec3 com(f.com[b][0],f.com[b][1],f.com[b][2]);
        const Vec3 committed(f.committed_com[b][0],f.committed_com[b][1],f.committed_com[b][2]);
        const auto translation=rigid_pin_derivatives(static_cast<int>(b),com,q,omega,params,f.dt);
        const auto orientation=rigid_pin_derivatives(static_cast<int>(b),committed,q,omega,params,f.dt,nullptr,true);
        f.translation_expected[b]=f.orientation_expected[b]=f.initial[b];
        for(int r=0;r<3;++r)for(int phase=0;phase<2;++phase) {
            if(!(phase?f.rotate[b]:f.translate[b]))continue;
            auto& actual=phase?f.orientation_expected[b]:f.translation_expected[b];
            const auto& expected=phase?orientation:translation;
            actual.gradient[r]+=(phase?expected.orientation_gradient:expected.translation_gradient)[r];
            for(int c=0;c<3;++c)actual.hessian[3*r+c]+=(phase?expected.orientation_orientation_hessian:expected.translation_translation_hessian)(r,c);
        }
    }
    return f;
}
void aggregate_and_failure_test() {
    auto f=assembly_fixture(.17);RigidPinScene pins{f.bodies,f.count,f.pins,f.offsets};
    for(std::size_t b=0;b<f.bodies;++b)for(bool second:{false,true}) {
        const auto kin=rigid_kinematics(f.previous_orientation[b],f.omega[b],f.dt);
        RigidDerivatives actual{};require(evaluate_rigid_body_pins(pins,b,f.com[b],kin,f.dt,f.kpin,f.time,second,actual)==0,"valid body pin aggregate rejected");
        auto params=SimParams::zeros();params.kpin=f.kpin;params.rigid_pin_time=f.time;
        for(std::size_t i=0;i<f.count;++i)params.rigid_pins.push_back(cpu_pin(f.pins[i]));
        const Vec3 com(f.com[b][0],f.com[b][1],f.com[b][2]),omega(f.omega[b][0],f.omega[b][1],f.omega[b][2]);
        const Vec4 q(f.previous_orientation[b][0],f.previous_orientation[b][1],f.previous_orientation[b][2],f.previous_orientation[b][3]);
        compare_derivatives(actual,rigid_pin_derivatives(static_cast<int>(b),com,q,omega,params,f.dt,nullptr,second));
    }
    const auto kin=rigid_kinematics(f.previous_orientation[0],f.omega[0],f.dt);
    RigidDerivatives result{};
    f.pins[1].initial_target[0]=std::numeric_limits<double>::quiet_NaN();
    require(evaluate_rigid_body_pins(pins,0,f.com[0],kin,f.dt,f.kpin,f.time,true,result)==RigidPinInvalidInput,"nonfinite pin was accepted");
    const RigidDerivatives zero{};require(std::memcmp(&result,&zero,sizeof(result))==0,"failed aggregate published a partial result");
    f.pins[1].initial_target[0]=0.;f.pins[1].body=3;
    require(evaluate_rigid_body_pins(pins,0,f.com[0],kin,f.dt,f.kpin,f.time,true,result)==RigidPinInvalidInput,"wrong pin owner accepted");
    f.pins[1].body=0;f.offsets[1]=f.count+1;
    require(evaluate_rigid_body_pins(pins,0,f.com[0],kin,f.dt,f.kpin,f.time,true,result)==RigidPinInvalidInput,"invalid pin offset accepted");
    f.offsets[1]=2;f.offsets[0]=1;
    require(evaluate_rigid_body_pins(pins,0,f.com[0],kin,f.dt,f.kpin,f.time,true,result)==RigidPinInvalidInput,"omitted leading pin accepted");
    f.offsets[0]=0;f.offsets[f.bodies]=f.count-1;
    require(evaluate_rigid_body_pins(pins,0,f.com[0],kin,f.dt,f.kpin,f.time,true,result)==RigidPinInvalidInput,"omitted trailing pin accepted");
    require(evaluate_rigid_body_pins(pins,0,nullptr,kin,0.,0.,std::numeric_limits<double>::quiet_NaN(),true,result)==0,"disabled pins lost CPU no-work semantics");
}
} // namespace

int main() {
    try {
        oracle_test();aggregate_and_failure_test();
#ifdef IPC_GPU_RIGID_PINS_HOST_REFERENCE
        std::cout<<"HOST ARITHMETIC ONLY (not CUDA validation): ";
#else
        for(double time:{0.,.17,1.3})rigid_pin_assembly_cuda(assembly_fixture(time));
        std::cout<<"CUDA rigid-pin CPU oracle and color assembly: ";
#endif
        std::cout<<comparisons<<" components, max normalized error "<<max_error<<"; virtual/rotating/multiple pins, COM/omega Hessians, current COM, disabled modes and failure checks PASS\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Rigid-pin test failed: "<<error.what()<<'\n';return 1;
    }
}

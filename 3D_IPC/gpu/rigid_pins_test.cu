#include "rigid_pins_test.h"
#include "rigid_pins.cuh"
#include "scene_types.h"
#include "test_support.h"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ipc_gpu::solver {
namespace {

__global__ void assemble(RigidScene scene,RigidState state,RigidWorkspace work,
    RigidPinScene pins,const int* colors,int color,bool orientation,double dt,
    double kpin,double time,int* error) {
    const auto body=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (body>=scene.bodies || atomicAdd(error,0)) return;
    if (colors[scene.body_blocks[body]]!=color || !(orientation?scene.rotate[body]:scene.translate[body])) return;
    const auto kin=rigid_kinematics(state.previous_orientation[body],state.omega[body],dt);
    RigidDerivatives pin{};
    const int status=evaluate_rigid_body_pins(pins,body,state.com[body],kin,dt,kpin,time,orientation,pin);
    if (status) {atomicCAS(error,0,status);return;}
    Derivatives system=work.systems[body];
    const double* gradient=orientation?pin.orientation_gradient:pin.translation_gradient;
    const double* hessian=orientation?pin.orientation_orientation_hessian:pin.translation_translation_hessian;
    for (int k=0;k<3;++k) system.gradient[k]+=gradient[k];
    for (int k=0;k<9;++k) system.hessian[k]+=hessian[k];
    for (int k=0;k<3;++k) if (!isfinite(system.gradient[k])) {atomicCAS(error,0,RigidPinInvalidDerivative);return;}
    for (int k=0;k<9;++k) if (!isfinite(system.hessian[k])) {atomicCAS(error,0,RigidPinInvalidDerivative);return;}
    work.systems[body]=system;
}

// Test-only launch wrapper for the shared device evaluator.
void assemble_rigid_pins_color(const RigidScene& scene,const RigidState& state,
    const RigidWorkspace& work,const RigidPinScene& pins,const int* colors,
    int color,bool orientation,double dt,double kpin,double time,int* error,
    cudaStream_t stream) {
    if (!pins.count || kpin<=0) return;
    if (!std::isfinite(dt) || dt<=0 || !std::isfinite(kpin) || !std::isfinite(time) ||
        color<0 || pins.bodies!=scene.bodies || !pins.pins || !pins.offsets ||
        !colors || !error || !work.systems || !scene.body_blocks || !scene.translate ||
        !scene.rotate || !state.com || !state.previous_orientation || !state.omega)
        throw std::invalid_argument("GPU rigid pin assembly has invalid inputs");
    if (!scene.bodies) throw std::invalid_argument("rigid pins require a body");
    constexpr unsigned width=128;
    assemble<<<static_cast<unsigned>((scene.bodies+width-1)/width),width,0,stream>>>(
        scene,state,work,pins,colors,color,orientation,dt,kpin,time,error);
    const auto status=cudaGetLastError();
    if (status!=cudaSuccess)
        throw std::runtime_error(std::string("GPU rigid pins CUDA launch: ")+cudaGetErrorString(status));
}

__global__ void oracle(const RigidPinOracleInput* input,std::size_t count,
    RigidDerivatives* result,int* valid,double* targets) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto p=input[i];
    const auto kin=rigid_kinematics(p.previous_orientation,p.omega,p.dt);
    valid[i]=evaluate_rigid_pin(p.pin,p.com,kin,p.dt,p.kpin,p.time,
        p.with_orientation_hessian!=0,result[i]) && rigid_pin_target(p.pin,p.time,targets+3*i);
}

void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void compare_system(const Derivatives& actual,const Derivatives& expected) {
    for(int k=0;k<3;++k)
        require(std::isfinite(actual.gradient[k])&&std::abs(actual.gradient[k]-expected.gradient[k])<=2e-10*(1.+std::abs(expected.gradient[k])),
            "rigid-pin color kernel gradient differs from CPU");
    for(int k=0;k<9;++k)
        require(std::isfinite(actual.hessian[k])&&std::abs(actual.hessian[k]-expected.hessian[k])<=2e-10*(1.+std::abs(expected.hessian[k])),
            "rigid-pin color kernel Hessian differs from CPU");
}
template<class T> std::vector<T> list(const T* begin,std::size_t count) {
    return {begin,begin+count};
}

} // namespace

void rigid_pin_oracle_cuda(const RigidPinOracleInput* input,std::size_t count,
    RigidDerivatives* result,int* valid,double* targets) {
    if(!count)return;
    GpuTestBuffer<RigidPinOracleInput> inputs(list(input,count));
    GpuTestBuffer<RigidDerivatives> outputs(count);
    GpuTestBuffer<int> flags(count);
    GpuTestBuffer<double> target(3*count);
    oracle<<<static_cast<unsigned>((count+127)/128),128>>>(inputs.data(),count,outputs.data(),flags.data(),target.data());
    gpu_cuda_check(cudaGetLastError());gpu_cuda_check(cudaDeviceSynchronize());
    gpu_cuda_check(cudaMemcpy(result,outputs.data(),count*sizeof(*result),cudaMemcpyDeviceToHost));
    gpu_cuda_check(cudaMemcpy(valid,flags.data(),count*sizeof(*valid),cudaMemcpyDeviceToHost));
    gpu_cuda_check(cudaMemcpy(targets,target.data(),3*count*sizeof(*targets),cudaMemcpyDeviceToHost));
}

void rigid_pin_assembly_cuda(const RigidPinAssemblyFixture& f) {
    require(f.bodies==4&&f.count==6,"unexpected rigid-pin oracle fixture extent");
    GpuTestBuffer<RigidPinConstraint> pin_data(list(f.pins,f.count));
    GpuTestBuffer<std::size_t> offsets(list(f.offsets,f.bodies+1));
    GpuTestBuffer<int> blocks(list(f.body_blocks,f.bodies)),colors(list(f.colors,8));
    GpuTestBuffer<unsigned char> translate(list(f.translate,f.bodies)),rotate(list(f.rotate,f.bodies));
    GpuTestBuffer<double> com(list(&f.com[0][0],3*f.bodies));
    GpuTestBuffer<double> q(list(&f.previous_orientation[0][0],4*f.bodies));
    GpuTestBuffer<double> omega(list(&f.omega[0][0],3*f.bodies));
    GpuTestBuffer<Derivatives> systems(list(f.initial,f.bodies));
    GpuTestBuffer<int> status(std::vector<int>{0});
    RigidScene scene{};scene.bodies=f.bodies;scene.body_blocks=blocks.data();
    scene.translate=translate.data();scene.rotate=rotate.data();
    RigidState state{};state.com=reinterpret_cast<double(*)[3]>(com.data());
    state.previous_orientation=reinterpret_cast<double(*)[4]>(q.data());state.omega=reinterpret_cast<double(*)[3]>(omega.data());
    RigidWorkspace workspace{};workspace.systems=systems.data();
    RigidPinScene pins{f.bodies,f.count,pin_data.data(),offsets.data()};
    cudaStream_t stream=nullptr;gpu_cuda_check(cudaStreamCreate(&stream));
    try {
        for(bool orientation:{false,true}) {
            gpu_cuda_check(cudaMemcpy(systems.data(),f.initial,f.bodies*sizeof(Derivatives),cudaMemcpyHostToDevice));
            if(orientation)gpu_cuda_check(cudaMemcpy(com.data(),f.committed_com,3*f.bodies*sizeof(double),cudaMemcpyHostToDevice));
            bool visited[4]{};
            for(int color:{0,1}) {
                assemble_rigid_pins_color(scene,state,workspace,pins,colors.data(),color,orientation,
                    f.dt,f.kpin,f.time,status.data(),stream);
                gpu_cuda_check(cudaStreamSynchronize(stream));
                require(status.download()[0]==0,"finite rigid-pin color assembly failed");
                const auto actual=systems.download();
                for(std::size_t body=0;body<f.bodies;++body) {
                    if(f.colors[f.body_blocks[body]]==color)visited[body]=true;
                    compare_system(actual[body],visited[body]?(orientation?f.orientation_expected[body]:f.translation_expected[body]):f.initial[body]);
                }
            }
        }
        // The second pin fails after a valid first pin. Its body must never
        // publish a partially accumulated spring block.
        auto bad=list(f.pins,f.count);bad[1].initial_target[0]=std::numeric_limits<double>::quiet_NaN();
        gpu_cuda_check(cudaMemcpy(pin_data.data(),bad.data(),bad.size()*sizeof(bad[0]),cudaMemcpyHostToDevice));
        gpu_cuda_check(cudaMemcpy(systems.data(),f.initial,f.bodies*sizeof(Derivatives),cudaMemcpyHostToDevice));
        assemble_rigid_pins_color(scene,state,workspace,pins,colors.data(),0,true,f.dt,f.kpin,f.time,status.data(),stream);
        gpu_cuda_check(cudaStreamSynchronize(stream));
        require(status.download()[0]==RigidPinInvalidInput,"nonfinite rigid pin did not fail closed on device");
        compare_system(systems.download()[0],f.initial[0]);
        // Existing error status blocks all later writes, including valid pins.
        gpu_cuda_check(cudaMemcpy(pin_data.data(),f.pins,f.count*sizeof(f.pins[0]),cudaMemcpyHostToDevice));
        gpu_cuda_check(cudaMemcpy(systems.data(),f.initial,f.bodies*sizeof(Derivatives),cudaMemcpyHostToDevice));
        assemble_rigid_pins_color(scene,state,workspace,pins,colors.data(),0,false,f.dt,f.kpin,f.time,status.data(),stream);
        gpu_cuda_check(cudaStreamSynchronize(stream));
        const auto unchanged=systems.download();
        for(std::size_t b=0;b<f.bodies;++b)compare_system(unchanged[b],f.initial[b]);
        bool rejected=false;
        try {
            assemble_rigid_pins_color(scene,state,workspace,pins,colors.data(),0,false,f.dt,f.kpin,
                std::numeric_limits<double>::quiet_NaN(),status.data(),stream);
        } catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"nonfinite rigid-pin time bypassed prelaunch validation");
        // Empty/inactive pins perform no launch, allocation or state access.
        assemble_rigid_pins_color({}, {}, {}, {}, nullptr,0,false,0,0,0,nullptr,stream);
        assemble_rigid_pins_color({}, {}, {}, pins,nullptr,0,false,0,0,0,nullptr,stream);
        gpu_cuda_check(cudaStreamDestroy(stream));
    } catch(...) {
        cudaStreamDestroy(stream);throw;
    }
}

} // namespace ipc_gpu::solver

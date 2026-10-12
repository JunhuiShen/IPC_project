#include "rigid_bridge.h"
#include "rigid_math.cuh"
#include "solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// This opt-in local check is useful on Macs without CUDA. The normal CMake
// target never defines it and always executes the genuine CUDA entry points.
#ifdef IPC_GPU_RIGID_HOST_ONLY
namespace ipc_gpu {
void compute_rigid(const RigidPreparedContact* inputs, std::size_t count, RigidResult* outputs) {
    if (count && (!inputs || !outputs)) throw std::invalid_argument("host reference: null pointer");
    for (std::size_t i = 0; i < count; ++i) outputs[i] = evaluate_rigid(inputs[i]);
}
void compute_rigid_inertia(const RigidInertia* inputs, std::size_t count, Derivatives* outputs) {
    if (count && (!inputs || !outputs)) throw std::invalid_argument("host reference: null pointer");
    for (std::size_t i = 0; i < count; ++i) outputs[i] = evaluate_rigid_inertia(inputs[i]);
}
} // namespace ipc_gpu
#endif

namespace {
constexpr double dt = .031, d_hat = .55, stiffness = 370.;
std::size_t components = 0;
double max_tolerance_fraction = 0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class A, class B>
void compare(const A& actual, const B& expected, const std::string& context) {
    require(actual.allFinite() && expected.allFinite(),context + ": nonfinite derivative");
    const double tolerance = 2e-10 * (1.0 + expected.norm());
    const double error = (actual-expected).norm();
    max_tolerance_fraction = std::max(max_tolerance_fraction,error/tolerance);
    require(error <= tolerance,context + ": relative norm error " + std::to_string(error/(1+expected.norm())));
    components += actual.size();
}

void compare(const RigidEnergyDerivatives& a, const RigidEnergyDerivatives& b, const std::string& label) {
    compare(a.translation_gradient,b.translation_gradient,label + " COM gradient");
    compare(a.orientation_gradient,b.orientation_gradient,label + " omega gradient");
    compare(a.translation_translation_hessian,b.translation_translation_hessian,label + " COM Hessian");
    compare(a.translation_orientation_hessian,b.translation_orientation_hessian,label + " mixed Hessian");
    compare(a.orientation_orientation_hessian,b.orientation_orientation_hessian,label + " omega Hessian");
}

void compare(const ipc_simd::RigidContactOutput& a, const ipc_simd::RigidContactOutput& b, const std::string& label) {
    compare(a.barrier,b.barrier,label + " barrier");
    compare(a.friction,b.friction,label + " friction");
}

ipc_simd::RigidContactInput fixture(int index, const QuaternionOmegaKinematics& kinematics) {
    ipc_simd::RigidContactInput input;
    input.segment_segment = (index/2)%2 != 0;
    input.side = index%2 == 0 ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
    input.kinematics = &kinematics;
    const int feature = (index/4)%6;
    if (!input.segment_segment) {
        input.positions = {Vec3(.2,.25,.17),Vec3::Zero(),Vec3(1,0,0),Vec3(0,1,0)};
        if (feature == 1) input.positions[0] = Vec3(.4,-.13,.16);
        if (feature == 2) input.positions[0] = Vec3(-.12,-.11,.16);
        if (feature == 3) input.positions[0][2] *= -1;
        if (feature == 4) input.positions[3] = Vec3(.6,0,0);
        if (feature == 5) input.positions[0][2] = .9;
    } else {
        input.positions = {Vec3(-.7,0,0),Vec3(.7,0,0),Vec3(0,-.6,.17),Vec3(0,.6,.17)};
        if (feature == 1) { input.positions[2][1] = .12; input.positions[3][1] = .9; }
        if (feature == 2) { input.positions[2] = Vec3(.9,.12,.17); input.positions[3] = Vec3(.9,.9,.17); }
        if (feature == 3) { input.positions[2] = Vec3(-.4,.12,.17); input.positions[3] = Vec3(.5,.12,.17); }
        if (feature == 4) input.positions[1] = input.positions[0];
        if (feature == 5) { input.positions[2][2] = .9; input.positions[3][2] = .9; }
    }
    const double slip_scale = index%7 == 0 ? 0.0 : (index%7 == 1 ? 1e-6 : 1.0);
    for (int role = 0; role < 4; ++role) {
        input.body_references[role] = input.positions[role] + Vec3(.4,-.7,.9);
        input.previous_positions[role] = input.positions[role] - slip_scale*(role+1)*Vec3(.001,-.003,.002);
    }
    return input;
}

std::vector<ipc_simd::RigidContactOutput> reference(
    const std::vector<ipc_simd::RigidContactInput>& inputs, std::size_t count,
    RigidDerivativeMode mode, double friction) {
    std::vector<ipc_simd::RigidContactOutput> result(count);
    for (std::size_t begin = 0; begin < count; begin += ipc_simd::contact_tile_width) {
        const auto n = std::min(ipc_simd::contact_tile_width,count-begin);
        ipc_simd::rigid_contact_derivatives_tile(inputs.data()+begin,n,d_hat,stiffness,
            friction,dt,.1,mode,result.data()+begin);
    }
    return result;
}

void check_contacts() {
    const auto kinematics = quaternion_omega_kinematics(Vec4(.8,.2,-.4,.4).normalized(),Vec3(1.2,-.9,1.7),dt,true);
    std::vector<ipc_simd::RigidContactInput> inputs(129);
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = fixture(static_cast<int>(i),kinematics);
    for (auto mode : {RigidDerivativeMode::Full,RigidDerivativeMode::Gradient,
            RigidDerivativeMode::TranslationHessian,RigidDerivativeMode::OrientationHessian}) {
        for (double friction : {0.0,.37}) {
            for (int update = 0; update < 4; ++update) {
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                    inputs[i].update_mode = static_cast<RigidBodyUpdateMode>(update);
                    const unsigned masks[] = {0,15,1,10};
                    inputs[i].body_role_mask = masks[(i/24)%4];
                }
                for (std::size_t count : {std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(17),std::size_t(32),std::size_t(33),inputs.size()}) {
                    std::vector<ipc_gpu::RigidPreparedContact> prepared(count);
                    ipc_gpu::RigidResult canary{};
                    canary.barrier.translation_gradient[0] = 1234567.25;
                    std::vector<ipc_gpu::RigidResult> results(count+2,canary);
                    ipc_gpu::prepare_rigid_contacts(inputs.data(),count,d_hat,stiffness,friction,dt,.1,mode,prepared.data());
                    ipc_gpu::compute_rigid(prepared.data(),count,results.data()+1);
                    require(std::memcmp(&results.front(),&canary,sizeof canary) == 0
                        && std::memcmp(&results.back(),&canary,sizeof canary) == 0,"rigid kernel overwrote a guard");
                    const auto expected = reference(inputs,count,mode,friction);
                    for (std::size_t i = 0; i < count; ++i)
                        compare(ipc_gpu::unpack_rigid(results[i+1]),expected[i],"contact " + std::to_string(i)
                            + " mode=" + std::to_string(int(mode)) + " mask=" + std::to_string(update));
                }
            }
        }
    }

    // Ill-conditioned segment geometry deliberately enters the production
    // robust/exact CPU path before the same GPU contraction.
    const std::array<Vec3,4> tiny_gap{
        Vec3(.30643120172698846,1.2537041974945615,.345215273017485),
        Vec3(.2716303759805404,1.2435156840614272,.3522099917587509),
        Vec3(.3476899357030978,1.254750762137517,.34482031348588066),
        Vec3(.12785112730426101,1.247125469633767,.34839148080369126)};
    inputs.resize(2);
    for (int i = 0; i < 2; ++i) {
        inputs[i] = fixture(2+i,kinematics);
        inputs[i].positions = tiny_gap;
        for (int r = 0; r < 4; ++r) {
            inputs[i].body_references[r] = tiny_gap[r] + Vec3(.4,-.7,.9);
            inputs[i].previous_positions[r] = tiny_gap[r] - (r+1)*Vec3(.001,-.003,.002);
        }
    }
    require(segment_segment_distance(tiny_gap[0],tiny_gap[1],tiny_gap[2],tiny_gap[3]).robust,"tiny-gap fixture must activate robust geometry");
    for (auto mode : {RigidDerivativeMode::Full,RigidDerivativeMode::Gradient,
            RigidDerivativeMode::TranslationHessian,RigidDerivativeMode::OrientationHessian}) {
        for (double friction : {0.0,.37}) {
            ipc_gpu::RigidPreparedContact prepared[2];
            ipc_gpu::RigidResult results[2];
            ipc_gpu::prepare_rigid_contacts(inputs.data(),2,d_hat,stiffness,friction,dt,.1,mode,prepared);
            ipc_gpu::compute_rigid(prepared,2,results);
            const auto expected = reference(inputs,2,mode,friction);
            for (int i = 0; i < 2; ++i) compare(ipc_gpu::unpack_rigid(results[i]),expected[i],"robust tiny gap");
        }
    }

    ipc_gpu::compute_rigid(nullptr,0,nullptr);
    auto missing = fixture(1,kinematics);
    missing.kinematics = nullptr;
    ipc_gpu::RigidPreparedContact prepared;
    bool rejected = false;
    try { ipc_gpu::prepare_rigid_contacts(&missing,1,d_hat,stiffness,0,dt,.1,RigidDerivativeMode::Full,&prepared); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected,"missing orientation kinematics must be rejected");
    ipc_gpu::prepare_rigid_contacts(&missing,1,d_hat,stiffness,0,dt,.1,RigidDerivativeMode::TranslationHessian,&prepared);
#ifndef IPC_GPU_RIGID_HOST_ONLY
    const auto valid_prepared = prepared;
    prepared.gradients[0][0] = std::numeric_limits<double>::quiet_NaN();
    ipc_gpu::RigidResult result;
    rejected = false;
    try { ipc_gpu::compute_rigid(&prepared,1,&result); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected,"CUDA rigid kernel must reject nonfinite output");
    ipc_gpu::compute_rigid(&valid_prepared,1,&result);
    require(ipc_gpu::unpack_rigid(result).barrier.translation_gradient.allFinite(),"CUDA workspace must recover after rejection");
#endif
}

void check_inertia() {
    for (int i = 0; i < 16; ++i) {
        const double step = i%3 == 0 ? 1e-5 : (i%3 == 1 ? dt : .3);
        const double mass = i%5 == 0 ? 0.0 : .3 + i;
        const Vec3 current(.3*i,-.04*i,.01*i), previous = current + Vec3(.02,-.01,.003);
        const Vec3 velocity(.4,-.03,.12);
        const auto translation = ipc_gpu::rigid_translation_inertia_gravity(current,previous,velocity,step,mass,-9.81);
        const Vec3 expected_g = inertia_translation_gradient(current,previous,velocity,step,mass)
            - gravitational_potential_gradient(mass,-9.81,step);
        compare(translation.first,expected_g,"translation inertia/gravity");
        compare(translation.second,inertia_translation_hessian(mass),"translation inertia Hessian");
        const Vec4 q = Vec4(.8,.1,-.3,.4).normalized();
        const Vec3 omega = i%4 == 0 ? Vec3::Zero() : (i%4 == 1 ? Vec3(1e-8,-2e-8,3e-8) : Vec3(.7,-.5,1.2)*(i+1));
        const Vec3 omega_n(.13,-.18,.31);
        Mat33 factor;
        factor << .7,.1,-.04, .12,.9,.08, -.07,.03,.5;
        const Mat33 moment = (factor*factor.transpose())*(1+i*.1);
        const bool full_cache = i%2 == 0;
        const auto kinematics = quaternion_omega_kinematics(q,omega,step,full_cache);
        const Mat33 predictor = rigid_rotation_predictor(q,omega_n,step);
        const auto* kin = i%3 == 0 ? nullptr : &kinematics;
        const auto* pred = i%3 == 1 ? nullptr : &predictor;
        const auto actual = ipc_gpu::rigid_rotation_inertia(omega,q,omega_n,step,moment,kin,pred);
        const auto expected = inertia_rotation_gradient_hessian(omega,q,omega_n,step,moment,kin,pred);
        compare(actual.first,expected.first,"rotation inertia gradient");
        compare(actual.second,expected.second,"rotation inertia Hessian");
    }
    ipc_gpu::compute_rigid_inertia(nullptr,0,nullptr);
}

void check_body_gather() {
    RefMesh mesh;
    DeformedState state;
    state.orientations = {Vec4(1,0,0,0)};
    const std::vector<Vec3> omega = {Vec3(.4,-.8,.6)};
    const auto kinematics = quaternion_omega_kinematics(state.orientations[0],omega[0],dt,true);
    const auto nt = fixture(1,kinematics), ss = fixture(2,kinematics);
    std::vector<Vec3> positions(nt.positions.begin(),nt.positions.end());
    positions.insert(positions.end(),ss.positions.begin(),ss.positions.end());
    state.deformed_positions.assign(nt.previous_positions.begin(),nt.previous_positions.end());
    state.deformed_positions.insert(state.deformed_positions.end(),ss.previous_positions.begin(),ss.previous_positions.end());
    mesh.node_to_rb = {-1,0,0,0,0,0,-1,-1};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    mesh.ref_positions = {{nt.body_references[1],nt.body_references[2],nt.body_references[3],ss.body_references[0],ss.body_references[1]}};
    const std::vector<int> local = {-1,0,1,2,3,4,-1,-1};
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(40);
    for (auto& p : cache.nt_pairs) { p.node = 0; p.tri_v[0] = 1; p.tri_v[1] = 2; p.tri_v[2] = 3; }
    cache.ss_pairs.resize(37);
    for (auto& p : cache.ss_pairs) for (int r = 0; r < 4; ++r) p.v[r] = r+4;
    std::vector<int> nt_indices(40), ss_indices(37);
    std::iota(nt_indices.begin(),nt_indices.end(),0);
    std::iota(ss_indices.begin(),ss_indices.end(),0);
    SimParams params;
    params.d_hat = d_hat; params.k_barrier = stiffness;
    params.friction_coefficient = .37; params.friction_velocity_epsilon = .1;
    int leader_calls = 0;
    const std::function<void()> leader = [&] {
        ++leader_calls;
        // Nested CUDA inertia proves callback execution precedes the contact
        // workspace lock rather than deadlocking inside compute_rigid.
        ipc_gpu::rigid_translation_inertia_gravity(Vec3::Zero(),Vec3::Zero(),Vec3::Zero(),dt,1.,-9.81);
    };
    for (int phase = 0; phase < 2; ++phase) {
        const auto mode = phase == 0 ? RigidDerivativeMode::TranslationHessian : RigidDerivativeMode::OrientationHessian;
        if (phase == 1)
            for (std::size_t node = 0; node < positions.size(); ++node)
                if (mesh.node_to_rb[node] == 0) positions[node] += Vec3(.004,-.003,.002);
        const auto expected = ipc_simd::rigid_contact_derivatives(0,mesh,state,cache,nt_indices,ss_indices,
            local,positions,omega,params,dt,mode,&kinematics);
#ifndef IPC_GPU_RIGID_HOST_ONLY
        const auto before = ipc_gpu::statistics();
#endif
        const auto actual = ipc_gpu::rigid_contact_derivatives(0,mesh,state,cache,nt_indices,ss_indices,
            local,positions,omega,params,dt,mode,&kinematics,true,&leader);
#ifndef IPC_GPU_RIGID_HOST_ONLY
        const auto after = ipc_gpu::statistics();
        require(after.rigid_contacts-before.rigid_contacts == 77,"whole-body contact count mismatch");
        require(after.launches-before.launches == 2,"expected one whole-body contact launch and one leader inertia launch");
#endif
        compare(actual,expected,"body gather phase " + std::to_string(phase));
    }
    require(leader_calls == 2,"body organizer must run leader work exactly once");
    params.friction_coefficient = 0.;
    state.deformed_positions.clear();
    const auto no_previous = ipc_gpu::rigid_contact_derivatives(0,mesh,state,cache,nt_indices,ss_indices,
        local,positions,omega,params,dt,RigidDerivativeMode::TranslationHessian,nullptr);
    require(no_previous.barrier.translation_gradient.allFinite(),"friction-free gather read absent previous positions");
    params.d_hat = 0.;
    const auto disabled = ipc_gpu::rigid_contact_derivatives(0,mesh,state,cache,nt_indices,ss_indices,
        local,positions,omega,params,dt,RigidDerivativeMode::Full,nullptr,true,&leader);
    require(leader_calls == 3 && disabled.barrier.translation_gradient.isZero(),"disabled contact leader contract");
}
} // namespace

int main() {
    try {
#ifndef IPC_GPU_RIGID_HOST_ONLY
        ipc_gpu::reset_statistics();
        std::cout << "CUDA device: " << ipc_gpu::device_name() << '\n';
#endif
        check_contacts();
        check_inertia();
        check_body_gather();
#ifdef IPC_GPU_RIGID_HOST_ONLY
        std::cout << "PASS HOST-ONLY rigid math/reference check (CUDA NOT TESTED): ";
#else
        const auto stats = ipc_gpu::statistics();
        require(stats.launches > 0 && stats.rigid_contacts > 0 && stats.rigid_inertia > 0,
            "rigid tests must launch both contact and inertia CUDA kernels");
        std::cout << "PASS CUDA rigid kernels: ";
#endif
        std::cout << components << " components; maximum tolerance fraction=" << max_tolerance_fraction << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL rigid kernel test: " << error.what() << '\n';
        return 1;
    }
}

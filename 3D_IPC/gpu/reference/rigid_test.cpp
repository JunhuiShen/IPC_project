#include "rigid.cuh"
#include "rigid_body_ipc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void gpu_rigid_test_cuda(const double*, std::size_t, ipc_gpu::RigidInertia*,
    ipc_gpu::Derivatives*, double*, double*, double*);

// Explicit local arithmetic test for a non-CUDA development machine. Never
// enabled in the GPU target; output explicitly identifies its limited scope.
#ifdef IPC_GPU_RIGID_HOST_REFERENCE
void gpu_rigid_test_cuda(const double* inputs, std::size_t count,
    ipc_gpu::RigidInertia* inertias, ipc_gpu::Derivatives* derivatives,
    double* jacobians, double* curvatures, double* positions) {
    for (std::size_t i = 0; i < count; ++i) {
        const double* p = inputs + 26*i;
        const auto k = ipc_gpu::solver::rigid_kinematics(p, p+7, p[10]);
        inertias[i] = ipc_gpu::solver::build_rigid_inertia(p,p+4,p+7,p[10],p+11);
        derivatives[i] = ipc_gpu::evaluate_rigid_inertia(inertias[i]);
        double curvature[3][9];
        ipc_gpu::solver::rigid_reference_derivatives(p+20,k,jacobians+9*i,curvature);
        for (int c = 0; c < 3; ++c) for (int j = 0; j < 9; ++j) curvatures[27*i+9*c+j] = curvature[c][j];
        ipc_gpu::solver::rigid_world_position(p+20,p+23,k.orientation,positions+3*i);
    }
}
#endif

static std::size_t comparisons = 0;
static double maximum_normalized_error = 0.0;
static void compare(double actual, double expected, const std::string& label) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) throw std::runtime_error(label + ": nonfinite result");
    const double normalized = std::abs(actual-expected)/(1.0+std::abs(expected));
    maximum_normalized_error = std::max(maximum_normalized_error, normalized);
    if (normalized > 2e-10)
        throw std::runtime_error(label + ": error=" + std::to_string(normalized));
    ++comparisons;
}

static std::vector<double> fixtures() {
    std::vector<double> result;
    const std::array<Vec4,4> orientations{
        Vec4(1,0,0,0), Vec4(.8,.2,-.4,.4).normalized(), Vec4(0,1,0,0),
        Vec4(-.3,.2,.7,-.5).normalized()};
    std::vector<Vec3> omegas{Vec3::Zero(), Vec3(1e-200,-2e-200,3e-200),
        Vec3(1e-14,-2e-14,3e-14), Vec3(1e-6,-2e-6,3e-6),
        Vec3(.2,-.7,1.4), Vec3(-7.0,3.2,11.0), Vec3(100,-70,130)};
    // Exercise both sides of the exact CPU Taylor branch at dt=1, as well as
    // all other dt choices below, without relying on randomized coverage.
    for (double factor : {1.0-1e-8, 1.0, 1.0+1e-8}) omegas.emplace_back(2e-4*factor,0,0);
    int sample = 0;
    for (const Vec4& q : orientations) for (double dt : {0.0,1.0/600.0,.031,1.0})
        for (const Vec3& omega : omegas) {
            std::array<double,26> p{};
            for (int k = 0; k < 4; ++k) p[k] = q[k];
            const Vec3 previous_omega = sample%3 == 0 ? Vec3::Zero() : Vec3(.3,-1.7,.8);
            for (int k = 0; k < 3; ++k) { p[4+k] = previous_omega[k]; p[7+k] = omega[k]; }
            p[10] = dt;
            Mat33 A;
            A << .7,.2,-.3, -.1,1.2,.4, .3,-.2,.9;
            const Mat33 moment = (1.0+.01*sample) * (A.transpose()*A);
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) p[11+3*r+c] = moment(r,c);
            const Vec3 X = sample%5 == 0 ? Vec3::Zero() : Vec3(.2,-.7,1.3);
            for (int k = 0; k < 3; ++k) { p[20+k] = X[k]; p[23+k] = .2*(k-1); }
            result.insert(result.end(),p.begin(),p.end()); ++sample;
        }
    return result;
}

static void verify(const std::vector<double>& input, std::size_t count) {
    std::vector<ipc_gpu::RigidInertia> inertias(count);
    std::vector<ipc_gpu::Derivatives> derivatives(count);
    std::vector<double> jacobians(9*count), curvatures(27*count), positions(3*count);
    gpu_rigid_test_cuda(input.data(),count,inertias.data(),derivatives.data(),
        jacobians.data(),curvatures.data(),positions.data());
    for (std::size_t i = 0; i < count; ++i) {
        const double* p = input.data()+26*i;
        const Vec4 q(p[0],p[1],p[2],p[3]);
        const Vec3 previous_omega(p[4],p[5],p[6]), omega(p[7],p[8],p[9]);
        const double dt = p[10];
        Mat33 moment;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) moment(r,c) = p[11+3*r+c];
        const Vec3 X(p[20],p[21],p[22]), com(p[23],p[24],p[25]);
        const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(q,omega,dt,true);
        const Mat33 predictor = ::rigid_rotation_predictor(q,previous_omega,dt);
        const auto expected_inertia = inertia_rotation_gradient_hessian(omega,q,previous_omega,dt,moment,&kinematics,&predictor);
        const Mat33 J = dx_domega(X,kinematics);
        const auto H = d2x_domega2(X,kinematics);
        const Vec3 position = world_space_position(X,com,kinematics.orientation);
        const auto& actual = inertias[i];
        const std::string sample = "sample " + std::to_string(i);
        for (int a = 0; a < 4; ++a) {
            compare(actual.orientation[a],kinematics.orientation[a],sample+" orientation");
            for (int b = 0; b < 3; ++b)
                compare(actual.orientation_jacobian[a][b],kinematics.orientation_jacobian(a,b),sample+" dq/domega");
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                compare(actual.orientation_hessians[a][3*r+c],kinematics.orientation_hessians[a](r,c),sample+" d2q/domega2");
        }
        for (int r = 0; r < 3; ++r) {
            compare(derivatives[i].gradient[r],expected_inertia.first[r],sample+" inertia gradient");
            compare(positions[3*i+r],position[r],sample+" proxy position");
            for (int c = 0; c < 3; ++c) {
                compare(actual.rotation_predictor[3*r+c],predictor(r,c),sample+" predictor");
                compare(derivatives[i].hessian[3*r+c],expected_inertia.second(r,c),sample+" inertia Hessian");
                compare(jacobians[9*i+3*r+c],J(r,c),sample+" dx/domega");
                for (int axis = 0; axis < 3; ++axis)
                    compare(curvatures[27*i+9*axis+3*r+c],H[axis](r,c),sample+" d2x/domega2");
            }
        }
    }
}

int main() {
    try {
        const auto input = fixtures();
        for (std::size_t count : {std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(33),std::size_t(129),input.size()/26})
            verify(input,count);
#ifdef IPC_GPU_RIGID_HOST_REFERENCE
        std::cout << "HOST ARITHMETIC ONLY (not CUDA validation): ";
#else
        std::cout << "CUDA GPU rigid preparation: ";
#endif
        std::cout << comparisons << " components match CPU; max normalized error "
                  << std::setprecision(8) << maximum_normalized_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU rigid preparation test failed: " << error.what() << '\n';
        return 1;
    }
}

#include "rigid.cuh"
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

// Test-only raw-input wrapper. Production kernels call the device functions
// directly on GPU state; these transfers exist solely for CPU comparison.
// Input stride 26: q_previous[4], omega_previous[3], omega[3], dt,
// second_moment[9], body_reference[3], com[3].
struct GpuRigidTestOutput {
    ipc_gpu::RigidInertia inertia;
    ipc_gpu::Derivatives derivatives;
    double jacobian[9], curvature[3][9], position[3];
};

static void cuda_check(cudaError_t error, const char* label) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}
template<class T> struct TestBuffer {
    T* pointer = nullptr;
    explicit TestBuffer(std::size_t count) {
        if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "test allocation");
    }
    ~TestBuffer() { if (pointer) cudaFree(pointer); }
};

__global__ void gpu_rigid_test_kernel(const double* input, std::size_t count,
                                          GpuRigidTestOutput* output) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const double* p = input + 26 * i;
        auto& out = output[i];
        const auto kinematics = ipc_gpu::solver::rigid_kinematics(p, p + 7, p[10]);
        out.inertia = ipc_gpu::solver::build_rigid_inertia(p, p + 4, p + 7, p[10], p + 11);
        out.derivatives = ipc_gpu::evaluate_rigid_inertia(out.inertia);
        ipc_gpu::solver::rigid_reference_derivatives(p + 20, kinematics, out.jacobian, out.curvature);
        ipc_gpu::solver::rigid_world_position(p + 20, p + 23, kinematics.orientation, out.position);
    }
}

void gpu_rigid_test_cuda(const double* inputs, std::size_t count,
    ipc_gpu::RigidInertia* inertias, ipc_gpu::Derivatives* derivatives,
    double* jacobians, double* curvatures, double* positions) {
    if (!count) return;
    TestBuffer<double> device_input(26 * count);
    TestBuffer<GpuRigidTestOutput> device_output(count);
    cuda_check(cudaMemcpy(device_input.pointer, inputs, 26 * count * sizeof(double), cudaMemcpyHostToDevice), "test input upload");
    gpu_rigid_test_kernel<<<static_cast<unsigned>((count + 127) / 128), 128>>>(device_input.pointer, count, device_output.pointer);
    cuda_check(cudaGetLastError(), "GPU rigid test kernel");
    std::vector<GpuRigidTestOutput> output(count);
    cuda_check(cudaMemcpy(output.data(), device_output.pointer, count * sizeof(GpuRigidTestOutput), cudaMemcpyDeviceToHost), "test output download");
    for (std::size_t i = 0; i < count; ++i) {
        inertias[i] = output[i].inertia; derivatives[i] = output[i].derivatives;
        for (int k = 0; k < 9; ++k) jacobians[9*i+k] = output[i].jacobian[k];
        for (int c = 0; c < 3; ++c) {
            positions[3*i+c] = output[i].position[c];
            for (int k = 0; k < 9; ++k) curvatures[27*i+9*c+k] = output[i].curvature[c][k];
        }
    }
}

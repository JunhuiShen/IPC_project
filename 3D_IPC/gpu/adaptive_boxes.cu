#include "adaptive_boxes.h"
#include "broadphase.h"
#include "elastic.h"
#include "device_memory.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ipc_gpu::solver {
namespace {
using Position = std::array<double, 3>;
static_assert(sizeof(Position) == 3 * sizeof(double));
constexpr unsigned block_width = 128;

unsigned blocks(std::size_t nodes) {
    if (nodes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("GPU adaptive boxes: too many nodes");
    return static_cast<unsigned>((nodes + block_width - 1) / block_width);
}

__device__ double norm3(const double x, const double y, const double z) {
    return sqrt(x * x + (y * y + z * z));
}

__global__ void begin_adaptive_boxes(
    DeformableScene scene, DeformableState state, double dt,
    double minimum, double maximum, bool initialize,
    double* previous_displacement, double* radii,
    double (*anchor)[3], int* error) {
    const auto n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= scene.nodes || !scene.deformable[n] || atomicAdd(error, 0)) return;
    for (int a = 0; a < 3; ++a) {
        if (!isfinite(state.positions[n][a]) || !isfinite(state.velocities[n][a])) {
            atomicCAS(error, 0, 3);
            return;
        }
        anchor[n][a] = state.positions[n][a];
    }
    if (initialize) previous_displacement[n] = maximum;
    const double inertial = norm3(state.velocities[n][0], state.velocities[n][1],
                                  state.velocities[n][2]) * dt;
    const double previous = previous_displacement[n];
    if (!isfinite(previous) || previous < 0 || isnan(inertial)) {
        atomicCAS(error, 0, 3);
        return;
    }
    const double padded = 1.2 * (previous < inertial ? inertial : previous);
    radii[n] = padded < minimum ? minimum : (maximum < padded ? maximum : padded);
}

__global__ void build_adaptive_boxes(
    DeformableScene scene, DeformableState state, const double* radii,
    BpAabb* boxes, int* error) {
    const auto n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= scene.nodes || !scene.deformable[n] || atomicAdd(error, 0)) return;
    BpAabb box{};
    for (int a = 0; a < 3; ++a) {
        box.lo[a] = state.positions[n][a] - radii[n];
        box.hi[a] = state.positions[n][a] + radii[n];
        if (!isfinite(box.lo[a]) || !isfinite(box.hi[a])) {
            atomicCAS(error, 0, 3);
            return;
        }
    }
    boxes[n] = box;
}

__global__ void finish_adaptive_boxes(
    DeformableScene scene, DeformableState state, const double (*anchor)[3],
    double* previous_displacement, int* error) {
    const auto n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= scene.nodes || !scene.deformable[n] || atomicAdd(error, 0)) return;
    const double displacement = norm3(state.positions[n][0] - anchor[n][0],
                                      state.positions[n][1] - anchor[n][1],
                                      state.positions[n][2] - anchor[n][2]);
    if (!isfinite(displacement)) {
        atomicCAS(error, 0, 3);
        return;
    }
    previous_displacement[n] = displacement;
}
} // namespace

struct GpuAdaptiveBoxes::Impl {
    DeviceArray<double> previous_displacement, radii;
    DeviceArray<Position> anchor;
    const unsigned char* deformable = nullptr;
    bool initialized = false;
    bool active = false;

    double (*anchors())[3] { return reinterpret_cast<double (*)[3]>(anchor.data()); }

    void validate_scene(const DeformableScene& scene, const DeformableState& state,
                        int* error) const {
        if (!error || (scene.nodes && (!scene.deformable || !state.positions)))
            throw std::invalid_argument("GPU adaptive boxes: missing device arrays");
    }
    void validate_active(const DeformableScene& scene, const DeformableState& state,
                         int* error) const {
        validate_scene(scene, state, error);
        if (!active || anchor.size() != scene.nodes || deformable != scene.deformable)
            throw std::logic_error("GPU adaptive boxes: call begin_solver for this scene first");
    }
};

GpuAdaptiveBoxes::GpuAdaptiveBoxes() : impl_(std::make_unique<Impl>()) {}
GpuAdaptiveBoxes::~GpuAdaptiveBoxes() = default;

void GpuAdaptiveBoxes::begin_solver(
    const DeformableScene& scene, const DeformableState& state,
    double dt, double minimum, double maximum, int* error) {
    auto& s = *impl_;
    s.validate_scene(scene, state, error);
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(minimum) || minimum < 0
        || !std::isfinite(maximum) || maximum <= 0 || maximum < minimum
        || (scene.nodes && !state.velocities))
        throw std::invalid_argument("GPU adaptive boxes: invalid radius, timestep, or velocities");
    if (s.active)
        throw std::logic_error("GPU adaptive boxes: finish_solver before the next begin_solver");
    const auto launch_blocks = blocks(scene.nodes);
    if (s.anchor.size() != scene.nodes || s.deformable != scene.deformable) {
        s.previous_displacement.resize(scene.nodes);
        s.radii.resize(scene.nodes);
        s.anchor.resize(scene.nodes);
        s.deformable = scene.deformable;
        s.initialized = false;
    }
    if (scene.nodes) {
        begin_adaptive_boxes<<<launch_blocks, block_width>>>(scene, state, dt, minimum, maximum,
            !s.initialized, s.previous_displacement.data(), s.radii.data(), s.anchors(), error);
        cuda_checked(cudaGetLastError());
    }
    s.initialized = true;
    s.active = true;
}

void GpuAdaptiveBoxes::build(
    const DeformableScene& scene, const DeformableState& state,
    BpAabb* boxes, int* error) {
    auto& s = *impl_;
    s.validate_active(scene, state, error);
    if (scene.nodes && !boxes)
        throw std::invalid_argument("GPU adaptive boxes: missing output boxes");
    if (scene.nodes) {
        build_adaptive_boxes<<<blocks(scene.nodes), block_width>>>(scene, state, s.radii.data(), boxes, error);
        cuda_checked(cudaGetLastError());
    }
}

void GpuAdaptiveBoxes::finish_solver(
    const DeformableScene& scene, const DeformableState& state, int* error) {
    auto& s = *impl_;
    s.validate_active(scene, state, error);
    if (scene.nodes) {
        finish_adaptive_boxes<<<blocks(scene.nodes), block_width>>>(scene, state, s.anchors(),
            s.previous_displacement.data(), error);
        cuda_checked(cudaGetLastError());
    }
    s.active = false;
}

void GpuAdaptiveBoxes::reset() {
    impl_->initialized = false;
    impl_->active = false;
}

} // namespace ipc_gpu::solver

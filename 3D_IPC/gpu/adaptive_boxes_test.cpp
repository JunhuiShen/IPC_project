#include "adaptive_boxes.h"
#include "broadphase.h"
#include "elastic.h"
#include "test_support.h"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace ipc_gpu::solver;
using V = std::array<double, 3>;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> static void upload(GpuTestBuffer<T>& buffer, const std::vector<T>& values) {
    if (!values.empty()) gpu_cuda_check(cudaMemcpy(buffer.data(), values.data(),
        values.size() * sizeof(T), cudaMemcpyHostToDevice));
}
static double (*triples(V* values))[3] { return reinterpret_cast<double (*)[3]>(values); }
static double cpu_norm(const V& value) {
    return Eigen::Vector3d(value[0], value[1], value[2]).norm();
}
static V subtract(const V& a, const V& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
static void check_boxes(const std::vector<BpAabb>& boxes, const std::vector<V>& positions,
                        const std::vector<double>& previous, const std::vector<V>& velocities,
                        const std::vector<unsigned char>& deformable,
                        const BpAabb& rigid_sentinel, double dt, double minimum, double maximum) {
    for (std::size_t n = 0; n < positions.size(); ++n) {
        if (!deformable[n]) {
            require(std::memcmp(&boxes[n], &rigid_sentinel, sizeof(BpAabb)) == 0,
                    "adaptive boxes changed a rigid proxy");
            continue;
        }
        // This is exactly the basic-v2 CPU policy, independently evaluated
        // with its Eigen vector norm; no production GPU arithmetic is reused.
        const double radius = std::clamp(std::max(previous[n], cpu_norm(velocities[n]) * dt)
            * 1.2, minimum, maximum);
        for (int a = 0; a < 3; ++a) {
            const double tolerance = 8 * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(positions[n][a]), radius});
            require(std::abs(boxes[n].lo[a] - (positions[n][a] - radius)) <= tolerance,
                    "adaptive lower bound differs from CPU policy");
            require(std::abs(boxes[n].hi[a] - (positions[n][a] + radius)) <= tolerance,
                    "adaptive upper bound differs from CPU policy");
        }
    }
}

int main() try {
    // Positions already include a large initial guess. Physical previous and
    // predicted positions deliberately differ from this solver-entry anchor.
    std::vector<V> x = {{0, 0, 0}, {1, 2, 3}, {-.4, .2, .1}, {.2, -.3, .4},
                        {2, 3, 4}, {.1, .1, .1}, {9, 8, 7}, {-8, -7, -6}};
    const auto entry = x;
    std::vector<V> velocity(x.size(), V{0, 0, 0});
    const std::vector<unsigned char> deformable = {1, 1, 1, 1, 1, 1, 1, 0};
    const std::vector<unsigned char> solid = {0, 0, 1, 0, 1, 1, 0, 0};
    const std::vector<unsigned char> pinned = {0, 0, 0, 1, 0, 0, 1, 1};
    const BpAabb sentinel{{-101, -102, -103}, {101, 102, 103}};
    GpuTestBuffer<V> positions(x), velocities(velocity);
    GpuTestBuffer<V> previous_positions(std::vector<V>(x.size(), V{-100, -100, -100}));
    GpuTestBuffer<V> predicted(std::vector<V>(x.size(), V{100, 100, 100}));
    GpuTestBuffer<unsigned char> is_deformable(deformable), is_solid(solid), is_pinned(pinned);
    GpuTestBuffer<BpAabb> boxes(std::vector<BpAabb>(x.size(), sentinel));
    GpuTestBuffer<int> error(std::vector<int>{0});
    DeformableScene scene{};
    scene.nodes = x.size(); scene.deformable = is_deformable.data();
    scene.solid = is_solid.data(); scene.pinned = is_pinned.data();
    DeformableState state{triples(positions.data()), triples(velocities.data()),
        triples(previous_positions.data()), triples(predicted.data())};
    GpuAdaptiveBoxes history;
    constexpr double dt = .01, minimum = .001, maximum = .02;
    std::vector<double> previous(x.size(), maximum);
    history.begin_solver(scene, state, dt, minimum, maximum, error.data());
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, maximum);

    // Rebuild midway through the solve must recenter but must NOT reinterpret
    // current displacement or changed velocities as new history.
    const std::vector<V> updates = {{0, 0, 0}, {.003, .004, 0}, {0, .00005, 0},
        {.002, .001, -.001}, {.001, 0, 0}, {.01, .02, 0}, {0, 0, 0}, {0, 0, 0}};
    for (std::size_t n = 0; n < x.size(); ++n)
        for (int a = 0; a < 3; ++a) x[n][a] += updates[n][a];
    upload(positions, x);
    upload(velocities, std::vector<V>(x.size(), V{100, 200, 300}));
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, maximum);
    history.finish_solver(scene, state, error.data());
    for (std::size_t n = 0; n < x.size(); ++n) previous[n] = cpu_norm(subtract(x[n], entry[n]));

    // Second substep includes stationary, low-motion, previous-displacement
    // dominated, inertial dominated, high-speed clamped, and moving soft pins.
    velocity[2] = {.03, .04, 0};
    velocity[4] = {.6, .8, 0};
    velocity[5] = {100, 200, 300};
    // Node 6 had a huge difference from physical previous_positions, but zero
    // solver movement; its radius must go to the minimum, not the maximum.
    upload(velocities, velocity);
    history.begin_solver(scene, state, dt, minimum, maximum, error.data());
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, maximum);
    upload(velocities, std::vector<V>(x.size(), V{100, 200, 300}));
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, maximum);
    bool rejected_nested_begin = false;
    try { history.begin_solver(scene, state, dt, minimum, maximum, error.data()); }
    catch (const std::logic_error&) { rejected_nested_begin = true; }
    require(rejected_nested_begin, "adaptive boxes accepted an unfinished nested solver call");
    history.finish_solver(scene, state, error.data()); // zero iterations, no displacement

    // A no-motion solver call clears the old history. New dt/bounds apply
    // without reinitializing previous displacement to the changed maximum.
    std::fill(previous.begin(), previous.end(), 0);
    std::fill(velocity.begin(), velocity.end(), V{0, 0, 0});
    velocity[3] = {.03, .04, .12}; // a soft pin still gets an inertial radius
    upload(velocities, velocity);
    history.begin_solver(scene, state, .1, .0002, .04, error.data());
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, .1, .0002, .04);
    history.finish_solver(scene, state, error.data());

    // Topology reset restarts at the (possibly changed) maximum radius.
    history.reset();
    std::fill(previous.begin(), previous.end(), .03);
    history.begin_solver(scene, state, dt, minimum, .03, error.data());
    history.build(scene, state, boxes.data(), error.data());
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, .03);
    history.finish_solver(scene, state, error.data());
    require(error.download()[0] == 0, "adaptive boxes produced a device error");

    // Existing failure suppresses all numerical writes (including history).
    upload(error, std::vector<int>{77});
    const auto saved_boxes = boxes.download();
    history.begin_solver(scene, state, dt, minimum, maximum, error.data());
    history.build(scene, state, boxes.data(), error.data());
    history.finish_solver(scene, state, error.data());
    const auto failed_boxes = boxes.download();
    require(std::memcmp(saved_boxes.data(), failed_boxes.data(), saved_boxes.size() * sizeof(BpAabb)) == 0,
            "adaptive boxes wrote after an existing error");
    require(error.download()[0] == 77, "adaptive boxes replaced an existing error");
    upload(error, std::vector<int>{0});
    history.reset();

    bool rejected_bounds = false;
    try { history.begin_solver(scene, state, dt, maximum, minimum, error.data()); }
    catch (const std::invalid_argument&) { rejected_bounds = true; }
    require(rejected_bounds, "adaptive boxes accepted reversed bounds");

    // A zero-node topology has no numerical work and no invalid zero launch.
    DeformableScene empty{};
    DeformableState empty_state{};
    history.begin_solver(empty, empty_state, dt, 0, maximum, error.data());
    history.build(empty, empty_state, nullptr, error.data());
    history.finish_solver(empty, empty_state, error.data());
    require(error.download()[0] == 0, "empty adaptive boxes produced a device error");

    // Returning to a nonempty topology reinitializes history to maximum.
    history.begin_solver(scene, state, dt, minimum, maximum, error.data());
    history.build(scene, state, boxes.data(), error.data());
    std::fill(previous.begin(), previous.end(), maximum);
    check_boxes(boxes.download(), x, previous, velocity, deformable, sentinel, dt, minimum, maximum);
    history.finish_solver(scene, state, error.data());
    require(error.download()[0] == 0, "adaptive boxes failed after topology resize");
    std::cout << "PASS adaptive deformable boxes vs Eigen CPU policy: initial maximum, "
        "substep history, initial-guess exclusion, fixed within-solve radii, recentering, "
        "inertial/previous/min/max branches, soft pins, solids, rigid exclusion, "
        "zero iterations, reset/resize, empty topology, invalid lifecycle and error guard\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}

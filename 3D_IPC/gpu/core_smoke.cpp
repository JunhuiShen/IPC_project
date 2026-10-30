#include "solver.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ipc_gpu::solver;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

SceneDescription points(std::size_t count) {
    SceneDescription scene;
    scene.positions.resize(count); scene.velocities.resize(count);
    scene.pin_targets.resize(count); scene.body_reference.resize(count);
    scene.deformable.assign(count, 1); scene.solid.assign(count, 0);
    scene.pinned.assign(count, 0); scene.surface.assign(count, 1);
    scene.mass.assign(count, 1); scene.node_body.assign(count, -1);
    scene.node_blocks.resize(count);
    for (std::size_t i = 0; i < count; ++i) scene.node_blocks[i] = static_cast<int>(i);
    scene.block_count = static_cast<int>(count);
    scene.triangle_offsets.assign(count + 1, 0);
    scene.solid_offsets.assign(count + 1, 0);
    scene.hinge_offsets.assign(count + 1, 0);
    scene.body_node_offsets = {0};
    return scene;
}
}

int main() try {
    using namespace ipc_gpu::solver;
    StepParameters step;
    step.dt = .01; step.elastic.dt2 = step.dt * step.dt;
    step.contact.dt = step.dt; step.sdf.dt = step.dt;
    step.contact.eps = 1e-12; step.contact.eps_v = .001;
    step.iterations = 2; step.rebuild_interval = 1;
    step.elastic.kpin = 100;

    GpuSolver empty(points(0));
    empty.advance_substep(step, {}, {});
    require(empty.snapshot().positions.empty(), "empty core scene changed size");

    auto scene = points(1);
    scene.pinned[0] = 1;
    scene.pin_targets[0] = {.001, 0, 0};
    GpuSolver solver(scene);
    // A rejected host boundary input must not invalidate untouched device state.
    bool rejected = false;
    try { solver.advance_substep(step, {}, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "core accepted wrong dense pin extent");
    solver.advance_substep(step, scene.pin_targets, {});
    const auto state = solver.snapshot();
    require(state.positions.size() == 1 && state.velocities.size() == 1,
            "core snapshot extent changed");
    require(state.positions[0][0] > 0 && state.positions[0][0] <= scene.pin_targets[0][0],
            "core soft-pin update failed");
    for (const auto& value : {state.positions[0], state.velocities[0]})
        for (double component : value)
            require(std::isfinite(component), "core produced nonfinite state");
    require(state.positions[0][1] == 0 && state.positions[0][2] == 0,
            "core changed an unforced coordinate");
    std::cout << "PASS GPU CUDA core: empty scene, dense input rejection, soft pin, finite snapshot\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

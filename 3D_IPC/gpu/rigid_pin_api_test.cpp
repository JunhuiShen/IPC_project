// Ordinary GPU bridge and rigid material-point pin regression.
// The optional CPU/CUDA Jacobi-DAT feature and its parity cases were retired.
// These retained cases compare the ordinary device rigid solver with the
// CPU scalar rigid solver, including force, stiffness and time controls.
#include "bridge.h"
#include "mesh_utils.h"
#include "../solver.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_rigid_pin_api {
using namespace ipc_gpu::solver;
[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("GPU CPU/GPU parity: " + what);
}

void require(bool value, const std::string& what) {
    if (!value) fail(what);
}

std::string tag(const std::string& label, std::size_t node, int axis) {
    return label + " node " + std::to_string(node) + " axis " + std::to_string(axis);
}

struct Fixture {
    RefMesh mesh;
    DeformedState state;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    SimParams parameters = SimParams::zeros();
    Fixture(std::vector<Vec3> positions = {}, std::vector<int> triangles = {}) {
        mesh.tris = std::move(triangles);
        mesh.initialize(positions);
        mesh.mass.assign(positions.size(), 1.0);
        mesh.node_to_rb.assign(positions.size(), -1);
        mesh.build_deformable_nodes();
        adjacency = build_incident_triangle_map(mesh.tris);
        state.deformed_positions = std::move(positions);
        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
        parameters.fps = 40.0;
        parameters.substeps = 1;
        parameters.max_global_iters = 1;
        parameters.fixed_iters = true;
        parameters.use_dat = false;
        parameters.use_ccd_guess = false;
        parameters.node_box_min = .001;
        parameters.node_box_max = .1;
        parameters.theta_box_min = parameters.theta_box_max = .1;
        parameters.node_box_update_count = 2;
        parameters.damping = .5;
    }
};

void refresh_topology(Fixture& fixture) {
    fixture.mesh.num_positions = fixture.state.deformed_positions.size();
    fixture.mesh.node_to_rb.resize(fixture.state.deformed_positions.size(), -1);
    fixture.mesh.build_deformable_nodes();
    fixture.adjacency = build_incident_triangle_map(fixture.mesh.tris);
}

struct GpuRun {
    GpuSolver solver;
    StepParameters step;
    std::vector<Vector3> pin_targets;
    std::vector<SdfCollider> colliders;
    explicit GpuRun(const Fixture& fixture)
        : solver(describe_scene(fixture.mesh, fixture.state, fixture.pins, fixture.parameters)),
          step(describe_step(fixture.parameters)),
          pin_targets(describe_pin_targets(fixture.state.deformed_positions.size(), fixture.pins)),
          colliders(describe_colliders(fixture.parameters)) {}
    void advance() { solver.advance_substep(step, pin_targets, colliders); }
    void advance(const StepParameters& parameters) {
        solver.advance_substep(parameters, pin_targets, colliders);
    }
    StateSnapshot snapshot() const { return solver.snapshot(); }
};

RigidPin rigid_pin(int body, const Vec3& local, const Vec3& target) {
    RigidPin pin;
    pin.body = body;
    pin.local_position = local;
    pin.initial_target = target;
    pin.rotation_center = local;
    pin.angular_velocity_z = 0.0;
    return pin;
}

Fixture rigid_pair() {
    Fixture fixture;
    fixture.parameters.use_dat = false;
    fixture.parameters.max_global_iters = 1;
    fixture.parameters.damping = 1.0;
    fixture.parameters.gravity = Vec3(0, -.5, 0);
    fixture.parameters.kpin = 4.0;
    fixture.parameters.rigid_pin_time = .25;
    fixture.parameters.d_hat = 0.;
    const int body0 = create_rigid_body({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)},
        Vec3::Zero(), Vec4(1, 0, 0, 0), Vec3(.05, 0, 0), 1.0, fixture.mesh,
        fixture.state, RigidBodyUpdateMode::TranslationOnly);
    const int body1 = create_rigid_body({Vec3(3, 0, 0), Vec3(4, 0, 0), Vec3(3, 1, 0)},
        Vec3(3, 0, 0), Vec4(1, 0, 0, 0), Vec3(0, .05, 0), 1.0, fixture.mesh,
        fixture.state, RigidBodyUpdateMode::TranslationOnly);
    require(body0 == 0 && body1 == 1, "unexpected rigid body indices");
    fixture.mesh.tris.insert(fixture.mesh.tris.end(),
        fixture.mesh.rb_nodes[body0].begin(), fixture.mesh.rb_nodes[body0].end());
    fixture.mesh.tris.insert(fixture.mesh.tris.end(),
        fixture.mesh.rb_nodes[body1].begin(), fixture.mesh.rb_nodes[body1].end());
    refresh_topology(fixture);
    // Two pins per body in interleaved parameter order (body 1, body 0, body 1,
    // body 0) with distinct material points and targets, so stable per-body
    // grouping and within-body parameter order are both observable.
    fixture.parameters.rigid_pins.push_back(rigid_pin(1, Vec3(.25, .1, 0), Vec3(3.25, -.10, 0)));
    fixture.parameters.rigid_pins.push_back(rigid_pin(0, Vec3(.5, 0, 0), Vec3(.5, .20, 0)));
    fixture.parameters.rigid_pins.push_back(rigid_pin(1, Vec3(.75, 0, 0), Vec3(3.75, -.05, 0)));
    fixture.parameters.rigid_pins.push_back(rigid_pin(0, Vec3(.1, .2, 0), Vec3(.1, .25, 0)));
    return fixture;
}

struct CpuRbState {
    std::vector<Vec3> com, omega;
    std::vector<Vec4> orientation;
};

CpuRbState cpu_rigid_reference(const Fixture& fixture) {
    CpuRbState reference;
    reference.com = fixture.state.x_coms;
    reference.orientation = fixture.state.orientations;
    reference.omega.assign(fixture.state.omega.size(), Vec3::Zero());
    const auto result = global_gauss_seidel_solver_rb(fixture.mesh, fixture.state,
        fixture.parameters, reference.com, reference.orientation, reference.omega);
    require(result.iterations == fixture.parameters.max_global_iters,
        "CPU rigid reference used a different sweep count");
    return reference;
}

void compare_rigid(const StateSnapshot& gpu, const CpuRbState& cpu, const std::string& label) {
    require(gpu.com.size() == cpu.com.size(), label + ": rigid body count mismatch");
    require(gpu.com_velocity.size() == cpu.com.size(), label + ": rigid COM velocity count mismatch");
    require(gpu.omega.size() == cpu.omega.size(), label + ": rigid omega count mismatch");
    require(gpu.orientation.size() == cpu.orientation.size(), label + ": rigid orientation count mismatch");
    for (std::size_t body = 0; body < cpu.com.size(); ++body) {
        const std::string prefix = label + " body " + std::to_string(body) + " axis ";
        for (int axis = 0; axis < 3; ++axis) {
            // abs(NaN - x) > tol is false, so both sides must be proven finite
            // before any tolerance comparison can be trusted.
            const double actual = gpu.com[body][axis];
            const double expected = cpu.com[body][axis];
            require(std::isfinite(actual) && std::isfinite(expected),
                prefix + std::to_string(axis) + ": nonfinite rigid COM");
            require(std::isfinite(gpu.com_velocity[body][axis]),
                prefix + std::to_string(axis) + ": nonfinite rigid COM velocity");
            if (std::abs(actual - expected) > 5e-9)
                fail(prefix + std::to_string(axis) + ": COM differs from CPU reference");
            const double omega = gpu.omega[body][axis];
            const double expected_omega = cpu.omega[body][axis];
            require(std::isfinite(omega) && std::isfinite(expected_omega),
                prefix + std::to_string(axis) + ": nonfinite rigid omega");
            if (std::abs(omega - expected_omega) > 5e-7)
                fail(prefix + std::to_string(axis) + ": omega differs from CPU reference");
        }
        // Translation-only bodies never enter an orientation stage, so the
        // quaternion must stay finite, unit length and equal to the CPU's.
        double norm2 = 0;
        for (int component = 0; component < 4; ++component) {
            const double actual = gpu.orientation[body][component];
            const double expected = cpu.orientation[body][component];
            const std::string component_prefix = label + " body " + std::to_string(body)
                + " orientation component " + std::to_string(component);
            require(std::isfinite(actual) && std::isfinite(expected),
                component_prefix + ": nonfinite rigid orientation");
            require(std::abs(actual) <= 1.0 + 1e-12,
                component_prefix + ": component outside the unit orientation extent");
            if (std::abs(actual - expected) > 1e-12)
                fail(component_prefix + ": orientation differs from CPU reference");
            norm2 += actual * actual;
        }
        require(std::abs(norm2 - 1.0) <= 1e-9,
            label + " body " + std::to_string(body) + ": orientation is not unit length");
    }
}

void check_rigid_pins() {
    const Fixture base = rigid_pair();
    // Stable, per-body grouping keeps the CPU parameter-list order within each
    // body and groups all bodies even when pins are interleaved.
    auto scene = describe_scene(base.mesh, base.state, base.pins, base.parameters);
    const auto& source_pins = base.parameters.rigid_pins;
    require(source_pins.size() == 4, "rigid pin fixture size");
    require(source_pins[0].body != source_pins[1].body,
        "rigid pin fixture is not interleaved by body");
    require(scene.rigid_pin_offsets.size() == 3, "rigid pin offsets size");
    // Expected grouping: every body's pins in their original parameter order.
    std::vector<std::vector<std::size_t>> expected(2);
    for (std::size_t i = 0; i < source_pins.size(); ++i) expected[source_pins[i].body].push_back(i);
    require(expected[0].size() == 2 && expected[1].size() == 2,
        "rigid pin fixture does not give every body two pins");
    require(scene.rigid_pin_offsets[0] == 0
        && scene.rigid_pin_offsets[1] == expected[0].size()
        && scene.rigid_pin_offsets[2] == source_pins.size(),
        "rigid pin stable body grouping changed");
    require(scene.rigid_pins.size() == source_pins.size(), "rigid pin count changed during grouping");
    for (std::size_t body = 0; body < expected.size(); ++body) {
        require(scene.rigid_pin_offsets[body + 1] - scene.rigid_pin_offsets[body]
            == expected[body].size(), "rigid pin body group size changed");
        for (std::size_t k = 0; k < expected[body].size(); ++k) {
            const auto& source = source_pins[expected[body][k]];
            const auto& value = scene.rigid_pins[scene.rigid_pin_offsets[body] + k];
            require(value.body == static_cast<int>(body) && value.body == source.body,
                "rigid pin body grouping/order changed");
            require(value.angular_velocity_z == source.angular_velocity_z,
                "rigid pin motion changed during grouping");
            for (int axis = 0; axis < 3; ++axis)
                require(value.initial_target[axis] == source.initial_target[axis]
                    && value.local_position[axis] == source.local_position[axis]
                    && value.rotation_center[axis] == source.rotation_center[axis],
                    "rigid pin material point/target changed during grouping");
        }
    }
    // The interleaved input really is reordered, and same-body pins keep
    // distinct material points and targets.
    require(scene.rigid_pins[0].local_position[0]
        == source_pins[expected[0][0]].local_position[0]
        && scene.rigid_pins[scene.rigid_pin_offsets[1]].local_position[0]
        == source_pins[expected[1][0]].local_position[0],
        "rigid pin parameter order within a body was not preserved");
    const auto& first_pin = source_pins[expected[0][0]];
    const auto& second_pin = source_pins[expected[0][1]];
    require(first_pin.local_position[0] != second_pin.local_position[0]
        || first_pin.local_position[1] != second_pin.local_position[1]
        || first_pin.local_position[2] != second_pin.local_position[2],
        "rigid pin fixture reuses a material point");
    require(first_pin.initial_target[0] != second_pin.initial_target[0]
        || first_pin.initial_target[1] != second_pin.initial_target[1]
        || first_pin.initial_target[2] != second_pin.initial_target[2],
        "rigid pin fixture reuses a target");
    // Time is a per-step value, not baked into the static constraint.
    SimParams advanced = base.parameters;
    advanced.rigid_pin_time = 1.75;
    require(describe_step(advanced).rigid_pin_time == 1.75, "rigid pin time was not forwarded");
    bool rejected = false;
    SimParams bad = base.parameters;
    bad.rigid_pins.push_back(rigid_pin(7, Vec3::Zero(), Vec3::Zero()));
    try { (void)describe_scene(base.mesh, base.state, base.pins, bad); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range rigid pin body was not rejected");

    // One no-contact translation-only substep must equal the CPU rigid solver.
    const CpuRbState reference = cpu_rigid_reference(base);
    GpuRun gpu(base);
    gpu.advance();
    const auto actual = gpu.snapshot();
    compare_rigid(actual, reference, "rigid pin");
    double displacement = 0;
    for (std::size_t body = 0; body < base.state.x_coms.size(); ++body)
        displacement += (Vec3(actual.com[body][0], actual.com[body][1], actual.com[body][2])
            - base.state.x_coms[body]).norm();
    require(displacement > 1e-6, "rigid pin force did not affect the state");
    // Both bodies are translation-only: no rotation stage may touch the pose.
    require(actual.orientation.size() == base.state.orientations.size(),
        "rigid pin orientation count changed");
    for (std::size_t body = 0; body < base.state.orientations.size(); ++body)
        for (int component = 0; component < 4; ++component)
            require(std::abs(actual.orientation[body][component]
                - base.state.orientations[body][component]) <= 1e-12,
                "translation-only rigid body orientation changed");

    // Without pins the same substep must differ: the pin force is real.
    Fixture no_pin = base;
    no_pin.parameters.rigid_pins.clear();
    GpuRun free_run(no_pin);
    free_run.advance();
    const auto free_state = free_run.snapshot();
    double separation = 0;
    for (std::size_t body = 0; body < base.state.x_coms.size(); ++body)
        for (int axis = 0; axis < 3; ++axis)
            separation += std::abs(actual.com[body][axis] - free_state.com[body][axis]);
    require(separation > 1e-6, "rigid pin had no observable effect on the COM");

    // A zero stiffness keeps the pins configured but contributes no force.
    Fixture zero_stiffness = base;
    zero_stiffness.parameters.kpin = 0.0;
    GpuRun zero_run(zero_stiffness);
    zero_run.advance();
    const auto zero_state = zero_run.snapshot();
    for (std::size_t body = 0; body < base.state.x_coms.size(); ++body)
        for (int axis = 0; axis < 3; ++axis)
            require(std::abs(zero_state.com[body][axis] - free_state.com[body][axis]) < 1e-9,
                "zero-stiffness pin changed the no-pin result");

    // The per-step time actually reaches the device: two identical runs at
    // different angular pin times must diverge.
    Fixture timed = base;
    timed.parameters.rigid_pins[0].angular_velocity_z = 1.0;
    timed.parameters.rigid_pins[0].rotation_center = Vec3::Zero();
    const std::size_t timed_body = static_cast<std::size_t>(timed.parameters.rigid_pins[0].body);
    timed.parameters.rigid_pin_time = 0.;
    GpuRun time0(timed);
    time0.advance();
    timed.parameters.rigid_pin_time = 1.5;
    GpuRun time1(timed);
    time1.advance();
    const auto time0_state = time0.snapshot();
    const auto time1_state = time1.snapshot();
    double time_separation = 0;
    for (int axis = 0; axis < 3; ++axis)
        time_separation += std::abs(time0_state.com[timed_body][axis]
            - time1_state.com[timed_body][axis]);
    require(time_separation > 1e-8, "rigid pin time was ignored by the device");
    std::cout << "Rigid pins: interleaved stable grouping, time mapping, CPU parity and force effect PASS\n";
}

void check_bridge_mapping_and_retired_mode_rejection() {
    SimParams params=SimParams::zeros();
    params.fixed_iters=true;
    params.rigid_pin_time=.375;params.kpin=2.5;params.max_global_iters=4;
    params.damping=.25;params.node_box_min=.002;params.node_box_max=.1;
    params.node_box_update_count=3;params.theta_box_max=.07;
    const auto step=describe_step(params);
    require(step.iterations==params.max_global_iters,"iteration mapping changed");
    require(step.damping==params.damping,"damping mapping changed");
    require(step.rigid_pin_time==params.rigid_pin_time,"rigid pin time mapping changed");
    require(step.elastic.kpin==params.kpin,"soft pin stiffness mapping changed");
    require(step.box_radius_min==params.node_box_min&&step.box_radius==params.node_box_max,
        "adaptive box radius mapping changed");
    require(step.rebuild_interval==params.node_box_update_count&&step.rotation_radius==params.theta_box_max,
        "rebuild/rotation mapping changed");
    for(int variant=0;variant<6;++variant) {
        SimParams bad=params;
        if(variant==0)bad.fixed_iters=false;
        if(variant==1)bad.use_ogc=true;
        if(variant==2)bad.use_cloth_grid=true;
        if(variant==3){bad.use_colored_ccd_guess=true;bad.use_colored_ccd_guess_level_based=true;}
        if(variant==4)bad.use_dat=true;
        if(variant==5)bad.use_jacobi_level_based_ccd=true;
        bool rejected=false;
        try{(void)describe_step(bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"unsupported solver/guess mode did not fail closed");
    }
    auto base=rigid_pair();base.parameters.use_dat=true;
    bool scene_rejected=false;
    try{(void)describe_scene(base.mesh,base.state,base.pins,base.parameters);}
    catch(const std::invalid_argument&){scene_rejected=true;}
    require(scene_rejected,"GPU scene setup accepted the retired DAT route");
    std::cout<<"Ordinary bridge mapping and CPU-only/retired solver rejection PASS\n";
}
} // namespace gpu_rigid_pin_api

int main() {
    using namespace gpu_rigid_pin_api;
    try {
        check_bridge_mapping_and_retired_mode_rejection();
        check_rigid_pins();
        std::cout<<"GPU ordinary rigid-pin API parity PASS\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}

// Genuine native CPU-vs-GPU API parity fixture for the GPU CUDA solver.
//
// Every check drives the real `ipc_gpu::solver::GpuSolver` (no host
// substitution, no mocked CUDA, no CPU reroute) and compares it with the real
// CPU reference entry points (`global_jacobi_solver_dat` and
// `global_gauss_seidel_solver_basic_rb`). The binary only runs meaningfully on
// a CUDA host; building it on a non-CUDA machine does not execute any GPU work.
//
// Floating-point tolerance policy. The CPU reference inverts a general 3x3
// Hessian with Eigen and accumulates in a different order than the device's
// rank-revealing 3x3 solve, so bitwise identity is explicitly NOT promised:
//   * conditioned scenes: positions 5e-9, velocities 5e-7 (velocity divides
//     the position round-off by dt, e.g. 5e-9 / 0.025 = 2e-7 < 5e-7),
//   * large-coordinate scene: ordinary (y,z) coordinates 1e-10 and at most
//     4 ULP for the ~1e12 x coordinate only. DAT must consume raw increments,
//     never rounded endpoint-minus-position differences.
// These bounds are absolute and deliberately tight; they must not be relaxed
// to hide a real disagreement.
#include "bridge.h"
#include "../DAT.h"
#include "jacobi_step.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solid_ipc.h"
#include "../solver.h"
#include "time_integration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cpu_parity {
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

// Optional execution paths are selected from the environment when the contact
// pipeline is constructed. Scope an override so the default path is restored.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* previous = std::getenv(name)) {
            had_value_ = true;
            previous_ = previous;
        }
        setenv(name, value, 1);
    }
    ~ScopedEnv() {
        if (had_value_) setenv(name_.c_str(), previous_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
private:
    std::string name_, previous_;
    bool had_value_ = false;
};

bool same_doubles(const std::vector<Vector3>& a, const std::vector<Vector3>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int k = 0; k < 3; ++k)
            if (a[i][k] != b[i][k]) return false;
    return true;
}
bool same_vec3(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int k = 0; k < 3; ++k)
            if (a[i][k] != b[i][k]) return false;
    return true;
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
        parameters.use_jacobi_dat = true;
        parameters.use_ticcd = false;
        parameters.use_ccd_guess = false;
        parameters.node_box_min = .001;
        parameters.node_box_max = .1;
        parameters.theta_box_min = parameters.theta_box_max = .1;
        parameters.node_box_update_count = 2;
        parameters.damping = .5;
        parameters.dat_gamma = .9;
        parameters.dat_query_radius = .5;
        parameters.dat_node_box_padding = 1.2;
    }
};

Fixture cloth();
Fixture solid();
Fixture mixed();
Fixture face_interior_contact();

void refresh_topology(Fixture& fixture) {
    fixture.mesh.num_positions = fixture.state.deformed_positions.size();
    fixture.mesh.node_to_rb.resize(fixture.state.deformed_positions.size(), -1);
    fixture.mesh.build_deformable_nodes();
    fixture.adjacency = build_incident_triangle_map(fixture.mesh.tris);
}

void uniform_velocity(Fixture& fixture, const Vec3& velocity) {
    for (auto& value : fixture.state.velocities) value = velocity;
}

// One genuine CPU DAT substep. The driver entry point is the same one the
// production cloth/general drivers call for --use_jacobi_dat.
SolverResult cpu_substep(Fixture& fixture, bool require_completion = true) {
    const auto previous = fixture.state.deformed_positions;
    auto next = previous;
    std::vector<Vec3> target;
    const double dt = fixture.parameters.dt();
    build_xhat(target, previous, fixture.state.velocities, dt);
    const auto frozen_target = target;
    BroadPhase phase;
    const auto result = global_jacobi_solver_dat(fixture.mesh, fixture.adjacency,
        fixture.pins, fixture.parameters, next, target, fixture.state.velocities,
        phase, "", &previous);
    // The CPU solver contract keeps xhat frozen for the whole substep.
    require(same_vec3(target, frozen_target), "CPU DAT mutated the frozen xhat target");
    if (require_completion) {
        require(result.iterations == fixture.parameters.max_global_iters,
            "CPU fixed-budget DAT reference used a different sweep count");
        if (fixture.parameters.max_global_iters > 0)
            require(result.converged, "CPU fixed-budget DAT reference did not converge");
    }
    update_velocity(fixture.state.velocities, next, previous, dt);
    fixture.state.deformed_positions = std::move(next);
    return result;
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

void compare_state(const StateSnapshot& gpu, const Fixture& cpu, double position_tolerance,
                   double velocity_tolerance, const std::string& label) {
    require(gpu.positions.size() == cpu.state.deformed_positions.size(),
        label + ": GPU/CPU position count mismatch");
    require(gpu.velocities.size() == cpu.state.velocities.size(),
        label + ": GPU/CPU velocity count mismatch");
    for (std::size_t node = 0; node < cpu.state.deformed_positions.size(); ++node)
        for (int axis = 0; axis < 3; ++axis) {
            const double actual = gpu.positions[node][axis];
            const double expected = cpu.state.deformed_positions[node][axis];
            require(std::isfinite(actual), tag(label, node, axis) + ": nonfinite GPU position");
            if (std::abs(actual - expected) > position_tolerance)
                fail(tag(label, node, axis) + ": position " + std::to_string(actual)
                    + " differs from CPU " + std::to_string(expected));
            const double actual_velocity = gpu.velocities[node][axis];
            const double expected_velocity = cpu.state.velocities[node][axis];
            require(std::isfinite(actual_velocity), tag(label, node, axis) + ": nonfinite GPU velocity");
            if (std::abs(actual_velocity - expected_velocity) > velocity_tolerance)
                fail(tag(label, node, axis) + ": velocity " + std::to_string(actual_velocity)
                    + " differs from CPU " + std::to_string(expected_velocity));
        }
}

void check_dat_parity(const Fixture& base, const std::string& label, int substeps) {
    Fixture cpu = base;
    GpuRun gpu(base);
    for (int substep = 0; substep < substeps; ++substep) {
        cpu_substep(cpu);
        gpu.advance();
    }
    compare_state(gpu.snapshot(), cpu, 5e-9, 5e-7, label);
    std::cout << "DAT parity " << label << ": " << substeps << " substep(s), fixed sweeps PASS\n";
}

// Fixed-budget sweeps for every deformable scene kind: one frozen-xhat substep
// at each of the two budgets the CPU reference must reproduce exactly
// (iterations == max_global_iters, converged under fixed_iters).
void check_fixed_sweep_budgets() {
    struct Kind { const char* label; Fixture (*make)(); };
    const Kind kinds[] = {{"cloth", &cloth}, {"solid", &solid}, {"mixed", &mixed}};
    for (const auto& kind : kinds)
        for (int sweeps : {1, 4}) {
            Fixture fixture = kind.make();
            fixture.parameters.max_global_iters = sweeps;
            check_dat_parity(fixture,
                std::string(kind.label) + " " + std::to_string(sweeps) + " sweep(s)", 1);
        }
}

// Face-interior point/triangle contact. The point projects strictly inside the
// triangle (barycentric weights .25/.25/.5), so all three triangle-vertex
// weights are nonzero and every one of the four self roles receives a barrier
// gradient. Contact is the only energy here (no elasticity/bending/gravity),
// so a node that does not move is a solver defect, not a material term.
void check_face_interior_all_roles() {
    const Fixture base = face_interior_contact();
    Fixture cpu = base;
    cpu_substep(cpu);
    for (std::size_t node = 0; node < base.state.deformed_positions.size(); ++node) {
        const double moved = (cpu.state.deformed_positions[node]
            - base.state.deformed_positions[node]).norm();
        require(moved > 1e-9, tag("face-interior CPU", node, 0)
            + ": role received no face-interior barrier displacement");
    }
    GpuRun gpu(base);
    gpu.advance();
    const StateSnapshot actual = gpu.snapshot();
    compare_state(actual, cpu, 5e-9, 5e-7, "face-interior");
    for (std::size_t node = 0; node < base.state.deformed_positions.size(); ++node) {
        const Vec3 delta(actual.positions[node][0] - base.state.deformed_positions[node][0],
            actual.positions[node][1] - base.state.deformed_positions[node][1],
            actual.positions[node][2] - base.state.deformed_positions[node][2]);
        require(delta.norm() > 1e-9, tag("face-interior GPU", node, 0)
            + ": role received no face-interior barrier displacement");
    }
    {
        // FrozenDeformableSnapshot assembly must not collapse to a
        // single-selected-role packet even with the optional compact
        // deformable-contact path enabled.
        ScopedEnv compact("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS", "1");
        GpuRun compact_gpu(base);
        compact_gpu.advance();
        const StateSnapshot compact_state = compact_gpu.snapshot();
        compare_state(compact_state, cpu, 5e-9, 5e-7, "face-interior compact");
        for (std::size_t node = 0; node < base.state.deformed_positions.size(); ++node) {
            const Vec3 delta(compact_state.positions[node][0] - base.state.deformed_positions[node][0],
                compact_state.positions[node][1] - base.state.deformed_positions[node][1],
                compact_state.positions[node][2] - base.state.deformed_positions[node][2]);
            require(delta.norm() > 1e-9, tag("face-interior compact", node, 0)
                + ": compact snapshot mode dropped this role");
        }
    }
    std::cout << "DAT face-interior contact: point and all three triangle roles updated PASS\n";
}

Fixture cloth() {
    Fixture fixture({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)},
        {0, 1, 2, 1, 3, 2});
    fixture.parameters.mu = 30;
    fixture.parameters.lambda = 20;
    fixture.parameters.kB = .003;
    fixture.parameters.gravity = Vec3(0, -.4, 0);
    fixture.state.deformed_positions[0] += Vec3(-.1, .03, .02);
    fixture.state.deformed_positions[3] += Vec3(.06, -.02, .04);
    uniform_velocity(fixture, Vec3(.06, -.03, .02));
    return fixture;
}

void append_solid(Fixture& fixture, const Vec3& offset) {
    const auto first = fixture.state.deformed_positions.size();
    create_solid({offset, offset + Vec3::UnitX(), offset + Vec3::UnitY(), offset + Vec3::UnitZ()},
        {0, 1, 2, 3}, 24., fixture.mesh, fixture.state);
    fixture.state.deformed_positions[first + 1].x() += .08;
    fixture.parameters.solid_mu = 25;
    fixture.parameters.solid_lambda = 35;
    refresh_topology(fixture);
}

Fixture solid() {
    Fixture fixture;
    append_solid(fixture, Vec3::Zero());
    uniform_velocity(fixture, Vec3(.04, -.02, .01));
    return fixture;
}

Fixture mixed() {
    Fixture fixture = cloth();
    append_solid(fixture, Vec3(4, 0, 0));
    uniform_velocity(fixture, Vec3(.05, -.025, .015));
    return fixture;
}

// Obtuse point/triangle contact: sqrt(.0101) < .2 activates the barrier for
// every one of the four self roles (point + three triangle vertices).
Fixture obtuse_contact() {
    Fixture fixture({Vec3(.5, -.01, .1), Vec3::Zero(), Vec3(1, 0, 0), Vec3(-10, .1, 0)},
        {1, 2, 3});
    fixture.parameters.d_hat = .2;
    fixture.parameters.k_barrier = 100.;
    fixture.parameters.damping = .2;
    fixture.parameters.max_global_iters = 2;
    for (int node = 0; node < 4; ++node)
        fixture.state.velocities[node] = Vec3(.2, -.1, .03) * (node + 1);
    return fixture;
}

// Two disjoint triangles whose nearest edges are within d_hat: exercises the
// segment/segment package with all four edge roles updated.
Fixture edge_contact() {
    Fixture fixture({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
                     Vec3(0, .05, .1), Vec3(1, .05, .1), Vec3(.5, 1, .1)},
        {0, 1, 2, 3, 4, 5});
    fixture.parameters.d_hat = .2;
    fixture.parameters.k_barrier = 40.;
    fixture.parameters.damping = .3;
    fixture.parameters.max_global_iters = 2;
    for (int node = 0; node < 6; ++node)
        fixture.state.velocities[node] = Vec3(.01 * (node + 1), -.02, .015);
    return fixture;
}

Fixture face_interior_contact() {
    Fixture fixture({Vec3(.25, .25, .1), Vec3::Zero(), Vec3(1, 0, 0), Vec3(0, 1, 0)},
        {1, 2, 3});
    require(fixture.parameters.mu == 0.0 && fixture.parameters.lambda == 0.0
        && fixture.parameters.kB == 0.0, "face-interior fixture started with elasticity");
    fixture.parameters.d_hat = .2;
    fixture.parameters.k_barrier = 100.;
    fixture.parameters.damping = .5;
    fixture.parameters.max_global_iters = 1;
    require(fixture.parameters.k_sdf == 0.0 && fixture.parameters.kpin == 0.0
        && fixture.parameters.gravity.isZero() && fixture.parameters.friction_coefficient == 0.0,
        "face-interior fixture must contain only the barrier contact");
    for (std::size_t node = 0; node < fixture.mesh.mass.size(); ++node)
        require(fixture.mesh.mass[node] == 1.0, "face-interior fixture mass");
    return fixture;
}

// Simultaneous friction, analytic SDF, and a soft pin on an active contact.
Fixture friction_sdf_pin() {
    Fixture fixture = obtuse_contact();
    fixture.parameters.friction_coefficient = .35;
    fixture.parameters.friction_velocity_epsilon = .02;
    fixture.parameters.k_sdf = 20.;
    fixture.parameters.eps_sdf = .2;
    fixture.parameters.kpin = 15.;
    fixture.parameters.max_global_iters = 3;
    fixture.parameters.sdf_planes.push_back({Vec3(0, 0, -.05), Vec3(0, 0, 1)});
    fixture.pins.push_back(Pin{2, fixture.state.deformed_positions[2] + Vec3(0, .05, 0)});
    return fixture;
}

// The raw Jacobi update would cross the triangle. DAT must truncate it.
Fixture dat_truncation() {
    Fixture fixture({Vec3(0, 0, .2), Vec3(-1, -1, 0), Vec3(2, -1, 0), Vec3(-1, 2, 0)},
        {1, 2, 3});
    const double dt = fixture.parameters.dt();
    fixture.state.velocities[0] = Vec3(0, 0, (-.8 - .2) / dt);
    for (int node = 1; node < 4; ++node)
        fixture.state.velocities[node] = Vec3(0, 0, (.6 - 0.0) / dt);
    fixture.parameters.dat_query_radius = 1.;
    fixture.parameters.damping = .5;
    return fixture;
}

void check_dat_truncation() {
    const Fixture base = dat_truncation();
    const auto previous = base.state.deformed_positions;
    std::vector<Vec3> xhat;
    build_xhat(xhat, previous, base.state.velocities, base.parameters.dt());
    const auto raw = jacobi_step(base.mesh, base.adjacency, base.pins, base.parameters,
        previous, xhat, base.state.velocities, &previous);
    Fixture cpu = base;
    cpu_substep(cpu);
    double deviation = 0;
    for (std::size_t node = 0; node < previous.size(); ++node)
        deviation = std::max(deviation, (previous[node] + raw[node] - cpu.state.deformed_positions[node]).norm());
    require(deviation > 1e-6, "DAT truncation fixture did not exercise the raw-update filter");
    check_dat_parity(base, "truncation", 1);
}

// ~1e12 world coordinates: DAT consumes raw increments, so ordinary y/z
// coordinates must match the CPU to 1e-10 while x is compared in ULP.
Fixture large_coordinate() {
    const double x = 1e12;
    const double ulp = std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
    Fixture fixture({Vec3(x + 100 * ulp, 0, 0), Vec3(x, -3, -3), Vec3(x, 3, -3), Vec3(x, 0, 3)},
        {1, 2, 3});
    fixture.parameters.fps = 1.;
    fixture.parameters.damping = 1.;
    fixture.parameters.dat_query_radius = 1.;
    fixture.parameters.gravity = Vec3(-2e-5, 0, 0);
    fixture.parameters.kpin = 1.;
    fixture.pins.push_back(Pin{0, fixture.state.deformed_positions[0]});
    fixture.state.velocities[0] = Vec3(0, .4, 0);
    for (int node = 1; node < 4; ++node)
        fixture.state.velocities[node] = Vec3(.4, 0, 0);
    return fixture;
}

void check_large_coordinate() {
    const Fixture base = large_coordinate();
    Fixture cpu = base;
    cpu_substep(cpu);
    GpuRun gpu(base);
    gpu.advance();
    const auto actual = gpu.snapshot();
    require(actual.positions.size() == cpu.state.deformed_positions.size(),
        "large-coordinate position count mismatch");
    for (std::size_t node = 0; node < cpu.state.deformed_positions.size(); ++node) {
        for (int axis = 0; axis < 3; ++axis) {
            const double expected = cpu.state.deformed_positions[node][axis];
            const double value = actual.positions[node][axis];
            require(std::isfinite(value), tag("large coordinate", node, axis) + ": nonfinite GPU position");
            const double ulp = 4. * (std::nextafter(std::max(std::abs(expected), std::abs(value)),
                std::numeric_limits<double>::infinity())
                - std::max(std::abs(expected), std::abs(value)));
            const double tolerance = axis == 0 ? ulp : 1e-10;
            if (std::abs(value - expected) > tolerance)
                fail(tag("large coordinate", node, axis) + ": raw-increment mismatch GPU "
                    + std::to_string(value) + " CPU " + std::to_string(expected));
            // Velocity stays an incremental quantity; dt = 1 here.
            const double velocity = actual.velocities[node][axis];
            const double expected_velocity = cpu.state.velocities[node][axis];
            require(std::isfinite(velocity), tag("large coordinate", node, axis) + ": nonfinite GPU velocity");
            if (std::abs(velocity - expected_velocity) > (axis == 0 ? ulp : 1e-7))
                fail(tag("large coordinate", node, axis) + ": velocity mismatch");
        }
    }
    // The inertial proposal for the pinned node must have survived, not just
    // been rounded away with the world coordinate.
    require(std::abs(actual.positions[0][1] - base.state.deformed_positions[0][1]) > 1e-3,
        "large-coordinate fixture produced no observable ordinary-coordinate motion");
    std::cout << "DAT large-coordinate raw increment: 1e-10 ordinary, 4 ULP world-x PASS\n";
}

void check_dat_guess_suppression() {
    const Fixture base = dat_truncation();
    GpuRun baseline(base);
    baseline.advance();
    const auto expected = baseline.snapshot();
    for (int variant = 0; variant < 3; ++variant) {
        StepParameters step = baseline.step;
        step.global_ccd_guess = true;
        step.colored_guess_level_based = true;
        step.colored_guess_fraction = .25;
        step.colored_guess_sweeps = variant == 0 ? 0 : 3;
        GpuRun run(base);
        run.advance(step);
        const auto actual = run.snapshot();
        require(same_doubles(actual.positions, expected.positions)
            && same_doubles(actual.velocities, expected.velocities),
            "Jacobi DAT honored an initial guess/colored sweep request");
    }
    std::cout << "DAT guess suppression: legacy/level flags with zero and nonzero sweeps PASS\n";
}

// Finite-but-inverse-overflow soft pin. With dt = 1 and mass 1,
// kpin = -nextafter(1, 0) makes H = m/dt^2 + kpin = 2^-53 ~ 1.1e-16 on every
// axis while the gradient kpin*(x - target) is ~1e308: both are finite, but the
// raw -damping * H^{-1} g genuinely overflows. Damping 0 must therefore skip
// the inverse and retain the state; any positive damping must fail closed
// instead of publishing an infinite proposal.
Fixture inverse_overflow() {
    Fixture fixture({Vec3::Zero()}, {});
    // The direct CPU local-system API expects an adjacency row even for an isolated node.
    fixture.adjacency.emplace(0, IncidentTriangles{});
    // dt must be pinned before anything populates the lazy dt/dt2 cache.
    fixture.parameters.fps = 1.;
    fixture.parameters.substeps = 1;
    require(fixture.parameters.dt() == 1.0 && fixture.parameters.dt2() == 1.0,
        "overflow fixture timestep");
    fixture.parameters.gravity = Vec3::Zero();
    fixture.parameters.d_hat = 0.;
    fixture.parameters.k_barrier = 0.;
    fixture.parameters.k_sdf = 0.;
    fixture.parameters.eps_sdf = 0.;
    fixture.parameters.friction_coefficient = 0.;
    fixture.parameters.damping = 0.;
    fixture.parameters.max_global_iters = 1;
    fixture.parameters.kpin = -std::nextafter(1.0, 0.0);
    // DAT needs only its own explicit positive query radius; the legacy
    // adaptive-box limits are irrelevant to this path and must not be needed.
    fixture.parameters.dat_query_radius = 1.;
    fixture.parameters.node_box_min = 0.;
    fixture.parameters.node_box_max = 0.;
    fixture.parameters.theta_box_min = 0.;
    fixture.parameters.theta_box_max = 0.;
    fixture.pins.push_back(Pin{0, Vec3(1e308, 0, 0)});
    require(fixture.mesh.mass.size() == 1 && fixture.mesh.mass[0] == 1.0,
        "overflow fixture mass");
    require(fixture.parameters.mu == 0.0 && fixture.parameters.lambda == 0.0
        && fixture.parameters.solid_mu == 0.0 && fixture.parameters.solid_lambda == 0.0
        && fixture.parameters.kB == 0.0 && fixture.parameters.k_sdf == 0.0
        && fixture.parameters.d_hat == 0.0 && fixture.parameters.k_barrier == 0.0
        && fixture.parameters.friction_coefficient == 0.0,
        "overflow fixture must contain only the soft-pin energy");
    require(fixture.pins.size() == 1 && fixture.pins[0].vertex_index == 0,
        "overflow fixture must pin the isolated node");
    return fixture;
}

void check_zero_damping_skips_inverse() {
    const Fixture base = inverse_overflow();
    // Show that the local system really is finite but numerically singular:
    // a huge finite gradient over a ~1e-16 Hessian, not a fabricated stub.
    std::vector<Vec3> xhat;
    build_xhat(xhat, base.state.deformed_positions, base.state.velocities, base.parameters.dt());
    const PinMap pin_map = build_pin_map(base.pins, 1);
    const auto local = compute_local_gradient_and_hessian_no_barrier(
        0, base.mesh, base.adjacency, base.pins, base.parameters,
        base.state.deformed_positions, xhat, &pin_map);
    require(std::isfinite(local.first.x()) && std::abs(local.first.x()) > 1e300,
        "overflow fixture did not produce a huge finite soft-pin gradient");
    const double cancellation = 1.0 + base.parameters.kpin;
    require(std::isfinite(local.second(0, 0))
        && std::abs(local.second(0, 0) - cancellation) <= 1e-30,
        "overflow fixture Hessian is not the finite cancellation value");
    require(local.second(0, 0) == local.second(1, 1)
        && local.second(1, 1) == local.second(2, 2),
        "overflow fixture Hessian is not the isotropic soft-pin value");

    // Zero damping: both implementations skip the inverse and retain the state.
    Fixture cpu = base;
    cpu_substep(cpu);
    require(same_vec3(cpu.state.deformed_positions, base.state.deformed_positions),
        "CPU zero damping moved a node");
    for (const auto& velocity : cpu.state.velocities)
        require(velocity.isZero(), "CPU zero damping changed a velocity");
    GpuRun gpu(base);
    gpu.advance();
    const auto actual = gpu.snapshot();
    for (std::size_t node = 0; node < base.state.deformed_positions.size(); ++node)
        for (int axis = 0; axis < 3; ++axis) {
            require(actual.positions[node][axis] == base.state.deformed_positions[node][axis],
                tag("zero damping GPU", node, axis) + ": position changed");
            require(actual.velocities[node][axis] == 0.0,
                tag("zero damping GPU", node, axis) + ": velocity not zeroed");
        }

    // Positive damping counterpart: the same finite system must now fail closed.
    Fixture positive = base;
    positive.parameters.damping = 1.;
    bool cpu_failed = false;
    try { cpu_substep(positive, false); }
    catch (const std::exception&) { cpu_failed = true; }
    require(cpu_failed, "CPU positive damping published an overflowing proposal");
    require(same_vec3(positive.state.deformed_positions, base.state.deformed_positions),
        "CPU failed overflow proposal still changed the state");
    GpuRun positive_gpu(positive);
    bool gpu_failed = false;
    try { positive_gpu.advance(); }
    catch (const std::exception&) { gpu_failed = true; }
    require(gpu_failed, "GPU positive damping published an overflowing proposal");
    bool snapshot_failed = false;
    try { (void)positive_gpu.snapshot(); }
    catch (const std::exception&) { snapshot_failed = true; }
    require(snapshot_failed, "GPU exposed a snapshot after a failed overflow substep");
    std::cout << "DAT zero damping: inverse bypassed on ~1e-16 Hessian; positive damping fails closed PASS\n";
}

void check_zero_sweeps() {
    Fixture base = obtuse_contact();
    base.parameters.max_global_iters = 0;
    Fixture cpu = base;
    cpu_substep(cpu, false);
    GpuRun gpu(base);
    gpu.advance();
    const auto actual = gpu.snapshot();
    compare_state(actual, cpu, 5e-9, 5e-7, "zero sweeps");
    for (std::size_t node = 0; node < base.state.deformed_positions.size(); ++node)
        for (int axis = 0; axis < 3; ++axis)
            require(std::abs(actual.positions[node][axis] - base.state.deformed_positions[node][axis]) < 1e-30,
                tag("zero sweeps", node, axis) + ": zero-sweep DAT moved a node");
    std::cout << "DAT zero sweeps: no sweep, velocity integration only PASS\n";
}

void check_invalid_pre_update_rejection() {
    const Fixture base = obtuse_contact();
    GpuRun gpu(base);
    const auto before = gpu.snapshot();
    for (int variant = 0; variant < 8; ++variant) {
        StepParameters bad = gpu.step;
        if (variant == 0) bad.dat_gamma = 1.0;
        if (variant == 1) bad.dat_gamma = 0.0;
        if (variant == 2) bad.dat_query_radius = -1.0;
        if (variant == 3) bad.dat_node_box_padding = .5;
        if (variant == 4) bad.damping = -1.0;
        if (variant == 5) bad.contact.d_hat = -1.0;
        if (variant == 6) bad.iterations = -1;
        if (variant == 7) bad.contact.dt = bad.dt * .5;
        bool rejected = false;
        try { gpu.advance(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid DAT step was not rejected before any update");
        const auto after = gpu.snapshot();
        require(same_doubles(after.positions, before.positions)
            && same_doubles(after.velocities, before.velocities),
            "invalid DAT step changed evolving state");
    }
    // The solver must stay usable after a rejected request.
    gpu.advance();
    std::cout << "DAT invalid requests rejected before update; solver remains usable PASS\n";
}

void check_rigid_dat_rejection() {
    // describe_scene rejects a rigid-body Jacobi DAT scene outright.
    SimParams params = SimParams::zeros();
    params.fixed_iters = true;
    params.use_jacobi_dat = true;
    params.dat_query_radius = .5;
    RefMesh mesh;
    DeformedState state;
    create_rigid_body({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)}, Vec3::Zero(),
        Vec4(1, 0, 0, 0), Vec3::Zero(), 1.0, mesh, state,
        RigidBodyUpdateMode::TranslationOnly);
    bool rejected = false;
    try { (void)describe_scene(mesh, state, {}, params); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "rigid-body Jacobi DAT scene was not rejected by the bridge");
}

void check_bridge_mapping() {
    SimParams params = SimParams::zeros();
    params.fixed_iters = true;
    params.use_jacobi_dat = true;
    params.dat_query_radius = 0.0;
    params.d_hat = .02;
    params.node_box_max = .1;
    params.dat_gamma = .8;
    params.dat_node_box_padding = 1.3;
    params.rigid_pin_time = .375;
    params.kpin = 2.5;
    params.max_global_iters = 4;
    params.damping = .25;
    params.node_box_min = .002;
    params.node_box_update_count = 3;
    params.theta_box_max = .07;
    auto step = describe_step(params);
    require(step.dat_query_radius == 1.5 * params.d_hat, "DAT automatic radius from d_hat changed");
    require(step.dat_gamma == params.dat_gamma && step.dat_node_box_padding == params.dat_node_box_padding,
        "DAT gamma/padding mapping changed");
    require(step.jacobi_dat && step.iterations == params.max_global_iters, "DAT iterations mapping changed");
    require(step.damping == params.damping, "DAT damping mapping changed");
    require(step.rigid_pin_time == params.rigid_pin_time, "rigid pin time mapping changed");
    require(step.elastic.kpin == params.kpin, "soft pin stiffness mapping changed");
    require(step.box_radius_min == params.node_box_min && step.box_radius == params.node_box_max,
        "adaptive box radius mapping changed");
    require(step.rebuild_interval == params.node_box_update_count && step.rotation_radius == params.theta_box_max,
        "rebuild/rotation mapping changed");
    require(!step.global_ccd_guess && step.colored_guess_sweeps == 0,
        "DAT bridge enabled an initial guess");
    params.dat_query_radius = .3;
    require(describe_step(params).dat_query_radius == .3, "explicit DAT query radius ignored");
    params.dat_query_radius = 0.0;
    params.d_hat = 0.0;
    require(describe_step(params).dat_query_radius == params.node_box_max,
        "DAT automatic radius fallback to node_box_max changed");
    // Colored/legacy guesses stay suppressed even with nonzero CPU sweep flags.
    params.use_colored_ccd_guess = true;
    params.colored_ccd_guess_iters = 3;
    params.use_verlet_guess = true;
    const auto suppressed = describe_step(params);
    require(suppressed.jacobi_dat && !suppressed.global_ccd_guess
        && suppressed.colored_guess_sweeps == 0, "DAT bridge forwarded a guess request");
    // Unsupported modes must fail closed instead of silently routing to CPU.
    for (int variant = 0; variant < 8; ++variant) {
        SimParams bad = params;
        bad.use_colored_ccd_guess = false;
        bad.colored_ccd_guess_iters = 0;
        bad.use_verlet_guess = false;
        bad.use_jacobi_dat = false;
        if (variant == 0) bad.fixed_iters = false;
        if (variant == 1) bad.use_ogc = true;
        if (variant == 2) bad.use_ogc_solver = true;
        if (variant == 3) bad.use_cloth_grid = true;
        if (variant == 4) bad.use_ticcd = true;
        if (variant == 5) {
            // Both colored modes at once is still an invalid CPU/GPU request.
            bad.use_colored_ccd_guess = true;
            bad.use_colored_ccd_guess_level_based = true;
        }
        if (variant == 6) bad.use_verlet_guess = true;
        if (variant == 7) bad.use_translation_guess = true;
        bool rejected = false;
        try { (void)describe_step(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unsupported solver/guess mode did not fail closed");
    }
    std::cout << "Bridge DAT mapping, radius resolution, guess suppression and fail-closed modes PASS\n";
}

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
    fixture.parameters.use_jacobi_dat = false;
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
    const auto result = global_gauss_seidel_solver_basic_rb(fixture.mesh, fixture.state,
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

// The bridge rejects a rigid DAT scene at construction; the solver's own
// advance_substep must also reject a well-formed DAT request on a rigid scene
// before touching any state, and stay usable afterwards.
void check_rigid_dat_advance_rejection() {
    const Fixture base = rigid_pair();
    require(base.parameters.use_jacobi_dat == false,
        "rigid DAT advance fixture must start from a valid non-DAT rigid scene");
    GpuRun gpu(base);
    const auto before = gpu.snapshot();
    SimParams dat_params = base.parameters;
    dat_params.use_jacobi_dat = true;
    const StepParameters dat_step = describe_step(dat_params);
    require(dat_step.jacobi_dat, "rigid DAT step description lost the DAT request");
    bool rejected = false;
    try { gpu.advance(dat_step); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "advance_substep accepted Jacobi DAT on a rigid scene");
    const auto after = gpu.snapshot();
    require(same_doubles(after.positions, before.positions)
        && same_doubles(after.velocities, before.velocities)
        && same_doubles(after.com, before.com)
        && same_doubles(after.com_velocity, before.com_velocity)
        && same_doubles(after.omega, before.omega),
        "rejected rigid DAT substep changed evolving state");
    require(after.orientation.size() == before.orientation.size(),
        "rejected rigid DAT substep changed the orientation count");
    for (std::size_t body = 0; body < before.orientation.size(); ++body)
        require(after.orientation[body] == before.orientation[body],
            "rejected rigid DAT substep changed a rigid orientation");
    gpu.advance(); // A rejected request must not poison the solver.
    std::cout << "GpuSolver rigid+DAT substep rejected pre-update; state intact PASS\n";
}

void check_runtime_failure_blocks_snapshot() {
    // The isolated point sits exactly on the plane of a nonincident triangle:
    // planar DAT cannot certify a separating plane.
    Fixture fixture({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(-1, -1, 0)},
        {1, 2, 3});
    fixture.parameters.mu = 1;
    fixture.parameters.lambda = 1;
    fixture.parameters.dat_query_radius = 1.;
    fixture.state.velocities[0] = Vec3(0, 0, .1);
    bool cpu_rejected = false;
    try { Fixture cpu = fixture; cpu_substep(cpu, false); }
    catch (const std::exception&) { cpu_rejected = true; }
    require(cpu_rejected, "CPU DAT accepted a zero-gap nonincident pair");
    GpuRun gpu(fixture);
    bool gpu_rejected = false;
    try { gpu.advance(); }
    catch (const std::exception&) { gpu_rejected = true; }
    require(gpu_rejected, "GPU DAT accepted a zero-gap nonincident pair");
    bool snapshot_rejected = false;
    try { (void)gpu.snapshot(); }
    catch (const std::exception&) { snapshot_rejected = true; }
    require(snapshot_rejected, "GPU published a snapshot after a failed DAT sweep");
    std::cout << "DAT zero-gap failure: pre-publish rejection and invalidated snapshot PASS\n";
}

} // namespace cpu_parity

int main() {
    using namespace cpu_parity;
    try {
        check_dat_parity(cloth(), "cloth", 2);
        check_dat_parity(solid(), "solid", 1);
        check_dat_parity(mixed(), "mixed", 2);
        check_dat_parity(obtuse_contact(), "obtuse PT", 1);
        check_dat_parity(edge_contact(), "edge/edge", 1);
        check_face_interior_all_roles();
        check_dat_parity(friction_sdf_pin(), "friction/SDF/soft pin", 2);
        check_fixed_sweep_budgets();
        check_dat_truncation();
        check_large_coordinate();
        check_dat_guess_suppression();
        check_zero_damping_skips_inverse();
        check_zero_sweeps();
        check_invalid_pre_update_rejection();
        check_rigid_dat_rejection();
        check_bridge_mapping();
        check_rigid_pins();
        check_rigid_dat_advance_rejection();
        check_runtime_failure_blocks_snapshot();
        std::cout << "GPU CPU/GPU API parity fixture PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

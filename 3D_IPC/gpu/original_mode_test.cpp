// Original-mode routing/contract test. Host-only helpers (device-capable
// headers, bridge mapping and CPU controls) run first; the solver routing
// checks construct the real GpuSolver, and the companion original_mode_test.cu
// exercises the device kernels directly.
#include "bridge.h"
#include "solver.h"
#include "exact.cuh"
#include "level_schedule.cuh"
#include "../solver.h"
#include "time_integration.h"
#include "simulation.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int gpu_original_mode_device_check();
int gpu_original_integration_check();

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

SimParams original_cloth_params() {
    SimParams p = SimParams::zeros();
    p.fixed_iters = true;
    p.use_ticcd = false;
    p.use_parallel = true;
    p.use_basic_experimental = true;
    p.use_basic_experimental_v2 = true;
    p.use_simd = true;
    p.use_ccd = true;
    p.use_ogc = false;
    p.d_hat = 0.005;
    p.friction_coefficient = 0.0;
    p.exact_computation_fallback = false;
    return p;
}

void check_step_mapping() {
    const auto original = ipc_gpu::solver::describe_step(original_cloth_params());
    require(!original.exact_computation_fallback,
        "original mode flag was not forwarded");
    require(original.original_reference_threads == 64,
        "default CPU64 reference team changed");
    require(original.original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "experimental-v2 original dispatch did not select the cooperative family");
    require(!original.colored_guess_enabled,
        "predictor-intent metadata must default to false");

    // The family is scheduler metadata, NOT a geometry toggle: the CPU builds
    // the same selected vertex set for both exact_computation_fallback values.
    // Only the per-vertex cutoff rule differs (validate_cooperative_step).
    SimParams exact = original_cloth_params();
    exact.exact_computation_fallback = true;
    require(ipc_gpu::solver::describe_step(exact).original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "exact mode lost the cooperative family metadata");
    exact.use_ccd = false;
    require(ipc_gpu::solver::describe_step(exact).original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "CCD-off exact mode lost the cooperative family metadata");

    // Friction or a zero d_hat removes the CPU contact sweep entirely.
    SimParams friction = original_cloth_params();
    friction.friction_coefficient = 0.1;
    require(ipc_gpu::solver::describe_step(friction).original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::StoredNoop,
        "friction dispatch must stay StoredNoop");
    SimParams no_gap = original_cloth_params();
    no_gap.d_hat = 0.0;
    require(ipc_gpu::solver::describe_step(no_gap).original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::StoredNoop,
        "zero d_hat dispatch must stay StoredNoop");

    // Basic and experimental-v1 dispatches remain the noncooperative CPU rule.
    SimParams basic = original_cloth_params();
    basic.use_basic_experimental = false;
    basic.use_basic_experimental_v2 = false;
    basic.use_simd = false;
    require(ipc_gpu::solver::describe_step(basic).original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::StoredNoop,
        "basic CPU dispatch must stay StoredNoop");

    require(ipc_gpu::solver::describe_step(original_cloth_params(), 1)
            .original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::StoredNoop,
        "reference team 1 must be the noncooperative StoredNoop rule");
    require(ipc_gpu::solver::describe_step(original_cloth_params(), 2)
            .original_reference_threads == 2,
        "explicit direct-API reference team was not preserved");

    for (int threads : {0, -1, 4097}) {
        bool rejected = false;
        try {
            (void)ipc_gpu::solver::describe_step(original_cloth_params(), threads);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "unbounded reference team was not rejected");
    }
}

// The original predictor's movement box must be padded with the CPU raw
// intended displacement, never with target-start. These are the exact binary64
// values produced by the current CPU initial_guess.cpp/solver.cpp at
// fraction 0.97 (compiled/run control, not a synthetic rounding example): the
// stored target and the raw displacement differ by 3 ULP, so a target-only pad
// silently shrinks the box.
void check_original_guess_box_padding() {
    const double start = std::strtod("-0x1.a461117f5232ap-1", nullptr);
    const double xhat = std::strtod("-0x1.6d4f72c869f3ap-1", nullptr);
    const double raw = (xhat - start) * 0.97;
    const double target = start + raw;
    require(raw == std::strtod("0x1.ab5587f1c28cbp-4", nullptr),
        "CPU raw fraction displacement changed");
    require(target == std::strtod("-0x1.6ef6608119e11p-1", nullptr),
        "CPU stored target changed");
    require(target - start == std::strtod("0x1.ab5587f1c28c8p-4", nullptr),
        "target-start still differs from raw by 3 ULP");
    const double scale = std::max(std::fabs(start), std::fabs(target));
    const double pad = std::max(1.0e-8,
        std::max(0.1 * std::fabs(raw), 64.0 * DBL_EPSILON * scale));
    const double target_only_pad = std::max(1.0e-8,
        std::max(0.1 * std::fabs(target - start), 64.0 * DBL_EPSILON * scale));
    require(pad == std::strtod("0x1.55de065b020a3p-7", nullptr),
        "raw-padded CPU box padding changed");
    require(target_only_pad == std::strtod("0x1.55de065b020a0p-7", nullptr),
        "target-only padding control changed");
    require(std::min(start, target) - pad
            == std::strtod("-0x1.a9b88998be3adp-1", nullptr),
        "raw-padded CPU box lower bound changed");
    require(std::min(start, target) - target_only_pad
            == std::strtod("-0x1.a9b88998be3acp-1", nullptr),
        "target-only box lower bound control changed");
}

void check_collision_only_certificate() {
    using ipc_gpu::exact::Status;
    const double triangle[4][3] = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
    // Unchanged healthy positive gap: Safe (must not be relabelled t0 contact).
    {
        const double start[4][3] = {
            {0.2, 0.2, 0.1}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::Safe, "healthy unchanged positive-gap endpoint was rejected");
    }
    // Truly zero initial geometric gap fails safely.
    {
        const double start[4][3] = {
            {0.2, 0.2, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::InitialContact,
                "true zero initial gap was not reported as InitialContact");
    }
    // The actual represented path crossing the triangle is rejected.
    {
        const double start[4][3] = {
            {0.2, 0.2, 0.1}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        end[0][2] = -0.1;
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::Unsafe, "crossing represented path was accepted");
    }
    // Two moving vertices are outside the approved collision-only contract;
    // the retired rounded-double translating-edge extension must not return.
    {
        const double start[4][3] = {
            {0.2, 0.2, 0.1}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        end[0][2] = 0.2;
        end[1][2] = 0.4;
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::Invalid, "unsupported two-moving path was not rejected");
    }
    // The all-zero primitive is an exact zero initial intersection, not a
    // fake-floor resource/unsafe case.
    {
        const double start[4][3] = {
            {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::InitialContact, "all-zero initial primitive was not InitialContact");
    }
    // A path that passes beside the triangle stays Safe.
    {
        const double start[4][3] = {
            {2.0, 2.0, 0.1}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        double end[4][3];
        std::memcpy(end, start, sizeof(end));
        end[0][2] = -0.1;
        require(ipc_gpu::exact::validate_collision_step<128>(start, end, true)
                == Status::Safe, "separated represented path was rejected");
    }
    (void)triangle;
}

// Actual GpuSolver routing control: the CPU OMP2 selected tiny step must be a
// no-op while the reference-team-1 StoredNoop control keeps the small move.
// This exercises describe_step -> describe_scene -> advance_substep and the
// device-computed cooperative node mask, not just the isolated components.
void check_solver_cooperative_route() {
    RefMesh mesh;
    DeformedState state;
    const std::size_t n = 1;
    state.deformed_positions.assign(n, Vec3::Zero());
    state.velocities.assign(n, Vec3::Zero());
    mesh.initialize(state.deformed_positions);
    mesh.mass.assign(n, 1.0);
    mesh.node_to_rb.assign(n, -1);
    mesh.build_deformable_nodes();
    SimParams p = SimParams::zeros();
    p.fps = 40; p.substeps = 1;
    p.fixed_iters = true; p.max_global_iters = 1;
    p.use_basic_experimental = true; p.use_basic_experimental_v2 = true;
    p.use_simd = true; p.use_parallel = true;
    p.use_ticcd = false; p.use_ccd = true;
    p.use_ccd_guess = false;
    p.use_colored_ccd_guess = false;
    p.use_colored_ccd_guess_level_based = false;
    p.exact_computation_fallback = false;
    p.exact_computation_fallback_2 = false;
    p.node_box_min = p.node_box_max = 0.1;
    p.node_box_update_count = 3;
    p.damping = 1.0; p.d_hat = 0.005; p.k_barrier = 0.0;
    state.velocities[0].x() = 2e-13;

    const auto cooperative = ipc_gpu::solver::describe_step(p, 2);
    require(cooperative.original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "solver cooperative family metadata missing");
    ipc_gpu::solver::GpuSolver selected(
        ipc_gpu::solver::describe_scene(mesh, state, {}, p));
    selected.advance_substep(cooperative,
        ipc_gpu::solver::describe_pin_targets(n, {}), {});
    require(selected.snapshot().positions[0][0] == 0.0,
        "selected cooperative tiny step was not a stored no-op");

    const auto stored = ipc_gpu::solver::describe_step(p, 1);
    require(stored.original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::StoredNoop,
        "reference team 1 did not select StoredNoop");
    ipc_gpu::solver::GpuSolver scalar(
        ipc_gpu::solver::describe_scene(mesh, state, {}, p));
    scalar.advance_substep(stored,
        ipc_gpu::solver::describe_pin_targets(n, {}), {});
    require(scalar.snapshot().positions[0][0] > 0.0,
        "StoredNoop control discarded the representable small move");

    // CPU validate_cooperative_step = exact && use_ccd && !ticcd && !ogc. The
    // selected cutoff is therefore used by exact mode too when CCD is off, and
    // the exact+CCD-on path must keep the same small move.
    SimParams exact = p;
    exact.exact_computation_fallback = true;
    exact.exact_computation_fallback_2 = true;
    auto exact_ccd_off_params = exact;
    exact_ccd_off_params.use_ccd = false;
    const auto exact_ccd_off = ipc_gpu::solver::describe_step(exact_ccd_off_params, 2);
    require(exact_ccd_off.exact_computation_fallback,
        "exact mode flag was not forwarded");
    require(exact_ccd_off.original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "exact+CCD-off lost the cooperative family metadata");
    ipc_gpu::solver::GpuSolver exact_off(
        ipc_gpu::solver::describe_scene(mesh, state, {}, exact_ccd_off_params));
    exact_off.advance_substep(exact_ccd_off,
        ipc_gpu::solver::describe_pin_targets(n, {}), {});
    require(exact_off.snapshot().positions[0][0] == 0.0,
        "exact+CCD-off selected cooperative tiny step was not a stored no-op");

    const auto exact_ccd_on = ipc_gpu::solver::describe_step(exact, 2);
    require(exact_ccd_on.original_update_convention
            == ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm,
        "exact+CCD-on lost the cooperative family metadata");
    ipc_gpu::solver::GpuSolver exact_on(
        ipc_gpu::solver::describe_scene(mesh, state, {}, exact));
    exact_on.advance_substep(exact_ccd_on,
        ipc_gpu::solver::describe_pin_targets(n, {}), {});
    require(exact_on.snapshot().positions[0][0] > 0.0,
        "exact+CCD-on discarded the representable small move");

    // Direct GpuSolver API must reject the unqualified original+direct-bound
    // combination before any state mutation.
    auto bound = ipc_gpu::solver::describe_step(p, 2);
    bound.ccd_exact_step_bound = true;
    const auto before = selected.snapshot();
    bool rejected = false;
    try {
        selected.advance_substep(bound,
            ipc_gpu::solver::describe_pin_targets(n, {}), {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "false + ccd_exact_step_bound was not rejected by the direct API");
    require(selected.snapshot().positions == before.positions,
        "rejected direct-bound request mutated solver state");
}

// Original predictor caller coverage: enabled-with-zero-budget is a real
// request (the CPU validates forced positions/targets before its zero-budget
// return), and both ORIGINAL predictor modes must accept fractions 0/0.97/1
// with budgets 0/1/3 without error. Sweeps are isolated by running the
// ordinary iterations at zero.
void check_original_predictor_budgets() {
    RefMesh mesh;
    DeformedState state;
    const std::size_t n = 1;
    state.deformed_positions.assign(n, Vec3::Zero());
    state.velocities.assign(n, Vec3::Zero());
    mesh.initialize(state.deformed_positions);
    mesh.mass.assign(n, 1.0);
    mesh.node_to_rb.assign(n, -1);
    mesh.build_deformable_nodes();
    state.velocities[0].x() = 0.5;
    SimParams p = SimParams::zeros();
    p.fps = 40;
    p.substeps = 1;
    p.fixed_iters = true;
    p.max_global_iters = 0;
    p.use_basic_experimental = true;
    p.use_basic_experimental_v2 = true;
    p.use_simd = true;
    p.use_parallel = true;
    p.use_ticcd = false;
    p.use_ccd = true;
    p.d_hat = 0.005;
    p.k_barrier = 0.0;
    p.node_box_min = p.node_box_max = 0.1;
    p.node_box_update_count = 3;
    p.exact_computation_fallback = false;
    p.exact_computation_fallback_2 = false;
    p.use_colored_ccd_guess = true;
    const auto scene = ipc_gpu::solver::describe_scene(mesh, state, {}, p);
    for (const bool level : {false, true}) {
        for (const double fraction : {0.0, 0.97, 1.0}) {
            for (const int budget : {0, 1, 3}) {
                SimParams guess = p;
                // The CPU parameter validator forbids selecting both modes.
                guess.use_colored_ccd_guess = !level;
                guess.use_colored_ccd_guess_level_based = level;
                guess.colored_ccd_guess_fraction = fraction;
                guess.colored_ccd_guess_iters = budget;
                const auto step = ipc_gpu::solver::describe_step(guess);
                require(step.colored_guess_enabled,
                    "predictor intent was not forwarded to the GPU solver");
                require(step.colored_guess_sweeps == budget,
                    "predictor budget was not forwarded");
                ipc_gpu::solver::GpuSolver solver(scene);
                solver.advance_substep(step, scene.pin_targets, {});
                const auto output = solver.snapshot();
                const double moved = output.positions[0][0];
                if (budget == 0 || fraction == 0.0) {
                    require(moved == 0.0,
                        "zero-budget or zero-fraction original predictor moved a vertex");
                    continue;
                }
                // The target is previous + fraction*(xhat-previous); with no
                // contacts the predictor reaches it in one sweep.
                require(moved > 0.0 && moved <= 0.5 * step.dt * (1.0 + 1e-6),
                    "original predictor target fraction was not applied");
            }
        }
    }
}

// Host-level (device-capable) scheduler helper coverage: disconnected
// CPU control for the joined whole-sweep stop rule (repair of the legacy
// original predictor). One sweep leaves the point within the target tolerance
// but NOT at its target, so a further requested sweep must not move it. The
// compiled rule is the shared predictor_sweep_bits/predictor_sweep_finished
// helper used by the device status kernels.
void check_predictor_stop_control() {
    using namespace ipc_gpu::solver::level_predictor;
    const double point = std::strtod("0x1.5799015c21d5fp-27", nullptr);
    const double target = std::strtod("0x1.5798ebe292cfcp-27", nullptr);
    const double residual = target - point;
    require(point != target,
        "coupled stop control lost its representable endpoint gap");
    require(residual * residual == std::strtod("0x1.cd2b2f87a149cp-94", nullptr),
        "coupled stop control residual changed");
    require(residual * residual < 1e-28,
        "coupled stop control is no longer inside the tolerance");
    const unsigned tolerance_only = predictor_sweep_bits(true, residual * residual);
    require(tolerance_only == 1u,
        "moved-but-converged vertex changed the sweep status bits");
    require(predictor_sweep_finished(tolerance_only),
        "tolerance-only sweep was not a whole-sweep stop");
    require(predictor_sweep_finished(predictor_sweep_bits(false, 0.0)),
        "no-stored-movement sweep did not stop");
    // Strict inequality: exactly 1e-28 is still an unfinished sweep.
    require(!predictor_sweep_finished(predictor_sweep_bits(true, 1e-28)),
        "a still-over-tolerance moving sweep stopped early");
    require(predictor_sweep_finished(
                predictor_sweep_bits(true, std::nextafter(1e-28, 0.0))),
        "the stop rule is not strict below 1e-28");
}

// Host-level (device-capable) scheduler helper coverage: disconnected
// components, all exact movement ties seeded, a zero-movement component and a
// rigid obstacle that is never scheduled. Fraction/sweep budget coverage is in
// the CUDA integration test and the protected acceptance.
void check_level_helper_schedule() {
    using namespace ipc_gpu::solver::level_predictor;
    constexpr int n = 4;
    // Component {0,1}; isolated 2; rigid isolated 3.
    const std::size_t row[n + 1] = {0, 1, 2, 2, 2};
    const int adjacency[2] = {1, 0};
    int colors[n], seen[n], levels[n], component[n], queue[n];
    int trial_order[n] = {0, 1, 2, 3}, order[n], offsets[n], level_offsets[n + 1];
    Scratch scratch{colors, seen, levels, component, queue, trial_order, order,
                    offsets, level_offsets, n};
    const unsigned char deformable[n] = {1, 1, 1, 0};
    const double movement[n] = {0.5, 0.5, 0.0, 0.0};
    std::uint64_t costs[n] = {0, 0, 0, 0};
    Shape shape{};
    require(color(n, row, adjacency, scratch, shape) == Status::Ok,
        "level helper coloring failed");
    require(shape.colors == 2, "level helper coloring lost the two-color graph");
    std::sort(trial_order, trial_order + n,
        TrialLess{colors, costs, deformable, false});
    require(schedule(n, row, adjacency, deformable, movement, scratch, shape)
            == Status::Ok, "level helper schedule failed");
    require(shape.active == 3, "rigid obstacle was scheduled as an update");
    require(levels[0] == 0 && levels[1] == 0,
        "exact maximum movement tie was not seeded in both neighbors");
    require(levels[2] == 0,
        "zero-movement disconnected component was not seeded");
    require(levels[3] == 0, "rigid-only component did not get level zero");
    bool saw_rigid = false;
    for (int i = 0; i < shape.active; ++i)
        saw_rigid = saw_rigid || order[i] == 3;
    require(!saw_rigid, "rigid obstacle appeared in scheduled update order");
}

struct OriginalRigidFixture {
    RefMesh mesh;
    DeformedState state;
    SimParams params=original_cloth_params();
    OriginalRigidFixture() {
        params.use_parallel=false;
        params.use_basic_experimental=params.use_basic_experimental_v2=params.use_simd=false;
        params.exact_computation_fallback_2=false;
        params.fps=40;params.substeps=1;params.max_global_iters=2;
        params.damping=1;params.gravity=Vec3(0,-.5,0);
        params.node_box_min=params.node_box_max=.5;
        params.theta_box_min=params.theta_box_max=.5;
        params.node_box_update_count=2;
        params.use_ccd_guess=false;params.d_hat=0;
    }
    int body(const std::vector<Vec3>& positions,const Vec3& velocity,
        const Vec3& omega,RigidBodyUpdateMode mode) {
        const int b=create_rigid_body(positions,velocity,Vec4(.8,.2,-.3,.4).normalized(),omega,1,mesh,state,mode);
        const auto& nodes=mesh.rb_nodes[b];
        mesh.tris.insert(mesh.tris.end(),nodes.begin(),nodes.begin()+3);
        mesh.build_deformable_nodes();
        return b;
    }
};

void original_rigid_cpu_step(OriginalRigidFixture& fixture) {
    auto centers=fixture.state.x_coms;
    auto orientation=fixture.state.orientations;
    std::vector<Vec3> omega(fixture.state.omega.size(),Vec3::Zero());
    const auto result=global_gauss_seidel_solver_basic_rb(fixture.mesh,fixture.state,
        fixture.params,centers,orientation,omega);
    require(result.iterations==fixture.params.max_global_iters,"original rigid CPU iteration budget differs");
    update_velocity(fixture.state.v_coms,centers,fixture.state.x_coms,fixture.params.dt());
    fixture.state.x_coms=std::move(centers);
    fixture.state.orientations=std::move(orientation);
    fixture.state.omega=std::move(omega);
    sync_rigid_body_particles(fixture.mesh,fixture.state);
}

void original_rigid_compare(const ipc_gpu::solver::StateSnapshot& gpu,const OriginalRigidFixture& cpu) {
    const auto close=[](double a,double b,double tolerance,const char* label) {
        require(std::isfinite(a)&&std::isfinite(b),std::string(label)+": nonfinite");
        require(std::abs(a-b)<=tolerance,std::string(label)+": original rigid CPU/GPU mismatch");
    };
    require(gpu.positions.size()==cpu.state.deformed_positions.size(),"original rigid proxy count differs");
    require(gpu.com.size()==cpu.state.x_coms.size(),"original rigid body count differs");
    for(std::size_t i=0;i<gpu.positions.size();++i)for(int k=0;k<3;++k) {
        close(gpu.positions[i][k],cpu.state.deformed_positions[i][k],5e-9,"proxy position");
        close(gpu.velocities[i][k],cpu.state.velocities[i][k],5e-7,"proxy velocity");
    }
    for(std::size_t b=0;b<gpu.com.size();++b) {
        for(int k=0;k<3;++k) {
            close(gpu.com[b][k],cpu.state.x_coms[b][k],5e-9,"COM");
            close(gpu.com_velocity[b][k],cpu.state.v_coms[b][k],5e-7,"COM velocity");
            close(gpu.omega[b][k],cpu.state.omega[b][k],5e-7,"angular velocity");
        }
        for(int k=0;k<4;++k)close(gpu.orientation[b][k],cpu.state.orientations[b][k],5e-9,"orientation");
    }
}

ipc_gpu::solver::StateSnapshot original_rigid_parity(OriginalRigidFixture fixture,int steps) {
    using namespace ipc_gpu::solver;
    GpuSolver gpu(describe_scene(fixture.mesh,fixture.state,{},fixture.params));
    const auto targets=describe_pin_targets(fixture.state.deformed_positions.size(),{});
    for(int substep=0;substep<steps;++substep) {
        original_rigid_cpu_step(fixture);
        gpu.advance_substep(describe_step(fixture.params),targets,{});
        original_rigid_compare(gpu.snapshot(),fixture);
        fixture.params.rigid_pin_time+=fixture.params.dt();
    }
    return gpu.snapshot();
}

void check_original_rigid_modes_and_pins() {
    OriginalRigidFixture base;
    const RigidBodyUpdateMode modes[]={RigidBodyUpdateMode::TranslationAndOrientation,
        RigidBodyUpdateMode::TranslationOnly,RigidBodyUpdateMode::OrientationOnly,RigidBodyUpdateMode::None};
    base.params.kpin=8;base.params.rigid_pin_time=.17;
    for(int b=0;b<4;++b) {
        const Vec3 offset(4*b,0,0);
        base.body({offset+Vec3(0,0,0),offset+Vec3(.4,0,0),offset+Vec3(0,.3,.1)},
            Vec3(.1,-.03,.04),Vec3(.2,-.1,.3),modes[b]);
        RigidPin pin;pin.body=b;pin.local_position=base.mesh.ref_positions[b][0];
        pin.initial_target=base.state.deformed_positions[base.mesh.rb_nodes[b][0]]+Vec3(.03,.04,-.02);
        pin.rotation_center=base.state.x_coms[b];pin.angular_velocity_z=.7;
        base.params.rigid_pins.push_back(pin);
    }
    const auto actual=original_rigid_parity(base,2);
    for(int b=0;b<4;++b) {
        if(!updates_rigid_translation(modes[b]))for(int k=0;k<3;++k)
            require(actual.com[b][k]==base.state.x_coms[b][k],"original rigid disabled COM moved");
        if(!updates_rigid_orientation(modes[b]))for(int k=0;k<4;++k)
            require(actual.orientation[b][k]==base.state.orientations[b][k],"original rigid disabled orientation moved");
    }
    auto no_pin=base;no_pin.params.rigid_pins.clear();
    const auto control=original_rigid_parity(no_pin,2);
    double difference=0;
    for(int k=0;k<3;++k)difference=std::max(difference,std::abs(actual.com[0][k]-control.com[0][k]));
    require(difference>1e-8,"original rigid generalized pins had no observable effect");
}

OriginalRigidFixture original_rigid_contact_fixture(bool crossing) {
    OriginalRigidFixture fixture;
    fixture.params.max_global_iters=crossing?1:2;
    fixture.params.gravity=Vec3::Zero();fixture.params.d_hat=.01;
    fixture.params.k_barrier=crossing?0:100;
    fixture.params.friction_coefficient=crossing?0:.2;
    fixture.params.friction_velocity_epsilon=.001;
    fixture.params.node_box_min=fixture.params.node_box_max=.05;
    fixture.body({Vec3(.1,.1,.004),Vec3(.2,.1,.004),Vec3(.1,.2,.004)},
        Vec3(.1,0,crossing?-1.0:-.02),Vec3(0,0,.1),
        crossing?RigidBodyUpdateMode::TranslationOnly:RigidBodyUpdateMode::TranslationAndOrientation);
    fixture.body({Vec3(-1,-1,0),Vec3(1,-1,0),Vec3(0,1,0)},Vec3::Zero(),Vec3::Zero(),RigidBodyUpdateMode::None);
    fixture.body({Vec3(10,0,0),Vec3(10.4,0,0),Vec3(10,.3,.1)},
        Vec3(.1,0,0),Vec3::Zero(),RigidBodyUpdateMode::TranslationOnly);
    return fixture;
}

void check_original_rigid_contacts() {
    for(bool crossing:{false,true}) {
        const auto base=original_rigid_contact_fixture(crossing);
        const auto actual=original_rigid_parity(base,1);
        require(actual.com[2][0]>base.state.x_coms[2][0]+1e-5,
            "original rigid contact handling froze an unrelated body");
        for(int node:base.mesh.rb_nodes[0])
            require(actual.positions[node][2]>0,"original rigid translation crossed the fixed triangle");
        if(crossing) {
            // The zero-stiffness crossing request isolates original linear
            // CCD: 0.004 / 0.025 TOI, then exactly the CPU's 90% factor.
            for(int node:base.mesh.rb_nodes[0])
                require(std::abs(actual.positions[node][2]-.0004)<5e-12,
                    "original rigid solver did not commit the 90 percent TOI step");
        } else {
            auto no_contact=base;no_contact.params.k_barrier=0;no_contact.params.friction_coefficient=0;
            const auto control=original_rigid_parity(no_contact,1);
            require(std::abs(actual.com[0][2]-control.com[0][2])>1e-8,
                "original rigid contact force had no observable effect");
        }
    }
}

}  // namespace

int main() {
    try {
        check_step_mapping();
        check_original_guess_box_padding();
        check_collision_only_certificate();
        check_solver_cooperative_route();
        check_original_predictor_budgets();
        check_predictor_stop_control();
        check_level_helper_schedule();
        check_original_rigid_modes_and_pins();
        check_original_rigid_contacts();
        require(gpu_original_mode_device_check() == 0,
                "device original contact/CCD smoke failed");
        require(gpu_original_integration_check() == 0,
                "original clipper/pipeline integration failed");
        std::cout << "original-mode routing and collision-only certificate PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "original-mode test FAIL: " << error.what() << "\n";
        return 1;
    }
}

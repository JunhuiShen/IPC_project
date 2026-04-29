#include "make_shape.h"
#include "physics.h"
#include "solver.h"
#include "broad_phase.h"
#include "parallel_helper.h"
#include <gtest/gtest.h>
#include <vector>

// Compare one Gauss-Seidel sweep from the serial solver (global_gauss_seidel_solver)
// against one sweep from the parallel solver (global_gauss_seidel_solver_parallel).
// Both start from the same configuration. We allow a small tolerance because the
// parallel solver uses a different commit order (colored batches with fresh Newton
// directions for later colors), so results are close but not bit-identical.

// Serial and parallel GS use different update orders (serial: vertex-by-vertex,
// parallel: Jacobi prediction + colored batch commits), so results differ by
// O(h) where h is the Newton step size. The gap shrinks as the solvers
// converge, so the multi-sweep tolerance is much tighter than the one-sweep
// tolerance. Both are empirical ceilings with ~2x headroom over the measured
// max vertex drift on the build_scene() mesh.
static constexpr double kTolOneSweep   = 2e-3;   // measured ~1.27e-3
static constexpr double kTolMultiSweep = 1e-4;   // measured ~3.49e-5

static void build_scene(RefMesh& ref_mesh, DeformedState& state,
                        std::vector<Pin>& pins, VertexTriangleMap& adj,
                        SimParams& params, std::vector<Vec2>& X) {
    params.fps             = 30.0;
    params.substeps        = 1;
    params.mu              = 10.0;
    params.lambda          = 10.0;
    params.density         = 1.0;
    params.thickness       = 0.1;
    params.kpin            = 1e7;
    params.gravity         = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs         = 1e-6;
    params.d_hat           = 0.01;
    params.use_parallel    = false;

    clear_model(ref_mesh, state, X, pins);
    int nx = 10, ny = 10;
    int base = build_square_mesh(ref_mesh, state, X, nx, ny, 2.0, 2.0, Vec3(0.2, -0.1, 0.3));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    append_pin(pins, base + ny * (nx + 1),      state.deformed_positions);
    append_pin(pins, base + ny * (nx + 1) + nx, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj = build_incident_triangle_map(ref_mesh.tris);
}

// One sweep from fresh state: serial vs parallel should produce similar results.
// They won't be bit-identical because the parallel solver uses Jacobi prediction +
// certified regions + colored commits, while the serial solver processes vertices
// sequentially. Both should reduce energy and produce finite, close results.
TEST(ParallelSerialConsistency, OneSweepSerialVsParallelFromFreshState) {
    RefMesh ref_mesh; VertexTriangleMap adj;
    std::vector<Pin> pins; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    DeformedState state;
    build_scene(ref_mesh, state, pins, adj, params, X);

    const auto color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris),
                                           static_cast<int>(state.deformed_positions.size()));

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs          = 0.0;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    // Serial solver
    std::vector<Vec3> serial_x = state.deformed_positions;
    {
        BroadPhase bp;
        global_gauss_seidel_solver(ref_mesh, adj, pins, sweep_params, serial_x, xhat,
                                   bp, state.velocities, color_groups);
    }

    // Parallel solver
    std::vector<Vec3> parallel_x = state.deformed_positions;
    {
        BroadPhase bp;
        global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, parallel_x, xhat,
                                            bp, state.velocities);
    }

    // Both should produce finite results
    for (int i = 0; i < static_cast<int>(serial_x.size()); ++i) {
        EXPECT_FALSE(std::isnan(serial_x[i].x())) << "serial vertex " << i << " NaN";
        EXPECT_FALSE(std::isnan(parallel_x[i].x())) << "parallel vertex " << i << " NaN";
    }

    // Both should reduce energy compared to initial
    double E_init = compute_incremental_potential_no_barrier(ref_mesh, pins, sweep_params, state.deformed_positions, xhat);
    double E_serial = compute_incremental_potential_no_barrier(ref_mesh, pins, sweep_params, serial_x, xhat);
    double E_parallel = compute_incremental_potential_no_barrier(ref_mesh, pins, sweep_params, parallel_x, xhat);
    EXPECT_LT(E_serial, E_init + 1e-10) << "serial should reduce energy";
    EXPECT_LT(E_parallel, E_init + 1e-10) << "parallel should reduce energy";

    // Results should be close (not identical due to different update orders)
    for (int i = 0; i < static_cast<int>(serial_x.size()); ++i) {
        EXPECT_NEAR(serial_x[i].x(), parallel_x[i].x(), kTolOneSweep) << "vertex " << i << " x";
        EXPECT_NEAR(serial_x[i].y(), parallel_x[i].y(), kTolOneSweep) << "vertex " << i << " y";
        EXPECT_NEAR(serial_x[i].z(), parallel_x[i].z(), kTolOneSweep) << "vertex " << i << " z";
    }
}

// After multiple sweeps, both solvers should converge to similar final states.
TEST(ParallelSerialConsistency, MultiSweepConvergesToSimilarState) {
    RefMesh ref_mesh; VertexTriangleMap adj;
    std::vector<Pin> pins; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    DeformedState state;
    build_scene(ref_mesh, state, pins, adj, params, X);

    const auto color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris),
                                           static_cast<int>(state.deformed_positions.size()));

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 10;
    sweep_params.tol_abs          = 1e-8;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    // Serial
    std::vector<Vec3> serial_x = state.deformed_positions;
    SolverResult serial_result;
    {
        BroadPhase bp;
        serial_result = global_gauss_seidel_solver(ref_mesh, adj, pins, sweep_params, serial_x, xhat,
                                                    bp, state.velocities, color_groups);
    }

    // Parallel
    std::vector<Vec3> parallel_x = state.deformed_positions;
    SolverResult parallel_result;
    {
        BroadPhase bp;
        parallel_result = global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, parallel_x, xhat,
                                                              bp, state.velocities);
    }

    // Both should have converged (residual decreased)
    EXPECT_LT(serial_result.final_residual, serial_result.initial_residual);
    EXPECT_LT(parallel_result.final_residual, parallel_result.initial_residual);

    // Final states should be close
    for (int i = 0; i < static_cast<int>(serial_x.size()); ++i) {
        EXPECT_NEAR(serial_x[i].x(), parallel_x[i].x(), kTolMultiSweep) << "vertex " << i << " x";
        EXPECT_NEAR(serial_x[i].y(), parallel_x[i].y(), kTolMultiSweep) << "vertex " << i << " y";
        EXPECT_NEAR(serial_x[i].z(), parallel_x[i].z(), kTolMultiSweep) << "vertex " << i << " z";
    }
}

// Compare global_gauss_seidel_solver and global_gauss_seidel_solver_parallel
// directly with a forced per-vertex coloring in index order (each vertex in
// its own color: {{0}, {1}, ..., {nv-1}}). The parallel solver normally
// builds a dynamic conflict-graph coloring per iteration; we bypass that by
// passing override_colors so both solvers walk 0, 1, ..., nv-1 in strict
// sequence with a broad-phase refresh after every commit -- i.e. both run
// as true Gauss-Seidel in the same order. Under this forcing the per-vertex
// math is literally identical (same Newton direction, same CCD filter step,
// same commit expression), so the resulting positions agree bit-for-bit.
TEST(ParallelSerialConsistency, OneSweepForcedOrderSerialEqualsParallel) {
    RefMesh ref_mesh; VertexTriangleMap adj;
    std::vector<Pin> pins; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
    DeformedState state;
    build_scene(ref_mesh, state, pins, adj, params, X);

    const int nv = static_cast<int>(state.deformed_positions.size());

    // Serial solver coloring: one big group listing every vertex in index
    // order. The serial solver iterates groups in sequence and vertices
    // within a group in list order, so this makes it walk 0..nv-1.
    std::vector<std::vector<int>> serial_colors(1);
    serial_colors[0].resize(nv);
    for (int i = 0; i < nv; ++i) serial_colors[0][i] = i;

    // Parallel solver coloring: one vertex per color in index order. The
    // parallel solver commits colors sequentially and refreshes the broad
    // phase at the end of each color, so a per-vertex coloring is the
    // per-vertex Gauss-Seidel schedule.
    std::vector<std::vector<int>> parallel_colors(nv);
    for (int i = 0; i < nv; ++i) parallel_colors[i] = {i};

    SimParams sweep_params = params;
    sweep_params.max_global_iters = 1;
    sweep_params.tol_abs          = 0.0;

    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, sweep_params.dt());

    // Serial
    std::vector<Vec3> serial_x = state.deformed_positions;
    {
        BroadPhase bp;
        global_gauss_seidel_solver(ref_mesh, adj, pins, sweep_params, serial_x, xhat,
                                   bp, state.velocities, serial_colors);
    }

    // Parallel with the same sweep order
    std::vector<Vec3> parallel_x = state.deformed_positions;
    {
        BroadPhase bp;
        global_gauss_seidel_solver_parallel(ref_mesh, adj, pins, sweep_params, parallel_x, xhat,
                                            bp, state.velocities, /*residual_history=*/nullptr,
                                            &parallel_colors);
    }

    // Both solvers run the same schedule; allow only tiny floating-point
    // differences from reduction/order noise.
    constexpr double kBitwiseDriftTol = 1e-10;
    for (int i = 0; i < nv; ++i) {
        EXPECT_NEAR(serial_x[i].x(), parallel_x[i].x(), kBitwiseDriftTol) << "vertex " << i << " x";
        EXPECT_NEAR(serial_x[i].y(), parallel_x[i].y(), kBitwiseDriftTol) << "vertex " << i << " y";
        EXPECT_NEAR(serial_x[i].z(), parallel_x[i].z(), kBitwiseDriftTol) << "vertex " << i << " z";
    }
}

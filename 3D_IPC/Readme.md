# 3D IPC — Incremental Potential Contact Simulation

A 3D physics simulation of deformable triangle meshes using
**Incremental Potential Contact (IPC)** and solved with a **nonlinear Gauss–Seidel solver**.

The simulation pipeline consists of:
- **CCD-projected initial guess**: advances toward the inertial predictor using parallel cubic CCD
- **Nonlinear Gauss–Seidel solver**: local 3x3 Newton solve per vertex with linear CCD step filtering
- **Incremental broad phase**: swept-AABB BVH with local refit after each vertex commit
- **Parallel solver**: certified-region conflict graph with greedy coloring for safe concurrent vertex updates

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.10+
- Eigen3
- OpenMP
- GoogleTest

## Build

    cd 3D_IPC
    cmake -B build
    cmake --build build --clean-first

## Run

    ./build/3D_sim

Useful examples:

    ./build/3D_sim --help
    ./build/3D_sim --format obj --outdir frames_obj
    ./build/3D_sim --format usd --outdir frames_usd
    ./build/3D_sim --restart_frame 30 --outdir frames_sim3d

By default, output frames are written to `frames_sim3d/` in Houdini `.geo` format:

    frame_0000.geo
    frame_0001.geo
    ...

If `--format obj` is used, output is `.obj`. If `--format usd`, output is `.usda` text.

Per-frame restart snapshots are also written as `state_NNNN.bin`.

## Tests

The test suite contains GoogleTests covering every layer of the pipeline:

    cmake -B build
    cmake --build build --clean-first
    ctest --test-dir build --output-on-failure

| Test file | Count | What it covers |
|-----------|-------|----------------|
| `ccd_test` | 40 | Linear and cubic CCD, degeneracy chain, stress tests, initial guess |
| `broad_phase_test` | 27 | AABB, BVH, pair generation, CCD candidates, conservativeness, incremental refresh, `query_single_node_ccd` vs brute-force |
| `ipc_math_test` | 28 | `matrix3d_inverse`, `segment_closest_point`, `filter_root`, `add_root`/`SmallRoots`, barycentric coords, serialize/deserialize round-trip, `set_mesh_topology` caching |
| `segment_segment_distance_test` | 17 | All 9 Voronoi regions + parallel + degenerate + symmetry + stress |
| `parallel_helper_test` | 17 | Jacobi predictions, conflict graph, coloring, parallel commits, solver correctness |
| `barrier_energy_test` | 14 | Scalar barrier, NT/SS gradient+Hessian FD convergence, activation boundary, near-parallel stress |
| `corotated_energy_test` | 13 | Energy, rest state, rotation/translation invariance, gradient/Hessian FD convergence, stress |
| `make_shape_test` | 11 | Adjacency maps, graph coloring |
| `node_triangle_distance_test` | 9 | All 7 proximity regions + signed distance + degenerate |
| `total_energy_test` | 7 | Barrier activation, directional derivative, per-vertex gradient/Hessian, slope-2 checks |
| `parallel_serial_consistency_test` | 2 | Serial vs parallel solver agreement |
| `visualization_test` | 2 | Debug OBJ export (no assertions — manual inspection) |
| `simulation_snapshot_test` | 1 | Golden-file regression (5-frame determinism) |
| `restart_test` | 1 | Checkpoint resume matches golden |

To list all discovered tests:

    ctest --test-dir build -N -V

You can also run any executable directly:

    ./build/ccd_test
    ./build/ipc_math_test

## CLI Arguments

Arguments are parsed as `--key value` (boolean flags can also be passed as just `--flag`):

- Time integration: `fps`, `substeps`, `num_frames`
- Physics: `mu`, `lambda`, `density`, `thickness`, `kpin`, `gx`, `gy`, `gz`
- Solver: `max_iters`, `tol_abs`, `step_weight`, `d_hat`, `use_parallel`
- Mesh/scene: `nx`, `ny`, `width`, `height`, `left_x`, `right_x`, `sheet_y`, `left_z`, `right_z`
- Output/restart: `outdir`, `format` (`obj|geo|usd`), `restart_frame`

## Console Output

Per-frame statistics are printed to stdout:

    Frame    1 | initial_residual=... | final_residual=... | global_iters=... | solver_time=... ms
    Frame    2 | initial_residual=... | final_residual=... | global_iters=... | solver_time=... ms
    ...


## Project Structure

    3D_IPC/
    ├── CMakeLists.txt
    ├── IPC_math.h / IPC_math.cpp
    │   type aliases, matrix utilities, SmallRoots, small helper functions
    │
    ├── make_shape.h / make_shape.cpp
    │   mesh construction, build_xhat(), update_velocity()
    │   vertex adjacency map, greedy coloring
    │
    ├── physics.h / physics.cpp
    │   incremental potential (no barrier): energy, per-vertex gradient/Hessian
    │   PinMap for O(1) pin lookup, OpenMP-parallel global residual
    │   serialize/deserialize simulation state
    │
    ├── corotated_energy.h / corotated_energy.cpp
    │   corotated 3x2 energy, per-vertex nodal gradient, and per-vertex nodal Hessian
    │
    ├── node_triangle_distance.h / node_triangle_distance.cpp
    │   node–triangle closest-point distance (7 regions + degenerate)
    │
    ├── segment_segment_distance.h / segment_segment_distance.cpp
    │   segment–segment closest-point distance (9 regions + parallel)
    │
    ├── barrier_energy.h / barrier_energy.cpp
    │   scalar barrier function b(delta; d_hat) and its derivatives
    │   node–triangle barrier: energy + per-DOF gradient/Hessian (with optional pre-computed distance)
    │   segment–segment barrier: energy + per-DOF gradient/Hessian (with optional pre-computed distance)
    │
    ├── ccd.h / ccd.cpp
    │   linear CCD for single-node Gauss–Seidel sweeps
    │   general cubic CCD for multi-vertex motion (initial guess)
    │   full degeneracy handling: cubic → quadratic → linear → coplanar → collinear
    │   stack-allocated SmallRoots for all polynomial solvers
    │
    ├── broad_phase.h / broad_phase.cpp
    │   swept-AABB broad phase with BVH queries
    │   incremental refresh and local ancestor-only BVH refit
    │   cached mesh topology (set_mesh_topology)
    │   lightweight query_single_node_ccd for per-vertex CCD
    │
    ├── solver.h / solver.cpp
    │   CCD-projected initial guess (parallel CCD evaluation)
    │   serial nonlinear Gauss–Seidel solver
    │   parallel Gauss–Seidel solver (certified-region conflict graph + colored commits)
    │
    ├── parallel_helper.h / parallel_helper.cpp
    │   Jacobi prediction, certified regions, conflict graph
    │   parallel commit computation and application
    │
    ├── simulation.h
    │   advance_one_frame(): time stepping driver
    │
    ├── visualization.h / visualization.cpp
    │   export_obj(), export_geo(), export_usd(), export_frame()
    │
    ├── args.h / ipc_args.h
    │   CLI argument parsing
    │
    ├── Tests 
    │   ├── ipc_math_test.cpp — core math helpers, serialize round-trip, topology caching
    │   ├── corotated_energy_test.cpp — FD convergence for elastic energy/gradient/Hessian
    │   ├── node_triangle_distance_test.cpp — all 7 regions + degenerate
    │   ├── segment_segment_distance_test.cpp — all 9 regions + parallel + stress
    │   ├── barrier_energy_test.cpp — FD convergence for NT/SS barrier gradient+Hessian
    │   ├── ccd_test.cpp — linear + cubic CCD, degeneracy chain, stress, initial guess
    │   ├── broad_phase_test.cpp — BVH, pairs, CCD candidates, query_single_node_ccd vs brute-force
    │   ├── make_shape_test.cpp — adjacency maps, greedy coloring
    │   ├── parallel_helper_test.cpp — predictions, conflict graph, commits, solver
    │   ├── total_energy_test.cpp — combined elastic+barrier FD convergence
    │   ├── simulation_snapshot_test.cpp — golden-file regression
    │   ├── restart_test.cpp — checkpoint resume
    │   ├── parallel_serial_consistency_test.cpp — serial vs parallel agreement
    │   └── visualization_test.cpp — debug OBJ export
    │
    └── Readme.md

## Notes

- corotated_energy implements element-level (triangle) physics
- physics provides no-barrier local gradient/Hessian terms and residual helpers
- barrier_energy provides per-pair barrier energy, gradient, and Hessian for both node–triangle and segment–segment primitives
- ccd provides both linear (single-node) and general cubic (multi-vertex) continuous collision detection
- broad_phase maintains barrier candidate pairs incrementally during sweeps
- solver performs nonlinear Gauss–Seidel iterations with CCD step filtering and CCD-projected initial guess
- simulation.h controls time stepping and scene setup

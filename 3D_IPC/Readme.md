# 3D IPC — Incremental Potential Contact Simulation

A 3D physics simulation of deformable triangle meshes using
**Incremental Potential Contact (IPC)** and solved with a **nonlinear Gauss–Seidel solver**.

The simulator is designed for experimenting with different strategies for:
- broad-phase collision candidate detection
- Newton step-size filtering hooks (CCD integration in progress)
- initial guess generation

These components can be swapped to compare different algorithmic variants.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.10+
- Eigen3
- OpenMP
- GoogleTest

## Build

    cd 3D_IPC
    cmake -B build
    cmake --build build  --clean-first

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
    frame_0002.geo
    ...

If `--format obj` is used:

    frame_0000.obj
    frame_0001.obj
    frame_0002.obj
    ...

If `--format usd` is used, files are written as USDA text (`.usda`):

    frame_0000.usda
    frame_0001.usda
    frame_0002.usda
    ...

Per-frame restart snapshots are also written as:

    state_0000.bin
    state_0001.bin
    ...

## Tests

Run the full suite with CTest:

    cmake -B build
    cmake --build build --clean-first
    ctest --test-dir build --output-on-failure

To list discovered tests:

    ctest --test-dir build -N -V

You can still run any executable directly (for example):

    ./build/total_energy_test
    ./build/simulation_snapshot_test

The test suite includes both GoogleTest-based tests and standalone numerical verification executables.

## CLI Arguments

Arguments are parsed as `--key value` (and boolean flags can also be passed as just `--flag`):

- Time integration: `fps`, `substeps`, `num_frames`
- Physics: `mu`, `lambda`, `density`, `thickness`, `kpin`, `gx`, `gy`, `gz`
- Solver: `max_iters`, `tol_abs`, `step_weight`, `d_hat`, `use_parallel`
- Mesh/scene: `nx`, `ny`, `width`, `height`, `left_x`, `right_x`, `sheet_y`, `left_z`, `right_z`
- Output/restart: `outdir`, `format` (`obj|geo|usd`), `restart_frame`

## Console Output

Per-frame statistics are printed to stdout:

    Frame    1 | initial_residual=... | final_residual=... | global_iters=...
    Frame    2 | initial_residual=... | final_residual=... | global_iters=...
    ...

## Project Structure

    3D_IPC/
    ├── CMakeLists.txt
    ├── IPC_math.h / IPC_math.cpp
    │   matrix utilities and small helper functions
    │
    ├── make_shape.h / make_shape.cpp
    │   mesh construction
    │   build_xhat()
    │   update_velocity()
    │
    ├── physics.h / physics.cpp
    │   incremental potential (no barrier): energy, per-vertex gradient, per-vertex Hessian
    │   barrier terms are added
    │
    ├── node_triangle_distance.h / node_triangle_distance.cpp
    │   node–triangle closest-point distance (7 regions + degenerate)
    │
    ├── segment_segment_distance.h / segment_segment_distance.cpp
    │   segment–segment closest-point distance (9 regions + parallel)
    │
    ├── barrier_energy.h / barrier_energy.cpp
    │   scalar barrier function b(delta; d_hat) and its derivatives
    │   node–triangle barrier: energy + per-DOF gradient/Hessian blocks
    │   segment–segment barrier: energy + per-DOF gradient/Hessian blocks
    │
    ├── corotated_energy.h / corotated_energy.cpp
    │   corotated 3x2 energy, per-vertex nodal gradient, and per-vertex nodal Hessian
    │
    ├── broad_phase.h / broad_phase.cpp
    │   swept-AABB broad phase with BVH queries
    │   incremental refresh and local ancestor-only BVH refit
    │
    ├── solver.h / solver.cpp
    │   nonlinear Gauss–Seidel solver 
    │
    ├── visualization.h / visualization.cpp
    │   export_obj()
    │   export_geo()
    │   export_usd()
    │   export_frame()
    │
    ├── simulation.cpp
    │   main simulation driver
    │
    ├── corotated_energy_test.cpp
    │   FD verification for corotated element energy, gradient, Hessian
    │
    ├── node_triangle_distance_test.cpp
    │   distance computation tests for all 7 regions + degenerate
    │
    ├── segment_segment_distance_test.cpp
    │   distance computation tests for all 9 regions + parallel + symmetry
    │
    ├── barrier_energy_test.cpp
    │   FD convergence tests for scalar barrier, node–triangle barrier, and segment–segment barrier 
    │
    ├── total_energy_test.cpp
    │   FD verification for full incremental potential including barrier terms
    │
    └── Readme.md

## Notes

- corotated_energy implements element-level (triangle) physics
- physics provides no-barrier local gradient/Hessian terms and residual helpers
- barrier_energy provides per-pair barrier energy, gradient, and Hessian for both node–triangle and segment–segment primitives
- broad_phase maintains barrier candidate pairs incrementally during sweeps
- solver performs nonlinear Gauss–Seidel iterations
- simulation.cpp controls time stepping and scene setup

## Future Work

- CCD line search
- support larger meshes
- improve visualization pipeline

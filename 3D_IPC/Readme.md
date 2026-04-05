# 3D IPC — Incremental Potential Contact Simulation

A 3D physics simulation of deformable triangle meshes using
**Incremental Potential Contact (IPC)** and solved with a **nonlinear Gauss–Seidel solver**.

The simulator is designed for experimenting with different strategies for:
- broad-phase collision candidate detection
- collision-safe Newton step filtering
- initial guess generation

These components can be swapped to compare different algorithmic variants.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.10+
- Eigen3

## Build

    cd 3D_IPC
    cmake -B build
    cmake --build build  --clean-first

## Run

    ./build/3D_sim

Output frames are written to `frames_sim3d/` as

    frame_0000.obj
    frame_0001.obj
    frame_0002.obj
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
    │   node–triangle barrier: energy, gradient (12-vector), Hessian (12x12)
    │   segment–segment barrier: energy, gradient (12-vector), Hessian (12x12)
    │
    ├── corotated_energy.h / corotated_energy.cpp
    │   corotated 3x2 energy, per-vertex nodal gradient, and per-vertex nodal Hessian
    │
    ├── Corotated32.h / Corotated32.cpp
    │   reference corotated 3x2 formulation used by corotated_energy
    │
    ├── solver.h / solver.cpp
    │   nonlinear Gauss–Seidel solver 
    │
    ├── visualization.h / visualization.cpp
    │   export_obj()
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
- physics assembles global energy, gradient, and Hessian
- barrier_energy provides per-pair barrier energy, gradient, and Hessian for both node–triangle and segment–segment primitives
- solver performs nonlinear Gauss–Seidel iterations
- simulation.cpp controls time stepping and scene setup

## Future Work

- CCD line search
- support larger meshes
- improve visualization pipeline

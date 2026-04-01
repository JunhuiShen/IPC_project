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

    ./build/corotated_energy_test
    ./build/node_triangle_distance_test
    ./build/segment_segment_distance_test
    ./build/barrier_energy_test
    ./build/total_energy_test
    ./build/make_shape_test

The first five tests use central finite differences to verify analytic gradients
(slope 2) and Hessians (slope 2 or ratio ~4 for the slope-2 check).
`make_shape_test` uses GoogleTest to verify mesh construction utilities.

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
    │   segment–segment closest-point distance (9 regions + degenerate)
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
- solver performs nonlinear Gauss–Seidel iterations
- simulation.cpp controls time stepping and scene setup

The corotated elasticity model in `corotated_energy.h/.cpp` is adapted from the
TGSL `Corotated32` formulation. The reference files `Corotated32.h/.cpp` are
included in this project, and `corotated_energy.cpp` wraps that formulation
through a simplified project-level interface.

## Future Work

- AABB broad-phase collision filtering
- wire barrier terms into the Gauss–Seidel solver
- CCD line search
- support larger meshes
- improve visualization pipeline

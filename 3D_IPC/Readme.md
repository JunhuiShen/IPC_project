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
    cmake --build build

## Run

    ./build/3D_sim

Output frames are written to `frames_sim3d/` as

    frame_0000.obj
    frame_0001.obj
    frame_0002.obj
    ...

## Console Output

Per-frame statistics are printed to stdout:

    Frame    1 | initial_residual=... | final_residual=... | global_iters=...
    Frame    2 | initial_residual=... | final_residual=... | global_iters=...
    ...

## Project Structure

    3D_IPC/
    ├── CMakeLists.txt
    ├── IPC_math.h / IPC_math.cpp
    │   matrix utilities
    │
    ├── make_triangle.h / make_triangle.cpp
    │   mesh construction
    │   build_xhat()
    │   update_velocity()
    │
    ├── physics.h / physics.cpp
    │   global energy assembly
    │   gradient and Hessian accumulation
    │
    ├── solver.h / solver.cpp
    │   nonlinear Gauss–Seidel solver
    │
    ├── visualization.h / visualization.cpp
    │   export_obj()
    │   export_frame()
    │
    ├── corotated_energy.h / corotated_energy.cpp
    │   project-level wrapper for triangle constitutive evaluation
    │
    ├── Corotated32.h / Corotated32.cpp
    │   reference corotated 3x2 formulation used by corotated_energy
    │
    ├── simulation.cpp
    │   main simulation driver
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

- support larger meshes
- add collision handling (e.g. barrier potentials, axis-aligned bounding boxes, continuous collision detection, trust-region methods)
- improve visualization pipeline

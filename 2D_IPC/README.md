# 2D IPC — Incremental Potential Contact Simulation

A 2D physics simulation of spring chains with contact handled using
**Incremental Potential Contact (IPC)** and solved with a **nonlinear Gauss–Seidel solver**.

The simulator is designed for experimenting with different strategies for:

- broad-phase collision candidate detection
- collision-safe Newton step filtering
- initial guess generation

These components can be swapped to compare different algorithmic variants.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.16+

## Build

    cd 2D_IPC
    cmake -B build
    cmake --build build

## Run

    ./build/simulation

Output frames are written to `frames_spring_IPC_bvh/` as

    frame_0001.obj
    frame_0002.obj
    ...


## Console Output

Per-frame statistics are printed to stdout:

    Frame    2 | initial_residual=... | final_residual=... | global_iters=...
    Frame    3 | initial_residual=... | final_residual=... | global_iters=...
    ...
    ===== Simulation Summary =====
    max_global_residual = ...
    avg_global_iters = ...
    total runtime = ... seconds

## Example Selection

Simulation scenes are selected in `src/simulation.cpp`.

Example:

    // ExampleType example_type = ExampleType::Example1;
    ExampleType example_type = ExampleType::Example2;

Available example scenes are defined in `src/example.cpp`.

## Strategy Selection

Algorithmic strategies are configured in `src/simulation.cpp`.

Example configuration:

    StepPolicy filtering_step_policy = StepPolicy::CCD;
    initial_guess::Type initial_guess_type = initial_guess::Type::CCD;

The broad-phase collision candidate detector used in the simulation is `BVHBroadPhase`.

## Step Filter Options

| Filter | Description |
|---|---|
| `CCD` | CCD-based collision-safe Newton step filter |
| `TrustRegion` | Distance-based trust-region Newton step filter |

## Initial Guess Options

| Initial Guess | Description |
|---|---|
| `CCD` | CCD-filtered initial guess |
| `TrustRegion` | Trust-region-filtered initial guess |
| `Affine` | Affine-motion-based initial guess |
| `Trivial` | No-motion initial guess |

## Swapping Strategies

Strategies can be changed in `src/simulation.cpp`.

Example:

    auto broad_phase = std::make_unique<BVHBroadPhase>();

    StepPolicy filtering_step_policy = StepPolicy::CCD;
    // StepPolicy filtering_step_policy = StepPolicy::TrustRegion;

    initial_guess::Type initial_guess_type = initial_guess::Type::CCD;
    // initial_guess::Type initial_guess_type = initial_guess::Type::TrustRegion;
    // initial_guess::Type initial_guess_type = initial_guess::Type::Trivial;
    // initial_guess::Type initial_guess_type = initial_guess::Type::Affine;

Available options:

| Role | Options |
|---|---|
| `BroadPhase` | `BVHBroadPhase` |
| `StepFilter` | `CCD`, `TrustRegion` |
| `InitialGuess` | `CCD`, `TrustRegion`, `Affine`, `Trivial` |

After changing a strategy, rebuild the project:

    cmake --build build

## Project Structure

    2D_IPC/
    ├── CMakeLists.txt
    └── src/
        ├── ipc_math.h
        │   Vec2, Vec, Mat2, basic math operations
        │
        ├── physics.h / physics.cpp
        │   spring forces
        │   barrier IPC potential
        │   analytic gradients and Hessians
        │
        ├── chain.h / chain.cpp
        │   Chain structure
        │   make_chain()
        │   build_xhat()
        │   update_velocity()
        │
        ├── visualization.h / visualization.cpp
        │   export_obj()
        │   export_frame()
        │
        ├── solver.h / solver.cpp
        │   nonlinear Gauss–Seidel solver
        │
        ├── example.h / example.cpp
        │   example scene construction
        │
        ├── simulation.cpp
        │   main simulation driver
        │
        ├── broad_phase/
        │   ├── broad_phase.h
        │   └── bvh.h / bvh.cpp
        │       swept AABB + BVH candidate detection
        │
        ├── step_filter/
        │   ├── step_filter.h
        │   ├── ccd.h / ccd.cpp
        │   └── trust_region.h / trust_region.cpp
        │       collision-safe Newton step filters
        │
        └── initial_guess/
            ├── initial_guess.h
            ├── initial_guess.cpp
            │   dispatcher for initial guess strategies
            │
            ├── trivial.h / trivial.cpp
            ├── affine.h / affine.cpp
            ├── ccd.h / ccd.cpp
            └── trust_region.h / trust_region.cpp

## Notes

- `BVHBroadPhase` performs broad-phase AABB candidate detection.
- `CCD` and `TrustRegion` step policies limit the Newton step to maintain collision safety.
- Initial guess strategies provide different warm-starts for the nonlinear solver.
- Output frames are exported as `.obj` files for visualization and debugging.

# 2D IPC — Incremental Potential Contact Simulation

A 2D physics simulation of two spring chains colliding, using barrier-based contact (IPC) with a nonlinear Gauss-Seidel solver.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.16+

## Build

```bash
cd 2D_IPC
cmake -B build
cmake --build build
```

## Run

```bash
./build/simulation
```

Output frames are written to `frames_spring_IPC_bvh/` as `frame_0001.obj`, `frame_0002.obj`, etc.

Per-frame stats are printed to stdout:

```
Frame    2 | initial_residual=... | final_residual=... | iters=...
Frame    3 | initial_residual=... | final_residual=... | iters=...
...
===== Simulation Summary =====
max_residual = ...
avg_iters    = ...
runtime      = ... seconds
```

## Swapping Strategies

All three strategies are selected in `src/simulation.cpp` by changing a single line each:

```cpp
auto broad_phase   = std::make_unique<BVHBroadPhase>();   // collision detection
auto step_filter   = std::make_unique<CCDFilter>();        // Newton step limiter
auto initial_guess = std::make_unique<CCDGuess>();         // warm-start strategy
```

| Role | Options |
|---|---|
| `BroadPhase` | `BVHBroadPhase` |
| `StepFilter` | `CCDFilter`, `TrustRegionFilter` |
| `InitialGuess` | `CCDGuess`, `TrustRegionGuess`, `AffineGuess`, `TrivialGuess` |

After changing a strategy, rebuild with `cmake --build build`.

## Project Structure

```
2D_IPC/
├── CMakeLists.txt
└── src/
    ├── math.h                        # Vec2, Vec, Mat2, inline math ops
    ├── physics.h / .cpp              # spring, barrier, incremental potential
    ├── chain.h / .cpp                # Chain struct, make_chain, build_xhat, update_velocity
    ├── visualization.h / .cpp        # export_obj, export_frame
    ├── solver.h / .cpp               # nonlinear Gauss-Seidel solver
    ├── simulation.cpp                # main() entry point
    ├── broad_phase/
    │   ├── broad_phase.h             # BroadPhase base class
    │   └── bvh.h / .cpp             # BVHBroadPhase — swept AABB + BVH
    ├── step_filter/
    │   ├── step_filter.h             # StepFilter base class
    │   ├── ccd.h / .cpp             # CCDFilter — continuous collision detection
    │   └── trust_region.h / .cpp    # TrustRegionFilter — distance-based limiter
    └── initial_guess/
        ├── initial_guess.h           # InitialGuess base class
        ├── trivial.h / .cpp         # TrivialGuess — xnew = x (no movement)
        ├── affine.h / .cpp          # AffineGuess — rigid-body velocity field
        ├── ccd.h / .cpp             # CCDGuess — CCD-filtered explicit step
        └── trust_region.h / .cpp    # TrustRegionGuess — trust-region explicit step
```

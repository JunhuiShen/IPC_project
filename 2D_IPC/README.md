# 2D IPC — Incremental Potential Contact Simulation

A 2D physics simulation of arbitrary spring-edge networks with contact handled using
**Incremental Potential Contact (IPC)** and solved with a **nonlinear Gauss–Seidel solver**.

The simulator is designed for experimenting with different strategies for:

- broad-phase collision candidate detection
- collision-safe Newton step filtering
- initial guess generation

These components can be swapped to compare different algorithmic variants.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.16+
- Eigen 3.4.0 -- fetched automatically by CMake (requires network on the first configure)
- GoogleTest
- OpenMP (optional; on macOS: `brew install libomp`)

## Notes for coding agents

When working on this project, Claude, Codex, and other coding agents should
first inspect the libraries and local helpers already used by the codebase
before writing new implementations. Start with `CMakeLists.txt` and the
relevant headers to see what Eigen, GoogleTest, OpenMP, and existing project
utilities provide. After configuration, inspect `build/_deps/` when fetched
dependencies are relevant. Prefer maintained library APIs and existing project
helpers over duplicating math, geometry, collision detection, testing, or
build logic.

## Build

    cd 2D_IPC
    cmake -B build
    cmake --build build --clean-first   # clean rebuild
    cmake --build build -j              # faster incremental parallel build

## Test

Run all tests and print details for any failures:

    ctest --test-dir build --output-on-failure

Rerun only the tests that failed in the previous test run:

    ctest --test-dir build --rerun-failed --output-on-failure

## Run

    ./build/simulation --example 1

A fully explicit run of Example 1:

    ./build/simulation --example 1 --substeps 3 \
        --step_policy ccd --initial_guess ccd \
        --format geo --outdir frames_2d

Output frames are written to `frames_2d/` by default:

    frame_0000.geo
    frame_0001.geo
    frame_0002.geo
    ...

Binary restart checkpoints are written beside the frames:

    state_0000.bin
    state_0001.bin
    ...

Resume from a checkpoint with:

    ./build/simulation --example 1 --restart_frame 30 --outdir frames_2d

Export every substep with:

    ./build/simulation --example 1 --write_substeps

The output directory is relative to the current working directory. When the
simulator is launched from `2D_IPC/`, the default location is
`2D_IPC/frames_2d/`.

## Console Output

Per-frame statistics are printed to stdout:

    Vertices: ... | Segments: ...
    Frame    1 | initial_residual=... | final_residual=... | global_iters=... | solver_time=... s
    Frame    2 | initial_residual=... | final_residual=... | global_iters=... | solver_time=... s
    ...
    ===== Simulation Summary =====
    max_global_residual = ...
    avg_global_iters = ...
    total_sim_time = ... seconds
    total_solver_time = ... seconds
    avg_solver_time = ... seconds/frame

## Solver

`main.cpp` handles CLI parsing, scene setup, restart, and frame export.
The per-frame driver is declared in `simulation.h` and implemented in
`simulation.cpp`.

The 2D solver is in `solver.cpp`.

Each frame is split into `substeps` substeps (default `3`). Each substep runs
nonlinear Gauss-Seidel iterations over global nodes:

- sizes each node trust region as
  `clamp(1.2 * previous_displacement, node_box_min, node_box_max)`
- builds blue node trust boxes
- builds each red segment box as the union of its endpoint blue boxes
- builds each green segment box by augmenting its red box by `d_hat`
- registers node-segment contact pairs where a blue box intersects a green box
- builds a mesh/contact conflict graph and colors it greedily
- computes a local 2x2 Newton update for each node
- clamps each update to its blue box and then applies either the CCD or
  distance-based trust-region contact filter
- keeps contact pairs and coloring fixed within one Gauss-Seidel iteration
- rebuilds boxes, contact pairs, and coloring every
  `node_box_update_count` iterations
- reports a mass-normalized residual

## Data Model

The runtime is topology-agnostic:

- `State2D` stores the evolving configuration: current positions, velocities,
  predicted positions, masses, and pin data.
- `RefMesh` stores the fixed reference information: explicit edge endpoint
  pairs, rest lengths, and incident-edge adjacency for each node.
- Elasticity, contact, coloring, CCD, and Newton updates all use global node IDs.

Edges do not need consecutive endpoints. Branches, loops, disconnected
components, and edges such as `(0, 7)` are valid. `Chain` is only a convenience
used by the bundled examples to generate initial geometry; the solver never
receives chain or block information.

`RefMesh` stores reference invariants rather than a separate array of reference
positions. Its rest lengths are computed from the initial `State2D::x`.

## Examples And Strategies

Example 1 contains two pinned chains swinging into each other. Its geometry is
defined in `example.cpp`. Algorithmic choices use CLI flags defined in
`ipc_args.h`.

    ./build/simulation --example 1 --step_policy ccd --initial_guess affine

The broad-phase collision candidate detector used in the simulation is
`BroadPhase`.

## Step Filter Options

| CLI value | Description |
|---|---|
| `ccd` | Linear point-segment CCD step filter |
| `trust_region` | Distance-based trust-region step filter; requires `eta <= 0.5` |

## Initial Guess Options

| CLI value | Description |
|---|---|
| `ccd` | CCD-filtered explicit prediction |
| `affine` | Affine-motion prediction |
| `trivial` | No-motion prediction |

Important CLI options:

| Option | Values/default |
|---|---|
| `example` | currently `1`; default `1` |
| `nodes` | nodes per chain; default `100` |
| `dt` / `substeps` | frame timestep `1/30`; `3` substeps |
| `num_frames` | default `120` |
| `gx` / `gy` | gravity; defaults `0` and `-9.81` |
| `k_spring` / `k_barrier` | defaults `1000` and `100` |
| `density` / `d_hat` | defaults `900` and `0.005` |
| `tol_abs` / `max_substep_iters` | defaults `1e-6` and `500` |
| `eta` | step safety factor; default `0.9`; use at most `0.5` with `trust_region` |
| `step_policy` | `ccd` or `trust_region` |
| `initial_guess` | `ccd`, `affine`, or `trivial` |
| `use_parallel` | color-parallel updates; default `true` |
| `node_box_min` / `node_box_max` | defaults `0.001` and `0.01` |
| `node_box_update_count` | active-set rebuild interval; default `1` |
| `format` / `outdir` | `geo` or `obj`; default directory `frames_2d` |
| `write_substeps` | exports every substep; default `false` |
| `restart_frame` | checkpoint frame; `-1` disables restart |

`d_hat` must be nonnegative and strictly smaller than half the minimum
reference edge length. The executable validates this after constructing the
selected scene and reports the allowed limit.

Run `./build/simulation --help` for the complete generated option list. CLI
strategy changes do not require rebuilding.

## Project Structure

    2D_IPC/
    ├── CMakeLists.txt
    ├── main.cpp
    │   CLI application setup, restart, and output
    ├── simulation.h / simulation.cpp
    │   substep loop and frame advancement
    ├── solver.h / solver.cpp
    ├── physics.h / physics.cpp
    │   local incremental-potential gradient and Hessian
    ├── spring_energy.h / spring_energy.cpp
    ├── barrier_energy.h / barrier_energy.cpp
    │   scalar IPC barrier plus node-segment gradient/Hessian
    ├── node_segment_distance.h / node_segment_distance.cpp
    ├── ogc_trust_region.h / ogc_trust_region.cpp
    ├── ccd.h / ccd.cpp
    ├── broad_phase.h / broad_phase.cpp
    │   AABB/BVH infrastructure, active-set cache, and swept candidate queries
    ├── parallel_helper.h / parallel_helper.cpp
    │   blue/red/green box construction, pair registration, adjacency, and coloring
    ├── state.h / state.cpp
    │   global dynamic state plus predictor and velocity updates
    ├── mesh.h / mesh.cpp
    │   explicit edge topology, rest lengths, and node-edge incidence
    ├── chain.h / chain.cpp
    │   optional chain geometry and assembly helpers for example scenes
    ├── example.h / example.cpp
    ├── restart.h / restart.cpp
    ├── visualization.h / visualization.cpp
    └── initial_guess/
        └── initial_guess.h / .cpp, trivial, affine, ccd

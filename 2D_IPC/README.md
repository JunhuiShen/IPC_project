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

    ./build/simulation

Example 1, with the default CCD step policy and CCD initial guess:

    ./build/simulation --example 1

A fully explicit version of the same run:

    ./build/simulation --example 1 --substeps 3 --step_policy ccd --initial_guess ccd --format geo --outdir frames_2d

Output frames are written to `frames_2d/` by default as

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

From the repository root, the default output folder is:

    2D_IPC/frames_2d/


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

Each frame is split into `substeps` substeps (default `3`). Each substep runs a
nonlinear Gauss-Seidel sweep over global nodes:

- builds a node-box `BVHBroadPhase` active set
- computes a local 2x2 Newton update for each node
- clamps each update with either CCD or trust-region step policy
- periodically rebuilds and recolors the active set according to
  `node_box_update_count`
- reports a mass-normalized residual

## Data Model

The runtime is topology-agnostic:

- `State2D` stores one global position, velocity, mass, and pin array.
- `RefMesh` stores explicit edge endpoint pairs, rest lengths, and incident-edge
  adjacency for each node.
- Elasticity, contact, coloring, CCD, and Newton updates all use global node IDs.

Edges do not need consecutive endpoints. Branches, loops, disconnected
components, and edges such as `(0, 7)` are valid. `Chain` is only a convenience
used by the bundled examples to generate initial geometry; the solver never
receives chain or block information.

## Example And Strategy Selection

Simulation scenes and algorithmic choices are selected with CLI flags defined in
`ipc_args.h`.

    ./build/simulation --example 2 --step_policy ccd --initial_guess affine

Available scenes are defined in `example.cpp`. The broad-phase collision
candidate detector used in the simulation is `BVHBroadPhase`.

## Step Filter Options

| Filter | Description |
|---|---|
| `CCD` | CCD-based collision-safe Newton step filter |
| `TrustRegion` | Distance-based trust-region Newton step filter |

## Initial Guess Options

| Initial Guess | Description |
|---|---|
| `CCD` | CCD-filtered initial guess |
| `Affine` | Affine-motion-based initial guess |
| `Trivial` | No-motion initial guess |
| `TrustRegion` | Accepted as a legacy alias for `Trivial` |

Available CLI options:

| Role | Options |
|---|---|
| `BroadPhase` | `BVHBroadPhase` |
| `substeps` | positive integer, default `3` |
| `max_substep_iters` | max Gauss-Seidel iterations per substep, default `500` |
| `step_policy` | `ccd`, `trust_region` |
| `k_barrier` | IPC barrier stiffness multiplier, default `100` |
| `density` | mass density, default `900` |
| `d_hat` | IPC contact activation distance |
| `num_frames` | number of output frames |
| `use_parallel` | boolean, enables color-parallel basic solver updates |
| `node_box_min` / `node_box_max` | lower/upper half-width clamp for parallel node boxes |
| `node_box_update_count` | Gauss-Seidel iterations between parallel broad-phase/contact recoloring rebuilds |
| `initial_guess` | `ccd`, `affine`, `trivial`, `trust_region` (= `trivial`) |
| `write_substeps` | boolean, writes `substep_XXXX.geo` / `.obj` |
| `restart_frame` | frame index for `state_XXXX.bin`; `-1` disables restart |

After changing a strategy, rebuild the project:

    cmake --build build

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
    ├── parallel_helper.h / parallel_helper.cpp
    │   conflict graph construction and greedy coloring
    ├── state.h / state.cpp
    │   global dynamic state plus predictor and velocity updates
    ├── mesh.h / mesh.cpp
    │   explicit edge topology, rest lengths, and node-edge incidence
    ├── chain.h / chain.cpp
    │   optional chain geometry and assembly helpers for example scenes
    ├── example.h / example.cpp
    ├── restart.h / restart.cpp
    ├── visualization.h / visualization.cpp
    ├── broad_phase/
    │   └── broad_phase.h, bvh.h / bvh.cpp
    └── initial_guess/
        └── initial_guess.h / .cpp, trivial, affine, ccd

## Notes

- `BVHBroadPhase` performs broad-phase AABB candidate detection.
- `CCD` and `TrustRegion` step policies limit the Newton step to maintain collision safety.
- Initial guess strategies provide different warm-starts for the nonlinear solver.
- Rest lengths, explicit edge topology, and per-node edge incidence live in one mesh-wide `RefMesh`.
- Output frames are exported as `.geo` by default; pass `--format obj` for `.obj`.

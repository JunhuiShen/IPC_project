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

## Testing

After configuring and building, run the full 2D test suite from the repository
root or from `2D_IPC/`:

    ctest --test-dir build --output-on-failure

Rerun only the tests that failed in the previous test run:

    ctest --test-dir build --rerun-failed --output-on-failure

Run a focused subset by matching the GoogleTest suite name:

    ctest --test-dir build -R SpringEnergy --output-on-failure
    ctest --test-dir build -R BarrierEnergy --output-on-failure
    ctest --test-dir build -R CCD --output-on-failure
    ctest --test-dir build -R ParallelHelper --output-on-failure

The main test groups are:

| Test group | Coverage |
|---|---|
| `SpringEnergy` | spring energy, local gradient, and local Hessian finite-difference convergence |
| `BarrierEnergy` | scalar/node-segment barrier energy, local gradient, and local Hessian finite-difference convergence |
| `SDFPenalty2D`, `SDFHeaviside2D`, `GroundSDF`, `CircleSDF` | Heaviside-squared SDF penalty with ground and circle signed-distance fields |
| `CCD` | point-segment and rigid-body continuous collision detection cases |
| `BVH` | AABB/BVH construction, query, and safe-step behavior |
| `ParallelHelper` | blue/red/green boxes, contact registration, conflict graph construction, and coloring |

The energy tests use centered finite differences. Energy-vs-gradient and
gradient-vs-Hessian errors should converge with slope in `[1.95, 2.05]` as the
step size is halved.

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

Scenes with visualization-only static collider geometry also write one holder
file beside the frames:

    static_colliders.geo

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
    Frame    1 | initial_residual=... | final_residual=... | final_rb_residual=... | global_iters=... | solver_time=... s
    Frame    2 | initial_residual=... | final_residual=... | final_rb_residual=... | global_iters=... | solver_time=... s
    ...
    ===== Simulation Summary =====
    max_global_residual = ...
    max_rb_residual = ...
    avg_global_iters = ...
    total_sim_time = ... seconds
    total_solver_time = ... seconds
    avg_solver_time = ... seconds/frame

`final_residual` is the mass-normalized nodal residual used by the convergence
check. Rigid-body scenes also report `final_rb_residual`, an unnormalized
reduced residual formed from net COM force and scalar torque (`J^T g`) for
rigid nodes; visualization-only static colliders are not part of either
residual.

## Solver

`simulation.cpp` handles CLI parsing, scene setup, restart, and frame export.
The per-frame driver is declared inline in `simulation.h`.

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

### Rigid-Body Variant

The 2D code also includes a reduced-coordinate rigid-body Gauss-Seidel path for
polygonal rigid bodies. A rigid body stores a center of mass `y`, orientation
`theta`, material-space vertex positions, total mass, and inertia tensor. During
the solve, its world vertices are reconstructed from

    x_i = R(theta) X_i + y 

The central implementation for this layer is `rigid_body_ipc.h/.cpp`: it owns
the 2D rigid transform helpers, material/world-space conversion, rigid-body
creation, inertia tensor construction, and reduced inertial energy derivatives.

Local nodal contact derivatives are pulled back to rigid-body coordinates by
the chain rule. Translation uses `dx_i/dy = I`; rotation uses
`dx_i/dtheta = {-r_i.y, r_i.x}` with `r_i = x_i - y`.

The rigid-body solver mirrors the node-wise parallel path at the contact-set level:

- builds arc-aware blue boxes for each rigid-body node from COM and rotation trust regions
- builds red segment boxes from endpoint blue boxes
- builds green segment boxes by augmenting red boxes by `d_hat`
- registers rigid-rigid node-segment contact pairs through the same BVH query used by the node solver
- colors a rigid-body conflict graph from those contact pairs
- updates same-color rigid bodies in parallel when `use_parallel` is enabled
- updates COM and orientation as separate scalar/vector blocks
- clamps COM and orientation updates to the trust regions that generated the active contact set

Rigid barrier derivatives live in `barrier_energy.h/.cpp` as
`local_barrier_grad_rb` and `local_barrier_hess_rb`. Rigid SDF penalty
derivatives live in `sdf_penalty_energy.h/.cpp` as
`sdf_penalty_gradient_rb` and `sdf_penalty_hessian_rb`. These helpers keep the
reduced-coordinate chain rule close to the corresponding energy term, while
`solver.cpp` is responsible for choosing the Gauss-Seidel update order and step
safety.

## Data Model

The runtime is topology-agnostic:

- `DeformedState` stores the evolving configuration as `std::vector<Vec2>`
  current positions and velocities.
- `RefMesh` stores the fixed reference information: explicit edge endpoint
  pairs, rest lengths, incident-edge adjacency for each node, and lumped mass.
- `Pin` stores soft positional constraints separately from the deformed state.
- Elasticity, contact, coloring, CCD, and Newton updates all use global node IDs.

Edges do not need consecutive endpoints. Branches, loops, disconnected
components, and edges such as `(0, 7)` are valid. `make_shape` helpers are only
conveniences used by the bundled examples to generate initial geometry; the
solver never receives chain or block information.

`RefMesh` stores reference invariants rather than a separate array of reference
positions. Its rest lengths are computed from the initial
`DeformedState::deformed_positions`.

## Examples And Strategies

Example geometry is defined in `example.cpp`. Algorithmic choices use CLI flags
defined in `ipc_args.h`.

Reference commands:

    ./build/simulation --example 1 --substeps 3 \
        --step_policy ccd --initial_guess ccd \
        --outdir frames_2d

    ./build/simulation --example 2 --eps_sdf 0.001 --k_sdf 1e3 \
        --num_frames 500 --max_substep_iters 50000 \
        --tol_abs 1e-10 --substeps 10

    ./build/simulation --example 3 --num_frames 100 \
        --outdir frames_hexagons_collide --k_sdf 1e10 \
        --eps_sdf .001 --max_substep_iters 5000 \
        --substeps 25 --tol_abs 1e-12 --gy 0

    ./build/simulation --example 4 --num_frames 300 \
        --outdir frames_box --max_substep_iters 5000 \
        --substeps 50 --tol_abs 1e-12

| Example | Description |
|---|---|
| `1` | two pinned chains swinging into each other with node-segment IPC contact |
| `2` | spinning/falling rigid hexagon above ground |
| `3` | two rigid hexagons colliding horizontally without gravity |
| `4` | many rigid polygons falling into an open-top SDF box |

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
| `verlet` | CCD-filtered prediction using `x + dt v + dt^2 g` |
| `trivial` | No-motion prediction |

Important CLI options:

| Option | Values/default |
|---|---|
| `example` | `1`, `2`, `3`, or `4`; default `1` |
| `nodes` | nodes per chain; default `100` |
| `dt` / `substeps` | frame timestep `1/30`; `3` substeps |
| `num_frames` | default `120` |
| `gx` / `gy` | gravity; defaults `0` and `-9.81` |
| `k_spring` / `kpin` / `k_barrier` | defaults `1000`, `5e6`, and `100` |
| `k_sdf` / `eps_sdf` | defaults `500` and `0.002`; SDF penalty parameters used by ground/circle SDFs, including Example 2's ground |
| `density` / `thickness` | defaults `900 kg/m^3` and `0.001 m` |
| `d_hat` | default `0.005` |
| `tol_abs` / `max_substep_iters` | defaults `1e-6` and `500` |
| `fixed_iters` | run exactly `max_substep_iters` solver sweeps with no convergence check; default `false` |
| `eta` | step safety factor; default `0.9`; use at most `0.5` with `trust_region` |
| `step_policy` | `ccd` or `trust_region` |
| `initial_guess` | `ccd`, `verlet`, or `trivial`; default `ccd` |
| `use_parallel` | color-parallel updates; default `true` |
| `node_box_min` / `node_box_max` | defaults `0.001` and `0.01` |
| `theta_box_min` / `theta_box_max` | rigid-body angular trust-region half-widths; defaults `0.001` and `0.05` |
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
    ├── simulation.h / simulation.cpp
    │   inline substep loop plus CLI application setup, restart, and output
    ├── solver.h / solver.cpp
    │   initial guesses, local Newton updates, and nonlinear Gauss-Seidel solver
    ├── physics.h / physics.cpp
    │   SimParams2D, DeformedState, RefMesh, Pin, serialization, and local physics terms
    ├── rigid_body_ipc.h / rigid_body_ipc.cpp
    │   2D rigid transforms, material/world-space conversion, rigid-body creation, inertia, and reduced inertial derivatives
    ├── spring_energy.h / spring_energy.cpp
    │   one-edge spring energy and per-node local gradient/Hessian blocks
    ├── barrier_energy.h / barrier_energy.cpp
    │   scalar IPC barrier and node-segment energy, gradient, Hessian, and rigid-body chain-rule blocks
    ├── sdf_penalty_energy.h / sdf_penalty_energy.cpp
    │   generic Heaviside-squared 2D SDF penalty plus ground/circle SDF evaluators and rigid-body chain-rule blocks
    ├── node_segment_distance.h / node_segment_distance.cpp
    ├── ogc_trust_region.h / ogc_trust_region.cpp
    ├── ccd.h / ccd.cpp
    ├── broad_phase.h / broad_phase.cpp
    │   AABB/BVH infrastructure, active-set cache, and swept candidate queries
    ├── parallel_helper.h / parallel_helper.cpp
    │   blue/red/green box construction, pair registration, adjacency, and coloring
    ├── make_shape.h / make_shape.cpp
    │   optional deformable and rigid shape builders for example scenes
    ├── example.h / example.cpp
    └── visualization.h / visualization.cpp

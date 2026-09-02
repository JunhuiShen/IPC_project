# 3D IPC -- Incremental Potential Contact Simulation

A 3D simulator for deformable triangle meshes (cloth / thin shells) and
reduced-coordinate rigid bodies built around **Incremental Potential Contact
(IPC)**. The default deformable and rigid-body solvers use conflict coloring
for parallel updates and broad-phase contact sets for IPC.

## Contents

- [Overview](#overview)
- [Solver algorithms](#solver-algorithms)
- [Getting started](#getting-started)
- [Command reference](#command-reference)
- [Built-in scenes](#built-in-scenes)
- [Runtime behavior](#runtime-behavior)
- [CLI reference](#cli-reference)
- [Source layout](#source-layout)
- [Test coverage](#test-coverage)
- [Development guidance](#development-guidance)
- [Acknowledgments](#acknowledgments)

## Overview

For deformable scenes, each time step minimizes an incremental potential made of:

- **Inertial term** -- implicit Euler predictor against the current velocity field.
- **Gravity** -- constant body-force potential `-m*g*x` set by `gx`, `gy`, `gz`
  (default `(0, -9.81, 0)` m/s^2).
- **Elastic term** -- corotated membrane energy + Grinspun-style discrete-shell
  hinge bending (`kB` controls bending stiffness; `kB = 0` disables it).
- **IPC log-barrier contact** -- node-triangle and segment-segment barriers built
  from a swept-AABB BVH broad phase.
- **SDF penalty contact** -- analytic signed-distance penalties (plane, cylinder, sphere)
  with stiffness `k_sdf` and active range `eps_sdf` (cloth's force-free rest
  is at `phi = eps_sdf`; set 0 for a hard quadratic at the surface).
- **Pin springs** -- soft positional constraints for fixed vertices.

Deformable scenes use one of two Gauss-Seidel solvers, selected by CLI flag:

- **`global_gauss_seidel_solver_basic`** (default) -- builds the broad phase
  every `node_box_update_count` iterations and sweeps every vertex with a local
  3x3 Newton step.
  Each step is clamped by either CCD (`--use_ccd`) or an OGC narrow phase
  (`--use_ogc`). It uses conflict-graph coloring for parallel-by-color
  commits when `--use_parallel` is enabled. It supports either convergence-
  based stopping or a fixed iteration count with `--fixed_iters`.
- **`global_gauss_seidel_solver_ogc`** (`--use_ogc_solver`) -- alternative
  OGC solver that refreshes vertex boxes through partial BVH leaf refits and
  rebuilds contact pairs before each later outer iteration. Padding is
  controlled by `--ogc_box_pad`.
  Requires `--fixed_iters`.

Rigid-body scenes use **`global_gauss_seidel_solver_basic_rb`**, which solves
one three-component COM position and one three-component angular-velocity
vector per body. Its blue/red/green box structure conservatively generates a
contact-only body conflict graph, then updates each color in parallel when
`--use_parallel` is enabled.

Our OGC narrow phase and solver implement the algorithm from Chen et al.
2025; see Acknowledgments.

## Solver algorithms

### Deformable solver

For deformable scenes, each substep runs nonlinear Gauss-Seidel iterations over
the mesh vertices:

- builds a blue trust-region box for each vertex with a heuristic size
- builds green triangle boxes as padded unions of their blue node boxes
- builds red edge boxes as endpoint-box unions and green edge boxes by padding
  the red boxes by `d_hat`
- registers node-triangle contacts from blue-node/green-triangle intersections
  and edge-edge contacts from green-edge/red-edge intersections
- builds a combined elastic/contact conflict graph and colors it greedily
- processes the color groups sequentially; vertices within one color group can
  be updated in parallel
- computes a local Newton update for each vertex
- clips each update to its blue trust-region box and then applies CCD to keep
  the complete motion path intersection-free
- keeps the contact pairs and coloring fixed between rebuilds, potentially
  across multiple Gauss-Seidel iterations
- rebuilds the boxes, contact pairs, conflict graph, and coloring every
  `node_box_update_count` iterations
- evaluates the global residual after each sweep and stops when the requested
  convergence tolerance is reached

In short, the solver repeatedly builds a conservative contact set and
performs collision-safe per-vertex Newton updates one color group at a time.

### Rigid-body solver

Although the rigid-body and deformable solvers share IPC barrier primitives,
green/red broad-phase construction, conflict coloring, and convergence logic,
the rigid-body formulation differs in five main ways:

- **Reduced state.** Surface nodes are not independent unknowns. Each body has
  three COM-position unknowns and three world-space angular-velocity unknowns;
  its surface nodes are reconstructed from fixed body-space offsets.
- **Quaternion orientation.** The four quaternion components are not solved as
  unconstrained coordinates. The candidate unit orientation is
  `q(omega) = exp((dt / 2) * omega) * q_n`, including the full-arc quaternion
  branch encoded by its sign.
- **Reduced Newton updates.** Each body alternates a 3x3 COM solve and a 3x3
  angular-velocity solve. The reduced inertial, IPC barrier, SDF, and friction
  derivatives are assembled directly in those coordinates.
- **Rigid trust regions and CCD.** A node's blue box combines an anchored COM
  translation box with the spherical cap swept by its allowed orientations.
  COM updates use rigid translation CCD; rotation updates are clipped to the
  quaternion cap and then use rigid-rotation CCD.
- **Per-body update labels.** `TranslationAndOrientation` is the default.
  `TranslationOnly`, `OrientationOnly`, and `None` independently disable the
  corresponding generalized-coordinate updates. Disabled bodies remain in
  broad phase and IPC contact, so they act as fixed reaction geometry rather
  than disappearing from collision handling.

Rigid bodies therefore have no per-vertex Newton variables or membrane and
bending energies. Their triangle and edge connectivity is retained only as the
collision surface used by the shared contact pipeline.

Rigid meshes are assumed valid: every triangle has three distinct vertices,
every edge has two distinct endpoints, and every primitive has one uniform
owner.

### General mixed solver

Scenes containing a deformable solid, or mixing deformable nodes with rigid
bodies, use `global_gauss_seidel_solver_basic_general`. It combines the two
specialized formulations as follows:

- **Unified block state.** Blocks are ordered as cloth vertices, solid
  vertices, then rigid bodies. A cloth or solid block owns one 3D position;
  a rigid block owns its reduced COM and orientation variables.
- **Elastic and contact conflicts.** One graph contains cloth/solid elastic
  connectivity and current collision candidates. Each color is completed
  before the next begins, while independent blocks within a color run in
  parallel.
- **Type-specific Newton updates.** Cloth blocks assemble membrane, bending,
  pin, contact, and SDF terms. Solid blocks assemble volumetric corotated and
  contact terms. Rigid blocks use the same reduced COM and angular-velocity
  updates as the rigid-only solver.
- **Shared live configuration.** Every accepted block update is written to the
  same current position state, so later colors see earlier cloth, solid, and
  rigid motion consistently.
- **Mixed broad phase and safe steps.** Contact rebuilding excludes tetrahedral
  interior node queries and impossible rigid self-contact. Deformable updates
  use per-vertex clamping; rigid updates use translation and rotation CCD.
- **Stopping behavior.** The solver either performs the requested fixed number
  of sweeps or evaluates separate cloth, solid, and rigid residual components
  and stops from their combined residual.

This gives mixed scenes one contact-consistent Gauss-Seidel solve instead of
advancing each material type independently.

## Getting started

### Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.21+
- OpenMP (on macOS: `brew install libomp`)
- GoogleTest
- Eigen 3.4.0 -- fetched automatically by CMake (requires network on
  first configure)
- Tight-Inclusion CCD -- fetched automatically by CMake (requires network on
  first configure)

### Build

Configure and compile from the `3D_IPC` directory using the commands in
[Build and test](#build-and-test).

Release builds enable interprocedural optimization when the compiler supports
it, allowing the solver and its energy kernels to be optimized together. Pass
`-DIPC_ENABLE_IPO=OFF` at configure time to disable it.

### First run

After building, use [Basic usage](#basic-usage) to launch the default twisting-
cloth scene or inspect every CLI option.

(`--fixed_iters` is required only by `global_gauss_seidel_solver_ogc`; the
default basic solver can instead use residual-based convergence.)

## Command reference

All copyable project commands are collected here. Run them from the
`3D_IPC` directory.

### Basic usage

```bash
# Default twisting-cloth scene
./build/3D_sim

# Complete argument list and current defaults
./build/3D_sim --help
```

### Build and test

```bash
# First configure
cmake -B build

# Incremental parallel build
cmake --build build -j

# Clean rebuild
cmake --build build --clean-first

# Complete test suite
ctest --test-dir build --output-on-failure

# List discovered tests
ctest --test-dir build -N -V

# Run selected test binaries directly
./build/ccd_test
./build/bending_energy_test
./build/parallel_helper_test
```

### Output and restart

```bash
# Export GEO, OBJ, PLY, or USD frames
./build/3D_sim --format geo --outdir frames_geo
./build/3D_sim --format obj --outdir frames_obj
./build/3D_sim --format ply --outdir frames_ply
./build/3D_sim --format usd --outdir frames_usd

# Resume after frame 30 using state_0030.bin in frames_sim3d
./build/3D_sim --restart_frame 30 --outdir frames_sim3d
```

Output defaults to Houdini `.geo` files in `frames_sim3d/`. Available formats
are `geo`, `obj`, `ply`, and `usd`. Every completed frame also writes a binary
`state_NNNN.bin` restart checkpoint.

### Initial-guess alternatives

```bash
# Translation-restricted initial guess instead of the default CCD guess
./build/3D_sim --use_ccd_guess false --use_translation_guess true --fixed_iters

# CCD-clipped Verlet predictor
./build/3D_sim --use_ccd_guess false --use_verlet_guess true
```

### Reference scene commands

Examples 1–3 use the documented cloth parameters:

```bash
# Example 1: square cloth twisted in place, 240 frames at 0.5 turns/s
./build/3D_sim --example 1 --num_frames 240 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --d_hat 0.005 --k_barrier 100 \
  --fixed_iters --max_substep_iters 10 --substeps 3 --node_box_update_count 10

# Example 2: two cylinders, 2.0 turns, twist then untwist
./build/3D_sim --example 2 --num_frames 900 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 5e6 \
  --d_hat 0.005 --k_barrier 100 --tcyl_max_turn 2.0 \
  --fixed_iters --max_substep_iters 10 --substeps 3 --node_box_update_count 10

# Example 3: one yawing cylinder, 4.0 turns at 0.30 turns/s
./build/3D_sim --example 3 --num_frames 850 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e8 \
  --d_hat 0.005 --k_barrier 100 --k_sdf 1e9 \
  --tu_max_turn 4.0 --tu_twist_rate 0.30 \
  --fixed_iters --max_substep_iters 10 --substeps 5 --node_box_update_count 10

# Example 4: avatar collider and dress loaded from a data directory
./build/3D_sim --example 4 --datadir /path/to/avatar_data
```

Examples 5–11 are rigid-body scenes. These commands mirror the corresponding
scene comments in `example.cpp`:

```bash
# Example 5: freely rotating tennis racket
./build/3D_sim --example 5 --num_frames 500 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir racket_output

# Example 6: freely rotating space tool
./build/3D_sim --example 6 --num_frames 2000 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir space_tool_output

# Example 7: rigid bodies dropped onto a ground plane
./build/3D_sim --example 7 --num_frames 200 --substeps 10 --tol_abs 1e-12 --tol_rel 1e-10 --outdir drop_box_output --format obj

# Example 8: head-on rigid polygon collision
./build/3D_sim --example 8 --num_frames 60 --max_substep_iters 500 --substeps 10 --tol_rel 1e-10 --rigid_density 25 --outdir polygon_collision_output --format obj

# Example 9: stationary stack of twenty rigid polygons
./build/3D_sim --example 9 --num_frames 100 --substeps 10 --d_hat 0.001 --eps_sdf 0.0002 --rigid_density 25 --gy 0 --outdir twenty_polygon_static_stack_output --format obj

# Example 10: five aligned rigid polygons
./build/3D_sim --example 10 --num_frames 100 --substeps 10 --rigid_density 25 --outdir five_polygon_aligned_stack_output --format obj

# Example 11: one hundred rigid polygons, fixed-iteration mode
./build/3D_sim --example 11 --num_frames 200 --substeps 10 --max_substep_iters 20 --fixed_iters --outdir hundred_polygon_box_fixed_iter_output --format obj

# Example 11: residual-convergence alternative
./build/3D_sim --example 11 --num_frames 200 --substeps 80 --max_substep_iters 5000 --outdir hundred_polygon_box_output --format obj
```

Examples 12–21 combine cloth, deformable solids, rigid bodies, and SDFs. These
commands also mirror `example.cpp`:

```bash
# Example 12: fifty rigid polygons dropped onto pinned cloth
./build/3D_sim --example 12 --num_frames 200 --substeps 10 --max_substep_iters 20 --fixed_iters --outdir fifty_polygons_on_pinned_cloth_output --format obj

# Example 13: one deformable solid dropped onto a ground plane
./build/3D_sim --example 13 --num_frames 200 --substeps 20 --max_substep_iters 50 --tol_abs 1e-8 --tol_rel 1e-5 --outdir single_solid_ground_drop_output --format obj

# Example 14: one deformable solid dropped onto pinned cloth
./build/3D_sim --example 14 --num_frames 200 --substeps 20 --max_substep_iters 30 --fixed_iters --E 1e8 --outdir stiff_cloth_solid_drop_output --format obj --d_hat 0.019 --k_barrier 500

# Example 15: rigid and deformable polygons dropped onto pinned cloth
./build/3D_sim --example 15 --num_frames 200 --substeps 20 --max_substep_iters 20 --fixed_iters --outdir twenty_rigid_deformable_polygons_on_pinned_cloth_output --format obj --E 1e8 --d_hat 0.019 --k_barrier 500

# Example 16: alternating rigid and deformable polygons on pinned cloth
./build/3D_sim --example 16 --num_frames 200 --substeps 20 --max_substep_iters 20 --fixed_iters --outdir ten_rigid_solid_flat_stack_on_cloth_output --format obj --E 1e8 --d_hat 0.019 --k_barrier 500

# Example 17: Bunny/Spot solids with rigid cubes and gears
./build/3D_sim --example 17 --datadir example_obj --num_frames 200 --fps 30 --substeps 20 --max_substep_iters 600 --fixed_iters --E 1.25e9 --nu 0.25 --thickness 0.001 --solid_E 1.25e5 --solid_nu 0.25 --d_hat 0.019 --k_barrier 1000 --outdir multi_physics_output --format obj

# Example 18: dynamic threaded bolt falling through a fixed nut
./build/3D_sim --example 18 --num_frames 200 --substeps 20 --max_substep_iters 10 --fixed_iters --outdir bolt_into_fixed_nut_output --format obj

# Example 19: deformable Armadillo between gear crushers
./build/3D_sim --example 19 --num_frames 300 --fps 60 --substeps 10 --max_substep_iters 10 --node_box_update_count 2 --fixed_iters --solid_E 290909 --solid_nu 0.454545 --d_hat 0.00025 --k_barrier 1000 --friction_coefficient 0.1 --friction_velocity_epsilon 0.01 --crusher_angular_speed 20 --outdir armadillo_gear_crusher_output --format obj

# Example 20: four Bunny/Spot/cube/gear rows dropped onto pinned cloth
./build/3D_sim --example 20 --num_frames 200 --substeps 20 --max_substep_iters 200 --fixed_iters --E 1.25e9 --nu 0.25 --thickness 0.001 --solid_E 1.25e5 --solid_nu 0.25 --d_hat 0.019 --k_barrier 1000 --outdir multi_physics_2_output --format obj

# Example 21: rolled cloth unrolling down an SDF ramp
./build/3D_sim --example 21 --num_frames 200 --substeps 20 --max_substep_iters 80 --fixed_iters --kB 0.0025 --friction_coefficient 0.1 --friction_velocity_epsilon 0.01 --outdir rolled_cloth_on_steep_ramp_output_new --format obj
```

## Built-in scenes

Built-in example scenes (`--example N`):

| `--example` | Scene |
|-------------|-------|
| `1` | Square cloth clamped on two edges and twisted (default) |
| `2` | Four closed-loop cloth strips wrapping two horizontal cylinders, twisted then untwisted |
| `3` | Rectangular cloth wrapping one horizontal cylinder; cylinder yaws about +y, twisting the cloth between two clamped top edges, then reverses to untwist |
| `4` | Avatar clothing scene loaded from `datadir` (`body_0000.obj` collider + `dress_0000.obj` simulated cloth) |
| `5` | Freely rotating rigid tennis racket with a prescribed initial angular velocity and no gravity |
| `6` | Freely rotating space tool initialized near its intermediate principal axis |
| `7` | Rigid box and hexagonal prism falling onto a ground plane |
| `8` | Two same-height rigid hexagonal prisms moving toward one another with zero gravity |
| `9` | Twenty rigid hexagonal prisms initialized as a stationary vertical stack on a ground plane |
| `10` | Five differently oriented rigid hexagonal prisms falling onto one another in a vertical column |
| `11` | One hundred mixed triangular, square, pentagonal, heptagonal, and octagonal prisms initialized in five layers and falling into a wide open-top box |
| `12` | Fifty mixed rigid prisms (triangle through dodecagon) falling onto a large rectangular cloth pinned at its four corners |
| `13` | One flat density-900 tetrahedralized deformable octagonal prism falling onto a horizontal ground plane |
| `14` | One flat tetrahedralized deformable octagonal prism falling onto a cloth pinned along two opposite sides |
| `15` | Ten flat small rigid and ten flat larger tetrahedralized deformable polygonal prisms, spanning 3 through 12 sides, falling onto a cloth pinned along two opposite sides |
| `16` | Ten flat polygonal prisms, alternating five rigid bodies and five deformable solids, falling onto one another above a cloth pinned along two opposite sides |
| `17` | Two repeating larger Bunny-solid, larger Spot-solid, smaller rigid-box, and smaller rigid-gear cycles stacked in one vertical column above a cloth pinned along two opposite sides |
| `18` | A fully dynamic threaded bolt starting deeply engaged and falling under gravity through a threaded nut with fixed translation and orientation |
| `19` | One tetrahedral Armadillo starting in the nip between two fixed-center, initially counter-rotating gear crushers |
| `20` | Four close-packed level rows of Bunny-solid, Spot-solid, rigid cube, and rigid gear dropping together onto the same opposite-edge-pinned cloth used by Example 14 |
| `21` | A rolled cloth pinned along its upper edge and unrolling down an analytic SDF incline onto an SDF ground plane |

### External scene assets

All paths below are repository-relative. Examples not listed here are
procedural or use the directory supplied through `--datadir`.

- **Examples 17 and 20:** `example_obj/bunny_coarse/bunny_2000f.1.node` and
  `.ele`, `example_obj/spot/spot_2000f.1.node` and `.ele`, plus
  `example_obj/gear_z18_coarse.obj`.
- **Example 18:** `example_obj/bolt_and_nut/bolt_coarse_bolt.obj` and
  `example_obj/bolt_and_nut/bolt_coarse_nut.obj`.
- **Example 19:** `example_obj/armadillo_coarse/armadillo_5000f.1.node` and
  `.ele`, plus `crusher_coarse_left.obj` and `crusher_coarse_right.obj` under
  `example_obj/crusher/`.

At startup, every scene reports its vertex and triangle counts; scenes with
rigid bodies also report their count. Rigid surface vertices are represented
by up to three COM and three rotation unknowns per body (subject to its update
label), while deformable solids retain their tetrahedral vertex degrees of
freedom.

## Runtime behavior

### Initial guesses

Initial guesses are selected before the nonlinear solver starts each substep.
The default is `ccd_initial_guess`. `--use_verlet_guess true` uses the
CCD-clipped Verlet predictor `xhat + dt^2 gravity`; `--use_translation_guess
true` instead starts from a single global translation `x_i^n + C`, so pass
`--use_ccd_guess false` when using it. This translation guess minimizes the
translation-restricted inertia + gravity + pin-spring objective in closed form,
then applies one cheap 3D Newton correction for SDF penalty contact. Elastic,
bending, and cloth-cloth IPC barrier terms are unchanged by a uniform
translation and therefore do not affect `C`.

### Friction

`friction_coefficient` is the global mesh/SDF Coulomb coefficient; zero
disables friction. `friction_velocity_epsilon` controls smoothing near zero
slip. The implementation uses the lagged PSD model from
[Vertex Block Descent, Section 3.6](https://doi.org/10.1145/3658179). Solver
parameters are not stored in checkpoints, so repeat both friction flags when
restarting a run.

### Output, restart, and timing

Output frames go to `frames_sim3d/` by default in Houdini `.geo` format
(`frame_0000.geo`, `frame_0001.geo`, ...). `--format obj` writes `.obj`;
`--format ply` writes `.ply`; `--format usd` writes `.usda` text. A binary
restart snapshot `state_NNNN.bin` is written alongside every frame. Checkpoints
store particle and reduced rigid-body state.

Per-frame statistics are printed to stdout:

    Frame    1 | initial_residual = X | final_residual = X | global_iters = X | solver_time = X.XXX ms

Residual fields are omitted for fixed-iteration solves. `global_iters` is the
sum across all substeps in that frame. After the run finishes, total / average
solver time and total simulation time are also printed.

Pass `--verbose` to print contact-cache rebuilds and the residual after every
Gauss-Seidel sweep. Mixed solves report cloth, solid, rigid-body, and total
residuals separately. With `--fixed_iters`, residuals are intentionally not
computed, so verbose output reports cache rebuilds only.

## CLI reference

Arguments are parsed as `--key value` (boolean flags can also be passed bare).
See `./build/3D_sim --help` for defaults and full descriptions.

| Group | Flags |
|-------|-------|
| Time integration | `fps`, `substeps`, `num_frames` |
| Physics | Shell: `E`, `nu`, `density`, `thickness`, `kB`; volumetric solid: `solid_E`, `solid_nu`, `solid_density`; rigid body: `rigid_density`; shared: `kpin`, `gx`, `gy`, `gz` |
| Solver core | `max_substep_iters`, `tol_abs`, `tol_rel`, `d_hat`, `k_barrier`, `friction_coefficient`, `friction_velocity_epsilon`, `k_sdf`, `eps_sdf`, `damping`, `fixed_iters`, `use_parallel`, `verbose`, `write_substeps` |
| CCD / step clamping | `use_ccd`, `use_ccd_guess`, `use_verlet_guess`, `use_translation_guess`, `use_ticcd` |
| OGC trust region | `use_ogc` (clip in basic solver), `use_ogc_solver` (per-iteration box/pair refresh solver), `ogc_box_pad` (BVH padding for the refresh; floored to `d_hat`) |
| Node-box sizing | `node_box_min`, `node_box_max` (translation/node-box radius limits in m), `theta_box_min`, `theta_box_max` (rigid orientation-box angular-radius limits in rad), `node_box_update_count` (GS iterations between broad-phase/contact-color rebuilds; default 10) |
| Scene | `example` (`1`..`21`), `sheet_y` + per-example knobs: `twist_rate`, `twist_nx`, `twist_ny`, `twist_size`, `tcyl_n_strips`, `tcyl_strip_w`, `tcyl_strip_span_z`, `tcyl_cloth_h`, `tcyl_nx`, `tcyl_ny`, `tcyl_radius`, `tcyl_length`, `tcyl_nu`, `tcyl_visual_shrink`, `tcyl_twist_rate`, `tcyl_settle_time`, `tcyl_ramp_time`, `tcyl_max_turn`, `tcyl_untwist`, `tcyl_hold_time`, `tu_size`, `tu_width`, `tu_nx`, `tu_ny`, `tu_twist_rate`, `tu_settle_time`, `tu_ramp_time`, `tu_max_turn`, `tu_untwist`, `tu_hold_time`, `tu_cyl_radius`, `tu_cyl_length`, `tu_cyl_nu`, `tu_visual_shrink`, `crusher_angular_speed` |
| Output / restart | `outdir`, `format` (`obj \| geo \| ply \| usd`), `restart_frame`, `datadir` |

Notes:
- Volumetric solids use `solid_E`, `solid_nu`, and `solid_density` independently
  of the shell parameters. The soft rubber-toy defaults are `solid_E = 5e4` Pa,
  `solid_nu = 0.45`, and `solid_density = 900` kg/m³. Increase `solid_E` when
  a stiffer solid is desired.
- Density-based rigid-body scenes use `--rigid_density`, whose default is
  `900` kg/m³. Examples 5 and 6 retain their calibrated total masses because
  their overlapping triangle-proxy geometry does not define a unique volume.
- `restart_frame` is CLI run-control handled in `simulation.cpp` (from `IPCArgs3D`), not a physics/solver runtime parameter.
- `SimParams` holds only runtime solver/physics fields used during substeps.
- The production defaults are `node_box_min/max = 0.001/0.01` m and
  `theta_box_min/max = 0.01/0.1` rad. These values size the allowed boxes; they
  do not force a minimum displacement. Boxes and coloring are rebuilt every 10
  iterations by default.

## Source layout

Source files are grouped here by role, not listed alphabetically, so a new
reader can jump to the layer they care about.

### Program entry

- `simulation.cpp` -- `3D_sim` entry point: parses args, builds a scene from
  `example.cpp`, runs the frame loop, handles restart, prints per-frame stats.
- `simulation.h` -- inline `advance_one_frame()` time-stepping driver; selects
  the substep initial guess before dispatching to the chosen solver.
- `example.h` / `example.cpp` -- built-in scene library selected by `--example`.
- `args.h`, `ipc_args.h` -- generic `--key value` argument parser and the
  `IPCArgs3D` struct that defines every CLI flag and its default.
- `output.h` / `output.cpp` -- frame and diagnostic output, including
  `export_obj`, `export_geo`, `export_ply`, `export_usd`, `export_frame`, broad-
  phase debug geometry, and `write_substep_data`.

### Mesh & physics state

- `make_shape.h` / `make_shape.cpp` -- square, cylinder, sphere, and OBJ mesh
  construction plus rest-shape rebuilding.
- `mesh_utils.h` / `mesh_utils.cpp` -- generic model reset, pin insertion,
  deformed-triangle assembly, and incident-triangle maps.
- `time_integration.h` / `time_integration.cpp` -- inertial target construction
  (`build_xhat`) and post-step velocity updates.
- `state_io.h` / `state_io.cpp` -- binary simulation checkpoint serialization
  and deserialization for particle positions and velocities.
- `physics.h` / `physics.cpp` -- top-level incremental potential. Accumulates
  inertial + elastic + (when `d_hat > 0`) barrier contributions into per-vertex
  gradients and Hessians, exposes `PinMap` for O(1) pin lookup, and runs the
  OpenMP-parallel global residual (mass-normalized by vertex mass).

### Energy terms

- `corotated_energy.h` / `corotated_energy.cpp` -- corotated membrane energy on
  each triangle, per-vertex nodal gradient and Hessian.
- `bending_energy.h` / `bending_energy.cpp` -- Grinspun-style discrete-shell
  hinge bending over adjacent triangle pairs; per-node gradient and PSD
  Gauss-Newton Hessian across all four hinge vertices. Enabled when `kB > 0`
  and enumerated via the `hinge_adj` cache built during `RefMesh` initialization.
- `barrier_energy.h` / `barrier_energy.cpp` -- scalar IPC log barrier
  `b(delta; d_hat)` and its derivatives, plus per-pair energy, gradient, and
  Hessian for node-triangle and segment-segment primitives.
- `sdf_penalty_energy.h` / `sdf_penalty_energy.cpp` -- analytic SDF primitives
  (plane, cylinder, sphere) and a smoothed one-sided SDF penalty with derivatives. 
  Used for static or driven colliders outside the IPC barrier pipeline.

### Geometric primitives

- `IPC_math.h` / `IPC_math.cpp` -- type aliases, 3x3 matrix utilities, and
  shared geometric helpers.
- `node_triangle_distance.h` / `node_triangle_distance.cpp` -- closest-point
  distance covering all 7 Voronoi regions plus degenerate triangles.
- `segment_segment_distance.h` / `segment_segment_distance.cpp` -- closest-point
  distance covering all 9 Voronoi regions plus parallel and degenerate cases.

### Collision detection

- `ccd.h` / `ccd.cpp` -- deformable and rigid CCD entry points:
  - `node_triangle_only_one_node_moves` and `segment_segment_only_one_node_moves`
    take a `bool use_ticcd` flag. When `true` they forward to
    Tight-Inclusion CCD; when `false` they use a **self-written closed-form
    "linear" backend** that is exact in principle when one of the four
    vertices moves over the step. The deformable per-vertex Gauss-Seidel path
    passes `params.use_ticcd` (CLI flag `--use_ticcd`; default `false` in the
    production CLI).
  - `segment_segment_same_displacement_linear_ccd` handles a translating edge
    against a fixed edge and is used by rigid-body COM stepping.
  - `node_triangle_general_ccd` and `segment_segment_general_ccd` are
    TICCD-only entry points used wherever multiple vertices move
    simultaneously (e.g. the CCD-projected initial guess in
    `ccd_initial_guess`).
  - `point_triangle_rb_rotation_ccd` and
    `segment_segment_rb_rotation_ccd` follow a rigid quaternion rotation
    against a fixed primitive. Rigid translation stepping uses the linear
    one-node and translating-edge routes directly; it does not dispatch
    through `params.use_ticcd`.

  **Numerical caveat.** The linear backend reduces each query to a small
  polynomial and falls back to a 2D coplanar test. Coefficient sign tests use
  tolerances scaled to the input magnitudes, and the discriminant clamp drops
  "almost-zero" roots, so near-coplanar / near-tangent configurations can
  produce slightly different TOIs than TICCD's certified interval bisection.
  We treat TICCD as the ground-truth reference; the linear path is offered as
  a faster alternative for the single-moving-DOF case but is **not** as
  numerically robust as TICCD. The coplanar fallback uses a stack-allocated
  `SmallRoots` buffer to avoid heap traffic.
- `broad_phase.h` / `broad_phase.cpp` -- AABB broad phase backed by a per-tree
  BVH. It accepts swept motion or pre-built node boxes, caches mesh topology via
  `set_mesh_topology`, and builds node-triangle and edge-edge candidates. Solver-
  specific initialization modes omit unused refit/incidence storage and prune
  rigid self-contact while preserving deterministic candidate order. The
  refittable mode retains `parent` pointers and per-tree `leaf_to_node` maps so
  `refit_bvh_leaf` and `incremental_refresh_vertex` can perform `O(log N)`
  partial refits for `global_gauss_seidel_solver_ogc`.
- `safe_step.h` / `safe_step.cpp` -- per-vertex node-box clipping, rigid
  quaternion-cap clipping, OGC trust-region bounds, and CCD safe stepping.

### Solver

- `initial_guess.h` / `initial_guess.cpp` -- CCD-projected, Verlet, and
  translation-restricted initial guesses selected by `advance_one_frame()`.
- `solver.h` / `solver.cpp` -- common solver result and organized deformable
  and rigid-body solver implementations:
  - `global_gauss_seidel_solver_basic` (default): broad-phase/contact-color
    data is rebuilt every `node_box_update_count` GS iterations and reused
    between rebuilds. Gauss-Seidel sweeps run via
    `per_vertex_safe_step`, step-clamped by linear/TICCD CCD or
    the OGC narrow phase (`--use_ogc`). With `--use_parallel`, the
    conflict-graph coloring built in `parallel_helper` drives parallel-by-color
    commits.
  - `global_gauss_seidel_solver_ogc` (`--use_ogc_solver`): per-iteration broad-
    phase box/pair refresh with `--ogc_box_pad`-padded node boxes, OGC clip
    unconditionally on, and partial BVH leaf refits via
    `incremental_refresh_vertex`.
  - `global_gauss_seidel_solver_basic_rb`: reduced-coordinate COM/rotation
    updates, rigid barrier and SDF assembly, anchored blue boxes, contact
    coloring, candidate-only barrier/CCD work, and exact solver-requested
    derivative blocks.
  - `global_gauss_seidel_solver_basic_general`: one block ordering for cloth
    vertices, solid vertices, and rigid bodies, with a shared compact elastic/
    contact conflict graph and parallel-by-color updates.

  The two deformable solvers share non-barrier per-vertex gradient/Hessian
  assembly and node-box mechanics, but use live versus frozen barrier
  stencils.
- `parallel_helper.h` / `parallel_helper.cpp` -- helpers for elastic
  adjacency, contact adjacency, rigid-node blue boxes, rigid contact ownership
  filtering, compact lower-neighbor conflict graphs, adjacency union, and
  deterministic greedy coloring.

### Rigid bodies

- `quaternion_math.h` / `quaternion_math.cpp` -- quaternion normalization,
  products, rotations, and derivative helpers.
- `rigid_body_ipc.h` / `rigid_body_ipc.cpp` -- reduced rigid-body creation,
  COM/orientation kinematics, inertial energy, and analytic derivatives.
- The rigid-body Gauss-Seidel solve and IPC barrier assembly live in
  `solver.cpp`; rigid broad-phase/conflict-color construction lives in
  `broad_phase` and `parallel_helper`; `advance_one_frame_rb` and particle
  synchronization live in `simulation.h`.

### Tooling

- `CMakeLists.txt` -- builds the `3D_sim` binary plus every test executable and
  `generate_golden`.
- `generate_golden.cpp` -- standalone utility that rewrites `golden_frames.txt`
  and `frame_50_checkpoint`, which are the fixtures consumed by
  `simulation_snapshot_test` and `restart_test`.

## Test coverage

CTest currently discovers 432 GoogleTest cases across 23 focused binaries. Use
the centralized [build and test commands](#build-and-test) to run or list them.

| Test binary | Cases | What it covers |
|-------------|------:|----------------|
| `barrier_energy_test` | 29 | Scalar and primitive IPC barriers, deformable/rigid derivatives, inactive contact, and validation |
| `bending_energy_test` | 19 | Hinge energy, dihedral angle, finite-difference derivatives, and rigid-motion invariance |
| `broad_phase_test` | 32 | AABBs, BVHs, pair generation/order, solver storage modes, CCD candidates, safe stepping, conservativeness, and partial refits |
| `ccd_test` | 54 | Linear single-moving-DOF CCD, scale/coplanar stress cases, TICCD general NT/SS wrappers, and rigid rotational CCD |
| `corotated_energy_test` | 11 | Membrane rest state, invariance, finite-difference derivatives, and stress cases |
| `friction_energy_test` | 21 | Smoothed Coulomb mesh/SDF contact, prescribed motion, frozen gradients, PSD Hessians, scaling, and validation |
| `initial_guess_test` | 5 | CCD, Verlet, and translation-restricted initial guesses |
| `io_test` | 11 | TetGen input, malformed-input handling, and validated OBJ output |
| `ipc_math_test` | 14 | Matrix inversion, segment closest points, barycentric coordinates, and topology caching |
| `make_shape_test` | 29 | Mesh construction, imported-scene normalization, Examples 19–21, and restart-safe prescribed SDF motion |
| `mesh_test` | 8 | Tetrahedral boundary extraction, TGSL ordering, topology validation, and scale-aware degeneracy checks |
| `node_triangle_distance_test` | 9 | All seven proximity regions, signed distance, and degenerate triangles |
| `parallel_helper_test` | 20 | Compact contact adjacency, rigid ownership/coloring, spherical-cap AABBs, and rigid blue boxes |
| `rigid_body_ipc_test` | 71 | Quaternion and reduced-body derivatives, mesh/SDF contact and friction, update labels, trust boxes, and rigid safe steps |
| `sdf_penalty_energy_test` | 20 | Plane, cylinder, and sphere penalties; rigid derivatives; material poses; and hard/soft limits |
| `segment_segment_distance_test` | 17 | All nine proximity regions, parallel/degenerate cases, symmetry, and stress cases |
| `solid_ipc_test` | 41 | Volumetric solids, mixed-solver integration, mesh/SDF friction, boundary filtering, and fixed-rigid reaction |
| `state_io_test` | 1 | Binary checkpoint round trip |
| `time_integration_test` | 2 | Scalar and large-array position-difference velocity updates |
| `volumetric_corotated_energy_test` | 14 | Tet energy and derivatives, TGSL parity, inverted elements, cache modes, and validation |
| `simulation_snapshot_test` | 1 | Golden-file regression over the 100-frame reference trajectory |
| `restart_test` | 1 | Checkpoint resume against the golden trajectory |
| `output_test` | 2 | Debug OBJ and BVH export |

## Development guidance

Before adding an implementation, check the libraries and local wrappers already
used by the project. Start from `CMakeLists.txt` and the relevant headers to see
what Eigen, Tight-Inclusion CCD, GoogleTest, and OpenMP provide. After CMake
configuration, inspect `build/_deps/` and `CPM_modules/` when relevant. Prefer
an existing project helper or maintained library API over duplicating math,
geometry, collision detection, testing, or build logic.

## Acknowledgments

Our general (multi-vertex motion) continuous collision detection is provided by
[**Tight-Inclusion CCD**](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):

> Bolun Wang, Zachary Ferguson, Teseo Schneider, Xin Jiang, Marco Attene, and
> Daniele Panozzo. *A Large-Scale Benchmark and an Inclusion-Based Algorithm
> for Continuous Collision Detection.* ACM Transactions on Graphics, 2021.

The library is fetched automatically at configure time via CMake's
`FetchContent`. See its repository for license and citation details.

Our friction model follows:

> Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. *Vertex Block Descent.*
> ACM Transactions on Graphics 43(4), Article 116, July 2024.
> [doi:10.1145/3658179](https://doi.org/10.1145/3658179)

Our OGC narrow phase and `global_gauss_seidel_solver_ogc` implement:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel.
> *Offset Geometric Contact.* ACM Transactions on Graphics 44(4):160, 2025.
> [doi:10.1145/3731205](https://doi.org/10.1145/3731205)

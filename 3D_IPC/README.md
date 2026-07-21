# 3D IPC -- Incremental Potential Contact Simulation

A 3D simulator for deformable triangle meshes (cloth / thin shells) built around
**Incremental Potential Contact (IPC)**. Each time step optimizes an incremental
potential with a **parallel-by-color nonlinear Gauss-Seidel solver**, using
continuous collision detection to keep every intermediate state intersection-free.

## What the simulator does

Per time step, the optimizer minimizes an incremental potential made of:

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

The nonlinear solve is driven by one of two Gauss-Seidel solvers, selected by
CLI flag:

- **`global_gauss_seidel_solver_basic`** (default) -- builds the broad phase
  every `node_box_update_count` iterations and sweeps every vertex with a local
  3x3 Newton step.
  Each step is clamped by either CCD (`--use_ccd`) or an OGC narrow phase
  (`--use_ogc`). It uses conflict-graph coloring for parallel-by-color
  commits when `--use_parallel` is enabled. It supports either convergence-
  based stopping or a fixed iteration count with `--fixed_iters`.
- **`global_gauss_seidel_solver_ogc`** (`--use_ogc_solver`) -- alternative
  OGC solver that rebuilds the broad phase per outer iteration and
  partial-refits the BVH after each per-vertex commit. Padding controlled by
  `--ogc_box_pad`.
  Requires `--fixed_iters`.

Our OGC narrow phase and solver implement the algorithm from Chen et al.
2025; see Acknowledgments.

## Solver Algorithm

Each substep runs nonlinear Gauss-Seidel iterations over the mesh vertices:

- builds a blue trust-region box for each vertex with a heuristic size
- builds red primitive boxes from the blue boxes of the vertices belonging to
  each triangle and edge
- builds green primitive boxes by augmenting the red boxes by `d_hat`
- registers vertex-triangle and edge-edge contact pairs from intersections of
  the corresponding boxes
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

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.21+
- OpenMP (on macOS: `brew install libomp`)
- GoogleTest
- Eigen 3.4.0 -- fetched automatically by CMake (requires network on
  first configure)
- Tight-Inclusion CCD -- fetched automatically by CMake (requires network on
  first configure)

## Notes for coding agents

When working on this project, Claude, Codex, and other coding agents should
first check the libraries and local wrappers that are already part of the
codebase before writing new implementations. Start from `CMakeLists.txt` and
the relevant headers to see what Eigen, Tight-Inclusion CCD, GoogleTest, OpenMP,
and any fetched or vendored sources already provide; after configuration, also
inspect `build/_deps/` and `CPM_modules/` when they are relevant. Prefer using
or adapting a maintained library API or an existing project helper over
duplicating math, geometry, collision detection, testing, or build logic, and
only add custom code when the available library path is not suitable.

## Build

    cd IPC_project/3D_IPC
    cmake -B build
    cmake --build build --clean-first   # clean rebuild
    cmake --build build -j              # faster incremental parallel build

Release builds enable interprocedural optimization when the compiler supports
it, allowing the solver and its energy kernels to be optimized together. Pass
`-DIPC_ENABLE_IPO=OFF` at configure time to disable it.

## Run

    ./build/3D_sim --fixed_iters        # default scene (twisting cloth)
    ./build/3D_sim --help               # full argument list

(`--fixed_iters` is required only by `global_gauss_seidel_solver_ogc`; the
default basic solver can instead use residual-based convergence.)

Built-in example scenes (`--example N`):

| `--example` | Scene |
|-------------|-------|
| `1` | Square cloth clamped on two edges and twisted (default) |
| `2` | Four closed-loop cloth strips wrapping two horizontal cylinders, twisted then untwisted |
| `3` | Rectangular cloth wrapping one horizontal cylinder; cylinder yaws about +y, twisting the cloth between two clamped top edges, then reverses to untwist |
| `4` | Avatar clothing scene loaded from `datadir` (`body_0000.obj` collider + `dress_0000.obj` simulated cloth) |
| `5` | Freely rotating rigid tennis racket with configurable initial angular velocity and no gravity |

Common invocations:

    ./build/3D_sim --example 1                              # twisting cloth
    ./build/3D_sim --example 2                              # two-cylinder twist
    ./build/3D_sim --example 3                              # cylinder yaws and twists cloth between two clamped top edges
    ./build/3D_sim --example 5                              # freely rotating rigid tennis racket
    ./build/3D_sim --use_ccd_guess false --use_translation_guess true --fixed_iters
    ./build/3D_sim --format obj --outdir frames_obj         # export .obj frames
    ./build/3D_sim --format usd --outdir frames_usd         # export .usda frames
    ./build/3D_sim --restart_frame 30 --outdir frames_sim3d # resume from checkpoint

Initial guesses are selected before the nonlinear solver starts each substep.
The default is `ccd_initial_guess`; `--use_translation_guess true` instead starts
from a single global translation `x_i^n + C`, so pass
`--use_ccd_guess false` when using it. This translation guess minimizes the
translation-restricted inertia + gravity + pin-spring objective in closed form,
then applies one cheap 3D Newton correction for SDF penalty contact. Elastic,
bending, and cloth-cloth IPC barrier terms are unchanged by a uniform
translation and therefore do not affect `C`.

Reference command for example 1 (square cloth twisted in place, 240 frames at
0.5 turns/s):

    ./build/3D_sim --example 1 --num_frames 240 \
      --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
      --d_hat 0.005 --k_barrier 100 \
      --fixed_iters --max_substep_iters 10 --substeps 5 --node_box_update_count 10

Reference command for example 2 (2.0 turns per cylinder, twist + untwist, 900
frames):

    ./build/3D_sim --example 2 --num_frames 900 \
        --E 115000 --nu 0.25 --kB 0.009 --kpin 5e6 \
        --d_hat 0.005 --k_barrier 100 \
        --tcyl_max_turn 2.0 \
        --fixed_iters --max_substep_iters 10 --node_box_update_count 10

Reference command for example 3 (4.0 turns at 0.30 turns/s, twist + untwist, 850 frames):

    ./build/3D_sim --example 3 --num_frames 850 \
        --E 115000 --nu 0.25 --kB 0.009 --kpin 1e8 \
        --d_hat 0.005 --k_barrier 100 --k_sdf 1e9 \
        --tu_max_turn 4.0 --tu_twist_rate 0.30 \
        --fixed_iters --max_substep_iters 10 --substeps 5 --node_box_update_count 10

Output frames go to `frames_sim3d/` by default in Houdini `.geo` format
(`frame_0000.geo`, `frame_0001.geo`, ...). `--format obj` writes `.obj`;
`--format ply` writes `.ply`; `--format usd` writes `.usda` text. A binary
restart snapshot `state_NNNN.bin` is written alongside every frame.

Per-frame statistics are printed to stdout:

    Frame    1 | global_iters =   X | solver_time = X.XXX ms

After the run finishes, total / average solver time and total simulation time
are also printed.

## CLI reference

Arguments are parsed as `--key value` (boolean flags can also be passed bare).
See `./build/3D_sim --help` for defaults and full descriptions.

| Group | Flags |
|-------|-------|
| Time integration | `fps`, `substeps`, `num_frames` |
| Physics | `E`, `nu`, `density`, `thickness`, `kB`, `kpin`, `gx`, `gy`, `gz` |
| Solver core | `max_substep_iters`, `tol_abs`, `tol_rel`, `d_hat`, `k_barrier`, `k_sdf`, `eps_sdf`, `fixed_iters`, `use_parallel`, `write_substeps` |
| CCD / step clamping | `use_ccd`, `use_ccd_guess`, `use_verlet_guess`, `use_translation_guess`, `use_ticcd` |
| OGC trust region | `use_ogc` (clip in basic solver), `use_ogc_solver` (new per-iter rebuild solver), `ogc_box_pad` (BVH padding for the per-iter rebuild; floored to `d_hat`) |
| Node-box sizing | `node_box_min`, `node_box_max` (basic solver clamp range for `R_vi = clamp(max(prev_disp, \|v_i\| dt) * 1.2, min, max)`), `node_box_update_count` (GS iterations between broad-phase/contact-color rebuilds) |
| Scene | `example` (`1`..`4`), `sheet_y` + per-example knobs: `twist_rate`, `twist_nx`, `twist_ny`, `twist_size`, `tcyl_n_strips`, `tcyl_strip_w`, `tcyl_strip_span_z`, `tcyl_cloth_h`, `tcyl_nx`, `tcyl_ny`, `tcyl_radius`, `tcyl_length`, `tcyl_nu`, `tcyl_visual_shrink`, `tcyl_twist_rate`, `tcyl_settle_time`, `tcyl_ramp_time`, `tcyl_max_turn`, `tcyl_untwist`, `tcyl_hold_time`, `tu_size`, `tu_width`, `tu_nx`, `tu_ny`, `tu_twist_rate`, `tu_settle_time`, `tu_ramp_time`, `tu_max_turn`, `tu_untwist`, `tu_hold_time`, `tu_cyl_radius`, `tu_cyl_length`, `tu_cyl_nu`, `tu_visual_shrink` |
| Output / restart | `outdir`, `format` (`obj \| geo \| ply \| usd`), `restart_frame`, `datadir` |

Notes:
- `restart_frame` is CLI run-control handled in `simulation.cpp` (from `IPCArgs3D`), not a physics/solver runtime parameter.
- `SimParams` holds only runtime solver/physics fields used during substeps.

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
- `visualization.h` / `visualization.cpp` -- `export_obj`, `export_geo`,
  `export_usd`, `export_frame`.

### Mesh & physics state

- `make_shape.h` / `make_shape.cpp` -- mesh construction, `build_xhat()`,
  `update_velocity()`, and incident-triangle maps.
- `physics.h` / `physics.cpp` -- top-level incremental potential. Accumulates
  inertial + elastic + (when `d_hat > 0`) barrier contributions into per-vertex
  gradients and Hessians, exposes `PinMap` for O(1) pin lookup, and runs the
  OpenMP-parallel global residual (mass-normalized by vertex mass).
  Serialize/deserialize of simulation state lives here.

### Energy terms

- `corotated_energy.h` / `corotated_energy.cpp` -- corotated membrane energy on
  each triangle, per-vertex nodal gradient and Hessian.
- `bending_energy.h` / `bending_energy.cpp` -- Grinspun-style discrete-shell
  hinge bending over adjacent triangle pairs; per-node gradient and PSD
  Gauss-Newton Hessian across all four hinge vertices. Enabled when `kB > 0`
  and enumerated via the `hinge_adj` cache built in `physics.cpp`.
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

- `ccd.h` / `ccd.cpp` -- four public CCD entry points behind two backends:
  - `node_triangle_only_one_node_moves` and `segment_segment_only_one_node_moves`
    take a `bool use_ticcd` flag. When `true` they forward to
    Tight-Inclusion CCD; when `false` they use a **self-written closed-form
    "linear" backend** that is exact in principle when one of the four
    vertices moves over the step (the case Gauss-Seidel queries always
    satisfy). The Gauss-Seidel solvers pass `params.use_ticcd` (CLI flag
    `--use_ticcd`; default `false` in the production CLI).
  - `node_triangle_general_ccd` and `segment_segment_general_ccd` are
    TICCD-only entry points used wherever multiple vertices move
    simultaneously (e.g. the CCD-projected initial guess in
    `ccd_initial_guess`).

  **Numerical caveat.** The linear backend reduces each query to a small
  polynomial and falls back to a 2D coplanar test. Coefficient sign tests use
  tolerances scaled to the input magnitudes, and the discriminant clamp drops
  "almost-zero" roots, so near-coplanar / near-tangent configurations can
  produce slightly different TOIs than TICCD's certified interval bisection.
  We treat TICCD as the ground-truth reference; the linear path is offered as
  a faster alternative for the single-moving-DOF case but is **not** as
  numerically robust as TICCD. The coplanar fallback uses a stack-allocated
  `SmallRoots` buffer to avoid heap traffic.
- `broad_phase.h` / `broad_phase.cpp` -- swept-AABB broad phase backed by a
  per-tree BVH. Caches mesh topology via `set_mesh_topology`; builds candidate
  node-triangle and edge-edge pairs from per-vertex AABBs; stores per-vertex
  incident pair lists used by one-node linear CCD and the OGC narrow phase.
  Adds `parent` pointers and per-tree `leaf_to_node` maps so
  `refit_bvh_leaf` and `incremental_refresh_vertex` can do `O(log N)` partial
  refits, used by `global_gauss_seidel_solver_ogc`.
- `ogc_trust_region.h` / `ogc_trust_region.cpp` -- OGC narrow-phase helpers
  (per-pair scaling and the per-vertex `compute_trust_region_bound_for_vertex`).
  See Acknowledgments.

### Solver

- `solver.h` / `solver.cpp` -- solver implementations selected by CLI flag:
  - `global_gauss_seidel_solver_basic` (default): broad-phase/contact-color
    data is rebuilt every `node_box_update_count` GS iterations and reused
    between rebuilds. Gauss-Seidel sweeps run via
    `BroadPhase::per_vertex_safe_step`, step-clamped by linear/TICCD CCD or
    the OGC narrow phase (`--use_ogc`). With `--use_parallel`, the
    conflict-graph coloring built in `parallel_helper` drives parallel-by-color
    commits.
  - `global_gauss_seidel_solver_ogc` (`--use_ogc_solver`): per-iteration broad-
    phase rebuild with `--ogc_box_pad`-padded node boxes, OGC clip
    unconditionally on, partial BVH refit per move via
    `incremental_refresh_vertex`.

  Both share the per-vertex Newton solve (`gs_vertex_delta`) and node-box clip
  mechanics. `ccd_initial_guess`, `translation_initial_guess`, and `update_one_vertex` live here.
- `parallel_helper.h` / `parallel_helper.cpp` -- helpers for elastic
  adjacency, contact adjacency, adjacency union, and deterministic greedy
  coloring.

### Tooling

- `CMakeLists.txt` -- builds the `3D_sim` binary plus every test executable and
  `generate_golden`.
- `generate_golden.cpp` -- standalone utility that rewrites `golden_frames.txt`
  and `frame_50_checkpoint`, which are the fixtures consumed by
  `simulation_snapshot_test` and `restart_test`.

## Tests

Every layer of the pipeline has a GoogleTest binary. To build and run them all:

    cmake -B build
    cmake --build build --clean-first
    ctest --test-dir build --output-on-failure

| Test binary | Cases | What it covers |
|-------------|-------|----------------|
| `ccd_test` | 48 | Linear CCD single-moving-DOF, scale/coplanar stress cases, and TICCD-backed general NT/SS wrappers |
| `rigid_body_ipc_test` | 20 | Quaternion kinematics/derivatives and rigid-body rotational CCD |
| `broad_phase_test` | 25 | AABB, BVH, pair generation/order, CCD candidates, conservativeness, and `incremental_refresh_vertex` partial refit |
| `ipc_math_test` | 15 | `matrix3d_inverse`, `segment_closest_point`, barycentric coordinates, serialize round-trip, and topology caching |
| `sdf_penalty_energy_test` | 15 | Plane / cylinder SDF energy + gradient + Hessian FD convergence, hard-quadratic limit, soft-barrier rest at `phi=eps` |
| `bending_energy_test` | 18 | Hinge energy, dihedral angle, gradient/Hessian FD convergence, rigid-motion invariance |
| `parallel_helper_test` | 2 | Exact contact adjacency and deterministic coloring/scratch reuse |
| `segment_segment_distance_test` | 17 | All 9 Voronoi regions + parallel + degenerate + symmetry + stress |
| `make_shape_test` | 6 | Incident-triangle maps and icosphere construction |
| `barrier_energy_test` | 16 | Scalar barrier, all NT/SS feature regions, force partition, derivative blocks, and stress cases |
| `corotated_energy_test` | 11 | Rest state, invariance, gradient/Hessian FD convergence, and stress cases |
| `total_energy_test` | 5 | Aggregate Hessian/gradient checks, production residual FD, bending wiring, and disabled-SDF no-op behavior |
| `initial_guess_test` | 4 | CCD no-candidate guess and translation guess closed forms for inertia/gravity, pins, and one-step plane-SDF correction |
| `node_triangle_distance_test` | 9 | All 7 proximity regions + signed distance + degenerate |
| `visualization_test` | 2 | Debug OBJ export (no assertions -- manual inspection) |
| `simulation_snapshot_test` | 1 | Golden-file regression (5-frame determinism) |
| `restart_test` | 1 | Checkpoint resume matches golden |

List every discovered test case:

    ctest --test-dir build -N -V

Run any suite directly:

    ./build/ccd_test
    ./build/bending_energy_test
    ./build/parallel_helper_test

## Acknowledgments

Our general (multi-vertex motion) continuous collision detection is provided by
[**Tight-Inclusion CCD**](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):

> Bolun Wang, Zachary Ferguson, Teseo Schneider, Xin Jiang, Marco Attene, and
> Daniele Panozzo. *A Large-Scale Benchmark and an Inclusion-Based Algorithm
> for Continuous Collision Detection.* ACM Transactions on Graphics, 2021.

The library is fetched automatically at configure time via CMake's
`FetchContent`. See its repository for license and citation details.

Our OGC narrow phase and `global_gauss_seidel_solver_ogc` implement:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel.
> *Offset Geometric Contact.* ACM Transactions on Graphics 44(4):160, 2025.
> [doi:10.1145/3731205](https://doi.org/10.1145/3731205)

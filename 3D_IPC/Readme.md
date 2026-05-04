# 3D IPC -- Incremental Potential Contact Simulation

A 3D simulator for deformable triangle meshes (cloth / thin shells) built around
**Incremental Potential Contact (IPC)**. Each time step optimizes an incremental
potential with a **nonlinear Gauss-Seidel solver** (serial or parallel), using
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
- **SDF penalty contact** -- analytic signed-distance penalties (plane, cylinder,
  sphere) with stiffness `k_sdf` and transition layer `eps_sdf`.
- **Pin springs** -- soft positional constraints for fixed vertices.

The nonlinear solve is driven by one of two Gauss-Seidel solvers, selected by
CLI flag:

- **`global_gauss_seidel_solver_basic`** (default) -- builds the broad phase
  once per substep and sweeps every vertex with a local 3x3 Newton step.
  Each step is clamped by either CCD (`--use_ccd`) or an OGC narrow phase
  (`--use_ogc`). Supports parallel-by-color via `--use_parallel`.
- **`global_gauss_seidel_solver_ogc`** (`--use_ogc_solver`) -- serial-only
  sibling that rebuilds the broad phase per outer iteration and partial-refits
  the BVH after each per-vertex commit. Padding controlled by `--ogc_box_pad`.

Our OGC narrow phase and solver implement the algorithm from Chen et al.
2025; see Acknowledgments.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.21+
- OpenMP (on macOS: `brew install libomp`)
- GoogleTest
- Eigen 3.4.0 -- fetched automatically by CMake (requires network on
  first configure)
- Tight-Inclusion CCD -- fetched automatically by CMake (requires network on
  first configure)

## Build

    cd 3D_IPC
    cmake -B build
    cmake --build build --clean-first

## Run

    ./build/3D_sim                      # default scene (cloth stack, high-res)
    ./build/3D_sim --help               # full argument list

Built-in example scenes (`--example N`):

| `--example` | Scene |
|-------------|-------|
| `1` | Two side-by-side pinned sheets |
| `2` | Cloth stack, low-res (ground cloth + 3 small cloths) |
| `3` | Cloth stack, high-res (ground cloth + 5 cloths, default) |
| `4` | Cloth stack dropped over a horizontal cylinder |
| `5` | Square cloth clamped on two edges and twisted |
| `6` | Cloth stack dropped over a sphere (icosphere collider) |

Common invocations:

    ./build/3D_sim --example 1                             # two pinned sheets
    ./build/3D_sim --example 2 --use_parallel              # low-res stack, parallel solver
    ./build/3D_sim --format obj --outdir frames_obj        # export .obj frames
    ./build/3D_sim --format usd --outdir frames_usd        # export .usda frames
    ./build/3D_sim --restart_frame 30 --outdir frames_sim3d # resume from checkpoint

Output frames go to `frames_sim3d/` by default in Houdini `.geo` format
(`frame_0000.geo`, `frame_0001.geo`, ...). `--format obj` writes `.obj`;
`--format ply` writes `.ply`; `--format usd` writes `.usda` text. A binary
restart snapshot `state_NNNN.bin` is written alongside every frame.

Per-frame statistics are printed to stdout:

    Frame    1 | global_iters=... | solver_time=... ms

## CLI reference

Arguments are parsed as `--key value` (boolean flags can also be passed bare).
See `./build/3D_sim --help` for defaults and full descriptions.

| Group | Flags |
|-------|-------|
| Time integration | `fps`, `substeps`, `num_frames` |
| Physics | `E`, `nu`, `density`, `thickness`, `kB`, `kpin`, `gx`, `gy`, `gz` |
| Solver core | `max_substep_iters`, `tol_abs`, `tol_rel`, `d_hat`, `k_barrier`, `k_sdf`, `eps_sdf`, `fixed_iters`, `use_parallel`, `use_gpu`, `experimental`, `write_substeps` |
| CCD / step clamping | `use_ccd`, `use_ccd_guess`, `use_ticcd` |
| OGC trust region | `use_ogc` (clip in basic solver), `use_ogc_solver` (new per-iter rebuild solver), `ogc_box_pad` (BVH padding for the per-iter rebuild; floored to `d_hat`) |
| Node-box sizing | `node_box_min`, `node_box_max` (clamp range for `R_vi = clamp(prev_disp * 1.2, min, max)`) |
| Mesh geometry | `nx`, `ny`, `width`, `height`, `left_x`, `right_x`, `sheet_y`, `left_z`, `right_z` |
| Scene | `example` (`1`..`6`) + per-example knobs: `drop_stack_count`, `drop_cloth_nx`, `drop_cloth_ny`, `drop_first_y`, `drop_spacing`, `drop_cloth_w`, `drop_cloth_h`, `cyl_nu`, `cyl_radius`, `cyl_length`, `cyl_cx/cy/cz`, `cyl_ground_size`, `twist_rate`, `twist_nx`, `twist_ny`, `twist_size`, `sphere_radius`, `sphere_cx/cy/cz`, `sphere_subdiv`, `sphere_cloth_size`, `sphere_ground_size`, `sphere_ground_nx` |
| Output / restart | `outdir`, `format` (`obj \| geo \| ply \| usd`), `restart_frame` |

Notes:
- `restart_frame` is CLI run-control handled in `simulation.cpp` (from `IPCArgs3D`), not a physics/solver runtime parameter.
- `SimParams` holds only runtime solver/physics fields used during substeps.

## Source layout

Source files are grouped here by role, not listed alphabetically, so a new
reader can jump to the layer they care about.

### Program entry

- `simulation.cpp` -- `3D_sim` entry point: parses args, builds a scene from
  `example.cpp`, runs the frame loop, handles restart, prints per-frame stats.
- `simulation.h` -- inline `advance_one_frame()` time-stepping driver (normal and
  twisting paths unified via optional twist-spec/update callback).
- `example.h` / `example.cpp` -- built-in scene library selected by `--example`.
- `args.h`, `ipc_args.h` -- generic `--key value` argument parser and the
  `IPCArgs3D` struct that defines every CLI flag and its default.
- `visualization.h` / `visualization.cpp` -- `export_obj`, `export_geo`,
  `export_usd`, `export_frame`.

### Mesh & physics state

- `make_shape.h` / `make_shape.cpp` -- mesh construction, `build_xhat()`,
  `update_velocity()`, vertex adjacency, greedy graph coloring.
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
  (plane, cylinder, sphere) and the shared Heaviside-ramp penalty with
  derivatives. Used for static colliders outside the IPC barrier pipeline.

### Geometric primitives

- `IPC_math.h` / `IPC_math.cpp` -- type aliases, 3x3 matrix utilities,
  `SmallRoots` stack-allocated polynomial root container, miscellaneous helpers.
- `node_triangle_distance.h` / `node_triangle_distance.cpp` -- closest-point
  distance covering all 7 Voronoi regions plus degenerate triangles.
- `segment_segment_distance.h` / `segment_segment_distance.cpp` -- closest-point
  distance covering all 9 Voronoi regions plus parallel and degenerate cases.

### Collision detection

- `ccd.h` / `ccd.cpp` -- four public CCD entry points behind two backends:
  - `node_triangle_only_one_node_moves` and `segment_segment_only_one_node_moves`
    take a `bool use_ticcd` flag (default `true`). When `true` they forward to
    Tight-Inclusion CCD; when `false` they use a **self-written closed-form
    "linear" backend** that is exact in principle when one of the four
    vertices moves over the step (the case Gauss-Seidel queries always
    satisfy). The Gauss-Seidel solvers pass `params.use_ticcd` (CLI flag
    `--use_ticcd`).
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
  node-triangle and edge-edge pairs from per-vertex AABBs; exposes per-vertex
  pair queries used by one-node linear CCD and the OGC narrow phase. SS pairs
  use the asymmetric red/green convention. Adds `parent` pointers and per-tree
  `leaf_to_node` maps so `refit_bvh_leaf` and `incremental_refresh_vertex` can
  do `O(log N)` partial refits, used by `global_gauss_seidel_solver_ogc`.
- `ogc_trust_region.h` / `ogc_trust_region.cpp` -- OGC narrow-phase helpers
  (per-pair scaling and the per-vertex `compute_trust_region_bound_for_vertex`).
  See Acknowledgments.

### Solver

- `solver.h` / `solver.cpp` -- two solvers selected by CLI flag:
  - `global_gauss_seidel_solver_basic` (`--experimental`, default): substep-
    frozen broad phase, Gauss-Seidel sweeps via `BroadPhase::per_vertex_safe_step`,
    step-clamped by linear/TICCD CCD or the OGC narrow phase (`--use_ogc`).
    With `--use_parallel`, the conflict-graph coloring built in
    `parallel_helper` drives parallel-by-color commits.
  - `global_gauss_seidel_solver_ogc` (`--use_ogc_solver`): per-iteration broad-
    phase rebuild with `--ogc_box_pad`-padded node boxes, serial sweep with
    OGC clip unconditionally on, partial BVH refit per move via
    `incremental_refresh_vertex`.

  Both share the per-vertex Newton solve (`gs_vertex_delta`) and node-box clip
  mechanics. `ccd_initial_guess` and `update_one_vertex` (single-vertex Newton +
  CCD helper) live here.
- `parallel_helper.h` / `parallel_helper.cpp` -- Jacobi delta prediction,
  conflict-graph construction, greedy coloring, and parallel commit apply for
  the basic solver under `--use_parallel`.

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
| `ccd_test` | 17 | Linear CCD single-moving-DOF cases plus TICCD-backed general NT/SS wrapper smoke tests |
| `broad_phase_test` | 30 | AABB, BVH, pair generation, CCD candidates, conservativeness, per-vertex pair query vs brute-force, asymmetric red/green SS convention, `incremental_refresh_vertex` partial refit |
| `ipc_math_test` | 27 | `matrix3d_inverse`, `segment_closest_point`, `filter_root`, `SmallRoots`, barycentric coords, serialize round-trip, topology caching |
| `sdf_penalty_energy_test` | 21 | Plane / cylinder / sphere SDF energy + gradient + Hessian FD convergence, ramp boundary |
| `bending_energy_test` | 20 | Hinge energy, dihedral angle, gradient/Hessian FD convergence, rigid-motion invariance |
| `parallel_helper_test` | 20 | Jacobi predictions, conflict graph, coloring, parallel commits, solver correctness |
| `segment_segment_distance_test` | 17 | All 9 Voronoi regions + parallel + degenerate + symmetry + stress |
| `make_shape_test` | 15 | Adjacency maps, greedy coloring |
| `barrier_energy_test` | 14 | Scalar barrier, NT/SS gradient/Hessian FD convergence, activation boundary, near-parallel stress |
| `corotated_energy_test` | 13 | Energy, rest state, rotation/translation invariance, gradient/Hessian FD convergence, stress |
| `total_energy_test` | 12 | Combined elastic + barrier FD convergence, barrier activation, per-vertex gradient/Hessian, slope-2 checks |
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

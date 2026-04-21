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

The nonlinear solve is driven by:

1. **CCD-projected initial guess** -- advance toward the inertial predictor using
   parallel cubic CCD so the starting iterate is already intersection-free.
2. **Gauss-Seidel iterations** -- local 3x3 Newton solve per vertex, each step
   filtered by linear CCD and accepted only if the global residual drops.
3. **Incremental broad phase** -- swept-AABB BVH with local ancestor-only refit
   after each accepted vertex update.
4. **Parallel mode** (`--use_parallel`) -- certified-region conflict graph with
   greedy coloring so independent vertices can be committed concurrently.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.10+
- Eigen3
- OpenMP (on macOS: `brew install libomp`)
- GoogleTest

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
`--format usd` writes `.usda` text. A binary restart snapshot `state_NNNN.bin`
is written alongside every frame.

Per-frame statistics are printed to stdout:

    Frame    1 | initial_residual=... | final_residual=... | global_iters=... | solver_time=... ms

## CLI reference

Arguments are parsed as `--key value` (boolean flags can also be passed bare).
See `./build/3D_sim --help` for defaults and full descriptions.

| Group | Flags |
|-------|-------|
| Time integration | `fps`, `substeps`, `num_frames` |
| Physics | `mu`, `lambda`, `density`, `thickness`, `kB`, `kpin`, `gx`, `gy`, `gz` |
| Solver | `max_substep_iters`, `tol_abs`, `tol_rel`, `step_weight`, `d_hat`, `k_sdf`, `eps_sdf`, `use_parallel` |
| Mesh geometry | `nx`, `ny`, `width`, `height`, `left_x`, `right_x`, `sheet_y`, `left_z`, `right_z` |
| Scene | `example` (`1`..`6`), plus per-example knobs: `drop_stack_count`, `drop_cloth_nx`, `drop_cloth_ny`, `drop_first_y`, `drop_spacing`, `twist_rate`, `twist_nx`, `twist_ny`, `twist_size`, `sphere_radius`, `sphere_cx`, `sphere_cy`, `sphere_cz`, `sphere_subdiv`, `sphere_cloth_size`, `sphere_ground_size` |
| Output / restart | `outdir`, `format` (`obj \| geo \| usd`), `restart_frame` |

## Source layout

Source files are grouped here by role, not listed alphabetically, so a new
reader can jump to the layer they care about.

### Program entry

- `simulation.cpp` -- `3D_sim` entry point: parses args, builds a scene from
  `example.cpp`, runs the frame loop, handles restart, prints per-frame stats.
- `simulation.h` -- inline `advance_one_frame()` time-stepping driver.
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
  OpenMP-parallel global residual. Serialize/deserialize of simulation state
  lives here.

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

- `ccd.h` / `ccd.cpp` -- linear node-triangle CCD for every single-moving-DOF
  configuration, linear segment-segment CCD for single-moving-endpoint sweeps,
  and general cubic CCD for multi-vertex motion (used by the initial guess).
  Full degeneracy chain: cubic -> quadratic -> linear -> coplanar -> collinear.
  All root solvers use the stack-allocated `SmallRoots` buffer.
- `broad_phase.h` / `broad_phase.cpp` -- swept-AABB broad phase backed by a BVH.
  Caches mesh topology via `set_mesh_topology`, supports incremental refresh
  and local ancestor-only BVH refit, and exposes `query_single_node_ccd` for
  per-vertex CCD which enumerates both node-triangle roles a moving vertex
  can play (lone node vs. triangle corner) as well as edge-edge pairs
  incident to it.

### Solver

- `solver.h` / `solver.cpp` -- CCD-projected initial guess, serial nonlinear
  Gauss-Seidel solver, and the parallel Gauss-Seidel solver with
  certified-region conflict graph and colored commits.
- `parallel_helper.h` / `parallel_helper.cpp` -- Jacobi prediction, certified
  region computation, conflict graph construction, and parallel commit apply.

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
| `ccd_test` | 47 | Linear CCD (all single-moving-DOF cases) and cubic CCD, degeneracy chain, stress, CCD-projected initial guess |
| `broad_phase_test` | 38 | AABB, BVH, pair generation, CCD candidates, conservativeness, incremental refresh, `query_single_node_ccd` vs brute-force |
| `ipc_math_test` | 27 | `matrix3d_inverse`, `segment_closest_point`, `filter_root`, `SmallRoots`, barycentric coords, serialize round-trip, topology caching |
| `parallel_helper_test` | 26 | Jacobi predictions, certified regions, conflict graph, coloring, parallel commits, solver correctness |
| `bending_energy_test` | 20 | Hinge energy, dihedral angle, gradient/Hessian FD convergence, rigid-motion invariance |
| `segment_segment_distance_test` | 17 | All 9 Voronoi regions + parallel + degenerate + symmetry + stress |
| `barrier_energy_test` | 14 | Scalar barrier, NT/SS gradient/Hessian FD convergence, activation boundary, near-parallel stress |
| `corotated_energy_test` | 13 | Energy, rest state, rotation/translation invariance, gradient/Hessian FD convergence, stress |
| `make_shape_test` | 11 | Adjacency maps, greedy coloring |
| `total_energy_test` | 10 | Combined elastic + barrier FD convergence, barrier activation, per-vertex gradient/Hessian, slope-2 checks |
| `node_triangle_distance_test` | 9 | All 7 proximity regions + signed distance + degenerate |
| `parallel_serial_consistency_test` | 3 | Serial vs parallel solver agreement |
| `visualization_test` | 2 | Debug OBJ export (no assertions -- manual inspection) |
| `simulation_snapshot_test` | 1 | Golden-file regression (5-frame determinism) |
| `restart_test` | 1 | Checkpoint resume matches golden |

List every discovered test case:

    ctest --test-dir build -N -V

Run any suite directly:

    ./build/ccd_test
    ./build/bending_energy_test
    ./build/parallel_helper_test

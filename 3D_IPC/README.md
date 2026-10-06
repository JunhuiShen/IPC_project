# 3D IPC — Incremental Potential Contact Simulation

A C++17 command-line simulator for cloth, deformable solids, rigid bodies, and
scenes that combine them. It uses Incremental Potential Contact (IPC) for mesh
collisions, with analytic signed-distance fields (SDFs) for prescribed obstacles.
The simulator exports mesh sequences and restart checkpoints for viewing and
further processing; it does not open a viewer.

## Contents

- [Getting started](#getting-started)
- [Built-in scenes](#built-in-scenes)
- [Common settings](#common-settings)
- [Output and restart](#output-and-restart)
- [Reference scene commands](#reference-scene-commands)
- [Build and test](#build-and-test)
- [Troubleshooting](#troubleshooting)
- [Source guide](#source-guide)
- [Acknowledgments](#acknowledgments)

## Getting started

### Requirements

- A C++17 compiler and CMake 3.21+.
- OpenMP. On macOS with AppleClang, CMake looks for Homebrew `libomp`.
- Boost 1.70+ with its CMake package configuration.
- Git and network access for the first configuration. CMake fetches Eigen
  3.4.0 and Tight-Inclusion CCD 1.0.6, plus their configured dependencies.
- GoogleTest when building tests. It is optional for the simulator-only build
  below.

### Build and run a small scene

From the repository root:

```sh
cd 3D_IPC
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DIPC_BUILD_TOOLS=OFF
cmake --build build --target 3D_sim -j 4
```

**All remaining commands run from `3D_IPC/`.** Start with a 100-vertex cloth:

```sh
OMP_NUM_THREADS=4 ./build/3D_sim \
  --example 1 --twist_nx 9 --twist_ny 9 \
  --fps 30 --num_frames 30 --substeps 3 \
  --max_substep_iters 20 --fixed_iters \
  --use_basic_experimental true --use_simd true \
  --format obj --outdir results/quickstart
```

This simulates one second and writes `frame_0000.obj` through `frame_0030.obj`,
plus matching `state_NNNN.bin` checkpoints. Frame 0 is the initial state. Open
an OBJ in a mesh viewer, or load the numbered files as a sequence in your
visualization tool. The reference commands below provide longer, larger runs.

A fresh run replaces the existing `--outdir` folder. Use a dedicated results
folder and a different path for each run you want to keep.

To see every option and its default:

```sh
./build/3D_sim --help
```

### Build options

Release builds enable interprocedural optimization when supported; use
`-DIPC_ENABLE_IPO=OFF` to disable it. On x86 GCC/Clang builds,
`IPC_ENABLE_NATIVE_ARCH=ON` enables the build machine's CPU instructions.
Use `-DIPC_ENABLE_NATIVE_ARCH=OFF` when building for another CPU. SIMD v2
uses the available backend or a scalar fallback; OpenMP controls parallel
threads independently.

## Built-in scenes

Select a scene with `--example N`; IDs run consecutively from 1 to 14.
Categories describe the simulated material models: **cloth** uses shell
meshes, **rigid body** uses rigid motion, and **solid** uses deformable
tetrahedral meshes. **Mixed** scenes below contain cloth, rigid bodies, and
solids together. Prescribed/SDF obstacles do not change a cloth-only category.
There is no solid-only scene among the categorized examples.

| ID | Category | Scene |
|---|---|---|
| 1 | Cloth only | Square cloth clamped on two edges and twisted; the default scene. |
| 2 | Cloth only | Four cloth strips wrapped around two cylinders that twist and untwist. |
| 3 | Cloth only | Cloth wrapped around one cylinder that twists and untwists. |
| 4 | — | Avatar collider and dress loaded from an external data directory. |
| 5 | — | Freely rotating rigid tennis racket without gravity. |
| 6 | — | Freely rotating space tool demonstrating the Dzhanibekov effect. |
| 7 | — | Ten rigid hexagonal prisms stacked above a ground plane. |
| 8 | Mixed | Two Bunny/Spot solid, rigid cube, and rigid gear cycles stacked above pinned cloth. |
| 9 | Rigid body only | A dynamic threaded bolt falling through a nut with fixed position and orientation. |
| 10 | Mixed | Four horizontal layers of side-lying Bunny/Spot solids, rigid cubes, and rigid gears dropping onto pinned cloth. |
| 11 | Cloth only | A pinned cloth roll unrolling down an SDF ramp onto the ground. |
| 12 | Cloth only | Three cloth layers with one fixed edge and one oscillating edge. |
| 13 | Rigid body only | A linked wrecking ball swinging into a wall of 560 rigid cubes. |
| 14 | Cloth only | Fifty free cloth sheets falling over an off-center fixed cylinder toward the ground. |

### Scene assets

Examples 8–10 and 13 use the supplied `example_obj/` assets. Paths below are
relative to **`3D_IPC/`**, so keep that working directory when running them.

| Examples | Required files |
|---|---|
| 8, 10 | `example_obj/bunny_coarse/bunny_2000f.1.node` and `.ele`; `example_obj/spot/spot_2000f.1.node` and `.ele`; `example_obj/gear_z18_coarse.obj`. |
| 9 | `example_obj/bolt_and_nut/bolt_coarse_bolt.obj` and `bolt_coarse_nut.obj`. |
| 13 | `example_obj/wrecking_ball/link.obj` and `ball.obj`; `--datadir` can point to `example_obj` or its `wrecking_ball` folder. |

Example 4 needs a directory supplied with `--datadir`, containing
`body_0000.obj` and `dress_0000.obj`. Other scenes are generated procedurally.
Resolution flags are scene-specific: `twist_nx/ny` affect Example 1,
`osc_nx/nz` affect Example 12, and `drop_stack_count` plus `drop_cloth_nx/ny`
affect Example 14. Its default 50 sheets contain 245,000 cloth vertices.

## Common settings

Arguments use `--key value`. Boolean flags accept `true`/`false` or `1`/`0`;
a bare boolean flag enables it. Scenes can set their own presets, so use the
reference commands as the starting point for a particular example.
Scenes use meters and seconds, Young's moduli in pascals, densities in kg/m³,
and +y as the up direction.

| Setting | Meaning |
|---|---|
| `--num_frames N --fps F` | Simulated duration is `N/F` seconds; `fps` is output frequency, not measured solver speed. |
| `--substeps S` | Substep duration is `1/(fps × substeps)`. More substeps increase work per frame. |
| `--max_substep_iters K --fixed_iters` | Exactly K solver sweeps per substep. Without `fixed_iters`, residual tolerances allow early stopping. |
| `--E --nu --density --thickness --kB` | Cloth material and bending properties; `kB=0` disables bending. |
| `--solid_E --solid_nu --solid_density` | Deformable-solid material properties, separate from cloth settings. |
| `--rigid_density` | Density for density-based rigid scenes; Examples 5–6 use calibrated masses. |
| `--d_hat --k_barrier` | Mesh contact activation distance and barrier stiffness. `d_hat=0` disables mesh barriers. |
| `--k_sdf --eps_sdf` | Prescribed-obstacle contact stiffness and soft contact range. |
| `--friction_coefficient` | Mesh/SDF Coulomb friction; zero disables it. `friction_velocity_epsilon` smooths near-zero slip. |
| `--use_parallel true` | Parallel solver updates; set the thread count with `OMP_NUM_THREADS`. |
| `--verbose` | Print solver diagnostics. Fixed-iteration solves omit residual evaluation. |

### Solver selection

| Solver | Flags |
|---|---|
| Original solver | Default; automatically chooses cloth, rigid, or general solving from scene geometry. |
| Experimental SIMD v2 | `--use_basic_experimental true --use_simd true`; supports cloth, rigid, and mixed scenes. |
| Experimental scalar cloth v1 | `--use_basic_experimental true --use_simd false`. |
| Basic cloth grid | `--use_cloth_grid true`; cloth only, with OGC disabled. See `--help` for grid sizing flags. |

`--use_colored_ccd_guess true` enables collision-colored initial guesses for
cloth/solid vertices, keeping rigid bodies fixed during the guess. It overrides
the other guess flags and is ignored in OGC mode. Alternative OGC solving
(`--use_ogc_solver`) requires `--fixed_iters`; mixed deformable/rigid scenes
use the basic/general route. Some scenes enforce their own guess/contact
presets in [example.cpp](example.cpp).

## Output and restart

`--outdir` chooses the results folder; `--format` accepts `geo` (default),
`obj`, `ply`, or `usd` (ASCII `.usda`). The default folder is `frames_sim3d/`.

| File | Contents |
|---|---|
| `frame_NNNN.*` | Simulated mesh at a completed frame; frame 0 is the initial state. |
| `state_NNNN.bin` | Particle and rigid-body state for restarting. |
| `static_colliders.*` | Prescribed obstacle geometry, when present. Load it alongside the simulated meshes. |
| `collider_NNNN.*` | Per-frame prescribed collider geometry for Examples 2–4. |

Houdini can use GEO output. OBJ and PLY provide mesh exports for other viewers.
`--write_substeps true` exports each substep and additional diagnostics; leave
it false for a normal frame sequence. Checkpoints still use frame numbers.

To continue the quick-start run from frame 30 through frame 60:

```sh
OMP_NUM_THREADS=4 ./build/3D_sim \
  --example 1 --twist_nx 9 --twist_ny 9 \
  --fps 30 --num_frames 60 --substeps 3 \
  --max_substep_iters 20 --fixed_iters \
  --use_basic_experimental true --use_simd true \
  --format obj --outdir results/quickstart --restart_frame 30
```

A restart preserves the folder and loads `state_0030.bin`. `num_frames` is the
final frame number, so this adds frames 31–60. Checkpoints store state, not run
settings: repeat the original scene, resolution, material, contact, and solver
options, including friction. Keep any required scene assets available.

The console reports mesh counts, solver iterations, and time per frame, then
total and average timing. Mixed solves with convergence checks also report
cloth, solid, and rigid residuals separately.

## Reference scene commands

These are full-scene presets, with larger meshes and iteration budgets than
the quick start. Example 14's command uses 64 OpenMP threads; adjust the thread
count for your machine. Change `--num_frames` and `--outdir` for shorter runs
or separate results.

<details>
<summary>Show commands for all 14 examples</summary>

All scene commands below enable SIMD v2 with
`--use_basic_experimental true --use_simd true`. Each build uses its available
SIMD backend or scalar fallbacks.

```bash
# Example 1: square cloth twisted in place, 240 frames at 0.5 turns/s
./build/3D_sim --example 1 --num_frames 240 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --d_hat 0.005 --k_barrier 100 \
  --node_box_min 0.001 --node_box_max 0.01 \
  --fixed_iters --max_substep_iters 6 --substeps 5 --node_box_update_count 10 \
  --use_basic_experimental true --use_simd true --use_parallel true \
  --outdir example1_output

# Example 2: two cylinders, 2.0 turns, twist then untwist
./build/3D_sim --example 2 --num_frames 900 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 5e6 \
  --d_hat 0.005 --k_barrier 100 --k_sdf 1e5 --eps_sdf 0.002 \
  --node_box_min 0.001 --node_box_max 0.01 --tcyl_max_turn 2.0 \
  --fixed_iters --max_substep_iters 6 --substeps 3 --node_box_update_count 10 \
  --use_basic_experimental true --use_simd true --use_parallel true \
  --outdir example2_output

# Example 3: one yawing cylinder, 4.0 turns at 0.30 turns/s
./build/3D_sim --example 3 --num_frames 850 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e8 \
  --d_hat 0.005 --k_barrier 100 --k_sdf 1e9 --eps_sdf 0.002 \
  --node_box_min 0.001 --node_box_max 0.01 \
  --tu_max_turn 4.0 --tu_twist_rate 0.30 \
  --fixed_iters --max_substep_iters 8 --substeps 5 --node_box_update_count 10 \
  --use_basic_experimental true --use_simd true --use_parallel true \
  --outdir example3_output

# Example 4: avatar collider and dress loaded from a data directory
./build/3D_sim --example 4 --datadir /path/to/avatar_data \
  --use_basic_experimental true --use_simd true
```

Examples 5, 6, and 7 use the corresponding scene presets from `example.cpp`:

```bash
# Example 5: freely rotating tennis racket
./build/3D_sim --example 5 --num_frames 500 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir racket_output \
  --use_basic_experimental true --use_simd true

# Example 6: freely rotating space tool
./build/3D_sim --example 6 --num_frames 2000 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir space_tool_output \
  --use_basic_experimental true --use_simd true

# Example 7: stationary stack of ten rigid polygons
./build/3D_sim --example 7 --num_frames 100 --substeps 10 --d_hat 0.001 --eps_sdf 0.0002 --rigid_density 25 --gy 0 --outdir twenty_polygon_static_stack_output --format obj \
  --use_basic_experimental true --use_simd true

```

Examples 8–14 use the following scene presets from `example.cpp`:

```bash
# Example 8: Bunny/Spot solids with rigid cubes and gears
./build/3D_sim --example 8 --datadir example_obj --num_frames 200 --fps 30 --substeps 20 --max_substep_iters 600 --fixed_iters --E 1.25e9 --nu 0.25 --thickness 0.001 --solid_E 1.25e5 --solid_nu 0.25 --d_hat 0.019 --k_barrier 1000 --outdir multi_physics_output --format obj \
  --use_basic_experimental true --use_simd true

# Example 9: dynamic threaded bolt falling through a fixed nut
./build/3D_sim --example 9 --num_frames 200 --substeps 20 --max_substep_iters 10 --fixed_iters --outdir bolt_into_fixed_nut_output --format obj \
  --use_basic_experimental true --use_simd true

# Example 10: four horizontal layers of differently ordered Bunny/Spot/cube/gear groups; Bunny and Spot lie on their sides
./build/3D_sim --example 10 --num_frames 200 --fps 30 --substeps 15 --max_substep_iters 25 --fixed_iters \
  --E 1.25e9 --nu 0.25 --thickness 0.001 --solid_E 1.25e5 --solid_nu 0.25 \
  --d_hat 0.019 --k_barrier 1000 --friction_coefficient 0 --use_ccd true \
  --node_box_update_count 10 --use_parallel true --use_basic_experimental true --use_simd true \
  --write_substeps false --format obj --outdir outputs/example10_general_v2 \
  --use_colored_ccd_guess true --colored_ccd_guess_iters 10

# Example 11: rolled cloth unrolling down an SDF ramp
./build/3D_sim --example 11 --num_frames 200 --substeps 20 --max_substep_iters 80 --fixed_iters --kB 0.0025 --friction_coefficient 0.1 --friction_velocity_epsilon 0.01 --outdir rolled_cloth_on_steep_ramp_output_new --format obj \
  --use_basic_experimental true --use_simd true

# Example 12: three cloth layers with one fixed edge and one oscillating edge
./build/3D_sim --example 12 \
  --use_basic_experimental true --use_simd true \
  --num_frames 200 --substeps 30 --max_substep_iters 100 --fixed_iters \
  --friction_coefficient 0.0 --friction_velocity_epsilon 0.01 \
  --kB 0.01 --osc_amplitude 0.01 --osc_frequency 6.0 --osc_length 1.0 --gy 0 \
  --outdir oscillating_cloth_layers_output --format geo

# Example 13: a rigid wrecking ball swings into a wall of 560 cubes
./build/3D_sim --example 13 \
  --use_basic_experimental true --use_simd true \
  --num_frames 200 --substeps 30 --max_substep_iters 200 --fixed_iters \
  --d_hat 0.001 --k_barrier 1e9 --k_sdf 1e8 --eps_sdf 0.002 \
  --friction_coefficient 0.1 --outdir wrecking_ball_tuned_output --format geo

# Example 14: fifty free cloth sheets fall over a fixed cylinder offset to the left and spread onto the ground.

OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
./build/3D_sim --example 14 --num_frames 10 \
  --fps 30 --substeps 50 --max_substep_iters 3 --fixed_iters \
  --use_basic_experimental --use_simd --use_parallel true --use_ccd true \
  --E 1e6 --nu 0.3 --kB 0.001 --d_hat 0.0048 --k_barrier 1000 \
  --k_sdf 1e5 --eps_sdf 0.015 \
  --node_box_min 0.0002 --node_box_max 0.005 --node_box_update_count 3 \
  --outdir results/example14_70x70/frames --format geo && \
OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
./build/3D_sim --example 14 --restart_frame 10 --num_frames 120 \
  --fps 30 --substeps 50 --max_substep_iters 2 --fixed_iters \
  --use_basic_experimental --use_simd --use_parallel true --use_ccd true \
  --E 1e6 --nu 0.3 --kB 0.001 --d_hat 0.0048 --k_barrier 1000 \
  --k_sdf 1e5 --eps_sdf 0.015 \
  --node_box_min 0.0002 --node_box_max 0.005 --node_box_update_count 2 \
  --outdir results/example14_70x70/frames --format geo
```

</details>

## Build and test

The quick-start build skips tests and developer tools. To enable and run the
test suite, install GoogleTest, then use:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

List discovered tests with `ctest --test-dir build -N`, or run a test binary
such as `./build/make_shape_test` directly. Set `-DIPC_BUILD_TOOLS=ON` to build
`generate_golden`, which rewrites regression fixtures; it is not needed to run
simulations.

## Troubleshooting

| Issue | What to check |
|---|---|
| CMake cannot find Boost, OpenMP, or GoogleTest | Install the dependency and, for nonstandard locations, supply its CMake package/prefix path. GoogleTest can be skipped with `-DBUILD_TESTING=OFF`. |
| First configuration cannot fetch dependencies | CMake needs Git and network access to download the pinned dependencies. |
| A scene cannot open an OBJ or TetGen file | Run from `3D_IPC/` and check the [scene assets](#scene-assets); Example 4 needs your own `--datadir`. |
| Startup rejects `d_hat` | Its effective value must not exceed half the shortest mesh edge. Smaller scene gaps may impose stricter limits; use the scene preset or lower it. |
| Results were replaced | Fresh runs recreate `outdir`; use a new folder or `--restart_frame` with an existing checkpoint. |
| A full scene takes much longer than the quick start | Mesh resolution, substeps, and iteration counts all affect cost. Start with fewer frames; only use resolution flags belonging to that scene. |

## Source guide

- [example.cpp](example.cpp): scene geometry, motion, and scene-specific presets.
- [ipc_args.h](ipc_args.h): complete CLI options and defaults.
- [simulation.cpp](simulation.cpp) and [simulation.h](simulation.h): frame loop,
  time stepping, solver selection, and output.
- [solver.cpp](solver.cpp), [physics.cpp](physics.cpp), and
  [solid_ipc.cpp](solid_ipc.cpp): cloth, rigid, and solid solving and energies.
- [broad_phase.cpp](broad_phase.cpp), [ccd.cpp](ccd.cpp), and
  [safe_step.cpp](safe_step.cpp): collision candidates and safe updates.
- [output.cpp](output.cpp) and [state_io.cpp](state_io.cpp): exports and restart state.
- [CMakeLists.txt](CMakeLists.txt): build options and dependencies.

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

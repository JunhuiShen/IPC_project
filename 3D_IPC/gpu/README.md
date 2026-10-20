# CUDA solver for 3D IPC

`3D_sim_gpu_resident` runs cloth, solid, rigid-body and contact updates on the
GPU. It shares scene definitions, command-line options and output formats with
[3D IPC](../README.md).

## Where to start reading

1. [The parent README](../README.md) introduces scenes, materials, options and
   output formats.
2. [CMakeLists.txt](CMakeLists.txt) defines the compiled targets.
   [GeneratedSources.cmake](cmake/GeneratedSources.cmake) generates the shared
   driver overlays; [Validation.cmake](cmake/Validation.cmake) defines the
   component tests; [Hybrid.cmake](cmake/Hybrid.cmake) holds the legacy targets.
3. [resident_driver_step.inc](resident_driver_step.inc) and
   [resident_bridge.cpp](resident_bridge.cpp) show how scene data, prescribed
   motion and timestep settings reach the GPU solver.
4. [resident_solver.h](resident_solver.h) defines `SceneDescription`,
   `StepParameters` and the public solver API. Read `ResidentSolver::advance_substep`
   in [resident_solver.cu](resident_solver.cu) for the update order described below.
5. Follow each stage into its pipeline and arithmetic headers using the source
   map. Corresponding `*_test.cpp` and `*_test.cu` callers show supported inputs,
   fallback cases and error behavior.

### Source map

| Area | Files and role |
|---|---|
| Driver integration | [resident_driver_step.inc](resident_driver_step.inc): replaces frame-advance dispatch in generated `simulation_resident.cpp`. |
| Host/device boundary | [resident_bridge.h](resident_bridge.h), [resident_bridge.cpp](resident_bridge.cpp): scene and step descriptions, prescribed inputs and snapshot transfer. |
| Persistent solver | [resident_solver.h](resident_solver.h), [resident_solver.cu](resident_solver.cu): device state ownership, rebuilds, iteration/color loops and failure handling. |
| Elastic assembly | [resident_elastic.h](resident_elastic.h), [resident_elastic.cu](resident_elastic.cu): incident gathers, derivative evaluation, reduction, proposals and deformable commits. |
| Elastic arithmetic | [cloth_math.cuh](cloth_math.cuh), [solid_math.cuh](solid_math.cuh), [rigid_math.cuh](rigid_math.cuh): material gradients, Hessians and rigid kinematics. |
| Broad phase and coloring | [resident_broadphase.cu](resident_broadphase.cu), [pbgs_coloring.cuh](pbgs_coloring.cuh), [pbgs_warp_ordered_exact.cuh](pbgs_warp_ordered_exact.cuh), [pbgs_warp_shared_forbidden_exact.cuh](pbgs_warp_shared_forbidden_exact.cuh), [parallel_scene_bounds.cuh](parallel_scene_bounds.cuh), [pbgs_label_compaction.cuh](pbgs_label_compaction.cuh): boxes, parallel scene bounds, candidates, dependency graph, ordered coloring and compact labels. |
| Contact pipeline | [resident_contact_pipeline.cu](resident_contact_pipeline.cu): gathers, ordinary evaluation, exact queues/retries and ordered assembly into local systems. |
| Contact arithmetic | [resident_contact.cuh](resident_contact.cuh), [resident_contact_selected_ordinary.cuh](resident_contact_selected_ordinary.cuh), [resident_contact_exact.cuh](resident_contact_exact.cuh), [resident_contact_cooperative.cuh](resident_contact_cooperative.cuh): ordinary derivatives and scalar/cooperative exact paths. |
| Compact point contacts | [resident_contact_compact_point.cu](resident_contact_compact_point.cu), [cooperative_compact_point_study.cu](cooperative_compact_point_study.cu), [compact_point_prefix.cuh](compact_point_prefix.cuh), [direct_native_point.cuh](direct_native_point.cuh): guarded point-contact routes and fallback queues. |
| Local solve | [resident_solve.cuh](resident_solve.cuh): `solve3` and its rank-deficient fallback. |
| Linear CCD | [resident_linear_pipeline.cu](resident_linear_pipeline.cu), [resident_ccd.cuh](resident_ccd.cuh), [resident_exact.cuh](resident_exact.cuh), [resident_direct_bound.cuh](resident_direct_bound.cuh): conservative clipping, exact predicates and optional exact step proposals. |
| Rigid updates | [resident_rigid_pipeline.cu](resident_rigid_pipeline.cu), [resident_rigid_ccd.cuh](resident_rigid_ccd.cuh): translation/orientation proposals, curved-motion CCD and commits. |
| Initial guesses and movement boxes | [resident_global_guess.cu](resident_global_guess.cu), [resident_adaptive_boxes.cu](resident_adaptive_boxes.cu), [resident_level_schedule.cu](resident_level_schedule.cu): predictor CCD, movement limits and optional trial/BFS level scheduling. |
| Scene records, SDFs and buffers | [resident_scene_types.h](resident_scene_types.h), [resident_sdf_types.h](resident_sdf_types.h), [resident_sdf.cuh](resident_sdf.cuh), [resident_scene.h](resident_scene.h), [resident_memory.h](resident_memory.h): public records, obstacle evaluation and device storage. |

Most `.cuh` files contain device arithmetic; `.cu` files contain kernels and
launch orchestration. The numerical libraries are `ipc_resident_solver` and
`ipc_resident_components`; `ipc_resident_bridge` connects them to shared scene
and IO support. `cooperative_compact_point_study.cu` is compiled into the solver
despite its name; runtime options select its routes.

## What one substep does

```text
generated driver → bridge → ResidentSolver::advance_substep
  save state / predict / initial guess
  fixed iterations:
    rebuild boxes, candidates and colors at the configured interval
    each color in sequence:
      elastic + contact → local solve → CCD → commit
  finish velocities / check status
snapshot when requested → shared mesh/checkpoint output
```

1. Upload prescribed pin/SDF inputs, save previous state and predict targets.
   Apply the requested global cloth guess or colored guess with its CCD path.
2. Initialize adaptive movement limits. During the fixed iteration loop,
   rebuild movement boxes, broad-phase candidates, dependency colors, contact
   incidence and enabled worklists at the configured interval.
3. Visit colors **sequentially**. Independent blocks within a color update in
   parallel; later colors see the preceding committed state.
4. For deformables, `assemble_elastic_color` and
   `ResidentContactPipeline::assemble_color` gather current data, evaluate
   derivatives and accumulate contributions in their defined order.
   `propose_deformable_color` solves `H * delta = gradient` and proposes
   `x - damping * delta`; CCD clips the proposal before
   `commit_deformable_endpoints` copies the accepted coordinates.
5. For rigid bodies, assemble, propose, clip and commit translation first.
   Then perform a fresh orientation assembly, curved-motion CCD and commit.
6. Finish velocities, synchronize and check device status. The driver requests
   snapshots for mesh/checkpoint output.

The host handles scene setup, prescribed inputs, allocation, launch scheduling
and output. Positions, velocities, rigid poses, gathered records, derivatives
and numerical fallbacks remain on the device between snapshots. Counts,
launch metadata and error status can still return to the host.

### Arithmetic and collision safety

Calculations use binary64 with CUDA `--fmad=false`; explicit source `fma` calls
are retained. Linear CCD uses conservative floating-point certificates and
device integer/rational fallback for uncertain queries. Contact geometry has
separate exact queues and capacity checks. Rigid rotation uses its curved-motion
CCD path. Exact predicates do not make the entire simulation arbitrary precision.

Preserve the models, clamps, fixed budgets, contribution order and fallback
contracts. CCD commits the certified represented endpoint. An unrecoverable
error invalidates the state and prevents output or reuse of partial updates.

## Dependencies

| Dependency | Version / configuration |
|---|---|
| CUDA Toolkit | 12.6 used, including CUDA runtime and cuSPARSE |
| NVIDIA GPU | RTX 6000 Ada used; default CUDA architecture `89` |
| C++ compiler | C++17; GCC 11 used |
| CMake | 3.24 or newer |
| Boost | **1.92.0** |
| Eigen | **3.4.0** |
| Tight-Inclusion | **1.0.6** |
| OpenMP | Required by the shared scene/IO support |

CMake fetches Eigen and Tight-Inclusion. Set `Boost_DIR` to your installed
Boost 1.92.0 CMake package directory.

## Build

Run from the `3D_IPC/` directory:

```sh
cmake -S gpu -B gpu/build -DCMAKE_BUILD_TYPE=Release \
  -DBoost_DIR=/path/to/boost-1.92.0/lib/cmake/Boost-1.92.0 \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build --target 3D_sim_gpu_resident -j 8
```

The shared CPU source defaults to the parent directory. To use another
compatible checkout, add `-DIPC_CPU_SOURCE=/path/to/3D_IPC`.
For offline builds, set `FETCHCONTENT_SOURCE_DIR_EIGEN3` and
`FETCHCONTENT_SOURCE_DIR_TIGHT_INCLUSION` to local dependency directories.

## Run

This small Example 1 run uses a 9-by-9 mesh:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
./gpu/build/3D_sim_gpu_resident \
  --example 1 --twist_nx 9 --twist_ny 9 --num_frames 2 --fps 30 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --fixed_iters --substeps 5 --max_substep_iters 2 \
  --node_box_update_count 1 --d_hat 0.005 --k_barrier 100 \
  --use_ccd true --use_ticcd false --exact_computation_fallback true \
  --use_ccd_guess false --use_colored_ccd_guess true \
  --colored_ccd_guess_iters 2 --write_substeps false \
  --format geo --outdir results/gpu_resident_smoke
```

Two optional CPU features also have resident GPU implementations. Both default
to false. `--use_colored_ccd_guess_level_based true` selects a discarded colored
CCD trial followed by component/BFS level and color groups. Choose one colored
mode; either suppresses the global guess and uses `--colored_ccd_guess_iters` actual sweeps.
To select level mode in the example above, replace `--use_colored_ccd_guess true`
with `--use_colored_ccd_guess false --use_colored_ccd_guess_level_based true`;
setting both colored-mode flags is rejected. Zero sweeps performs no trial.
`--ccd_exact_step_bound true` proposes an exact
partial step for rejected deformable endpoints. Every proposal still passes
GPU CCD and rounded-endpoint checks; an inconclusive proposal retains the
existing conservative fallback. These options keep numerical work on the GPU.
The GPU's existing conservative CCD can choose different steps from the CPU.

Use a new output directory for each fresh run. Output includes frame meshes
and `state_NNNN.bin` checkpoints, starting at frame 0. Restart in the same
directory with `--restart_frame N`, preserving the original scene and solver
settings. `--num_frames` specifies the final frame number.
See [output and restart](../README.md#output-and-restart).

The resident driver requires fixed iterations, conflict-graph coloring and
exact fallback. Use `--use_ticcd false`. Colors execute sequentially, with
independent updates within each color running in parallel. Numerical fallback
and CCD execute on the GPU; an unrecoverable device error stops the substep.

## Validation

```sh
cmake --build gpu/build --target gpu_resident_validation gpu_resident_core_validation -j 8
ctest --test-dir gpu/build -R '^gpu_resident_' --output-on-failure
```

The validation targets cover elastic assembly, broad phase, coloring,
contacts, local solves, CCD, SDFs, exact arithmetic and integrated updates.

## Standalone numerical core

To build the CUDA libraries without shared CPU scene/IO support, Boost, Eigen
or Tight-Inclusion:

```sh
cmake -S gpu -B gpu/build-core -DCMAKE_BUILD_TYPE=Release \
  -DIPC_GPU_BUILD_DRIVER=OFF -DIPC_GPU_BUILD_VALIDATION=OFF \
  -DIPC_GPU_BUILD_HYBRID=OFF \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build-core --target gpu_resident_core_validation -j 8
ctest --test-dir gpu/build-core -R '^gpu_resident_core_smoke$' --output-on-failure
```

This builds `ipc_resident_solver`, `ipc_resident_components` and a public-API
smoke caller. The shared-scene simulator is built by the normal configuration.

## Runtime diagnostics

`IPC_GPU_COLORING` selects `greedy` (default), `degree`, `spatial` or `cusparse`.
The ordered `degree` and `spatial` modes use warp-cooperative coloring with
shared marks when its palette fits the shared-memory budget; larger palettes
use global marks. `IPC_GPU_WARP_SHARED_MARKS=0` selects global marks, and
`IPC_GPU_WARP_ORDERED_COLOR=0` selects serial coloring with the same labels and
execution order. Scene bounds use parallel reduction, with the original serial
scan for exceptional coordinates. `IPC_GPU_PARALLEL_SCENE_BOUNDS=0` selects the
serial scan for all coordinates.

- `IPC_GPU_PROFILE_CSV=/new/path/profile.csv` enables per-frame stage timings.
- `IPC_GPU_CONTACT_QUEUE_CSV=/new/path/queues.csv` enables contact queue counts.

Set environment options before launching the executable. CUDA stage timings
measure stream intervals, which can include gaps while the host submits work;
they are not kernel-active totals. Instrumentation adds overhead, so use
uninstrumented runs for performance comparisons. Options for contact routing and worklists are parsed in
[resident_contact_pipeline.cu](resident_contact_pipeline.cu),
[resident_contact_schedule.h](resident_contact_schedule.h) and
[resident_elastic.h](resident_elastic.h).

## Legacy hybrid reference

The legacy hybrid executable `3D_sim_gpu` remains available as an explicit
build target. Use `3D_sim_gpu_resident` for the resident solver.

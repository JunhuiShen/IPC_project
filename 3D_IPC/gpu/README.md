# CUDA solver for 3D IPC

**`3D_sim_gpu_resident` is the current simulator.** It keeps evolving cloth,
solid, rigid-body, and contact state and numerical substeps on the GPU,
including device exact-arithmetic fallbacks. It shares [3D IPC](../README.md)'s
scene setup, material data, command-line options, and mesh/checkpoint formats.
The solver is experimental: qualification applies to a particular build,
configuration, and scene; GPU execution alone establishes no general speedup.

## Where to start reading

1. [The parent README](../README.md) explains scenes, materials, and outputs.
2. [CMakeLists.txt](CMakeLists.txt) defines the actual compiled targets and
   includes three build modules: [GeneratedSources.cmake](cmake/GeneratedSources.cmake)
   for checked overlays, [Hybrid.cmake](cmake/Hybrid.cmake) for explicit legacy
   targets, and [Validation.cmake](cmake/Validation.cmake) for optional tests.
3. [resident_bridge.cpp](resident_bridge.cpp) maps project data and parameters
   into the resident interfaces.
4. [resident_solver.cu](resident_solver.cu), especially
   `ResidentSolver::advance_substep`, shows the full update order.
5. Follow the stage below into its pipeline and arithmetic headers. Read the
   corresponding `*_test*` files for supported cases and error behavior.

### Source map

| Area | Files and role |
|---|---|
| Driver integration | [resident_driver_step.inc](resident_driver_step.inc) replaces the CPU frame-advance dispatch in generated `simulation_resident.cpp`. |
| Host/device boundary | [resident_bridge.h](resident_bridge.h), [resident_bridge.cpp](resident_bridge.cpp): `describe_scene`, `describe_step`, prescribed inputs, and `copy_snapshot`. |
| Persistent solver | [resident_solver.h](resident_solver.h), [resident_solver.cu](resident_solver.cu): state ownership, rebuilds, iteration/color loops, and failure handling. |
| Elastic assembly | [resident_elastic.h](resident_elastic.h), [resident_elastic.cu](resident_elastic.cu): gather incident records, evaluate, reduce, propose, and commit deformable updates. |
| Elastic arithmetic | [cloth_math.cuh](cloth_math.cuh), [solid_math.cuh](solid_math.cuh), [rigid_math.cuh](rigid_math.cuh): material gradients, Hessians, and rigid kinematics. |
| Broad phase and colors | [resident_broadphase.cu](resident_broadphase.cu), [pbgs_coloring.cuh](pbgs_coloring.cuh), [pbgs_label_compaction.cuh](pbgs_label_compaction.cuh): boxes, candidates, graph/color construction, and compact labels. |
| Contact pipeline | [resident_contact_pipeline.cu](resident_contact_pipeline.cu): gather/fast evaluation, exact queues/retries, and ordered assembly into local systems. |
| Contact arithmetic | [resident_contact.cuh](resident_contact.cuh), [resident_contact_exact.cuh](resident_contact_exact.cuh), [resident_contact_cooperative.cuh](resident_contact_cooperative.cuh): contact derivatives and scalar/cooperative exact paths. |
| Compact point tiers | [resident_contact_compact_point.cu](resident_contact_compact_point.cu), [cooperative_compact_point_study.cu](cooperative_compact_point_study.cu), [compact_point_prefix.cuh](compact_point_prefix.cuh), [direct_native_point.cuh](direct_native_point.cuh): guarded point-contact routes with existing fallback queues. |
| Local solve | [resident_solve.cuh](resident_solve.cuh): `solve3`, including the rank-deficient fallback. |
| Linear CCD | [resident_linear_pipeline.cu](resident_linear_pipeline.cu), [resident_ccd.cuh](resident_ccd.cuh), [resident_exact.cuh](resident_exact.cuh): clipping certificates and device exact predicates. |
| Rigid updates | [resident_rigid_pipeline.cu](resident_rigid_pipeline.cu), [resident_rigid_ccd.cuh](resident_rigid_ccd.cuh): translation/orientation proposals, curved-motion CCD, and commits. |
| Initial guess and boxes | [resident_global_guess.cu](resident_global_guess.cu), [resident_global_guess.cuh](resident_global_guess.cuh), [resident_adaptive_boxes.cu](resident_adaptive_boxes.cu): predictor CCD and movement limits. |
| Public records | [resident_sdf_types.h](resident_sdf_types.h), [resident_scene_types.h](resident_scene_types.h): plain SDF/scene records used by the host API and numerical callers. |
| SDFs and storage | [resident_sdf.cuh](resident_sdf.cuh), [resident_scene.h](resident_scene.h), [resident_memory.h](resident_memory.h): obstacle evaluation, complete numerical includes, and device buffers. |

Most `.cuh` headers hold device arithmetic; `.cu` files combine kernels and
launch orchestration. CMake links nine CUDA units into `ipc_resident_solver`, two into
`ipc_resident_components`, and the C++ bridge into `ipc_resident_bridge`.
The components library publicly links CUDA runtime and cuSPARSE dependencies.

The public solver header includes the type-only scene boundary rather than the
SDF evaluator. Record definitions and defaults were moved verbatim; public solver
signatures and mathematical bodies stay unchanged. Numerical callers can still
include `resident_scene.h` for its complete evaluator contract. This reduces
header fanout; it does not establish a compile-time or runtime speedup.

A name containing `study` does **not** mean test-only:
`cooperative_compact_point_study.cu` is compiled into the resident solver.
Runtime flags select its use. Active-point/finite-CCD selections are local to
that unit; the linear pipeline also selects finite-CCD locally. Do not enable selectors
globally or infer production inclusion from a filename alone.

## What one substep does

```text
generated driver → bridge → ResidentSolver::advance_substep
  predict/rebuild → colors → elastic + contact → local solve → CCD → commit
  finish velocities → snapshot → shared mesh/checkpoint output
```

The host loads meshes, builds immutable rest data, supplies prescribed pin/SDF
motion, allocates storage, and launches work. Positions, velocities, rigid
poses, gathered records, derivatives, and numerical fallback work stay on the
device between output snapshots. Counts, launch metadata, and error status
can still return to the host.

1. Save previous state, predict targets, and apply the requested initial guess.
   The global cloth guess and colored guess have separate CCD paths.
2. Rebuild movement boxes, broad-phase candidates, dependency colors, and
   optional worklists at the configured interval.
3. Visit colors **sequentially**. Independent blocks within a color update in
   parallel; later colors see the preceding committed state.
4. For deformables, `assemble_elastic_color` and
   `ResidentContactPipeline::assemble_color` gather current data, evaluate
   derivatives, and accumulate incident contributions in their defined order.
5. `propose_deformable_color` solves `H * delta = gradient`, then proposes
   `x - damping * delta`. CCD clips the proposal; `commit_deformable_endpoints`
   copies the accepted represented coordinates.
6. Rigid translation is assembled, clipped, and committed before a fresh
   orientation assembly/curved-CCD/commit. The two phases are sequential.
7. Finish velocities and check device status before exposing a snapshot.

Preserve this order, fixed budgets, models, signs, clamps, degeneracy behavior,
and fallback/error contracts. Fixed sweep completion does not prove convergence.

### Arithmetic and collision safety

Resident calculations use binary64 and CUDA targets use `--fmad=false`, with
explicit `fma` calls retained where the source requires them. Do not substitute
fast math when comparing or qualifying these paths.

Linear CCD first uses conservative floating-point certificates; uncertain
queries enter the existing device integer/rational fallback and retry paths.
Contact geometry has its own exact-arithmetic queues and capacity checks.
There is no CPU numerical retry. Exact predicates do not make the entire
simulation arbitrary precision: square roots, logarithms, derivatives, and
evolving state still involve rounded floating-point arithmetic.

Uncertainty is not permission to accept motion. Commit uses the certified
rounded endpoint rather than recomputing it. An unrecoverable device error
invalidates the substep/state and blocks output or reuse of partial updates.
Rigid rotation retains the existing curved-motion algorithm; linear exact
predicate tests do not establish a stronger guarantee for that algorithm.

## Build and run

Run the following commands from `3D_IPC/`, the repository root.

### Requirements

- CMake 3.24+, a C++17 compiler, and an NVIDIA GPU with a compatible CUDA toolkit.
- OpenMP, Boost, Eigen, and Tight-Inclusion; see [the main build instructions](../README.md#build-and-test).
- The tested Linux configuration uses CUDA 12.6, GCC 11, and an RTX 6000 Ada
  GPU (`sm_89`). Other toolkit/compiler/device combinations need validation.

These commands assume `nvcc` and `g++-11` are on `PATH`; substitute their paths
when needed. Architecture 89 is also the current CMake default.

```sh
cmake -S gpu -B gpu/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build -j 8
```

The ordinary build selects `3D_sim_gpu_resident` and its required libraries and
host support. Optional test and control targets are excluded from `ALL`.
An explicit `--target 3D_sim_gpu_resident` selects the same simulator.
Build and run the resident validation callers separately:

```sh
cmake --build gpu/build --target gpu_resident_validation -j 8
ctest --test-dir gpu/build -R '^gpu_resident_' --output-on-failure
```

Initial configuration may fetch dependencies.
Offline builds can supply
`FETCHCONTENT_SOURCE_DIR_EIGEN3` and `FETCHCONTENT_SOURCE_DIR_TIGHT_INCLUSION`.
Use a dedicated Tight-Inclusion source copy: its configuration generates a
header in that tree.

The selected CPU tree supplies scenes, types, setup, and outputs, not a resident
numerical fallback. `IPC_CPU_SOURCE` defaults to the parent directory; to select a
separate compatible checkout, add `-DIPC_CPU_SOURCE=/path/to/compatible/3D_IPC`.
CMake leaves the selected CPU base unchanged and writes checked driver/solver
overlays into the build directory. The shared generators require unique
insertion anchors and write only changed content, so a no-change reconfigure
preserves generated-source timestamps. [check_overlay.cmake](check_overlay.cmake)
uses those same generators for its explicit overlay checks. Reconfigure after
changing the selected CPU source.

### Small resident run

Use a fresh output directory for each run:

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

This writes frame meshes and `state_NNNN.bin` checkpoints, including frame 0.
The resident driver refuses to replace an existing directory on a fresh run.
Restart with `--restart_frame N` in the same directory, repeating the original
scene/material/solver settings; `--num_frames` is the final frame number.
See [output and restart](../README.md#output-and-restart) for file formats.

OpenMP controls host work, not CUDA block width. The resident executable selects
the GPU solver; CPU SIMD/experimental-solver flags do not choose its kernels.
Keep fixed iteration/substep budgets when comparing configurations.

## Defaults, experiments, and diagnostics

Scene/solver arguments and `IPC_GPU_*` environment switches are different
interfaces; leave them unset for quickstart defaults. Selected examples:

| Environment setting | Default / purpose |
|---|---|
| `IPC_GPU_COLOR_PAIR_QUEUES` | `1`; per-color contact queues. |
| `IPC_GPU_COOPERATIVE_EXACT_CONTACT` | `1`; cooperative exact evaluation. |
| `IPC_GPU_ELASTIC_COLOR_WORKLISTS` | `0`; opt-in stable elastic color worklists. |
| `IPC_GPU_ELASTIC_NODE_THREADS`, `IPC_GPU_ELASTIC_ELEMENT_THREADS` | `1`, `8` when worklists are enabled; powers of two through 128. |
| `IPC_GPU_COMPACT_POINT_FIRST_TIER`, `IPC_GPU_COMPACT_POINT_COOPERATIVE`, `IPC_GPU_CONTACT_INPLACE_RATIO` | `0`; optional compact point routing. Cooperative routing also requires the first tier, cooperative exact path, and in-place ratio selection. |
| `IPC_GPU_COMPACT_DEFORMABLE_CONTACTS`, `IPC_GPU_CONTACT_OUTPUT_ACTIVITY` | `0`; optional compact output/storage behavior. |
| `IPC_GPU_PROFILE_CSV` | Unset; optional per-frame stage timing CSV at a fresh path. |
| `IPC_GPU_CONTACT_QUEUE_CSV` | Unset; optional queue diagnostics, including extra count readbacks. |

See [resident_contact_pipeline.cu](resident_contact_pipeline.cu),
[resident_contact_schedule.h](resident_contact_schedule.h), and
[resident_elastic.h](resident_elastic.h) for parsing, dependencies, and bounds.
Options are often captured at object construction; set them before launch.
Record the complete environment when comparing builds. An opt-in route is
not automatically the selected or fastest full-solver configuration.

Stage CUDA-event intervals can include idle gaps and overlap host spans; do not
add these times. Instrumentation adds overhead. Compare full solver times on
matched representative workloads before attributing application gains to a kernel.

### Studied Example 1 environment

This separate profile controls contact routing/storage, worklists, CCD launches,
coloring, and host scheduling. It is not enabled by the default quickstart:

```sh
export OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE
export IPC_GPU_COMPACT_DEFORMABLE_CONTACTS=1 IPC_GPU_BOX_CONTACT_PRUNING=1
export IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY=1 IPC_GPU_GLOBAL_GUESS_WORKERS=8192 IPC_GPU_GLOBAL_GUESS_THREADS=32
export IPC_GPU_ELASTIC_COLOR_WORKLISTS=1 IPC_GPU_CONTACT_OUTPUT_ACTIVITY=1
export IPC_GPU_CONTACT_PREFETCH_REDUCE=1 IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS=1
export IPC_GPU_CONTACT_INPLACE_RATIO=1 IPC_GPU_CCD_CURRENT_COLOR_LAUNCH=1
export IPC_GPU_DEFER_CONTACT_INPUTS=1 IPC_GPU_CANONICAL_POINT_HESSIAN=1
export IPC_GPU_COLORING=degree IPC_GPU_COMPACT_POINT_FIRST_TIER=1 IPC_GPU_COMPACT_POINT_COOPERATIVE=1
export OMP_PROC_BIND=close OMP_PLACES=cores
```

The first eight exports contain the studied 19 settings; the last records host
affinity. These flags alone do not reproduce a measured runtime: retain the
exact build, scene, fixed budgets, and output/checkpoint options with each result.

The current source includes the active point and finite-CCD selections.
Compact membrane, guarded bending, and Schur prototypes remain separate;
the names “compact contact” and “compact membrane” do not mean the same feature.

## Supported scope and validation

The resident path implements cloth, tetrahedral solids, rigid bodies, mixed
scenes, mesh barriers/friction, and plane/cylinder/sphere SDFs. The bridge
currently requires fixed iterations, conflict-graph coloring, and exact
fallback enabled. OGC/grid modes and standalone Verlet/translation guesses
are rejected. Set `--use_ticcd false`: the per-update Tight-Inclusion backend
is not the resident linear clipper; the global predictor has its own port.

`*_test.cpp` and `*_test.cu` files are component/integration callers, not solver
entry points. The validation target covers elastic assembly/worklists,
broad-phase/coloring, contact queues/derivatives, local solves, CCD/exact pools,
SDFs, initial guesses, and integrated updates. Python `validate*`, `profile*`,
`analyze*`, and tuning scripts support particular experiments; inspect their
paths and assumptions before reuse. They do not select the simulator backend.

Changes need numerical/source-expression checks, guard/fallback/error cases,
actual pipeline callers, and appropriate sanitizer coverage. Combined solver
changes also need representative full runs and collision/physical checks.
CPU and GPU trajectories need not be bitwise identical. Passing local tests
or saved-frame collision checks does not prove swept-path or global physical
acceptance for every scene.

## Legacy hybrid reference

`3D_sim_gpu` is retained for historical comparison and source reuse. Its
generated `solver_gpu.cpp` keeps CPU update and safety stages. It is not the
current solver or part of the ordinary build/run instructions above.
See [color_bridge.h](color_bridge.h), [general_bridge.cpp](general_bridge.cpp),
[rigid_bridge.cpp](rigid_bridge.cpp), and the `*_sweep.inc` hooks. Its separate
`ipc_gpu_hybrid_overlay` archive links ahead of the unchanged CPU base;
resident targets use that base without this legacy overlay. Shared material
records and useful legacy implementations remain preserved.

Historical callers can explicitly build `3D_sim_gpu`, `3D_sim_cpu_control`,
or the `gpu_hybrid_validation` aggregate. None joins ordinary `ALL`.
Keep results from the resident, hybrid, and CPU-control executables distinct.

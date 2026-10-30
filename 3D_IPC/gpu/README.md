# CUDA solver for 3D IPC

`3D_sim_gpu` is the single public GPU simulator. It runs cloth, solid,
rigid-body and contact updates on the GPU through the current device solver
backend, and shares scene definitions, command-line options and output formats
with [3D IPC](../README.md). The public executable is the current device
solver driver; the retired hybrid comparison targets are no longer built.

## Where to start reading

1. [The parent README](../README.md) introduces scenes, materials, options and
   output formats.
2. [CMakeLists.txt](CMakeLists.txt) defines the compiled targets.
   [GeneratedSources.cmake](cmake/GeneratedSources.cmake) generates the GPU
   driver; [Validation.cmake](cmake/Validation.cmake) defines the component
   tests.
3. [driver_step.inc](driver_step.inc) and [bridge.cpp](bridge.cpp) show how
   scene data, prescribed motion and timestep settings reach the GPU solver.
   The generator copies the shared example driver into the build tree as
   `simulation_gpu.cpp`, replaces its frame dispatch with that fragment, and
   rewrites the copied quoted `solver.h`/`DAT.h` includes to the read-only CPU
   tree so the GPU `solver.h`/`dat.h` headers cannot shadow them.
4. [solver.h](solver.h) defines `SceneDescription`, `StepParameters` and the
   public solver API. Read `GpuSolver::advance_substep` in
   [solver.cu](solver.cu) for the update order described below.
5. Follow each stage into its pipeline and arithmetic headers using the source
   map. Corresponding `*_test.cpp` and `*_test.cu` callers show supported
   inputs, fallback cases and error behavior.

### Source map

| Area | Files and role |
|---|---|
| Driver integration | [driver_step.inc](driver_step.inc): replaces frame-advance dispatch in generated `simulation_gpu.cpp`. |
| Host/device boundary | [bridge.h](bridge.h), [bridge.cpp](bridge.cpp): scene and step descriptions, prescribed inputs and snapshot transfer. |
| Persistent solver | [solver.h](solver.h), [solver.cu](solver.cu): device state ownership, rebuilds, iteration/color loops and failure handling. |
| Elastic assembly | [elastic.h](elastic.h), [elastic.cu](elastic.cu): incident gathers, derivative evaluation, reduction, proposals and deformable commits. |
| Elastic arithmetic | [cloth_math.cuh](cloth_math.cuh), [solid_math.cuh](solid_math.cuh), [rigid_math.cuh](rigid_math.cuh): material gradients, Hessians and rigid kinematics. |
| Broad phase and coloring | [broadphase.cu](broadphase.cu), [pbgs_coloring.cuh](pbgs_coloring.cuh), [pbgs_warp_ordered_exact.cuh](pbgs_warp_ordered_exact.cuh), [pbgs_warp_shared_forbidden_exact.cuh](pbgs_warp_shared_forbidden_exact.cuh), [parallel_scene_bounds.cuh](parallel_scene_bounds.cuh), [pbgs_label_compaction.cuh](pbgs_label_compaction.cuh): boxes, parallel scene bounds, candidates, dependency graph, ordered coloring and compact labels. |
| Contact pipeline | [contact_pipeline.cu](contact_pipeline.cu): gathers, ordinary evaluation, exact queues/retries and ordered assembly into local systems. |
| Contact arithmetic | [contact.cuh](contact.cuh), [contact_selected_ordinary.cuh](contact_selected_ordinary.cuh), [contact_exact.cuh](contact_exact.cuh), [contact_cooperative.cuh](contact_cooperative.cuh): ordinary derivatives and scalar/cooperative exact paths. [original_contact.cuh](original_contact.cuh): the qualified original (`exact_computation_fallback == false`) geometry/force evaluator. |
| Compact point contacts | [contact_compact_point.cu](contact_compact_point.cu), [cooperative_compact_point_study.cu](cooperative_compact_point_study.cu), [compact_point_prefix.cuh](compact_point_prefix.cuh), [direct_native_point.cuh](direct_native_point.cuh): guarded point-contact routes and fallback queues. |
| Local solve | [solve.cuh](solve.cuh): `solve3` and its rank-deficient fallback. |
| Linear CCD | [linear_pipeline.cu](linear_pipeline.cu), [ccd.cuh](ccd.cuh), [exact.cuh](exact.cuh), [direct_bound.cuh](direct_bound.cuh), [original_ccd.cuh](original_ccd.cuh): conservative clipping, exact predicates, the qualified original TOI resolver and optional exact step proposals. |
| Rigid updates | [rigid_pipeline.cu](rigid_pipeline.cu), [rigid_pins.cuh](rigid_pins.cuh), [rigid_ccd.cuh](rigid_ccd.cuh): translation/orientation proposals, material-point pins, curved-motion CCD and commits. |
| Initial guesses and movement boxes | [global_guess.cu](global_guess.cu), [adaptive_boxes.cu](adaptive_boxes.cu), [level_schedule.cu](level_schedule.cu), [original_guess.cuh](original_guess.cuh): predictor CCD, movement limits, optional trial/BFS level scheduling and the CPU false predictor certificate/floor helpers. |
| Simultaneous DAT path | [dat.h](dat.h), [dat.cu](dat.cu): the optional Jacobi-with-DAT route, its budget checks and per-sweep publishing. |
| Scene records, SDFs and buffers | [scene_types.h](scene_types.h), [sdf_types.h](sdf_types.h), [sdf.cuh](sdf.cuh), [scene.h](scene.h), [device_memory.h](device_memory.h): public records, obstacle evaluation and device storage. |

Most `.cuh` files contain device arithmetic; `.cu` files contain kernels and
launch orchestration. The numerical libraries are `ipc_gpu_solver` and
`ipc_gpu_components`; `ipc_gpu_bridge` connects them to shared scene and IO
support. `cooperative_compact_point_study.cu` is compiled into the solver
despite its name; runtime options select its routes.

## What one substep does

```text
generated driver → bridge → GpuSolver::advance_substep
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
   `GpuContactPipeline::assemble_color` gather current data, evaluate derivatives
   and accumulate contributions in their defined order.
   `propose_deformable_color` solves `H * delta = gradient` and proposes
   `x - damping * delta`; CCD clips the proposal before
   `commit_deformable_endpoints` copies the accepted coordinates.
5. For rigid bodies, assemble, propose, clip and commit translation first.
   Then perform a fresh orientation assembly, curved-motion CCD and commit.
6. Finish velocities, synchronize and check device status. The driver requests
   snapshots for mesh/checkpoint output.

The host handles scene setup, prescribed inputs, allocation, launch scheduling
and output. Positions, velocities, rigid poses, gathered records, derivatives
and numerical fallbacks remain on the device between snapshots. Counts, launch
metadata and error status can still return to the host.

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
| Boost | **1.92.0** (exact, checked at configure time) |
| Eigen | **3.4.0** |
| Tight-Inclusion | **1.0.6** |
| OpenMP | Required by the shared scene/IO support |

CMake fetches Eigen 3.4.0 and Tight-Inclusion 1.0.6. Set `Boost_DIR` to your
installed Boost 1.92.0 CMake package directory; another Boost version stops
configuration instead of silently mixing headers.

## Build

Run from the `3D_IPC/` directory:

```sh
cmake -S gpu -B gpu/build -DCMAKE_BUILD_TYPE=Release \
  -DBoost_DIR=/path/to/boost-1.92.0/lib/cmake/Boost-1.92.0 \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build --target 3D_sim_gpu -j 8
```

`3D_sim_gpu` is built from the generated current GPU driver and links
`ipc_gpu_bridge`, `ipc_gpu_solver` and `ipc_gpu_components`. The shared CPU
source defaults to the parent directory. To use another compatible checkout,
add `-DIPC_CPU_SOURCE=/path/to/3D_IPC`.
`IPC_GPU_BUILD_DRIVER` and `IPC_GPU_BUILD_VALIDATION` select the driver and the
validation targets independently. For offline builds, set
`FETCHCONTENT_SOURCE_DIR_EIGEN3` and `FETCHCONTENT_SOURCE_DIR_TIGHT_INCLUSION`
to local dependency directories.

## Run

This small Example 1 run uses a 9-by-9 mesh:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
./gpu/build/3D_sim_gpu \
  --example 1 --twist_nx 9 --twist_ny 9 --num_frames 2 --fps 30 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --fixed_iters --substeps 5 --max_substep_iters 2 \
  --node_box_update_count 1 --d_hat 0.005 --k_barrier 100 \
  --use_ccd true --use_ticcd false --exact_computation_fallback true \
  --use_ccd_guess false --use_colored_ccd_guess true \
  --colored_ccd_guess_iters 2 --write_substeps false \
  --format geo --outdir results/gpu_smoke
```

The driver prints `GPU device solver backend` after its preflight mode check
accepts the options, so a log without that line is not a device solve.

Two optional CPU features also have GPU implementations. Both default to false.
`--use_colored_ccd_guess_level_based true` selects a discarded colored CCD trial
followed by component/BFS level and color groups. Choose one colored mode;
either suppresses the global guess and uses `--colored_ccd_guess_iters` actual
sweeps. To select level mode in the example above, replace
`--use_colored_ccd_guess true` with `--use_colored_ccd_guess false
--use_colored_ccd_guess_level_based true`; setting both colored-mode flags is
rejected. Zero sweeps performs no trial. `--colored_ccd_guess_fraction` accepts
finite values in `[0, 1]` (default `1`). Both colored modes use the fixed target
`x^n + fraction * (x_hat - x^n)` for every sweep and the level trial. This
changes only the initial guess; the solver keeps its full `x_hat`, and rigid
bodies stay fixed during the guess. `--ccd_exact_step_bound true` proposes an
exact partial step for rejected deformable endpoints. Every proposal still
passes GPU CCD and rounded-endpoint checks; an inconclusive proposal retains the
existing conservative fallback. These options keep numerical work on the GPU.
The GPU's existing conservative CCD can choose different steps from the CPU.

Use a new output directory for each fresh run. Output includes frame meshes
and `state_NNNN.bin` checkpoints, starting at frame 0. Restart in the same
directory with `--restart_frame N`, preserving the original scene and solver
settings. `--num_frames` specifies the final frame number.
See [output and restart](../README.md#output-and-restart).

### Optional Jacobi-with-DAT path

`--use_jacobi_dat true` selects the GPU port of the simultaneous Jacobi update
with planar Divide and Truncate. It is deformable-only: cloth and solid scenes
without rigid bodies. It honours the CPU fixed budget (`--fixed_iters` with
`--max_substep_iters`), starts every substep from the previous positions and
therefore skips all initial-guess flags. `--dat_gamma` (default `0.9`),
`--dat_node_box_padding` (default `1.2`) and `--dat_query_radius` (default `0`,
automatic) keep their CPU meaning and validation.

```sh
./gpu/build/3D_sim_gpu --example 1 --twist_nx 9 --twist_ny 9 \
  --num_frames 1 --substeps 3 --max_substep_iters 20 --fixed_iters \
  --use_jacobi_dat --dat_gamma 0.9 --format geo --outdir results/gpu_dat
```

The public DAT parameters are exactly `--dat_gamma`, `--dat_query_radius` and
`--dat_node_box_padding`. DAT truncates raw displacements, not candidate lists:
each sweep limits the displacement by its Euclidean norm against
`0.5 * gamma * rq`, where `rq` is the query radius (`--dat_query_radius`,
automatic when 0), and then applies the plane factors.
`--dat_node_box_padding` scales only the BVH box half-extent used to find
candidates; it must not change the displacement cap `0.5 * gamma * rq`.
Candidate lists stay uncapped; exceeding the device work budget
is an error, never a silent truncation, so the path fails closed. Accepted
displacements are published once, at the sweep boundary, which is what makes
the update simultaneous. The filter is surface-only: DAT rejects mesh surface
contact only. It has no tetrahedron-inversion guard and no SDF collision
filter; SDF penalties remain part of the Jacobi update.

### Rigid pins and motor-driven motion

Rigid material-point pins and motor-driven prescribed pin motion (for example
the bike-chain scene) are supported. The device solver reproduces the CPU
pin-time behaviour of the *same dispatch*: a pure rigid-body scene advances
`t_next`, while a mixed/general scene uses `params.rigid_pin_time`. That CPU
quirk is intentional parity, not a GPU deviation. Compare a GPU scene only with
a CPU run that uses the same dispatch; comparing across dispatches is not a
valid parity test. Motor-driven mixed scenes remain outside the native
qualification performed so far.

## Validation

```sh
cmake --build gpu/build --target gpu_validation gpu_core_validation -j 8
ctest --test-dir gpu/build -L gpu_solver --output-on-failure
```

Every test on the current backend carries the `gpu_solver` label, so the label
selects only the current solver component suites.
`gpu_validation` builds the CPU-oracle parity targets (`gpu_dat`,
`gpu_dat_geometry`, `gpu_rigid_pins`, `gpu_cpu_parity`) together with the
component suites for elastic assembly, broad phase, coloring, contacts, local
solves, CCD, SDFs and exact arithmetic. The independent original-mode targets
`gpu_original_contact_test`, `gpu_original_ccd_test` and
`gpu_original_mode_test` compile the qualified original components and the
mode-routing/collision-only integration checks; they are registered with
CTest but stay out of the default ALL build.

`python3 gpu/validate.py --build gpu/build` is a small modern component-only
CTest launcher: it runs `ctest --test-dir <build> --output-on-failure
--no-tests=error --parallel 1 -L gpu_solver`, forwards the child output and
propagates its failure. `--build` selects the GPU build directory (default
`gpu/build`) and `--ctest` selects the CTest executable (default: discovered
`ctest`). The launcher does not configure or build, delete directories, run a
solver or benchmark, and it makes no CPU/GPU trajectory or alignment claim.
These suites provide modern component regression coverage only; separate
source-bound CPU/GPU numerical, caller, sanitizer and short physical validation
remain independent acceptance steps. This launcher neither runs those checks nor
claims that they have passed. No speedup is claimed.

`python3 gpu/example1_comparison.py --self-test` checks the named Example 1
comparison protocols offline. Comparisons bind one explicit protocol
(`readme_example1`, `historical_example1` or `legacy_colored_guess`), never
mixing documented settings with older baselines. The current README profile
(`readme_example1`) is bound to both implementations; the GPU runs its original
non-exact mode for that deformable cloth protocol. Its GPU environment pins
`OMP_NUM_THREADS=1`/`OMP_THREAD_LIMIT=1`: those are host execution resources
for device setup and output. The modelled CPU dispatch still uses the logical
64-thread reference team (`original_reference_threads=64`), so the selected
cooperative set and its `1e-28` rule do not follow the GPU host team. The
README's own CPU environment and recipe bytes are unchanged.

## Standalone numerical core

To build the CUDA libraries without shared CPU scene/IO support, Boost, Eigen
or Tight-Inclusion:

```sh
cmake -S gpu -B gpu/build-core -DCMAKE_BUILD_TYPE=Release \
  -DIPC_GPU_BUILD_DRIVER=OFF -DIPC_GPU_BUILD_VALIDATION=OFF \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build-core --target gpu_core_validation -j 8
ctest --test-dir gpu/build-core -R '^gpu_core_smoke$' --output-on-failure
```

This builds `ipc_gpu_solver`, `ipc_gpu_components` and a public-API smoke
caller. The shared-scene simulator is built by the normal configuration.

## Original and robust modes

`--exact_computation_fallback true` keeps the robust exact-fallback path.
`--exact_computation_fallback false` selects the original non-exact CPU mode
ported on the device:

- contact assembly keeps `evaluate_contact_original` and its contribution
  order while using the existing warp reduction by default. Optional compact,
  selected-node, component and prefetch reductions remain available. The
  robust selected-ordinary evaluator and cooperative exact queues retain their
  separate geometry requirements;
- for the CPU SIMD energy path on a pure cloth scene with CCD enabled, main
  solver rebuilds can discard contacts proved inactive over all four movement
  boxes. The certificate uses the original CPU projection rule. Raw incidence
  counts remain separate from retained counts: cooperative selection ranks
  by raw cost, then tests the retained cost of the first ranked vertex. These
  counts and the selected-node mask stay on the device;
- ordinary deformable updates use the CPU componentwise blue-box clip and the
  stored raw displacement; a collision commits `0.9 * TOI`, otherwise weight
  one, and the actual represented endpoint/path is certified by a collision-only
  check. That resolve runs as a bounded pool of whole-node workers (each with a
  disjoint reusable exact arena slice) and only reads immutable current
  positions, so independent same-color vertices still update in parallel; the
  whole launch joins before any commit or the next color;
- linear CCD uses the qualified `original_ccd` fast query plus its bounded
  exact resolver. An original failure is never sent to the robust exact queue;
- the CPU original cooperative selected-node `1e-28` squared cutoff is modelled
  from `original_reference_threads` (`1` selects the noncooperative rule). This
  is scheduler-FAMILY metadata, not a geometry toggle: the cloth
  `ColoredContactSweep` family builds the same selected set for both
  `exact_computation_fallback` values, and the CPU
  `validate_cooperative_step` rule (`exact && use_ccd && !ticcd && !ogc`) only
  chooses between the exact stored-change test and the squared cutoff. The GPU
  therefore applies the cutoff to selected vertices in original mode with CCD
  on or off, and in robust exact mode with CCD disabled; exact + CCD on keeps
  representable small moves. Solid/mixed/rigid scenes stay `StoredNoop`. The
  selected-node mask is computed on the device from the current canonical
  colors, incidence counts and logical team, validated once per call, never
  applied globally, and never inherited by a predictor. When that cutoff is
  active, a non-`greedy` `IPC_GPU_COLORING` is rejected rather than assumed to
  reproduce the CPU canonical color order;
- both original predictors share the CPU false preparation: fixed target
  fraction, the raw intended displacement retained separately from the stored
  target, per-axis `max(1e-8, 0.1*|raw dx|, 64*eps*scale)` movement boxes (the
  raw displacement is required because `target-start` can round smaller), fixed
  `1e-8` gap, the CPU false projected graph certificate (not the robust outward
  certificate), and contact-only ascending greedy coloring without heavy-first
  reordering. The level-based predictor adds one discarded probe, all exact
  movement ties per component and BFS levels; the legacy predictor gains
  neither and spends its first sweep from the requested budget. Both stop after
  a joined whole sweep on no stored movement or every target squared error
  below `1e-28`, computed on the device. After each positive step the original
  predictor applies its per-pair `min(start gap, 1e-8)` + fixed 48
  tested-endpoint refinement; that floor is not applied to ordinary
  `OriginalSolver`, which has no positive separation floor;
- DAT false uses original contact assembly but keeps its own Jacobi increments,
  budgets and plane/cap/filter model.
- an original predictor request with a zero sweep budget is still a request:
  `StepParameters::colored_guess_enabled` (default `false`) lets a caller ask
  for the CPU validation of forced positions/targets without sweeps. It is
  appended metadata, not a geometry toggle, and direct API callers that leave
  it `false` keep the old behaviour.

Original mode supports deformable scenes. The direct `advance_substep` entry
rejects rigid bodies before advancing state; the generated CLI driver also
checks this scene restriction before creating output. Tight-Inclusion CCD,
OGC and the cloth grid remain unsupported in both modes.

The collision-only represented-path check is an extra physical safety layer,
not part of the CPU TOI model. It accepts an unchanged healthy positive-gap
endpoint and fails a truly zero initial gap, but it can clip slightly more than
the CPU false path; that additional clipping is intentional and observable in
the original-mode integration tests. A true initial finite-primitive
intersection maps to the explicit initial-contact device error (not an unsafe
backtrack or a silent freeze). The check supports at most one moving vertex;
multi-vertex motions such as a translating edge are rejected rather than
certified from rounded `end-start` equality.
Each check first asks the existing outward-rounded complete-trajectory
certificate for a proved zero-gap separation; a proof accepts the path, and an
inconclusive answer falls through to the unchanged exact predicate (no positive
floor, no `DBL_MIN`/denormal fake gap). Exact capacity is retried once from 256
to 2048 limbs only while the worker's arena is healthy. The bounded original
resolve allocates at most 16 reusable arenas of 4 MiB, independent of the scene
size.

## Known constraints

The ordinary device solver requires `--fixed_iters` and conflict-graph coloring;
the driver rejects unported modes before it touches the output directory. OGC
solving, the cloth grid and Tight-Inclusion CCD (`--use_ticcd true`) are not
available for ordinary solving, so use `--use_ticcd false`. The DAT path adds
its own restriction to deformable scenes. Colors execute sequentially, with
independent updates within each color running in parallel. Numerical fallback
and CCD must stay on the GPU; an unrecoverable device error stops the substep.

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
[contact_pipeline.cu](contact_pipeline.cu),
[contact_schedule.h](contact_schedule.h) and
[elastic.h](elastic.h).

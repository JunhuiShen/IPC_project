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
   rewrites the copied quoted `solver.h` include to the read-only CPU
   tree so the GPU `solver.h` header cannot shadow it.
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
| Contact pipeline | [contact_pipeline.cu](contact_pipeline.cu), [contact_record_index.cuh](contact_record_index.cuh): gathers, bounded packet addressing, ordinary evaluation, exact queues/retries and ordered assembly into local systems. |
| Contact arithmetic | [contact.cuh](contact.cuh), [contact_selected_ordinary.cuh](contact_selected_ordinary.cuh), [contact_exact.cuh](contact_exact.cuh), [contact_cooperative.cuh](contact_cooperative.cuh): ordinary derivatives and scalar/cooperative exact paths. [original_contact.cuh](original_contact.cuh): the qualified original (`exact_computation_fallback == false`) geometry/force evaluator. |
| Compact point contacts | [contact_compact_point.cu](contact_compact_point.cu), [cooperative_compact_point_study.cu](cooperative_compact_point_study.cu), [compact_point_prefix.cuh](compact_point_prefix.cuh), [direct_native_point.cuh](direct_native_point.cuh): guarded point-contact routes and fallback queues. |
| Local solve | [solve.cuh](solve.cuh): `solve3` and its rank-deficient fallback. |
| Linear CCD | [linear_pipeline.cu](linear_pipeline.cu), [ccd.cuh](ccd.cuh), [exact.cuh](exact.cuh), [direct_bound.cuh](direct_bound.cuh), [original_ccd.cuh](original_ccd.cuh): conservative clipping, exact predicates, the qualified original TOI resolver and optional exact step proposals. |
| Rigid updates | [rigid_pipeline.cu](rigid_pipeline.cu), [rigid_pins.cuh](rigid_pins.cuh), [rigid_ccd.cuh](rigid_ccd.cuh): translation/orientation proposals, material-point pins, curved-motion CCD and commits. |
| Initial guesses and movement boxes | [global_guess.cu](global_guess.cu), [adaptive_boxes.cu](adaptive_boxes.cu), [level_schedule.cu](level_schedule.cu), [original_guess.cuh](original_guess.cuh): predictor CCD, movement limits, optional trial/BFS level scheduling and the CPU false predictor certificate/floor helpers. |
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
cmake -S gpu -B gpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBoost_DIR=/path/to/boost-1.92.0/lib/cmake/Boost-1.92.0 \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/build --target 3D_sim_gpu -j 8
```

`IPC_GPU_FAST_DEV_BUILD` defaults to `ON`. It uses PTXAS O1 and explicitly
outlines scalar comparisons in the two largest linear CCD units. Integer-valued
operations retain their existing definitions; `--fmad=false` and exact-arithmetic
budgets are unchanged. Other CUDA units retain inline comparisons and Release
optimization. Set `-DIPC_GPU_FAST_DEV_BUILD=OFF` for inline comparisons and
PTXAS O3 in the linear CCD units.

The exact kernel families are compiled separately in `linear_resolve_*.cu`.
Each kernel keeps its allocator and device helpers in the same compilation
unit. Ninja compiles one GPU-target source at a time by default; the shared CPU
support libraries can still build in parallel. This limits overlapping compiler
working sets. A caller-supplied `CMAKE_JOB_POOL_COMPILE` overrides that default.

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
  --use_ccd true --exact_computation_fallback true \
  --use_ccd_guess false --use_colored_ccd_guess true \
  --colored_ccd_guess_iters 2 --write_substeps false \
  --format geo --outdir results/gpu_smoke
```

The driver prints `GPU device solver backend` after its preflight mode check
accepts the options, so a log without that line is not a device solve.

For full Examples 1 and 9, use the corresponding [scene command](../README.md#example-scene-commands)
with `./gpu/build/3D_sim_gpu` in place of `./build/3D_sim`. Keep the documented
physical settings, including `--exact_computation_fallback false` for both.
No `IPC_GPU_*` overrides are needed for the selected execution defaults. The
support and CPU/GPU numerical-policy constraints below still apply.

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

### CPU-only DAT and OGC modes

The optional CUDA Jacobi-DAT path has been retired. The shared `--use_dat`
selector now requests the CPU planar DAT solver, so both GPU scene setup and
step conversion reject it before any update. `--use_ogc` is CPU-only
as well. The GPU driver never substitutes a CPU solve for either request.

The ordinary GPU colored solver, its CCD paths, and rigid material-point pins
remain available.

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

The integrated color-local/rigid-compaction routes passed the corrected
236-check joint native test and combined memcheck, synccheck and initcheck
with zero reported errors. The native binary used the explicit environment
options listed below; the final local default literals were not rebuilt.

The joint contact suite includes a Robust movement-box pruning fixture. CTest
sets `IPC_GPU_BOX_CONTACT_PRUNING=1` for `gpu_contact_pipeline`, including through
`gpu/validate.py`. For direct runs of the contact executable or Compute Sanitizer,
set it before startup:

```sh
IPC_GPU_BOX_CONTACT_PRUNING=1 gpu/build/gpu_contact_pipeline_test
IPC_GPU_BOX_CONTACT_PRUNING=1 compute-sanitizer --tool memcheck gpu/build/gpu_contact_pipeline_test
```

Use the same setting for direct synccheck/initcheck runs. This is a test-process
setting; production `IPC_GPU_BOX_CONTACT_PRUNING` still defaults to `0`.

`gpu_validation` builds the ordinary rigid-pin parity targets (`gpu_rigid_pins`,
`gpu_rigid_pin_api`) together with the
component suites for elastic assembly, broad phase, coloring, contacts, local
solves, CCD, SDFs and exact arithmetic. The independent original-mode targets
`gpu_original_contact_test`, `gpu_original_ccd_test` and
`gpu_original_mode_test` compile the qualified original components and the
mode-routing/collision-only integration checks; they are registered with
CTest but stay out of the default ALL build.
`gpu_bridge_mode_host` checks ordinary settings conversion and rejection of the
CPU-only DAT selector without executing device code.

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
claims that they have passed. The launcher itself makes no performance claim.

The ordinary solver uses a reusable 262,144-pair workspace. Set
`IPC_GPU_CONTACT_PACKET_CAPACITY` to a positive integer through 1,048,576 to
trade packet memory for fewer launches; `65536` restores the previous capacity.
The direct `GpuContactPipeline()` constructor retains its 65,536-pair default. All candidate pairs remain in the device
lists; ordered accumulation and failure-before-publication checks are retained.
Original GS allocates no unused exact-contact arenas.

The rounding and ordinary rigid math tests also compile as ordinary C++ without
CUDA. These host checks cover numerical helpers; native CUDA compilation and
device execution are still required to qualify the kernels.

Maintained optimization tests join the explicit `gpu_validation` target. Their
host oracles, genuine public Solver regressions, private-authority facades and
CUDA evaluator checks have separate scopes. Run the complete registered suite
on a supported CUDA device after changing source or defaults:

```sh
cmake --build gpu/build --target gpu_validation -j 8
ctest --test-dir gpu/build -L gpu_solver --output-on-failure
```

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
  order. Eligible deformable sweeps use compact packets and prefetched ordered
  reduction by default; unsupported layouts retain the previous reduction.
  Selected-node and component reductions remain available. The robust
  selected-ordinary evaluator and cooperative exact queues retain their
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
  check. Ordinary pair queries run across active-node CTAs; uncertain queries
  enter a bounded queue with independent exact-worker arenas. Per-node minima
  and earliest CSR failures retain the original result. Queue overflow restores
  the original proposals and replays the complete color with whole-node
  workers. Current positions remain immutable until the joined color commits;
- linear CCD uses the qualified `original_ccd` fast query plus its bounded
  exact resolver. An original failure is never sent to the robust exact queue;
- the CPU original cooperative selected-node `1e-28` squared cutoff is modelled
  from `original_reference_threads` (`1` selects the noncooperative rule). This
  is scheduler-FAMILY metadata, not a geometry toggle: the cloth
  `ColoredContactSweep` family builds the same selected set for both
  `exact_computation_fallback` values, and the CPU
  `validate_cooperative_step` rule (`exact && use_ccd && !ogc`) only
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
- an original predictor request with a zero sweep budget is still a request:
  `StepParameters::colored_guess_enabled` (default `false`) lets a caller ask
  for the CPU validation of forced positions/targets without sweeps. It is
  appended metadata, not a geometry toggle, and direct API callers that leave
  it `false` keep the old behaviour.

Original mode also supports rigid and mixed scenes. Rigid translation uses
the original CPU CCD and raw-displacement commit rule; represented-path checks
may conservatively shorten a body's step. Rigid rotation retains curved-motion
CCD. All numerical checks run on the device. Per-vertex CCD uses the built-in
linear backend; the global predictor retains its Tight-Inclusion-compatible
device implementation. OGC and the cloth grid remain unsupported in both modes.

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
resolve defaults to 16 reusable arenas of 4 MiB. Its configurable limit is
256; a larger rigid fallback CTA width can raise the shared reservation.
The reservation is bounded independently of the scene size.

## Original node CCD partitioning

Original node CCD uses `IPC_GPU_NODE_CSR_PARTITION=1` by default. When an active
color has fewer node rows than available CTAs, it distributes each row's fast
queries across several CTAs in the existing grid, up to 1,024 CTAs of 128
threads. Larger active sets retain the previous row mapping. Set
`IPC_GPU_NODE_CSR_PARTITION=0` before creating a `GpuLinearClipper`, or in the
driver's environment, to restore the previous mapping. The option is captured
once per clipper.

This changes execution distribution only. `--exact_computation_fallback false`
continues to select the Original numerical path, including the full raw TOI
minimum, `0.9 * TOI` step, stored-displacement endpoint and represented-path
checks. Both raw and stored-path passes remain; no queries are skipped.
Precision, tolerances, exact retries, queue bounds, overflow replay, existing
backtracking budgets and commit order are unchanged. It adds no GPU allocation
and preserves the existing launch count. It does not select Robust CCD.

The native and independent integer-coverage tests exercise long/duplicated rows,
late TOI minima, ordered faults, exact retry, overflow/reuse and predictor rules.
Host tests do not establish CUDA correctness. Build and run the maintained checks:

```sh
cmake --build gpu/build --target gpu_node_csr_partition_native_test gpu_node_csr_partition_test -j 8
ctest --test-dir gpu/build -R '^gpu_node_csr_partition_(native|host)$' --output-on-failure
```

## Cooperative Original node controller

`IPC_GPU_NODE_CCD_COOPERATIVE` defaults to `1` for Original node batches. Set it
to `0` before constructing a `GpuLinearClipper` to use the fixed host schedule.
Owners capture the option once. Robust CCD and rigid clipping keep their routes.

An eligible color runs the existing prepare, fast-query, exact-worker, gate and
consume stages inside one cooperative grid. Every thread joins between stages;
joined completion ends idle rounds. Raw and represented/stored-path passes,
full TOI minima, `0.9 * TOI`, CSR failure ordering, exact arenas, retries,
65 trial candidates, queue-overflow replay and publication remain unchanged.

Eligibility uses actual kernel residency and node/worker coverage. Ineligible
plans use the fixed schedule. `IPC_GPU_NODE_CCD_ONE_WAVE` defaults to `1` and
bounds the grid while retaining those coverage requirements; `0` uses the full
resident grid. CUDA errors propagate. Device stack limits are not changed, and
no numerical arrays are downloaded for scheduling.

The native test retains the complete Original node fixture and resource-fallback
checks. Return code `77` reports fallback coverage only and is a CTest skip, not
cooperative qualification. The separate host test checks stage order, joined
completion and residency bounds. Both join `gpu_validation` and stay out of ALL.

```sh
cmake --build gpu/build --target gpu_node_cooperative_native_test gpu_node_controller_test -j 8
ctest --test-dir gpu/build -R '^gpu_(node_cooperative_native|node_controller_host)$' --output-on-failure
```

## Rigid proxy execution

`IPC_GPU_PARALLEL_RIGID_PROXIES` defaults on for owners constructed with
validated, uniquely owned rigid topology. The
solver uses that trusted path after scene validation. Set the option to `0`
before owner construction to restore the previous proxy loops; `1` selects the
parallel path explicitly. Owners capture the option once. Raw/default owners
and free-function callers retain the legacy route, including alias and invalid
input behaviour.

This distributes rigid proxy endpoint copies, trusted translation-trial proxy
writes, accepted rotation-position construction and final velocity updates
across CSR slots. Scalar COM checks and ordered error handling remain in place. It retains the
existing scalar formulas and copies the certified endpoints exactly. Original
`0.9 * TOI`, every raw and represented/stored-path query, curved-motion CCD,
precision and retry budgets are unchanged. It does not select Robust CCD or
change rigid boxes, angular caps or the CPU/GPU policy differences below.
Finite preflight precedes publication; finish/rotation faults replay the old
ordered path rather than publishing a partial parallel result.

No per-node scratch is added. The two owners lazily allocate one 4-byte fault
flag each; the rotation clipper's existing 112-byte context is unchanged. Proxy
launches use 128 threads and at most 65,535 CTAs, striding larger body sets.
Untrusted topology stays on the prior path.

The native proxy tests cover exact OFF/ON/public-oracle endpoints and status,
large/tail/scrambled/empty rows, mode masks, ordered faults, captured options
and a nondefault stream. The trusted-owner supplemental fixture and the
existing raw rigid-pipeline fixture remain separate. The two host-only tests
check integer slot coverage and proxy-value formulas against independent
oracles; host checks do not establish CUDA correctness. All four targets
join `gpu_validation`, stay out of ALL, and use the `gpu_solver` CTest label;
the host tests also carry `host_math`.

```sh
cmake --build gpu/build --target gpu_rigid_proxy_parallel_test gpu_rigid_pipeline_proxy_trusted_test gpu_rigid_proxy_index_test gpu_rigid_proxy_values_test -j 8
ctest --test-dir gpu/build -R '^gpu_(rigid_proxy_(parallel|index|values)|rigid_pipeline_proxy_trusted)$' --output-on-failure
```

## Known constraints

The ordinary device solver requires `--fixed_iters` and conflict-graph coloring;
the driver rejects unported modes before it touches the output directory. OGC
solving and the cloth grid are not available for ordinary solving. Colors execute
sequentially, with independent updates within each color running in parallel.
Numerical fallback and CCD must stay on the GPU; an unrecoverable device error stops the substep.

Original-mode rigid support has native regression coverage on NVIDIA RTX 6000
Ada with CUDA 12.6, including basic CPU parity, rigid/mixed scenes, tiny motion,
pins and collision checks. Those historical checks do not qualify the current
feature-retirement edits without a fresh CUDA build and device run.
CUDA needs a host thread for launch orchestration and I/O. Use a positive
`OMP_NUM_THREADS` (for example `1`); this does not move solver arithmetic to CPU.

Short CPU parity tests do not establish identical long trajectories. The CPU
and GPU use different BVH traversal/contact accumulation orders, and these
fixed-iteration contact examples can amplify roundoff-sized initial differences.
The current rigid GPU path also retains maximum-radius boxes, stored-orientation
anchors and an additional angular proposal cap. The supplied Example 9 command
(`--use_simd false`) selects the scalar CPU rigid solver, which uses history-based
boxes and anchors regenerated from its angular state. It always reconstructs
accepted rotation/proxy values. The tiny-angular-step publication rule belongs
to the SIMD CPU rigid/general backend and does not apply to that command.
In particular,
`theta_box_min` does not currently select a rigid GPU minimum radius.
Performance comparisons must state the numerical policy and distinguish a
warm continuation from a checkpoint restart, which resets movement histories.
The full current README Examples 1 and 9 failed the unchanged strict CPU
trajectory bounds. GPU-to-GPU byte preservation, saved-endpoint intersection
checks and seed-sensitivity controls have separate scopes and do not establish
full CPU trajectory parity.

## Runtime diagnostics

The execution options below default to `1`; set an option to `0` for its
previous route. They preserve geometry formulas, precision, candidate membership
and the original contact addition order.

| Option | Execution path |
| --- | --- |
| `IPC_GPU_ORIGINAL_NODE_CCD_BATCH` | Parallel ordinary node CCD with bounded exact queries and complete-color overflow fallback. |
| `IPC_GPU_NODE_CCD_COOPERATIVE` | Cooperative execution of the existing Original node stages when the actual resident grid covers nodes/workers; fixed scheduling otherwise. |
| `IPC_GPU_RIGID_CCD_BATCH` | Grid-wide rigid translation queries for a unique active body; deterministic minima and earliest failure, with scalar/CTA fallback. |
| `IPC_GPU_RIGID_ROTATION_BATCH` | Grid-wide curved-motion queries for a unique rotating body; other color layouts retain the previous clipper. |
| `IPC_GPU_PARALLEL_RIGID_PROXIES` | Parallel copies and velocity updates for trusted rigid proxy topology; raw/default owners retain the legacy route. |
| `IPC_GPU_RIGID_CONTACT_PULLBACK_CACHE` | Reuse per-node pullbacks and per-pair rigid terms within one assembly. |
| `IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION` | Omit redundant color-pair lists for validated two-body owners when complete rigid incidence is sufficient. |
| `IPC_GPU_TWO_RIGID_DENSE_CSR` | Use complete two-row dense incidence after device validation; raw, pruned and unsupported inputs keep generic construction. |
| `IPC_GPU_TWO_RIGID_KIND_PACKETS` | Omit wholly wrong-kind tiles after validated kind-prefix inspection; executed packet bounds and per-kind contribution order stay unchanged. |
| `IPC_GPU_RIGID_ONE_SLOT_TERMS` | Store one rigid term record per packet pair for proven unique selected ownership; other layouts retain four slots. |
| `IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER` | Evaluate geometry first for eligible zero-friction Original rigid pairs; full force evaluation retains every active/error pair and leaves raw/CSR/CCD membership intact. |
| `IPC_GPU_RIGID_COMPACT_REDUCE` | Stable active-row compaction before ordered tiled rigid accumulation, with the previous scan above 64 bodies or unsupported row sizes. |
| `IPC_GPU_CONTACT_OUTPUT_ACTIVITY` | Skip inactive output records. |
| `IPC_GPU_COLOR_LOCAL_PACKETS` | Recover compact/selected/warp/prefetch routes when the global contact list exceeds the packet cap but the indexed current-color list fits; larger colors retain global tiling. |
| `IPC_GPU_COMPACT_DEFORMABLE_CONTACTS` | Compact records for eligible untiled or color-local deformable sweeps; Frozen and other phases use full records. |
| `IPC_GPU_CONTACT_PREFETCH_REDUCE` | Prefetch ordered CSR records for selected deformable owners in non-Frozen sweeps, in Original or Robust mode; unsupported layouts use the existing reduction. |
| `IPC_GPU_SMALL_BLOCK_GRAPH` | Sorted bitset graph construction through 512 blocks; larger graphs retain the previous construction. |
| `IPC_GPU_CANONICAL_WARP_COLOR` | Warp execution of ascending greedy coloring with identical canonical labels. |

Color-local storage preserves the original global CSR/contact addition order
and validates contact kinds across the global list. Missing mappings fail closed;
unsupported colors retain the previous tiled path. Single force faults retain
exact codes and seed/SDF/pin precedence. With multiple independent force faults,
concurrent evaluation may select a different nonzero diagnostic than coarse
tile/kind ordering; any error still prevents publication. Legacy OFF arbitration
and CCD first-unsafe/error ordering are unchanged.

`IPC_GPU_ORIGINAL_CCD_WORKERS` selects 1 through 256 exact workers (default 16).
Each worker reserves 4 MiB of limbs plus allocator slots. Rigid fallback CTA
widths can raise that shared arena reservation; their default remains one
thread. Both batch queues default to 1,048,576 entries: 8 MiB for node/rank
pairs and 4 MiB for rigid ranks. Queue overflow replays the previous resolver
without dropping queries. These settings change execution and scratch memory,
not arithmetic capacities or CCD tolerances.

At the ordinary solver's default 262,144-pair capacity, full output records
occupy at most 354 MiB. The rigid-term cache uses 48 MiB for eligible trusted
two-body Original non-Frozen one-slot assembly, or up to 192 MiB for the complete four-owner layout.
The sparse Original force queue adds `4 * capacity + 4` bytes when eligible
(about 1 MiB at this cap). Stable rigid compaction adds about 2.016 MiB for two
bodies, or 64.5 MiB at its 64-body limit. Buffers grow lazily and retain their
high-water capacities. A 1,048,576-pair setting multiplies per-packet ceilings by
four; it does not bound raw candidate, incidence or topology storage.
Eligible compact deformable phases add 216 bytes per packet slot, at most
54 MiB at this cap; those records are explicitly released for Frozen or other
noncompact phases.

The color-local global-pair-to-slot map costs `4 * N` bytes for its retained
global contact capacity `N`, is included in `storage_bytes().packet_bytes`, and
is **not bounded by the packet cap**. Its allocation and the rigid kept-index/
count scratch survive fallback, empty and Frozen reuse. Lowering packet
capacity reduces the numerical packet/cache ceilings, not this global map or
existing incidence/topology memory.

Rigid movement boxes still use the GPU's existing maximum-radius policy. The
CPU's history-based radius shrink is a separate numerical policy; these
execution changes do not claim CPU box-policy equivalence.

`IPC_GPU_COLORING` selects `greedy` (default), `degree`, `spatial` or `cusparse`.
The ordered `degree` and `spatial` modes use warp-cooperative coloring with
shared marks when its palette fits the shared-memory budget; larger palettes
use global marks. `IPC_GPU_WARP_SHARED_MARKS=0` selects global marks, and
`IPC_GPU_WARP_ORDERED_COLOR=0` selects serial coloring with the same labels and
execution order. Scene bounds use parallel reduction, with the original serial
scan for exceptional coordinates. `IPC_GPU_PARALLEL_SCENE_BOUNDS=0` selects the
serial scan for all coordinates.

Set environment options before launching the executable. Options for contact
routing and worklists are parsed in
[contact_pipeline.cu](contact_pipeline.cu),
[contact_schedule.h](contact_schedule.h) and
[elastic.h](elastic.h).

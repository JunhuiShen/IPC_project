# CUDA cloth, solid, and rigid solvers

There are **two different executables** in this folder:

| Executable | Numerical work |
| --- | --- |
| `3D_sim_gpu_resident` | GPU-resident solver, including GPU exact fallbacks; current implementation |
| `3D_sim_gpu` | Earlier hybrid prototype; many numerical stages still run on CPU |

The CPU source was fast-forwarded to `71c7716573658` on October 10, 2026, and
the server's `../build/3D_sim` was rebuilt, as requested. GPU-specific changes
remain in this folder; they do not modify the production CPU algorithms.
Both GPU implementations are experimental, not production-validated.

## GPU-resident solver

`resident_solver.cu` keeps positions, velocities, rigid COMs/quaternions,
incident element records, collision candidates, and color groups on the device.
The numerical substep is:

1. Predict deformable targets and save previous deformable/rigid state.
2. Apply the selected initial guess entirely on GPU. The cloth-only default
   is the standard all-moving global CCD guess: Tight-Inclusion BFS queries,
   a global minimum TOI, and the CPU's 0.9 safety factor. The opt-in colored
   guess instead sweeps colors toward the original target. As on CPU, general
   solid/mixed scenes ignore the legacy global guess; their colored guess
   keeps rigid bodies fixed.
3. Rebuild blue/primitive boxes, BVHs, contact candidates, dependency graph and
   colors on GPU at the requested node-box rebuild interval.
4. For each color, gather current element/contact data, compute private
   derivatives, accumulate in incident order, solve, CCD-clip, and commit.
   Cloth membrane/bending, solid tetrahedra, points/pins/gravity/inertia,
   barrier contact, friction and plane/cylinder/sphere SDFs run on GPU.
5. A rigid body performs COM assembly/update/commit, then freshly gathers for
   its orientation assembly/update/commit. Its proxy vertices remain resident.
6. Update velocities on GPU; download a snapshot only for requested frame or
   substep output/checkpoints.

The CPU still loads meshes, prepares immutable rest data, supplies prescribed
pin/SDF motion, launches kernels, reads allocation counts/error status, and
writes files. It does **not** evaluate a numerical solver or CCD fallback for
the resident executable. This is a device-resident numerical solver, not a
claim that CUDA host launch/runtime calls or file I/O disappear.

### Exact fallback and safety

Linear CCD first tries conservative floating-point interval certificates.
Uncertain cases are queued for **GPU multiword integer/rational predicates**;
there is no CPU retry. Exact conversion starts from the original binary64
coordinates before subtraction. A 128-limb fast exact path retries with
2,048 limbs (65,536 bits), covering the documented degree/input-range bounds
in `resident_exact.cuh`. Contact geometry/derivatives similarly retry from
256 to 1,600 limbs. GPU scratch arenas avoid enormous per-thread stack frames.
Exact geometry, rational derivative terms and predicates do not make the whole
physics arbitrary precision: final square roots, logarithms, barrier/friction
values and solver state remain binary64.

Each linear trial checks all incident pairs at the **same proposed endpoint**.
The broad phase also protects the coordinate-scaled numerical gap independently
of `d_hat`: GPU reductions set conservative query padding without enlarging the
allowed movement boxes. Initial-guess boxes account for both start and target
coordinate scales, including when `d_hat` is zero.
Rejected proposals are halved; after 64 unsuccessful trials the unchanged
endpoint is itself certified. Nothing is accepted solely because a retry
budget expired. Accepted rounded coordinates are copied verbatim at commit.
Detected true zero-gap contact, invalid/nonfinite state, or allocation failure
invalidates the substep and prevents export/reuse of the partially updated
state; these are not triggers for a CPU fallback.

Rigid rotation uses the existing CPU algorithm's analytic curved-motion CCD,
ported to CUDA; it is neither a linear-chord substitute nor a new exact-rational
curved CCD algorithm. The exact fallback described above is for linear CCD and
contact geometry. Its tests do not establish a stronger formal guarantee for
the pre-existing rotational CCD algorithm.

### Build and validate on biot

Use the existing configured `gpu/build` on the server, or configure as below
(offline dependency settings are explained in the hybrid build section):

```sh
cmake -S gpu -B gpu/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build gpu/build --target gpu_resident_validation -j 8
ctest --test-dir gpu/build -R gpu_resident --output-on-failure
```

The CUDA build uses double precision, no fast math, and `--fmad=false`.
Twelve resident suites cover elastic kernels, BVH/graph completeness, contact
derivatives, rank-deficient small solves, rigid kinematics/curved CCD, SDFs,
exact linear CCD, each update pipeline and an integrated active-contact
cloth/solid/rigid fixture. The latter exercises four substeps with friction
0 and 0.1 and with the colored initial guess, checking finite state and positive
contact gaps. Exact tests include full-range integer retries and cases where
ordinary subtraction cannot resolve the geometry.

### Run the resident solver

For a small cloth smoke test, from the repository root:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
./gpu/build/3D_sim_gpu_resident \
  --example 1 --twist_nx 9 --twist_ny 9 --num_frames 2 --fps 30 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --fixed_iters --substeps 5 --max_substep_iters 2 \
  --node_box_update_count 1 --d_hat 0.005 --k_barrier 100 \
  --friction_coefficient 0 --use_ccd true --use_ticcd false \
  --exact_computation_fallback true --use_ccd_guess false \
  --use_colored_ccd_guess true --colored_ccd_guess_iters 2 \
  --write_substeps false --format geo --outdir outputs/gpu_resident_cloth_NEW
```

Use a new output directory: the resident driver refuses to replace an existing
directory for a fresh run. OMP does not control CUDA worker count. CPU SIMD,
experimental-v2 and CPU scheduling flags do not select GPU kernels; choosing
the resident executable does. Supported material/scene/output parameters remain
those of the project driver, including cloth, solid, rigid and mixed scenes.

The root README's full Example 1 command is supported with the resident
executable substituted and a fresh output directory. Its default
`--use_ccd_guess true` / `--use_colored_ccd_guess false` combination is no
longer replaced by a colored guess. GEO output uses the shared driver writer.

Fixed iterations and conflict-graph coloring are currently required. Grid/OGC,
the `--use_ticcd true` per-vertex backend, disabling exact fallback, and
Verlet/translation initial guesses are explicitly rejected rather than routed
to CPU. This per-vertex backend limitation does not apply to the standard
all-moving initial guess, whose Tight-Inclusion algorithm is ported to CUDA.
As in the general CPU solver, the particle
`use_ccd` switch does not disable rigid CCD or CCD inside the colored guess.
It also does not disable barrier energy; that is controlled by contact parameters.

### Current limitations and interpretation

- This remains experimental, **not a demonstrated speedup over CPU**. Inactive
  contact rejection and parallel exact-contact queues removed the previous
  dominant bottleneck in a two-frame test; greedy coloring remains serial and
  per-color launches/whole-array filtering can be expensive. That short test
  used a colored initial guess and is not the README-default full-run result.
- Deformable node boxes now use the CPU adaptive rule:
  `clamp(1.2 * max(previous_solver_displacement, norm(v) * dt), min, max)`.
  History persists across substeps and excludes initial-guess displacement.
  Radii stay fixed within a solve; rebuilds recenter them. Rigid COM boxes and
  angular caps still use fixed `node_box_max`/`theta_box_max` values.
- Certified linear backtracking is not the CPU earliest-TOI-times-0.9 step
  selection. Rotational clipping retains its 0.9 factor. Together with different
  coloring and floating-point order, this means no byte-identical trajectory or
  identical convergence-rate claim.
- Passing kernel and short mixed-scene tests is not full validation of long
  production scenes. In particular, a completed full 200/240-frame resident
  benchmark is still required; the initial two-frame result cannot predict it.

### README-compatible global initial guess

`resident_global_guess.cuh/.cu` ports the CPU standard predictor's all-moving
Tight-Inclusion breadth-first queries: minimum separation `1e-10`, geometric
tolerance `1e-6`, maximum refinement count `1000000`, and zero TOI allowed.
It uses the same reconstructed motion endpoints and swept-box padding as
`ccd_initial_guess`, then reduces the minimum collision time on the GPU and
applies `omega = toi >= 1 ? 1 : 0.9 * toi` to the entire cloth.

Per-query queues grow through explicit device retries rather than dropping
unresolved cells. Host orchestration reads only counts/error status, not query
geometry or TOIs. Arithmetic/depth/resource failures stop before committing
the guess. This ports the CPU predictor's existing interval algorithm; it does
not replace the separate one-moving-vertex exact safety checks in solver sweeps.

`gpu_resident_global_guess_test` compares all-moving point/triangle and
edge/edge queries against the CPU library, including known crossings,
degeneracies, and a wide-frontier case requiring queue growth. Adaptive radius
history is checked independently by `gpu_resident_adaptive_boxes_test`.

Server validation for this port (October 10, 2026): all 17 resident tests
passed. The standard-guess test matched the CPU TOI exactly on all 541
fixtures, including the 262,144-entry queue-growth stress case. CUDA memcheck
reported zero errors for adaptive boxes, the integrated solver, and the
540-case `--quick` global-guess test. Quick mode deliberately omits the wide
frontier and uses nonparallel whole-scene test geometry; the full stress test
was checked without memcheck. A two-frame full-resolution README smoke run
also completed and its GEO meshes passed topology/finite-coordinate checks.
These are correctness checks, not the completed 240-frame performance result.

The full comparison uses `wait_example1_comparison.py --readme-command
--frames 240 --cpu-first`, which verifies the root README argument list and
uses identical 64-thread OpenMP settings for both applications. Only the
executable and output directories differ. Geometry and timing are downloaded
by `download_example1_when_ready.py`; `report_example1_full.py` reports a
speed ratio only after both complete runs have passed output validation.

### All-frame stage profiling (same numerical logic)

Set `IPC_GPU_PROFILE_CSV` to a fresh CSV path to enable resident-solver timing.
The profiler records every completed frame, with hierarchical host and CUDA
stream intervals, direct-child-subtracted exclusive times, candidate/color
counts, and global-guess queue retries. It pools events and resolves them after
the existing substep synchronization, rather than synchronizing every kernel.
Contact gathering and fast evaluation remain a fused kernel and are labeled
as such. No numerical kernels or launch parameters are changed by these hooks.

`profile_example1_full.py` runs all 240 frames using the root README arguments,
after an idle-server check, and preserves the accepted CPU baseline. Its
`analyze_example1_full_profile.py` report includes all individual frames,
60-frame chronological ranges, and candidate-count quartiles. Candidate counts
are broadphase work proxies, not counts of active contacts. It saves SHA-256
hashes for every GEO and binary checkpoint, providing a full-precision reference
for later optimization checks. `download_example1_profile.py` downloads the
reports and GEO files; binary checkpoints remain on the server with their hashes.

CUDA event intervals include host-fed stream gaps, not just active kernels.
Host and CUDA intervals overlap: never add them together. Inclusive stage times
contain their children; sum exclusive times only. Instrumentation overhead is
reported but must not be subtracted as an exact correction to clean runtime.
The regular printed solver timer also includes final profiling CSV output.

Before the full profile, `validate_example1_profile.py` confirmed byte-identical
GEO and binary checkpoints between the preserved GPU executable, the instrumented
executable with profiling disabled, and profiling enabled: frames 1–2, plus
contact-heavy frame 111 restarted from the same frame-110 checkpoint. These are
correctness tests, not a completed full-run optimization result. CPU source and
its production executable are unchanged.

### Exact-contact optimization and validation

#### Result-preserving contact/worklist trial (October 11)

Three additional execution switches remain opt-in:

- `IPC_GPU_COMPACT_DEFORMABLE_CONTACTS=1`: stores the selected deformable
  role's gradient, self-Hessian and relative friction derivatives in a
  216-byte hot record rather than the 1,416-byte full contact record. Packing
  does no arithmetic; accumulation order is unchanged. Rigid and multiple-role
  masks retain the full record. This path bypasses the original full-record
  warp reducer; the round-4 compact-aware experiment below is separate.
- `IPC_GPU_BOX_CONTACT_PRUNING=1`: after graph coloring, removes incidences
  certified separated over their entire blue boxes by more than both `d_hat`
  and the coordinate-scaled CCD gap floor. The original graph/color IDs stay
  unchanged. Certificates are invalidated at every rebuild and reused by
  linear CCD. Initially only all-deformable contacts qualify; rigid arcs are
  deliberately excluded. Ordinary solver steps without CCD do not prune.
- `IPC_GPU_ELASTIC_COLOR_WORKLISTS=1`: GPU-built stable node/incident lists
  limit gather, compute, reduction, solve and commit launches to the current
  color. Arrays and per-node accumulation order remain unchanged. Optional
  `IPC_GPU_ELASTIC_NODE_THREADS` / `IPC_GPU_ELASTIC_ELEMENT_THREADS` default
  to 1 / 8, respectively, to spread small colors across multiprocessors.

These switches do **not** weaken exact-fallback thresholds, change precision,
or replace the existing iterative CCD certification/halving rule with CPU TOI
clipping. CPU and GPU numerical trajectories are not claimed identical.
New certificate, compact-record, pruning, and elastic-worklist tests augment
the existing resident tests. `validate_contact_round3.py` gates repeated clean
early/middle/late trials and an optional continuous 240-frame checkpoint-byte
comparison on a successful, hash-bound isolated build/test record. A faster
short trial alone is not a measured full-simulation speedup.

The isolated round-3 build completed all 240 README Example 1 frames with
903.817696 solver seconds versus the prior 1066.265610 seconds (1.179735x),
and all 482 GEO/BIN files matched. This is GPU-to-GPU identity, not CPU parity.

#### Round-4 execution experiments (opt-in)

- `IPC_GPU_CONTACT_SELECTED_NODE_REDUCE=1` launches the original scalar
  contact accumulation only for blocks of the current color.
  `IPC_GPU_CONTACT_SELECTED_NODE_THREADS` accepts powers of two from 1 to 128
  (default 8). Mapping and launch counts refresh on rebuild; arithmetic is
  unchanged. The feature itself defaults to 0.
- `IPC_GPU_COMPACT_WARP_REDUCE=1` assigns independent gradient/Hessian
  components of high-degree deformable nodes to lanes while retaining each
  component's original incident/role/addition order. It requires compact
  contacts, uses selected scalar reduction below
  `IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE` (default 32, positive integer),
  and defaults to 0. It is not a tree summation.
- `IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS=1` distributes the seven exact
  residual/weight conversions across private lane arenas before Hessian
  preparation. The default is 0; scalar replay and large-capacity retry remain.
- `IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY=1` skips unused graph/color work in
  the global initial guess, retaining identical candidates and their order.
  It defaults to 0. The ordinary solver and colored guess still build graphs.
- `IPC_GPU_GLOBAL_GUESS_WORKERS` (1..8192, default 512) and
  `IPC_GPU_GLOBAL_GUESS_THREADS` (1/2/4/8/16/32, default 32) vary independent
  query scheduling. The 64 MiB scratch budget still caps effective workers;
  heap operations, tolerances and capacity retries do not change.

`finish_contact_round4.py` builds an isolated snapshot, executes regression
and sanitizer tests, and seals their evidence. `experiment_contact_round4.py`
then compares individual configurations on early/middle/late cold restarts,
confirms selected combinations with two alternating-order repetitions, and
runs a continuous 240-frame byte/timing check only for a confirmed improvement.
Phase profiles are separate from clean timing. Its explicit monitoring policy
permits routine editor I/O while still rejecting competing jobs, transfers,
swapping and excessive external CPU work. Neither script pushes Git changes
or replaces production executables.

The completed round-4 trial measured **752.128292 solver seconds** over all
240 README Example 1 frames, versus round 3's 903.817696 seconds: **1.201680x
faster, or 16.8% less solver time**. Average solver time was 3.133868 seconds
per frame; total simulation time was 755.691938 seconds. All 482 GEO/BIN files
were byte-identical to the previous GPU reference. The 26 scheduled resident
regression tests, global-guess quick test, contact/cooperative memory checks,
and global-guess initialization check passed. The quick global-guess suite
does not include the wide-frontier stress fixture.

The winning configuration retained the round-3 execution settings and enabled
only the global-guess changes from this round:

```sh
export IPC_GPU_COMPACT_DEFORMABLE_CONTACTS=1
export IPC_GPU_BOX_CONTACT_PRUNING=1
export IPC_GPU_ELASTIC_COLOR_WORKLISTS=1
export IPC_GPU_COLOR_PAIR_QUEUES=1
export IPC_GPU_WARP_CONTACT_REDUCE=0
export IPC_GPU_COOPERATIVE_EXACT_CONTACT=1
export IPC_GPU_CONTACT_EXACT_SMALL_WORKERS=1024
export IPC_GPU_CONTACT_EXACT_SMALL_THREADS=1
export IPC_GPU_CONTACT_EXACT_LARGE_WORKERS=256
export IPC_GPU_CONTACT_EXACT_LARGE_THREADS=4
export IPC_GPU_CONTACT_SELECTED_NODE_REDUCE=0
export IPC_GPU_COMPACT_WARP_REDUCE=0
export IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS=0
export IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY=1
export IPC_GPU_GLOBAL_GUESS_WORKERS=8192
export IPC_GPU_GLOBAL_GUESS_THREADS=32
```

These are opt-in run settings, not changed defaults. Selected-node/compact-warp
reduction and parallel exact conversions were not selected for the full run.
The GPU remains 20.30x slower than the preserved 37.049037-second, 64-thread
CPU baseline; GPU-to-GPU byte identity does not establish CPU/GPU trajectory
identity or comparable numerical work.

The exact rational-to-binary64 conversion reuses its largest shifted integer
denominator and shifts it right exactly between quotient bits. It retains the
same subtraction, rounding-to-nearest/ties-to-even, capacity failures, and GPU
fallback. It does not lower precision or change contact/CCD thresholds.
`gpu_resident_ratio_host` and `gpu_resident_ratio_cuda` compare the old and new
implementations, including boundary rounding and device allocator failures.

Exact-contact launch scheduling can be tuned once per pipeline through
`IPC_GPU_CONTACT_EXACT_SMALL_WORKERS` (default 1024, maximum 4096),
`IPC_GPU_CONTACT_EXACT_SMALL_THREADS` (candidate default 1),
`IPC_GPU_CONTACT_EXACT_LARGE_WORKERS` (default 256, maximum 1024), and
`IPC_GPU_CONTACT_EXACT_LARGE_THREADS` (default 4). Threads per block must be
1, 2, 4, 8, 16, or 32, and divide the worker count. Both passes retain their
256- and 1600-limb capacities. The ordered per-node contact accumulation is
unchanged. Larger worker populations require more GPU scratch memory.

`IPC_GPU_CONTACT_QUEUE_CSV=/fresh/path.csv` enables per-color exact/retry queue
counts; it deliberately synchronizes for diagnostics and must be OFF for timing.
`tune_example1_contact.py` runs clean, idle-guarded early/middle/late trials against
a preserved executable. It compares both GEO and binary checkpoints against
matching cold restarts, not against continuous-run checkpoint hashes.
`validate_optimized_example1_full.py` runs the complete README simulation with
timers disabled and compares every output checkpoint against the original full
GPU manifest. Short-trial speedups are not full-simulation speedup claims.

### CPU update and matching GPU changes — October 10, 2026

The CPU base is now `71c7716573658789867ada8078f1342df681c026`
(`optimize contact geometry and evaluation and helper scheduling`). The local
and server source trees were fast-forwarded; the server CPU executable was
rebuilt. The existing server-only GoogleTest compatibility edit was preserved.

The resident port carries over the applicable work-reduction ideas:

- Contact evaluation computes only the Hessian blocks consumed by the active
  update. Deformable nodes need their self block; rigid point/triangle COM
  updates need the point `H00`; rigid orientation needs the selected upper
  cross blocks. Gradients and relative friction retain their existing formulas.
- Exact contact evaluation reuses per-endpoint geometric derivatives within
  the current contact. It does not cache geometry across state changes.
- Rigid orientation assembly prepares each needed proxy's Jacobian/curvature
  once, on GPU, and reuses it for its incident contacts and SDF contributions.
  This storage is refreshed for every orientation assembly, after COM commits.
- Independent initial endpoint certificates run per contact on GPU, so a
  contact-heavy deformable node can use multiple CUDA workers. Only proven-safe
  certificates are reused at the identical first endpoint. Backtracking
  invalidates that reuse and checks every incident pair again. The exact cold
  path still runs on GPU; its arithmetic and gap thresholds are unchanged.
- GPU-generated per-color stage masks suppress empty deformable/COM/orientation
  launches. Fixed bodies remain in collision detection and graph construction;
  soft-pinned deformable nodes remain updateable.

CPU OpenMP helper counts, chunk sizes and thread-local scratch pools are not
copied literally to CUDA. The resident solver already retains its device
allocation capacity and uses a single unified dependency graph. The CPU's new
rigid-only driver API and Example 15 scene dispatch are inherited through the
checked driver generator. `check_overlay.cmake` also verifies that the generated
resident driver contains no CPU `advance_one_frame*` calls.

After building, the correctness-only runner is:

```sh
python3 gpu/validate_resident_update.py --output NEW_VALIDATION_DIRECTORY
```

It checks all resident suites, rigid kernel compatibility, CUDA memory and
uninitialized-memory checks, and small Example 1 / Example 15 driver runs.
The Example 15 smoke test explicitly reduces the 200-sheet scene to two tiny
sheets; it is not validation of the full production-size scene. This update
does not establish a CPU/GPU speedup or byte-identical trajectories.

Validation on the server passed: 262 CPU regression tests (including the long
CCD integration test), all 12 resident GPU suites, the shared rigid-kernel
suite, four CUDA memory checks and two uninitialized-memory checks (zero
errors). The two small driver runs completed with finite output meshes.
The CPU tests used the system C++ runtime where the older Conda runtime could
not load the rebuilt test binary; no global runtime settings were changed.
The report, commands, executable hashes, logs and smoke meshes are in
[`validation_update_71c7716_20261010.bnurUZ/report.json`](validation_update_71c7716_20261010.bnurUZ/report.json),
mirrored locally and on the server. The old performance queue remains stopped;
these validation timings are not benchmark results.

### Resident verification — October 9, 2026

The final server build passed all 12 resident suites. CUDA compute-sanitizer
reported **zero errors** for exact CCD, exact contact and the integrated mixed
solver; the mixed solver also passed uninitialized-memory checking. The actual
resident driver completed a two-frame, 100-vertex Example 1 smoke run with the
colored CCD guess. These are correctness checks, not performance measurements:
another user's CPU job and our GPU sanitizers were active during portions of
validation. Production `build/3D_sim` retained its original SHA-256 hash.

The machine-readable report, complete logs and small smoke-run geometry are in
[`validation_resident_20261009.H43ECK/report.json`](validation_resident_20261009.H43ECK/report.json),
mirrored on the server and locally. No Git push was performed.

## Earlier hybrid prototype (historical results below)

This is an **experimental hybrid CPU/GPU implementation**, not a complete
GPU-resident simulator. Production CPU files and `../build/3D_sim` are unchanged.
The starting CPU commit is `eb16601aafddef90b8db62186e66f1047f613c9b`.

## Implemented scope

- GPU, double precision: cloth membrane (including 2x2 polar/eigen preparation),
  bending PSD derivatives, inertia, gravity, and pins; solid tetrahedral
  elasticity including signed 3x3 QR-SVD/polar preparation and PSD PBGS blocks;
  rigid translation/rotation inertia and contact/friction generalized-coordinate
  pullbacks (COM/omega gradients, Hessians, mixed blocks, quaternion curvature).
- CPU: mesh gathering, ordered incident-element accumulation, SDF and deformable
  contact/friction; rigid closest-feature geometry, Cartesian barrier/friction
  derivatives, quaternion kinematics and frozen-friction preparation;
  small node/body solves, safe-step/CCD (including exact fallbacks),
  BVH, coloring, initial guesses, integration, and output.
- Whole-color GPU batches preserve conflict-graph Gauss–Seidel color order.
  Vertices update only after all that color's GPU derivatives have returned.
- Cloth-only, solid-only, rigid-only, and mixed general-v2 routing are supported.
  Grid coloring, OGC, and serial driver selection are explicitly rejected.

This stage does **not** promise speedup or byte-identical CPU trajectories.
Host/device transfers, CPU gather/accumulation, and small colors can dominate.
The existing cooperative CPU contact/CCD sweep is retained: a collective
before-color hook submits GPU derivatives, then the same CPU helpers and
acceptance rules resume. Other modes use whole-color device compute followed
by the existing independent CPU per-vertex update/safe-step functions.

For mixed scenes, `general_bridge.cpp` snapshots all deformable nodes in a
color, evaluates point/triangle/tet/hinge records on CUDA, then the existing
general batch scheduler handles contact and node updates. It retains CPU
cooperative helpers, but starts an OpenMP team per color. Rigid bodies preserve
the original **COM update/commit, then orientation update/commit** sequence;
each body phase regathers current geometry before CUDA contact pullback. The
rigid contact organizer uses a whole-body CUDA batch instead of CPU SIMD contact
tiles. Its robust host geometry preparation is currently serial per body, and
the old CPU contact-helper sharing is not used for that GPU batch (CCD helpers
are unchanged). This is a correctness-first extension, not a tuned GPU solver.

The CUDA workspace/stream is serialized with a mutex across submitting host
threads. Rigid inertia currently launches once per body phase, even for one
record; this has significant overhead. No hidden CPU fallback is used for any
of the GPU derivative families listed above.

## Following SIMDIFICATION_PLAN.md

`color_bridge.h` gathers per-incident-element AoS snapshots from a color, then
`kernels.cu` computes private AoS derivatives, then the original host arithmetic
accumulates contributions in incident order. Each CUDA thread handles one record.
There are no per-vertex device atomics or full-mesh SoA arrays. Tail records are
bounds checked. Allocations are reused between colors.

This starts with the plan's simple thread-per-element option. It deliberately
uses neither shared-memory staging nor a tile transpose yet. Adjacent AoS
records **do not imply optimal field-wise coalescing** for large structs; this
must be measured before choosing a padded/shared-memory or warp-shuffle tile.
See NVIDIA's [memory-access explanation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

The CPU reference's eigenvalue floor (`1e-12`), unclamped metric inverse, and
bending PSD model are preserved. Nonfinite device results raise an error before
the color updates. CUDA uses no fast math or implicit FMA contraction; explicit
FMA expressions retain their specified operation. Numerical comparisons, rather
than cross-device bitwise equality, validate this implementation.

## Build on biot

CUDA 12.6, GCC 11, RTX 6000 Ada (SM 89):

```sh
cmake -S gpu -B gpu/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build gpu/build --target \
  3D_sim_gpu 3D_sim_cpu_control \
  gpu_kernel_test gpu_solid_kernel_test gpu_rigid_kernel_test \
  gpu_contact_fixture cpu_contact_fixture gpu_mixed_fixture cpu_mixed_fixture -j 8
ctest --test-dir gpu/build --output-on-failure
python3 gpu/validate.py --threads 8 --full-size --scene-smoke
```

For offline configuration the server build additionally points
`FETCHCONTENT_SOURCE_DIR_EIGEN3` and `FETCHCONTENT_SOURCE_DIR_TIGHT_INCLUSION`
at private copies in `gpu/deps/`. Do not share the production Tight-Inclusion
source directory: its configuration writes a header into that directory.

CMake reads the CPU source and makes a checked solver/driver overlay **only
inside `gpu/build/`**. If insertion anchors change, configuration fails for
review. Reconfigure/revalidate after an upstream change. No production source
patch, CUDA installation, Git commit, or push is performed by this build.

## Run

From the CPU repository root, with a **new output directory** (the existing
driver replaces output files on a fresh run):

```sh
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
./gpu/build/3D_sim_gpu \
  --example 1 --num_frames 2 --fps 30 \
  --E 115000 --nu 0.25 --kB 0.009 --kpin 1e9 --twist_rate 0.5 \
  --fixed_iters --max_substep_iters 10 --substeps 5 \
  --node_box_update_count 10 --d_hat 0.005 --k_barrier 100 \
  --friction_coefficient 0 --use_ccd true \
  --use_parallel true --use_basic_experimental true --use_simd true \
  --write_substeps false --format obj --outdir outputs/gpu_example1_NEW
```

OMP controls the remaining CPU work, not the number of GPU threads. The CUDA
block width is 128, with bounds-checked record counts. Final output includes GPU
device name, successful launch/record counts, and CUDA-event upload/compute/
download durations; these are not total solver time.

## Verification

`gpu_kernel_test` compares point, membrane, and bending gradients/Hessians with
the CPU SIMD reference at counts 0, 1, 31, 32, 33, 127, 128, 129. It covers pins,
all hinge roles, thin and degenerate geometry, disabled terms, tail canaries,
nonfinite rejection/recovery, and device counters. Regular tolerances are
`2e-9 + 2e-9 * max(abs(cpu),abs(gpu))`; thin-case tolerances are explicitly
reported by the test. Those tolerances are tests, not a promise about arbitrary
long nonlinear trajectories.

`3D_sim_cpu_control` uses the same host flags/core/dependencies but the original
CPU solver. Integration comparisons must report position errors and successful
CUDA calls as well as runtime, and must cover contact-heavy trajectories before
production use. The separate resident implementation above now covers device
gather/reduction, contact, small solves and device-only exact fallbacks; the
hybrid executable described in this section has not changed into that solver.

`contact_fixture.cpp` supplies two bent cloth sheets with an initial gap below
`d_hat`, opposing velocities, pins, bending, and actual active normal barriers.
It runs four substeps with friction 0 and 0.1 in separate matched CPU/GPU
executables. The validation runner compares all final vertex positions and
records contact counts/gaps. This does not replace long full-scene validation.

`gpu_solid_kernel_test` compares 36,576 components against scalar volumetric
derivatives, including all four node roles, signed inversions, rank-deficient
tets, material switches, and launch tails. `gpu_rigid_kernel_test` covers
translation/orientation/full/gradient-only modes, both primitive sides, body
update masks, friction, near-degenerate contacts, current COM snapshots,
whole-body batching, and inertia. Their optional host-arithmetic test defines
are diagnostic only; host results explicitly do **not** validate CUDA.

`mixed_fixture.cpp` includes cloth with bending, a four-tet solid with an
interior node, and a rigid triangle. All three cross-object contact types are
active. It compares positions, velocities, COMs, quaternions, and omega over
four substeps, with friction 0 and 0.1, while requiring positive CUDA counters
for every derivative family. `validate.py` also runs device memcheck for all
three standalone derivative tests. Passing these tests is not a claim of long
trajectory equivalence or byte-identical CPU/GPU results.

## Initial cloth-only server results (October 9, 2026)

- Device-vs-CPU SIMD reference: 75,948 derivative components passed; CUDA
  compute-sanitizer memcheck reported zero errors.
- 12x12-subdivision cloth, three frames: collision off/on, friction 0.1, and
  bending disabled passed; maximum coordinate difference `4.44e-16`.
- 10,000-vertex Example 1, two frames, five substeps, ten iterations:
  collision-off max coordinate difference `5.77e-15`; collision-on `8.88e-16`.
- Active-contact two-sheet fixture, four substeps: friction 0 and 0.1 passed;
  maximum coordinate difference `5.55e-17` in both cases.
- One short, eight-CPU-thread timing sample (not a benchmark): collision off
  CPU `0.283 s`, hybrid `0.998 s`; collision on CPU `0.919 s`, hybrid `2.575 s`.
  The hybrid is currently slower. CUDA kernel time was only `20.6 ms` / `53.0 ms`
  respectively; transfers plus remaining host work dominate. In the cooperative
  contact prepass, nested host gather/unpack loops currently execute serially
  under the submitting worker with the default OpenMP nesting policy.

Reports are in the server's `gpu/validation_*/report.json`; source and tests
are mirrored in the local `gpu/` folder. No result claims byte identity or
production readiness, and no Git push has been performed.

## All-three extension verification (October 9, 2026)

The actual CUDA build passes all three kernel tests on the RTX 6000 Ada:
75,948 point/cloth components, 36,576 solid components, and 455,652 rigid
components. All three compute-sanitizer memchecks pass. The mixed fixture
executes every CUDA family, with maximum position error `5.00e-16` without
friction and `5.55e-17` with friction 0.1; maximum velocity error is `3.34e-15`.
These results cover four substeps, not a long nonlinear trajectory.

The application-driver checks also pass: one frame of mixed Example 10
(`15` substeps, `2` iterations, friction `0.1`) differs by at most `2.18e-14`
in position; one frame of rigid-only Example 9 (`10` substeps, `2` iterations)
differs by at most `7.84e-15`. The Example 10 first frame has no active rigid
contacts, so the active mixed-contact fixture above is an essential extra test.

Single eight-host-thread timing samples, not repeated/warmed benchmarks:

| Scene / one frame | CPU control solver | Hybrid GPU solver |
| --- | ---: | ---: |
| Example 10, 15×2 | 2.566 s | 3.164 s |
| Example 9, 10×2 | 109.860 s | 19.765 s |

These are complete solver times, including CPU work, transfers and scheduler
differences—not isolated GPU compute speedups. Performance is scene-dependent;
the mixed scene is slower in this sample. Full report and logs:
[`validation_8so1zypq/report.json`](validation_8so1zypq/report.json), mirrored
locally. Simulation geometry outputs remain in that server validation folder.

For an existing cloth, solid, rigid, or mixed command, replace `./build/3D_sim`
with `./gpu/build/3D_sim_gpu`, retain `--use_parallel true
--use_basic_experimental true --use_simd true`, and choose a **new** output
directory. Keep normal scene parameters; GPU support does not remove contact,
time-step, or convergence requirements. Use the CPU production executable for
unsupported grid/OGC modes. This extension does not promise a general speedup.

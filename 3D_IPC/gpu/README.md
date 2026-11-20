# GPU solvers for 3D IPC

There are three GPU solvers: **reference DAT**, **DAT with our energies**, and
**our Gauss–Seidel (GS) solver**. All use Newton's GPU infrastructure and
NVIDIA Warp, which compiles the simulation kernels to CUDA.

## The three solvers

| Solver | Select with | Energies and contact | Supported scenes in this driver |
|---|---|---|---|
| Reference DAT | `--solver dat-reference` | Newton's original material and contact laws | Shared Newton twisting-cloth example |
| DAT with our energies | `--solver dat` | Our elasticity, bending, pins, IPC/friction and SDF terms | Cloth and deformable solids; also the shared cloth example |
| Our GS solver (default) | `--solver ipc` | Our elasticity, bending, pins, IPC/friction and SDF terms | Cloth, solids, rigid bodies and mixed scenes |

**Reference DAT** runs the pinned upstream `SolverVBD` with its original
energies. **DAT with our energies** keeps that solver's inertial predictor by default,
color order, collision schedule, local solve and Divide and Truncate (DAT)
motion limits, while substituting our energy evaluations. This lets us compare
the energy models within the same DAT algorithm. An explicit level-based CCD
initial-guess option replaces only prediction. A separate `--color-ccd` option
replaces DAT motion limits with our CCD for actual VBD color updates.
Add `--color-ccd-contact-forces` to assemble IPC barrier and friction forces in
that path; without this flag, its contact-force ablation stays unchanged.

**Our GS solver** uses our local updates and continuous collision detection
(CCD) to accept motion. It inherits Newton's timestep and structural color
loop: colors run in order, and vertices within a color are processed in
parallel. Rigid bodies use our rigid GS implementation.

The two DAT variants load the pristine
[newton_reference](third_party/newton_reference) snapshot. GS loads the
[newton](third_party/newton) fork with our physics hooks. Run different
snapshots in separate processes. The older C++/CUDA engine in
[reference/](reference/README.md) is separate from `--solver dat-reference`.

## Source guide

Start with [run.py](run.py): `main()` selects a solver, and `FrameGraph`
advances our scene examples through substeps and CUDA Graph replay. The shared
Newton cloth example uses its own frame loop. Then follow one of these paths:

| Solver | Start here | Follow into |
|---|---|---|
| Reference DAT | [`SolverDATReference`](__init__.py), a lazy direct alias to upstream `SolverVBD` | [solver_vbd.py](third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py) for the timestep and color loop; [particle_vbd_kernels.py](third_party/newton_reference/newton/_src/solvers/vbd/particle_vbd_kernels.py) for local solves and DAT truncation |
| DAT with our energies | [solver_dat.py](solver_dat.py), which substitutes energy calls in the upstream particle loop | [dat_energies.py](dat_energies.py) for materials, local energy assembly and IPC/friction on Newton's contact rows |
| Our GS solver | [solver.py](solver.py), which supplies our particle update and acceptance hooks | [solver.py](solver.py) for local assembly, schedules and GS stages; [ccd.py](ccd.py) for motion checks; [policies.py](policies.py) for CCD initial guesses; [rigid.py](rigid.py) for rigid GS |

For our scene examples, data flows through:

```text
Native scene definitions (../example.cpp)
    -> export_scene.cpp       scene/rest data and motion descriptions
    -> scene.py               Newton Model + our energy data
    -> run.py / FrameGraph    motion.py, substeps and solver.step()
    -> output.py              meshes and restart checkpoints
```

Within a substep, prediction and collision detection establish the working
state, then the solver visits vertex color groups. Each color accumulates
forces and Hessians, solves local systems, and limits motion through DAT or
CCD. Velocities are updated at the end. The reference and our-energy DAT paths share this control
flow; GS supplies its own prediction and acceptance hooks.

The own-energy VBD path (`--solver dat`, including its per-color CCD mode)
uses the project's corotated membrane gradient and analytic 3x3 self-Hessian,
not the previous native Neo-Hookean PSD surrogate. Bending uses the project's
dihedral gradient and rank-one Gauss-Newton self-Hessian, as on CPU; solid
elasticity already uses the project's frozen-rotation self-Hessian. These
material routines use FP32, with binary64 evaluation on the GPU for very thin
or ill-conditioned triangles/hinges to avoid cancellation. Local buffers remain
FP32. The native VBD local inverse and update schedule
remain in place. The analytic membrane Hessian can be indefinite; no PSD
projection is added. This does not imply byte-identical CPU trajectories.

Supporting code is grouped by purpose:

| Purpose | Files |
|---|---|
| Material equations | [energy.py](energy.py) wraps the cloth, solid and bending equations in [cloth_math.cuh](cloth_math.cuh) and [solid_math.cuh](solid_math.cuh) |
| Mesh contact | [dat_energies.py](dat_energies.py) contains native DAT contacts; [contact.py](contact.py) contains the ordinary, compact and exact GS contact paths. |
| VBD color CCD | [dat_color_schedule.py](dat_color_schedule.py) drives combined colors; [dat_color_ccd.py](dat_color_ccd.py) gathers each color's candidate pairs and clips its vertex updates through [ccd_direct_fp32.cuh](ccd_direct_fp32.cuh) or [ccd_plane_fp32.cuh](ccd_plane_fp32.cuh) |
| Analytic obstacles | [sdf.py](sdf.py) assembles signed-distance-field (SDF) terms |
| CUDA compilation | [warp_build.py](warp_build.py) includes our `.cuh` arithmetic in Warp's compiled kernels |
| Prescribed motion | [motion.py](motion.py) updates scene targets on the GPU |
| Rigid bodies | [rigid.py](rigid.py) contains rigid equations, graph preprocessing and GS stages; [rigid.cuh](rigid.cuh) holds their native arithmetic |
| Shared DAT example | [dat_twist.py](dat_twist.py) runs the Newton example; `run_own()` selects our energies and `attach_stock_ipc()` attaches their inputs in the same module |
| Tests | [warp_tests/](warp_tests/) covers GS and shared equations; [dat_tests/](dat_tests/) covers the DAT integration |

## Build

The production solver requires an NVIDIA CUDA GPU and a compatible driver.
Dependencies are:

| Component | Requirement |
|---|---|
| Python | 3.11 or newer |
| NVIDIA Warp | 1.19.0, using the CUDA 12 or CUDA 13 requirements file |
| NumPy / USD | 2.4.6 / usd-core 26.8, pinned in those files |
| Native scene exporter | C++17, CMake 3.24+, Boost 1.70+ and OpenMP |
| Fetched native dependencies | Eigen 3.4.0 and Tight-Inclusion 1.0.6 |

Newton is included in this repository. Warp compiles the GPU kernels; the
main build needs no separate Newton installation, `nvcc` or cuSPARSE. See the
[parent README](../README.md#getting-started) for native dependency setup.

From `3D_IPC/`, on Linux x86-64 with CUDA 12:

```sh
python3.11 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r gpu/newton_dat/requirements-cuda12.txt
cmake -S gpu -B gpu/build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv-gpu/bin/python"
cmake --build gpu/build --target 3D_sim_gpu --parallel
./gpu/build/3D_sim_gpu --check
```

For CUDA 13, install [requirements.txt](newton_dat/requirements.txt) instead.
The build produces a Python launcher, `3D_sim_gpu`, and the C++ scene exporter,
`3D_export_newton_scene`. Simulation runs in the Warp/CUDA kernels.

## Run

All commands below run from `3D_IPC/`.

### Compare the two DAT variants

Use the Python entry point for the shared Newton cloth example:

```sh
# Reference DAT: original energies
.venv-gpu/bin/python -m gpu.run --solver dat-reference \
  --viewer null --num-frames 60 --test

# Same DAT algorithm and scene, with our energies
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --viewer null --num-frames 60 --test
```

These commands use Newton's example options, including `--viewer` and
`--num-frames`. The generated `3D_sim_gpu` launcher is for our native scenes
and supplies a scene exporter, so use Python directly for this shared example.

The same stock scene can use our GS solver with all collision processing off:

```sh
.venv-gpu/bin/python -m gpu.run --solver ipc --stock-cloth-twist \
  --no-collisions --viewer null --num-frames 480 --test
```

`--no-collisions` is a deformable GS option. It disables contact detection,
barrier/friction and SDF terms, CCD (including predictor CCD), and movement-box
limits. Material energies, pins, inertia and the requested velocity predictor
remain active; material-only GS colors are built once and reused. Defaults
keep collisions enabled. The corresponding own-energy DAT option is
`--no-planar-dat`, which also removes DAT motion limits during the VBD solve.
Candidate generation, coloring, initial-guess CCD and per-color CCD can be enabled separately
as described below.

### Use our level-based CCD initial guess with VBD

The modified DAT/VBD path can use the existing GPU level-based CCD predictor:

```sh
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --level-based-ccd-guess --colored-ccd-guess-iters 1 \
  --colored-ccd-guess-fraction 1.0 \
  --viewer null --num-frames 480 --test
```

`--colored-ccd-guess-iters` is the maximum number of **updating sweeps**;
the level method uses an additional discarded probe to construct levels.
Both the probe and updating sweeps clip each vertex independently: a blocked
vertex does not shorten unrelated vertices in the same color or level/color
group. The shared retry condition only requests another CCD verification
after rounding; it does not set those vertices' clipping fractions.
Early exit remains enabled. The target fraction must be in `[0,1]`. The
defaults for this opt-in are one updating sweep and fraction `1.0`; without
`--level-based-ccd-guess`, the native VBD predictor remains unchanged. These
hyphenated predictor options require `--solver dat` in this driver.

This predictor uses the existing deformable mesh vertex–triangle and edge–edge
CCD implementation and requires surface triangles. It does not add rigid-body
prediction or CCD against analytic SDF colliders. Its target is the step-start
position plus `fraction * dt * velocity`; VBD keeps its gravity-inclusive
inertial target for the subsequent energy solve.

It also works with `--no-planar-dat`, including boxes and combined coloring:

```sh
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --level-based-ccd-guess --colored-ccd-guess-iters 1 \
  --colored-ccd-guess-fraction 1.0 \
  --viewer null --num-frames 480 --test
```

Without `--color-ccd`, this combination performs **CCD during the initial guess only**. The later
VBD iterations still have contact forces, friction and DAT motion limits off;
the combined coloring does not enforce collision-free motion. Reports record
predictor CCD separately from collision processing during the solve.

Our DAT path has an opt-in boxes-only stage for measuring broad-phase setup
while collision response stays off:

```sh
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --no-planar-dat --build-broad-phase-boxes --viewer null --num-frames 480 --test
```

`--build-broad-phase-boxes` requires `--solver dat --no-planar-dat`. It builds
our adaptive node boxes, face/edge AABBs and BVHs. The node radius uses
`node_box_min`/`node_box_max`. The query padding is `d_hat`, raised to at least
the linear CCD gap floor when `--color-ccd` is enabled. Boxes refit
after prediction and every `node_box_update_count` solver iterations, with
hierarchy rebuilds when the scene requests them. Without the separate predictor
or coloring options, queries, contact rows,
barrier/friction, CCD, collision recoloring and DAT motion limits remain off.
Motion remains unrestricted, so these boxes do not bound future motion
between refreshes. Default runs keep this stage off. Native-scene checkpoint
signatures record the option and cannot be mixed with boxes-off runs.

Add `--color-barrier-pairs` to query the BVHs for candidate point-triangle and
edge-edge pairs and build combined material/pair colors. The resulting colors
drive the VBD vertex sweeps; changed sweep ordering can change the trajectory.
Without `--color-ccd`, contact forces, friction, per-update CCD and DAT motion
limits remain disabled during the VBD solve. The
candidate boxes still do not guarantee collision-free motion.

```sh
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --viewer null --num-frames 480 --test
```

`--color-barrier-pairs` requires all three flags `--solver dat`,
`--no-planar-dat` and `--build-broad-phase-boxes`. Candidate row storage uses
`--vertex-contact-capacity` and `--edge-contact-capacity`. Native-scene restart
signatures record coloring separately, so coloring and boxes-only runs cannot
share a checkpoint.

### Apply our CCD to actual VBD color updates

Add `--color-ccd` to clip each VBD color's proposed vertex updates with our
FP32 CCD. `--color-ccd-method direct_rounding` remains the default;
`--color-ccd-method plane` selects the alternative that tests multiple
separating directions:

```sh
.venv-gpu/bin/python -m gpu.run --solver dat --stock-cloth-twist \
  --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs --color-ccd \
  --level-based-ccd-guess --colored-ccd-guess-iters 1 \
  --colored-ccd-guess-fraction 1.0 \
  --viewer null --num-frames 480 --test
```

This option requires `--solver dat --no-planar-dat --build-broad-phase-boxes
--color-barrier-pairs`. The trajectory exporter uses the same prerequisites
with `--variant own` in place of `--solver dat`. It gathers the current color's
candidate pairs, then reduces incident CCD bounds with `atomicMin` into each
moving vertex's own fraction, as in the level-based predictor. A blocked
vertex does not shorten unrelated vertices in its color. CCD clips actual
VBD updates. By default, contact forces, friction, SDF terms and planar DAT
limits remain off. The material local solve and combined color order are unchanged.

For experiments with `--color-ccd-method plane`, add
`--color-ccd-accept-uncertified` to publish the original box-clamped proposal
when per-color CCD would leave a vertex stationary. Certified partial steps
still apply. The fallback can let primitives cross.

Add `--color-ccd-contact-forces` to evaluate IPC barrier and friction contributions
from the current positions before each color's material solve. Each contact
contributes only to vertices in that color; its gradients and Hessians enter
the local update before CCD clips the proposed movement. Friction uses the
previous substep positions as its motion reference. The same candidate graph
is reused, and planar DAT remains disabled. This flag requires `--color-ccd`;
it uses the scene's `d_hat`, `k_barrier`, friction coefficient and friction
regularization. Zero friction coefficient disables the friction contribution.
For the shared twisting scene these defaults are `d_hat=0.002`,
`k_barrier=100`, friction coefficient `0.2` and velocity regularization `0.01`.
SDF contacts are not enabled by this flag.

This per-color contact path now uses the project's complete barrier
self-Hessian (including distance curvature) and project frozen-friction
gradient/Hessian. Contact geometry and derivatives use binary64 arithmetic
on the GPU, then contribute to FP32 local systems; both per-color CCD methods
remain FP32 without rational arithmetic or bisection. The separate
Planar DAT contact adapter still retains native rank-one contact blocks and
native friction.

Initial-guess CCD remains a separate option. With no level-based CCD predictor,
`--color-ccd` starts VBD at the step-start positions and keeps the native
inertial target for the energy solve; its first color updates pass through CCD.
With `--level-based-ccd-guess`, the selected predictor runs before those color
updates. `--ccd-method` selects only that predictor's method;
`--color-ccd-method` independently selects actual VBD color clipping.
To use plane-based CCD in both places, pass
`--level-based-ccd-guess --ccd-method plane --color-ccd --color-ccd-method plane`.
This selects our plane-based CCD, not Planar DAT. Native checkpoint signatures
distinguish color-CCD runs and their clipping method from coloring-only runs.

For synchronized timing and saved samples of the fixed 2,500-vertex,
480-frame, 60 Hz, 10-substep × 4-iteration scene:

```sh
.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant gs --no-collisions --output results/stock_gs_off.npz

.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant own --no-planar-dat --build-broad-phase-boxes \
  --output results/stock_dat_boxes_only.npz

.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --output results/stock_dat_pair_coloring.npz

.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --level-based-ccd-guess --colored-ccd-guess-iters 1 \
  --colored-ccd-guess-fraction 1.0 \
  --output results/stock_vbd_level_guess.npz

.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --color-ccd --level-based-ccd-guess --ccd-method direct_rounding \
  --colored-ccd-guess-iters 1 --colored-ccd-guess-fraction 1.0 \
  --output results/stock_vbd_color_ccd.npz

# Same CCD acceptance, with barrier and friction forces enabled:
.venv-gpu/bin/python gpu/newton_dat/export_twist_trajectory.py \
  --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs \
  --color-ccd --color-ccd-contact-forces \
  --level-based-ccd-guess --ccd-method direct_rounding \
  --colored-ccd-guess-iters 1 --colored-ccd-guess-fraction 1.0 \
  --vertex-contact-capacity 512 --edge-contact-capacity 1024 \
  --output results/stock_vbd_color_ccd_contact_forces.npz
```

The trajectory exporter requires `--variant own --no-planar-dat` for the
boxes-only option. Its JSON metadata records the boxes stage separately from
collision queries, contact energies and planar DAT, which remain disabled.
With `--color-barrier-pairs`, metadata reports candidate queries, combined
coloring and use of those colors for sweeps as active. Without `--color-ccd`,
collision response stays disabled. With `--color-ccd`, metadata records solve
collision processing and response as enabled, the fixed color CCD method, and
contact energies, friction and SDF energies as disabled unless
`--color-ccd-contact-forces` is selected. That flag records contact assembly
separately, including the actual barrier stiffness and friction coefficient;
barrier/friction enabled fields account for zero coefficients. SDF and planar
DAT stay off. Native restart signatures distinguish the contact-force setting. With
`--level-based-ccd-guess`, metadata additionally records the
updating cap, target fraction and predictor CCD; this does not enable DAT or
barrier energies during the solve. Device statistics cover coloring checkpoints and are downloaded
once after timing; the statistics download is excluded from measured steps.

### Run our scene examples

Use the [parent README's scene options](../README.md#example-scene-commands)
with `./gpu/build/3D_sim_gpu`. The following commands use a small cloth mesh:

```sh
# Our GS solver with IPC contact and CCD
./gpu/build/3D_sim_gpu --solver ipc --example 1 \
  --twist_nx 9 --twist_ny 9 --fps 30 --num_frames 30 \
  --fixed_iters --substeps 3 --max_substep_iters 3 \
  --format geo --outdir results/gpu_gs

# DAT with our energies
./gpu/build/3D_sim_gpu --solver dat --example 1 \
  --twist_nx 9 --twist_ny 9 --fps 30 --num_frames 30 \
  --fixed_iters --substeps 10 --max_substep_iters 10 \
  --use_ccd_guess false --use_colored_ccd_guess false \
  --use_colored_ccd_guess_level_based false --dat_inertial_predictor true \
  --format geo --outdir results/gpu_dat
```

GS requires `--fixed_iters`. DAT uses its native inertial predictor by default
and rejects the ordinary and non-level colored CCD-guess flags. To replace its
predictor, add `--level-based-ccd-guess` with the optional updating cap and
fraction shown above. Native scenes also accept the existing
`--use_colored_ccd_guess_level_based true --colored_ccd_guess_iters N
--colored_ccd_guess_fraction F` scene flags, with the other two guess flags
disabled. The hyphenated opt-in explicitly selects level-based prediction and
overrides the scene's other predictor flags. For `--color-ccd`, the
step-start positions are retained when no active CCD predictor is selected,
as described above. Our DAT path
currently rejects rigid bodies; use GS for rigid and mixed scenes. OGC,
cloth-grid and Jacobi modes are not available in this GPU driver.

For deformable GS experiments, `--gs-ccd-accept-uncertified` uses the
box-clamped proposal when update CCD would leave a vertex stationary.
For the level-based predictor, the separate opt-in
`--gs-ccd-guess-accept-uncertified` uses the predicted position if CCD
would otherwise leave the vertex stationary. The discarded probe and
updating sweeps use the same acceptance rule. Both options keep certified
partial steps and can allow crossings. On native scenes, select the
level-based guess with `--use_colored_ccd_guess false
--use_colored_ccd_guess_level_based true --colored_ccd_guess_iters 1`.
On `--stock-cloth-twist`, the predictor opt-in selects one level-based
updating sweep.

For native scene runs:

- Use `--num_frames` with an underscore. `--substeps` sets the steps per
  output frame; `--max_substep_iters` sets solver sweeps per substep.
- Use an empty output directory. Meshes and `state_NNNN.bin` checkpoints start
  at frame 0. Resume with `--restart_frame N` in the same directory and keep
  scene, material and solver settings consistent.
- If collision rows overflow, increase `--vertex-contact-capacity` and
  `--edge-contact-capacity`, for example to `512` and `1024`.
- Add `--benchmark` for timing without mesh/checkpoint output, or
  `--no-graphs` to disable CUDA Graph replay while debugging.

State and local systems use float32. Our GPU CCD also uses float32 throughout
its trajectory certificates, step fractions and endpoint interpolation, with
outward-rounded bounds in [ccd_fp32.cuh](ccd_fp32.cuh). An uncertain moving
trajectory is shortened and its represented endpoint checked again; if it
cannot be certified, the update can stop at its starting position. CCD has no
float64 or exact-arithmetic fallback. Very small gaps, degenerate geometry or
large coordinate offsets can therefore produce smaller accepted steps than
the previous higher-precision implementation.

The reference DAT GPU kernels also use float32. Their plane-based truncation
is a different motion-limiting algorithm, so equal precision does not imply
equal cost or accepted steps. Our DAT variant's `contact_precision` option
controls the scalar IPC barrier evaluation, independently of CCD. Existing
`exact_computation_fallback` and `ccd_exact_step_bound` scene settings no longer
select higher-precision CCD; the former still affects other contact geometry.
Other geometry and energy routines may continue to use higher precision.
Shared broad-phase movement radii are converted inward from their float64
storage before CCD uses them.

The shared twisting-trajectory exporter accepts `--ccd-method` with
`--level-based-ccd-guess`. `interval` remains the default. Two experimental
methods require contact-independent vertex groups (at most one moving vertex
in each candidate pair):

- `plane`: tests up to ten starting-geometry directions, directly bounds the
  projected gaps and reserves a 0.99 fraction margin.
- `direct_rounding`: proposes one closest-feature direction in FP32, checks
  every cross-primitive vertex gap, and includes an explicit bound for the
  rounding in the final position interpolation. It follows the CPU direct
  proposal's structure, but does not use its exact closest-feature arithmetic.

The direct method can conservatively reject separated near-parallel edges
when FP32 feature classification chooses an unsuitable axis. The plane method
can accept such moves by testing additional separating directions. Both
remain available; selecting a method can change the trajectory and contact work.

Both use independent per-vertex fractions and verify shortened stored endpoints
after reducing all incident contacts. They use neither bisection nor a
higher-precision fallback; an uncertified move stays at the starting position.
Both conservatively require a fixed direction to certify the full `1e-7`
gap floor. In particular, they do not reproduce the CPU fallback's exact
`min(starting gap, gap floor)` policy for starts below that floor. The
`direct_rounding` method leaves rigid-body CCD unchanged. Changes in accepted
steps can change trajectories and later contact/color work, so end-to-end
timings should be reported separately from timings on identical CCD inputs.

GPU update ordering and movement budgets differ from the CPU solver, so their
trajectories can differ. OpenMP and CPU SIMD flags do not set CUDA parallelism.

## Tests

Run GS and DAT tests in separate processes so each loads its own Newton
snapshot:

```sh
IPC_NEWTON_SCENE_EXPORTER="$PWD/gpu/build/3D_export_newton_scene" \
  .venv-gpu/bin/python -m unittest discover -s gpu/warp_tests -t . -v
.venv-gpu/bin/python -m unittest discover -s gpu/dat_tests -t . -v
python3 gpu/newton_dat/verify_import.py
python3 gpu/newton_dat/verify_import.py --reference
```

The FP32 CCD regressions include analytic vertex–triangle and edge–edge
crossings, degenerate primitives, near-coplanar motion, represented endpoint
rounding, large coordinates, subnormal scales and invalid inputs. A C++17
compiler can run the standalone host checks without Warp:

```sh
python3 -m unittest gpu.warp_tests.test_ccd_fp32.Float32CertificateHostTests -v
```

With Warp installed, the complete `test_ccd_fp32` module also checks the actual
CPU/CUDA kernel path and per-node acceptance. CUDA checks run when a CUDA
device is available; host checks do not establish GPU performance.

The [import notes](newton_dat/README.md) describe the pinned Newton snapshots
and source verification.

## Optional FP32 CPU-style local solve

`SolverDAT(..., local_solve="cpu_fp32", local_solve_damping=1.0)` uses a
scale-aware 3×3 inverse and a rank-revealing, minimum-norm QR fallback. Its
local arithmetic is entirely FP32, including the fallback. It preserves
negative Hessian eigenvalues; it does not add a PSD projection or line search.
Both scalar and tiled kernels support it, with either fixed or combined colors.
`local_solve="native"` remains the default for comparison.

This follows the CPU inverse/pseudoinverse policy, with FP32 rank tolerances;
it does not reproduce Eigen's double-precision operation order. Material and
contact precision policies remain unchanged. CUDA local arithmetic uses explicit
round-to-nearest FP32 instructions without changing material compiler flags.
Tests are in `gpu/dat_tests/test_local_solve_fp32.py`.

## Credits

This implementation builds on the **Newton Developers' GPU VBD/DAT code**.
We reuse its timestep/color-loop structure, model/state layout, BVHs and
cooperative-kernel organization. Our DAT path uses its planar truncation
kernels with our energies.

**DAT paper:** Anka He Chen, Jerry Hsu, Youssef Ayman and Miles Macklin.
*Divide and Truncate: A Penetration and Inversion Free Framework for Coupled
Multi-physics Systems*, SIGGRAPH 2026.
[Paper](https://arxiv.org/abs/2604.15513) ·
[Authors' project and released code](https://ankachan.github.io/Projects/DAT/).
The underlying optimizer is [Vertex Block Descent](https://doi.org/10.1145/3658179).

The Newton snapshots are pinned to
[`f89e82f`](https://github.com/newton-physics/newton/tree/f89e82fed043fe78d1844b26d3f289a25db03810).
The [Apache-2.0 license](third_party/newton/LICENSE.md) and additional notices
are preserved. [FORK.json](third_party/newton/FORK.json) records the GS fork's
two local patches.

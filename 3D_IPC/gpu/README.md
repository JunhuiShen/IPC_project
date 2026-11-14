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
energies. **DAT with our energies** keeps that solver's inertial predictor,
color order, collision schedule, local solve and Divide and Truncate (DAT)
motion limits, while substituting our energy evaluations. This lets us compare
the energy models within the same DAT algorithm.

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
| Reference DAT | [solver_dat_reference.py](solver_dat_reference.py), a direct alias to upstream `SolverVBD` | [solver_vbd.py](third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py) for the timestep and color loop; [particle_vbd_kernels.py](third_party/newton_reference/newton/_src/solvers/vbd/particle_vbd_kernels.py) for local solves and DAT truncation |
| DAT with our energies | [solver_dat.py](solver_dat.py), which substitutes energy calls in the upstream particle loop | [dat_material_kernels.py](dat_material_kernels.py) and [dat_materials.py](dat_materials.py) for material terms; [dat_contacts.py](dat_contacts.py) for IPC/friction on Newton's contact rows |
| Our GS solver | [solver.py](solver.py), which supplies our particle update and acceptance hooks | [kernels.py](kernels.py) for local assembly/solves; [ccd.py](ccd.py) for motion checks; [policies.py](policies.py) for CCD initial guesses; [body_solver.py](body_solver.py) for rigid GS |

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

Supporting code is grouped by purpose:

| Purpose | Files |
|---|---|
| Material equations | [energy.py](energy.py) wraps the cloth, solid and bending equations in [cloth_math.cuh](cloth_math.cuh) and [solid_math.cuh](solid_math.cuh) |
| Mesh contact | [contact.py](contact.py) provides our IPC/friction evaluation; [dat_contact_fast.py](dat_contact_fast.py) provides DAT's mixed-precision evaluator |
| Analytic obstacles | [sdf.py](sdf.py) assembles signed-distance-field (SDF) terms |
| CUDA compilation | [warp_build.py](warp_build.py) includes our `.cuh` arithmetic in Warp's compiled kernels |
| Prescribed motion | [motion.py](motion.py) updates scene targets on the GPU |
| Rigid bodies | [body_solver.py](body_solver.py) schedules rigid GS; [rigid.py](rigid.py) supplies rigid equations |
| Shared DAT example | [dat_twist.py](dat_twist.py) runs the Newton example; [stock_twist.py](stock_twist.py) selects our energies; [dat_stock_scene.py](dat_stock_scene.py) attaches their inputs |
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

GS requires `--fixed_iters`. DAT uses its native inertial predictor and
rejects the three GS CCD-guess flags, so disable them as shown. Our DAT path
currently rejects rigid bodies; use GS for rigid and mixed scenes. OGC,
cloth-grid and Jacobi modes are not available in this GPU driver.

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

State and local systems use float32; sensitive geometry uses higher precision
and exact fallback. GPU update ordering and movement budgets differ from the
CPU solver, so their trajectories can differ. OpenMP and CPU SIMD flags do
not set CUDA parallelism.

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

The [import notes](newton_dat/README.md) describe the pinned Newton snapshots
and source verification.

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

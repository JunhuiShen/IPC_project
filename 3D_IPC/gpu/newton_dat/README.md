# Direct import of the authors' GPU DAT implementation

The exact original Newton source is copied into
[../third_party/newton_reference](../third_party/newton_reference).
It is pinned to the exact revision validated on `biot`:
[`f89e82fed043fe78d1844b26d3f289a25db03810`](https://github.com/newton-physics/newton/tree/f89e82fed043fe78d1844b26d3f289a25db03810).
All 1,115 imported files are regular local files and match the original
checksums. A separate [GS fork](../third_party/newton) contains two declared
patches for physics hooks and empty-mesh BVHs. The reference and energy-only
DAT solvers use the pristine snapshot.

This includes the **complete `newton/` Python package**: runtime, official
examples, upstream tests and license notices. Original package metadata,
dependency lockfile, citation and VBD documentation are also preserved.
[`UPSTREAM.json`](../third_party/newton_reference/UPSTREAM.json) lists every file and its
SHA-256 checksum. Repository-level CI/development files are excluded.

The [DAT authors' project](https://ankachan.github.io/Projects/DAT/) links to
Newton as its released GPU implementation. It uses NVIDIA Warp to compile
the Python kernel definitions into CUDA. Gaia's original C++/CUDA VBD engine
is a separate implementation.

## Actual solver source

| Upstream file | Contents |
|---|---|
| [solver_vbd.py](../third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py) | Full VBD/AVBD solver, launch scheduling and DAT integration |
| [particle_vbd_kernels.py](../third_party/newton_reference/newton/_src/solvers/vbd/particle_vbd_kernels.py) | Cloth/solid kernels, tiled elasticity and soft self-contact DAT |
| [rigid_vbd_kernels.py](../third_party/newton_reference/newton/_src/solvers/vbd/rigid_vbd_kernels.py) | Rigid equations, rigid-soft DAT and trajectory checks |
| [interval_arithmetic.py](../third_party/newton_reference/newton/_src/solvers/vbd/interval_arithmetic.py) | Interval arithmetic helpers |
| [example_cloth_twist.py](../third_party/newton_reference/newton/examples/cloth/example_cloth_twist.py) | Authors' original twisting-cloth scene |

The supporting model, geometry, collision, coupling, tendon and utility code
is present in the same package. The local patch extracts the particle contact,
elasticity and commit operations into overridable hooks, preserving their
original implementations as the defaults. See
[FORK.json](../third_party/newton/FORK.json) and the
[hook patch](patches/0001-particle-physics-hooks.patch). A second
[patch](patches/0002-empty-triangle-bvh.patch) avoids constructing or querying
an empty triangle BVH, matching the existing upstream treatment of empty edges.

Our GS solver (`--solver ipc`) lives in [solver.py](../solver.py). It subclasses
the GS fork's `SolverVBD`, inherits its timestep and particle color loop, and
supplies our energies, IPC contact and CCD through those hooks.
[DAT with our energies](../solver_dat.py) instead uses the pristine snapshot
and retains its DAT motion limits while substituting energy evaluations.
[Reference DAT](../solver_dat_reference.py) exposes the pristine solver directly.
See the [GPU README](../README.md) for the three solver choices and run commands.

## Install on biot or another CUDA 12 Linux x86-64 machine

Use Python 3.11. Run from `3D_IPC/`:

```sh
python3.11 -m venv gpu/newton_dat/.venv
source gpu/newton_dat/.venv/bin/activate
python -m pip install -r gpu/newton_dat/requirements-cuda12.txt
```

For the standard CUDA 13 Warp wheel, use `requirements.txt` instead and a
compatible NVIDIA driver. The CUDA 12 file deliberately selects the exact
Linux x86-64 wheel used for the benchmark. The `square_cloth.usd` asset comes
with the pinned Warp wheel. The headless example needs no GUI, MuJoCo or
PyTorch packages.

## Run the shared twisting-cloth test

```sh
python -m gpu.run --solver dat-reference \
  --device cuda:0 --viewer null --test --benchmark
```

This imports the unmodified local reference solver and runs the shared
twisting-cloth example through [dat_twist.py](../dat_twist.py). Use
`--solver dat --stock-cloth-twist` to run the same scene with our energies.

To use the source as an installed editable package instead of setting
`PYTHONPATH`, after installing dependencies run:

```sh
python -m pip install --no-deps -e gpu/third_party/newton_reference
python -m newton.examples cloth_twist --device cuda:0 --viewer null --test --benchmark
```

Newton's original `pyproject.toml` supplies the editable-build backend.
Installing `newton[examples]` is unnecessary for this headless scene and would
pull in additional optional dependencies.

## Source verification and GPU validation

```sh
python3 gpu/newton_dat/verify_import.py --reference
python3 gpu/newton_dat/verify_import.py
```

Each check verifies all 1,115 imported files, including the complete
1,108-file Python package. `--reference` requires every original checksum.
The default checks the GS fork against its original checksums and two declared
patches. Unrecorded changes fail verification.

The public GPU driver also exposes this as `--solver dat-reference`.
`--solver dat --stock-cloth-twist` runs the same official scene with our energy
evaluations substituted. Both retain the original initial guess, coloring,
collision schedule, DAT clipping, local inverse and frame loop. Our energies
change the constitutive/contact laws and exclude the stock viscous material
damping. See the [GPU README](../README.md) for both commands.

Upstream code is Apache-2.0; see [LICENSE.md](../third_party/newton/LICENSE.md)
and [additional notices](../third_party/newton/newton/licenses).

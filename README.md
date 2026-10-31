# Incremental Potential Contact Simulators

A C++17 research codebase for **Incremental Potential Contact (IPC)**, pairing
a compact 2D testbed with a full 3D simulator. Both projects explore nonlinear
Gauss--Seidel solvers, conservative contact sets, collision-safe local updates,
and reduced-coordinate rigid-body dynamics.

## Choose a simulator

| | [`2D_IPC/`](2D_IPC/) | [`3D_IPC/`](3D_IPC/) |
|---|---|---|
| Geometry | Explicit spring-edge networks and polygonal rigid bodies | Cloth / thin shells, tetrahedral deformable solids, and triangle-mesh rigid bodies |
| IPC pairs | Node--segment | Node--triangle and segment--segment |
| Elasticity | Axial springs | Corotated membrane, discrete-shell bending, and volumetric corotated energy |
| Static contact | Analytic 2D SDFs | Plane, cylinder, and sphere SDFs |
| Solver | Per-node or reduced rigid-body nonlinear Gauss--Seidel | Cloth, solid, rigid, or combined nonlinear Gauss--Seidel; SIMD v2 and optional Jacobi-DAT |
| Safe steps | Linear CCD or a distance-based trust region | Linear and curved rigid-motion CCD, Tight-Inclusion, OGC, or DAT |
| Parallelism | Conflict-colored OpenMP updates in the basic deformable and rigid-body solvers | OpenMP and SIMD on CPU; a separate CUDA device solver |
| Output | Houdini `.geo`, Wavefront `.obj`, binary checkpoints | Houdini `.geo`, Wavefront `.obj`, `.ply`, ASCII USD `.usda`, binary checkpoints |

See the subproject guides for setup, scene commands, CLI options, output
conventions, and source maps:

- [2D simulator guide](2D_IPC/README.md)
- [3D simulator guide](3D_IPC/README.md)
- [3D CUDA solver guide](3D_IPC/gpu/README.md)

## How the solvers fit together

The Gauss--Seidel paths in both simulators follow the same broad loop:

1. Construct a substep predictor from the current positions and velocities.
2. Minimize an incremental potential containing inertia, elasticity or rigid
   inertia, constraints, and contact.
3. Build conservative contact candidates with AABB/BVH broad phases.
4. Color the elastic/contact conflict graph and apply local Newton updates in
   nonlinear Gauss--Seidel sweeps.
5. Clamp each update with CCD or an Offset Geometric Contact (OGC) trust region
   so intermediate iterates remain collision-safe.
6. Update velocities, export the frame, and write a restart checkpoint.

The 2D project is the smaller testbed for spring networks and planar rigid
bodies. The 3D project is the full simulator: it adds shell membrane and bending
energies, node--triangle and edge--edge contact, quaternion rigid-body motion, tetrahedral deformable solids, combined solving,
alternative OGC and Jacobi-DAT solvers, a CUDA backend, richer output, and regression fixtures.

## Requirements

- A C++17 compiler
- CMake 3.24+ for fresh dependency downloads; declared CPU project minima are
  3.16 for 2D and 3.21 for 3D with populated dependency sources
- GoogleTest for test targets (required in 2D; optional in 3D with
  `-DBUILD_TESTING=OFF`)
- Boost 1.70+ with its CMake package configuration for 3D
- Git for the fetched 3D dependencies
- CUDA Toolkit, an NVIDIA GPU, and exactly Boost 1.92.0 for the optional
  [3D CUDA build](3D_IPC/gpu/README.md#build)
- OpenMP (optional in 2D and required in 3D; on macOS, install Homebrew
  `libomp`)
- Network access during the first configure so CMake can fetch Eigen 3.4.0;
  the 3D build also fetches Tight-Inclusion CCD 1.0.6

The subprojects have separate CMake builds; there is no top-level CMake target.

## Build and test

From the repository root, configure and build either project independently:

```sh
# 2D
cmake -S 2D_IPC -B 2D_IPC/build
cmake --build 2D_IPC/build -j
ctest --test-dir 2D_IPC/build --output-on-failure

# 3D
cmake -S 3D_IPC -B 3D_IPC/build
cmake --build 3D_IPC/build -j
ctest --test-dir 3D_IPC/build --output-on-failure
```

Useful 3D configuration switches are `-DIPC_BUILD_SIMULATOR=OFF`,
`-DIPC_BUILD_TOOLS=OFF`, `-DIPC_ENABLE_IPO=OFF`, and `-DBUILD_TESTING=OFF`.

## Run

Print the generated CLI reference:

```sh
./2D_IPC/build/simulation --help
./3D_IPC/build/3D_sim --help
```

Small smoke runs from the repository root:

```sh
./2D_IPC/build/simulation \
  --example 1 --nodes 20 --num_frames 1 --outdir 2D_IPC/frames_smoke

./3D_IPC/build/3D_sim \
  --example 1 --twist_nx 9 --twist_ny 9 --num_frames 1 \
  --max_substep_iters 20 --fixed_iters --format obj \
  --outdir 3D_IPC/frames_smoke
```

A fresh run replaces its output folder; choose a dedicated path. Frame 0 is
the initial mesh, and each completed frame has a matching restart checkpoint.
For a guided first run, categorized scenes, asset paths, and restart examples,
see the [3D quick guide](3D_IPC/README.md#getting-started). Its scene commands
run from `3D_IPC/`, where the bundled `example_obj/` assets live.

## Repository layout

```text
IPC_project/
├── 2D_IPC/           2D simulator, tests, and detailed guide
├── 3D_IPC/           3D simulator, tests, fixtures, and detailed guide
│   └── gpu/          CUDA solver, tests, and GPU guide
└── README.md         repository overview
```

## References and acknowledgments

The core IPC contact formulation follows:

> Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis Zorin,
> Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. *Incremental Potential
> Contact: Intersection- and Inversion-Free, Large-Deformation Dynamics.*
> ACM Transactions on Graphics 39(4), Article 49, 2020.
> [Project page](https://ipc-sim.github.io/)

Rigid-body IPC background and reference scenes:

> Zachary Ferguson, Minchen Li, Teseo Schneider, Francisca Gil-Ureta, Timothy
> Langlois, Chenfanfu Jiang, Denis Zorin, Danny M. Kaufman, and Daniele Panozzo.
> *Intersection-Free Rigid Body Dynamics.* ACM Transactions on Graphics 40(4),
> Article 183, 2021. [Project page](https://ipc-sim.github.io/rigid-ipc/)

The OGC narrow phase and 3D OGC solver implement:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
> Yuksel. *Offset Geometric Contact.* ACM Transactions on Graphics 44(4):160,
> 2025. [doi:10.1145/3731205](https://doi.org/10.1145/3731205)

The friction model follows:

> Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. *Vertex Block Descent.*
> ACM Transactions on Graphics 43(4), Article 116, 2024.
> [doi:10.1145/3658179](https://doi.org/10.1145/3658179)

The optional Jacobi-DAT path follows:

> Anka He Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. *Divide and Truncate:
> A Penetration and Inversion Free Framework for Coupled Multi-physics Systems.*
> 2026. [arXiv:2604.15513](https://arxiv.org/abs/2604.15513)

General-motion 3D CCD uses
[Tight-Inclusion CCD](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):

> Bolun Wang, Zachary Ferguson, Teseo Schneider, Xin Jiang, Marco Attene, and
> Daniele Panozzo. *A Large Scale Benchmark and an Inclusion-Based Algorithm
> for Continuous Collision Detection.* ACM Transactions on Graphics 40(5),
> Article 188, 2021. [doi:10.1145/3460775](https://doi.org/10.1145/3460775)

The discrete-shell bending model follows:

> Eitan Grinspun, Anil N. Hirani, Mathieu Desbrun, and Peter Schröder.
> *Discrete Shells.* ACM SIGGRAPH/Eurographics Symposium on Computer Animation,
> 62–67, 2003. [Project page](https://multires.caltech.edu/pubs/DS-CDROM/)

The corotated elasticity model is described in:

> Alexey Stomakhin, Russell Howes, Craig Schroeder, and Joseph M. Teran.
> *Energetically Consistent Invertible Elasticity.* ACM SIGGRAPH/Eurographics
> Symposium on Computer Animation, 25–32, 2012.
> [doi:10.2312/SCA/SCA12/025-032](https://doi.org/10.2312/SCA/SCA12/025-032)

The bundled SVD implementation and its CPU/GPU adaptations follow:

> Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, and Joseph Teran. *Implicit-shifted
> Symmetric QR Singular Value Decomposition of 3x3 Matrices.* UCLA Mathematics
> Department Technical Report CAM16-19, 2016.
> [Technical report](https://ww3.math.ucla.edu/camreport/cam16-19.pdf)

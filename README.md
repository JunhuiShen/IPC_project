# Incremental Potential Contact Simulators

A C++17 research codebase for **Incremental Potential Contact (IPC)**, pairing
a compact 2D testbed with a full 3D simulator. Both projects explore nonlinear
Gauss--Seidel solvers, conservative contact sets, collision-safe local updates,
and reduced-coordinate rigid-body dynamics.

## Choose a simulator

| | [`2D_IPC/`](2D_IPC/) | [`3D_IPC/`](3D_IPC/) |
|---|---|---|
| Geometry | Explicit spring-edge networks and polygonal rigid bodies | Deformable triangle meshes (cloth / thin shells) and triangle-mesh rigid bodies |
| IPC pairs | Node--segment | Node--triangle and segment--segment |
| Elasticity | Axial springs | Corotated membrane energy and discrete-shell hinge bending |
| Static contact | Analytic 2D SDFs | Plane, cylinder, and sphere SDFs |
| Solver | Per-node or reduced rigid-body nonlinear Gauss--Seidel | Per-vertex basic/OGC or reduced rigid-body nonlinear Gauss--Seidel |
| Safe steps | Linear CCD or a distance-based trust region | Linear and Tight-Inclusion CCD routes or an OGC trust region |
| Parallelism | Conflict-colored OpenMP updates in the basic deformable and rigid-body solvers | Conflict-colored OpenMP updates in the basic deformable and rigid-body solvers |
| Output | Houdini `.geo`, Wavefront `.obj`, binary checkpoints | Houdini `.geo`, Wavefront `.obj`, `.ply`, ASCII USD `.usda`, binary checkpoints |

See the subproject guides for the complete algorithms, scene catalogues, CLI
options, output conventions, and source maps:

- [2D simulator guide](2D_IPC/README.md)
- [3D simulator guide](3D_IPC/README.md)

## How the solvers fit together

Both simulators follow the same broad loop:

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
energies, node--triangle and edge--edge contact, quaternion rigid-body motion,
alternative OGC solver mechanics, richer output, and regression fixtures.

## Requirements

- A C++17 compiler
- CMake 3.16+ for 2D and CMake 3.21+ for 3D
- GoogleTest (required by the configured test targets)
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
  --outdir 3D_IPC/frames_smoke
```

Some representative scenes:

```sh
# 2D rigid polygons colliding without gravity
./2D_IPC/build/simulation \
  --example 3 --gy 0 --num_frames 100 --outdir 2D_IPC/frames_collision

# 3D freely rotating tennis racket
./3D_IPC/build/3D_sim \
  --example 5 --format obj --outdir 3D_IPC/frames_racket

# 3D rigid prisms falling onto one another
./3D_IPC/build/3D_sim \
  --example 10 --substeps 10 --format obj --outdir 3D_IPC/frames_prisms
```

## Repository layout

```text
IPC_project/
├── 2D_IPC/           2D simulator, tests, and detailed guide
├── 3D_IPC/           3D simulator, tests, fixtures, and detailed guide
└── README.md         repository overview
```

## References and acknowledgments

The OGC narrow phase and 3D OGC solver implement:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
> Yuksel. *Offset Geometric Contact.* ACM Transactions on Graphics 44(4):160,
> 2025. [doi:10.1145/3731205](https://doi.org/10.1145/3731205)

General-motion 3D CCD uses
[Tight-Inclusion CCD](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):

> Bolun Wang, Zachary Ferguson, Teseo Schneider, Xin Jiang, Marco Attene, and
> Daniele Panozzo. *A Large-Scale Benchmark and an Inclusion-Based Algorithm
> for Continuous Collision Detection.* ACM Transactions on Graphics, 2021.

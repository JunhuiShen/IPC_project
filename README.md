# IPC Project

Research codebase for **Incremental Potential Contact (IPC)** simulation,
covering both 2D spring chains and 3D deformable triangle meshes
(cloth / thin shells). Each time step optimizes an incremental potential
with a **nonlinear Gauss–Seidel solver**, using continuous collision
detection (or a distance-based trust region) to keep every intermediate
state intersection-free.

## Subprojects

| Directory | What it is |
|-----------|------------|
| [`2D_IPC/`](2D_IPC/) | 2D spring-chain testbed for swapping broad-phase, step-filter, and initial-guess strategies. Lightweight, dependency-free. |
| [`3D_IPC/`](3D_IPC/) | Full 3D simulator for deformable triangle meshes (cloth / thin shells). CCD- and OGC-based step clamping, parallel-by-color GS, optional GPU pipeline, USD/OBJ/PLY/GEO export. |

Each subproject has its own README with build instructions, CLI flags,
example scenes, and a source-layout map. Start there:

- [`2D_IPC/README.md`](2D_IPC/README.md)
- [`3D_IPC/Readme.md`](3D_IPC/Readme.md)

## Shared themes

Both subprojects share the same algorithmic skeleton, which makes it easy
to port ideas between them:

- **Incremental potential.** Inertial + elastic + IPC log-barrier
  contact, minimized per substep.
- **Nonlinear Gauss–Seidel.** Per-vertex local Newton step, swept across
  the mesh once per outer iteration.
- **Step safety.** Either continuous collision detection (CCD) or a
  distance-based trust region clamps each per-vertex move to keep the
  iterate intersection-free.
- **Pluggable strategies.** Broad phase, step filter, and initial guess
  are interchangeable so different combinations can be benchmarked.

## Build

Each subproject builds independently with CMake:

    cd 2D_IPC && cmake -B build && cmake --build build
    cd 3D_IPC && cmake -B build && cmake --build build

The 3D project additionally requires OpenMP, GoogleTest, Eigen, and
Tight-Inclusion CCD (the latter two fetched automatically).

## Acknowledgments

3D OGC narrow phase and `global_gauss_seidel_solver_ogc` implement:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
> Yuksel. *Offset Geometric Contact.* ACM Transactions on Graphics
> 44(4):160, 2025. [doi:10.1145/3731205](https://doi.org/10.1145/3731205)

3D general-motion CCD uses
[Tight-Inclusion CCD](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):

> Bolun Wang, Zachary Ferguson, Teseo Schneider, Xin Jiang, Marco Attene,
> and Daniele Panozzo. *A Large-Scale Benchmark and an Inclusion-Based
> Algorithm for Continuous Collision Detection.* ACM Transactions on
> Graphics, 2021.

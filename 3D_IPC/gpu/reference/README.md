# Native reference engine

This directory preserves the previous C++/CUDA engine and all of its independent
numerical, controller, ownership and allocation-lifetime regression tests. It is
not part of the production Newton/Warp GPU build in `gpu/`. Shared arithmetic
headers here forward to their maintained implementations in the parent directory.
Test equations, fixtures and tolerances are unchanged.

Build and run its validation targets separately on a CUDA development machine:

```sh
cmake -S gpu/reference -B gpu/reference/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu/reference/build --target gpu_validation
ctest --test-dir gpu/reference/build --output-on-failure -L gpu_solver
```

The reference project requires CMake 3.24+, a CUDA toolkit/compiler, and the
same native CPU dependencies, including Boost 1.92.0. Pass local dependency
locations through the usual CMake cache variables when needed. Validation is
enabled by default. Its former native simulator is disabled by default and
can be built explicitly with `-DIPC_GPU_BUILD_DRIVER=ON`; the production
`gpu/build/3D_sim_gpu` continues to use Newton/Warp.

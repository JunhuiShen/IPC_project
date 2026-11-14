#pragma once

// Device-only broad-phase geometry shared by the native and Warp compilers.
// Keep this header free of host containers and CUDA runtime dependencies.
namespace ipc_gpu {
namespace solver {

struct BpAabb { double lo[3], hi[3]; };
struct BpTriangle { int vertices[3]; };
struct BpEdge { int vertices[2]; };
struct BpDependency { int vertices[4]; int count; };
struct BpContact {
    int vertices[4];
    // 0: node/triangle [point,a,b,c], 1: edge/edge [a,b,c,d].
    int kind;
};

} // namespace solver
} // namespace ipc_gpu

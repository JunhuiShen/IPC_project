#pragma once
#include "gpu_mesh.h"
#include "physics.h"

// --------------------------------------------------------------------------
// gpu_compute_local_gradient_and_hessian_no_barrier
//
// GPU equivalent of compute_local_gradient_and_hessian_no_barrier in
// physics.cpp. Replaces every unordered_map / vector<Vec3> access with
// direct indexing into flat DeviceBuffer arrays.
//
// In the stub (gpu_physics_stub.cpp), DeviceBuffer::ptr is a regular CPU
// heap pointer so this runs on the CPU and validates the data layout.
// In the real CUDA build (gpu_physics.cu) this becomes a __device__ function
// called from within a kernel, one thread per vertex.
//
// d_x    : flat positions   [num_verts * 3], interleaved xyz
// d_xhat : flat inertia target [num_verts * 3], interleaved xyz
// --------------------------------------------------------------------------
std::pair<Vec3, Mat33> gpu_compute_local_gradient_and_hessian_no_barrier(
    int vi,
    const GPURefMesh&           mesh,
    const GPUAdjacency&         adj,
    const GPUPins&              pins,
    const GPUPinMap&            pin_map,
    const GPUSimParams&         params,
    const DeviceBuffer<double>& d_x,
    const DeviceBuffer<double>& d_xhat);

// --------------------------------------------------------------------------
// gpu_compute_local_gradient_and_hessian
//
// Full gradient + hessian including barrier terms.
// Calls gpu_compute_local_gradient_and_hessian_no_barrier then adds
// node-triangle and segment-segment barrier contributions from the
// GPUBroadPhaseCache CSR per-vertex lookups.
// When params.d_hat == 0 the barrier terms are skipped and this is
// identical to the no_barrier version.
// --------------------------------------------------------------------------
std::pair<Vec3, Mat33> gpu_compute_local_gradient_and_hessian(
    int vi,
    const GPURefMesh&           mesh,
    const GPUAdjacency&         adj,
    const GPUPins&              pins,
    const GPUPinMap&            pin_map,
    const GPUSimParams&         params,
    const GPUBroadPhaseCache&   bp,
    const DeviceBuffer<double>& d_x,
    const DeviceBuffer<double>& d_xhat);

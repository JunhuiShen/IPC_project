// gpu_solver.cu
// CUDA implementation of gpu_solver.h (gpu_build_jacobi_predictions and
// gpu_parallel_commit).  The CPU stub is gpu_solver_stub.cpp — read that
// first; the algorithm is identical, only the execution model changes.
//
// STATUS: both kernels are structurally described but not yet written.
//         See TODO 1 and TODO 2 below.
//
// TODO 3 — add CUDA to LANGUAGES in CMakeLists.txt and swap stub for this file
//
// PORTING GUIDE
// -------------
// The CPU stub uses Eigen Vec3 / Mat33 and std::vector.  On the GPU those
// must be replaced with plain double arrays passed by pointer, because Eigen
// is not available in __device__ code and STL containers cannot live in
// device memory.  The math is identical — the stub is the source of truth
// for correctness.
//
// Data flow for a single outer iteration:
//
//   CPU                              GPU
//   ---                              ---
//   cudaMemcpy x → d_x              ─────────────────────────────────────
//   launch jacobi_predict_kernel ──►  one thread per vertex vi:
//                                        compute_local_newton_direction_device
//                                        build_certified_region_device
//                                        write d_predictions[vi]
//   cudaMemcpy d_predictions → predictions (CPU)
//   build_conflict_graph (CPU)
//   greedy_color_conflict_graph (CPU)
//   for each color group:
//     cudaMemcpy group → d_group
//     launch parallel_commit_kernel ► one thread per group member:
//                                        clip_step_to_certified_region_device
//                                        compute_safe_step_device (CCD/TR)
//                                        write d_commits[local_idx]
//     cudaMemcpy d_commits → commits (CPU)
//     apply_parallel_commits (CPU)    ─────────────────────────────────────

#include "gpu_solver.h"
#include <cuda_runtime.h>
#include <vector>

// --------------------------------------------------------------------------
// Device helper: 3x3 matrix inverse (cofactor method)
// --------------------------------------------------------------------------
__device__ static void mat33_inverse(const double H[9], double inv[9]) {
    double det = H[0]*(H[4]*H[8] - H[5]*H[7])
               - H[1]*(H[3]*H[8] - H[5]*H[6])
               + H[2]*(H[3]*H[7] - H[4]*H[6]);
    double id = 1.0 / det;
    inv[0] =  id * (H[4]*H[8] - H[5]*H[7]);
    inv[1] = -id * (H[1]*H[8] - H[2]*H[7]);
    inv[2] =  id * (H[1]*H[5] - H[2]*H[4]);
    inv[3] = -id * (H[3]*H[8] - H[5]*H[6]);
    inv[4] =  id * (H[0]*H[8] - H[2]*H[6]);
    inv[5] = -id * (H[0]*H[5] - H[2]*H[3]);
    inv[6] =  id * (H[3]*H[7] - H[4]*H[6]);
    inv[7] = -id * (H[0]*H[7] - H[1]*H[6]);
    inv[8] =  id * (H[0]*H[4] - H[1]*H[3]);
}

// --------------------------------------------------------------------------
// TODO 1 — jacobi_predict_kernel
//
// One thread per vertex vi.  Computes the Newton direction and certified
// region, then writes into the flat prediction arrays.
//
// Device functions to implement (port from cpu counterparts, raw doubles):
//
//   __device__ void compute_local_newton_direction_device(
//       int vi,
//       const int*    tris,          // [num_tris * 3]
//       const double* Dm_inv,        // [num_tris * 4]
//       const double* area,          // [num_tris]
//       const double* mass,          // [nv]
//       const int*    hinge_v,       // [num_hinges * 4]
//       const double* hinge_bar_theta,
//       const double* hinge_ce,
//       const int*    hinge_adj_offsets, const int* hinge_adj_hi, const int* hinge_adj_role,
//       const int*    adj_offsets, const int* adj_tri_idx, const int* adj_tri_local,
//       const int*    vnt_offsets, const int* vnt_pair_idx, const int* vnt_dof,
//       const int*    nt_data,       // [num_nt_pairs * 4]: node, tv0, tv1, tv2
//       const int*    vss_offsets, const int* vss_pair_idx, const int* vss_dof,
//       const int*    ss_data,       // [num_ss_pairs * 4]: v0, v1, v2, v3
//       const int*    pin_indices, const double* pin_targets, const int* pin_map,
//       GPUSimParams  params,
//       const double* x, const double* xhat,
//       double        g[3],          // OUT
//       double        H[9],          // OUT
//       double        delta[3]);     // OUT: H^{-1} * g
//   Reference: compute_local_newton_direction in parallel_helper.cpp
//              (elastic/inertia part: gpu_compute_local_gradient_and_hessian_no_barrier
//               in gpu_physics_stub.cpp; barrier part: barrier_energy.cpp)
//
//   __device__ void build_certified_region_device(
//       int vi, const double* x, const double delta[3],
//       const int* node_to_tris_offsets, const int* node_to_tris_data,   // CSR
//       const double* tri_box_min, const double* tri_box_max,            // [num_tris * 3]
//       const int* node_to_edges_offsets, const int* node_to_edges_data, // CSR
//       const double* edge_box_min, const double* edge_box_max,          // [num_edges * 3]
//       double d_hat,
//       double box_min[3], double box_max[3]);  // OUT: certified region AABB
//   Reference: build_certified_region_for_vertex in parallel_helper.cpp
//
// Suggested GPUPrediction layout (flat device arrays, one entry per vertex):
//   double* d_g          [nv * 3]
//   double* d_H          [nv * 9]
//   double* d_delta      [nv * 3]
//   double* d_box_min    [nv * 3]
//   double* d_box_max    [nv * 3]
//
// Kernel signature:
//
//   __global__ static void jacobi_predict_kernel(
//       int nv,
//       // mesh arrays (same layout as GPURefMesh / GPUAdjacency) ...
//       // broad-phase CSR arrays (GPUBroadPhaseCache) ...
//       // pin arrays ...
//       GPUSimParams params,
//       const double* d_x, const double* d_xhat,
//       double* d_g, double* d_H, double* d_delta,
//       double* d_box_min, double* d_box_max);
//
// Host-side wrapper gpu_build_jacobi_predictions should:
//   1. Upload x, xhat, broad-phase CSR → device (or reuse if already there)
//   2. cudaMalloc the flat prediction arrays
//   3. Launch jacobi_predict_kernel<<<(nv+255)/256, 256>>>
//   4. cudaMemcpy prediction arrays → CPU JacobiPrediction structs
//   5. cudaFree temporaries
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// TODO 2 — parallel_commit_kernel
//
// One thread per vertex in the color group.  Given the frozen predictions and
// the current positions, computes the safe Newton step for each vertex.
//
// Device functions to implement (port from cpu counterparts, raw doubles):
//
//   __device__ double clip_step_to_certified_region_device(
//       int vi, const double* x,
//       const double delta[3],
//       const double box_min[3], const double box_max[3]);
//   Reference: clip_step_to_certified_region in parallel_helper.cpp
//
//   __device__ double compute_safe_step_device(
//       int vi, const double* x, const double delta[3],
//       GPUSimParams params,
//       // broad-phase CSR query for vertex vi ...
//       const double* tri_box_min, const double* tri_box_max,
//       const double* edge_box_min, const double* edge_box_max,
//       const int* nt_data, const int* ss_data);
//   Reference: compute_safe_step_for_vertex in parallel_helper.cpp
//   (CCD: ccd.cpp; trust-region: trust_region.cpp)
//
// Flat device output arrays per group member:
//   double* d_x_after    [group_size * 3]
//   int*    d_vi         [group_size]
//   int*    d_valid      [group_size]
//
// Kernel signature:
//
//   __global__ static void parallel_commit_kernel(
//       const int* d_group, int n,          // color group
//       bool use_cached_prediction,
//       // flat prediction arrays from TODO 1 (d_g, d_H, d_delta, d_box_min, d_box_max) ...
//       // broad-phase CSR arrays ...
//       GPUSimParams params,
//       const double* d_x, const double* d_xhat,
//       double* d_x_after, int* d_valid);   // OUT
//
// Host-side wrapper gpu_parallel_commit should:
//   1. Upload group indices + prediction arrays to device
//   2. cudaMalloc d_x_after, d_valid
//   3. Launch parallel_commit_kernel<<<(n+255)/256, 256>>>
//   4. cudaMemcpy d_x_after, d_valid → CPU ParallelCommit structs
//   5. cudaFree temporaries
//   6. Return std::vector<ParallelCommit>
// --------------------------------------------------------------------------

void gpu_build_jacobi_predictions(
    const RefMesh&,
    const VertexTriangleMap&,
    const std::vector<Pin>&,
    const SimParams&,
    const std::vector<Vec3>&,
    const std::vector<Vec3>&,
    const BroadPhase::Cache&,
    std::vector<JacobiPrediction>&,
    const PinMap*)
{
    // TODO 1: implement jacobi_predict_kernel above, then replace this stub.
}

std::vector<ParallelCommit> gpu_parallel_commit(
    const std::vector<int>&,
    bool,
    const std::vector<JacobiPrediction>&,
    const RefMesh&,
    const VertexTriangleMap&,
    const std::vector<Pin>&,
    const SimParams&,
    const std::vector<Vec3>&,
    const std::vector<Vec3>&,
    const BroadPhase&,
    const PinMap*)
{
    // TODO 2: implement parallel_commit_kernel above, then replace this stub.
    return {};
}

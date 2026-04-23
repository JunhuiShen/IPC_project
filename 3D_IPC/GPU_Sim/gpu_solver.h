#pragma once
// gpu_solver.h
// Declares the two GPU-accelerated phases of the Jacobi-prediction solver.
//
// The bridge (gpu_solver_bridge.cpp) calls these two functions.
// Without CUDA the stub (gpu_solver_stub.cpp) provides CPU+OpenMP equivalents.
// With CUDA, gpu_solver.cu provides the kernel implementations.
//
// Everything else in the iteration loop — build_conflict_graph,
// greedy_color_conflict_graph, commit write-back — stays on the CPU and
// is called directly from the bridge; no GPU version is needed.

#include "gpu_mesh.h"
#include "../parallel_helper.h"   // JacobiPrediction
#include "../physics.h"
#include "../broad_phase.h"
#include <vector>

struct ParallelCommit {
    int vi = -1;
    Vec3 delta = Vec3::Zero();
    double alpha_clip = 1.0;
    double ccd_step = 1.0;
    Vec3 x_after = Vec3::Zero();
    bool valid = false;
};

// --------------------------------------------------------------------------
// gpu_build_jacobi_predictions
//
// Phase 1 of the parallel solver: compute the Newton direction and certified
// region for every vertex simultaneously.
//
// CPU stub (gpu_solver_stub.cpp): runs build_jacobi_prediction_deltas and
// build_blue_boxes.
//
// CUDA (gpu_solver.cu TODO 1): launch one thread per vertex; each thread
// runs the device-side port of compute_local_newton_direction and
// build_certified_region_for_vertex, then writes into d_predictions.
// Download d_predictions to `predictions` before returning.
// --------------------------------------------------------------------------
void gpu_build_jacobi_predictions(
    const RefMesh&                 ref_mesh,
    const VertexTriangleMap&       adj,
    const std::vector<Pin>&        pins,
    const SimParams&               params,
    const std::vector<Vec3>&       x,
    const std::vector<Vec3>&       xhat,
    const BroadPhase::Cache&       bp_cache,
    std::vector<JacobiPrediction>& predictions,   // OUT
    const PinMap*                  pin_map = nullptr);

// --------------------------------------------------------------------------
// gpu_parallel_commit
//
// Phase 2 of the parallel solver: for each vertex in a color group, compute
// the certified step (clipping to region if not color-0) then run single-node
// CCD. Returns one ParallelCommit per group member.
//
// CPU stub (gpu_solver_stub.cpp): runs the same commit math with
// #pragma omp parallel for.
//
// CUDA (gpu_solver.cu TODO 2): upload group indices + predictions to device,
// launch one thread per group member calling the device-side port of
// the commit math (raw double arrays), download commits.
// --------------------------------------------------------------------------
std::vector<ParallelCommit> gpu_parallel_commit(
    const std::vector<int>&              group,
    bool                                 use_cached_prediction,
    const std::vector<JacobiPrediction>& predictions,
    const RefMesh&                       ref_mesh,
    const VertexTriangleMap&             adj,
    const std::vector<Pin>&              pins,
    const SimParams&                     params,
    const std::vector<Vec3>&             x,
    const std::vector<Vec3>&             xhat,
    const BroadPhase&                    broad_phase,
    const PinMap*                        pin_map = nullptr);

#pragma once
// gpu_solver_bridge.h
// Provides gpu_gauss_seidel_solver: a drop-in replacement for
// global_gauss_seidel_solver that routes the inner GS sweep through the
// GPU implementation (or the CPU stub when CUDA is unavailable).
//
// The broad_phase argument is used for both initialization and CPU-side
// residual evaluation.  On each call the GPU structures are rebuilt from the
// CPU ref_mesh / adj / pins so that the caller does not need to manage any
// GPU-specific state.

#include "../physics.h"
#include "../solver.h"
#include "../broad_phase.h"
#include <vector>

SolverResult gpu_gauss_seidel_solver(
    const RefMesh&                            ref_mesh,
    const VertexTriangleMap&                  adj,
    const std::vector<Pin>&                   pins,
    const SimParams&                          params,
    std::vector<Vec3>&                        xnew,
    const std::vector<Vec3>&                  xhat,
    BroadPhase&                               broad_phase,
    const std::vector<Vec3>&                  v,
    std::vector<double>*                      residual_history = nullptr);

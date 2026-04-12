// solver.h
#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

struct SolverResult {
    double initial_residual = 0.0;
    double final_residual = 0.0;
    int iterations = 0;
    bool converged = false;
};

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh);

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
                       const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                        const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        BroadPhase& broad_phase,
                                        const std::vector<Vec3>& v,
                                        const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history = nullptr);

// override_colors: if non-null, skip the dynamic conflict-graph coloring and
// walk the supplied color_groups each iteration instead. Used by tests to
// force a specific sweep order so the parallel path can be compared
// bit-for-bit against the serial solver.
SolverResult global_gauss_seidel_solver_parallel(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                 const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, BroadPhase& broad_phase,
                                                 const std::vector<Vec3>& v, std::vector<double>* residual_history = nullptr,
                                                 const std::vector<std::vector<int>>* override_colors = nullptr);
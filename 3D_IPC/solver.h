#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

struct SolverResult {
    double initial_residual = 0.0;
    double final_residual = 0.0;
    int iterations = 0;
};

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
                       const std::vector<NodeTrianglePair>& nt_pairs,
                       const std::vector<SegmentSegmentPair>& ss_pairs);

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                        const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        BroadPhase& broad_phase,
                                        const std::vector<Vec3>& v,
                                        const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history = nullptr);

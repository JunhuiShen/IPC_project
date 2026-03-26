#pragma once
#include "physics.h"
#include <vector>

struct SolverResult {
    double initial_residual = 0.0;
    double final_residual = 0.0;
    int iterations = 0;
};

void update_one_vertex(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                       const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xhat, std::vector<Vec3>& x);

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                                        const std::vector<Pin>& pins, const SimParams& params, std::vector<Vec3>& xnew,
                                        const std::vector<Vec3>& xhat, std::vector<double>* residual_history = nullptr);

// solver.h
#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <string>
#include <vector>

struct SolverResult {
    int    iterations = 0;
    bool   converged  = false;
    bool   has_residual = false;
    double initial_residual = 0.0;
    double final_residual = 0.0;
};

inline void accumulate_solver_result(SolverResult& agg, const SolverResult& sub, bool first){
    if (first) agg.converged = sub.converged;
    else       agg.converged = agg.converged && sub.converged;
    agg.iterations += sub.iterations;
    if (sub.has_residual) {
        if (!agg.has_residual)
            agg.initial_residual = sub.initial_residual;
        agg.final_residual = std::max(
            agg.final_residual, sub.final_residual);
        agg.has_residual = true;
    }
}

// Deformable solvers implemented in solver.cpp.
SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, BroadPhase& broad_phase, const std::string& outdir = "", bool verbose = false);

SolverResult global_gauss_seidel_solver_ogc(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, const std::string& outdir = "");

// Rigid-body solver implemented in solver.cpp. To start from the previous
// collision-free state, initialize x_com_new and q_new from state and
// initialize omega_new to zero.
SolverResult global_gauss_seidel_solver_basic_rb(const RefMesh& ref_mesh, const DeformedState& state, const SimParams& params, std::vector<Vec3>& x_com_new, std::vector<Vec4>& q_new, std::vector<Vec3>& omega_new, bool verbose = false);

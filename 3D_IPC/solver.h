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
    bool   has_residual_components = false;
    double initial_cloth_residual = 0.0;
    double final_cloth_residual = 0.0;
    double initial_rigid_residual = 0.0;
    double final_rigid_residual = 0.0;
};

inline void accumulate_solver_result(SolverResult& agg, const SolverResult& sub, bool first){
    if (first) agg.converged = sub.converged;
    else       agg.converged = agg.converged && sub.converged;
    agg.iterations += sub.iterations;
    if (sub.has_residual) {
        if (!agg.has_residual) {
            agg.initial_residual = sub.initial_residual;
        }
        if (sub.has_residual_components) {
            if (!agg.has_residual_components) {
                agg.initial_cloth_residual = sub.initial_cloth_residual;
                agg.initial_rigid_residual = sub.initial_rigid_residual;
            }
            agg.final_cloth_residual = std::max(agg.final_cloth_residual, sub.final_cloth_residual);
            agg.final_rigid_residual = std::max(agg.final_rigid_residual, sub.final_rigid_residual);
            agg.initial_residual = agg.initial_cloth_residual + agg.initial_rigid_residual;
            agg.final_residual = agg.final_cloth_residual + agg.final_rigid_residual;
            agg.has_residual_components = true;
        } else {
            agg.final_residual = std::max(agg.final_residual, sub.final_residual);
        }
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

// General basic solver. Deformable nodes (node_to_rb == -1) are advanced as
// independent particle blocks, while every rigid body is advanced through its
// COM and angular-velocity blocks. All blocks share one live position array
// and one contact cache, enabling cloth-cloth, cloth-rigid, and rigid-rigid
// contact in the same Gauss-Seidel solve.
SolverResult global_gauss_seidel_solver_basic_general(
    const RefMesh& ref_mesh, const DeformedState& state,
    const VertexTriangleMap& adj, const std::vector<Pin>& pins,
    const SimParams& params, std::vector<Vec3>& xnew,
    const std::vector<Vec3>& xhat,
    std::vector<Vec3>& x_com_new, std::vector<Vec4>& q_new,
    std::vector<Vec3>& omega_new, BroadPhase& broad_phase,
    const std::string& outdir = "", bool verbose = false);

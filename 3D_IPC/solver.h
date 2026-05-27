// solver.h
#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <string>
#include <vector>

struct SolverResult {
    int    iterations = 0;
    bool   converged  = false;
};

inline void accumulate_solver_result(SolverResult& agg, const SolverResult& sub, bool first){
    if (first) agg.converged = sub.converged;
    else       agg.converged = agg.converged && sub.converged;
    agg.iterations += sub.iterations;
}

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh);

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                              const std::vector<Pin>& pins, const SimParams& params,
                                              std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                              const std::vector<Vec3>& v,
                                              const std::string& outdir = "",
                                              bool verbose = false);

SolverResult global_gauss_seidel_solver_ogc(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                            const std::vector<Pin>& pins, const SimParams& params,
                                            std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                            const std::vector<Vec3>& v,
                                            const std::string& outdir = "");

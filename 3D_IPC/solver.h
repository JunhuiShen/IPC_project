// solver.h
#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <string>
#include <vector>

struct SolverResult {
    double initial_residual = 0.0;
    double final_residual   = 0.0;
    int    iterations       = 0;
    bool   converged        = false;
};

// Fold one substep's result into the frame aggregate. `first` preserves the
// leading substep's initial_residual and seeds `converged` before AND-folding.
inline void accumulate_solver_result(SolverResult& agg, const SolverResult& sub, bool first){
    if (first) {
        agg.initial_residual = sub.initial_residual;
        agg.converged        = sub.converged;
    } else {
        agg.converged = agg.converged && sub.converged;
    }
    agg.final_residual       = sub.final_residual;
    agg.iterations          += sub.iterations;
}

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh);
std::vector<Vec3> trust_region_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, double d_hat);

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
                       const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                              const std::vector<Pin>& pins, const SimParams& params,
                                              std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                              const std::vector<Vec3>& v,
                                              std::vector<double>* residual_history = nullptr,
                                              const std::string& outdir = "");

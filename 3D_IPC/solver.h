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

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase = nullptr);

std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                           const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                           const SimParams& params);

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                              const std::vector<Pin>& pins, const SimParams& params,
                                              std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                              const std::vector<Vec3>& v,
                                              BroadPhase& broad_phase,
                                              const std::string& outdir = "",
                                              bool verbose = false);

SolverResult global_gauss_seidel_solver_ogc(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                            const std::vector<Pin>& pins, const SimParams& params,
                                            std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                            const std::vector<Vec3>& v,
                                            const std::string& outdir = "");


namespace rb_solver {

struct ComUpdate {
    int rb = -1;
    Vec3 dx_com = Vec3::Zero(); // Newton direction H^{-1}g; subtract when committing.
    double step = 1.0;          // Reserved for rigid-body CCD/trust-region clipping.
};

struct OrientationUpdate {
    int rb = -1;
    Vec3 domega = Vec3::Zero(); // Newton direction H^{-1}g in omega coordinates.
    double step = 1.0;          // Reserved for rotational CCD/trust-region clipping.
};

ComUpdate compute_com_update(
    int rb, const DeformedState& state, const Vec3& x_com,
    const SimParams& params, double dt, double total_mass);

OrientationUpdate compute_orientation_update(
    int rb, const DeformedState& state, const RefMesh& ref_mesh,
    const Vec3& omega, double dt);

void commit_com_update(
    const ComUpdate& update, std::vector<Vec3>& x_coms);

void commit_orientation_update(
    const OrientationUpdate& update, const DeformedState& state,
    std::vector<Vec4>& orientations, std::vector<Vec3>& omega,
    double dt);

} // namespace rb_solver


SolverResult global_gauss_seidel_solver_basic_rb(
    const RefMesh& ref_mesh, const DeformedState& state,
    const SimParams& params, std::vector<Vec3>& x_coms,
    std::vector<Vec4>& orientations, std::vector<Vec3>& omega,
    bool verbose = false);

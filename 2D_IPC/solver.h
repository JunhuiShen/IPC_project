#pragma once

#include "broad_phase.h"
#include "physics.h"
#include <vector>

struct SolveResult {
    double final_residual;
    int    iterations_used;
    bool   has_rigid_residual = false;
    double final_rigid_residual = 0.0;
};

Vec trivial_initial_guess(const DeformedState& state);

double ccd_initial_guess_safe_step(const Vec& x, const Vec& v,
                                   const std::vector<NodeSegmentPair>& pairs,
                                   double dt, double eta = 0.9);

Vec ccd_initial_guess(const DeformedState& state, const RefMesh& ref_mesh,
                      const std::vector<Pin>& pins, double dt, double eta);

Vec verlet_initial_guess(const DeformedState& state, const RefMesh& ref_mesh,
                         const std::vector<Pin>& pins, double dt, double eta,
                         const Vec2& gravity);

SolveResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const DeformedState& state, const Vec& xhat,
                                              Vec& xnew, const SimParams2D& params, BroadPhase& broad_phase, std::vector<double>* residual_history = nullptr);

// y_current and theta_current are in/out: initialise to starting DOFs before
// calling; they hold the converged values on return.
SolveResult global_gauss_seidel_solver_rb(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const DeformedState& state, Vec& xnew,
                                           Vec& y_current, std::vector<double>& theta_current, const SimParams2D& params, std::vector<double>* residual_history = nullptr);

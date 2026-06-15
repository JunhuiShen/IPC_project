#pragma once

#include "broad_phase.h"
#include "physics.h"
#include <vector>

struct SolveResult {
    double final_residual;
    int    iterations_used;
};

SolveResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh,
                                              const std::vector<Pin>& pins,
                                              const DeformedState& state,
                                              const Vec& xhat,
                                              Vec& x,
                                              double dt, double k_spring, const Vec2& g_accel,
                                              double d_hat, double k_barrier,
                                              int max_iters, double tol_abs, double eta,
                                              double node_box_min, double node_box_max, int node_box_update_count,
                                              BroadPhase& broad_phase, bool use_ccd_step_policy,
                                              bool use_parallel,
                                              std::vector<double>* residual_history = nullptr);

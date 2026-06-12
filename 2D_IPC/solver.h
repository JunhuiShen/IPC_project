#pragma once

#include "ipc_math.h"
#include "physics.h"
#include "chain.h"
#include "broad_phase/broad_phase.h"
#include <vector>

struct BlockView {
    Vec*                        x;
    const Vec*                  xhat;
    const Vec*                  xpin;
    const std::vector<double>*  mass;
    const RefMesh*              ref_mesh;
    int                         rest_length_offset;
    const std::vector<char>*    is_pinned;
    int                         offset;

    int size() const { return static_cast<int>(mass->size()); }
};

struct SolveResult {
    double final_residual;
    int    iterations_used;
};

SolveResult global_gauss_seidel_solver_basic(std::vector<BlockView>& blocks,
                                              Vec& x_global, const Vec& v_vel_global,
                                              double dt, double k, const Vec2& g_accel,
                                              double dhat, double k_barrier,
                                              int max_iters, double tol_abs, double eta,
                                              BroadPhase& broad_phase, bool use_ccd_step_policy,
                                              std::vector<double>* residual_history = nullptr);

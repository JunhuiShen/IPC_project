#pragma once

#include "ipc_math.h"
#include "physics.h"
#include "broad_phase/broad_phase.h"
#include "step_filter/step_filter.h"
#include <vector>

struct BlockView {
    Vec*                        x;
    const Vec*                  xhat;
    const Vec*                  xpin;
    const std::vector<double>*  mass;
    const std::vector<double>*  L;
    const std::vector<char>*    is_pinned;
    int                         offset;

    int size() const { return static_cast<int>(mass->size()); }
};

struct SolveResult {
    double final_residual;
    int    iterations_used;
};

SolveResult solve(std::vector<BlockView>& blocks,
                  Vec& x_global, const Vec& v_vel_global,
                  double dt, double k, const Vec2& g_accel,
                  double dhat, int max_iters, double tol_abs, double eta,
                  BroadPhase& broad_phase, StepFilter& step_filter,
                  std::vector<double>* residual_history = nullptr);

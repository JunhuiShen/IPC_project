#pragma once

#include "broad_phase.h"
#include "initial_guess/initial_guess.h"
#include "mesh.h"
#include "solver.h"
#include "state.h"

#include <algorithm>
#include <functional>

struct SimParams2D {
    double frame_dt = 1.0 / 30.0;
    int    substeps = 3;
    double k_spring = 1000.0;
    double k_barrier = 100.0;
    Vec2   gravity{0.0, -9.81};
    double d_hat = 0.005;
    double tol_abs = 1e-6;
    int    max_substep_iters = 500;
    double eta = 0.9;
    bool   use_parallel = false;
    double node_box_min = 0.001;
    double node_box_max = 0.01;
    int    node_box_update_count = 1;
    bool   use_ccd_step_policy = true;
    InitialGuessType initial_guess_type = InitialGuessType::CCD;

    double substep_dt() const {
        return frame_dt / static_cast<double>(std::max(1, substeps));
    }
};

struct AdvanceResult2D {
    double first_initial_residual = 0.0;
    double max_final_residual = 0.0;
    int    total_iterations = 0;
    int    substeps_completed = 0;
};

using SubstepCallback2D = std::function<void(int, const Vec&)>;

AdvanceResult2D advance_one_frame(
        State2D& state, const RefMesh& ref_mesh,
        const SimParams2D& params, BroadPhase& broad_phase,
        int frame_index, SubstepCallback2D on_substep = nullptr);

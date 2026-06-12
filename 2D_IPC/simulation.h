#pragma once

#include "broad_phase/broad_phase.h"
#include "chain.h"
#include "initial_guess/initial_guess.h"
#include "solver.h"
#include "visualization.h"
#include <algorithm>
#include <functional>
#include <vector>

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
    bool   use_ccd_step_policy = true;
    bool   write_substeps = false;
    initial_guess::Type initial_guess_type = initial_guess::Type::CCD;

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

inline std::vector<char> build_segment_valid_from_ref_mesh(const RefMesh& ref_mesh) {
    std::vector<char> segment_valid(std::max(0, ref_mesh.num_positions - 1), 0);
    for (const auto& e : ref_mesh.edges) {
        if (e.second == e.first + 1 && e.first >= 0 && e.first < static_cast<int>(segment_valid.size())) {
            segment_valid[e.first] = 1;
        }
    }
    return segment_valid;
}

inline void build_x_combined_from_chains(Vec& x_combined,
                                         const std::vector<Chain>& chains,
                                         const std::vector<int>& offsets) {
    x_combined.assign(2 * static_cast<int>(x_combined.size() / 2), 0.0);
    for (std::size_t b = 0; b < chains.size(); ++b) {
        scatter_positions(x_combined, chains[b].x, offsets[b], chains[b].N);
    }
}

inline AdvanceResult2D advance_one_frame(std::vector<Chain>& chains,
                                         const RefMesh& ref_mesh,
                                         const std::vector<int>& offsets,
                                         const SimParams2D& params,
                                         BroadPhase& broad_phase,
                                         int frame_index,
                                         SubstepCallback2D on_substep = nullptr) {
    const int nblocks = static_cast<int>(chains.size());
    const int total_nodes = ref_mesh.num_positions;
    const double dt = params.substep_dt();
    const std::vector<char> segment_valid = build_segment_valid_from_ref_mesh(ref_mesh);

    AdvanceResult2D aggregate;
    Vec x_combined(2 * total_nodes, 0.0);
    Vec v_combined(2 * total_nodes, 0.0);
    std::vector<Vec> xnew_blocks(nblocks);

    auto make_guess_blocks = [&]() {
        std::vector<initial_guess::BlockRef> guess_blocks;
        guess_blocks.reserve(nblocks);
        for (int b = 0; b < nblocks; ++b) {
            guess_blocks.push_back({&chains[b], &xnew_blocks[b], offsets[b]});
        }
        return guess_blocks;
    };

    auto make_solver_blocks = [&]() {
        std::vector<BlockView> blocks;
        blocks.reserve(nblocks);
        for (int b = 0; b < nblocks; ++b) {
            blocks.push_back({&xnew_blocks[b], &chains[b].xhat, &chains[b].xpin,
                              &chains[b].mass, &ref_mesh, ref_mesh.chain_rest_offsets[b],
                              &chains[b].is_pinned, offsets[b]});
        }
        return blocks;
    };

    for (int sub = 0; sub < std::max(1, params.substeps); ++sub) {
        for (int b = 0; b < nblocks; ++b) {
            build_xhat(chains[b], dt);
            xnew_blocks[b] = chains[b].x;
        }

        std::vector<initial_guess::BlockRef> guess_blocks = make_guess_blocks();
        initial_guess::apply(params.initial_guess_type, guess_blocks, x_combined, v_combined,
                             segment_valid, dt, params.d_hat, params.eta);

        std::vector<BlockView> blocks = make_solver_blocks();
        std::vector<double> res_hist;
        SolveResult sub_result = global_gauss_seidel_solver_basic(
                blocks, x_combined, v_combined, dt, params.k_spring, params.gravity,
                params.d_hat, params.k_barrier, params.max_substep_iters, params.tol_abs, params.eta,
                broad_phase, params.use_ccd_step_policy, &res_hist);

        if (aggregate.substeps_completed == 0 && !res_hist.empty()) {
            aggregate.first_initial_residual = res_hist.front();
        }
        aggregate.max_final_residual = std::max(aggregate.max_final_residual, sub_result.final_residual);
        aggregate.total_iterations += sub_result.iterations_used;
        aggregate.substeps_completed += 1;

        for (int b = 0; b < nblocks; ++b) {
            update_velocity(chains[b], xnew_blocks[b], dt);
        }

        if (on_substep) {
            build_x_combined_from_chains(x_combined, chains, offsets);
            const int global_substep = (frame_index - 1) * std::max(1, params.substeps) + sub;
            on_substep(global_substep, x_combined);
        }
    }

    return aggregate;
}

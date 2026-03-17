#include "ipc_math.h"
#include "chain.h"
#include "visualization.h"
#include "solver.h"
#include "broad_phase/bvh.h"
#include "step_filter/ccd.h"
#include "step_filter/trust_region.h"
#include "initial_guess/initial_guess.h"
#include "example.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace fs = std::__fs::filesystem;
using namespace math;

namespace {

    enum class StepPolicy {
        CCD,
        TrustRegion
    };

} // namespace

int main() {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    const std::string outdir = "frames_spring_IPC_bvh";

    if (fs::exists(outdir)) {
        fs::remove_all(outdir);
    }
    fs::create_directories(outdir);

    // ------------------------------------------------------
    // Parameters
    // ------------------------------------------------------
    const double dt = 1.0 / 30.0;
    const Vec2 g_accel{0.0, -9.81};
    const double k_spring = 20.0;
    const int max_global_iters = 5000;
    const double tol_abs = 1e-6;
    const double dhat = 0.1;
    const int number_of_nodes = 100; // number of nodes per chain

    // ------------------------------------------------------
    // Strategy choices
    // ------------------------------------------------------
//     const ExampleType example_type = ExampleType::Example1;
    const ExampleType example_type = ExampleType::Example2;

    auto broad_phase = std::make_unique<BVHBroadPhase>();

    const StepPolicy filtering_step_policy = StepPolicy::CCD;
//     const StepPolicy filtering_step_policy = StepPolicy::TrustRegion;

    const initial_guess::Type initial_guess_type = initial_guess::Type::CCD;
//     const initial_guess::Type initial_guess_type = initial_guess::Type::TrustRegion;
    // const initial_guess::Type initial_guess_type = initial_guess::Type::Trivial;
    // const initial_guess::Type initial_guess_type = initial_guess::Type::Affine;

    double eta = 0.9;
    if (initial_guess_type == initial_guess::Type::TrustRegion && filtering_step_policy == StepPolicy::TrustRegion) {
        eta = 0.4;
    } else if (initial_guess_type == initial_guess::Type::CCD && filtering_step_policy == StepPolicy::CCD) {
        eta = 0.9;
    }

    // Reset CCD stats
    step_filter::ccd::reset_stats();

    // ------------------------------------------------------
    // Build example
    // ------------------------------------------------------
    ExampleScene scene = build_example(example_type, number_of_nodes);
    std::vector<Chain> chains = std::move(scene.chains);
    const int total_frames = scene.total_frames;

    // ------------------------------------------------------
    // Global indexing data
    // ------------------------------------------------------
    const int nblocks = static_cast<int>(chains.size());

    std::vector<int> offsets(nblocks, 0);
    for (int b = 1; b < nblocks; ++b) {
        offsets[b] = offsets[b - 1] + chains[b - 1].N;
    }

    int total_nodes = 0;
    for (const auto& c : chains) {
        total_nodes += c.N;
    }

    std::vector<char> segment_valid(std::max(0, total_nodes - 1), 0);
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i + 1 < chains[b].N; ++i) {
            segment_valid[offsets[b] + i] = 1;
        }
    }

    std::vector<std::pair<int, int>> edges_combined;
    for (int b = 0; b < nblocks; ++b) {
        for (const auto& e : chains[b].edges) {
            edges_combined.emplace_back(e.first + offsets[b], e.second + offsets[b]);
        }
    }

    // ------------------------------------------------------
    // Global state buffers
    // ------------------------------------------------------
    Vec x_combined(2 * total_nodes, 0.0);
    Vec v_combined(2 * total_nodes, 0.0);

    std::vector<Vec> xnew_blocks(nblocks);
    for (int b = 0; b < nblocks; ++b) {
        xnew_blocks[b] = chains[b].x;
    }

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
            blocks.push_back({&xnew_blocks[b], &chains[b].xhat, &chains[b].xpin, &chains[b].mass,
                              &chains[b].rest_lengths, &chains[b].is_pinned, offsets[b]});
        }
        return blocks;
    };

    // ------------------------------------------------------
    // Initial export
    // ------------------------------------------------------
    {
        std::vector<initial_guess::BlockRef> guess_blocks = make_guess_blocks();
        initial_guess::build_x_combined_from_current_positions(x_combined, guess_blocks);
        export_frame(outdir, 1, x_combined, edges_combined);
    }

    double max_global_residual = 0.0;
    int sum_global_iters_used = 0;

    // ------------------------------------------------------
    // Time stepping
    // ------------------------------------------------------
    for (int frame = 2; frame <= total_frames + 1; ++frame) {
        for (int b = 0; b < nblocks; ++b) {
            build_xhat(chains[b], dt);
        }

        std::vector<initial_guess::BlockRef> guess_blocks = make_guess_blocks();
        initial_guess::apply(initial_guess_type, guess_blocks, x_combined, v_combined,
                             segment_valid, dt, dhat, eta);

        std::vector<BlockView> blocks = make_solver_blocks();

        std::vector<double> res_hist;
        SolveResult result{};

        if (filtering_step_policy == StepPolicy::CCD) {
            CCDFilter filter;
            result = solve(blocks, x_combined, v_combined, dt, k_spring, g_accel,
                           dhat, max_global_iters, tol_abs, eta,
                           *broad_phase, filter, &res_hist);
        } else {
            TrustRegionFilter filter;
            result = solve(blocks, x_combined, v_combined, dt, k_spring, g_accel,
                           dhat, max_global_iters, tol_abs, eta,
                           *broad_phase, filter, &res_hist);
        }

        const double initial_residual = res_hist.empty() ? 0.0 : res_hist.front();
        const double global_residual = result.final_residual;
        const int iters_used = result.iterations_used;

        max_global_residual = std::max(max_global_residual, global_residual);
        sum_global_iters_used += iters_used;

        for (int b = 0; b < nblocks; ++b) {
            update_velocity(chains[b], xnew_blocks[b], dt);
        }

        initial_guess::build_x_combined_from_current_positions(x_combined, guess_blocks);
        export_frame(outdir, frame, x_combined, edges_combined);

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual=" << std::scientific << initial_residual
                  << " | final_residual="   << std::scientific << global_residual
                  << " | global_iters="     << std::setw(3) << iters_used
                  << '\n';
    }

    auto t_end = clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    const double avg_global_iters_used =
            (total_frames > 0) ? double(sum_global_iters_used) / total_frames : 0.0;

    std::cout << "\n===== Simulation Summary =====\n";
    std::cout << "max_global_residual = " << std::scientific << max_global_residual << "\n";
    std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
    std::cout << "total runtime = " << elapsed.count() << " seconds\n";
    std::cout << "ccd_total_tests = " << step_filter::ccd::total_tests << "\n";
    std::cout << "ccd_total_collisions = " << step_filter::ccd::total_collisions << "\n";

    return 0;
}

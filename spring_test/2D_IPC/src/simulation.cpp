#include "ipc_math.h"
#include "chain.h"
#include "visualization.h"
#include "solver.h"
#include "broad_phase/bvh.h"
#include "step_filter/ccd.h"
#include "step_filter/trust_region.h"
#include "initial_guess/ccd.h"
#include "initial_guess/trust_region.h"
#include "initial_guess/trivial.h"
#include "initial_guess/affine.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;
using namespace math;

int main() {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    const std::string outdir = "frames_spring_IPC_bvh";
    fs::create_directory(outdir);

    // ── Parameters ──────────────────────────────────────────────────────────
    const double dt             = 1.0 / 30.0;
    const Vec2   g_accel        = {0.0, -9.81};
    const double k_spring       = 20.0;
    const int    total_frames   = 150;
    const int    max_iters      = 800;
    const double tol_abs        = 1e-6;
    const double dhat           = 0.1;
    const int    N_nodes        = 11;
    const double eta            = 0.9;   // safety margin (CCD mode)

    // ── Strategy selection (swap any line to change behavior) ────────────────
    auto broad_phase   = std::make_unique<BVHBroadPhase>();
    auto step_filter   = std::make_unique<CCDFilter>();
    auto initial_guess = std::make_unique<CCDGuess>();

    // ── Build chains ─────────────────────────────────────────────────────────
    Chain left  = make_chain({-0.1,  1.5}, {-0.1, -1.5}, N_nodes, 0.08);
    Chain right = make_chain({ 0.1,  1.5}, { 0.1, -1.5}, N_nodes, 0.08);

    // Pin the top node of each chain
    set_xi(left.xpin,  0, get_xi(left.x,  0));
    set_xi(right.xpin, 0, get_xi(right.x, 0));

    // Initial velocities: chains rush toward each other
    for (int i = 0; i < left.N;  ++i) set_xi(left.v,  i, {-6.0, 0.0});
    for (int i = 0; i < right.N; ++i) set_xi(right.v, i, { 6.0, 0.0});

    // ── Combined edge list (used only for export) ─────────────────────────────
    std::vector<std::pair<int,int>> edges_combined = left.edges;
    for (const auto& e : right.edges)
        edges_combined.emplace_back(e.first + left.N, e.second + left.N);

    // ── Global state buffers ──────────────────────────────────────────────────
    const int total_nodes = left.N + right.N;
    Vec x_combined(2 * total_nodes, 0.0);
    Vec v_combined(2 * total_nodes, 0.0);
    Vec xnew_left  = left.x;
    Vec xnew_right = right.x;

    combine_positions(x_combined, left.x, right.x, left.N, right.N);
    export_frame(outdir, 1, x_combined, edges_combined);

    // ── Statistics ────────────────────────────────────────────────────────────
    double max_residual    = 0.0;
    int    sum_iters       = 0;

    // ── Time-stepping loop ────────────────────────────────────────────────────
    for (int frame = 2; frame <= total_frames + 1; ++frame) {

        // Linear extrapolation target
        build_xhat(left,  dt);
        build_xhat(right, dt);

        // Initial guess for the nonlinear solve
        initial_guess->apply(left, right,
                             xnew_left, xnew_right,
                             x_combined, v_combined,
                             dt, eta);

        combine_positions(x_combined, xnew_left, xnew_right, left.N, right.N);

        // Solver block views (non-owning pointers into chain data)
        std::vector<BlockView> blocks = {
            {&xnew_left,  &left.xhat,  &left.xpin,  &left.mass,  &left.rest_lengths,  0},
            {&xnew_right, &right.xhat, &right.xpin, &right.mass, &right.rest_lengths, left.N},
        };

        // Nonlinear Gauss-Seidel solve
        std::vector<double> res_hist;
        auto result = solve(blocks, x_combined, v_combined,
                            dt, k_spring, g_accel,
                            dhat, max_iters, tol_abs, eta,
                            *broad_phase, *step_filter, &res_hist);

        max_residual = std::max(max_residual, result.final_residual);
        sum_iters   += result.iterations_used;

        // Velocity update and position commit
        update_velocity(left,  xnew_left,  dt);
        update_velocity(right, xnew_right, dt);

        combine_positions(x_combined, left.x, right.x, left.N, right.N);
        export_frame(outdir, frame, x_combined, edges_combined);

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual=" << std::scientific << res_hist.front()
                  << " | final_residual="   << std::scientific << result.final_residual
                  << " | iters=" << std::setw(3) << result.iterations_used
                  << '\n';
    }

    auto elapsed = std::chrono::duration<double>(clock::now() - t_start).count();

    std::cout << "\n===== Simulation Summary =====\n"
              << "max_residual = " << std::scientific << max_residual << "\n"
              << "avg_iters    = " << std::fixed << (1.0 * sum_iters / total_frames) << "\n"
              << "runtime      = " << elapsed << " seconds\n";

    return 0;
}

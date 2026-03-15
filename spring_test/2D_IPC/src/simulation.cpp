//#include "ipc_math.h"
//#include "chain.h"
//#include "physics.h"
//#include "visualization.h"
//#include "solver.h"
//#include "broad_phase/bvh.h"
//#include "step_filter/ccd.h"
//#include "step_filter/trust_region.h"
//#include "initial_guess/initial_guess.h"
//
//#include <iostream>
//#include <iomanip>
//#include <chrono>
//#include <filesystem>
//#include <memory>
//#include <vector>
//#include <algorithm>
//#include <stdexcept>
//#include <cmath>
//
//namespace fs = std::__fs::filesystem;
//using namespace math;
//using namespace physics;
//
//namespace {
//
//    enum class ExampleType {
//        Example1,
//        Example2
//    };
//
//    enum class StepPolicy {
//        CCD,
//        TrustRegion
//    };
//
//    struct GuessBlock {
//        Chain* chain;
//        Vec* xnew;
//        int offset;
//    };
//
//    // ------------------------------------------------------
//    // Global assembly helpers
//    // ------------------------------------------------------
//
//    inline int total_nodes(const std::vector<GuessBlock>& blocks) {
//        int total = 0;
//        for (const auto& b : blocks)
//            total += b.chain->N;
//        return total;
//    }
//
//    inline void scatter_block_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block) {
//        for (int i = 0; i < N_block; ++i)
//            set_xi(x_combined, offset + i, get_xi(x_block, i));
//    }
//
//    inline void scatter_block_velocities(Vec& v_combined, const Vec& v_block, int offset, int N_block) {
//        for (int i = 0; i < N_block; ++i)
//            set_xi(v_combined, offset + i, get_xi(v_block, i));
//    }
//
//    inline void build_x_combined_from_current_positions(Vec& x_combined,
//                                                        const std::vector<GuessBlock>& blocks) {
//        for (const auto& b : blocks)
//            scatter_block_positions(x_combined, b.chain->x, b.offset, b.chain->N);
//    }
//
//    inline void build_x_combined_from_xnew(Vec& x_combined,
//                                           const std::vector<GuessBlock>& blocks) {
//        for (const auto& b : blocks)
//            scatter_block_positions(x_combined, *b.xnew, b.offset, b.chain->N);
//    }
//
//    inline void build_v_combined_from_chain_velocities(Vec& v_combined,
//                                                       const std::vector<GuessBlock>& blocks) {
//        for (const auto& b : blocks)
//            scatter_block_velocities(v_combined, b.chain->v, b.offset, b.chain->N);
//    }
//
//    // ------------------------------------------------------
//    // Local initial-guess helpers
//    // These restore the behavior of your monolithic code.
//    // ------------------------------------------------------
//
//    struct AffineParams {
//        double omega;
//        Vec2 vhat;
//        Vec2 xcom;
//    };
//
//    inline AffineParams compute_affine_params_global(const std::vector<GuessBlock>& blocks) {
//        Vec2 xcom{0.0, 0.0};
//        double M = 0.0;
//
//        for (const auto& b : blocks) {
//            const Chain& c = *b.chain;
//            for (int i = 0; i < c.N; ++i) {
//                if (c.is_pinned[i]) continue;
//                Vec2 xi = get_xi(c.x, i);
//                xcom.x += c.mass[i] * xi.x;
//                xcom.y += c.mass[i] * xi.y;
//                M += c.mass[i];
//            }
//        }
//
//        if (M <= 1e-12)
//            return {0.0, {0.0, 0.0}, {0.0, 0.0}};
//
//        xcom.x /= M;
//        xcom.y /= M;
//
//        double G[3][3] = {{0.0}};
//        double bvec[3] = {0.0, 0.0, 0.0};
//
//        for (const auto& br : blocks) {
//            const Chain& c = *br.chain;
//            for (int i = 0; i < c.N; ++i) {
//                if (c.is_pinned[i]) continue;
//
//                Vec2 Xi = get_xi(c.x, i);
//                Vec2 Vi = get_xi(c.v, i);
//                Vec2 d{Xi.x - xcom.x, Xi.y - xcom.y};
//
//                Vec2 U1{-d.y, d.x};
//                Vec2 U2{1.0, 0.0};
//                Vec2 U3{0.0, 1.0};
//                Vec2 U[3] = {U1, U2, U3};
//
//                double w = c.mass[i];
//
//                for (int k = 0; k < 3; ++k) {
//                    bvec[k] += w * (U[k].x * Vi.x + U[k].y * Vi.y);
//                    for (int j = 0; j < 3; ++j) {
//                        G[k][j] += w * (U[k].x * U[j].x + U[k].y * U[j].y);
//                    }
//                }
//            }
//        }
//
//        double omega  = (std::abs(G[0][0]) > 1e-12) ? bvec[0] / G[0][0] : 0.0;
//        double vhat_x = (std::abs(G[1][1]) > 1e-12) ? bvec[1] / G[1][1] : 0.0;
//        double vhat_y = (std::abs(G[2][2]) > 1e-12) ? bvec[2] / G[2][2] : 0.0;
//
//        return {omega, {vhat_x, vhat_y}, xcom};
//    }
//
//    inline Vec2 affine_velocity_at(const AffineParams& ap, const Vec2& x) {
//        Vec2 d{x.x - ap.xcom.x, x.y - ap.xcom.y};
//        return {ap.vhat.x - ap.omega * d.y, ap.vhat.y + ap.omega * d.x};
//    }
//
//    inline void build_v_combined_from_affine(Vec& v_combined,
//                                             const std::vector<GuessBlock>& blocks,
//                                             const AffineParams& ap) {
//        for (const auto& b : blocks) {
//            const Chain& c = *b.chain;
//            for (int i = 0; i < c.N; ++i) {
//                Vec2 xi = get_xi(c.x, i);
//                Vec2 vi = c.is_pinned[i] ? Vec2{0.0, 0.0} : affine_velocity_at(ap, xi);
//                set_xi(v_combined, b.offset + i, vi);
//            }
//        }
//    }
//
//    inline double ccd_point_segment_safe_step(const Vec2& x1, const Vec2& dx1,
//                                              const Vec2& x2, const Vec2& dx2,
//                                              const Vec2& x3, const Vec2& dx3,
//                                              double eta = 0.9, double eps = 1e-12) {
//        Vec2 x21 = sub(x1, x2);
//        Vec2 x32 = sub(x3, x2);
//        Vec2 dx21 = sub(dx1, dx2);
//        Vec2 dx32 = sub(dx3, dx2);
//
//        double a = cross(dx32, dx21);
//        double b = cross(dx32, x21) + cross(x32, dx21);
//        double c = cross(x32, x21);
//
//        double t_candidates[2];
//        int num_roots = 0;
//
//        if (std::fabs(a) < eps) {
//            if (std::fabs(b) < eps) return 1.0;
//            double t = -c / b;
//            if (t >= 0.0 && t <= 1.0)
//                t_candidates[num_roots++] = t;
//        } else {
//            double D = b * b - 4.0 * a * c;
//            if (D < 0.0) return 1.0;
//
//            double sqrtD = std::sqrt(std::max(D, 0.0));
//            double s = (b >= 0.0) ? 1.0 : -1.0;
//            double q = -0.5 * (b + s * sqrtD);
//
//            double t1 = q / a;
//            double t2 = c / q;
//
//            if (t1 >= 0.0 && t1 <= 1.0)
//                t_candidates[num_roots++] = t1;
//            if (t2 >= 0.0 && t2 <= 1.0)
//                t_candidates[num_roots++] = t2;
//        }
//
//        if (num_roots == 0) return 1.0;
//
//        double t_star = t_candidates[0];
//        if (num_roots == 2 && t_candidates[1] < t_star)
//            t_star = t_candidates[1];
//
//        Vec2 x1t = add(x1, scale(dx1, t_star));
//        Vec2 x2t = add(x2, scale(dx2, t_star));
//        Vec2 x3t = add(x3, scale(dx3, t_star));
//
//        Vec2 seg = sub(x3t, x2t);
//        Vec2 rel = sub(x1t, x2t);
//
//        double seg_len2 = norm2(seg);
//        if (seg_len2 < eps) return 1.0;
//
//        double s_param = dot(rel, seg) / seg_len2;
//        if (s_param < 0.0 || s_param > 1.0) return 1.0;
//
//        if (t_star <= 1e-12) return 0.0;
//        return eta * t_star;
//    }
//
//    inline double trust_region_weight_local(const Vec2& xi, const Vec2& dxi,
//                                            const Vec2& xj, const Vec2& dxj,
//                                            const Vec2& xk, const Vec2& dxk,
//                                            double eta = 0.4) {
//        double s;
//        Vec2 p{}, r{};
//        double d0 = node_segment_distance(xi, xj, xk, s, p, r);
//
//        constexpr double eps = 1e-12;
//        d0 = std::max(d0, eps);
//
//        const double M = norm(dxi) + norm(dxj) + norm(dxk);
//
//        if (M <= eps)
//            return 1.0;
//
//        const double w = eta * d0 / M;
//        return std::max(0.0, std::min(1.0, w));
//    }
//
//    inline double compute_initial_guess_ccd_step(const Vec& x_combined, const Vec& v_combined,
//                                                 const std::vector<NodeSegmentPair>& pairs,
//                                                 double dt, double eta) {
//        double omega = 1.0;
//
//        for (const auto& c : pairs) {
//            Vec2 xi = get_xi(x_combined, c.node);
//            Vec2 xj = get_xi(x_combined, c.seg0);
//            Vec2 xk = get_xi(x_combined, c.seg1);
//
//            Vec2 vi = get_xi(v_combined, c.node);
//            Vec2 vj = get_xi(v_combined, c.seg0);
//            Vec2 vk = get_xi(v_combined, c.seg1);
//
//            Vec2 dxi{dt * vi.x, dt * vi.y};
//            Vec2 dxj{dt * vj.x, dt * vj.y};
//            Vec2 dxk{dt * vk.x, dt * vk.y};
//
//            omega = std::min(omega, ccd_point_segment_safe_step(xi, dxi, xj, dxj, xk, dxk, eta));
//            if (omega <= 0.0)
//                return 0.0;
//        }
//
//        return omega;
//    }
//
//    inline double compute_initial_guess_trust_region_step(const Vec& x_combined, const Vec& v_combined,
//                                                          const std::vector<NodeSegmentPair>& pairs,
//                                                          double dt, double eta) {
//        double omega = 1.0;
//
//        for (const auto& c : pairs) {
//            Vec2 xi = get_xi(x_combined, c.node);
//            Vec2 xj = get_xi(x_combined, c.seg0);
//            Vec2 xk = get_xi(x_combined, c.seg1);
//
//            Vec2 vi = get_xi(v_combined, c.node);
//            Vec2 vj = get_xi(v_combined, c.seg0);
//            Vec2 vk = get_xi(v_combined, c.seg1);
//
//            Vec2 dxi{dt * vi.x, dt * vi.y};
//            Vec2 dxj{dt * vj.x, dt * vj.y};
//            Vec2 dxk{dt * vk.x, dt * vk.y};
//
//            omega = std::min(omega, trust_region_weight_local(xi, dxi, xj, dxj, xk, dxk, eta));
//            if (omega <= 0.0)
//                return 0.0;
//        }
//
//        return omega;
//    }
//
//    inline void apply_initial_guess(initial_guess::Type initial_guess_type,
//                                    const std::vector<GuessBlock>& blocks,
//                                    Vec& x_combined,
//                                    Vec& v_combined,
//                                    const std::vector<char>& segment_valid,
//                                    double dt,
//                                    double eta,
//                                    BVHBroadPhase& broad_phase) {
//        const int total = total_nodes(blocks);
//        x_combined.assign(2 * total, 0.0);
//        v_combined.assign(2 * total, 0.0);
//
//        if (initial_guess_type == initial_guess::Type::Trivial) {
//            for (const auto& b : blocks)
//                *b.xnew = b.chain->x;
//
//            build_v_combined_from_chain_velocities(v_combined, blocks);
//            build_x_combined_from_current_positions(x_combined, blocks);
//            return;
//        }
//
//        if (initial_guess_type == initial_guess::Type::Affine) {
//            AffineParams ap = compute_affine_params_global(blocks);
//            build_v_combined_from_affine(v_combined, blocks, ap);
//
//            for (const auto& b : blocks) {
//                Chain& c = *b.chain;
//                Vec& xnew = *b.xnew;
//                xnew = c.x;
//
//                for (int i = 0; i < c.N; ++i) {
//                    Vec2 xi = get_xi(c.x, i);
//
//                    if (c.is_pinned[i]) {
//                        set_xi(xnew, i, xi);
//                        continue;
//                    }
//
//                    Vec2 v_aff = affine_velocity_at(ap, xi);
//                    set_xi(xnew, i, {xi.x + dt * v_aff.x, xi.y + dt * v_aff.y});
//                }
//            }
//
//            build_x_combined_from_xnew(x_combined, blocks);
//            return;
//        }
//
//        if (initial_guess_type == initial_guess::Type::CCD) {
//            build_x_combined_from_current_positions(x_combined, blocks);
//            build_v_combined_from_chain_velocities(v_combined, blocks);
//
//            auto pairs = broad_phase.build_ccd_candidates(x_combined, v_combined, segment_valid, dt);
//            double omega = compute_initial_guess_ccd_step(x_combined, v_combined, pairs, dt, eta);
//
//            for (const auto& b : blocks) {
//                Chain& c = *b.chain;
//                Vec& xnew = *b.xnew;
//                xnew = c.x;
//
//                for (int i = 0; i < c.N; ++i) {
//                    Vec2 xi = get_xi(c.x, i);
//
//                    if (c.is_pinned[i]) {
//                        set_xi(xnew, i, xi);
//                        continue;
//                    }
//
//                    Vec2 vi = get_xi(c.v, i);
//                    set_xi(xnew, i, {xi.x + omega * dt * vi.x,
//                                     xi.y + omega * dt * vi.y});
//                }
//            }
//
//            build_x_combined_from_xnew(x_combined, blocks);
//            return;
//        }
//
//        if (initial_guess_type == initial_guess::Type::TrustRegion) {
//            build_x_combined_from_current_positions(x_combined, blocks);
//            build_v_combined_from_chain_velocities(v_combined, blocks);
//
//            double vmax = 0.0;
//            for (int i = 0; i < total; ++i)
//                vmax = std::max(vmax, norm(get_xi(v_combined, i)));
//
//            double motion_pad = dt * vmax / eta;
//            auto pairs = broad_phase.build_trust_region_candidates(
//                    x_combined, v_combined, segment_valid, dt, motion_pad
//            );
//
//            double alpha = compute_initial_guess_trust_region_step(x_combined, v_combined, pairs, dt, eta);
//            alpha = std::max(0.0, std::min(1.0, alpha));
//
//            for (const auto& b : blocks) {
//                Chain& c = *b.chain;
//                Vec& xnew = *b.xnew;
//                xnew = c.x;
//
//                for (int i = 0; i < c.N; ++i) {
//                    Vec2 xi = get_xi(c.x, i);
//
//                    if (c.is_pinned[i]) {
//                        set_xi(xnew, i, xi);
//                        continue;
//                    }
//
//                    Vec2 vi = get_xi(c.v, i);
//                    set_xi(xnew, i, {xi.x + alpha * dt * vi.x,
//                                     xi.y + alpha * dt * vi.y});
//                }
//            }
//
//            build_x_combined_from_xnew(x_combined, blocks);
//            return;
//        }
//
//        throw std::runtime_error("Unknown initial_guess::Type");
//    }
//
//} // anonymous namespace
//
//int main() {
//    using clock = std::chrono::high_resolution_clock;
//    auto t_start = clock::now();
//
//    const std::string outdir = "frames_spring_IPC_bvh";
//
//    if (fs::exists(outdir)) {
//        fs::remove_all(outdir);
//    }
//    fs::create_directories(outdir);
//
//    // ------------------------------------------------------
//    // Parameters
//    // ------------------------------------------------------
//    double dt = 1.0 / 30.0;
//    Vec2 g_accel{0.0, -9.81};
//    double k_spring = 20.0;
//    int max_global_iters = 1000;
//    double tol_abs = 1e-6;
//    double dhat = 0.1;
//    int number_of_nodes = 11;
//
//    // ExampleType example_type = ExampleType::Example1;
//    ExampleType example_type = ExampleType::Example2;
//
//    auto broad_phase = std::make_unique<BVHBroadPhase>();
//
//    StepPolicy filtering_step_policy = StepPolicy::CCD;
//    // StepPolicy filtering_step_policy = StepPolicy::TrustRegion;
//
//    initial_guess::Type initial_guess_type = initial_guess::Type::CCD;
//    // initial_guess::Type initial_guess_type = initial_guess::Type::TrustRegion;
//    // initial_guess::Type initial_guess_type = initial_guess::Type::Trivial;
//    // initial_guess::Type initial_guess_type = initial_guess::Type::Affine;
//
//    double eta;
//    if (initial_guess_type == initial_guess::Type::TrustRegion &&
//        filtering_step_policy == StepPolicy::TrustRegion) {
//        eta = 0.4;
//    } else if (initial_guess_type == initial_guess::Type::CCD &&
//               filtering_step_policy == StepPolicy::CCD) {
//        eta = 0.9;
//    } else {
//        eta = 0.9;
//    }
//
//    int total_frames = 150;
//
//    // ------------------------------------------------------
//    // Build example as a list of chains
//    // ------------------------------------------------------
//    std::vector<Chain> chains;
//
//    if (example_type == ExampleType::Example1) {
//        total_frames = 150;
//
//        Chain chain1 = make_chain({-0.1,  1.5}, {-0.1, -1.5}, number_of_nodes, 0.05);
//        Chain chain2 = make_chain({ 0.1,  1.5}, { 0.1, -1.5}, number_of_nodes, 0.05);
//
//        chain1.is_pinned[0] = 1;
//        chain2.is_pinned[0] = 1;
//
//        set_xi(chain1.xpin, 0, get_xi(chain1.x, 0));
//        set_xi(chain2.xpin, 0, get_xi(chain2.x, 0));
//
//        for (int i = 0; i < chain1.N; ++i)
//            set_xi(chain1.v, i, {-6.0, 0.0});
//
//        for (int i = 0; i < chain2.N; ++i)
//            set_xi(chain2.v, i, {6.0, 0.0});
//
//        chains.push_back(chain1);
//        chains.push_back(chain2);
//    }
//    else if (example_type == ExampleType::Example2) {
//        total_frames = 60;
//
//        Chain chain1 = make_chain({-0.8, 1.2}, { 1.6, 0.0}, number_of_nodes, 0.05);
//        Chain chain2 = make_chain({-0.4, 2.0}, { 2.0, 0.8}, number_of_nodes, 0.05);
//        Chain chain3 = make_chain({ 0.0, 2.8}, { 2.4, 1.6}, number_of_nodes, 0.05);
//        Chain ground = make_chain({-2.0, -1.8}, { 2.0, -1.8}, 2, 1.0);
//
//        ground.is_pinned[0] = 1;
//        ground.is_pinned[1] = 1;
//
//        set_xi(ground.xpin, 0, get_xi(ground.x, 0));
//        set_xi(ground.xpin, 1, get_xi(ground.x, 1));
//
//        for (int i = 0; i < chain1.N; ++i)
//            set_xi(chain1.v, i, {0.0, 0.0});
//
//        for (int i = 0; i < chain2.N; ++i)
//            set_xi(chain2.v, i, {0.0, 0.0});
//
//        for (int i = 0; i < chain3.N; ++i)
//            set_xi(chain3.v, i, {0.0, 0.0});
//
//        for (int i = 0; i < ground.N; ++i)
//            set_xi(ground.v, i, {0.0, 0.0});
//
//        chains.push_back(chain1);
//        chains.push_back(chain2);
//        chains.push_back(chain3);
//        chains.push_back(ground);
//    }
//
//    // ------------------------------------------------------
//    // Global indexing data
//    // ------------------------------------------------------
//    const int nblocks = static_cast<int>(chains.size());
//
//    std::vector<int> offsets(nblocks, 0);
//    for (int b = 1; b < nblocks; ++b)
//        offsets[b] = offsets[b - 1] + chains[b - 1].N;
//
//    int total_nodes_global = 0;
//    for (const auto& c : chains)
//        total_nodes_global += c.N;
//
//    std::vector<char> segment_valid(std::max(0, total_nodes_global - 1), 0);
//    for (int b = 0; b < nblocks; ++b) {
//        for (int i = 0; i + 1 < chains[b].N; ++i)
//            segment_valid[offsets[b] + i] = 1;
//    }
//
//    std::vector<std::pair<int, int>> edges_combined;
//    for (int b = 0; b < nblocks; ++b) {
//        for (const auto& e : chains[b].edges)
//            edges_combined.emplace_back(e.first + offsets[b], e.second + offsets[b]);
//    }
//
//    Vec x_combined(2 * total_nodes_global, 0.0);
//    Vec v_combined(2 * total_nodes_global, 0.0);
//
//    std::vector<Vec> xnew_blocks(nblocks);
//    for (int b = 0; b < nblocks; ++b)
//        xnew_blocks[b] = chains[b].x;
//
//    auto make_guess_blocks = [&]() {
//        std::vector<GuessBlock> guess_blocks;
//        guess_blocks.reserve(nblocks);
//        for (int b = 0; b < nblocks; ++b)
//            guess_blocks.push_back({&chains[b], &xnew_blocks[b], offsets[b]});
//        return guess_blocks;
//    };
//
//    auto make_solver_blocks = [&]() {
//        std::vector<BlockView> blocks;
//        blocks.reserve(nblocks);
//        for (int b = 0; b < nblocks; ++b) {
//            blocks.push_back({
//                                     &xnew_blocks[b],
//                                     &chains[b].xhat,
//                                     &chains[b].xpin,
//                                     &chains[b].mass,
//                                     &chains[b].rest_lengths,
//                                     &chains[b].is_pinned,
//                                     offsets[b]
//                             });
//        }
//        return blocks;
//    };
//
//    {
//        std::vector<GuessBlock> guess_blocks = make_guess_blocks();
//        build_x_combined_from_current_positions(x_combined, guess_blocks);
//        export_frame(outdir, 1, x_combined, edges_combined);
//    }
//
//    double max_global_residual = 0.0;
//    int sum_global_iters_used = 0;
//
//    // ------------------------------------------------------
//    // Time stepping
//    // ------------------------------------------------------
//    for (int frame = 2; frame <= total_frames + 1; ++frame) {
//
//        for (int b = 0; b < nblocks; ++b)
//            build_xhat(chains[b], dt);
//
//        std::vector<GuessBlock> guess_blocks = make_guess_blocks();
//
//        apply_initial_guess(initial_guess_type,
//                            guess_blocks,
//                            x_combined,
//                            v_combined,
//                            segment_valid,
//                            dt,
//                            eta,
//                            *broad_phase);
//
//        std::vector<BlockView> blocks = make_solver_blocks();
//
//        std::vector<double> res_hist;
//        SolveResult result;
//        if (filtering_step_policy == StepPolicy::CCD) {
//            CCDFilter filter;
//            result = solve(blocks, x_combined, v_combined,
//                           dt, k_spring, g_accel,
//                           dhat, max_global_iters, tol_abs, eta,
//                           *broad_phase, filter, &res_hist);
//        } else {
//            TrustRegionFilter filter;
//            result = solve(blocks, x_combined, v_combined,
//                           dt, k_spring, g_accel,
//                           dhat, max_global_iters, tol_abs, eta,
//                           *broad_phase, filter, &res_hist);
//        }
//
//        double global_residual = result.final_residual;
//        int iters_used = result.iterations_used;
//
//        max_global_residual = std::max(max_global_residual, global_residual);
//        sum_global_iters_used += iters_used;
//
//        for (int b = 0; b < nblocks; ++b)
//            update_velocity(chains[b], xnew_blocks[b], dt);
//
//        build_x_combined_from_current_positions(x_combined, guess_blocks);
//        export_frame(outdir, frame, x_combined, edges_combined);
//
//        std::cout << "Frame " << std::setw(4) << frame
//                  << " | initial_residual=" << std::scientific << res_hist.front()
//                  << " | final_residual="   << std::scientific << global_residual
//                  << " | global_iters="     << std::setw(3) << iters_used
//                  << '\n';
//    }
//
//    auto t_end = clock::now();
//    std::chrono::duration<double> elapsed = t_end - t_start;
//
//    double avg_global_iters_used = 1.0 * sum_global_iters_used / total_frames;
//
//    std::cout << "\n===== Simulation Summary =====\n";
//    std::cout << "max_global_residual = " << std::scientific << max_global_residual << "\n";
//    std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
//    std::cout << "total runtime = " << elapsed.count() << " seconds\n";
//
//    return 0;
//}

#include "ipc_math.h"
#include "chain.h"
#include "visualization.h"
#include "solver.h"
#include "broad_phase/bvh.h"
#include "step_filter/ccd.h"
#include "step_filter/trust_region.h"
#include "initial_guess/initial_guess.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>
#include <algorithm>

namespace fs = std::__fs::filesystem;
using namespace math;

namespace {

    enum class ExampleType {
        Example1,
        Example2
    };

    enum class StepPolicy {
        CCD,
        TrustRegion
    };

} // anonymous namespace

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
    double dt = 1.0 / 30.0;
    Vec2 g_accel{0.0, -9.81};
    double k_spring = 20.0;
    int max_global_iters = 1000;
    double tol_abs = 1e-6;
    double dhat = 0.1;
    int number_of_nodes = 11;

    // ------------------------------------------------------
    // Strategy choices
    // ------------------------------------------------------
    // ExampleType example_type = ExampleType::Example1;
    ExampleType example_type = ExampleType::Example2;

    auto broad_phase = std::make_unique<BVHBroadPhase>();

    StepPolicy filtering_step_policy = StepPolicy::CCD;
    // StepPolicy filtering_step_policy = StepPolicy::TrustRegion;

    initial_guess::Type initial_guess_type = initial_guess::Type::CCD;
    // initial_guess::Type initial_guess_type = initial_guess::Type::TrustRegion;
    // initial_guess::Type initial_guess_type = initial_guess::Type::Trivial;
    // initial_guess::Type initial_guess_type = initial_guess::Type::Affine;

    // Monolithic-code eta rule:
    // only use 0.4 when BOTH initial guess and step filtering are TrustRegion
    double eta;
    if (initial_guess_type == initial_guess::Type::TrustRegion &&
        filtering_step_policy == StepPolicy::TrustRegion) {
        eta = 0.4;
    } else if (initial_guess_type == initial_guess::Type::CCD &&
               filtering_step_policy == StepPolicy::CCD) {
        eta = 0.9;
    } else {
        eta = 0.9;
    }

    int total_frames = 150;

    // ------------------------------------------------------
    // Build example as a list of chains
    // ------------------------------------------------------
    std::vector<Chain> chains;

    if (example_type == ExampleType::Example1) {
        total_frames = 150;

        Chain chain1 = make_chain({-0.1,  1.5}, {-0.1, -1.5}, number_of_nodes, 0.05);
        Chain chain2 = make_chain({ 0.1,  1.5}, { 0.1, -1.5}, number_of_nodes, 0.05);

        chain1.is_pinned[0] = 1;
        chain2.is_pinned[0] = 1;

        set_xi(chain1.xpin, 0, get_xi(chain1.x, 0));
        set_xi(chain2.xpin, 0, get_xi(chain2.x, 0));

        for (int i = 0; i < chain1.N; ++i)
            set_xi(chain1.v, i, {-6.0, 0.0});

        for (int i = 0; i < chain2.N; ++i)
            set_xi(chain2.v, i, {6.0, 0.0});

        chains.push_back(chain1);
        chains.push_back(chain2);
    }
    else if (example_type == ExampleType::Example2) {
        total_frames = 60;

        Chain chain1 = make_chain({-0.8, 1.2}, { 1.6, 0.0}, number_of_nodes, 0.05);
        Chain chain2 = make_chain({-0.4, 2.0}, { 2.0, 0.8}, number_of_nodes, 0.05);
        Chain chain3 = make_chain({ 0.0, 2.8}, { 2.4, 1.6}, number_of_nodes, 0.05);
        Chain ground = make_chain({-2.0, -1.8}, { 2.0, -1.8}, 2, 1.0);

        ground.is_pinned[0] = 1;
        ground.is_pinned[1] = 1;

        set_xi(ground.xpin, 0, get_xi(ground.x, 0));
        set_xi(ground.xpin, 1, get_xi(ground.x, 1));

        for (int i = 0; i < chain1.N; ++i)
            set_xi(chain1.v, i, {0.0, 0.0});

        for (int i = 0; i < chain2.N; ++i)
            set_xi(chain2.v, i, {0.0, 0.0});

        for (int i = 0; i < chain3.N; ++i)
            set_xi(chain3.v, i, {0.0, 0.0});

        for (int i = 0; i < ground.N; ++i)
            set_xi(ground.v, i, {0.0, 0.0});

        chains.push_back(chain1);
        chains.push_back(chain2);
        chains.push_back(chain3);
        chains.push_back(ground);
    }

    // ------------------------------------------------------
    // Global indexing data
    // ------------------------------------------------------
    const int nblocks = static_cast<int>(chains.size());

    std::vector<int> offsets(nblocks, 0);
    for (int b = 1; b < nblocks; ++b)
        offsets[b] = offsets[b - 1] + chains[b - 1].N;

    int total_nodes = 0;
    for (const auto& c : chains)
        total_nodes += c.N;

    // Global valid-segment array
    std::vector<char> segment_valid(std::max(0, total_nodes - 1), 0);
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i + 1 < chains[b].N; ++i)
            segment_valid[offsets[b] + i] = 1;
    }

    // Combined edge list (for export only)
    std::vector<std::pair<int, int>> edges_combined;
    for (int b = 0; b < nblocks; ++b) {
        for (const auto& e : chains[b].edges)
            edges_combined.emplace_back(e.first + offsets[b], e.second + offsets[b]);
    }

    // ------------------------------------------------------
    // Global state buffers
    // ------------------------------------------------------
    Vec x_combined(2 * total_nodes, 0.0);
    Vec v_combined(2 * total_nodes, 0.0);

    // Per-block unknowns
    std::vector<Vec> xnew_blocks(nblocks);
    for (int b = 0; b < nblocks; ++b)
        xnew_blocks[b] = chains[b].x;

    auto make_guess_blocks = [&]() {
        std::vector<initial_guess::BlockRef> guess_blocks;
        guess_blocks.reserve(nblocks);
        for (int b = 0; b < nblocks; ++b)
            guess_blocks.push_back({&chains[b], &xnew_blocks[b], offsets[b]});
        return guess_blocks;
    };

    auto make_solver_blocks = [&]() {
        std::vector<BlockView> blocks;
        blocks.reserve(nblocks);
        for (int b = 0; b < nblocks; ++b) {
            blocks.push_back({
                                     &xnew_blocks[b],
                                     &chains[b].xhat,
                                     &chains[b].xpin,
                                     &chains[b].mass,
                                     &chains[b].rest_lengths,
                                     &chains[b].is_pinned,
                                     offsets[b]
                             });
        }
        return blocks;
    };

    // ------------------------------------------------------
    // Initial export: frame 1 is the initial configuration
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

        // Linear extrapolation
        for (int b = 0; b < nblocks; ++b)
            build_xhat(chains[b], dt);

        // Initial guess
        std::vector<initial_guess::BlockRef> guess_blocks = make_guess_blocks();
        initial_guess::apply(initial_guess_type,
                             guess_blocks,
                             x_combined,
                             v_combined,
                             segment_valid,
                             dt,
                             dhat,
                             eta);

        // Solver block views
        std::vector<BlockView> blocks = make_solver_blocks();

        // Nonlinear GS solve
        std::vector<double> res_hist;
        SolveResult result;

        if (filtering_step_policy == StepPolicy::CCD) {
            CCDFilter filter;
            result = solve(blocks, x_combined, v_combined,
                           dt, k_spring, g_accel,
                           dhat, max_global_iters, tol_abs, eta,
                           *broad_phase, filter, &res_hist);
        } else {
            TrustRegionFilter filter;
            result = solve(blocks, x_combined, v_combined,
                           dt, k_spring, g_accel,
                           dhat, max_global_iters, tol_abs, eta,
                           *broad_phase, filter, &res_hist);
        }

        double global_residual = result.final_residual;
        int iters_used = result.iterations_used;

        max_global_residual = std::max(max_global_residual, global_residual);
        sum_global_iters_used += iters_used;

        // Velocity update
        for (int b = 0; b < nblocks; ++b)
            update_velocity(chains[b], xnew_blocks[b], dt);

        // Export solved positions
        initial_guess::build_x_combined_from_current_positions(x_combined, guess_blocks);
        export_frame(outdir, frame, x_combined, edges_combined);

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual=" << std::scientific << res_hist.front()
                  << " | final_residual="   << std::scientific << global_residual
                  << " | global_iters="     << std::setw(3) << iters_used
                  << '\n';
    }

    auto t_end = clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    double avg_global_iters_used = 1.0 * sum_global_iters_used / total_frames;

    std::cout << "\n===== Simulation Summary =====\n";
    std::cout << "max_global_residual = " << std::scientific << max_global_residual << "\n";
    std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
    std::cout << "total runtime = " << elapsed.count() << " seconds\n";

    return 0;
}
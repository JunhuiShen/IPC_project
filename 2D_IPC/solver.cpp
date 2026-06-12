#include "solver.h"
#include "barrier_energy.h"
#include "ccd.h"
#include "ogc_trust_region.h"
#include "parallel_helper.h"
#include <algorithm>
#include <cstdint>
#include <unordered_set>

using namespace math;
using namespace physics;

struct NodeUpdate {
    const BlockView* block = nullptr;
    int local_i = -1;
    int who = -1;
    Vec2 dx{0.0, 0.0};
    double omega = 0.0;
};

static std::uint64_t contact_pair_key(const NodeSegmentPair& p) {
    return (std::uint64_t(std::uint32_t(p.node)) << 32) | std::uint32_t(p.seg0);
}

static Vec2 compute_local_gradient(int i, const BlockView& b,
                                   double dt, double k, const Vec2& g_accel,
                                   const std::vector<NodeSegmentPair>& barrier_pairs,
                                   double d_hat, double k_barrier, const Vec& x_global) {
    Vec2 gi = local_grad_no_barrier(i, *b.x, *b.xhat, *b.xpin,
                                    *b.mass, b.ref_mesh->rest_lengths, b.rest_length_offset, *b.is_pinned,
                                    dt, k, g_accel);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x_global, c.node, c.seg0, c.seg1, d_hat);
        gi.x += dt * dt * k_barrier * gb.x;
        gi.y += dt * dt * k_barrier * gb.y;
    }

    return gi;
}

static Mat2 compute_local_hessian(int i, const BlockView& b,
                                  double dt, double k,
                                  const std::vector<NodeSegmentPair>& barrier_pairs,
                                  double d_hat, double k_barrier, const Vec& x_global) {
    Mat2 Hi = local_hess_no_barrier(i, *b.x, *b.mass,
                                    b.ref_mesh->rest_lengths, b.rest_length_offset,
                                    *b.is_pinned, dt, k);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Mat2 Hb = local_barrier_hess(who, x_global, c.node, c.seg0, c.seg1, d_hat);
        Hi.a11 += dt * dt * k_barrier * Hb.a11;
        Hi.a12 += dt * dt * k_barrier * Hb.a12;
        Hi.a21 += dt * dt * k_barrier * Hb.a21;
        Hi.a22 += dt * dt * k_barrier * Hb.a22;
    }

    return Hi;
}

static double compute_global_residual(const BlockView& b,
                                      double dt, double k, const Vec2& g_accel,
                                      const std::vector<NodeSegmentPair>& barrier_pairs,
                                      double d_hat, double k_barrier, const Vec& x_global) {
    double r = 0.0;

    for (int i = 0; i < b.size(); ++i) {
        Vec2 g = compute_local_gradient(i, b, dt, k, g_accel, barrier_pairs, d_hat, k_barrier, x_global);
        const double mi = std::max((*b.mass)[i], 1e-12);
        r = std::max(r, std::max(std::abs(g.x), std::abs(g.y)) / mi);
    }

    return r;
}

static double compute_ccd_safe_step(int who_global, const Vec2& dx,
                                    const Vec& x_global,
                                    const std::vector<NodeSegmentPair>& candidates,
                                    double eta) {
    double omega = 1.0;
    Vec2 full{-dx.x, -dx.y};

    for (const auto& c : candidates) {
        if (who_global != c.node && who_global != c.seg0 && who_global != c.seg1)
            continue;

        Vec2 xi = get_xi(x_global, c.node);
        Vec2 xj = get_xi(x_global, c.seg0);
        Vec2 xk = get_xi(x_global, c.seg1);

        Vec2 dxi{}, dxj{}, dxk{};
        if      (who_global == c.node) dxi = full;
        else if (who_global == c.seg0) dxj = full;
        else if (who_global == c.seg1) dxk = full;

        omega = std::min(omega, step_filter::ccd::safe_step(xi, dxi, xj, dxj, xk, dxk, eta));
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

static double compute_trust_region_safe_step(int who_global, const Vec2& dx,
                                             const Vec& x_global,
                                             const std::vector<NodeSegmentPair>& candidates,
                                             double eta) {
    double omega = 1.0;
    Vec2 full{-dx.x, -dx.y};

    for (const auto& c : candidates) {
        if (who_global != c.node && who_global != c.seg0 && who_global != c.seg1)
            continue;

        Vec2 xi = get_xi(x_global, c.node);
        Vec2 xj = get_xi(x_global, c.seg0);
        Vec2 xk = get_xi(x_global, c.seg1);

        Vec2 dxi{}, dxj{}, dxk{};
        if      (who_global == c.node) dxi = full;
        else if (who_global == c.seg0) dxj = full;
        else if (who_global == c.seg1) dxk = full;

        omega = std::min(omega, trust_region_node_segment_gauss_seidel(
                xi, dxi, xj, dxj, xk, dxk, eta).omega);
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

static void update_one_node(int local_i, const BlockView& b,
                            Vec& x_global, BroadPhase& broad_phase,
                            const Vec& v_vel_global,
                            const std::vector<char>& segment_valid,
                            double dt, double k, const Vec2& g_accel,
                            double d_hat, double k_barrier, double eta,
                            bool use_ccd_step_policy) {
    Vec2 gi = compute_local_gradient(local_i, b,
                                     dt, k, g_accel,
                                     broad_phase.pairs(), d_hat, k_barrier, x_global);

    Mat2 Hi = compute_local_hessian(local_i, b,
                                    dt, k,
                                    broad_phase.pairs(), d_hat, k_barrier, x_global);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);

    const int who = b.offset + local_i;

    // Encode Newton step as a velocity over one dt
    Vec v_newton(v_vel_global.size(), 0.0);
    set_xi(v_newton, who, {-dx.x / dt, -dx.y / dt});

    std::vector<NodeSegmentPair> filtering_candidate_set;

    if (use_ccd_step_policy) {
        filtering_candidate_set = broad_phase.build_ccd_candidates_for_node(
                who, x_global, v_newton, segment_valid, dt
        );
    } else {
        double motion_pad = norm(dx) / eta;
        filtering_candidate_set = broad_phase.build_trust_region_candidates(
                x_global, v_newton, segment_valid, dt, motion_pad
        );
    }

    double omega = use_ccd_step_policy
            ? compute_ccd_safe_step(who, dx, x_global, filtering_candidate_set, eta)
            : compute_trust_region_safe_step(who, dx, x_global, filtering_candidate_set, eta);

    Vec2 xi = get_xi(*b.x, local_i);
    xi.x -= omega * dx.x;
    xi.y -= omega * dx.y;

    set_xi(*b.x, local_i, xi);
    set_xi(x_global, who, xi);

    // Incrementally refresh the persistent barrier broad phase
    broad_phase.refresh(x_global, v_vel_global, who,
                        dt, /*node_pad=*/d_hat, /*seg_pad=*/0.0);
}

static NodeUpdate compute_parallel_node_update(int local_i, const BlockView& b,
                                               const Vec& x_global, BroadPhase& broad_phase,
                                               const Vec& v_vel_global,
                                               const std::vector<char>& segment_valid,
                                               double dt, double k, const Vec2& g_accel,
                                               double d_hat, double k_barrier, double eta,
                                               bool use_ccd_step_policy) {
    Vec2 gi = compute_local_gradient(local_i, b,
                                     dt, k, g_accel,
                                     broad_phase.pairs(), d_hat, k_barrier, x_global);

    Mat2 Hi = compute_local_hessian(local_i, b,
                                    dt, k,
                                    broad_phase.pairs(), d_hat, k_barrier, x_global);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);
    const int who = b.offset + local_i;

    Vec2 xi = get_xi(x_global, who);
    Vec2 displacement{-dx.x, -dx.y};
    double omega = broad_phase.node_box_safe_step(who, xi, displacement);

    Vec v_newton(v_vel_global.size(), 0.0);
    set_xi(v_newton, who, {-dx.x / dt, -dx.y / dt});

    std::vector<NodeSegmentPair> filtering_candidate_set;
    if (use_ccd_step_policy) {
        filtering_candidate_set = broad_phase.build_ccd_candidates_for_node(
                who, x_global, v_newton, segment_valid, dt);
    } else {
        double motion_pad = norm(dx) / eta;
        filtering_candidate_set = broad_phase.build_trust_region_candidates(
                x_global, v_newton, segment_valid, dt, motion_pad);
    }

    const double contact_omega = use_ccd_step_policy
            ? compute_ccd_safe_step(who, dx, x_global, filtering_candidate_set, eta)
            : compute_trust_region_safe_step(who, dx, x_global, filtering_candidate_set, eta);
    omega = std::min(omega, contact_omega);

    return {&b, local_i, who, dx, omega};
}

static void commit_node_update(const NodeUpdate& update, Vec& x_global) {
    if (update.block == nullptr) return;
    const BlockView& b = *update.block;

    Vec2 xi = get_xi(*b.x, update.local_i);
    xi.x -= update.omega * update.dx.x;
    xi.y -= update.omega * update.dx.y;

    set_xi(*b.x, update.local_i, xi);
    set_xi(x_global, update.who, xi);
}

// ======================================================
// Public API
// ======================================================

SolveResult global_gauss_seidel_solver_basic(std::vector<BlockView>& blocks,
                                              Vec& x_global, const Vec& v_vel_global,
                                              double dt, double k, const Vec2& g_accel,
                                              double d_hat, double k_barrier,
                                              int max_iters, double tol_abs, double eta,
                                              double node_box_min, double node_box_max, int node_box_update_count,
                                              BroadPhase& broad_phase, bool use_ccd_step_policy,
                                              bool use_parallel,
                                              std::vector<double>* residual_history) {

    const int total_nodes = static_cast<int>(x_global.size() / 2);

    // Build global valid-segment array from block layout
    std::vector<char> segment_valid(std::max(0, total_nodes - 1), 0);
    for (const auto& b : blocks) {
        for (int i = 0; i + 1 < b.size(); ++i)
            segment_valid[b.offset + i] = 1;
    }

    const Vec x_substep_start = x_global;
    static std::vector<double> prev_disp;
    if (static_cast<int>(prev_disp.size()) != total_nodes)
        prev_disp.assign(total_nodes, node_box_max);

    std::vector<std::vector<int>> color_groups;
    std::vector<NodeSegmentPair> conflict_pairs;
    std::unordered_set<std::uint64_t> conflict_pair_keys;
    auto update_prev_disp = [&]() {
        for (int i = 0; i < total_nodes; ++i) {
            Vec2 xi = get_xi(x_global, i);
            Vec2 x0 = get_xi(x_substep_start, i);
            prev_disp[i] = norm(xi - x0);
        }
    };

    auto build_parallel_active_set = [&]() {
        std::vector<double> node_radii(total_nodes, node_box_min);
        constexpr double node_box_padding = 1.2;
        for (int i = 0; i < total_nodes; ++i) {
            const double inertial = norm(get_xi(v_vel_global, i)) * dt;
            const double raw = std::max(prev_disp[i], inertial) * node_box_padding;
            node_radii[i] = std::clamp(raw, node_box_min, node_box_max);
        }

        broad_phase.initialize_node_radii(x_global, segment_valid, node_radii, d_hat);

        for (const NodeSegmentPair& p : broad_phase.pairs()) {
            if (conflict_pair_keys.insert(contact_pair_key(p)).second)
                conflict_pairs.push_back(p);
        }

        const auto graph = build_conflict_graph(blocks, conflict_pairs, total_nodes);
        color_groups = greedy_color_conflict_graph(graph);
    };

    build_parallel_active_set();

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        double r = 0.0;
        for (const auto& b : blocks) {
            r = std::max(r, compute_global_residual(
                    b, dt, k, g_accel,
                    broad_phase.pairs(), d_hat, k_barrier, x_global
            ));
        }
        return r;
    };

    double r = eval_residual();
    if (residual_history) residual_history->push_back(r);

    if (r < tol_abs) {
        update_prev_disp();
        return {r, 0};
    }

    const auto global_to_block_local = build_global_to_block_local(blocks, total_nodes);
    const int rebuild_every = std::max(1, node_box_update_count);

    for (int it = 1; it < max_iters; ++it) {
        if (it > 1 && (it - 1) % rebuild_every == 0)
            build_parallel_active_set();

        for (const auto& group : color_groups) {
            std::vector<NodeUpdate> updates(group.size());

            #pragma omp parallel for if(use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                const int who = group[idx];
                const auto [block_index, local_i] = global_to_block_local[who];
                if (block_index < 0) continue;

                updates[idx] = compute_parallel_node_update(
                        local_i, blocks[block_index], x_global, broad_phase,
                        v_vel_global, segment_valid,
                        dt, k, g_accel, d_hat, k_barrier, eta, use_ccd_step_policy);
            }

            for (const NodeUpdate& update : updates)
                commit_node_update(update, x_global);
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (r < tol_abs) {
            update_prev_disp();
            return {r, it};
        }
    }

    update_prev_disp();
    return {r, max_iters};
}

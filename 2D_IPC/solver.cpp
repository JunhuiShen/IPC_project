#include "solver.h"
#include "barrier_energy.h"
#include "ccd.h"
#include "ogc_trust_region.h"
#include <algorithm>

using namespace math;
using namespace physics;

static Vec2 compute_local_gradient(int i, const BlockView& b,
                                   double dt, double k, const Vec2& g_accel,
                                   const std::vector<NodeSegmentPair>& barrier_pairs,
                                   double dhat, double k_barrier, const Vec& x_global) {
    Vec2 gi = local_grad_no_barrier(i, *b.x, *b.xhat, *b.xpin,
                                    *b.mass, b.ref_mesh->rest_lengths, b.rest_length_offset, *b.is_pinned,
                                    dt, k, g_accel);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x_global, c.node, c.seg0, c.seg1, dhat);
        gi.x += dt * dt * k_barrier * gb.x;
        gi.y += dt * dt * k_barrier * gb.y;
    }

    return gi;
}

static Mat2 compute_local_hessian(int i, const BlockView& b,
                                  double dt, double k,
                                  const std::vector<NodeSegmentPair>& barrier_pairs,
                                  double dhat, double k_barrier, const Vec& x_global) {
    Mat2 Hi = local_hess_no_barrier(i, *b.x, *b.mass,
                                    b.ref_mesh->rest_lengths, b.rest_length_offset,
                                    *b.is_pinned, dt, k);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Mat2 Hb = local_barrier_hess(who, x_global, c.node, c.seg0, c.seg1, dhat);
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
                                      double dhat, double k_barrier, const Vec& x_global) {
    double r = 0.0;

    for (int i = 0; i < b.size(); ++i) {
        Vec2 g = compute_local_gradient(i, b, dt, k, g_accel, barrier_pairs, dhat, k_barrier, x_global);
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
                            double dhat, double k_barrier, double eta,
                            bool use_ccd_step_policy) {
    Vec2 gi = compute_local_gradient(local_i, b,
                                     dt, k, g_accel,
                                     broad_phase.pairs(), dhat, k_barrier, x_global);

    Mat2 Hi = compute_local_hessian(local_i, b,
                                    dt, k,
                                    broad_phase.pairs(), dhat, k_barrier, x_global);

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
                        dt, /*node_pad=*/dhat, /*seg_pad=*/0.0);
}

// ======================================================
// Public API
// ======================================================

SolveResult global_gauss_seidel_solver_basic(std::vector<BlockView>& blocks,
                                              Vec& x_global, const Vec& v_vel_global,
                                              double dt, double k, const Vec2& g_accel,
                                              double dhat, double k_barrier,
                                              int max_iters, double tol_abs, double eta,
                                              BroadPhase& broad_phase, bool use_ccd_step_policy,
                                              std::vector<double>* residual_history) {

    const int total_nodes = static_cast<int>(x_global.size() / 2);

    // Build global valid-segment array from block layout
    std::vector<char> segment_valid(std::max(0, total_nodes - 1), 0);
    for (const auto& b : blocks) {
        for (int i = 0; i + 1 < b.size(); ++i)
            segment_valid[b.offset + i] = 1;
    }

    // Persistent barrier cache
    broad_phase.initialize(x_global, v_vel_global, segment_valid, dt, dhat);

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        double r = 0.0;
        for (const auto& b : blocks) {
            r = std::max(r, compute_global_residual(
                    b, dt, k, g_accel,
                    broad_phase.pairs(), dhat, k_barrier, x_global
            ));
        }
        return r;
    };

    double r = eval_residual();
    if (residual_history) residual_history->push_back(r);

    if (r < tol_abs)
        return {r, 0};

    for (int it = 1; it < max_iters; ++it) {
        for (const auto& b : blocks) {
            for (int i = 0; i < b.size(); ++i) {
                update_one_node(i, b, x_global, broad_phase, v_vel_global,
                                segment_valid,
                                dt, k, g_accel, dhat, k_barrier, eta, use_ccd_step_policy);
            }
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (r < tol_abs)
            return {r, it};
    }

    return {r, max_iters};
}

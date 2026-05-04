#include "solver.h"
#include "step_filter/ccd.h"
#include <algorithm>
#include <stdexcept>

using namespace math;
using namespace physics;

// ======================================================
// Internal helpers
// ======================================================

static Mat2 mat2_inverse(const Mat2& H) {
    double det = H.a11 * H.a22 - H.a12 * H.a21;
    if (std::abs(det) < 1e-12)
        throw std::runtime_error("Singular matrix in mat2_inverse()");
    double inv = 1.0 / det;
    return {H.a22 * inv, -H.a12 * inv, -H.a21 * inv, H.a11 * inv};
}

static Vec2 compute_local_gradient(int i, const BlockView& b,
                                   double dt, double k, const Vec2& g_accel,
                                   const std::vector<NodeSegmentPair>& barrier_pairs,
                                   double dhat, const Vec& x_global) {
    Vec2 gi = local_grad_no_barrier(i, *b.x, *b.xhat, *b.xpin,
                                    *b.mass, *b.L, *b.is_pinned,
                                    dt, k, g_accel);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x_global, c.node, c.seg0, c.seg1, dhat);
        gi.x += dt * dt * gb.x;
        gi.y += dt * dt * gb.y;
    }

    return gi;
}

static Mat2 compute_local_hessian(int i, const BlockView& b,
                                  double dt, double k,
                                  const std::vector<NodeSegmentPair>& barrier_pairs,
                                  double dhat, const Vec& x_global) {
    Mat2 Hi = local_hess_no_barrier(i, *b.x, *b.mass, *b.L, *b.is_pinned, dt, k);

    const int who = b.offset + i;

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Mat2 Hb = local_barrier_hess(who, x_global, c.node, c.seg0, c.seg1, dhat);
        Hi.a11 += dt * dt * Hb.a11;
        Hi.a12 += dt * dt * Hb.a12;
        Hi.a21 += dt * dt * Hb.a21;
        Hi.a22 += dt * dt * Hb.a22;
    }

    return Hi;
}

static double compute_global_residual(const BlockView& b,
                                      double dt, double k, const Vec2& g_accel,
                                      const std::vector<NodeSegmentPair>& barrier_pairs,
                                      double dhat, const Vec& x_global) {
    double r = 0.0;

    for (int i = 0; i < b.size(); ++i) {
        Vec2 g = compute_local_gradient(i, b, dt, k, g_accel, barrier_pairs, dhat, x_global);
        r = std::max(r, std::max(std::abs(g.x), std::abs(g.y)));
    }

    return r;
}

static void update_one_node(int local_i, const BlockView& b,
                            Vec& x_global, BroadPhase& broad_phase,
                            const Vec& v_vel_global,
                            const std::vector<char>& segment_valid,
                            double dt, double k, const Vec2& g_accel,
                            double dhat, double eta,
                            StepFilter& step_filter) {
    Vec2 gi = compute_local_gradient(local_i, b,
                                     dt, k, g_accel,
                                     broad_phase.pairs(), dhat, x_global);

    Mat2 Hi = compute_local_hessian(local_i, b,
                                    dt, k,
                                    broad_phase.pairs(), dhat, x_global);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);

    const int who = b.offset + local_i;

    // Encode Newton step as a velocity over one dt
    Vec v_newton(v_vel_global.size(), 0.0);
    set_xi(v_newton, who, {-dx.x / dt, -dx.y / dt});

    std::vector<NodeSegmentPair> filtering_candidate_set;

    if (dynamic_cast<CCDFilter*>(&step_filter) != nullptr) {
        filtering_candidate_set = broad_phase.build_ccd_candidates_for_node(
                who, x_global, v_newton, segment_valid, dt
        );
    } else {
        double motion_pad = norm(dx) / eta;
        filtering_candidate_set = broad_phase.build_trust_region_candidates(
                x_global, v_newton, segment_valid, dt, motion_pad
        );
    }

    double omega = step_filter.compute_safe_step(
            who, dx, x_global, filtering_candidate_set, eta
    );

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

SolveResult solve(std::vector<BlockView>& blocks,
                  Vec& x_global, const Vec& v_vel_global,
                  double dt, double k, const Vec2& g_accel,
                  double dhat, int max_iters, double tol_abs, double eta,
                  BroadPhase& broad_phase, StepFilter& step_filter,
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
                    broad_phase.pairs(), dhat, x_global
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
                                dt, k, g_accel, dhat, eta, step_filter);
            }
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (r < tol_abs)
            return {r, it};
    }

    return {r, max_iters};
}

#include "solver.h"
#include "barrier_energy.h"
#include "ccd.h"
#include "ogc_trust_region.h"
#include "parallel_helper.h"
#include "physics.h"
#include <algorithm>

struct NodeUpdate {
    int who = -1;
    Vec2 dx{0.0, 0.0};
    double omega = 0.0;
};

static Vec2 compute_local_gradient(int who, const RefMesh& ref_mesh,
                                   const std::vector<Pin>& pins, const PinMap& pin_map,
                                   const DeformedState& state, const Vec& x, const Vec& xhat,
                                   double dt, double k_spring, const Vec2& g_accel,
                                   const std::vector<NodeSegmentPair>& barrier_pairs,
                                   double d_hat, double k_barrier) {
    Vec2 gi = local_grad_no_barrier(
            who, x, xhat, ref_mesh, pins, &pin_map, dt, k_spring, g_accel);

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x, c.node, c.seg0, c.seg1, d_hat);
        gi.x += dt * dt * k_barrier * gb.x;
        gi.y += dt * dt * k_barrier * gb.y;
    }

    return gi;
}

static Mat2 compute_local_hessian(int who, const RefMesh& ref_mesh,
                                  const std::vector<Pin>& pins, const PinMap& pin_map,
                                  const DeformedState& state, const Vec& x,
                                  double dt, double k_spring,
                                  const std::vector<NodeSegmentPair>& barrier_pairs,
                                  double d_hat, double k_barrier) {
    Mat2 Hi = local_hess_no_barrier(
            who, x, ref_mesh, pins, &pin_map, dt, k_spring);

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Mat2 Hb = local_barrier_hess(who, x, c.node, c.seg0, c.seg1, d_hat);
        Hi.a11 += dt * dt * k_barrier * Hb.a11;
        Hi.a12 += dt * dt * k_barrier * Hb.a12;
        Hi.a21 += dt * dt * k_barrier * Hb.a21;
        Hi.a22 += dt * dt * k_barrier * Hb.a22;
    }

    return Hi;
}

static double compute_global_residual(const RefMesh& ref_mesh,
                                      const std::vector<Pin>& pins, const PinMap& pin_map,
                                      const DeformedState& state, const Vec& x, const Vec& xhat,
                                      double dt, double k_spring, const Vec2& g_accel,
                                      const std::vector<NodeSegmentPair>& barrier_pairs,
                                      double d_hat, double k_barrier) {
    double r = 0.0;

    for (int i = 0; i < static_cast<int>(state.deformed_positions.size()); ++i) {
        Vec2 g = compute_local_gradient(
                i, ref_mesh, pins, pin_map, state, x, xhat, dt, k_spring, g_accel,
                barrier_pairs, d_hat, k_barrier);
        const double mi = std::max(ref_mesh.mass[i], 1e-12);
        r = std::max(r, std::max(std::abs(g.x), std::abs(g.y)) / mi);
    }

    return r;
}

static double compute_ccd_safe_step(int who, const Vec2& dx,
                                    const Vec& x,
                                    const std::vector<NodeSegmentPair>& candidates,
                                    double eta) {
    double omega = 1.0;
    Vec2 full{-dx.x, -dx.y};

    for (const auto& c : candidates) {
        if (who != c.node && who != c.seg0 && who != c.seg1) continue;

        Vec2 xi = get_xi(x, c.node);
        Vec2 xj = get_xi(x, c.seg0);
        Vec2 xk = get_xi(x, c.seg1);

        Vec2 dxi{}, dxj{}, dxk{};
        if      (who == c.node) dxi = full;
        else if (who == c.seg0) dxj = full;
        else if (who == c.seg1) dxk = full;

        omega = std::min(omega, point_segment_ccd_safe_step(xi, dxi, xj, dxj, xk, dxk, eta));
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

static double compute_trust_region_safe_step(
        int who, const Vec2& dx, const Vec& x,
        const std::vector<NodeSegmentPair>& candidates, double eta) {
    double omega = 1.0;
    Vec2 full{-dx.x, -dx.y};

    for (const auto& c : candidates) {
        if (who != c.node && who != c.seg0 && who != c.seg1) continue;

        Vec2 xi = get_xi(x, c.node);
        Vec2 xj = get_xi(x, c.seg0);
        Vec2 xk = get_xi(x, c.seg1);

        Vec2 dxi{}, dxj{}, dxk{};
        if      (who == c.node) dxi = full;
        else if (who == c.seg0) dxj = full;
        else if (who == c.seg1) dxk = full;

        omega = std::min(omega, trust_region_node_segment_gauss_seidel(
                xi, dxi, xj, dxj, xk, dxk, eta).omega);
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

static NodeUpdate compute_node_update(
        int who, const RefMesh& ref_mesh, const DeformedState& state,
        const std::vector<Pin>& pins, const PinMap& pin_map,
        const Vec& x, const Vec& xhat, BroadPhase& broad_phase,
        double dt, double k_spring, const Vec2& g_accel,
        double d_hat, double k_barrier, double eta,
        bool use_ccd_step_policy) {
    Vec2 gi = compute_local_gradient(
            who, ref_mesh, pins, pin_map, state, x, xhat, dt, k_spring, g_accel,
            broad_phase.pairs(), d_hat, k_barrier);
    Mat2 Hi = compute_local_hessian(
            who, ref_mesh, pins, pin_map, state, x, dt, k_spring,
            broad_phase.pairs(), d_hat, k_barrier);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);
    Vec2 xi = get_xi(x, who);
    Vec2 displacement{-dx.x, -dx.y};
    double omega = broad_phase.node_box_safe_step(who, xi, displacement);

    Vec v_newton(x.size(), Vec2{0.0, 0.0});
    set_xi(v_newton, who, {-dx.x / dt, -dx.y / dt});

    std::vector<NodeSegmentPair> filtering_candidates;
    if (use_ccd_step_policy) {
        filtering_candidates = broad_phase.build_ccd_candidates_for_node(
                who, x, v_newton, ref_mesh.edges, dt);
    } else {
        const double motion_pad = norm(dx) / eta;
        filtering_candidates = broad_phase.build_trust_region_candidates(
                x, v_newton, ref_mesh.edges, dt, motion_pad);
    }

    const double contact_omega = use_ccd_step_policy
            ? compute_ccd_safe_step(who, dx, x, filtering_candidates, eta)
            : compute_trust_region_safe_step(who, dx, x, filtering_candidates, eta);

    return {who, dx, std::min(omega, contact_omega)};
}

static void commit_node_update(const NodeUpdate& update, Vec& x) {
    if (update.who < 0) return;
    Vec2 xi = get_xi(x, update.who);
    xi.x -= update.omega * update.dx.x;
    xi.y -= update.omega * update.dx.y;
    set_xi(x, update.who, xi);
}

SolveResult global_gauss_seidel_solver_basic(
        const RefMesh& ref_mesh, const std::vector<Pin>& pins,
        const DeformedState& state, const Vec& xhat,
        Vec& x,
        double dt, double k_spring, const Vec2& g_accel,
        double d_hat, double k_barrier,
        int max_iters, double tol_abs, double eta,
        double node_box_min, double node_box_max, int node_box_update_count,
        BroadPhase& broad_phase, bool use_ccd_step_policy,
        bool use_parallel, std::vector<double>* residual_history) {
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const Vec x_substep_start = x;
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    static std::vector<double> prev_disp;
    if (static_cast<int>(prev_disp.size()) != total_nodes)
        prev_disp.assign(total_nodes, node_box_max);

    const auto elastic_adj = build_elastic_adj(ref_mesh.edges, total_nodes);
    std::vector<std::vector<int>> color_groups;

    auto update_prev_disp = [&]() {
        for (int i = 0; i < total_nodes; ++i) {
            prev_disp[i] = norm(get_xi(x, i) - get_xi(x_substep_start, i));
        }
    };

    auto build_parallel_active_set = [&]() {
        std::vector<double> node_radii(total_nodes, node_box_min);
        constexpr double node_box_padding = 1.2;
        for (int i = 0; i < total_nodes; ++i) {
            const double raw = prev_disp[i] * node_box_padding;
            node_radii[i] = std::clamp(raw, node_box_min, node_box_max);
        }

        std::vector<AABB> blue_boxes;
        RedBoxes red_boxes;
        GreenBoxes green_boxes;
        build_blue_boxes(x, node_radii, blue_boxes);
        build_red_boxes(ref_mesh.edges, blue_boxes, red_boxes);
        build_green_boxes(red_boxes, d_hat, green_boxes);
        broad_phase.mutable_cache() = register_barrier_pairs_from_blue_and_green(
            ref_mesh.edges, blue_boxes, green_boxes);

        const auto contact_adj = build_contact_adj(broad_phase.pairs(), total_nodes);
        color_groups = greedy_color_conflict_graph(union_adjacency(elastic_adj, contact_adj));
    };

    build_parallel_active_set();
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(
            ref_mesh, pins, pin_map, state, x, xhat, dt, k_spring, g_accel,
            broad_phase.pairs(), d_hat, k_barrier);
    };

    double r = eval_residual();
    if (residual_history) residual_history->push_back(r);

    if (r < tol_abs) {
        update_prev_disp();
        return {r, 0};
    }

    const int rebuild_every = std::max(1, node_box_update_count);
    for (int it = 1; it < max_iters; ++it) {
        if (it > 1 && (it - 1) % rebuild_every == 0)
            build_parallel_active_set();

        for (const auto& group : color_groups) {
            std::vector<NodeUpdate> updates(group.size());

            #pragma omp parallel for if(use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                updates[idx] = compute_node_update(
                        group[idx], ref_mesh, state, pins, pin_map, x, xhat, broad_phase,
                        dt, k_spring, g_accel, d_hat, k_barrier, eta,
                        use_ccd_step_policy);
            }

            for (const NodeUpdate& update : updates)
                commit_node_update(update, x);
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

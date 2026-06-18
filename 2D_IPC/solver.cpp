#include "solver.h"
#include "barrier_energy.h"
#include "ccd.h"
#include "ogc_trust_region.h"
#include "parallel_helper.h"
#include "physics.h"
#include "rigid_body_ipc.h"
#include <algorithm>
#include <limits>

Vec trivial_initial_guess(const DeformedState& state) {
    return state.deformed_positions;
}

double ccd_initial_guess_safe_step(const Vec& x, const Vec& v,
                                   const std::vector<NodeSegmentPair>& pairs,
                                   double dt, double eta) {
    double omega = 1.0;

    for (const auto& c : pairs) {
        Vec2 xi = get_xi(x, c.node);
        Vec2 xj = get_xi(x, c.seg0);
        Vec2 xk = get_xi(x, c.seg1);

        Vec2 vi = get_xi(v, c.node);
        Vec2 vj = get_xi(v, c.seg0);
        Vec2 vk = get_xi(v, c.seg1);

        Vec2 dxi{dt * vi.x, dt * vi.y};
        Vec2 dxj{dt * vj.x, dt * vj.y};
        Vec2 dxk{dt * vk.x, dt * vk.y};

        omega = std::min(omega, point_segment_ccd_safe_step(xi, dxi, xj, dxj, xk, dxk, eta));
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

Vec ccd_initial_guess(const DeformedState& state, const RefMesh& ref_mesh,
                      const std::vector<Pin>& pins, double dt, double eta) {
    Vec xnew = state.deformed_positions;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    BroadPhase broad_phase;
    auto pairs = broad_phase.build_ccd_candidates(
            state.deformed_positions, state.velocities, ref_mesh.edges, dt);
    const double omega = ccd_initial_guess_safe_step(
            state.deformed_positions, state.velocities, pairs, dt, eta);

    for (int i = 0; i < total_nodes; ++i) {
        Vec2 xi = get_xi(state.deformed_positions, i);
        if (pin_map[i] >= 0) {
            set_xi(xnew, i, pins[pin_map[i]].target_position);
            continue;
        }

        Vec2 vi = get_xi(state.velocities, i);
        set_xi(xnew, i, {xi.x + omega * dt * vi.x, xi.y + omega * dt * vi.y});
    }

    return xnew;
}

Vec verlet_initial_guess(const DeformedState& state, const RefMesh& ref_mesh,
                         const std::vector<Pin>& pins, double dt, double eta,
                         const Vec2& gravity) {
    DeformedState predictor = state;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());

    for (int i = 0; i < total_nodes; ++i) {
        Vec2 vi = get_xi(state.velocities, i);
        vi.x += dt * gravity.x;
        vi.y += dt * gravity.y;
        set_xi(predictor.velocities, i, vi);
    }

    return ccd_initial_guess(predictor, ref_mesh, pins, dt, eta);
}

struct NodeUpdate {
    int who = -1;
    Vec2 dx{0.0, 0.0};
    double omega = 0.0;
};

static bool node_in_rigid_body(const std::vector<int>& rb_nodes, int node) {
    return std::find(rb_nodes.begin(), rb_nodes.end(), node) != rb_nodes.end();
}

static bool sdf_min_evaluation(const std::vector<GroundSDF>& grounds,
                               const std::vector<CircleSDF>& circles,
                               const Vec2& xi,
                               SDFEvaluation& out) {
    bool any = false;
    out.phi = std::numeric_limits<double>::infinity();

    for (const GroundSDF& ground : grounds) {
        const SDFEvaluation sdf = evaluate_sdf(ground, xi);
        if (!any || sdf.phi < out.phi) {
            out = sdf;
            any = true;
        }
    }

    for (const CircleSDF& circle : circles) {
        const SDFEvaluation sdf = evaluate_sdf(circle, xi);
        if (!any || sdf.phi < out.phi) {
            out = sdf;
            any = true;
        }
    }

    return any;
}

static bool sdf_min_evaluation(const SimParams2D& params, const Vec2& xi,
                               SDFEvaluation& out) {
    return sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, xi, out);
}

static Vec2 compute_local_gradient(int who, const RefMesh& ref_mesh,
                                   const std::vector<Pin>& pins, const PinMap& pin_map,
                                   const DeformedState& state, const Vec& x, const Vec& xhat,
                                   const SimParams2D& params,
                                   const std::vector<NodeSegmentPair>& barrier_pairs,
                                   double dt) {
    Vec2 gi = local_grad_no_barrier(
            who, x, xhat, ref_mesh, pins, &pin_map, dt, params.k_spring, params.gravity);

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x, c.node, c.seg0, c.seg1, params.d_hat);
        gi.x += dt * dt * params.k_barrier * gb.x;
        gi.y += dt * dt * params.k_barrier * gb.y;
    }

    if (params.k_sdf > 0.0) {
        SDFEvaluation sdf;
        if (sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, get_xi(x, who), sdf)) {
            Vec2 gs = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
            gi.x += dt * dt * gs.x;
            gi.y += dt * dt * gs.y;
        }
    }

    return gi;
}

static Mat2 compute_local_hessian(int who, const RefMesh& ref_mesh,
                                  const std::vector<Pin>& pins, const PinMap& pin_map,
                                  const DeformedState& state, const Vec& x,
                                  const SimParams2D& params,
                                  const std::vector<NodeSegmentPair>& barrier_pairs,
                                  double dt) {
    Mat2 Hi = local_hess_no_barrier(
            who, x, ref_mesh, pins, &pin_map, dt, params.k_spring);

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Mat2 Hb = local_barrier_hess(who, x, c.node, c.seg0, c.seg1, params.d_hat);
        Hi.a11 += dt * dt * params.k_barrier * Hb.a11;
        Hi.a12 += dt * dt * params.k_barrier * Hb.a12;
        Hi.a21 += dt * dt * params.k_barrier * Hb.a21;
        Hi.a22 += dt * dt * params.k_barrier * Hb.a22;
    }

    if (params.k_sdf > 0.0) {
        SDFEvaluation sdf;
        if (sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, get_xi(x, who), sdf)) {
            Mat2 Hs = sdf_penalty_hessian(
                    sdf, params.k_sdf, params.eps_sdf, /*include_curvature=*/false);
            Hi.a11 += dt * dt * Hs.a11;
            Hi.a12 += dt * dt * Hs.a12;
            Hi.a21 += dt * dt * Hs.a21;
            Hi.a22 += dt * dt * Hs.a22;
        }
    }

    return Hi;
}

static double compute_global_residual(const RefMesh& ref_mesh,
                                      const std::vector<Pin>& pins, const PinMap& pin_map,
                                      const DeformedState& state, const Vec& x, const Vec& xhat,
                                      const SimParams2D& params,
                                      const std::vector<NodeSegmentPair>& barrier_pairs,
                                      double dt) {
    double r = 0.0;

    for (int i = 0; i < static_cast<int>(state.deformed_positions.size()); ++i) {
        Vec2 g = compute_local_gradient(
                i, ref_mesh, pins, pin_map, state, x, xhat, params,
                barrier_pairs, dt);
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

struct ComUpdate {
    int rb = -1;
    Vec2 dy{0.0, 0.0};   // H^{-1} g
    double omega = 1.0;  // CCD-safe step fraction
};

static bool contains_node(const std::vector<int>& nodes, int node) {
    return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
}

static ComUpdate compute_com_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const std::vector<int>& rb_nodes,
    const Vec& x, const Vec2& y_current, const std::vector<NodeSegmentPair>& ccd_pairs, double dt, double eta, double m_total) {

    const Vec2& y_n = state.x_coms[rb];
    const Vec2& vhat_n = state.v_coms[rb];

    Vec2 g = inertia_translation_gradient(y_current, y_n, vhat_n, dt, m_total);
    Mat2 H = inertia_translation_hessian(m_total);

    Vec2 dy = matvec(mat2_inverse(H), g);
    Vec2 full_step{-dy.x, -dy.y};

    double omega = 1.0;

    for (const NodeSegmentPair& c : ccd_pairs) {
        const bool node_moves = contains_node(rb_nodes, c.node);
        const bool seg0_moves = contains_node(rb_nodes, c.seg0);
        const bool seg1_moves = contains_node(rb_nodes, c.seg1);

        if (!node_moves && !seg0_moves && !seg1_moves) continue;
        if (node_moves && seg0_moves && seg1_moves) continue;

        Vec2 dxi{0.0, 0.0};
        Vec2 dxj{0.0, 0.0};
        Vec2 dxk{0.0, 0.0};

        if (node_moves) dxi = full_step;
        if (seg0_moves) dxj = full_step;
        if (seg1_moves) dxk = full_step;

        omega = std::min(omega, point_segment_ccd_safe_step(get_xi(x, c.node), dxi, get_xi(x, c.seg0), dxj, get_xi(x, c.seg1), dxk, eta));

        if (omega <= 0.0) return {rb, dy, 0.0};
    }

    return {rb, dy, omega};
}

static NodeUpdate compute_node_update(
        int who, const RefMesh& ref_mesh, const DeformedState& state,
        const std::vector<Pin>& pins, const PinMap& pin_map,
        const Vec& x, const Vec& xhat, BroadPhase& broad_phase,
        const SimParams2D& params,
        double dt) {
    Vec2 gi = compute_local_gradient(
            who, ref_mesh, pins, pin_map, state, x, xhat, params,
            broad_phase.pairs(), dt);
    Mat2 Hi = compute_local_hessian(
            who, ref_mesh, pins, pin_map, state, x, params,
            broad_phase.pairs(), dt);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);
    Vec2 xi = get_xi(x, who);
    Vec2 displacement{-dx.x, -dx.y};
    double omega = broad_phase.node_box_safe_step(who, xi, displacement);

    Vec v_newton(x.size(), Vec2{0.0, 0.0});
    set_xi(v_newton, who, {-dx.x / dt, -dx.y / dt});

    std::vector<NodeSegmentPair> filtering_candidates;
    if (params.use_ccd_step_policy) {
        filtering_candidates = broad_phase.build_ccd_candidates_for_node(
                who, x, v_newton, ref_mesh.edges, dt);
    } else {
        const double motion_pad = norm(dx) / params.eta;
        filtering_candidates = broad_phase.build_trust_region_candidates(
                x, v_newton, ref_mesh.edges, dt, motion_pad);
    }

    const double contact_omega = params.use_ccd_step_policy
            ? compute_ccd_safe_step(who, dx, x, filtering_candidates, params.eta)
            : compute_trust_region_safe_step(who, dx, x, filtering_candidates, params.eta);

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
        Vec& xnew,
        const SimParams2D& params,
        BroadPhase& broad_phase,
        std::vector<double>* residual_history) {
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const Vec xnew_substep_start = xnew;
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    static std::vector<double> prev_disp;
    if (static_cast<int>(prev_disp.size()) != total_nodes)
        prev_disp.assign(total_nodes, params.node_box_max);

    const auto elastic_adj = build_elastic_adj(ref_mesh.edges, total_nodes);
    std::vector<std::vector<int>> color_groups;

    auto update_prev_disp = [&]() {
        for (int i = 0; i < total_nodes; ++i) {
            prev_disp[i] = norm(get_xi(xnew, i) - get_xi(xnew_substep_start, i));
        }
    };

    auto build_parallel_active_set = [&]() {
        std::vector<double> node_radii(total_nodes, params.node_box_min);
        constexpr double node_box_padding = 1.2;
        for (int i = 0; i < total_nodes; ++i) {
            const double raw = prev_disp[i] * node_box_padding;
            node_radii[i] = std::clamp(raw, params.node_box_min, params.node_box_max);
        }

        std::vector<AABB> blue_boxes;
        RedBoxes red_boxes;
        GreenBoxes green_boxes;
        build_blue_boxes(xnew, node_radii, blue_boxes);
        build_red_boxes(ref_mesh.edges, blue_boxes, red_boxes);
        build_green_boxes(red_boxes, params.d_hat, green_boxes);
        broad_phase.mutable_cache() = register_barrier_pairs_from_blue_and_green(
            ref_mesh.edges, blue_boxes, green_boxes);

        const auto contact_adj = build_contact_adj(broad_phase.pairs(), total_nodes);
        color_groups = greedy_color_conflict_graph(union_adjacency(elastic_adj, contact_adj));
    };

    build_parallel_active_set();
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(
            ref_mesh, pins, pin_map, state, xnew, xhat, params,
            broad_phase.pairs(), params.substep_dt());
    };

    double r = eval_residual();
    if (residual_history) residual_history->push_back(r);

    if (r < params.tol_abs) {
        update_prev_disp();
        return {r, 0};
    }

    const int rebuild_every = std::max(1, params.node_box_update_count);
    for (int it = 1; it < params.max_substep_iters; ++it) {
        if (it > 1 && (it - 1) % rebuild_every == 0)
            build_parallel_active_set();

        for (const auto& group : color_groups) {
            std::vector<NodeUpdate> updates(group.size());

            #pragma omp parallel for if(params.use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                updates[idx] = compute_node_update(
                        group[idx], ref_mesh, state, pins, pin_map, xnew, xhat, broad_phase,
                        params, params.substep_dt());
            }

            for (const NodeUpdate& update : updates)
                commit_node_update(update, xnew);
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (r < params.tol_abs) {
            update_prev_disp();
            return {r, it};
        }
    }

    update_prev_disp();
    return {r, params.max_substep_iters};
}

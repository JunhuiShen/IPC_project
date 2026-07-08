#include "solver.h"
#include "barrier_energy.h"
#include "ccd.h"
#include "ogc_trust_region.h"
#include "parallel_helper.h"
#include "physics.h"
#include "rigid_body_ipc.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

Eigen::Vector2d to_eigen(const Vec2& v) {
    return {v.x, v.y};
}

namespace {

bool residual_converged(double residual, double initial_residual, const SimParams2D& params) {
    double effective_tol = 0.0;
    if (params.tol_abs > 0.0) {
        effective_tol = std::max(effective_tol, params.tol_abs);
    }
    if (params.tol_rel > 0.0 && std::isfinite(initial_residual)) {
        effective_tol = std::max(effective_tol, params.tol_rel * initial_residual);
    }
    return residual <= effective_tol;
}

double ccd_initial_guess_safe_step(const Vec& x, const Vec& v, const std::vector<NodeSegmentPair>& pairs,  double dt, double eta) {
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

} // namespace

namespace initial_guess {

Vec trivial_initial_guess(const DeformedState& state) {
    return state.deformed_positions;
}

Vec ccd_initial_guess(const DeformedState& state, const RefMesh& ref_mesh, const std::vector<Pin>& pins, double dt, double eta) {
    Vec xnew = state.deformed_positions;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    BroadPhase broad_phase;
    auto pairs = broad_phase.build_ccd_candidates(state.deformed_positions, state.velocities, ref_mesh.edges, dt);
    const double omega = ccd_initial_guess_safe_step(state.deformed_positions, state.velocities, pairs, dt, eta);

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

Vec verlet_initial_guess(const DeformedState& state, const RefMesh& ref_mesh,const std::vector<Pin>& pins, double dt, double eta, const Vec2& gravity) {
    DeformedState predictor = state;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());

    for (int i = 0; i < total_nodes; ++i) {
        Vec2 vi = get_xi(state.velocities, i);
        vi.x += dt * gravity.x;
        vi.y += dt * gravity.y;
        set_xi(predictor.velocities, i, vi);
    }

    return initial_guess::ccd_initial_guess(predictor, ref_mesh, pins, dt, eta);
}

} // namespace initial_guess

bool sdf_min_evaluation(const std::vector<GroundSDF>& grounds, const std::vector<CircleSDF>& circles,
                        const std::vector<PlaneSDF>& planes, const Vec2& xi, SDFEvaluation& out) {
    bool any = false;
    out.phi = std::numeric_limits<double>::infinity();

    for (const GroundSDF& ground : grounds) {
        const SDFEvaluation sdf = evaluate_sdf(ground, xi);
        if (!any || sdf.phi < out.phi) { out = sdf; any = true; }
    }
    for (const CircleSDF& circle : circles) {
        const SDFEvaluation sdf = evaluate_sdf(circle, xi);
        if (!any || sdf.phi < out.phi) { out = sdf; any = true; }
    }
    for (const PlaneSDF& plane : planes) {
        const SDFEvaluation sdf = evaluate_sdf(plane, xi);
        if (!any || sdf.phi < out.phi) { out = sdf; any = true; }
    }

    return any;
}

int pin_index_for_node(const PinMap& pin_map, int node) {
    if (node < 0 || node >= static_cast<int>(pin_map.size())) return -1;
    return pin_map[node];
}

namespace node_solver {

struct NodeUpdate {
    int who = -1;
    Vec2 dx{0.0, 0.0};
    double omega = 0.0;
};

Vec2 compute_local_gradient(int who, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const PinMap& pin_map, const Vec& x, const Vec& xhat,  const SimParams2D& params,
    const std::vector<NodeSegmentPair>& barrier_pairs, double dt) {
    Vec2 gi = local_grad_no_barrier(
            who, x, xhat, ref_mesh, pins, &pin_map, dt, params.k_spring, params.kpin, params.gravity);

    for (const auto& c : barrier_pairs) {
        if (c.node != who && c.seg0 != who && c.seg1 != who) continue;

        Vec2 gb = local_barrier_grad(who, x, c.node, c.seg0, c.seg1, params.d_hat);
        gi.x += dt * dt * params.k_barrier * gb.x;
        gi.y += dt * dt * params.k_barrier * gb.y;
    }

    if (params.k_sdf > 0.0) {
        SDFEvaluation sdf;
        if (sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, params.sdf_planes, get_xi(x, who), sdf)) {
            Vec2 gs = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
            gi.x += dt * dt * gs.x;
            gi.y += dt * dt * gs.y;
        }
    }

    return gi;
}

Mat2 compute_local_hessian(int who, const RefMesh& ref_mesh, const PinMap& pin_map,
    const Vec& x, const SimParams2D& params, const std::vector<NodeSegmentPair>& barrier_pairs, double dt) {
    Mat2 Hi = local_hess_no_barrier(who, x, ref_mesh, &pin_map, dt, params.k_spring, params.kpin);

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
        if (sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, params.sdf_planes, get_xi(x, who), sdf)) {
            Mat2 Hs = sdf_penalty_hessian(sdf, params.k_sdf, params.eps_sdf, /*include_curvature=*/false);
            Hi.a11 += dt * dt * Hs.a11;
            Hi.a12 += dt * dt * Hs.a12;
            Hi.a21 += dt * dt * Hs.a21;
            Hi.a22 += dt * dt * Hs.a22;
        }
    }

    return Hi;
}

double compute_global_residual(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const PinMap& pin_map,  const DeformedState& state, const Vec& x, const Vec& xhat,
                                      const SimParams2D& params, const std::vector<NodeSegmentPair>& barrier_pairs,  double dt) {
    double r = 0.0;

    for (int i = 0; i < static_cast<int>(state.deformed_positions.size()); ++i) {
        Vec2 g = compute_local_gradient(i, ref_mesh, pins, pin_map, x, xhat, params, barrier_pairs, dt);
        const double mi = std::max(ref_mesh.mass[i], 1e-12);
        r = std::max(r, std::max(std::abs(g.x), std::abs(g.y)) / mi);
    }

    return r;
}

double compute_ccd_safe_step(int who, const Vec2& dx, const Vec& x, const std::vector<NodeSegmentPair>& candidates, double eta) {
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

double compute_trust_region_safe_step(int who, const Vec2& dx, const Vec& x, const std::vector<NodeSegmentPair>& candidates, double eta) {
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

        omega = std::min(omega, trust_region_node_segment_gauss_seidel(xi, dxi, xj, dxj, xk, dxk, eta).omega);
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

NodeUpdate compute_node_update(int who, const RefMesh& ref_mesh,
    const std::vector<Pin>& pins, const PinMap& pin_map, const Vec& x, const Vec& xhat, BroadPhase& broad_phase,
    const SimParams2D& params, double dt) {

    Vec2 gi = compute_local_gradient(who, ref_mesh, pins, pin_map, x, xhat, params, broad_phase.pairs(), dt);
    Mat2 Hi = compute_local_hessian(who, ref_mesh, pin_map, x, params, broad_phase.pairs(), dt);

    Vec2 dx = matvec(mat2_inverse(Hi), gi);
    Vec2 xi = get_xi(x, who);
    Vec2 displacement{-dx.x, -dx.y};
    double omega = 1.0;
    const auto& blue_boxes = broad_phase.cache().blue_boxes;
    if (who >= 0 && who < static_cast<int>(blue_boxes.size())) {
        omega = std::min(omega, box_safe_step(blue_boxes[who], xi, displacement));
    }

    const double contact_omega = params.use_ccd_step_policy
        ? compute_ccd_safe_step(who, dx, x, broad_phase.pairs(), params.eta)
        : compute_trust_region_safe_step(who, dx, x, broad_phase.pairs(), params.eta);

    return {who, dx, std::min(omega, contact_omega)};
}

void commit_node_update(const NodeUpdate& update, Vec& x) {
    if (update.who < 0) return;
    Vec2 xi = get_xi(x, update.who);
    xi.x -= update.omega * update.dx.x;
    xi.y -= update.omega * update.dx.y;
    set_xi(x, update.who, xi);
}

} // namespace node_solver

namespace rb_solver {

// Build a global node -> rigid-body ownership map
// Example: if rb_nodes[0] = {0, 1, 2} and rb_nodes[1] = {3, 4}, then node_to_rb = {0, 0, 0, 1, 1} for 5 total nodes
// Thus nodes 0,1, and 2 belong to RB 0, and nodes 3 and 4 belong to RB 1
std::vector<int> build_node_to_rb(const RefMesh& ref_mesh, int total_nodes) {
    std::vector<int> node_to_rb(total_nodes, -1);
    for (int rb = 0; rb < static_cast<int>(ref_mesh.rb_nodes.size()); ++rb) {
        for (int node : ref_mesh.rb_nodes[rb]) {
            if (node >= 0 && node < total_nodes) node_to_rb[node] = rb;
        }
    }
    return node_to_rb;
}

// Return the rigid body that owns this node, or -1 if none
int owning_rb_for_node(const std::vector<int>& node_to_rb, int node) {
    if (node < 0) return -1;
    if (node >= static_cast<int>(node_to_rb.size())) return -1;
    return node_to_rb[node];
}

// Keep only registered node-segment pairs that connect different rigid bodies.
std::vector<NodeSegmentPair> filter_rigid_body_barrier_pairs(const std::vector<NodeSegmentPair>& pairs, const std::vector<int>& node_to_rb) {
    std::vector<NodeSegmentPair> filtered;
    filtered.reserve(pairs.size());

    for (const NodeSegmentPair& c : pairs) {
        const int node_rb = owning_rb_for_node(node_to_rb, c.node);
        const int seg0_rb = owning_rb_for_node(node_to_rb, c.seg0);
        const int seg1_rb = owning_rb_for_node(node_to_rb, c.seg1);
        if (node_rb < 0 || seg0_rb < 0 || seg1_rb < 0) continue;
        if (node_rb == seg0_rb && node_rb == seg1_rb) continue;
        filtered.push_back(c);
    }

    return filtered;
}

// Convert node-segment contact candidates into an RB conflict graph
// Two rigid bodies share an edge if one active contact pair depends on both
std::vector<std::vector<int>> build_rb_contact_adj(const std::vector<NodeSegmentPair>& pairs, const std::vector<int>& node_to_rb, int num_rbs) {
    std::vector<std::vector<int>> graph(num_rbs);

    auto add_rb_edge = [&](int a, int b) {
        if (a == b || a < 0 || b < 0 || a >= num_rbs || b >= num_rbs) return;
        graph[a].push_back(b);
        graph[b].push_back(a);
    };

    for (const NodeSegmentPair& c : pairs) {
        const int rbs[3] = {
            owning_rb_for_node(node_to_rb, c.node),
            owning_rb_for_node(node_to_rb, c.seg0),
            owning_rb_for_node(node_to_rb, c.seg1),
        };

        for (int a = 0; a < 3; ++a) {
            for (int b = a + 1; b < 3; ++b) {
                add_rb_edge(rbs[a], rbs[b]);
            }
        }
    }

    for (auto& neighbors : graph) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    return graph;
}

double interval_safe_step(double lo, double hi, double value, double delta) {
    constexpr double eps = 1.0e-12;
    if (value < lo - eps || value > hi + eps) return 0.0;
    if (delta > 0.0) return std::clamp((hi - value) / delta, 0.0, 1.0);
    if (delta < 0.0) return std::clamp((lo - value) / delta, 0.0, 1.0);
    return 1.0;
}

void add_rigid_pin_gradient_terms(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& y_current, const std::vector<Pin>& pins, const PinMap& pin_map,
    double dt, double kpin, Vec2& translation, double& rotation) {
    for (int node : rb_nodes) {
        const int pin_index = pin_index_for_node(pin_map, node);
        if (pin_index < 0) continue;

        const Vec2 xi = get_xi(x, node);
        const Vec2 xpin = pins[pin_index].target_position;
        const Vec2 gx{kpin * (xi.x - xpin.x), kpin * (xi.y - xpin.y)};
        const Vec2 r = xi - y_current;
        const Vec2 dx_dtheta{-r.y, r.x};

        translation.x += dt * dt * gx.x;
        translation.y += dt * dt * gx.y;
        rotation += dt * dt * dot(dx_dtheta, gx);
    }
}

// Returns the generalized gradient [dE/dy_x, dE/dy_y, dE/dtheta] for rigid body rb.
Eigen::Vector3d compute_local_gradient_rb(int rb, const RefMesh& ref_mesh, const DeformedState& state, const Vec& x, const Vec2& y_current, double theta_current, double m_total,
                                                  const std::vector<Pin>& pins, const PinMap& pin_map, const SimParams2D& params, const std::vector<NodeSegmentPair>& barrier_pairs,
                                                  const std::vector<int>& node_to_rb, double dt) {
    // Translation: inertia + gravity
    Vec2 gt = inertia_translation_gradient(y_current, state.x_coms[rb], state.v_coms[rb], dt, m_total);
    const Vec2 g_grav = gravitational_gradient(m_total, params.gravity.y, dt);
    gt.x -= g_grav.x;
    gt.y -= g_grav.y;

    // Rotation: inertia
    double gr = inertia_rotation_gradient(theta_current, state.theta[rb], state.omega[rb], ref_mesh.inertia_tensor[rb], dt);

    const auto& rb_nodes = ref_mesh.rb_nodes[rb];
    add_rigid_pin_gradient_terms(rb_nodes, x, y_current, pins, pin_map, dt, params.kpin, gt, gr);

    for (const auto& c : barrier_pairs) {
        const bool node_moves = owning_rb_for_node(node_to_rb, c.node) == rb;
        const bool seg0_moves = owning_rb_for_node(node_to_rb, c.seg0) == rb;
        const bool seg1_moves = owning_rb_for_node(node_to_rb, c.seg1) == rb;
        if (!node_moves && !seg0_moves && !seg1_moves) continue;
        if (node_moves && seg0_moves && seg1_moves) continue;

        const RigidBarrierGradient gb = local_barrier_grad_rb(rb_nodes, x, y_current, c.node, c.seg0, c.seg1, params.d_hat);
        gt.x += dt * dt * params.k_barrier * gb.translation.x;
        gt.y += dt * dt * params.k_barrier * gb.translation.y;
        gr += dt * dt * params.k_barrier * gb.rotation;
    }

    if (params.k_sdf > 0.0) {
        for (int node : rb_nodes) {
            SDFEvaluation sdf;
            if (sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, params.sdf_planes, get_xi(x, node), sdf)) {
                const Vec2 xi = get_xi(x, node);
                const Vec2 gs = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
                gt.x += dt * dt * gs.x;
                gt.y += dt * dt * gs.y;
                const Vec2 r = xi - y_current;
                const Vec2 dxdtheta{-r.y, r.x};
                gr += dt * dt * dot(dxdtheta,gs);
            }
        }
    }

    return {gt.x, gt.y, gr};
}

double compute_rigid_body_unnormalized_residual(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const PinMap& pin_map,
                                                       const DeformedState& state, const Vec& x, const Vec& y_current, const std::vector<double>& theta_current,
                                                       const SimParams2D& params, const std::vector<NodeSegmentPair>& barrier_pairs,
                                                       const std::vector<int>& node_to_rb, double dt) {
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());
    double residual = 0.0;

    for (int rb = 0; rb < num_rbs; ++rb) {
        const double m_total = ref_mesh.total_mass[rb];
        const Eigen::Vector3d g = compute_local_gradient_rb(rb, ref_mesh, state, x, y_current[rb], theta_current[rb], m_total, pins, pin_map, params, barrier_pairs, node_to_rb, dt);

        residual += g.norm();
    }

    return residual;
}

struct ComUpdate {
    int rb = -1;
    Vec2 dy{0.0, 0.0};   // H^{-1} g
    double omega = 1.0;  // CCD-safe step fraction
};

struct ThetaUpdate {
    int rb = -1;
    double dtheta = 0.0; // H^{-1} g
    double omega = 1.0;  // CCD-safe step fraction
};

void sync_rb_positions(int rb, const RefMesh& ref_mesh, const Vec& y_current, const std::vector<double>& theta_current, Vec& x) {
    const Vec2& y = y_current[rb];
    const double theta = theta_current[rb];
    const auto& nodes = ref_mesh.rb_nodes[rb];
    const auto& ref_pos = ref_mesh.ref_positions[rb];
    for (int j = 0; j < static_cast<int>(nodes.size()); ++j)
        set_xi(x, nodes[j], world_space_position(ref_pos[j], y, theta));
}

void add_rigid_sdf_translation_terms(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& y_current, const SimParams2D& params, double dt, Vec2& g, Mat2& H) {
    if (params.k_sdf <= 0.0) return;

    for (int node : rb_nodes) {
        SDFEvaluation sdf;
        if (!sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, params.sdf_planes, get_xi(x, node), sdf)) continue;

        const Vec2 xi = get_xi(x, node);
        const RigidSDFGradient gs = sdf_penalty_gradient_rb(sdf, xi, y_current, params.k_sdf, params.eps_sdf);
        const RigidSDFHessian Hs = sdf_penalty_hessian_rb(sdf, xi, y_current, params.k_sdf, params.eps_sdf, /*include_sdf_curvature=*/false, /*include_rigid_curvature=*/false);

        g.x += dt * dt * gs.translation.x;
        g.y += dt * dt * gs.translation.y;
        H.a11 += dt * dt * Hs.translation_translation.a11;
        H.a12 += dt * dt * Hs.translation_translation.a12;
        H.a21 += dt * dt * Hs.translation_translation.a21;
        H.a22 += dt * dt * Hs.translation_translation.a22;
    }
}

void add_rigid_pin_translation_terms(const std::vector<int>& rb_nodes, const Vec& x, const std::vector<Pin>& pins, const PinMap& pin_map, double dt, double kpin, Vec2& g, Mat2& H) {
    for (int node : rb_nodes) {
        const int pin_index = pin_index_for_node(pin_map, node);
        if (pin_index < 0) continue;

        const Vec2 xi = get_xi(x, node);
        const Vec2 xpin = pins[pin_index].target_position;

        g.x += dt * dt * kpin * (xi.x - xpin.x);
        g.y += dt * dt * kpin * (xi.y - xpin.y);
        H.a11 += dt * dt * kpin;
        H.a22 += dt * dt * kpin;
    }
}

void add_rigid_pin_rotation_terms(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& y_current, const std::vector<Pin>& pins, const PinMap& pin_map, double dt, double kpin, double& g, double& H) {
    for (int node : rb_nodes) {
        const int pin_index = pin_index_for_node(pin_map, node);
        if (pin_index < 0) continue;

        const Vec2 xi = get_xi(x, node);
        const Vec2 xpin = pins[pin_index].target_position;
        const Vec2 gx{kpin * (xi.x - xpin.x), kpin * (xi.y - xpin.y)};
        const Vec2 r = xi - y_current;
        const Vec2 dx_dtheta{-r.y, r.x};
        const Vec2 d2x_dtheta2{-r.x, -r.y};

        g += dt * dt * dot(dx_dtheta, gx);
        H += dt * dt * (kpin * dot(dx_dtheta, dx_dtheta) + dot(gx, d2x_dtheta2));
    }
}

void add_rigid_sdf_rotation_terms(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& y_current, const SimParams2D& params, double dt, double& g, double& H) {
    if (params.k_sdf <= 0.0) return;

    for (int node : rb_nodes) {
        const Vec2 xi = get_xi(x, node);
        SDFEvaluation sdf;
        if (!sdf_min_evaluation(params.sdf_grounds, params.sdf_circles, params.sdf_planes, xi, sdf)) continue;

        const RigidSDFGradient gs = sdf_penalty_gradient_rb(sdf, xi, y_current, params.k_sdf, params.eps_sdf);
        const RigidSDFHessian Hs = sdf_penalty_hessian_rb(sdf, xi, y_current, params.k_sdf, params.eps_sdf, /*include_sdf_curvature=*/false, /*include_rigid_curvature=*/false);

        g += dt * dt * gs.rotation;
        H += dt * dt * Hs.rotation_rotation;
    }
}

void add_rigid_barrier_translation_terms(int rb, const std::vector<int>& rb_nodes, const std::vector<int>& node_to_rb, const Vec& x, const Vec2& y_current, const std::vector<NodeSegmentPair>& barrier_pairs, const SimParams2D& params, double dt, Vec2& g, Mat2& H) {
    for (const NodeSegmentPair& c : barrier_pairs) {
        const bool node_moves = owning_rb_for_node(node_to_rb, c.node) == rb;
        const bool seg0_moves = owning_rb_for_node(node_to_rb, c.seg0) == rb;
        const bool seg1_moves = owning_rb_for_node(node_to_rb, c.seg1) == rb;
        if (!node_moves && !seg0_moves && !seg1_moves) continue;
        if (node_moves && seg0_moves && seg1_moves) continue;

        const RigidBarrierGradient gb = local_barrier_grad_rb(rb_nodes, x, y_current, c.node, c.seg0, c.seg1, params.d_hat);
        const RigidBarrierHessian Hb = local_barrier_hess_rb(rb_nodes, x, y_current, c.node, c.seg0, c.seg1, params.d_hat);

        g.x += dt * dt * params.k_barrier * gb.translation.x;
        g.y += dt * dt * params.k_barrier * gb.translation.y;
        H.a11 += dt * dt * params.k_barrier * Hb.translation_translation.a11;
        H.a12 += dt * dt * params.k_barrier * Hb.translation_translation.a12;
        H.a21 += dt * dt * params.k_barrier * Hb.translation_translation.a21;
        H.a22 += dt * dt * params.k_barrier * Hb.translation_translation.a22;
    }
}

void add_rigid_barrier_rotation_terms(int rb, const std::vector<int>& rb_nodes, const std::vector<int>& node_to_rb, const Vec& x, const Vec2& y_current, const std::vector<NodeSegmentPair>& barrier_pairs, const SimParams2D& params, double dt, double& g, double& H) {
    for (const NodeSegmentPair& c : barrier_pairs) {
        const bool node_moves = owning_rb_for_node(node_to_rb, c.node) == rb;
        const bool seg0_moves = owning_rb_for_node(node_to_rb, c.seg0) == rb;
        const bool seg1_moves = owning_rb_for_node(node_to_rb, c.seg1) == rb;
        if (!node_moves && !seg0_moves && !seg1_moves) continue;
        if (node_moves && seg0_moves && seg1_moves) continue;

        const RigidBarrierGradient gb = local_barrier_grad_rb(rb_nodes, x, y_current, c.node, c.seg0, c.seg1, params.d_hat);
        const RigidBarrierHessian Hb = local_barrier_hess_rb(rb_nodes, x, y_current, c.node, c.seg0, c.seg1, params.d_hat);

        g += dt * dt * params.k_barrier * gb.rotation;
        H += dt * dt * params.k_barrier * Hb.rotation_rotation;
    }
}

ComUpdate compute_com_update(int rb, const DeformedState& state, const std::vector<int>& rb_nodes,
    const Vec& x, const Vec2& y_current, const AABB& com_box, const std::vector<Pin>& pins, const PinMap& pin_map,
    const std::vector<NodeSegmentPair>& ccd_pairs, const std::vector<int>& node_to_rb, const SimParams2D& params, double dt, double eta, double m_total) {

    const Vec2& y_n = state.x_coms[rb];
    const Vec2& vhat_n = state.v_coms[rb];

    Vec2 g = inertia_translation_gradient(y_current, y_n, vhat_n, dt, m_total);

    // add gravitational potential gradient
    const Vec2 g_grav = gravitational_gradient(m_total, params.gravity.y, dt);
    g.x -= g_grav.x;
    g.y -= g_grav.y;

    Mat2 H = inertia_translation_hessian(m_total);
    add_rigid_pin_translation_terms(rb_nodes, x, pins, pin_map, dt, params.kpin, g, H);
    add_rigid_barrier_translation_terms(rb, rb_nodes, node_to_rb, x, y_current, ccd_pairs, params, dt, g, H);
    add_rigid_sdf_translation_terms(rb_nodes, x, y_current, params, dt, g, H);

    Vec2 dy = matvec(mat2_inverse(H), g);
    Vec2 full_step{-dy.x, -dy.y};

    double omega = 1.0;

    for (const NodeSegmentPair& c : ccd_pairs) {
        const bool node_moves = owning_rb_for_node(node_to_rb, c.node) == rb;
        const bool seg0_moves = owning_rb_for_node(node_to_rb, c.seg0) == rb;
        const bool seg1_moves = owning_rb_for_node(node_to_rb, c.seg1) == rb;

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

    const Vec2 displacement{-dy.x, -dy.y};
    omega = std::min(omega, box_safe_step(com_box, y_current, displacement));
    return {rb, dy, omega};
}

ThetaUpdate compute_theta_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const std::vector<int>& rb_nodes,
    const Vec& x, const Vec2& y_current, double theta_current, const std::vector<Pin>& pins, const PinMap& pin_map,
    const std::vector<NodeSegmentPair>& ccd_pairs, const std::vector<int>& node_to_rb, const SimParams2D& params, double dt, double eta,
    double theta_box_center, double theta_box_radius) {

    const double theta_n = state.theta[rb];
    const double omega_n = state.omega[rb];
    const Mat2& I = ref_mesh.inertia_tensor[rb];

    double g = inertia_rotation_gradient(theta_current, theta_n, omega_n, I, dt);
    double H = inertia_rotation_hessian(theta_current, theta_n, omega_n, I, dt);
    add_rigid_pin_rotation_terms(rb_nodes, x, y_current, pins, pin_map, dt, params.kpin, g, H);
    add_rigid_barrier_rotation_terms(rb, rb_nodes, node_to_rb, x, y_current, ccd_pairs, params, dt, g, H);
    add_rigid_sdf_rotation_terms(rb_nodes, x, y_current, params, dt, g, H);

    const double dtheta = g / H;
    const double theta_new = theta_current - dtheta;
    const Vec2 x_com_vec = y_current;
    const Eigen::Vector2d x_com = to_eigen(x_com_vec);

    double step = 1.0;
    for (const NodeSegmentPair& c : ccd_pairs) {
        const bool node_moves = owning_rb_for_node(node_to_rb, c.node) == rb;
        const bool seg0_moves = owning_rb_for_node(node_to_rb, c.seg0) == rb;
        const bool seg1_moves = owning_rb_for_node(node_to_rb, c.seg1) == rb;

        if (!node_moves && !seg0_moves && !seg1_moves) continue;
        if (node_moves && seg0_moves && seg1_moves) continue;

        if (node_moves && !seg0_moves && !seg1_moves) {
            step = std::min(step, point_segment_rb_rotation_safe_step(to_eigen(get_xi(x, c.node)), x_com, theta_current, theta_new, to_eigen(get_xi(x, c.seg0)), to_eigen(get_xi(x, c.seg1)), eta));
            if (step <= 0.0) return {rb, dtheta, 0.0};
            continue;
        }

        // A moving rigid segment vs a fixed world node can be treated in the rigid body’s material frame as a fixed material segment vs the external node rotating by -dtheta
        if (!node_moves && seg0_moves && seg1_moves) {
            const Vec2 material_point = material_space_position(get_xi(x, c.node), x_com_vec, theta_current);
            const Vec2 material_seg0 = material_space_position(get_xi(x, c.seg0), x_com_vec, theta_current);
            const Vec2 material_seg1 = material_space_position(get_xi(x, c.seg1), x_com_vec, theta_current);

            step = std::min(step, point_segment_rb_rotation_safe_step(to_eigen(material_point), Eigen::Vector2d::Zero(), 0.0, -(theta_new - theta_current), to_eigen(material_seg0), to_eigen(material_seg1), eta));
            if (step <= 0.0) return {rb, dtheta, 0.0};
            continue;
        }

        return {rb, dtheta, 0.0};
    }

    const double low = theta_box_center - theta_box_radius;
    const double high = theta_box_center + theta_box_radius;
    step = std::min(step, interval_safe_step(low, high, theta_current, -dtheta));
    return {rb, dtheta, step};
}

void commit_com_update(const ComUpdate& update, Vec& y_current, const RefMesh& ref_mesh, const std::vector<double>& theta_current, Vec& x) {
    if (update.rb < 0) return;
    y_current[update.rb].x -= update.omega * update.dy.x;
    y_current[update.rb].y -= update.omega * update.dy.y;
    sync_rb_positions(update.rb, ref_mesh, y_current, theta_current, x);
}

void commit_theta_update(const ThetaUpdate& update, std::vector<double>& theta_current, const RefMesh& ref_mesh, const Vec& y_current, Vec& x) {
    if (update.rb < 0) return;
    theta_current[update.rb] -= update.omega * update.dtheta;
    sync_rb_positions(update.rb, ref_mesh, y_current, theta_current, x);
}

} // namespace rb_solver

SolveResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const DeformedState& state, const Vec& xhat,  Vec& xnew,
        const SimParams2D& params, BroadPhase& broad_phase, std::vector<double>* residual_history) {
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
        broad_phase.mutable_cache() = register_barrier_pairs_from_blue_and_green(ref_mesh.edges, blue_boxes, green_boxes);

        const auto contact_adj = build_contact_adj(broad_phase.pairs(), total_nodes);
        color_groups = greedy_color_conflict_graph(union_adjacency(elastic_adj, contact_adj));
    };

    build_parallel_active_set();
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return node_solver::compute_global_residual(ref_mesh, pins, pin_map, state, xnew, xhat, params, broad_phase.pairs(), params.substep_dt());
    };

    double r = eval_residual();
    const double initial_residual = r;
    if (residual_history) residual_history->push_back(r);

    if (!params.fixed_iters && residual_converged(r, initial_residual, params)) {
        update_prev_disp();
        return {r, 0};
    }

    const int rebuild_every = std::max(1, params.node_box_update_count);
    for (int it = 1; it <= params.max_substep_iters; ++it) {
        if (it > 1 && (it - 1) % rebuild_every == 0)
            build_parallel_active_set();

        for (const auto& group : color_groups) {
            std::vector<node_solver::NodeUpdate> updates(group.size());

            #pragma omp parallel for if(params.use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                updates[idx] = node_solver::compute_node_update(group[idx], ref_mesh, pins, pin_map, xnew, xhat, broad_phase, params, params.substep_dt());
            }

            for (const node_solver::NodeUpdate& update : updates)
                node_solver::commit_node_update(update, xnew);
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (!params.fixed_iters && residual_converged(r, initial_residual, params)) {
            update_prev_disp();
            return {r, it};
        }
    }

    update_prev_disp();
    return {r, params.max_substep_iters};
}

SolveResult global_gauss_seidel_solver_rb(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const DeformedState& state, Vec& xnew,
        Vec& y_current, std::vector<double>& theta_current, const SimParams2D& params, BroadPhase& broad_phase, std::vector<double>* residual_history) {
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const Vec y_substep_start = y_current;
    const std::vector<double> theta_substep_start = theta_current;
    const PinMap pin_map = build_pin_map(pins, total_nodes);
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());

    static std::vector<double> prev_com_disp;
    static std::vector<double> prev_theta_disp;
    if (static_cast<int>(prev_com_disp.size()) != num_rbs) {
        prev_com_disp.assign(num_rbs, params.node_box_max);
        prev_theta_disp.assign(num_rbs, params.theta_box_max);
    }

    const std::vector<int>& node_to_rb = ref_mesh.node_to_rb;
    std::vector<std::vector<int>> color_groups;
    std::vector<AABB> com_boxes;
    std::vector<double> theta_box_centers;
    std::vector<double> theta_radii;

    for (int rb = 0; rb < num_rbs; ++rb)
        rb_solver::sync_rb_positions(rb, ref_mesh, y_current, theta_current, xnew);

    auto update_prev_disp = [&]() {
        for (int rb = 0; rb < num_rbs; ++rb) {
            prev_com_disp[rb] = norm(y_current[rb] - y_substep_start[rb]);
            prev_theta_disp[rb] = std::abs(theta_current[rb] - theta_substep_start[rb]);
        }
    };

    auto build_parallel_active_set = [&]() {
        std::vector<double> com_radii(num_rbs, params.node_box_min);
        theta_radii.assign(num_rbs, params.theta_box_min);
        com_boxes.resize(num_rbs);
        theta_box_centers = theta_current;

        constexpr double node_box_padding = 1.2;
        for (int rb = 0; rb < num_rbs; ++rb) {
            const double raw_com = prev_com_disp[rb] * node_box_padding;
            com_radii[rb] = std::clamp(raw_com, params.node_box_min, params.node_box_max);

            const double raw_theta = prev_theta_disp[rb] * node_box_padding;
            theta_radii[rb] = std::clamp(raw_theta, params.theta_box_min, params.theta_box_max);

            const Vec2& y = y_current[rb];
            const double r_box = com_radii[rb];
            com_boxes[rb] = AABB(Vec2(y.x - r_box, y.y - r_box), Vec2(y.x + r_box, y.y + r_box));
        }

        std::vector<AABB> blue_boxes;
        RedBoxes red_boxes;
        GreenBoxes green_boxes;
        build_blue_boxes_rb(xnew, y_current, theta_current, theta_radii, com_radii, ref_mesh.rb_nodes, ref_mesh.ref_positions, blue_boxes);
        build_red_boxes(ref_mesh.edges, blue_boxes, red_boxes);
        build_green_boxes(red_boxes, params.d_hat, green_boxes);

        BroadPhase::Cache cache = register_barrier_pairs_from_blue_and_green(ref_mesh.edges, blue_boxes, green_boxes);
        cache.pairs = rb_solver::filter_rigid_body_barrier_pairs(cache.pairs, node_to_rb);
        broad_phase.mutable_cache() = cache;

        const auto contact_adj = rb_solver::build_rb_contact_adj(broad_phase.pairs(), node_to_rb, num_rbs);
        color_groups = greedy_color_conflict_graph(contact_adj);
    };

    build_parallel_active_set();
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return rb_solver::compute_rigid_body_unnormalized_residual(ref_mesh, pins, pin_map, state, xnew, y_current, theta_current, params, broad_phase.pairs(), node_to_rb, params.substep_dt());
    };

    auto make_result = [&](double residual, int iterations) {
        SolveResult result{residual, iterations};
        result.has_rigid_residual = true;
        result.final_rigid_residual = rb_solver::compute_rigid_body_unnormalized_residual(ref_mesh, pins, pin_map, state, xnew, y_current, theta_current, params, broad_phase.pairs(), node_to_rb, params.substep_dt());
        return result;
    };

    double r = eval_residual();
    const double initial_residual = r;
    if (residual_history) residual_history->push_back(r);

    if (!params.fixed_iters && residual_converged(r, initial_residual, params)) {
        update_prev_disp();
        return make_result(r, 0);
    }

    const int rebuild_every = std::max(1, params.node_box_update_count);
    for (int it = 1; it <= params.max_substep_iters; ++it) {
        if (it > 1 && (it - 1) % rebuild_every == 0)
            build_parallel_active_set();

        for (const auto& group : color_groups) {
            std::vector<rb_solver::ComUpdate> updates(group.size());

            #pragma omp parallel for if(params.use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                const int rb = group[idx];
                const auto& rb_nodes = ref_mesh.rb_nodes[rb];
                const double m_total = ref_mesh.total_mass[rb];
                updates[idx] = rb_solver::compute_com_update(rb, state, rb_nodes, xnew, y_current[rb], com_boxes[rb], pins, pin_map, broad_phase.pairs(), node_to_rb, params, params.substep_dt(), params.eta, m_total);
            }

            for (const rb_solver::ComUpdate& update : updates) {
                rb_solver::commit_com_update(update, y_current, ref_mesh, theta_current, xnew);
            }
        }

        for (const auto& group : color_groups) {
            std::vector<rb_solver::ThetaUpdate> updates(group.size());

            #pragma omp parallel for if(params.use_parallel && group.size() > 1)
            for (int idx = 0; idx < static_cast<int>(group.size()); ++idx) {
                const int rb = group[idx];
                const auto& rb_nodes = ref_mesh.rb_nodes[rb];
                updates[idx] = rb_solver::compute_theta_update(rb, state, ref_mesh, rb_nodes, xnew, y_current[rb], theta_current[rb], pins, pin_map, broad_phase.pairs(), node_to_rb, params, params.substep_dt(), params.eta, theta_box_centers[rb], theta_radii[rb]);
            }

            for (const rb_solver::ThetaUpdate& update : updates) {
                rb_solver::commit_theta_update(update, theta_current, ref_mesh, y_current, xnew);
            }
        }

        r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (!params.fixed_iters && residual_converged(r, initial_residual, params)) {
            update_prev_disp();
            return make_result(r, it);
        }
    }

    update_prev_disp();
    return make_result(r, params.max_substep_iters);
}

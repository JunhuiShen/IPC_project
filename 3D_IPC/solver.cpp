#include "solver.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "parallel_helper.h"
#include "trust_region.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include "barrier_energy.h"

#include <algorithm>
#include <limits>
#include <cstdio>
#include <string>


// CCD initial guess
std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh) {
    const int nv = static_cast<int>(x.size());

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase ccd_bp;
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0);
    const auto& cache = ccd_bp.cache();

    double toi_min = 1.0;

    const int n_nt = static_cast<int>(cache.nt_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_nt; ++i) {
        const auto& p = cache.nt_pairs[i];
        toi_min = std::min(toi_min, node_triangle_general_ccd(
            x[p.node],     dx[p.node],
            x[p.tri_v[0]], dx[p.tri_v[0]],
            x[p.tri_v[1]], dx[p.tri_v[1]],
            x[p.tri_v[2]], dx[p.tri_v[2]]));
    }

    const int n_ss = static_cast<int>(cache.ss_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_ss; ++i) {
        const auto& p = cache.ss_pairs[i];
        toi_min = std::min(toi_min, segment_segment_general_ccd(
            x[p.v[0]], dx[p.v[0]],
            x[p.v[1]], dx[p.v[1]],
            x[p.v[2]], dx[p.v[2]],
            x[p.v[3]], dx[p.v[3]]));
    }

    const double omega = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];

    return xnew;
}

// Trust-region initial guess
std::vector<Vec3> trust_region_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, double /*d_hat*/) {
    const int nv = static_cast<int>(x.size());
    constexpr double gamma_P = 0.4;  // 0 < gamma_P < 0.5

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase ccd_bp;
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0);
    const auto& cache = ccd_bp.cache();

    // Paper Eq. 21: b[v] = gamma_P * min over every candidate pair touching v
    // of d0. No d_hat threshold -- the invariant ||x_v - x_v^prev|| <= b_v
    // must hold for all pairs, not only those already inside the barrier.
    std::vector<double> b(nv, std::numeric_limits<double>::infinity());

    for (const auto& p : cache.nt_pairs) {
        const double d0 = node_triangle_distance(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]).distance;
        b[p.node]     = std::min(b[p.node],     d0);
        b[p.tri_v[0]] = std::min(b[p.tri_v[0]], d0);
        b[p.tri_v[1]] = std::min(b[p.tri_v[1]], d0);
        b[p.tri_v[2]] = std::min(b[p.tri_v[2]], d0);
    }
    for (const auto& p : cache.ss_pairs) {
        const double d0 = segment_segment_distance(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]).distance;
        b[p.v[0]] = std::min(b[p.v[0]], d0);
        b[p.v[1]] = std::min(b[p.v[1]], d0);
        b[p.v[2]] = std::min(b[p.v[2]], d0);
        b[p.v[3]] = std::min(b[p.v[3]], d0);
    }
    for (int i = 0; i < nv; ++i) b[i] *= gamma_P;

    // Per-vertex truncation (eq. 28).
    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) {
        const double len = dx[i].norm();
        if (len <= b[i]) {
            xnew[i] = xhat[i];                       // keep the full proposed guess
        } else {
            xnew[i] = x[i] + (b[i] / len) * dx[i];   // truncate to the b_v ball
        }
    }
    return xnew;
}

Vec3 gs_vertex_delta(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    return matrix3d_inverse(H) * g;
}

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    const Vec3 delta = matrix3d_inverse(H) * g;

    double step = 1.0;
    if (params.d_hat > 0.0) {
        const Vec3 dx = -delta;
        const bool tr = params.use_trust_region;

        const auto pairs = broad_phase.query_pairs_for_vertex(x, vi, dx, ref_mesh);

        double safe_min = 1.0;

        // vi as the lone moving node
        for (const auto& p : pairs.nt_node_pairs) {
            if (tr) {
                auto r = trust_region_vertex_triangle_gauss_seidel(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                auto r = node_triangle_only_one_node_moves(
                    x[p.node],     dx,
                    x[p.tri_v[0]], Vec3::Zero(),
                    x[p.tri_v[1]], Vec3::Zero(),
                    x[p.tri_v[2]], Vec3::Zero());
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        // vi as one moving triangle vertex
        for (const auto& p : pairs.nt_face_pairs) {
            if (tr) {
                auto r = trust_region_vertex_triangle_gauss_seidel(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                dxv[p.vi_local] = dx;
                auto r = node_triangle_only_one_node_moves(
                    x[p.node],     Vec3::Zero(),
                    x[p.tri_v[0]], dxv[0],
                    x[p.tri_v[1]], dxv[1],
                    x[p.tri_v[2]], dxv[2]);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        for (const auto& p : pairs.ss_pairs) {
            if (tr) {
                auto r = trust_region_edge_edge_gauss_seidel(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                CCDResult r;
                if (p.vi_dof == 0)
                    r = segment_segment_only_one_node_moves(x[p.v[0]], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]]);
                else
                    r = segment_segment_only_one_node_moves(x[p.v[1]], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]]);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        step = tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
    }

    x[vi] -= step * delta;
}

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v, const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history, const std::string& outdir) {

    if (!params.fixed_iters) {
        fprintf(stderr, "global_gauss_seidel_solver_basic: params.fixed_iters must be true\n");
        exit(1);
    }

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    const int nv = static_cast<int>(xnew.size());
    const PinMap pm = build_pin_map(pins, nv);
    std::vector<AABB> blue_boxes(nv);
    for (int i = 0; i < nv; ++i) {
        blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(params.node_box_size), xnew[i] + Vec3::Constant(params.node_box_size));
    }
    BroadPhase broad_phase;
    broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat);

    //residual tracking: not going to actually do this and will demand running with fixed iterations
    if (residual_history) residual_history->clear();
    SolverResult result;
    result.initial_residual = 0.0;
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;
    if (residual_history) {
        const int reserve_n = std::max(0, params.max_global_iters);
        residual_history->reserve(static_cast<std::size_t>(reserve_n) + 1);
        residual_history->push_back(result.initial_residual);
    }

    if (params.d_hat > 0.0 && params.write_barrier_distances) {
        static int substep_counter = 0;
        const std::string dist_path = (outdir.empty() ? "" : outdir + "/") + "barrier_distances_" + std::to_string(substep_counter++) + ".txt";
        if (FILE* dist_file = fopen(dist_path.c_str(), "w")) {
            fprintf(dist_file, "# substep %d  d_hat=%.6e\n# type node/v0 v1 v2 v3 distance force_norm_sum\n", substep_counter - 1, params.d_hat);
            const auto& bpc = broad_phase.cache();
            for (const auto& p : bpc.nt_pairs) {
                const auto dr = node_triangle_distance(xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]], xnew[p.tri_v[2]]);
                double fsum = 0.0;
                for (int dof = 0; dof < 4; ++dof)
                    fsum += node_triangle_barrier_gradient(xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]], xnew[p.tri_v[2]], params.d_hat, dof, 1e-12, &dr).norm();
                fprintf(dist_file, "NT %d %d %d %d %.10e %.10e\n", p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2], dr.distance, fsum);
            }
            for (const auto& p : bpc.ss_pairs) {
                const auto dr = segment_segment_distance(xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]]);
                double fsum = 0.0;
                for (int dof = 0; dof < 4; ++dof)
                    fsum += segment_segment_barrier_gradient(xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]], params.d_hat, dof, 1e-12, &dr).norm();
                fprintf(dist_file, "SS %d %d %d %d %.10e %.10e\n", p.v[0], p.v[1], p.v[2], p.v[3], dr.distance, fsum);
            }
            fclose(dist_file);
        }
    }

    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        broad_phase.per_vertex_safe_step(xnew, [&](int vi){ return xnew[vi] - gs_vertex_delta(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm); },
                                         /*safety=*/0.9, /*clip_to_node_box=*/true, /*clip_ccd=*/params.use_ccd);
        result.final_residual = 0.0;
        result.iterations     = iter;
        if (residual_history) residual_history->push_back(result.final_residual);
    }

    if (params.fixed_iters) result.converged = true;
    return result;
}

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, BroadPhase& broad_phase,
                                        const std::vector<Vec3>& v, const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history) {
    const double dt         = params.dt();
    const double dhat       = params.d_hat;
    const bool   use_barrier = dhat > 0.0;
    const PinMap pm = build_pin_map(pins, static_cast<int>(xnew.size()));

    if (use_barrier) {
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);
    }

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat,
                                       broad_phase, &pm);
    };

    SolverResult result;
    result.initial_residual = params.fixed_iters ? 0.0 : eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    // Effective stopping threshold: max of absolute floor and relative multiple
    // of the initial residual (tol_rel == 0 keeps legacy absolute-only behavior).
    const double effective_tol = std::max(params.tol_abs,
                                          params.tol_rel * result.initial_residual);

    if (residual_history) {
        const int reserve_n = std::max(0, params.max_global_iters);
        residual_history->reserve(static_cast<std::size_t>(reserve_n) + 1);
        residual_history->push_back(result.initial_residual);
    }
    if (!params.fixed_iters && result.initial_residual < effective_tol) {
        result.converged = true;
        return result;
    }

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (const auto& group : color_groups) {
            for (int vi : group) {
                update_one_vertex(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm);
            }
        }

        result.final_residual = params.fixed_iters ? 0.0 : eval_residual();
        result.iterations     = iter;


        if (residual_history) residual_history->push_back(result.final_residual);
        if (!params.fixed_iters && result.final_residual < effective_tol) {
            result.converged = true;
            return result;
        }
    }

    if (params.fixed_iters) result.converged = true;
    return result;
}

SolverResult global_gauss_seidel_solver_parallel(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                 const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                                 BroadPhase& broad_phase, const std::vector<Vec3>& v, std::vector<double>* residual_history,
                                                 const std::vector<std::vector<int>>* override_colors) {
    const double time_step = params.dt();
    const double dhat = params.d_hat;
    const bool barrier_enabled = dhat > 0.0;
    const int num_vertices = static_cast<int>(xnew.size());
    const PinMap pin_map = build_pin_map(pins, num_vertices);

    // One initialize() call in barrier mode to populate topology in cache
    // (especially edge list) and seed iter 1's Jacobi prediction with a
    // conservative swept pair set. Subsequent iters rebuild pairs from
    // blue/red/green boxes.
    if (barrier_enabled) {
        broad_phase.initialize(xnew, v, ref_mesh, time_step, dhat);
    }

    // Static elastic adjacency.
    const std::vector<std::vector<int>> elastic_adjacency = build_elastic_adj(ref_mesh, adj, num_vertices);

    // Per-iter pair cache. Seeded from the initialize() swept-AABB pairs so
    // iter 1's Jacobi prediction has a non-empty contact pair list; then
    // overwritten each iter by register_barrier_pairs_from_blue_and_green.
    BroadPhase::Cache barrier_pair_cache;
    const BroadPhase::Cache* active_pair_cache = nullptr;
    if (barrier_enabled) {
        active_pair_cache = &broad_phase.cache();
    } else {
        barrier_pair_cache.vertex_nt.assign(num_vertices, {});
        barrier_pair_cache.vertex_ss.assign(num_vertices, {});
        active_pair_cache = &barrier_pair_cache;
    }

    if (residual_history) {
        residual_history->clear();
        const int history_reserve_iters = std::max(0, params.max_global_iters);
        residual_history->reserve(static_cast<std::size_t>(history_reserve_iters) + 1);
    }

    SolverResult result;
    result.final_residual = 0.0;
    result.iterations     = 0;
    double residual_tolerance = params.tol_abs;
    const int color_rebuild_interval = std::max(1, params.color_rebuild_interval);
    SweptBvhCache swept_bvh_cache;
    std::vector<std::vector<int>> cached_color_groups;
    std::vector<JacobiPrediction> jacobi_predictions;
    jacobi_predictions.reserve(num_vertices);
    std::vector<AABB> blue_boxes(num_vertices);
    std::vector<double> trust_blue_radii;
    RedBoxes red_boxes;
    GreenBoxes green_boxes;
    std::vector<Vec3> positions_before_iter;
    std::vector<Vec3> replay_positions;
    if (barrier_enabled && params.ccd_check) {
        positions_before_iter.resize(num_vertices);
        replay_positions.resize(num_vertices);
    }
    if (override_colors) result.last_num_colors = static_cast<int>(override_colors->size());

    auto color_groups_for_iteration = [&](int outer_iter) -> const std::vector<std::vector<int>>& {
        if (override_colors) return *override_colors;

        const bool should_rebuild_color_groups =
            cached_color_groups.empty() || ((outer_iter - 1) % color_rebuild_interval == 0);
        if (should_rebuild_color_groups) {
            const std::vector<std::vector<int>> contact_adjacency =
                barrier_enabled ? build_contact_adj(*active_pair_cache, num_vertices) : std::vector<std::vector<int>>(num_vertices);
            const std::vector<std::vector<int>> merged_adjacency = union_adjacency(elastic_adjacency, contact_adjacency);
            // Step 5: define colors for parallel GS updates.
            cached_color_groups = greedy_color_conflict_graph(
                build_conflict_graph(ref_mesh, pins, *active_pair_cache, jacobi_predictions, &adj, &merged_adjacency, &swept_bvh_cache),
                jacobi_predictions);
        }
        return cached_color_groups;
    };

    auto apply_colored_gauss_seidel = [&](const std::vector<std::vector<int>>& color_groups) {
        for (const auto& color_group : color_groups) {
            if (color_group.empty()) continue;

            #pragma omp parallel for schedule(static) if(params.use_parallel && color_group.size() >= 16)
            for (int local_idx = 0; local_idx < static_cast<int>(color_group.size()); ++local_idx) {
                const int vertex = color_group[local_idx];
                Vec3 unused_gradient, fresh_delta;
                Mat33 unused_hessian;

                // Compute GS direction for this vertex against current state.
                compute_local_newton_direction(vertex, ref_mesh, adj, pins, params, xnew, xhat,
                                               *active_pair_cache, unused_gradient, unused_hessian, fresh_delta, &pin_map);

                // Clamp to the blue box only when barriers are enabled.
                // With barriers off, skip certified-region clipping entirely.
                Vec3 clipped_delta = fresh_delta;
                if (barrier_enabled) {
                    const double clip_alpha =
                        clip_step_to_certified_region(vertex, xnew, fresh_delta, blue_boxes[vertex]);
                    clipped_delta = clip_alpha * fresh_delta;
                }

                // CCD against the current active barrier-pair cache.
                const double safe_step = compute_safe_step_for_vertex(vertex, ref_mesh, params, xnew, clipped_delta, *active_pair_cache);
                xnew[vertex] -= safe_step * clipped_delta;
            }
        }
    };

    auto has_ccd_replay_violation = [&](const std::vector<std::vector<int>>& color_groups) -> bool {
        if (!barrier_enabled || !params.ccd_check) return false;

        replay_positions = positions_before_iter;
        for (const auto& color_group : color_groups) {
            for (int vertex : color_group) {
                const Vec3 move = xnew[vertex] - positions_before_iter[vertex];
                if (move.squaredNorm() <= 0.0) {
                    replay_positions[vertex] = xnew[vertex];
                    continue;
                }
                const double replay_safe_step = compute_safe_step_for_vertex(
                    vertex, ref_mesh, params, replay_positions, -move, *active_pair_cache);
                if (replay_safe_step < 1.0 - 1.0e-12) return true;
                replay_positions[vertex] = xnew[vertex];
            }
        }
        return false;
    };

    for (int outer_iter = 1; outer_iter <= params.max_global_iters; ++outer_iter) {
        if (barrier_enabled && params.ccd_check) positions_before_iter = xnew;

        // Step 1: Jacobi predictions against the current pair cache.
        build_jacobi_prediction_deltas(ref_mesh, adj, pins, params, xnew, xhat, *active_pair_cache, jacobi_predictions, &pin_map);

        // Residual from predictions.
        const double residual_inf_norm =
            params.fixed_iters ? 0.0 : compute_prediction_residual_inf_norm(ref_mesh, jacobi_predictions, params.use_parallel);
        if (outer_iter == 1) {
            result.initial_residual = residual_inf_norm;
            residual_tolerance = std::max(params.tol_abs, params.tol_rel * residual_inf_norm);
        }
        result.final_residual = residual_inf_norm;
        if (residual_history) residual_history->push_back(residual_inf_norm);
        const bool should_stop_on_residual = !params.fixed_iters && residual_inf_norm < residual_tolerance;
        if (should_stop_on_residual) {
            result.iterations = outer_iter - 1;
            result.converged  = true;
            return result;
        }

        // Step 2: define node certified regions, i.e. "blue boxes".
        // With trust-region on, the radius is the b_v = gamma_P * d0_min
        const std::vector<double>* blue_box_radii = nullptr;
        if (barrier_enabled && params.use_trust_region) {
            trust_blue_radii.resize(num_vertices);
            #pragma omp parallel for if(params.use_parallel)
            for (int vi = 0; vi < num_vertices; ++vi) {
                const double b = compute_trust_region_bound_for_vertex(vi, xnew, *active_pair_cache, 0.4);
                trust_blue_radii[vi] = std::isfinite(b) ? b : jacobi_predictions[vi].delta.norm();
            }
            blue_box_radii = &trust_blue_radii;
        }
        
        build_blue_boxes(xnew, params.use_parallel, jacobi_predictions, &blue_boxes, blue_box_radii);

        // Step 3: define (red) edge and triangle boxes from node (blue) boxes.
        if (barrier_enabled) {
            const auto& mesh_edges = broad_phase.cache().edges;
            build_red_boxes(ref_mesh, mesh_edges, blue_boxes, red_boxes);

            // Step 4: define padded (green) boxes and register barrier pairs.
            build_green_boxes(red_boxes, dhat, green_boxes);
            barrier_pair_cache = register_barrier_pairs_from_blue_and_green(
                ref_mesh, mesh_edges, blue_boxes, green_boxes);
            active_pair_cache = &barrier_pair_cache;
        }

        // Step 5: define/reuse conflict graph colors.
        const std::vector<std::vector<int>>& color_groups_to_apply = color_groups_for_iteration(outer_iter);
        result.last_num_colors = static_cast<int>(color_groups_to_apply.size());

        // Step 6: colored GS.
        apply_colored_gauss_seidel(color_groups_to_apply);

        // Step 7: replay CCD sanity check in the same coloring order.
        if (has_ccd_replay_violation(color_groups_to_apply)) result.ccd_violations += 1;

        result.iterations = outer_iter;
    }

    // Step 8: optional residual tolerance (skipped when params.fixed_iters).
    if (params.fixed_iters) {
        result.final_residual = 0.0;
        result.converged      = true;
    } else {
        result.final_residual = compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pin_map);
    }

    return result;
}

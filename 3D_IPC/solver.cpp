#include "solver.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "parallel_helper.h"

#include <cstdio>

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
        double t = node_triangle_general_ccd(
                x[p.node],     dx[p.node],
                x[p.tri_v[0]], dx[p.tri_v[0]],
                x[p.tri_v[1]], dx[p.tri_v[1]],
                x[p.tri_v[2]], dx[p.tri_v[2]]);
        toi_min = std::min(toi_min, t);
    }

    const int n_ss = static_cast<int>(cache.ss_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_ss; ++i) {
        const auto& p = cache.ss_pairs[i];
        double t = segment_segment_general_ccd(
                x[p.v[0]], dx[p.v[0]],
                x[p.v[1]], dx[p.v[1]],
                x[p.v[2]], dx[p.v[2]],
                x[p.v[3]], dx[p.v[3]]);
        toi_min = std::min(toi_min, t);
    }

    double omega = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];

    return xnew;
}

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2 = params.dt2();

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2 * bg;
            H += dt2 * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2 * bg;
            H += dt2 * bH;
        }
    }

    const Vec3 delta = matrix3d_inverse(H) * g;

    double step = 1.0;
    if (params.d_hat > 0.0) {
        const Vec3 dx = -delta;

        const auto ccd = broad_phase.query_single_node_ccd(x, vi, dx, ref_mesh);

        double toi_min = 1.0;

        // vi as the lone moving node
        for (const auto& p : ccd.nt_node_pairs) {
            auto r = node_triangle_only_one_node_moves(
                x[p.node],     dx,
                x[p.tri_v[0]], Vec3::Zero(),
                x[p.tri_v[1]], Vec3::Zero(),
                x[p.tri_v[2]], Vec3::Zero());
            if (r.collision) toi_min = std::min(toi_min, r.t);
        }

        // vi as one moving triangle vertex
        for (const auto& p : ccd.nt_face_pairs) {
            Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
            dxv[p.vi_local] = dx;
            auto r = node_triangle_only_one_node_moves(
                x[p.node],     Vec3::Zero(),
                x[p.tri_v[0]], dxv[0],
                x[p.tri_v[1]], dxv[1],
                x[p.tri_v[2]], dxv[2]);
            if (r.collision) toi_min = std::min(toi_min, r.t);
        }

        for (const auto& p : ccd.ss_pairs) {
            CCDResult r;
            if (p.vi_dof == 0)
                r = segment_segment_only_one_node_moves(x[p.v[0]], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]]);
            else
                r = segment_segment_only_one_node_moves(x[p.v[1]], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]]);
            if (r.collision) toi_min = std::min(toi_min, r.t);
        }

        step = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;
    }

    x[vi] -= step * delta;
}

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, BroadPhase& broad_phase,
                                        const std::vector<Vec3>& v, const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history) {
    const double dt         = params.dt();
    const double dhat       = params.d_hat;
    const bool   use_barrier = dhat > 0.0;
    const double node_pad   = dhat;
    const double tri_pad    = 0.0;
    const double edge_pad   = dhat * 0.5;
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
    result.initial_residual = eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < params.tol_abs) {
        result.converged = true;
        return result;
    }

    const int nv = static_cast<int>(xnew.size());

    // Skip per-vertex refresh when motion < 80% of edge_pad. edge_pad is the
    // smallest broad-phase pad; staying under it keeps the pair-set valid.
    const double refresh_skip_thresh  = 0.8 * edge_pad;
    const double refresh_skip_thresh2 = refresh_skip_thresh * refresh_skip_thresh;
    std::vector<Vec3> x_at_last_refresh = xnew;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        std::vector<Vec3> x_before;
        if (use_barrier && params.ccd_check) x_before = xnew;

        for (const auto& group : color_groups) {
            for (int vi : group) {
                update_one_vertex(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm);
                if (use_barrier) {
                    const Vec3 d = xnew[vi] - x_at_last_refresh[vi];
                    if (d.squaredNorm() >= refresh_skip_thresh2) {
                        broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                        x_at_last_refresh[vi] = xnew[vi];
                    }
                }
            }
        }

        result.final_residual = eval_residual();
        result.iterations     = iter;

        // Optional post-sweep CCD penetration check. toi < 1 means the
        // per-vertex line-search missed a pair and the sweep tunneled.
        if (use_barrier && params.ccd_check) {
            std::vector<Vec3> dx(nv);
            for (int i = 0; i < nv; ++i) dx[i] = xnew[i] - x_before[i];

            BroadPhase ccd_bp;
            ccd_bp.build_ccd_candidates(x_before, dx, ref_mesh, 1.0);
            const auto& ccd_cache = ccd_bp.cache();

            double toi_min = 1.0;
            int hit_type = 0;  // 0=none, 1=node-tri, 2=seg-seg

            for (int i = 0; i < static_cast<int>(ccd_cache.nt_pairs.size()); ++i) {
                const auto& p = ccd_cache.nt_pairs[i];
                double t = node_triangle_general_ccd(
                    x_before[p.node],     dx[p.node],
                    x_before[p.tri_v[0]], dx[p.tri_v[0]],
                    x_before[p.tri_v[1]], dx[p.tri_v[1]],
                    x_before[p.tri_v[2]], dx[p.tri_v[2]]);
                if (t < toi_min) { toi_min = t; hit_type = 1; }
            }

            for (int i = 0; i < static_cast<int>(ccd_cache.ss_pairs.size()); ++i) {
                const auto& p = ccd_cache.ss_pairs[i];
                double t = segment_segment_general_ccd(
                    x_before[p.v[0]], dx[p.v[0]],
                    x_before[p.v[1]], dx[p.v[1]],
                    x_before[p.v[2]], dx[p.v[2]],
                    x_before[p.v[3]], dx[p.v[3]]);
                if (t < toi_min) { toi_min = t; hit_type = 2; }
            }

            if (toi_min < 1.0) {
                std::fprintf(stderr,
                    "[serial sanity] CCD violation after iter %d: toi=%.6e (%s)\n",
                    iter, toi_min, hit_type == 1 ? "node-triangle" : "segment-segment");
            }
        }

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < params.tol_abs) {
            result.converged = true;
            return result;
        }
    }

    return result;
}

SolverResult global_gauss_seidel_solver_parallel(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                 const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                                 BroadPhase& broad_phase, const std::vector<Vec3>& v, std::vector<double>* residual_history,
                                                 const std::vector<std::vector<int>>* override_colors) {
    const double dt          = params.dt();
    const double dhat        = params.d_hat;
    const bool   use_barrier = dhat > 0.0;
    const double node_pad    = dhat;
    const double tri_pad     = 0.0;
    const double edge_pad    = dhat * 0.5;
    const PinMap pm = build_pin_map(pins, static_cast<int>(xnew.size()));

    SolverResult result;

    if (use_barrier) {
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);
    }

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pm);
    };

    result.initial_residual = eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < params.tol_abs) {
        result.converged = true;
        return result;
    }

    // Skip per-vertex refresh when motion < 80% of edge_pad. edge_pad is the
    // smallest broad-phase pad; staying under it keeps the pair-set valid.
    const double refresh_skip_thresh  = 0.8 * edge_pad;
    const double refresh_skip_thresh2 = refresh_skip_thresh * refresh_skip_thresh;

    // Position snapshot at the last broad-phase refresh per vertex.
    std::vector<Vec3> x_at_last_refresh = xnew;

    // Coloring is rebuilt every outer iteration.
    std::vector<std::vector<int>> cached_color_groups;
    bool cached_colors_valid = false;
    constexpr int color_rebuild_interval = 1;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        std::vector<Vec3> x_before;
        if (use_barrier && params.ccd_check) x_before = xnew;

        // Phase 1: Jacobi prediction
        std::vector<JacobiPrediction> predictions;
        build_jacobi_predictions(ref_mesh, adj, pins, params, xnew, xhat, broad_phase.cache(), predictions, &pm);

        // Tests can pin a schedule via override_colors.
        const std::vector<std::vector<int>>* color_groups_ptr = override_colors;
        if (!color_groups_ptr) {
            const bool need_rebuild = !cached_colors_valid ||
                                      ((iter - 1) % color_rebuild_interval) == 0;
            if (need_rebuild) {
                const auto conflict_graph = build_conflict_graph(ref_mesh, pins, broad_phase.cache(), predictions, &adj);
                cached_color_groups = greedy_color_conflict_graph(conflict_graph, predictions);
                cached_colors_valid = true;
            }
            color_groups_ptr = &cached_color_groups;
        }
        const auto& color_groups = *color_groups_ptr;

        // Phase 2: colored Gauss-Seidel
        for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
            const auto& group = color_groups[color_idx];
            if (group.empty()) continue;

            const bool use_cached_prediction = (color_idx == 0);
            std::vector<ParallelCommit> commits(group.size());

            #pragma omp parallel for schedule(static) if(params.use_parallel && group.size() >= 16)
            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int vi = group[local_idx];
                commits[local_idx] = compute_parallel_commit_for_vertex(vi, use_cached_prediction, predictions[vi],
                    ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pm);
            }
            apply_parallel_commits(commits, xnew);

            if (use_barrier) {
                for (int vi : group) {
                    const Vec3 d = xnew[vi] - x_at_last_refresh[vi];
                    if (d.squaredNorm() < refresh_skip_thresh2) continue;
                    broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                    x_at_last_refresh[vi] = xnew[vi];
                }
            }
        }

        result.last_num_colors = static_cast<int>(color_groups.size());

        // Optional post-sweep CCD penetration check. toi < 1 means a pair
        // was missed by single-node CCD and the sweep tunneled.
        if (use_barrier && params.ccd_check) {
            const int nv_local = static_cast<int>(xnew.size());
            std::vector<Vec3> dx(nv_local);
            for (int i = 0; i < nv_local; ++i) dx[i] = xnew[i] - x_before[i];

            BroadPhase ccd_bp;
            ccd_bp.build_ccd_candidates(x_before, dx, ref_mesh, 1.0);
            const auto& ccd_cache = ccd_bp.cache();

            double toi_min = 1.0;
            int hit_type = 0;

            for (int i = 0; i < static_cast<int>(ccd_cache.nt_pairs.size()); ++i) {
                const auto& p = ccd_cache.nt_pairs[i];
                double t = node_triangle_general_ccd(
                    x_before[p.node],     dx[p.node],
                    x_before[p.tri_v[0]], dx[p.tri_v[0]],
                    x_before[p.tri_v[1]], dx[p.tri_v[1]],
                    x_before[p.tri_v[2]], dx[p.tri_v[2]]);
                if (t < toi_min) { toi_min = t; hit_type = 1; }
            }
            for (int i = 0; i < static_cast<int>(ccd_cache.ss_pairs.size()); ++i) {
                const auto& p = ccd_cache.ss_pairs[i];
                double t = segment_segment_general_ccd(
                    x_before[p.v[0]], dx[p.v[0]],
                    x_before[p.v[1]], dx[p.v[1]],
                    x_before[p.v[2]], dx[p.v[2]],
                    x_before[p.v[3]], dx[p.v[3]]);
                if (t < toi_min) { toi_min = t; hit_type = 2; }
            }

            if (toi_min < 1.0) {
                result.ccd_violations += 1;
                std::fprintf(stderr,
                    "[parallel sanity] CCD violation after iter %d: toi=%.6e (%s)\n",
                    iter, toi_min, hit_type == 1 ? "node-triangle" : "segment-segment");
            }
        }

        result.final_residual = eval_residual();
        result.iterations     = iter;

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < params.tol_abs) {
            result.converged = true;
            return result;
        }
    }

    return result;
}
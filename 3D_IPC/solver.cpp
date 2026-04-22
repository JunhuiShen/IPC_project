#include "solver.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "parallel_helper.h"
#include "trust_region.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <limits>


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
std::vector<Vec3> trust_region_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, double d_hat) {
    const int nv = static_cast<int>(x.size());
    constexpr double gamma_P = 0.4;  // 0 < gamma_P < 0.5

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase ccd_bp;
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0);
    const auto& cache = ccd_bp.cache();

    // b[v] = min over every barrier-active candidate pair that touches v of d0,
    // eventually multiplied by gamma_P.
    std::vector<double> b(nv, std::numeric_limits<double>::infinity());

    for (const auto& p : cache.nt_pairs) {
        const double d0 = node_triangle_distance(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]).distance;
        if (d0 >= d_hat) continue;
        b[p.node]     = std::min(b[p.node],     d0);
        b[p.tri_v[0]] = std::min(b[p.tri_v[0]], d0);
        b[p.tri_v[1]] = std::min(b[p.tri_v[1]], d0);
        b[p.tri_v[2]] = std::min(b[p.tri_v[2]], d0);
    }
    for (const auto& p : cache.ss_pairs) {
        const double d0 = segment_segment_distance(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]).distance;
        if (d0 >= d_hat) continue;
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
        const bool tr = params.use_trust_region;

        const auto pairs = broad_phase.query_pairs_for_vertex(x, vi, dx, ref_mesh);

        double safe_min = 1.0;

        // vi as the lone moving node
        for (const auto& p : pairs.nt_node_pairs) {
            if (tr) {
                auto r = trust_region_vertex_triangle_gauss_seidel(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
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
                if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
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
                if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
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
    const double dt          = params.dt();
    const double dhat        = params.d_hat;
    const bool   use_barrier = dhat > 0.0;
    const int    nv          = static_cast<int>(xnew.size());
    const PinMap pm          = build_pin_map(pins, nv);

    // One initialize() call to populate mesh topology (edges, node_to_tris,
    // node_to_edges) and to seed iter 1's Jacobi prediction with a
    // conservative velocity-swept pair set. Every subsequent iter rebuilds
    // the pair cache from blue/green boxes.
    if (use_barrier) {
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);
    } else if (!broad_phase.has_topology()) {
        broad_phase.set_mesh_topology(ref_mesh, nv);
    }

    // Static elastic adjacency.
    const std::vector<std::vector<int>> elastic_adj = build_elastic_adj(ref_mesh, adj, nv);

    // Per-iter pair cache. Seeded from the initialize() swept-AABB pairs so
    // iter 1's Jacobi prediction has a non-empty contact pair list; then
    // overwritten each iter by register_barrier_pairs_from_blue_green.
    BroadPhase::Cache iter_cache_storage;
    const BroadPhase::Cache* iter_cache = nullptr;
    if (use_barrier) {
        iter_cache = &broad_phase.cache();
    } else {
        iter_cache_storage.vertex_nt.assign(nv, {});
        iter_cache_storage.vertex_ss.assign(nv, {});
        iter_cache = &iter_cache_storage;
    }

    if (residual_history) {
        residual_history->clear();
        const int reserve_n = std::max(0, params.max_global_iters);
        residual_history->reserve(static_cast<std::size_t>(reserve_n) + 1);
    }

    SolverResult result;
    result.final_residual = 0.0;
    result.iterations     = 0;
    double effective_tol  = params.tol_abs;
    const int color_rebuild_interval = std::max(1, params.color_rebuild_interval);
    SweptBvhCache sw_cache;
    std::vector<std::vector<int>> cached_colors;
    const std::vector<std::vector<int>>* last_color_groups = override_colors;
    std::vector<JacobiPrediction> predictions;
    predictions.reserve(nv);
    std::vector<AABB> blue_boxes(nv);
    std::vector<Vec3> x_before;
    std::vector<Vec3> x_replay;
    if (use_barrier && params.ccd_check) {
        x_before.resize(nv);
        x_replay.resize(nv);
    }

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if (use_barrier && params.ccd_check) x_before = xnew;

        // Step 1: Jacobi predictions against the current pair cache.
        build_jacobi_prediction_deltas(ref_mesh, adj, pins, params, xnew, xhat, *iter_cache, predictions, &pm);

        // Residual from predictions.
        double residual_now = 0.0;
        if (!params.fixed_iters) {
            double r_inf = 0.0;
            #pragma omp parallel for reduction(max:r_inf) schedule(static)
            for (int i = 0; i < nv; ++i) {
                Vec3 g = predictions[i].g;
                const double m = ref_mesh.mass[i];
                if (m > 0.0) g /= m;
                r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
            }
            residual_now = r_inf;
        }
        if (iter == 1) {
            result.initial_residual = residual_now;
            effective_tol = std::max(params.tol_abs, params.tol_rel * residual_now);
        }
        result.final_residual = residual_now;
        if (residual_history) residual_history->push_back(residual_now);
        if (!params.fixed_iters && residual_now < effective_tol) {
            result.iterations = iter - 1;
            if (last_color_groups) {
                result.last_num_colors = static_cast<int>(last_color_groups->size());
                result.color_groups_parallel = *last_color_groups;
            }
            result.converged  = true;
            return result;
        }

        // Step 2: define node certified regions, i.e. "blue boxes".
        build_blue_boxes_from_deltas(xnew, params.use_parallel, predictions, &blue_boxes);

        // Step 3: define (red) edge and triangle boxes from node (blue) boxes.
        if (use_barrier) {
            iter_cache_storage = register_barrier_pairs_from_blue_green(
                ref_mesh, broad_phase.cache().edges, blue_boxes, dhat);
            iter_cache = &iter_cache_storage;
        }

        // Step 4: define/reuse conflict graph colors. We rebuild every
        // color_rebuild_interval iterations; skipped iterations reuse the last
        // coloring to avoid the recoloring overhead.
        const std::vector<std::vector<int>>* color_groups_ptr = override_colors;
        if (!color_groups_ptr) {
            const bool rebuild_colors = cached_colors.empty() || ((iter - 1) % color_rebuild_interval == 0);
            if (rebuild_colors) {
                const std::vector<std::vector<int>> base_adj = union_adjacency(
                    elastic_adj,
                    use_barrier ? build_contact_adj(*iter_cache, nv) : std::vector<std::vector<int>>(nv));
                // Step 5: define colors for parallel GS updates.
                cached_colors = greedy_color_conflict_graph(
                    build_conflict_graph(ref_mesh, pins, *iter_cache, predictions, &adj, &base_adj, &sw_cache),
                    predictions);
            }
            color_groups_ptr = &cached_colors;
        }
        const std::vector<std::vector<int>>& color_groups = *color_groups_ptr;
        last_color_groups = &color_groups;

        // Step 6: colored GS.
        for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
            const auto& group = color_groups[color_idx];
            if (group.empty()) continue;

            #pragma omp parallel for schedule(static) if(params.use_parallel && group.size() >= 16)
            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int vi = group[local_idx];

                Vec3 g_fresh, delta_fresh;
                Mat33 H_fresh;

                // Compute GS delta
                compute_local_newton_direction(vi, ref_mesh, adj, pins, params, xnew, xhat,
                                               *iter_cache, g_fresh, H_fresh, delta_fresh, &pm);

                // Clamp the step to stay inside the node's blue box.
                const Vec3 delta = (use_barrier ? clip_step_to_certified_region(vi, xnew, delta_fresh, blue_boxes[vi]) : 1.0) * delta_fresh;

                // CCD against the current iter_cache pairs.
                const double ccd_step = compute_safe_step_for_vertex(vi, ref_mesh, params, xnew, delta, *iter_cache);

                xnew[vi] -= ccd_step * delta;
            }
        }

        result.last_num_colors       = static_cast<int>(color_groups.size());

        // Step 7: CCD sanity check on the actual executed colored path.
        if (use_barrier && params.ccd_check) {
            bool violation = false;
            x_replay = x_before;
            for (const auto& group : color_groups) {
                for (int vi : group) {
                    const Vec3 move = xnew[vi] - x_before[vi];
                    if (move.squaredNorm() <= 0.0) {
                        x_replay[vi] = xnew[vi];
                        continue;
                    }
                    const double safe_replay = compute_safe_step_for_vertex(
                        vi, ref_mesh, params, x_replay, -move, *iter_cache);
                    if (safe_replay < 1.0 - 1.0e-12) {
                        violation = true;
                        break;
                    }
                    x_replay[vi] = xnew[vi];
                }
                if (violation) break;
            }
            if (violation) result.ccd_violations += 1;
        }

        result.iterations = iter;
    }

    // Step 8: optional residual tolerance (skipped when params.fixed_iters).
    if (params.fixed_iters) {
        result.final_residual = 0.0;
        result.converged      = true;
    } else {
        result.final_residual = compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pm);
    }
    if (last_color_groups) {
        result.last_num_colors = static_cast<int>(last_color_groups->size());
        result.color_groups_parallel = *last_color_groups;
    }

    return result;
}

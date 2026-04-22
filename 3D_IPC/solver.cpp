#include "solver.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "parallel_helper.h"
#include "trust_region.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <cstdint>
#include <cstdio>
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

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (!params.fixed_iters && result.initial_residual < effective_tol) {
        result.converged = true;
        return result;
    }

    const int nv = static_cast<int>(xnew.size());

    const double node_pad = dhat;
    const double tri_pad  = 0.0;
    const double edge_pad = dhat * 0.5;
    const bool   do_refresh = use_barrier && params.use_incremental_refresh;
    // Refresh once accumulated drift could have moved pairs past half the
    // swept-box pad; any later and the current BVH leaves may already miss
    // newly-reachable pairs.
    const double skip_thresh2 = (0.8 * edge_pad) * (0.8 * edge_pad);
    std::vector<Vec3> x_at_last_refresh;
    if (do_refresh) x_at_last_refresh = xnew;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {

        for (const auto& group : color_groups) {
            for (int vi : group) {
                update_one_vertex(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm);
                if (do_refresh) {
                    const Vec3 d = xnew[vi] - x_at_last_refresh[vi];
                    if (d.squaredNorm() >= skip_thresh2) {
                        broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                        x_at_last_refresh[vi] = xnew[vi];
                    }
                }
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

SolverResult global_gauss_seidel_solver_parallel_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
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

    // Snapshot the edge list.
    const std::vector<std::array<int, 2>> edges = broad_phase.cache().edges;

    // Static elastic adjacency.
    const std::vector<std::vector<int>> elastic_adj = build_elastic_adj(ref_mesh, adj, nv);

    // Per-iter pair cache. Seeded from the initialize() swept-AABB pairs so
    // iter 1's Jacobi prediction has a non-empty contact pair list; then
    // overwritten each iter by register_barrier_pairs_from_blue_green.
    BroadPhase::Cache iter_cache;
    if (use_barrier) {
        iter_cache = broad_phase.cache();
    } else {
        iter_cache.vertex_nt.assign(nv, {});
        iter_cache.vertex_ss.assign(nv, {});
    }

    if (residual_history) residual_history->clear();

    SolverResult result;
    result.final_residual = 0.0;
    result.iterations     = 0;
    double effective_tol  = params.tol_abs;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {

        // Step 1: Jacobi predictions against the current pair cache.
        std::vector<JacobiPrediction> predictions;
        build_jacobi_predictions(ref_mesh, adj, pins, params, xnew, xhat, iter_cache, predictions, &pm);

        // Residual from predictions.
        double residual_now = 0.0;
        if (!params.fixed_iters) {
            const bool normalize = params.mass_normalize_residual;
            double r_inf = 0.0;
            #pragma omp parallel for reduction(max:r_inf) schedule(static)
            for (int i = 0; i < nv; ++i) {
                Vec3 g = predictions[i].g;
                if (normalize) {
                    const double m = ref_mesh.mass[i];
                    if (m > 0.0) g /= m;
                }
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
            result.converged  = true;
            return result;
        }

        // Step 2: define node certified regions, i.e. "blue boxes".
        std::vector<AABB> blue_boxes(nv);
        for (int vi = 0; vi < nv; ++vi) {
            const double r = predictions[vi].delta.norm();
            const Vec3 half = Vec3::Constant(r);
            AABB box;
            box.expand(xnew[vi] - half);
            box.expand(xnew[vi] + half);
            blue_boxes[vi] = box;
        }

        // Step 3: define (red) edge and triangle boxes from node (blue) boxes.
        if (use_barrier) {
            iter_cache = register_barrier_pairs_from_blue_green(ref_mesh, edges, blue_boxes, dhat);
        }

        // Step 4: define conflict graph from barrier pair and elastic mesh connectivity.
        const std::vector<std::vector<int>> contact_adj = use_barrier? build_contact_adj(iter_cache, nv): std::vector<std::vector<int>>(nv);
        const std::vector<std::vector<int>> conflict = union_adjacency(elastic_adj, contact_adj);

        // Step 5: define colors for parallel GS updates.
        const std::vector<std::vector<int>> local_colors = override_colors? std::vector<std::vector<int>>() : greedy_color_conflict_graph(conflict, predictions);
        const std::vector<std::vector<int>>& color_groups = override_colors ? *override_colors : local_colors;

        // Step 6: colored GS.
        for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
            const auto& group = color_groups[color_idx];
            if (group.empty()) continue;

            #pragma omp parallel for schedule(static) if(params.use_parallel && group.size() >= 16)
            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int vi = group[local_idx];

                Vec3 g_fresh, delta_fresh;
                Mat33 H_fresh;
                compute_local_newton_direction(vi, ref_mesh, adj, pins, params, xnew, xhat,
                                               iter_cache, g_fresh, H_fresh, delta_fresh, &pm);

                // (a) clamp the step to stay inside the node's blue box.
                const double alpha_clip = use_barrier? clip_step_to_certified_region(vi, xnew, delta_fresh, blue_boxes[vi]): 1.0;
                const Vec3 delta = alpha_clip * delta_fresh;

                // (b) CCD against the current iter_cache pairs.
                const double ccd_step = compute_safe_step_for_vertex(vi, ref_mesh, params, xnew, delta, iter_cache);

                xnew[vi] -= ccd_step * delta;
            }
        }

        result.last_num_colors       = static_cast<int>(color_groups.size());
        result.color_groups_parallel = color_groups;
        result.recolor_count        += 1;


        result.iterations = iter;
    }

    // Step 8: optional residual tolerance (skipped when params.fixed_iters).
    if (params.fixed_iters) {
        result.final_residual = 0.0;
        result.converged      = true;
    } else {
        result.final_residual = compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pm);
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
    const PinMap pm = build_pin_map(pins, static_cast<int>(xnew.size()));

    SolverResult result;

    if (use_barrier) {
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);
    }

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, &pm);
    };

    // The iter==1 residual check inside the loop reads from predictions[] and
    // sets initial_residual + effective_tol on the fly. We only fall back to
    // eval_residual() at exit if the loop terminates on max_substep_iters.
    result.final_residual = 0.0;
    result.iterations     = 0;
    double effective_tol  = params.tol_abs;  // overwritten on iter 1

    // Per-solver-call caches. Emptiness / size / version mismatch encodes
    // "stale" — no separate validity flags.
    const int color_interval = (params.color_rebuild_interval > 0)
                               ? params.color_rebuild_interval : 1;
    std::vector<std::vector<int>> cached_color_groups;
    std::vector<std::vector<int>> cached_elastic_adj;
    std::vector<std::vector<int>> cached_base_adj;
    std::uint64_t                 cached_bp_version = 0;
    SweptBvhCache                 cached_sw_bvh;
    std::vector<AABB>             frozen_certified_regions;
    std::uint64_t                 last_recolor_bp_version = 0;

    const double node_pad = dhat;
    const double tri_pad  = 0.0;
    const double edge_pad = dhat * 0.5;
    const bool   do_refresh = use_barrier && params.use_incremental_refresh;
    // Threshold matches the serial solver; see that comment for the rationale.
    const double skip_thresh2 = (0.8 * edge_pad) * (0.8 * edge_pad);
    std::vector<Vec3> x_at_last_refresh;
    if (do_refresh) x_at_last_refresh = xnew;

    // Reuses the g_i already computed by build_jacobi_predictions, so the
    // convergence check is essentially free vs. a full compute_global_residual.
    auto residual_from_predictions = [&](const std::vector<JacobiPrediction>& preds) {
        const bool normalize = params.mass_normalize_residual;
        const int nv_local = static_cast<int>(preds.size());
        double r_inf = 0.0;
        #pragma omp parallel for reduction(max:r_inf) schedule(static)
        for (int i = 0; i < nv_local; ++i) {
            Vec3 g = preds[i].g;
            if (normalize) {
                const double m = ref_mesh.mass[i];
                if (m > 0.0) g /= m;
            }
            r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
        }
        return r_inf;
    };

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {

        std::vector<JacobiPrediction> predictions;
        build_jacobi_predictions(ref_mesh, adj, pins, params, xnew, xhat, broad_phase.cache(), predictions, &pm);

        // Freeze certified-region AABBs inside the recolor window. The sweep
        // clips every delta into its box; with boxes held constant, the
        // conflict graph built below is bit-identical to the one the cached
        // coloring was produced from, so the cached coloring stays valid —
        // unless an incremental refresh added new barrier pairs to bp_cache,
        // in which case bp_v_now has advanced and we must recolor to pick up
        // the new conflict edges before the next sweep races on them.
        const std::uint64_t bp_v_now = use_barrier ? broad_phase.version() : 0;
        const bool have_frozen       = !frozen_certified_regions.empty();
        const bool periodic_tick     = ((iter - 1) % color_interval == 0);
        const bool bp_changed        = have_frozen && bp_v_now != last_recolor_bp_version;
        const bool recolor_this_iter = !have_frozen || periodic_tick || bp_changed;

        if (recolor_this_iter) {
            frozen_certified_regions.resize(predictions.size());
            for (std::size_t i = 0; i < predictions.size(); ++i)
                frozen_certified_regions[i] = predictions[i].certified_region;
        } else {
            for (std::size_t i = 0; i < predictions.size(); ++i)
                predictions[i].certified_region = frozen_certified_regions[i];
        }

        const double residual_now = params.fixed_iters ? 0.0 : residual_from_predictions(predictions);
        if (iter == 1) {
            result.initial_residual = residual_now;
            effective_tol = std::max(params.tol_abs, params.tol_rel * residual_now);
        }
        result.final_residual = residual_now;
        if (residual_history) residual_history->push_back(residual_now);
        if (!params.fixed_iters && residual_now < effective_tol) {
            result.iterations = iter - 1;
            result.converged  = true;
            return result;
        }

        // Tests can pin a schedule via override_colors.
        const std::vector<std::vector<int>>* color_groups_ptr = override_colors;
        if (!color_groups_ptr) {
            const int nv_preds = static_cast<int>(predictions.size());

            if (static_cast<int>(cached_elastic_adj.size()) != nv_preds) {
                cached_elastic_adj = build_elastic_adj(ref_mesh, adj, nv_preds);
                cached_base_adj.clear();  // derived; rebuild below
            }
            const std::uint64_t bp_v = use_barrier ? broad_phase.version() : 0;
            if (static_cast<int>(cached_base_adj.size()) != nv_preds || bp_v != cached_bp_version) {
                auto contact_adj = use_barrier ? build_contact_adj(broad_phase.cache(), nv_preds)
                                               : std::vector<std::vector<int>>(nv_preds);
                cached_base_adj   = union_adjacency(cached_elastic_adj, contact_adj);
                cached_bp_version = bp_v;
            }

            if (cached_color_groups.empty() || recolor_this_iter) {
                const auto conflict_graph = build_conflict_graph(ref_mesh, pins, broad_phase.cache(), predictions, &adj, &cached_base_adj, &cached_sw_bvh);
                cached_color_groups = greedy_color_conflict_graph(conflict_graph, predictions);
                result.color_groups_parallel = cached_color_groups; // for visualization
                result.recolor_count += 1;
                last_recolor_bp_version = bp_v_now;
            } else {
                result.recolor_skipped += 1;
            }
            color_groups_ptr = &cached_color_groups;
        }
        const auto& color_groups = *color_groups_ptr;

        // Color 0 reuses the cached prediction.delta; later colors see a
        // partially-updated xnew and recompute from scratch.
        for (std::size_t color_idx = 0; color_idx < color_groups.size(); ++color_idx) {
            const auto& group = color_groups[color_idx];
            if (group.empty()) continue;

            const bool use_cached_prediction = (color_idx == 0);
            const auto& bp_cache = broad_phase.cache();

            // Within a color, the conflict graph guarantees no two vertices
            // share an elastic, bending, or barrier pair, so writing xnew[vi]
            // immediately cannot race with another thread's reads.
            #pragma omp parallel for schedule(static) if(params.use_parallel && group.size() >= 16)
            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int vi = group[local_idx];
                const JacobiPrediction& prediction = predictions[vi];

                Vec3 delta = prediction.delta;
                if (!use_cached_prediction) {
                    Vec3 g_fresh, delta_fresh;
                    Mat33 H_fresh;
                    compute_local_newton_direction(vi, ref_mesh, adj, pins, params, xnew, xhat,
                                                   bp_cache, g_fresh, H_fresh, delta_fresh, &pm);
                    const double alpha_clip = use_barrier
                        ? clip_step_to_certified_region(vi, xnew, delta_fresh, prediction.certified_region)
                        : 1.0;
                    delta = alpha_clip * delta_fresh;
                }
                const double ccd_step = compute_safe_step_for_vertex(
                        vi, ref_mesh, params, xnew, delta, broad_phase.cache());
                xnew[vi] -= ccd_step * delta;
            }

            // refresh() mutates shared bp_cache state — run after the parallel
            // sweep finishes, serially over the group's moved vertices.
            if (do_refresh) {
                for (int vi : group) {
                    const Vec3 d = xnew[vi] - x_at_last_refresh[vi];
                    if (d.squaredNorm() < skip_thresh2) continue;
                    broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                    x_at_last_refresh[vi] = xnew[vi];
                }
            }
        }

        result.last_num_colors = static_cast<int>(color_groups.size());

        result.iterations = iter;
    }

    // residual_now was measured before the last sweep; recompute once so the
    // reported final_residual reflects the post-sweep state.
    if (params.fixed_iters) {
        result.final_residual = 0.0;
        result.converged = true;
    } else {
        result.final_residual = eval_residual();
    }
    if (residual_history) residual_history->push_back(result.final_residual);
    return result;
}
#include "parallel_helper.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "trust_region.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline int ph_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static inline int ph_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static void compute_local_newton_direction(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                           const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                           const BroadPhase::Cache& bp_cache, Vec3& g_out, Mat33& H_out, Vec3& delta_out,
                                           const PinMap* pin_map = nullptr){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2 = params.dt2();

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian( x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2 * bg;
            H += dt2 * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2 * bg;
            H += dt2 * bH;
        }
    }

    g_out = g;
    H_out = H;

    delta_out = matrix3d_inverse(H) * g;
}

// Certified region for vertex vi: isotropic cube centered at x[vi] with
// half-edge |delta|_2, unioned with incident tri/edge boxes. Every contributing
// sub-AABB is padded by d_hat/2 so disjoint certified regions are guaranteed
// d_hat-separated. The isotropic cube allows a fresh Newton step move in any direction 
// up to the cached step length without being clipped by a thin perpendicular extent. 
// edge_boxes already carry d_hat/2 from the broad phase; tri_boxes carry 0, 
// so only the cube and tri_boxes are grown here.
static AABB build_certified_region_for_vertex(int vi, const std::vector<Vec3>& x, const Vec3& delta, const BroadPhase::Cache& bp_cache, double d_hat){
    AABB box;

    const Vec3 half_dhat(0.5 * d_hat, 0.5 * d_hat, 0.5 * d_hat);
    const Vec3 jacobi_step_extent = Vec3::Constant(delta.norm());
    const Vec3& center = x[vi];

    box.expand(center - jacobi_step_extent - half_dhat);
    box.expand(center + jacobi_step_extent + half_dhat);

    if (vi >= 0 && vi < static_cast<int>(bp_cache.node_to_tris.size())) {
        for (int tri_idx : bp_cache.node_to_tris[vi]) {
            const AABB& tb = bp_cache.tri_boxes[tri_idx];
            box.expand(tb.min - half_dhat);
            box.expand(tb.max + half_dhat);
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.node_to_edges.size())) {
        for (int edge_idx : bp_cache.node_to_edges[vi]) {
            box.expand(bp_cache.edge_boxes[edge_idx]);
        }
    }

    return box;
}

void build_jacobi_predictions(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                              const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                              const BroadPhase::Cache& bp_cache, std::vector<JacobiPrediction>& predictions,
                              const PinMap* pin_map){
    const int nv = static_cast<int>(x.size());
    predictions.clear();
    predictions.resize(nv);

    #pragma omp parallel for if(params.use_parallel)
    for (int vi = 0; vi < nv; ++vi) {
        predictions[vi].active = true;

        Vec3 g, delta;
        Mat33 H;
        compute_local_newton_direction(vi, ref_mesh, adj, pins, params, x, xhat, bp_cache, g, H, delta, pin_map);

        predictions[vi].g = g;
        predictions[vi].H = H;
        predictions[vi].delta = delta;
        predictions[vi].certified_region = build_certified_region_for_vertex(vi, x, predictions[vi].delta, bp_cache, params.d_hat);
    }
}

std::vector<std::vector<int>> build_conflict_graph(const RefMesh& ref_mesh, const std::vector<Pin>& /*pins*/,
                                                   const BroadPhase::Cache& bp_cache, const std::vector<JacobiPrediction>& predictions,
                                                   const VertexTriangleMap* adj){
    const int nv = static_cast<int>(predictions.size());
    std::vector<std::vector<int>> graph(nv);

    VertexTriangleMap local_adj;
    if (!adj) {
        local_adj = build_incident_triangle_map(ref_mesh.tris);
        adj = &local_adj;
    }

    std::vector<AABB> active_boxes;
    std::vector<int> active_ids;
    active_boxes.reserve(nv);
    active_ids.reserve(nv);
    for (int i = 0; i < nv; ++i) {
        if (!predictions[i].active) continue;
        active_ids.push_back(i);
        active_boxes.push_back(predictions[i].certified_region);
    }
    std::vector<BVHNode> sw_bvh_nodes;
    const int sw_root = build_bvh(active_boxes, sw_bvh_nodes);

    // Each thread accumulates edges into its own per-vertex list; we merge
    // + dedup at the end. Order-independent, so a pure refactor of the
    // serial version.
    const int T = ph_max_threads();
    std::vector<std::vector<std::vector<int>>> local_nbr(T, std::vector<std::vector<int>>(nv));

    const int n_nt = static_cast<int>(bp_cache.nt_pairs.size());
    const int n_ss = static_cast<int>(bp_cache.ss_pairs.size());
    const int n_active = static_cast<int>(active_ids.size());

    #pragma omp parallel
    {
        auto& lg = local_nbr[ph_thread_num()];

        // Elastic coupling: vertices that share a triangle.
        if (adj) {
            #pragma omp for schedule(static) nowait
            for (int vi = 0; vi < nv; ++vi) {
                if (!predictions[vi].active) continue;
                auto it = adj->find(vi);
                if (it == adj->end()) continue;
                for (const auto& [ti, local_a] : it->second) {
                    for (int local_b = 0; local_b < 3; ++local_b) {
                        int vj = tri_vertex(ref_mesh, ti, local_b);
                        if (vj == vi || vj < 0 || vj >= nv || !predictions[vj].active) continue;
                        lg[vi].push_back(vj);
                        lg[vj].push_back(vi);
                    }
                }
            }
        }

        // Barrier node-triangle pairs.
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n_nt; ++i) {
            const auto& p = bp_cache.nt_pairs[i];
            const int verts[4] = { p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2] };
            for (int a = 0; a < 4; ++a) {
                if (verts[a] < 0 || verts[a] >= nv || !predictions[verts[a]].active) continue;
                for (int b = a + 1; b < 4; ++b) {
                    if (verts[b] < 0 || verts[b] >= nv || !predictions[verts[b]].active) continue;
                    lg[verts[a]].push_back(verts[b]);
                    lg[verts[b]].push_back(verts[a]);
                }
            }
        }

        // Barrier segment-segment pairs.
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n_ss; ++i) {
            const auto& p = bp_cache.ss_pairs[i];
            for (int a = 0; a < 4; ++a) {
                const int va = p.v[a];
                if (va < 0 || va >= nv || !predictions[va].active) continue;
                for (int b = a + 1; b < 4; ++b) {
                    const int vb = p.v[b];
                    if (vb < 0 || vb >= nv || !predictions[vb].active) continue;
                    lg[va].push_back(vb);
                    lg[vb].push_back(va);
                }
            }
        }

        // Swept-region overlap: BVH query per active vertex.
        #pragma omp for schedule(static)
        for (int ai = 0; ai < n_active; ++ai) {
            std::vector<int> hits;
            query_bvh(sw_bvh_nodes, sw_root, active_boxes[ai], hits);
            const int vi = active_ids[ai];
            for (int leaf : hits) {
                const int vj = active_ids[leaf];
                if (vj > vi) {
                    lg[vi].push_back(vj);
                    lg[vj].push_back(vi);
                }
            }
        }
    }

    // Merge per-thread locals + dedup, parallel over vertices.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        std::size_t total = 0;
        for (int t = 0; t < T; ++t) total += local_nbr[t][vi].size();
        graph[vi].reserve(total);
        for (int t = 0; t < T; ++t) {
            auto& src = local_nbr[t][vi];
            graph[vi].insert(graph[vi].end(), src.begin(), src.end());
        }
        std::sort(graph[vi].begin(), graph[vi].end());
        graph[vi].erase(std::unique(graph[vi].begin(), graph[vi].end()), graph[vi].end());
    }

    return graph;
}

std::vector<std::vector<int>> greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, const std::vector<JacobiPrediction>& predictions){
    const int nv = static_cast<int>(graph.size());
    std::vector<int> color(nv, -1);
    int max_color = -1;

    for (int vi = 0; vi < nv; ++vi) {
        if (!predictions[vi].active) continue;

        std::vector<char> used(max_color + 2, 0);
        for (int nb : graph[vi]) {
            if (nb >= 0 && nb < nv && color[nb] >= 0) {
                used[color[nb]] = 1;
            }
        }

        int c = 0;
        while (c < static_cast<int>(used.size()) && used[c]) ++c;
        color[vi] = c;
        max_color = std::max(max_color, c);
    }

    std::vector<std::vector<int>> groups(max_color + 1);
    for (int vi = 0; vi < nv; ++vi) {
        if (color[vi] >= 0) groups[color[vi]].push_back(vi);
    }

    return groups;
}

static double clip_step_to_certified_region(int vi, const std::vector<Vec3>& x, const Vec3& fresh_delta, const AABB& certified_region){
    double alpha = 1.0;

    for (int k = 0; k < 3; ++k) {
        const double x0 = x[vi](k);
        const double x1 = x0 - fresh_delta(k);
        const double lo = certified_region.min(k);
        const double hi = certified_region.max(k);

        if (x1 < lo) {
            const double denom = x0 - x1;
            if (std::abs(denom) > 1.0e-16) {
                alpha = std::min(alpha, (x0 - lo) / denom);
            }
        } else if (x1 > hi) {
            const double denom = x1 - x0;
            if (std::abs(denom) > 1.0e-16) {
                alpha = std::min(alpha, (hi - x0) / denom);
            }
        }
    }

    return clamp_scalar(alpha, 0.0, 1.0);
}

double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params, const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase& broad_phase){
    if (params.d_hat <= 0.0) return 1.0;

    const Vec3 dx = -delta;
    const bool tr = params.use_trust_region;

    const auto pairs = broad_phase.query_pairs_for_vertex(x, vi, dx, ref_mesh);

    double safe_min = 1.0;

    // vi is the lone moving node against static triangles.
    for (const auto& p : pairs.nt_node_pairs) {
        if (tr) {
            auto r = trust_region_vertex_triangle_gauss_seidel(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
            if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
        } else {
            CCDResult r = node_triangle_only_one_node_moves(
                x[p.node],     dx,
                x[p.tri_v[0]], Vec3::Zero(),
                x[p.tri_v[1]], Vec3::Zero(),
                x[p.tri_v[2]], Vec3::Zero());
            if (r.collision) safe_min = std::min(safe_min, r.t);
        }
    }

    // vi is one corner of a moving triangle against a static node.
    for (const auto& p : pairs.nt_face_pairs) {
        if (tr) {
            auto r = trust_region_vertex_triangle_gauss_seidel(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
            if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
        } else {
            Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
            dxv[p.vi_local] = dx;
            CCDResult r = node_triangle_only_one_node_moves(
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

    return tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
}

ParallelCommit compute_parallel_commit_for_vertex(int vi, bool use_cached_prediction, const JacobiPrediction& prediction,
                                                  const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                  const SimParams& params, const std::vector<Vec3>& x_current, const std::vector<Vec3>& xhat,
                                                  const BroadPhase& broad_phase, const PinMap* pin_map){
    const auto& bp_cache = broad_phase.cache();
    ParallelCommit out;
    out.vi = vi;

    Vec3 delta = prediction.delta;

    if (!use_cached_prediction) {
        Vec3 g_fresh, delta_fresh;
        Mat33 H_fresh;
        compute_local_newton_direction(vi, ref_mesh, adj, pins, params, x_current, xhat, bp_cache, g_fresh, H_fresh, delta_fresh, pin_map);

        out.alpha_clip = clip_step_to_certified_region(vi, x_current, delta_fresh, prediction.certified_region);
        delta = out.alpha_clip * delta_fresh;
    }

    out.delta = delta;

    // The conflict graph ensures no new barrier pair can arise
    // within this batch, so single-node CCD against the current barrier pairs suffices.
    out.ccd_step = compute_safe_step_for_vertex(vi, ref_mesh, params, x_current, delta, broad_phase);

    out.x_after = x_current[vi] - out.ccd_step * delta;
    out.valid = true;
    return out;
}

void apply_parallel_commits(const std::vector<ParallelCommit>& commits, std::vector<Vec3>& xnew){
    for (const auto& commit : commits) {
        if (!commit.valid) continue;
        xnew[commit.vi] = commit.x_after;
    }
}

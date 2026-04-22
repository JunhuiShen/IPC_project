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

void compute_local_newton_direction(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, Vec3& g_out, Mat33& H_out, Vec3& delta_out,
                                    const PinMap* pin_map,
                                    const std::vector<TriPrecompute>* tri_cache,
                                    const std::vector<HingePrecompute>* hinge_cache){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map, tri_cache, hinge_cache);

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

void build_jacobi_prediction_deltas(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, std::vector<JacobiPrediction>& predictions,
                                    const PinMap* pin_map){
    const int nv = static_cast<int>(x.size());
    predictions.assign(nv, {});

    // Per-triangle F/cache/P/dPdF and per-hinge cache are shared across every
    // corner of their stencil; the per-vertex call reads them instead of
    // recomputing from x.
    std::vector<TriPrecompute>   tri_cache;
    std::vector<HingePrecompute> hinge_cache;
    build_elastic_precompute(ref_mesh, x, params, /*want_hessian=*/true, tri_cache);
    build_bending_precompute(ref_mesh, x, params, hinge_cache);

    #pragma omp parallel for if(params.use_parallel)
    for (int vi = 0; vi < nv; ++vi) {
        predictions[vi].active = true;

        Vec3 g, delta;
        Mat33 H;
        compute_local_newton_direction(vi, ref_mesh, adj, pins, params, x, xhat, bp_cache, g, H, delta, pin_map, &tri_cache, &hinge_cache);

        predictions[vi].g = g;
        predictions[vi].delta = delta;
    }
}

void build_blue_boxes_from_deltas(const std::vector<Vec3>& x,
                                  bool use_parallel,
                                  std::vector<JacobiPrediction>& predictions,
                                  std::vector<AABB>* blue_boxes_out){
    const int nv = static_cast<int>(predictions.size());
    if (blue_boxes_out) blue_boxes_out->resize(nv);

    #pragma omp parallel for if(use_parallel)
    for (int vi = 0; vi < nv; ++vi) {
        AABB box;
        box.expand(x[vi] - Vec3::Constant(predictions[vi].delta.norm()));
        box.expand(x[vi] + Vec3::Constant(predictions[vi].delta.norm()));
        predictions[vi].certified_region = box;
        if (blue_boxes_out) (*blue_boxes_out)[vi] = box;
    }
}

std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv){
    std::vector<std::vector<int>> out(nv);
    #pragma omp parallel for schedule(static)
    for (int vi = 0; vi < nv; ++vi) {
        auto it = adj.find(vi);
        if (it == adj.end()) continue;
        std::vector<int>& row = out[vi];
        for (const auto& [ti, local_a] : it->second) {
            for (int local_b = 0; local_b < 3; ++local_b) {
                const int vj = tri_vertex(ref_mesh, ti, local_b);
                if (vj == vi || vj < 0 || vj >= nv) continue;
                row.push_back(vj);
            }
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return out;
}

std::vector<std::vector<int>> build_contact_adj(const BroadPhase::Cache& bp_cache, int nv){
    const int T = ph_max_threads();
    std::vector<std::vector<std::vector<int>>> local_nbr(T, std::vector<std::vector<int>>(nv));
    const int n_nt = static_cast<int>(bp_cache.nt_pairs.size());
    const int n_ss = static_cast<int>(bp_cache.ss_pairs.size());

    #pragma omp parallel
    {
        auto& lg = local_nbr[ph_thread_num()];
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n_nt; ++i) {
            const auto& p = bp_cache.nt_pairs[i];
            const int verts[4] = { p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2] };
            for (int a = 0; a < 4; ++a) {
                if (verts[a] < 0 || verts[a] >= nv) continue;
                for (int b = a + 1; b < 4; ++b) {
                    if (verts[b] < 0 || verts[b] >= nv) continue;
                    lg[verts[a]].push_back(verts[b]);
                    lg[verts[b]].push_back(verts[a]);
                }
            }
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n_ss; ++i) {
            const auto& p = bp_cache.ss_pairs[i];
            for (int a = 0; a < 4; ++a) {
                const int va = p.v[a];
                if (va < 0 || va >= nv) continue;
                for (int b = a + 1; b < 4; ++b) {
                    const int vb = p.v[b];
                    if (vb < 0 || vb >= nv) continue;
                    lg[va].push_back(vb);
                    lg[vb].push_back(va);
                }
            }
        }
    }

    std::vector<std::vector<int>> out(nv);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        std::size_t total = 0;
        for (int t = 0; t < T; ++t) total += local_nbr[t][vi].size();
        out[vi].reserve(total);
        for (int t = 0; t < T; ++t) {
            out[vi].insert(out[vi].end(), local_nbr[t][vi].begin(), local_nbr[t][vi].end());
        }
        std::sort(out[vi].begin(), out[vi].end());
        out[vi].erase(std::unique(out[vi].begin(), out[vi].end()), out[vi].end());
    }
    return out;
}

std::vector<std::vector<int>> union_adjacency(const std::vector<std::vector<int>>& a,
                                              const std::vector<std::vector<int>>& b){
    const int nv = static_cast<int>(std::max(a.size(), b.size()));
    std::vector<std::vector<int>> out(nv);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        const auto* pa = (vi < static_cast<int>(a.size())) ? &a[vi] : nullptr;
        const auto* pb = (vi < static_cast<int>(b.size())) ? &b[vi] : nullptr;
        if (!pa || pa->empty()) { if (pb) out[vi] = *pb; continue; }
        if (!pb || pb->empty()) { out[vi] = *pa; continue; }
        out[vi].reserve(pa->size() + pb->size());
        std::set_union(pa->begin(), pa->end(), pb->begin(), pb->end(), std::back_inserter(out[vi]));
    }
    return out;
}

std::vector<std::vector<int>> build_conflict_graph(const RefMesh& ref_mesh, const std::vector<Pin>& /*pins*/,
                                                   const BroadPhase::Cache& bp_cache, const std::vector<JacobiPrediction>& predictions,
                                                   const VertexTriangleMap* adj,
                                                   const std::vector<std::vector<int>>* base_adj,
                                                   SweptBvhCache* sw_cache){
    const int nv = static_cast<int>(predictions.size());
    std::vector<std::vector<int>> graph(nv);

    VertexTriangleMap local_adj;
    if (!adj && !base_adj) {
        local_adj = build_incident_triangle_map(ref_mesh.tris);
        adj = &local_adj;
    }

    // Stage active AABBs into cache storage (if provided) so refit_bvh can
    // reuse the node layout; otherwise use function-local buffers.
    std::vector<AABB>  local_active_boxes;
    std::vector<int>   local_active_ids;
    std::vector<AABB>& active_boxes = sw_cache ? sw_cache->active_boxes : local_active_boxes;
    std::vector<int>&  active_ids   = sw_cache ? sw_cache->active_ids   : local_active_ids;
    const int prev_n_active = static_cast<int>(active_ids.size());
    const int prev_first_id = prev_n_active > 0 ? active_ids.front() : -1;
    const int prev_last_id  = prev_n_active > 0 ? active_ids.back()  : -1;

    active_boxes.clear();
    active_ids.clear();
    active_boxes.reserve(nv);
    active_ids.reserve(nv);
    for (int i = 0; i < nv; ++i) {
        if (!predictions[i].active) continue;
        active_ids.push_back(i);
        active_boxes.push_back(predictions[i].certified_region);
    }

    std::vector<BVHNode>  local_sw_bvh_nodes;
    std::vector<BVHNode>& sw_bvh_nodes = sw_cache ? sw_cache->nodes : local_sw_bvh_nodes;
    int sw_root = -1;

    // refit_bvh is only safe if leaf->primitive mapping is unchanged. Cheap
    // guard: same size and same first/last ids (ids are produced by a
    // deterministic scan over predictions).
    const int n_active_now = static_cast<int>(active_ids.size());
    const bool cache_topology_matches =
        sw_cache && sw_cache->valid && !sw_bvh_nodes.empty() &&
        prev_n_active == n_active_now &&
        (n_active_now == 0 ||
         (active_ids.front() == prev_first_id && active_ids.back() == prev_last_id));
    if (cache_topology_matches) {
        refit_bvh(sw_bvh_nodes, active_boxes);
        sw_root = sw_cache->root;
    } else {
        sw_bvh_nodes.clear();
        sw_root = build_bvh(active_boxes, sw_bvh_nodes);
        if (sw_cache) {
            sw_cache->root  = sw_root;
            sw_cache->valid = true;
        }
    }

    const int T = ph_max_threads();
    std::vector<std::vector<std::vector<int>>> local_nbr(T, std::vector<std::vector<int>>(nv));

    const int n_nt     = static_cast<int>(bp_cache.nt_pairs.size());
    const int n_ss     = static_cast<int>(bp_cache.ss_pairs.size());
    const int n_active = static_cast<int>(active_ids.size());
    const bool have_base = (base_adj != nullptr);

    // Elastic + barrier-pair edges (skipped when caller supplies base_adj).
    if (!have_base) {
        if (adj) {
            #pragma omp parallel
            {
                auto& lg = local_nbr[ph_thread_num()];
                #pragma omp for schedule(static)
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
        }

        #pragma omp parallel
        {
            auto& lg = local_nbr[ph_thread_num()];
            #pragma omp for schedule(static)
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
        }

        #pragma omp parallel
        {
            auto& lg = local_nbr[ph_thread_num()];
            #pragma omp for schedule(static)
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
        }
    }

    // Swept-region self-query: pairs whose certified-region AABBs overlap.
    #pragma omp parallel
    {
        auto& lg = local_nbr[ph_thread_num()];
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

    // Merge per-thread locals, seed with base_adj if provided, then dedup.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        std::size_t base_sz = (have_base && vi < static_cast<int>(base_adj->size())) ? (*base_adj)[vi].size() : 0;
        std::size_t total = base_sz;
        for (int t = 0; t < T; ++t) total += local_nbr[t][vi].size();
        graph[vi].reserve(total);
        if (have_base && base_sz) {
            graph[vi].insert(graph[vi].end(), (*base_adj)[vi].begin(), (*base_adj)[vi].end());
        }
        for (int t = 0; t < T; ++t) {
            graph[vi].insert(graph[vi].end(), local_nbr[t][vi].begin(), local_nbr[t][vi].end());
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

double clip_step_to_certified_region(int vi, const std::vector<Vec3>& x, const Vec3& fresh_delta, const AABB& certified_region){
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
                                    const BroadPhase::Cache& bp_cache){
    if (params.d_hat <= 0.0) return 1.0;

    const Vec3 dx = -delta;
    const bool tr = params.use_trust_region;

    double safe_min = 1.0;

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_nt.size())) {
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            // dof: 0=node, 1=tri_v[0], 2=tri_v[1], 3=tri_v[2]
            if (entry.dof == 0) {
                // vi is the lone moving node vs a static triangle.
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
            } else {
                // vi is a moving tri corner vs a static node.
                if (tr) {
                    auto r = trust_region_vertex_triangle_gauss_seidel(
                        x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                    if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
                } else {
                    Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                    dxv[entry.dof - 1] = dx;
                    CCDResult r = node_triangle_only_one_node_moves(
                        x[p.node],     Vec3::Zero(),
                        x[p.tri_v[0]], dxv[0],
                        x[p.tri_v[1]], dxv[1],
                        x[p.tri_v[2]], dxv[2]);
                    if (r.collision) safe_min = std::min(safe_min, r.t);
                }
            }
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_ss.size())) {
        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            // dof: 0=v[0], 1=v[1], 2=v[2], 3=v[3]. vi is on the first edge iff dof in {0,1}.
            if (tr) {
                auto r = trust_region_edge_edge_gauss_seidel(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], dx);
                if (r.d0 < params.d_hat) safe_min = std::min(safe_min, r.omega);
            } else {
                CCDResult r;
                if (entry.dof == 0)
                    r = segment_segment_only_one_node_moves(x[p.v[0]], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]]);
                else if (entry.dof == 1)
                    r = segment_segment_only_one_node_moves(x[p.v[1]], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]]);
                else if (entry.dof == 2)
                    r = segment_segment_only_one_node_moves(x[p.v[2]], dx, x[p.v[3]], x[p.v[0]], x[p.v[1]]);
                else
                    r = segment_segment_only_one_node_moves(x[p.v[3]], dx, x[p.v[2]], x[p.v[0]], x[p.v[1]]);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }
    }

    return tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
}

BroadPhase::Cache register_barrier_pairs_from_blue_green(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges,
                                                         const std::vector<AABB>& blue_boxes, double d_hat){
    const int nv = static_cast<int>(blue_boxes.size());
    const int nt = num_tris(ref_mesh);
    const int ne = static_cast<int>(edges.size());

    BroadPhase::Cache cache;
    cache.edges = edges;
    cache.vertex_nt.assign(nv, {});
    cache.vertex_ss.assign(nv, {});

    if (nv == 0) return cache;

    // Green triangle boxes: union of blue corners, then pad by d_hat on each face.
    std::vector<AABB> green_tri(nt);
    for (int t = 0; t < nt; ++t) {
        const int a = tri_vertex(ref_mesh, t, 0);
        const int b = tri_vertex(ref_mesh, t, 1);
        const int c = tri_vertex(ref_mesh, t, 2);
        green_tri[t] = blue_boxes[a];
        green_tri[t].expand(blue_boxes[b]);
        green_tri[t].expand(blue_boxes[c]);
        green_tri[t].min.array() -= d_hat;
        green_tri[t].max.array() += d_hat;
    }

    // Green edge boxes: union of blue endpoints + d_hat pad.
    std::vector<AABB> green_edge(ne);
    for (int e = 0; e < ne; ++e) {
        green_edge[e] = blue_boxes[edges[e][0]];
        green_edge[e].expand(blue_boxes[edges[e][1]]);
        green_edge[e].min.array() -= d_hat;
        green_edge[e].max.array() += d_hat;
    }

    std::vector<BVHNode> tri_bvh;
    const int tri_root = build_bvh(green_tri, tri_bvh);
    std::vector<BVHNode> edge_bvh;
    const int edge_root = build_bvh(green_edge, edge_bvh);

    // Node-triangle pairs: register (n, t) when B(n) and G(t) have intersections and n is not a corner of t.
    std::vector<std::vector<int>> nt_hits(nv);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int vi = 0; vi < nv; ++vi) {
        if (tri_root < 0) continue;
        query_bvh(tri_bvh, tri_root, blue_boxes[vi], nt_hits[vi]);
    }

    for (int vi = 0; vi < nv; ++vi) {
        for (int t : nt_hits[vi]) {
            const int a = tri_vertex(ref_mesh, t, 0);
            const int b = tri_vertex(ref_mesh, t, 1);
            const int c = tri_vertex(ref_mesh, t, 2);
            if (vi == a || vi == b || vi == c) continue;
            const std::size_t idx = cache.nt_pairs.size();
            cache.nt_pairs.push_back(NodeTrianglePair{vi, {a, b, c}});
            cache.vertex_nt[vi].push_back({idx, 0});
            cache.vertex_nt[a].push_back({idx, 1});
            cache.vertex_nt[b].push_back({idx, 2});
            cache.vertex_nt[c].push_back({idx, 3});
        }
    }

    // Segment-segment pairs: register (e, f) when G(e) and G(f) have intersections and they share no vertex.
    std::vector<std::vector<int>> ss_hits(ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (edge_root < 0) continue;
        query_bvh(edge_bvh, edge_root, green_edge[e], ss_hits[e]);
    }
    for (int e = 0; e < ne; ++e) {
        const int ea = edges[e][0];
        const int eb = edges[e][1];
        for (int other : ss_hits[e]) {
            if (other <= e) continue;
            const int oa = edges[other][0];
            const int ob = edges[other][1];
            if (ea == oa || ea == ob || eb == oa || eb == ob) continue;
            const std::size_t idx = cache.ss_pairs.size();
            cache.ss_pairs.push_back(SegmentSegmentPair{{ea, eb, oa, ob}});
            cache.vertex_ss[ea].push_back({idx, 0});
            cache.vertex_ss[eb].push_back({idx, 1});
            cache.vertex_ss[oa].push_back({idx, 2});
            cache.vertex_ss[ob].push_back({idx, 3});
        }
    }

    return cache;
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

    // The conflict graph ensures no new barrier pair can arise within this
    // batch, so single-node CCD against the current barrier pairs suffices.
    out.ccd_step = compute_safe_step_for_vertex(vi, ref_mesh, params, x_current, delta, bp_cache);

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

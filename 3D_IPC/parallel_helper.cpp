#include "parallel_helper.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

// Certified region = segment (x[vi] → x[vi] - delta) ∪ incident tris/edges,
// inflated by d_hat. While vi stays inside this box its incident primitives
// stay d_hat-separated from any non-overlapping other vertex's region, so the
// two vertices can be updated in parallel.
static AABB build_certified_region_for_vertex(int vi, const std::vector<Vec3>& x, const Vec3& delta, const BroadPhase::Cache& bp_cache, double d_hat){
    AABB box;

    box.expand(x[vi]);
    box.expand(x[vi] - delta);

    if (vi >= 0 && vi < static_cast<int>(bp_cache.node_to_tris.size())) {
        for (int tri_idx : bp_cache.node_to_tris[vi]) {
            box.expand(bp_cache.tri_boxes[tri_idx]);
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.node_to_edges.size())) {
        for (int edge_idx : bp_cache.node_to_edges[vi]) {
            box.expand(bp_cache.edge_boxes[edge_idx]);
        }
    }

    box.min.array() -= d_hat;
    box.max.array() += d_hat;
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

static void add_conflict_edge(int a, int b, std::vector<std::vector<int>>& graph){
    if (a == b) return;
    graph[a].push_back(b);
    graph[b].push_back(a);
}

std::vector<std::vector<int>> build_conflict_graph(const RefMesh& ref_mesh, const std::vector<Pin>& /*pins*/,
                                                   const BroadPhase::Cache& bp_cache, const std::vector<JacobiPrediction>& predictions,
                                                   const VertexTriangleMap* adj){
    const int nv = static_cast<int>(predictions.size());
    std::vector<std::vector<int>> graph(nv);

    // Elastic coupling: two vertices that share a triangle.
    if (adj) {
        for (const auto& [vi, tri_list] : *adj) {
            if (vi < 0 || vi >= nv || !predictions[vi].active) continue;
            for (const auto& [ti, local_a] : tri_list) {
                for (int local_b = 0; local_b < 3; ++local_b) {
                    int vj = tri_vertex(ref_mesh, ti, local_b);
                    if (vj == vi || vj < 0 || vj >= nv || !predictions[vj].active) continue;
                    add_conflict_edge(vi, vj, graph);
                }
            }
        }
    } else {
        const auto elastic_adj = ::build_vertex_adjacency_map(ref_mesh.tris);
        for (const auto& kv : elastic_adj) {
            const int vi = kv.first;
            const auto& nbrs = kv.second;
            if (vi < 0 || vi >= nv) continue;
            if (!predictions[vi].active) continue;
            for (int vj : nbrs) {
                if (vj < 0 || vj >= nv) continue;
                if (!predictions[vj].active) continue;
                add_conflict_edge(vi, vj, graph);
            }
        }
    }

    // Barrier coupling: any two vertices that appear in the same contact pair.
    for (const auto& p : bp_cache.nt_pairs) {
        const int verts[4] = { p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2] };
        for (int a = 0; a < 4; ++a) {
            if (verts[a] < 0 || verts[a] >= nv || !predictions[verts[a]].active) continue;
            for (int b = a + 1; b < 4; ++b) {
                if (verts[b] < 0 || verts[b] >= nv || !predictions[verts[b]].active) continue;
                add_conflict_edge(verts[a], verts[b], graph);
            }
        }
    }

    for (const auto& p : bp_cache.ss_pairs) {
        for (int a = 0; a < 4; ++a) {
            const int va = p.v[a];
            if (va < 0 || va >= nv || !predictions[va].active) continue;
            for (int b = a + 1; b < 4; ++b) {
                const int vb = p.v[b];
                if (vb < 0 || vb >= nv || !predictions[vb].active) continue;
                add_conflict_edge(va, vb, graph);
            }
        }
    }

    // Swept-region overlap: BVH query of each certified region against the rest.
    {
        std::vector<AABB> active_boxes;
        std::vector<int> active_ids;
        active_boxes.reserve(nv);
        active_ids.reserve(nv);
        for (int i = 0; i < nv; ++i) {
            if (!predictions[i].active) continue;
            active_ids.push_back(i);
            active_boxes.push_back(predictions[i].certified_region);
        }

        std::vector<BVHNode> bvh_nodes;
        int root = build_bvh(active_boxes, bvh_nodes);

        std::vector<int> hits;
        for (int ai = 0; ai < static_cast<int>(active_ids.size()); ++ai) {
            hits.clear();
            query_bvh(bvh_nodes, root, active_boxes[ai], hits);
            const int vi = active_ids[ai];
            for (int leaf : hits) {
                const int vj = active_ids[leaf];
                if (vj > vi) add_conflict_edge(vi, vj, graph);
            }
        }
    }

    for (auto& nbrs : graph) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
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

static double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params, const std::vector<Vec3>& x, const Vec3& delta,
                                           const BroadPhase& broad_phase){
    if (params.d_hat <= 0.0) return 1.0;

    const Vec3 dx = -delta;

    const auto ccd = broad_phase.query_single_node_ccd(x, vi, dx, ref_mesh);

    double toi_min = 1.0;

    // vi is the lone moving node against static triangles.
    for (const auto& p : ccd.nt_node_pairs) {
        CCDResult r = node_triangle_only_one_node_moves(
            x[p.node],     dx,
            x[p.tri_v[0]], Vec3::Zero(),
            x[p.tri_v[1]], Vec3::Zero(),
            x[p.tri_v[2]], Vec3::Zero());
        if (r.collision) toi_min = std::min(toi_min, r.t);
    }

    // vi is one corner of a moving triangle against a static node.
    for (const auto& p : ccd.nt_face_pairs) {
        Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
        dxv[p.vi_local] = dx;
        CCDResult r = node_triangle_only_one_node_moves(
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

    return (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;
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

    // Correctness: the conflict graph ensures no new barrier pair can arise
    // within this batch, so single-node CCD against the current B̃ suffices.
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

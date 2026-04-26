#include "broad_phase.h"

#include <map>
#include <set>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

// BVH build / refit / query
int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    out.clear();
    if (boxes.empty()) return -1;

    std::vector<int> idx(boxes.size());
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) idx[i] = i;

    struct BuildTask {
        int node_idx;
        int start;
        int end;
    };

    out.emplace_back();
    std::vector<BuildTask> stack;
    stack.push_back({0, 0, static_cast<int>(idx.size())});

    while (!stack.empty()) {
        const BuildTask task = stack.back();
        stack.pop_back();

        AABB node_box;
        for (int i = task.start; i < task.end; ++i) {
            node_box.expand(boxes[idx[i]]);
        }
        out[task.node_idx].bbox = node_box;

        const int count = task.end - task.start;
        if (count == 1) {
            out[task.node_idx].leafIndex = idx[task.start];
            continue;
        }

        const Vec3 e = node_box.extent();
        int axis = 0;
        if (e.y() > e.x() && e.y() >= e.z()) axis = 1;
        else if (e.z() > e.x() && e.z() >= e.y()) axis = 2;

        const int mid = task.start + count / 2;
        std::nth_element(
                idx.begin() + task.start,
                idx.begin() + mid,
                idx.begin() + task.end,
                [&](int a, int b) { return boxes[a].centroid()[axis] < boxes[b].centroid()[axis]; });

        const int left = static_cast<int>(out.size());
        out.emplace_back();
        const int right = static_cast<int>(out.size());
        out.emplace_back();

        out[task.node_idx].left = left;
        out[task.node_idx].right = right;

        stack.push_back({right, mid, task.end});
        stack.push_back({left, task.start, mid});
    }

    return 0;
}

void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes) {
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        BVHNode& n = nodes[i];
        if (n.leafIndex >= 0) {
            n.bbox = boxes[n.leafIndex];
        } else {
            n.bbox = AABB();
            n.bbox.expand(nodes[n.left].bbox);
            n.bbox.expand(nodes[n.right].bbox);
        }
    }
}

void query_bvh(const std::vector<BVHNode>& nodes, int root, const AABB& query, std::vector<int>& hits) {
    if (root < 0) return;

    int stack[256];
    int top = 0;
    stack[top++] = root;

    while (top > 0) {
        const BVHNode& n = nodes[stack[--top]];
        if (!aabb_intersects(n.bbox, query)) continue;

        if (n.leafIndex >= 0) {
            hits.push_back(n.leafIndex);
        } else {
            stack[top++] = n.left;
            stack[top++] = n.right;
        }
    }
}

// Local helpers
namespace {

    struct Edge {
        int v0 = -1;
        int v1 = -1;
    };

    static inline Edge canonical_edge(int a, int b) {
        if (a > b) std::swap(a, b);
        return {a, b};
    }

    static void build_unique_edges_and_adjacency(const RefMesh& mesh, int nv, std::vector<std::array<int, 2>>& out_edges,
                                                 std::vector<std::vector<int>>& out_node_to_edges, std::vector<std::vector<int>>& out_node_to_tris) {
        out_edges.clear();
        out_node_to_edges.assign(nv, {});
        out_node_to_tris.assign(nv, {});

        std::map<std::pair<int, int>, int> edge_to_idx;

        const int nt = num_tris(mesh);
        for (int t = 0; t < nt; ++t) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);

            out_node_to_tris[a].push_back(t);
            out_node_to_tris[b].push_back(t);
            out_node_to_tris[c].push_back(t);

            const Edge e0 = canonical_edge(a, b);
            const Edge e1 = canonical_edge(b, c);
            const Edge e2 = canonical_edge(c, a);

            const std::array<Edge, 3> tri_edges = {e0, e1, e2};
            for (const Edge& e : tri_edges) {
                const std::pair<int, int> key(e.v0, e.v1);
                auto it = edge_to_idx.find(key);
                int edge_idx = -1;
                if (it == edge_to_idx.end()) {
                    edge_idx = static_cast<int>(out_edges.size());
                    edge_to_idx.emplace(key, edge_idx);
                    out_edges.push_back({e.v0, e.v1});
                } else {
                    edge_idx = it->second;
                }

                out_node_to_edges[e.v0].push_back(edge_idx);
                out_node_to_edges[e.v1].push_back(edge_idx);
            }
        }

        for (int i = 0; i < nv; ++i) {
            auto& tris = out_node_to_tris[i];
            std::sort(tris.begin(), tris.end());
            tris.erase(std::unique(tris.begin(), tris.end()), tris.end());

            auto& edges = out_node_to_edges[i];
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        }
    }

    static inline bool share_vertex(const Edge& e0, const Edge& e1) {
        return e0.v0 == e1.v0 || e0.v0 == e1.v1 ||
               e0.v1 == e1.v0 || e0.v1 == e1.v1;
    }

    static inline bool node_in_triangle(int node, int a, int b, int c) {
        return node == a || node == b || node == c;
    }

    static inline AABB build_node_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int node, double dt, double pad) {
        AABB box;
        const Vec3 x0 = x[node];
        const Vec3 x1 = x[node] + dt * v[node];

        box.expand(x0);
        box.expand(x1);

        box.min.array() -= pad;
        box.max.array() += pad;

        return box;
    }

    static inline AABB build_triangle_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, int c, double dt, double pad) {
        AABB box;

        box.expand(x[a]);
        box.expand(x[b]);
        box.expand(x[c]);

        box.expand(x[a] + dt * v[a]);
        box.expand(x[b] + dt * v[b]);
        box.expand(x[c] + dt * v[c]);

        box.min.array() -= pad;
        box.max.array() += pad;

        return box;
    }

    static inline AABB build_edge_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, double dt, double pad) {
        AABB box;

        box.expand(x[a]);
        box.expand(x[b]);

        box.expand(x[a] + dt * v[a]);
        box.expand(x[b] + dt * v[b]);

        box.min.array() -= pad;
        box.max.array() += pad;

        return box;
    }

    using VPE = BroadPhase::Cache::VertexPairEntry;

    static inline void remove_vertex_entry(std::vector<VPE>& vec, std::size_t pair_idx) {
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (vec[i].pair_index == pair_idx) {
                vec[i] = vec.back();
                vec.pop_back();
                return;
            }
        }
    }

    static inline void update_vertex_entry(std::vector<VPE>& vec, std::size_t old_idx, std::size_t new_idx) {
        for (auto& e : vec) {
            if (e.pair_index == old_idx) {
                e.pair_index = new_idx;
                return;
            }
        }
    }

    static inline void add_nt_pair(BroadPhase::Cache& cache, int node, int tri_idx, const RefMesh& mesh) {
        const std::uint64_t key = BroadPhase::nt_key(node, tri_idx);
        if (cache.nt_pair_index.count(key)) return;

        NodeTrianglePair p;
        p.node = node;
        p.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
        p.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
        p.tri_v[2] = tri_vertex(mesh, tri_idx, 2);

        const std::size_t idx = cache.nt_pairs.size();
        cache.nt_pair_index[key] = idx;
        cache.nt_pairs.push_back(p);
        cache.nt_pair_tri.push_back(tri_idx);

        if (!cache.vertex_nt.empty()) {
            cache.vertex_nt[p.node].push_back({idx, 0});
            cache.vertex_nt[p.tri_v[0]].push_back({idx, 1});
            cache.vertex_nt[p.tri_v[1]].push_back({idx, 2});
            cache.vertex_nt[p.tri_v[2]].push_back({idx, 3});
        }
    }

    static inline void add_ss_pair(BroadPhase::Cache& cache, int e0, int e1) {
        int a = e0;
        int b = e1;
        if (a > b) std::swap(a, b);

        const std::uint64_t key = BroadPhase::ss_key(a, b);
        if (cache.ss_pair_index.count(key)) return;

        SegmentSegmentPair p;
        p.v[0] = cache.edges[a][0];
        p.v[1] = cache.edges[a][1];
        p.v[2] = cache.edges[b][0];
        p.v[3] = cache.edges[b][1];

        const std::size_t idx = cache.ss_pairs.size();
        cache.ss_pair_index[key] = idx;
        cache.ss_pairs.push_back(p);
        cache.ss_pair_edges.push_back({a, b});

        if (!cache.vertex_ss.empty()) {
            cache.vertex_ss[p.v[0]].push_back({idx, 0});
            cache.vertex_ss[p.v[1]].push_back({idx, 1});
            cache.vertex_ss[p.v[2]].push_back({idx, 2});
            cache.vertex_ss[p.v[3]].push_back({idx, 3});
        }
    }

    static inline void erase_nt_pair_at(BroadPhase::Cache& cache, std::size_t idx) {
        const std::size_t last = cache.nt_pairs.size() - 1;
        const auto& victim = cache.nt_pairs[idx];
        const int victim_node = victim.node;
        const int victim_tri = cache.nt_pair_tri[idx];

        if (!cache.vertex_nt.empty()) {
            remove_vertex_entry(cache.vertex_nt[victim.node], idx);
            remove_vertex_entry(cache.vertex_nt[victim.tri_v[0]], idx);
            remove_vertex_entry(cache.vertex_nt[victim.tri_v[1]], idx);
            remove_vertex_entry(cache.vertex_nt[victim.tri_v[2]], idx);

            if (idx != last) {
                const auto& moved = cache.nt_pairs[last];
                update_vertex_entry(cache.vertex_nt[moved.node], last, idx);
                update_vertex_entry(cache.vertex_nt[moved.tri_v[0]], last, idx);
                update_vertex_entry(cache.vertex_nt[moved.tri_v[1]], last, idx);
                update_vertex_entry(cache.vertex_nt[moved.tri_v[2]], last, idx);
            }
        }

        if (idx != last) {
            cache.nt_pairs[idx] = cache.nt_pairs[last];
            cache.nt_pair_tri[idx] = cache.nt_pair_tri[last];

            const int moved_node = cache.nt_pairs[idx].node;
            const int moved_tri = cache.nt_pair_tri[idx];
            cache.nt_pair_index[BroadPhase::nt_key(moved_node, moved_tri)] = idx;
        }

        cache.nt_pairs.pop_back();
        cache.nt_pair_tri.pop_back();
        cache.nt_pair_index.erase(BroadPhase::nt_key(victim_node, victim_tri));
    }

    static inline void erase_ss_pair_at(BroadPhase::Cache& cache, std::size_t idx) {
        const std::size_t last = cache.ss_pairs.size() - 1;
        const auto& victim = cache.ss_pairs[idx];
        const std::array<int, 2> victim_edges = cache.ss_pair_edges[idx];

        if (!cache.vertex_ss.empty()) {
            remove_vertex_entry(cache.vertex_ss[victim.v[0]], idx);
            remove_vertex_entry(cache.vertex_ss[victim.v[1]], idx);
            remove_vertex_entry(cache.vertex_ss[victim.v[2]], idx);
            remove_vertex_entry(cache.vertex_ss[victim.v[3]], idx);

            if (idx != last) {
                const auto& moved = cache.ss_pairs[last];
                update_vertex_entry(cache.vertex_ss[moved.v[0]], last, idx);
                update_vertex_entry(cache.vertex_ss[moved.v[1]], last, idx);
                update_vertex_entry(cache.vertex_ss[moved.v[2]], last, idx);
                update_vertex_entry(cache.vertex_ss[moved.v[3]], last, idx);
            }
        }

        if (idx != last) {
            cache.ss_pairs[idx] = cache.ss_pairs[last];
            cache.ss_pair_edges[idx] = cache.ss_pair_edges[last];

            const std::array<int, 2> moved_edges = cache.ss_pair_edges[idx];
            cache.ss_pair_index[BroadPhase::ss_key(moved_edges[0], moved_edges[1])] = idx;
        }

        cache.ss_pairs.pop_back();
        cache.ss_pair_edges.pop_back();
        cache.ss_pair_index.erase(BroadPhase::ss_key(victim_edges[0], victim_edges[1]));
    }

}

// Broad phase
void BroadPhase::set_mesh_topology(const RefMesh& mesh, int nv) {
    build_unique_edges_and_adjacency(mesh, nv, topo_.edges, topo_.node_to_edges, topo_.node_to_tris);
    topology_valid_ = true;
}

void BroadPhase::build(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad) {
    Cache c;

    const int nv = static_cast<int>(x.size());
    const int nt = num_tris(mesh);

    c.vertex_nt.resize(nv);
    c.vertex_ss.resize(nv);

    if (!topology_valid_) set_mesh_topology(mesh, nv);
    c.edges = topo_.edges;
    c.node_to_edges = topo_.node_to_edges;
    c.node_to_tris = topo_.node_to_tris;
    const int ne = static_cast<int>(c.edges.size());

    c.node_boxes.resize(nv);
    for (int i = 0; i < nv; ++i) {
        c.node_boxes[i] = build_node_box(x, v, i, dt, node_pad);
    }

    c.tri_boxes.resize(nt);
    for (int t = 0; t < nt; ++t) {
        const int a = tri_vertex(mesh, t, 0);
        const int b = tri_vertex(mesh, t, 1);
        const int cc = tri_vertex(mesh, t, 2);
        c.tri_boxes[t] = build_triangle_box(x, v, a, b, cc, dt, tri_pad);
    }

    c.edge_boxes.resize(ne);
    for (int e = 0; e < ne; ++e) {
        c.edge_boxes[e] = build_edge_box(x, v, c.edges[e][0], c.edges[e][1], dt, edge_pad);
    }

    c.tri_root = build_bvh(c.tri_boxes, c.tri_bvh_nodes);
    c.edge_root = build_bvh(c.edge_boxes, c.edge_bvh_nodes);
    c.node_root = build_bvh(c.node_boxes, c.node_bvh_nodes);

    // Parallel BVH queries, serial pair-insert (add_*_pair mutates shared state).
    std::vector<std::vector<int>> node_hits(nv);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int node = 0; node < nv; ++node) {
        if (c.tri_root < 0) continue;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
    }
    for (int node = 0; node < nv; ++node) {
        for (int t : node_hits[node]) {
            const int a  = tri_vertex(mesh, t, 0);
            const int b  = tri_vertex(mesh, t, 1);
            const int cc = tri_vertex(mesh, t, 2);
            if (node_in_triangle(node, a, b, cc)) continue;
            add_nt_pair(c, node, t, mesh);
        }
    }

    std::vector<std::vector<int>> edge_hits(ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (c.edge_root < 0) continue;
        query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], edge_hits[e]);
    }
    for (int e = 0; e < ne; ++e) {
        const Edge e0{c.edges[e][0], c.edges[e][1]};
        for (int other : edge_hits[e]) {
            if (other == e) continue;
            const Edge e1{c.edges[other][0], c.edges[other][1]};
            if (share_vertex(e0, e1)) continue;
            add_ss_pair(c, e, other);
        }
    }

    cache_ = std::move(c);
}

void BroadPhase::initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat) {
    build(x, v, mesh, dt, /*node_pad=*/dhat, /*tri_pad=*/0.0, /*edge_pad=*/dhat * 0.5);
    ++version_;
}

void BroadPhase::initialize(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, double d_hat) {
    const int nv = static_cast<int>(vertex_boxes.size());
    const int nt = num_tris(mesh);

    Cache c;
    c.vertex_nt.resize(nv);
    c.vertex_ss.resize(nv);

    if (!topology_valid_) set_mesh_topology(mesh, nv);
    c.edges = topo_.edges;
    c.node_to_edges = topo_.node_to_edges;
    c.node_to_tris = topo_.node_to_tris;
    const int ne = static_cast<int>(c.edges.size());

    c.node_boxes = vertex_boxes;

    const Vec3 pad = d_hat * Vec3::Ones();
    c.tri_boxes.resize(nt);
    for (int t = 0; t < nt; ++t) {
        const int a  = tri_vertex(mesh, t, 0);
        const int b  = tri_vertex(mesh, t, 1);
        const int cc = tri_vertex(mesh, t, 2);
        c.tri_boxes[t] = vertex_boxes[a];
        c.tri_boxes[t].expand(vertex_boxes[b]);
        c.tri_boxes[t].expand(vertex_boxes[cc]);
        c.tri_boxes[t].min -= pad;
        c.tri_boxes[t].max += pad;
    }

    c.edge_boxes.resize(ne);
    for (int e = 0; e < ne; ++e) {
        c.edge_boxes[e] = vertex_boxes[c.edges[e][0]];
        c.edge_boxes[e].expand(vertex_boxes[c.edges[e][1]]);
        c.edge_boxes[e].min -= pad;
        c.edge_boxes[e].max += pad;
    }

    c.tri_root  = build_bvh(c.tri_boxes,  c.tri_bvh_nodes);
    c.edge_root = build_bvh(c.edge_boxes, c.edge_bvh_nodes);
    c.node_root = build_bvh(c.node_boxes, c.node_bvh_nodes);

    std::vector<std::vector<int>> node_hits(nv);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int node = 0; node < nv; ++node) {
        if (c.tri_root < 0) continue;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
    }
    for (int node = 0; node < nv; ++node) {
        for (int t : node_hits[node]) {
            const int a  = tri_vertex(mesh, t, 0);
            const int b  = tri_vertex(mesh, t, 1);
            const int cc = tri_vertex(mesh, t, 2);
            if (node_in_triangle(node, a, b, cc)) continue;
            add_nt_pair(c, node, t, mesh);
        }
    }

    std::vector<std::vector<int>> edge_hits(ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (c.edge_root < 0) continue;
        query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], edge_hits[e]);
    }
    for (int e = 0; e < ne; ++e) {
        const Edge e0{c.edges[e][0], c.edges[e][1]};
        for (int other : edge_hits[e]) {
            if (other == e) continue;
            const Edge e1{c.edges[other][0], c.edges[other][1]};
            if (share_vertex(e0, e1)) continue;
            add_ss_pair(c, e, other);
        }
    }

    cache_ = std::move(c);
    ++version_;
}

double BroadPhase::ccd_min_toi(const std::vector<Vec3>& x, const std::vector<Vec3>& x_new) const {
    const int nv = static_cast<int>(x.size());
    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = x_new[i] - x[i];

    double toi_min = 1.0;

    for (const auto& p : cache_.nt_pairs) {
        toi_min = std::min(toi_min, node_triangle_general_ccd(
            x[p.node],     dx[p.node],
            x[p.tri_v[0]], dx[p.tri_v[0]],
            x[p.tri_v[1]], dx[p.tri_v[1]],
            x[p.tri_v[2]], dx[p.tri_v[2]]));
    }

    for (const auto& p : cache_.ss_pairs) {
        toi_min = std::min(toi_min, segment_segment_general_ccd(
            x[p.v[0]], dx[p.v[0]],
            x[p.v[1]], dx[p.v[1]],
            x[p.v[2]], dx[p.v[2]],
            x[p.v[3]], dx[p.v[3]]));
    }

    return toi_min;
}

void BroadPhase::per_vertex_safe_step(
        std::vector<Vec3>& x, const std::function<Vec3(int)>& x_new_fn, double safety, bool clip_to_node_box) const {
    const int nv = static_cast<int>(x.size());

    for (int vi = 0; vi < nv; ++vi) {
        if (clip_to_node_box) {
            const AABB& box = cache_.node_boxes[vi];
            if ((x[vi].array() < box.min.array()).any() || (x[vi].array() > box.max.array()).any()) {
                fprintf(stderr, "per_vertex_safe_step: x[%d] = (%.6f, %.6f, %.6f) is outside box [(%.6f,%.6f,%.6f),(%.6f,%.6f,%.6f)]\n",
                    vi,
                    x[vi].x(), x[vi].y(), x[vi].z(),
                    box.min.x(), box.min.y(), box.min.z(),
                    box.max.x(), box.max.y(), box.max.z());
                exit(1);
            }
        }
        const Vec3 x_new = clip_to_node_box
            ? [&]{ constexpr double inset = 1e-10;
                   const AABB& box = cache_.node_boxes[vi];
                   return x_new_fn(vi).cwiseMax(box.min + Vec3::Constant(inset))
                                      .cwiseMin(box.max - Vec3::Constant(inset)); }()
            : x_new_fn(vi);
        const Vec3 dx = x_new - x[vi];
        if (dx.squaredNorm() < 1e-28) continue;

        double toi_min = 1.0;

        for (const auto& entry : cache_.vertex_nt[vi]) {
            const auto& p = cache_.nt_pairs[entry.pair_index];
            CCDResult r;
            if (entry.dof == 0) {
                // vi is the lone node; triangle is stationary
                r = node_triangle_only_one_node_moves(
                    x[vi],       dx,
                    x[p.tri_v[0]], Vec3::Zero(),
                    x[p.tri_v[1]], Vec3::Zero(),
                    x[p.tri_v[2]], Vec3::Zero());
            } else {
                // vi is a triangle corner; external node is stationary
                Vec3 d0 = Vec3::Zero(), d1 = Vec3::Zero(), d2 = Vec3::Zero();
                if      (entry.dof == 1) d0 = dx;
                else if (entry.dof == 2) d1 = dx;
                else                     d2 = dx;
                r = node_triangle_only_one_node_moves(
                    x[p.node],     Vec3::Zero(),
                    x[p.tri_v[0]], d0,
                    x[p.tri_v[1]], d1,
                    x[p.tri_v[2]], d2);
            }
            if (r.collision) toi_min = std::min(toi_min, r.t);
        }

        for (const auto& entry : cache_.vertex_ss[vi]) {
            const auto& p = cache_.ss_pairs[entry.pair_index];
            // always put vi in the x1 (moving) slot; swap edges if vi is on edge 2
            CCDResult r;
            if (entry.dof == 0)
                r = segment_segment_only_one_node_moves(
                    x[vi], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]]);
            else if (entry.dof == 1)
                r = segment_segment_only_one_node_moves(
                    x[vi], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]]);
            else if (entry.dof == 2)
                r = segment_segment_only_one_node_moves(
                    x[vi], dx, x[p.v[3]], x[p.v[0]], x[p.v[1]]);
            else
                r = segment_segment_only_one_node_moves(
                    x[vi], dx, x[p.v[2]], x[p.v[0]], x[p.v[1]]);
            if (r.collision) toi_min = std::min(toi_min, r.t);
        }

        const double step = (toi_min < 1.0) ? safety * toi_min : 1.0;
        x[vi] = x[vi] + step * dx;
    }
}

void BroadPhase::build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt) {
    constexpr double epsilon_pad = 1.0e-10;  // fp tie-breaker, not a safety pad
    build(x, v, mesh, dt, /*node_pad=*/epsilon_pad, /*tri_pad=*/epsilon_pad, /*edge_pad=*/epsilon_pad);
}

BroadPhase::VertexPairs BroadPhase::query_pairs_for_vertex(
        const std::vector<Vec3>& x, int vi, const Vec3& dx, const RefMesh& mesh) const {
    constexpr double epsilon_pad = 1.0e-10;
    VertexPairs result;

    AABB vi_box;
    vi_box.expand(x[vi]);
    vi_box.expand(x[vi] + dx);
    vi_box.min.array() -= epsilon_pad;
    vi_box.max.array() += epsilon_pad;

    // vi as lone-moving node vs external triangles.
    if (cache_.tri_root >= 0) {
        std::vector<int> hits;
        query_bvh(cache_.tri_bvh_nodes, cache_.tri_root, vi_box, hits);
        for (int t : hits) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);
            if (vi == a || vi == b || vi == c) continue;
            result.nt_node_pairs.push_back({vi, {a, b, c}});
        }
    }

    // vi as a moving triangle corner vs external nodes.
    if (cache_.node_root >= 0 && vi < static_cast<int>(cache_.node_to_tris.size())) {
        for (int t : cache_.node_to_tris[vi]) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);

            AABB tri_box;
            tri_box.expand(x[a]);
            tri_box.expand(x[b]);
            tri_box.expand(x[c]);
            if (a == vi) tri_box.expand(x[a] + dx);
            if (b == vi) tri_box.expand(x[b] + dx);
            if (c == vi) tri_box.expand(x[c] + dx);
            tri_box.min.array() -= epsilon_pad;
            tri_box.max.array() += epsilon_pad;

            std::vector<int> hits;
            query_bvh(cache_.node_bvh_nodes, cache_.node_root, tri_box, hits);
            for (int X : hits) {
                if (X == a || X == b || X == c) continue;
                const int vi_local = (a == vi) ? 0 : ((b == vi) ? 1 : 2);
                result.nt_face_pairs.push_back({X, {a, b, c}, vi_local});
            }
        }
    }

    // Segment-segment: edges incident to vi vs non-sharing edges.
    if (cache_.edge_root >= 0 && vi < static_cast<int>(cache_.node_to_edges.size())) {
        for (int ei : cache_.node_to_edges[vi]) {
            const int ea = cache_.edges[ei][0];
            const int eb = cache_.edges[ei][1];

            AABB edge_box;
            edge_box.expand(x[ea]);
            edge_box.expand(x[eb]);
            if (ea == vi) edge_box.expand(x[ea] + dx);
            if (eb == vi) edge_box.expand(x[eb] + dx);
            edge_box.min.array() -= epsilon_pad;
            edge_box.max.array() += epsilon_pad;

            std::vector<int> hits;
            query_bvh(cache_.edge_bvh_nodes, cache_.edge_root, edge_box, hits);
            for (int ej : hits) {
                if (ej == ei) continue;
                const int oa = cache_.edges[ej][0];
                const int ob = cache_.edges[ej][1];
                if (ea == oa || ea == ob || eb == oa || eb == ob) continue;
                // Dedup when both edges are incident to vi: emit only once (ei < ej).
                if ((oa == vi || ob == vi) && ei > ej) continue;

                int vi_dof = (vi == ea) ? 0 : 1;
                result.ss_pairs.push_back({{ea, eb, oa, ob}, vi_dof});
            }
        }
    }

    return result;
}

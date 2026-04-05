#include "broad_phase.h"

#include <map>
#include <set>
#include <tuple>

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

    static inline void add_nt_pair(BroadPhase::Cache& cache, int node, int tri_idx, const RefMesh& mesh) {
        const std::uint64_t key = BroadPhase::nt_key(node, tri_idx);
        if (cache.nt_pair_index.count(key)) return;

        NodeTrianglePair p;
        p.node = node;
        p.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
        p.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
        p.tri_v[2] = tri_vertex(mesh, tri_idx, 2);

        cache.nt_pair_index[key] = cache.nt_pairs.size();
        cache.nt_pairs.push_back(p);
        cache.nt_pair_tri.push_back(tri_idx);
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

        cache.ss_pair_index[key] = cache.ss_pairs.size();
        cache.ss_pairs.push_back(p);
        cache.ss_pair_edges.push_back({a, b});
    }

    static inline void erase_nt_pair_at(BroadPhase::Cache& cache, std::size_t idx) {
        const std::size_t last = cache.nt_pairs.size() - 1;
        const int victim_node = cache.nt_pairs[idx].node;
        const int victim_tri = cache.nt_pair_tri[idx];

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
        const std::array<int, 2> victim_edges = cache.ss_pair_edges[idx];

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

    static inline void remove_nt_pairs_touching_node(BroadPhase::Cache& cache, int node) {
        for (std::size_t i = cache.nt_pairs.size(); i > 0; --i) {
            if (cache.nt_pairs[i - 1].node == node) {
                erase_nt_pair_at(cache, i - 1);
            }
        }
    }

    static inline void remove_nt_pairs_touching_triangle(BroadPhase::Cache& cache, int tri_idx) {
        for (std::size_t i = cache.nt_pair_tri.size(); i > 0; --i) {
            if (cache.nt_pair_tri[i - 1] == tri_idx) {
                erase_nt_pair_at(cache, i - 1);
            }
        }
    }

    static inline void remove_ss_pairs_touching_edge(BroadPhase::Cache& cache, int edge_idx) {
        for (std::size_t i = cache.ss_pair_edges.size(); i > 0; --i) {
            const auto& e = cache.ss_pair_edges[i - 1];
            if (e[0] == edge_idx || e[1] == edge_idx) {
                erase_ss_pair_at(cache, i - 1);
            }
        }
    }

    static inline void query_node_against_triangles(BroadPhase::Cache& cache, int node, const RefMesh& mesh) {
        if (cache.tri_root < 0) return;

        std::vector<int> hits;
        query_bvh(cache.tri_bvh_nodes, cache.tri_root, cache.node_boxes[node], hits);

        for (int t : hits) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);
            if (node_in_triangle(node, a, b, c)) continue;
            add_nt_pair(cache, node, t, mesh);
        }
    }

    static inline void scan_triangle_against_all_nodes(BroadPhase::Cache& cache, int tri_idx, const RefMesh& mesh) {
        if (tri_idx < 0 || tri_idx >= static_cast<int>(cache.tri_boxes.size())) return;

        const int a = tri_vertex(mesh, tri_idx, 0);
        const int b = tri_vertex(mesh, tri_idx, 1);
        const int c = tri_vertex(mesh, tri_idx, 2);
        const AABB& tri_box = cache.tri_boxes[tri_idx];

        const int nv = static_cast<int>(cache.node_boxes.size());
        for (int node = 0; node < nv; ++node) {
            if (node_in_triangle(node, a, b, c)) continue;
            if (!aabb_intersects(cache.node_boxes[node], tri_box)) continue;
            add_nt_pair(cache, node, tri_idx, mesh);
        }
    }

    static inline void query_edge_against_edges(BroadPhase::Cache& cache, int edge_idx) {
        if (cache.edge_root < 0) return;
        if (edge_idx < 0 || edge_idx >= static_cast<int>(cache.edge_boxes.size())) return;

        const Edge e0{cache.edges[edge_idx][0], cache.edges[edge_idx][1]};

        std::vector<int> hits;
        query_bvh(cache.edge_bvh_nodes, cache.edge_root, cache.edge_boxes[edge_idx], hits);

        for (int other : hits) {
            if (other == edge_idx) continue;

            const Edge e1{cache.edges[other][0], cache.edges[other][1]};
            if (share_vertex(e0, e1)) continue;

            add_ss_pair(cache, edge_idx, other);
        }
    }

    static void build_bvh_topology(const std::vector<BVHNode>& nodes, int num_primitives,
                                   std::vector<int>& out_parent, std::vector<int>& out_leaf_node) {
        out_parent.assign(nodes.size(), -1);
        out_leaf_node.assign(num_primitives, -1);

        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            const BVHNode& n = nodes[i];
            if (n.leafIndex >= 0) {
                if (n.leafIndex >= 0 && n.leafIndex < static_cast<int>(out_leaf_node.size())) {
                    out_leaf_node[n.leafIndex] = i;
                }
            } else {
                if (n.left >= 0) out_parent[n.left] = i;
                if (n.right >= 0) out_parent[n.right] = i;
            }
        }
    }

    static void refit_bvh_locally(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes,
                                  const std::vector<int>& parent, const std::vector<int>& leaf_node,
                                  const std::vector<int>& dirty_primitives) {
        if (nodes.empty()) return;

        std::vector<int> stack;
        std::vector<unsigned char> enqueued(nodes.size(), 0);

        auto enqueue_once = [&](int node_idx) {
            if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
            if (!enqueued[node_idx]) {
                enqueued[node_idx] = 1;
                stack.push_back(node_idx);
            }
        };

        for (int prim_idx : dirty_primitives) {
            if (prim_idx < 0 || prim_idx >= static_cast<int>(leaf_node.size())) continue;
            const int leaf = leaf_node[prim_idx];
            if (leaf < 0) continue;
            nodes[leaf].bbox = boxes[prim_idx];
            enqueue_once(parent[leaf]);
        }

        while (!stack.empty()) {
            const int idx = stack.back();
            stack.pop_back();

            BVHNode& n = nodes[idx];
            if (n.leafIndex >= 0) {
                n.bbox = boxes[n.leafIndex];
            } else {
                n.bbox = AABB();
                n.bbox.expand(nodes[n.left].bbox);
                n.bbox.expand(nodes[n.right].bbox);
            }

            enqueue_once(parent[idx]);
        }
    }

}

// Broad phase
void BroadPhase::build(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad) {
    Cache c;

    const int nv = static_cast<int>(x.size());
    const int nt = num_tris(mesh);

    build_unique_edges_and_adjacency(mesh, nv, c.edges, c.node_to_edges, c.node_to_tris);
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
    build_bvh_topology(c.tri_bvh_nodes, static_cast<int>(c.tri_boxes.size()), c.tri_bvh_parent, c.tri_leaf_node);
    build_bvh_topology(c.edge_bvh_nodes, static_cast<int>(c.edge_boxes.size()), c.edge_bvh_parent, c.edge_leaf_node);

    for (int node = 0; node < nv; ++node) {
        query_node_against_triangles(c, node, mesh);
    }

    for (int e = 0; e < ne; ++e) {
        query_edge_against_edges(c, e);
    }

    cache_ = std::move(c);
}

void BroadPhase::initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat) {
    build(x, v, mesh, dt, /*node_pad=*/dhat, /*tri_pad=*/0.0, /*edge_pad=*/dhat * 0.5);
}

void BroadPhase::refresh(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, int moved_node, double dt, double node_pad, double tri_pad, double edge_pad) {
    const int nv = static_cast<int>(cache_.node_boxes.size());
    if (moved_node < 0 || moved_node >= nv) return;

    cache_.node_boxes[moved_node] = build_node_box(x, v, moved_node, dt, node_pad);

    const std::vector<int>& incident_tris = cache_.node_to_tris[moved_node];
    for (int t : incident_tris) {
        const int a = tri_vertex(mesh, t, 0);
        const int b = tri_vertex(mesh, t, 1);
        const int c = tri_vertex(mesh, t, 2);
        cache_.tri_boxes[t] = build_triangle_box(x, v, a, b, c, dt, tri_pad);
    }

    const std::vector<int>& incident_edges = cache_.node_to_edges[moved_node];
    for (int e : incident_edges) {
        cache_.edge_boxes[e] = build_edge_box(x, v, cache_.edges[e][0], cache_.edges[e][1], dt, edge_pad);
    }

    if (cache_.tri_root >= 0) {
        refit_bvh_locally(cache_.tri_bvh_nodes, cache_.tri_boxes, cache_.tri_bvh_parent,
                          cache_.tri_leaf_node, incident_tris);
    }
    if (cache_.edge_root >= 0) {
        refit_bvh_locally(cache_.edge_bvh_nodes, cache_.edge_boxes, cache_.edge_bvh_parent,
                          cache_.edge_leaf_node, incident_edges);
    }

    remove_nt_pairs_touching_node(cache_, moved_node);
    for (int t : incident_tris) {
        remove_nt_pairs_touching_triangle(cache_, t);
    }

    for (int e : incident_edges) {
        remove_ss_pairs_touching_edge(cache_, e);
    }

    query_node_against_triangles(cache_, moved_node, mesh);

    for (int t : incident_tris) {
        scan_triangle_against_all_nodes(cache_, t, mesh);
    }

    for (int e : incident_edges) {
        query_edge_against_edges(cache_, e);
    }
}

void BroadPhase::build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, std::vector<NodeTrianglePair>& out_nt, std::vector<SegmentSegmentPair>& out_ss) {
    BroadPhase tmp;
    tmp.build(x, v, mesh, dt, /*node_pad=*/0.0, /*tri_pad=*/0.0, /*edge_pad=*/0.0);

    out_nt = tmp.nt_pairs();
    out_ss = tmp.ss_pairs();
}

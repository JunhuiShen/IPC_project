#include "broad_phase.h"

#include <set>
#include <tuple>

// ======================================================
// BVH build / refit / query
// ======================================================

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

    out.emplace_back(); // root
    std::vector<BuildTask> stack;
    stack.push_back({0, 0, static_cast<int>(idx.size())});

    while (!stack.empty()) {
        const BuildTask task = stack.back();
        stack.pop_back();

        AABB node_box;
        for (int i = task.start; i < task.end; ++i) node_box.expand(boxes[idx[i]]);
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
        std::nth_element(idx.begin() + task.start, idx.begin() + mid, idx.begin() + task.end,
                         [&](int a, int b) { return boxes[a].centroid()[axis] < boxes[b].centroid()[axis]; });

        const int left = static_cast<int>(out.size());
        out.emplace_back();
        const int right = static_cast<int>(out.size());
        out.emplace_back();
        out[task.node_idx].left = left;
        out[task.node_idx].right = right;

        // DFS order with explicit stack; push right first so left is handled next.
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

// ======================================================
// Local helpers
// ======================================================

namespace {

    struct Edge {
        int v0 = -1;
        int v1 = -1;
    };

    static inline Edge canonical_edge(int a, int b) {
        if (a > b) std::swap(a, b);
        return {a, b};
    }

    static std::vector<Edge> build_unique_edges(const RefMesh& mesh) {
        std::set<std::pair<int,int>> edge_set;

        const int nt = num_tris(mesh);
        for (int t = 0; t < nt; ++t) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);

            const Edge e0 = canonical_edge(a, b);
            const Edge e1 = canonical_edge(b, c);
            const Edge e2 = canonical_edge(c, a);

            edge_set.insert({e0.v0, e0.v1});
            edge_set.insert({e1.v0, e1.v1});
            edge_set.insert({e2.v0, e2.v1});
        }

        std::vector<Edge> edges;
        edges.reserve(edge_set.size());
        for (const auto& [a, b] : edge_set)
            edges.push_back({a, b});

        return edges;
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

    static inline void add_nt_pair(BroadPhase3D::Cache& cache, int node, int tri_idx, const RefMesh& mesh) {
        const std::uint64_t key = BroadPhase3D::nt_key(node, tri_idx);
        if (cache.nt_pair_index.count(key)) return;

        NodeTrianglePair p;
        p.node     = node;
        p.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
        p.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
        p.tri_v[2] = tri_vertex(mesh, tri_idx, 2);

        cache.nt_pair_index[key] = cache.nt_pairs.size();
        cache.nt_pairs.push_back(p);
    }

    static inline void add_ss_pair(BroadPhase3D::Cache& cache, int e0, int e1, const std::vector<Edge>& edges) {
        int a = e0, b = e1;
        if (a > b) std::swap(a, b);

        const std::uint64_t key = BroadPhase3D::ss_key(a, b);
        if (cache.ss_pair_index.count(key)) return;

        SegmentSegmentPair p;
        p.v[0] = edges[a].v0;
        p.v[1] = edges[a].v1;
        p.v[2] = edges[b].v0;
        p.v[3] = edges[b].v1;

        cache.ss_pair_index[key] = cache.ss_pairs.size();
        cache.ss_pairs.push_back(p);
    }

} // namespace

// ======================================================
// BroadPhase3D
// ======================================================

void BroadPhase3D::build(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad) {
    Cache c;

    const int nv = static_cast<int>(x.size());
    const int nt = num_tris(mesh);
    const std::vector<Edge> edges = build_unique_edges(mesh);
    const int ne = static_cast<int>(edges.size());

    c.node_boxes.resize(nv);
    for (int i = 0; i < nv; ++i)
        c.node_boxes[i] = build_node_box(x, v, i, dt, node_pad);

    c.tri_boxes.resize(nt);
    for (int t = 0; t < nt; ++t) {
        const int a = tri_vertex(mesh, t, 0);
        const int b = tri_vertex(mesh, t, 1);
        const int cc = tri_vertex(mesh, t, 2);
        c.tri_boxes[t] = build_triangle_box(x, v, a, b, cc, dt, tri_pad);
    }

    c.edge_boxes.resize(ne);
    for (int e = 0; e < ne; ++e)
        c.edge_boxes[e] = build_edge_box(x, v, edges[e].v0, edges[e].v1, dt, edge_pad);

    c.tri_root  = build_bvh(c.tri_boxes,  c.tri_bvh_nodes);
    c.edge_root = build_bvh(c.edge_boxes, c.edge_bvh_nodes);

    // Node-triangle candidates
    for (int node = 0; node < nv; ++node) {
        std::vector<int> hits;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], hits);

        for (int t : hits) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int cc = tri_vertex(mesh, t, 2);

            if (node_in_triangle(node, a, b, cc)) continue;
            add_nt_pair(c, node, t, mesh);
        }
    }

    // Segment-segment candidates
    for (int e = 0; e < ne; ++e) {
        std::vector<int> hits;
        query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], hits);

        for (int f : hits) {
            if (f <= e) continue;
            if (share_vertex(edges[e], edges[f])) continue;
            add_ss_pair(c, e, f, edges);
        }
    }

    cache_ = std::move(c);
}

void BroadPhase3D::initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat) {
    build(x, v, mesh, dt, /*node_pad=*/dhat, /*tri_pad=*/dhat, /*edge_pad=*/dhat);
}

void BroadPhase3D::refresh(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, int /*moved_node*/, double dt, double node_pad, double tri_pad, double edge_pad) {
    // Correct baseline implementation:
    // rebuild globally after a local move.
    build(x, v, mesh, dt, node_pad, tri_pad, edge_pad);
}

void BroadPhase3D::build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, std::vector<NodeTrianglePair>& out_nt, std::vector<SegmentSegmentPair>& out_ss) {
    BroadPhase3D tmp;
    tmp.build(x, v, mesh, dt, /*node_pad=*/0.0, /*tri_pad=*/0.0, /*edge_pad=*/0.0);

    out_nt = tmp.nt_pairs();
    out_ss = tmp.ss_pairs();
}

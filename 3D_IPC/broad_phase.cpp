#include "broad_phase.h"

#include <cmath>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

// BVH build / refit / query
namespace {
double point_aabb_squared_distance(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    double distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        double distance = 0.0;
        if (p[axis] < lo[axis])
            distance = lo[axis] - p[axis];
        else if (p[axis] > hi[axis])
            distance = p[axis] - hi[axis];
        distance_squared += distance * distance;
    }
    return distance_squared;
}

inline int build_bvh_impl(const std::vector<AABB>& boxes, std::vector<BVHNode>& out, std::vector<int>* leaf_to_node) {
    out.clear();
    if (leaf_to_node) leaf_to_node->assign(boxes.size(), -1);
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
            const int leaf = idx[task.start];
            out[task.node_idx].leafIndex = leaf;
            if (leaf_to_node) (*leaf_to_node)[leaf] = task.node_idx;
            continue;
        }

        const Vec3 e = node_box.extent();
        int axis = 0;
        if (e.y() > e.x() && e.y() >= e.z()) axis = 1;
        else if (e.z() > e.x() && e.z() >= e.y()) axis = 2;

        const int mid = task.start + count / 2;
        std::nth_element( idx.begin() + task.start, idx.begin() + mid, idx.begin() + task.end, [&](int a, int b) {
                    return boxes[a].min[axis] + boxes[a].max[axis] < boxes[b].min[axis] + boxes[b].max[axis];
                });

        const int left = static_cast<int>(out.size());
        out.emplace_back();
        const int right = static_cast<int>(out.size());
        out.emplace_back();

        out[task.node_idx].left = left;
        out[task.node_idx].right = right;
        out[left].parent = task.node_idx;
        out[right].parent = task.node_idx;

        stack.push_back({right, mid, task.end});
        stack.push_back({left, task.start, mid});
    }

    return 0;
}
}  // namespace

bool node_triangle_aabbs_within_distance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, double distance_squared) {
    const Vec3 lo = a.cwiseMin(b).cwiseMin(c);
    const Vec3 hi = a.cwiseMax(b).cwiseMax(c);
    if (point_aabb_squared_distance(p, lo, hi) > distance_squared) return false;
    const Vec3 normal = (b - a).cross(c - a);
    const double normal_squared = normal.squaredNorm();
    const double scaled_distance = (p - a).dot(normal);
    const double scaled_distance_squared = scaled_distance * scaled_distance;
    constexpr double conservative_roundoff = 1.0 + 1.0e-10;
    return !std::isfinite(scaled_distance_squared) || !std::isfinite(normal_squared) || scaled_distance_squared <= conservative_roundoff * distance_squared * normal_squared;
}

bool segment_aabbs_within_distance(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1, double distance_squared) {
    const Vec3 alo = a0.cwiseMin(a1);
    const Vec3 ahi = a0.cwiseMax(a1);
    const Vec3 blo = b0.cwiseMin(b1);
    const Vec3 bhi = b0.cwiseMax(b1);
    double aabb_distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        double distance = 0.0;
        if (ahi[axis] < blo[axis])
            distance = blo[axis] - ahi[axis];
        else if (bhi[axis] < alo[axis])
            distance = alo[axis] - bhi[axis];
        aabb_distance_squared += distance * distance;
    }
    if (aabb_distance_squared > distance_squared) return false;
    const Vec3 normal = (a1 - a0).cross(b1 - b0);
    const double normal_squared = normal.squaredNorm();
    const double scaled_distance = (b0 - a0).dot(normal);
    const double scaled_distance_squared = scaled_distance * scaled_distance;
    constexpr double conservative_roundoff = 1.0 + 1.0e-10;
    return !std::isfinite(scaled_distance_squared) || !std::isfinite(normal_squared) || scaled_distance_squared <= conservative_roundoff * distance_squared * normal_squared;
}

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    return build_bvh_impl(boxes, out, nullptr);
}

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out, std::vector<int>& leaf_to_node) {
    return build_bvh_impl(boxes, out, &leaf_to_node);
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

void refit_bvh_leaf(std::vector<BVHNode>& nodes, const std::vector<int>& leaf_to_node, int leafIndex, const AABB& new_box) {
    if (leafIndex < 0 || leafIndex >= static_cast<int>(leaf_to_node.size())) return;
    int idx = leaf_to_node[leafIndex];
    if (idx < 0) return;

    nodes[idx].bbox = new_box;
    for (int parent = nodes[idx].parent; parent >= 0; parent = nodes[parent].parent) {
        AABB combined = nodes[nodes[parent].left].bbox;
        combined.expand(nodes[nodes[parent].right].bbox);
        const AABB& prev = nodes[parent].bbox;
        if (combined.min == prev.min && combined.max == prev.max) break;
        nodes[parent].bbox = combined;
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

    static std::vector<unsigned char> build_tet_interior_node_mask(
        const RefMesh& mesh, int nv) {
        std::vector<unsigned char> is_interior(
            static_cast<std::size_t>(nv), 0);
        for (const int node : mesh.tet_nodes) {
            if (node >= 0 && node < nv)
                is_interior[static_cast<std::size_t>(node)] = 1;
        }
        for (const int node : mesh.surface_nodes) {
            if (node >= 0 && node < nv)
                is_interior[static_cast<std::size_t>(node)] = 0;
        }
        return is_interior;
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

    static inline int rigid_owner(const RefMesh& mesh, const int node) {
        return node >= 0 && node < static_cast<int>(mesh.node_to_rb.size()) ? mesh.node_to_rb[static_cast<std::size_t>(node)] : -1;
    }

    static inline int common_rigid_owner(const RefMesh& mesh, const int first, const int second) {
        const int owner = rigid_owner(mesh, first);
        return owner >= 0 && rigid_owner(mesh, second) == owner ? owner : -1;
    }

    static void build_bvh_rigid_owners(const std::vector<BVHNode>& nodes, const std::vector<int>& leaf_rigid_owner, std::vector<int>& node_rigid_owner) {
        node_rigid_owner.resize(nodes.size());
        for (int node_index = static_cast<int>(nodes.size()) - 1; node_index >= 0; --node_index) {
            const BVHNode& node = nodes[static_cast<std::size_t>(node_index)];
            if (node.leafIndex >= 0) {
                node_rigid_owner[static_cast<std::size_t>(node_index)] = leaf_rigid_owner[static_cast<std::size_t>(node.leafIndex)];
                continue;
            }
            const int left_owner = node_rigid_owner[static_cast<std::size_t>(node.left)];
            const int right_owner = node_rigid_owner[static_cast<std::size_t>(node.right)];
            node_rigid_owner[static_cast<std::size_t>(node_index)] = left_owner >= 0 && left_owner == right_owner ? left_owner : -1;
        }
    }

    static void query_bvh_excluding_rigid_owner(const std::vector<BVHNode>& nodes, const std::vector<int>& node_rigid_owner, const int root, const AABB& query, const int excluded_owner, std::vector<int>& hits) {
        if (root < 0)
            return;

        int stack[256];
        int top = 0;
        stack[top++] = root;
        while (top > 0) {
            const int node_index = stack[--top];
            if (excluded_owner >= 0 && node_rigid_owner[static_cast<std::size_t>(node_index)] == excluded_owner) {
                continue;
            }
            const BVHNode& node = nodes[static_cast<std::size_t>(node_index)];
            if (!aabb_intersects(node.bbox, query))
                continue;
            if (node.leafIndex >= 0) {
                hits.push_back(node.leafIndex);
            } else {
                stack[top++] = node.left;
                stack[top++] = node.right;
            }
        }
    }

    static inline bool is_rigid_self_nt_pair(const RefMesh& mesh, const int node, const int tri_idx) {
        const int owner = rigid_owner(mesh, node);
        return owner >= 0 && rigid_owner(mesh, tri_vertex(mesh, tri_idx, 0)) == owner && rigid_owner(mesh, tri_vertex(mesh, tri_idx, 1)) == owner && rigid_owner(mesh, tri_vertex(mesh, tri_idx, 2)) == owner;
    }

    static inline bool is_rigid_self_ss_pair(const BroadPhase::Cache& cache, const RefMesh& mesh, const int e0, const int e1) {
        const int owner = rigid_owner(mesh, cache.edges[e0][0]);
        return owner >= 0 && rigid_owner(mesh, cache.edges[e0][1]) == owner && rigid_owner(mesh, cache.edges[e1][0]) == owner && rigid_owner(mesh, cache.edges[e1][1]) == owner;
    }

    static inline bool stores_vertex_incidence(const BroadPhase::InitializationMode mode, const RefMesh& mesh, const int node) {
        if (mode == BroadPhase::InitializationMode::Refittable || mode == BroadPhase::InitializationMode::DeformableSolver)
            return true;
        if (mode == BroadPhase::InitializationMode::GeneralSolver)
            return rigid_owner(mesh, node) < 0;
        return false;
    }

    static inline void append_nt_pair(BroadPhase::Cache& cache, int node, int tri_idx, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        const std::size_t idx = cache.nt_pairs.size();

        NodeTrianglePair p;
        p.node = node;
        p.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
        p.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
        p.tri_v[2] = tri_vertex(mesh, tri_idx, 2);

        cache.nt_pairs.push_back(p);
        cache.nt_pair_tri.push_back(tri_idx);

        if (!cache.vertex_nt.empty()) {
            if (stores_vertex_incidence(mode, mesh, p.node))
                cache.vertex_nt[p.node].push_back({idx, 0});
            if (stores_vertex_incidence(mode, mesh, p.tri_v[0]))
                cache.vertex_nt[p.tri_v[0]].push_back({idx, 1});
            if (stores_vertex_incidence(mode, mesh, p.tri_v[1]))
                cache.vertex_nt[p.tri_v[1]].push_back({idx, 2});
            if (stores_vertex_incidence(mode, mesh, p.tri_v[2]))
                cache.vertex_nt[p.tri_v[2]].push_back({idx, 3});
        }
    }

    static inline void append_ss_pair(BroadPhase::Cache& cache, int e0, int e1, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        int a = e0;
        int b = e1;
        if (a > b) std::swap(a, b);

        const std::size_t idx = cache.ss_pairs.size();

        SegmentSegmentPair p;
        p.v[0] = cache.edges[a][0];
        p.v[1] = cache.edges[a][1];
        p.v[2] = cache.edges[b][0];
        p.v[3] = cache.edges[b][1];

        cache.ss_pairs.push_back(p);
        cache.ss_pair_edges.push_back({a, b});

        if (!cache.vertex_ss.empty()) {
            for (int role = 0; role < 4; ++role) {
                if (stores_vertex_incidence(mode, mesh, p.v[role]))
                    cache.vertex_ss[p.v[role]].push_back({idx, role});
            }
        }
    }

    static inline bool earlier_edge_query_already_reported_pair(const BroadPhase::Cache& cache, int current_edge, int hit_edge) {
        if (hit_edge >= current_edge) return false;

        // Edge-query results are consumed in increasing edge order. Check
        // whether the already-consumed query for hit_edge also found
        // current_edge; if so, this unordered pair was already appended.
        if (current_edge < static_cast<int>(cache.red_edge_boxes.size()))
            return aabb_intersects(cache.edge_boxes[hit_edge], cache.red_edge_boxes[static_cast<std::size_t>(current_edge)]);

        // Velocity-based BVHs do not keep leaf indices, so consult the saved
        // query results directly when refresh_pairs() is called on one.
        const auto& earlier_hits = cache.edge_hits[hit_edge];
        return std::find(earlier_hits.begin(), earlier_hits.end(), current_edge)!= earlier_hits.end();
    }

    static inline bool accepts_nt_pair(const RefMesh& mesh, const int node, const int tri_idx, const BroadPhase::InitializationMode mode) {
        const int a = tri_vertex(mesh, tri_idx, 0);
        const int b = tri_vertex(mesh, tri_idx, 1);
        const int c = tri_vertex(mesh, tri_idx, 2);
        if (node_in_triangle(node, a, b, c))
            return false;
        return mode == BroadPhase::InitializationMode::Refittable || mode == BroadPhase::InitializationMode::DeformableSolver || !is_rigid_self_nt_pair(mesh, node, tri_idx);
    }

    static inline bool accepts_ss_pair(const BroadPhase::Cache& cache, const RefMesh& mesh, const int edge, const int other, const BroadPhase::InitializationMode mode) {
        if (other == edge)
            return false;
        const Edge e0{cache.edges[edge][0], cache.edges[edge][1]};
        const Edge e1{cache.edges[other][0], cache.edges[other][1]};
        if (share_vertex(e0, e1))
            return false;
        if (earlier_edge_query_already_reported_pair(cache, edge, other))
            return false;
        return mode == BroadPhase::InitializationMode::Refittable || mode == BroadPhase::InitializationMode::DeformableSolver || !is_rigid_self_ss_pair(cache, mesh, edge, other);
    }

    static inline void write_nt_pair(BroadPhase::Cache& cache, const std::size_t pair_index, const int node, const int tri_idx, const RefMesh& mesh) {
        NodeTrianglePair& pair = cache.nt_pairs[pair_index];
        pair.node = node;
        pair.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
        pair.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
        pair.tri_v[2] = tri_vertex(mesh, tri_idx, 2);
        cache.nt_pair_tri[pair_index] = tri_idx;
    }

    static inline void write_ss_pair(BroadPhase::Cache& cache, const std::size_t pair_index, const int first_edge, const int second_edge) {
        const int a = std::min(first_edge, second_edge);
        const int b = std::max(first_edge, second_edge);
        SegmentSegmentPair& pair = cache.ss_pairs[pair_index];
        pair.v[0] = cache.edges[a][0];
        pair.v[1] = cache.edges[a][1];
        pair.v[2] = cache.edges[b][0];
        pair.v[3] = cache.edges[b][1];
        cache.ss_pair_edges[pair_index] = {a, b};
    }

    static void build_solver_vertex_incidence(BroadPhase::Cache& cache, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        if (mode != BroadPhase::InitializationMode::DeformableSolver && mode != BroadPhase::InitializationMode::GeneralSolver)
            return;

        const std::size_t nv = cache.vertex_nt.size();
        int num_workers = 1;
        #ifdef _OPENMP
        num_workers = std::max(1, omp_get_max_threads());
        #endif
        std::vector<std::size_t> counts(static_cast<std::size_t>(num_workers) * nv, 0);

        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            std::size_t* worker_counts = counts.data() + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.nt_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.nt_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const NodeTrianglePair& pair = cache.nt_pairs[pair_index];
                if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.node) < 0) ++worker_counts[static_cast<std::size_t>(pair.node)];
                for (int role = 0; role < 3; ++role) if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.tri_v[role]) < 0) ++worker_counts[static_cast<std::size_t>(pair.tri_v[role])];
            }
        }
        for (std::size_t node = 0; node < nv; ++node) {
            std::size_t offset = 0;
            for (int worker = 0; worker < num_workers; ++worker) {
                std::size_t& count = counts[static_cast<std::size_t>(worker) * nv + node];
                const std::size_t worker_count = count;
                count = offset;
                offset += worker_count;
            }
            cache.vertex_nt[node].resize(offset);
        }
        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            std::size_t* worker_offsets = counts.data() + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.nt_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.nt_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const NodeTrianglePair& pair = cache.nt_pairs[pair_index];
                if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.node) < 0) cache.vertex_nt[static_cast<std::size_t>(pair.node)][worker_offsets[static_cast<std::size_t>(pair.node)]++] = {pair_index, 0};
                for (int role = 0; role < 3; ++role) if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.tri_v[role]) < 0) cache.vertex_nt[static_cast<std::size_t>(pair.tri_v[role])][worker_offsets[static_cast<std::size_t>(pair.tri_v[role])]++] = {pair_index, role + 1};
            }
        }

        std::fill(counts.begin(), counts.end(), 0);
        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            std::size_t* worker_counts = counts.data() + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.ss_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.ss_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const SegmentSegmentPair& pair = cache.ss_pairs[pair_index];
                for (int role = 0; role < 4; ++role) if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.v[role]) < 0) ++worker_counts[static_cast<std::size_t>(pair.v[role])];
            }
        }
        for (std::size_t node = 0; node < nv; ++node) {
            std::size_t offset = 0;
            for (int worker = 0; worker < num_workers; ++worker) {
                std::size_t& count = counts[static_cast<std::size_t>(worker) * nv + node];
                const std::size_t worker_count = count;
                count = offset;
                offset += worker_count;
            }
            cache.vertex_ss[node].resize(offset);
        }
        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            std::size_t* worker_offsets = counts.data() + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.ss_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.ss_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const SegmentSegmentPair& pair = cache.ss_pairs[pair_index];
                for (int role = 0; role < 4; ++role) if (mode == BroadPhase::InitializationMode::DeformableSolver || rigid_owner(mesh, pair.v[role]) < 0) cache.vertex_ss[static_cast<std::size_t>(pair.v[role])][worker_offsets[static_cast<std::size_t>(pair.v[role])]++] = {pair_index, role};
            }
        }
    }

    static void materialize_solver_pairs(BroadPhase::Cache& cache, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        const int nv = static_cast<int>(cache.node_hits.size());
        const int ne = static_cast<int>(cache.edge_hits.size());

        // Each query owns one contiguous output interval. Parallel counting,
        // a deterministic prefix sum, and parallel writes preserve the exact
        // legacy pair order without a serial push_back bottleneck.
        std::vector<std::size_t> nt_offsets(static_cast<std::size_t>(nv) + 1, 0);
        #pragma omp parallel for schedule(dynamic, 32)
        for (int node = 0; node < nv; ++node) {
            std::size_t count = 0;
            for (const int tri_idx : cache.node_hits[node])
                count += accepts_nt_pair(mesh, node, tri_idx, mode);
            nt_offsets[static_cast<std::size_t>(node) + 1] = count;
        }
        for (int node = 0; node < nv; ++node) {
            nt_offsets[static_cast<std::size_t>(node) + 1] += nt_offsets[static_cast<std::size_t>(node)];
        }
        cache.nt_pairs.resize(nt_offsets.back());
        cache.nt_pair_tri.resize(nt_offsets.back());
        #pragma omp parallel for schedule(dynamic, 32)
        for (int node = 0; node < nv; ++node) {
            std::size_t pair_index = nt_offsets[static_cast<std::size_t>(node)];
            for (const int tri_idx : cache.node_hits[node]) {
                if (accepts_nt_pair(mesh, node, tri_idx, mode))
                    write_nt_pair(cache, pair_index++, node, tri_idx, mesh);
            }
        }

        std::vector<std::size_t> ss_offsets(static_cast<std::size_t>(ne) + 1, 0);
        #pragma omp parallel for schedule(dynamic, 32)
        for (int edge = 0; edge < ne; ++edge) {
            std::size_t count = 0;
            for (const int other : cache.edge_hits[edge])
                count += accepts_ss_pair(cache, mesh, edge, other, mode);
            ss_offsets[static_cast<std::size_t>(edge) + 1] = count;
        }
        for (int edge = 0; edge < ne; ++edge) {
            ss_offsets[static_cast<std::size_t>(edge) + 1] += ss_offsets[static_cast<std::size_t>(edge)];
        }
        cache.ss_pairs.resize(ss_offsets.back());
        cache.ss_pair_edges.resize(ss_offsets.back());
        #pragma omp parallel for schedule(dynamic, 32)
        for (int edge = 0; edge < ne; ++edge) {
            std::size_t pair_index = ss_offsets[static_cast<std::size_t>(edge)];
            for (const int other : cache.edge_hits[edge]) {
                if (accepts_ss_pair(cache, mesh, edge, other, mode))
                    write_ss_pair(cache, pair_index++, edge, other);
            }
        }

        build_solver_vertex_incidence(cache, mesh, mode);
    }

    // Recycle broad-phase storage/topology to avoid allocation churn; per-build boxes, BVHs, and pairs are cleared.
    static BroadPhase::Cache take_reusable_cache(BroadPhase::Cache& old_cache, const int nv, const BroadPhase::InitializationMode mode) {
        BroadPhase::Cache c = std::move(old_cache);

        c.node_boxes.clear();
        c.tri_boxes.clear();
        c.edge_boxes.clear();

        c.tri_bvh_nodes.clear();
        c.edge_bvh_nodes.clear();
        c.node_bvh_nodes.clear();
        c.tri_leaf_to_node.clear();
        c.edge_leaf_to_node.clear();
        c.node_leaf_to_node.clear();
        c.tri_bvh_rigid_owner.clear();
        c.edge_bvh_rigid_owner.clear();

        c.node_root = -1;
        c.tri_root = -1;
        c.edge_root = -1;

        c.nt_pairs.clear();
        c.ss_pairs.clear();
        c.nt_pair_tri.clear();
        c.ss_pair_edges.clear();
        if (mode == BroadPhase::InitializationMode::RigidSolver) {
            c.vertex_nt.clear();
            c.vertex_ss.clear();
        } else {
            if (static_cast<int>(c.vertex_nt.size()) == nv) {
                for (auto& row : c.vertex_nt)
                    row.clear();
            } else {
                c.vertex_nt.assign(nv, {});
            }
            if (static_cast<int>(c.vertex_ss.size()) == nv) {
                for (auto& row : c.vertex_ss)
                    row.clear();
            } else {
                c.vertex_ss.assign(nv, {});
            }
        }
        return c;
    }

    static std::vector<std::vector<int>>& prepare_hit_rows(std::vector<std::vector<int>>& rows, int n) {
        if (static_cast<int>(rows.size()) == n) {
            for (auto& row : rows) row.clear();
        } else {
            rows.assign(n, {});
        }
        return rows;
    }

}

// Broad phase
void BroadPhase::set_mesh_topology(const RefMesh& mesh, int nv) {
    build_unique_edges_and_adjacency(mesh, nv, topo_.edges, topo_.node_to_edges, topo_.node_to_tris);
    topo_.tri_rigid_owner.resize(static_cast<std::size_t>(num_tris(mesh)));
    for (int tri_idx = 0; tri_idx < num_tris(mesh); ++tri_idx) {
        const int owner = common_rigid_owner(mesh, tri_vertex(mesh, tri_idx, 0), tri_vertex(mesh, tri_idx, 1));
        topo_.tri_rigid_owner[static_cast<std::size_t>(tri_idx)] = owner >= 0 && rigid_owner(mesh, tri_vertex(mesh, tri_idx, 2)) == owner ? owner : -1;
    }
    topo_.edge_rigid_owner.resize(topo_.edges.size());
    for (std::size_t edge = 0; edge < topo_.edges.size(); ++edge) {
        topo_.edge_rigid_owner[edge] = common_rigid_owner(mesh, topo_.edges[edge][0], topo_.edges[edge][1]);
    }
    topo_.surface_nt_query_nodes.clear();
    topo_.surface_nt_query_nodes_valid = false;
    cache_.edges.clear();
    cache_.node_to_edges.clear();
    cache_.node_to_tris.clear();
    topology_valid_ = true;
}

const std::vector<int>& BroadPhase::surface_nt_query_nodes(
    const RefMesh& mesh, const int nv) {
    if (topo_.surface_nt_query_nodes_valid)
        return topo_.surface_nt_query_nodes;

    const std::vector<unsigned char> tet_interior_nodes = build_tet_interior_node_mask(mesh, nv);
    topo_.surface_nt_query_nodes.clear();
    topo_.surface_nt_query_nodes.reserve(static_cast<std::size_t>(nv));
    for (int node = 0; node < nv; ++node) {
        if (tet_interior_nodes[static_cast<std::size_t>(node)] == 0)
            topo_.surface_nt_query_nodes.push_back(node);
    }
    topo_.surface_nt_query_nodes_valid = true;
    return topo_.surface_nt_query_nodes;
}

void BroadPhase::build(
    const std::vector<Vec3>& x, const std::vector<Vec3>& v,
    const RefMesh& mesh, double dt, double node_pad, double tri_pad,
    double edge_pad, const bool exclude_tet_interior_nt_queries) {
    const int nv = static_cast<int>(x.size());
    const int nt = num_tris(mesh);
    exclude_tet_interior_nt_queries_ = exclude_tet_interior_nt_queries;

    constexpr InitializationMode mode = InitializationMode::Refittable;
    Cache c = take_reusable_cache(cache_, nv, mode);
    c.excludes_tet_interior_nt_queries = exclude_tet_interior_nt_queries;

    if (!topology_valid_) set_mesh_topology(mesh, nv);
    if (c.edges.empty()) c.edges = topo_.edges;
    if (c.node_to_edges.empty()) c.node_to_edges = topo_.node_to_edges;
    if (c.node_to_tris.empty()) c.node_to_tris = topo_.node_to_tris;
    const int ne = static_cast<int>(c.edges.size());
    const std::vector<int>* surface_query_nodes = exclude_tet_interior_nt_queries ? &surface_nt_query_nodes(mesh, nv) : nullptr;

    c.node_boxes.resize(nv);
    for (int i = 0; i < nv; ++i) {
        c.node_boxes[i] = build_node_box(x, v, i, dt, node_pad);
    }

    c.tri_boxes.resize(nt);
    #pragma omp parallel for schedule(static)
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

    #pragma omp parallel sections
    {
        #pragma omp section
        { c.tri_root = build_bvh(c.tri_boxes, c.tri_bvh_nodes); }
        #pragma omp section
        { c.edge_root = build_bvh(c.edge_boxes, c.edge_bvh_nodes); }
        #pragma omp section
        { c.node_root = build_bvh(c.node_boxes, c.node_bvh_nodes); }
    }

    // Parallel BVH queries, followed by serial ordered pair insertion.
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    const int num_nt_query_nodes = exclude_tet_interior_nt_queries ? static_cast<int>(surface_query_nodes->size()) : nv;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int query_index = 0; query_index < num_nt_query_nodes; ++query_index) {
        const int node = exclude_tet_interior_nt_queries ? (*surface_query_nodes)[static_cast<std::size_t>(query_index)] : query_index;
        if (c.tri_root < 0) continue;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
    }
    for (int node = 0; node < nv; ++node) {
        for (int t : node_hits[node]) {
            const int a  = tri_vertex(mesh, t, 0);
            const int b  = tri_vertex(mesh, t, 1);
            const int cc = tri_vertex(mesh, t, 2);
            if (node_in_triangle(node, a, b, cc)) continue;
            append_nt_pair(c, node, t, mesh, mode);
        }
    }

    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (c.edge_root < 0) continue;
        query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], edge_hits[e]);
    }
    for (int e = 0; e < ne; ++e) {
        const Edge e0{c.edges[e][0], c.edges[e][1]};
        for (int other : edge_hits[e]) {
            if (other <= e) continue;
            const Edge e1{c.edges[other][0], c.edges[other][1]};
            if (share_vertex(e0, e1)) continue;
            append_ss_pair(c, e, other, mesh, mode);
        }
    }

    cache_ = std::move(c);
}

void BroadPhase::initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat) {
    build(
        x, v, mesh, dt, /*node_pad=*/dhat, /*tri_pad=*/0.0,
        /*edge_pad=*/dhat * 0.5,
        /*exclude_tet_interior_nt_queries=*/false);
}

void BroadPhase::initialize_surface_nodes(
    const std::vector<Vec3>& x, const std::vector<Vec3>& v,
    const RefMesh& mesh, const double dt, const double dhat) {
    build(
        x, v, mesh, dt, /*node_pad=*/dhat, /*tri_pad=*/0.0,
        /*edge_pad=*/dhat * 0.5,
        /*exclude_tet_interior_nt_queries=*/true);
}

void BroadPhase::initialize_surface_nodes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, const double d_hat, const InitializationMode mode) {
    initialize_from_vertex_boxes(vertex_boxes, mesh, d_hat, /*exclude_tet_interior_nt_queries=*/true, mode);
}

void BroadPhase::initialize(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, const double d_hat, const InitializationMode mode) {
    initialize_from_vertex_boxes(vertex_boxes, mesh, d_hat, /*exclude_tet_interior_nt_queries=*/false, mode);
}

void BroadPhase::initialize_from_vertex_boxes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, const double d_hat, const bool exclude_tet_interior_nt_queries, const InitializationMode mode) {
    const int nv = static_cast<int>(vertex_boxes.size());
    const int nt = num_tris(mesh);
    exclude_tet_interior_nt_queries_ = exclude_tet_interior_nt_queries;

    Cache c = take_reusable_cache(cache_, nv, mode);
    c.excludes_tet_interior_nt_queries = exclude_tet_interior_nt_queries;

    if (!topology_valid_) set_mesh_topology(mesh, nv);
    if (c.edges.empty()) c.edges = topo_.edges;
    if (c.node_to_edges.empty()) c.node_to_edges = topo_.node_to_edges;
    if (c.node_to_tris.empty()) c.node_to_tris = topo_.node_to_tris;
    const int ne = static_cast<int>(c.edges.size());
    const std::vector<int>* surface_query_nodes = exclude_tet_interior_nt_queries ? &surface_nt_query_nodes(mesh, nv) : nullptr;

    // Blue boxes: one certified motion box per vertex.
    c.node_boxes = vertex_boxes;

    const Vec3 pad = d_hat * Vec3::Ones();
    c.tri_boxes.resize(nt);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; ++t) {
        const int a  = tri_vertex(mesh, t, 0);
        const int b  = tri_vertex(mesh, t, 1);
        const int cc = tri_vertex(mesh, t, 2);
        // Green triangle boxes: union of incident blue boxes, padded by d_hat.
        c.tri_boxes[t] = vertex_boxes[a];
        c.tri_boxes[t].expand(vertex_boxes[b]);
        c.tri_boxes[t].expand(vertex_boxes[cc]);
        c.tri_boxes[t].min -= pad;
        c.tri_boxes[t].max += pad;
    }

    c.edge_boxes.resize(ne);
    std::vector<AABB>& red_edge_boxes = c.red_edge_boxes;
    red_edge_boxes.resize(ne);
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; ++e) {
        // Red edge boxes are unpadded edge unions; edge_boxes stores the padded green boxes.
        red_edge_boxes[e] = vertex_boxes[c.edges[e][0]];
        red_edge_boxes[e].expand(vertex_boxes[c.edges[e][1]]);
        c.edge_boxes[e] = red_edge_boxes[e];
        c.edge_boxes[e].min -= pad;
        c.edge_boxes[e].max += pad;
    }

    if (mode == InitializationMode::Refittable) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                c.tri_root = build_bvh(c.tri_boxes, c.tri_bvh_nodes, c.tri_leaf_to_node);
            }
            #pragma omp section
            {
                c.edge_root = build_bvh(red_edge_boxes, c.edge_bvh_nodes, c.edge_leaf_to_node);
            }
            #pragma omp section
            {
                c.node_root = build_bvh(c.node_boxes, c.node_bvh_nodes, c.node_leaf_to_node);
            }
        }
    } else {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                c.tri_root = build_bvh(c.tri_boxes, c.tri_bvh_nodes);
                if (mode != InitializationMode::DeformableSolver) build_bvh_rigid_owners(c.tri_bvh_nodes, topo_.tri_rigid_owner, c.tri_bvh_rigid_owner);
            }
            #pragma omp section
            {
                c.edge_root = build_bvh(red_edge_boxes, c.edge_bvh_nodes);
                if (mode != InitializationMode::DeformableSolver) build_bvh_rigid_owners(c.edge_bvh_nodes, topo_.edge_rigid_owner, c.edge_bvh_rigid_owner);
            }
        }
    }

    // NT candidates: blue node boxes queried against green triangle boxes.
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    const int num_nt_query_nodes = exclude_tet_interior_nt_queries ? static_cast<int>(surface_query_nodes->size()) : nv;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int query_index = 0; query_index < num_nt_query_nodes; ++query_index) {
        const int node = exclude_tet_interior_nt_queries ? (*surface_query_nodes)[static_cast<std::size_t>(query_index)] : query_index;
        if (c.tri_root < 0) continue;
        if (mode == InitializationMode::Refittable || mode == InitializationMode::DeformableSolver) {
            query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
        } else {
            const int owner = rigid_owner(mesh, node);
            query_bvh_excluding_rigid_owner(c.tri_bvh_nodes, c.tri_bvh_rigid_owner, c.tri_root, c.node_boxes[node], owner, node_hits[node]);
        }
    }

    // SS candidates: green edge boxes queried against red edge boxes.
    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (c.edge_root < 0) continue;
        if (mode == InitializationMode::Refittable || mode == InitializationMode::DeformableSolver) {
            query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], edge_hits[e]);
        } else {
            const int owner = topo_.edge_rigid_owner[static_cast<std::size_t>(e)];
            query_bvh_excluding_rigid_owner(c.edge_bvh_nodes, c.edge_bvh_rigid_owner, c.edge_root, c.edge_boxes[e], owner, edge_hits[e]);
        }
    }
    if (mode == InitializationMode::Refittable) {
        for (int node = 0; node < nv; ++node) {
            for (const int tri_idx : node_hits[node]) {
                if (accepts_nt_pair(mesh, node, tri_idx, mode))
                    append_nt_pair(c, node, tri_idx, mesh, mode);
            }
        }
        for (int edge = 0; edge < ne; ++edge) {
            for (const int other : edge_hits[edge]) {
                if (accepts_ss_pair(c, mesh, edge, other, mode))
                    append_ss_pair(c, edge, other, mesh, mode);
            }
        }
    } else {
        materialize_solver_pairs(c, mesh, mode);
    }

    cache_ = std::move(c);
}

void BroadPhase::refresh_pairs(const RefMesh& mesh) {
    Cache& c = cache_;
    const int nv = static_cast<int>(c.node_boxes.size());
    const int ne = static_cast<int>(c.edges.size());

    c.nt_pairs.clear();
    c.nt_pair_tri.clear();
    for (auto& v : c.vertex_nt) v.clear();

    c.ss_pairs.clear();
    c.ss_pair_edges.clear();
    for (auto& v : c.vertex_ss) v.clear();

    const std::vector<int>* surface_query_nodes = exclude_tet_interior_nt_queries_ ? &surface_nt_query_nodes(mesh, nv) : nullptr;
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    const int num_nt_query_nodes = exclude_tet_interior_nt_queries_ ? static_cast<int>(surface_query_nodes->size()) : nv;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int query_index = 0; query_index < num_nt_query_nodes; ++query_index) {
        const int node = exclude_tet_interior_nt_queries_ ? (*surface_query_nodes)[static_cast<std::size_t>(query_index)] : query_index;
        if (c.tri_root < 0) continue;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
    }
    for (int node = 0; node < nv; ++node) {
        for (int t : node_hits[node]) {
            const int a  = tri_vertex(mesh, t, 0);
            const int b  = tri_vertex(mesh, t, 1);
            const int cc = tri_vertex(mesh, t, 2);
            if (node_in_triangle(node, a, b, cc)) continue;
            append_nt_pair(c, node, t, mesh, InitializationMode::Refittable);
        }
    }

    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
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
            if (earlier_edge_query_already_reported_pair(c, e, other)) continue;
            append_ss_pair(c, e, other, mesh, InitializationMode::Refittable);
        }
    }
}

void incremental_refresh_vertex(BroadPhase::Cache& c, int vi, const std::vector<Vec3>& x, const RefMesh& mesh, double box_pad, double node_box_radius_padded) {
    if (vi < 0 || vi >= static_cast<int>(c.node_boxes.size())) return;

    const Vec3 r = Vec3::Constant(node_box_radius_padded);
    c.node_boxes[vi] = AABB(x[vi] - r, x[vi] + r);
    refit_bvh_leaf(c.node_bvh_nodes, c.node_leaf_to_node, vi, c.node_boxes[vi]);

    const Vec3 pad = Vec3::Constant(box_pad);
    for (int t : c.node_to_tris[vi]) {
        AABB tb = c.node_boxes[tri_vertex(mesh, t, 0)];
        tb.expand(c.node_boxes[tri_vertex(mesh, t, 1)]);
        tb.expand(c.node_boxes[tri_vertex(mesh, t, 2)]);
        tb.min -= pad; tb.max += pad;
        c.tri_boxes[t] = tb;
        refit_bvh_leaf(c.tri_bvh_nodes, c.tri_leaf_to_node, t, tb);
    }
    
    for (int e : c.node_to_edges[vi]) {
        AABB red = c.node_boxes[c.edges[e][0]];
        red.expand(c.node_boxes[c.edges[e][1]]);
        c.edge_boxes[e] = AABB(red.min - pad, red.max + pad);
        refit_bvh_leaf(c.edge_bvh_nodes, c.edge_leaf_to_node, e, red);
    }
}

void BroadPhase::build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt) {
    constexpr double epsilon_pad = 1.0e-10;  // fp tie-breaker, not a safety pad
    build(
        x, v, mesh, dt, /*node_pad=*/epsilon_pad,
        /*tri_pad=*/epsilon_pad, /*edge_pad=*/epsilon_pad,
        /*exclude_tet_interior_nt_queries=*/false);
}

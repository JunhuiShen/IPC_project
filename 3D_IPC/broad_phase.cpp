#include "broad_phase.h"
#include "contact_scheduling.h"

#include <cmath>
#include <cassert>
#include <memory>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void BroadPhase::discard_separated_contact_incidence(
    const std::vector<unsigned char>& nt_separated,
    const std::vector<unsigned char>& ss_separated,
    bool parallel,
    std::vector<std::vector<int>>* body_nt,
    std::vector<std::vector<int>>* body_ss) {
    if (nt_separated.size() != cache_.nt_pairs.size()
        || ss_separated.size() != cache_.ss_pairs.size()) {
        throw std::invalid_argument("separation flags must match contact pairs");
    }
    if ((body_nt == nullptr) != (body_ss == nullptr)
        || (body_nt && body_nt->size() != body_ss->size())) {
        throw std::invalid_argument("rigid contact incidence must have matching body rows");
    }
    const int count = static_cast<int>(cache_.vertex_nt.size());
    #pragma omp parallel for schedule(static) if(parallel && count >= 128)
    for (int node = 0; node < count; ++node) {
        auto& nt = cache_.vertex_nt[node];
        auto& ss = cache_.vertex_ss[node];
        nt.erase(std::remove_if(nt.begin(), nt.end(), [&](const auto& entry) {
            return nt_separated[entry.pair_index] != 0;
        }), nt.end());
        ss.erase(std::remove_if(ss.begin(), ss.end(), [&](const auto& entry) {
            return ss_separated[entry.pair_index] != 0;
        }), ss.end());
    }
    if (body_nt) {
        const int bodies = static_cast<int>(body_nt->size());
        #pragma omp parallel for schedule(static) if(parallel && bodies >= 8)
        for (int body = 0; body < bodies; ++body) {
            auto& nt = (*body_nt)[body];
            auto& ss = (*body_ss)[body];
            nt.erase(std::remove_if(nt.begin(), nt.end(), [&](int pair) {
                return nt_separated[pair] != 0;
            }), nt.end());
            ss.erase(std::remove_if(ss.begin(), ss.end(), [&](int pair) {
                return ss_separated[pair] != 0;
            }), ss.end());
        }
    }
}

// BVH build / refit / query
namespace {
// Smaller subtrees stay within a worker to amortize task scheduling.
constexpr int bvh_task_grain = 1024;
// Integer scan: each worker scans its own interval, then adds the totals of
// preceding intervals. Only the small worker-total scan is serial.
void scan_contact_offsets(std::vector<std::size_t>& values) {
    if (values.size() < 1024 || omp_get_max_threads() == 1) {
        for (std::size_t i = 1; i < values.size(); ++i) values[i] += values[i - 1];
        return;
    }
    solver_detail::BorrowedContactBuffer<std::size_t, std::allocator<std::size_t>> total_storage;
    total_storage.resize(static_cast<std::size_t>(omp_get_max_threads()));
    auto& totals = total_storage.values();
    std::fill(totals.begin(), totals.end(), 0);
    #pragma omp parallel
    {
        const int worker = omp_get_thread_num();
        const int workers = omp_get_num_threads();
        const std::size_t begin = values.size() * worker / workers;
        const std::size_t end = values.size() * (worker + 1) / workers;
        std::size_t sum = 0;
        for (std::size_t i = begin; i < end; ++i) {
            sum += values[i];
            values[i] = sum;
        }
        totals[worker] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            std::size_t offset = 0;
            for (int i = 0; i < workers; ++i) {
                const std::size_t count = totals[i];
                totals[i] = offset;
                offset += count;
            }
        }
        const std::size_t offset = totals[worker];
        for (std::size_t i = begin; i < end; ++i) values[i] += offset;
    }
}


double point_aabb_squared_distance(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    double distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        // A point can lie outside ordered bounds in only one direction.
        // Keep the original axis-by-axis squared-distance accumulation.
        const double distance = std::max({0.0, lo[axis] - p[axis], p[axis] - hi[axis]});
        distance_squared += distance * distance;
    }
    return distance_squared;
}

// Child slots are assigned from subtree sizes, matching the original serial
// allocation order. Tasks partition disjoint index ranges and never grow out.
void build_bvh_subtree(const std::vector<AABB>& boxes, std::vector<int>& idx,
    std::vector<BVHNode>& out, std::vector<int>* leaf_to_node,
    int node_index, int children_begin, int start, int end,
    const std::vector<int>* leaf_owners, std::vector<int>* owners,
    const std::vector<Vec3>* centroid_sums) {
    AABB node_box;
    for (int i = start; i < end; ++i) node_box.expand(boxes[idx[i]]);
    out[node_index].bbox = node_box;
    // Reused nodes must lose old leaf/child links before this subtree is filled.
    out[node_index].left = -1;
    out[node_index].right = -1;
    out[node_index].leafIndex = -1;
    const int count = end - start;
    if (count == 1) {
        const int leaf = idx[start];
        out[node_index].leafIndex = leaf;
        if (leaf_to_node) (*leaf_to_node)[leaf] = node_index;
        if (owners) (*owners)[node_index] = (*leaf_owners)[leaf];
        return;
    }
    const Vec3 extent = node_box.extent();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() >= extent.z()) axis = 1;
    else if (extent.z() > extent.x() && extent.z() >= extent.y()) axis = 2;
    const int mid = start + count / 2;
    if (centroid_sums) {
        std::nth_element(idx.begin() + start, idx.begin() + mid, idx.begin() + end,
            [&](int a, int b) { return (*centroid_sums)[a][axis] < (*centroid_sums)[b][axis]; });
    } else {
        std::nth_element(idx.begin() + start, idx.begin() + mid, idx.begin() + end,
            [&](int a, int b) {
                return boxes[a].min[axis] + boxes[a].max[axis]
                    < boxes[b].min[axis] + boxes[b].max[axis];
            });
    }
    const int left = children_begin;
    const int right = children_begin + 1;
    const int left_children = children_begin + 2;
    const int right_children = children_begin + 2 * (mid - start);
    out[node_index].left = left;
    out[node_index].right = right;
    out[left].parent = node_index;
    out[right].parent = node_index;
    if (count >= (boxes.size() <= 16384 ? 256 : bvh_task_grain)) {
        #pragma omp taskgroup
        {
            #pragma omp task shared(boxes, idx, out) firstprivate(leaf_to_node, left, left_children, start, mid, leaf_owners, owners, centroid_sums)
            build_bvh_subtree(boxes, idx, out, leaf_to_node, left, left_children, start, mid, leaf_owners, owners, centroid_sums);
            #pragma omp task shared(boxes, idx, out) firstprivate(leaf_to_node, right, right_children, mid, end, leaf_owners, owners, centroid_sums)
            build_bvh_subtree(boxes, idx, out, leaf_to_node, right, right_children, mid, end, leaf_owners, owners, centroid_sums);
        }
    } else {
        build_bvh_subtree(boxes, idx, out, leaf_to_node, left, left_children, start, mid, leaf_owners, owners, centroid_sums);
        build_bvh_subtree(boxes, idx, out, leaf_to_node, right, right_children, mid, end, leaf_owners, owners, centroid_sums);
    }
    if (owners) {
        const int a = (*owners)[left], b = (*owners)[right];
        (*owners)[node_index] = a >= 0 && a == b ? a : -1;
    }
}

inline int build_bvh_impl(const std::vector<AABB>& boxes, std::vector<BVHNode>& out,
    std::vector<int>* leaf_to_node, const std::vector<int>* leaf_owners = nullptr,
    std::vector<int>* owners = nullptr) {
    if (owners) owners->resize(boxes.empty() ? 0 : 2 * boxes.size() - 1);
    if (leaf_to_node) leaf_to_node->assign(boxes.size(), -1);
    if (boxes.empty()) {
        out.clear();
        return -1;
    }
    out.resize(2 * boxes.size() - 1);
    out[0].parent = -1;
    struct BuildScratch {
        std::vector<int> indices;
        std::vector<Vec3> centroids;
    };
    // A loan owns its arrays until all construction tasks join. Concurrent or
    // nested builds receive separate arrays, and every used entry is rewritten.
    solver_detail::BorrowedContactBuffer<BuildScratch, std::allocator<BuildScratch>> build_storage;
    build_storage.resize(1);
    auto& idx = build_storage.values()[0].indices;
    idx.resize(boxes.size());
    // Large builds repeatedly compare the same box centroids. Keep exactly
    // the original min+max sums in compact AoS records, without multiplying
    // by one half or changing tie behavior. Tasks own separate entries.
    auto& centroid_sums = build_storage.values()[0].centroids;
    centroid_sums.resize(boxes.size() >= 16384 ? boxes.size() : 0);
    #pragma omp taskloop shared(boxes, idx, centroid_sums) grainsize(8192) if(omp_in_parallel() && boxes.size() >= 16384)
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
        idx[i] = i;
        if (!centroid_sums.empty())
            for (int axis = 0; axis < 3; ++axis)
                centroid_sums[i][axis] = boxes[i].min[axis] + boxes[i].max[axis];
    }
    const auto* centers = centroid_sums.empty() ? nullptr : &centroid_sums;
    // BroadPhase already builds its trees in parallel sections. Their tasks
    // share that team; standalone callers get a team only for large trees.
    if (!omp_in_parallel() && boxes.size() >= 256) {
        #pragma omp parallel
        {
            #pragma omp single
            build_bvh_subtree(boxes, idx, out, leaf_to_node, 0, 1, 0, static_cast<int>(boxes.size()), leaf_owners, owners, centers);
        }
    } else {
        build_bvh_subtree(boxes, idx, out, leaf_to_node, 0, 1, 0, static_cast<int>(boxes.size()), leaf_owners, owners, centers);
    }
    return 0;
}
}  // namespace

bool node_triangle_aabbs_within_distance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, double distance_squared, bool* aabb_rejected) {
    if (aabb_rejected) *aabb_rejected = false;
    const Vec3 lo = a.cwiseMin(b).cwiseMin(c);
    const Vec3 hi = a.cwiseMax(b).cwiseMax(c);
    if (point_aabb_squared_distance(p, lo, hi) > distance_squared) {
        if (aabb_rejected) *aabb_rejected = true;
        return false;
    }
    const Vec3 normal = (b - a).cross(c - a);
    const double normal_squared = normal.squaredNorm();
    const double scaled_distance = (p - a).dot(normal);
    const double scaled_distance_squared = scaled_distance * scaled_distance;
    constexpr double conservative_roundoff = 1.0 + 1.0e-10;
    return !std::isfinite(scaled_distance_squared) || !std::isfinite(normal_squared) || scaled_distance_squared <= conservative_roundoff * distance_squared * normal_squared;
}

bool segment_aabbs_within_distance(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1, double distance_squared, bool* aabb_rejected) {
    if (aabb_rejected) *aabb_rejected = false;
    const Vec3 alo = a0.cwiseMin(a1);
    const Vec3 ahi = a0.cwiseMax(a1);
    const Vec3 blo = b0.cwiseMin(b1);
    const Vec3 bhi = b0.cwiseMax(b1);
    double aabb_distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        // Ordered bounds can have a positive gap in at most one direction.
        // Preserve the original axis-by-axis squared-distance accumulation.
        const double distance = std::max({
            0.0, blo[axis] - ahi[axis], alo[axis] - bhi[axis]});
        aabb_distance_squared += distance * distance;
    }
    if (aabb_distance_squared > distance_squared) {
        if (aabb_rejected) *aabb_rejected = true;
        return false;
    }
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

void query_bvh_tile(const std::vector<BVHNode>& nodes, int root,
    const AABB* queries, std::size_t count, std::vector<int>* const* hits) {
    assert(count <= bvh_query_tile_width);
    if (root < 0 || count == 0) return;
#if defined(__AVX512F__)
    // The organizer gathers AoS boxes; only this small local tile is SoA.
    // Shared tree bounds are broadcast across independent query lanes.
    alignas(64) double lo[3][bvh_query_tile_width], hi[3][bvh_query_tile_width];
    for (std::size_t lane = 0; lane < bvh_query_tile_width; ++lane) {
        const AABB& query = queries[std::min(lane, count - 1)];
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis][lane] = query.min[axis];
            hi[axis][lane] = query.max[axis];
        }
    }
    const __m512d qlo[] = {_mm512_load_pd(lo[0]), _mm512_load_pd(lo[1]), _mm512_load_pd(lo[2])};
    const __m512d qhi[] = {_mm512_load_pd(hi[0]), _mm512_load_pd(hi[1]), _mm512_load_pd(hi[2])};
    struct Visit { int node; unsigned lanes; };
    Visit stack[256];
    int top = 0;
    stack[top++] = {root, (1u << count) - 1u};
    while (top > 0) {
        const Visit visit = stack[--top];
        const BVHNode& node = nodes[visit.node];
        unsigned lanes = visit.lanes;
        for (int axis = 0; axis < 3; ++axis) {
            // Ordered inclusive comparisons also match NaN and touching boxes.
            lanes &= _mm512_cmp_pd_mask(_mm512_set1_pd(node.bbox.min[axis]), qhi[axis], _CMP_LE_OQ);
            lanes &= _mm512_cmp_pd_mask(_mm512_set1_pd(node.bbox.max[axis]), qlo[axis], _CMP_GE_OQ);
        }
        if (!lanes) continue;
        if (node.leafIndex >= 0) {
            while (lanes) {
                const int lane = __builtin_ctz(lanes);
                lanes &= lanes - 1u;
                hits[lane]->push_back(node.leafIndex);
            }
        } else {
            // Right-first DFS preserves each lane's original scalar hit order.
            stack[top++] = {node.left, lanes};
            stack[top++] = {node.right, lanes};
        }
    }
#else
    for (std::size_t lane = 0; lane < count; ++lane)
        query_bvh(nodes, root, queries[lane], *hits[lane]);
#endif
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
        const int nt = num_tris(mesh);
        out_node_to_tris.resize(nv);
        out_node_to_edges.resize(nv);
        std::vector<int> counts(nv, 0);
        #pragma omp parallel for schedule(static) if(nt >= 128)
        for (int t = 0; t < nt; ++t) {
            for (int role = 0; role < 3; ++role) {
                const int node = tri_vertex(mesh, t, role);
                #pragma omp atomic update
                ++counts[node];
            }
        }
        #pragma omp parallel for schedule(static) if(nv >= 128)
        for (int node = 0; node < nv; ++node) {
            out_node_to_tris[node].resize(counts[node]);
            counts[node] = 0;
        }
        #pragma omp parallel for schedule(static) if(nt >= 128)
        for (int t = 0; t < nt; ++t) {
            for (int role = 0; role < 3; ++role) {
                const int node = tri_vertex(mesh, t, role);
                int slot;
                #pragma omp atomic capture
                slot = counts[node]++;
                out_node_to_tris[node][slot] = t;
            }
        }

        // Each edge is owned by its smaller endpoint. Record its first
        // triangle/role occurrence, preserving the legacy edge numbering.
        std::vector<std::vector<std::pair<int, int>>> owned_edges(nv);
        std::vector<std::size_t> edge_offsets(static_cast<std::size_t>(3) * nt + 1, 0);
        #pragma omp parallel for schedule(dynamic, 16) if(nv >= 128)
        for (int node = 0; node < nv; ++node) {
            auto& triangles = out_node_to_tris[node];
            std::sort(triangles.begin(), triangles.end());
            triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
            auto& edges = owned_edges[node];
            for (const int t : triangles) {
                for (int role = 0; role < 3; ++role) {
                    const Edge edge = canonical_edge(tri_vertex(mesh, t, role), tri_vertex(mesh, t, (role + 1) % 3));
                    if (edge.v0 == node) edges.emplace_back(edge.v1, 3 * t + role);
                }
            }
            std::sort(edges.begin(), edges.end());
            int previous = -1;
            for (const auto& [neighbor, occurrence] : edges) {
                if (neighbor != previous) edge_offsets[occurrence + 1] = 1;
                previous = neighbor;
            }
        }
        scan_contact_offsets(edge_offsets);
        out_edges.resize(edge_offsets.back());
        std::vector<int> occurrence_to_edge(static_cast<std::size_t>(3) * nt);
        #pragma omp parallel for schedule(dynamic, 16) if(nv >= 128)
        for (int node = 0; node < nv; ++node) {
            int previous = -1, edge_id = -1;
            for (const auto& [neighbor, occurrence] : owned_edges[node]) {
                if (neighbor != previous) {
                    edge_id = static_cast<int>(edge_offsets[occurrence]);
                    out_edges[edge_id] = {node, neighbor};
                }
                occurrence_to_edge[occurrence] = edge_id;
                previous = neighbor;
            }
        }
        #pragma omp parallel for schedule(dynamic, 16) if(nv >= 128)
        for (int node = 0; node < nv; ++node) {
            auto& edges = out_node_to_edges[node];
            edges.clear();
            for (const int t : out_node_to_tris[node]) {
                for (int role = 0; role < 3; ++role) {
                    if (tri_vertex(mesh, t, role) == node || tri_vertex(mesh, t, (role + 1) % 3) == node)
                        edges.push_back(occurrence_to_edge[3 * t + role]);
                }
            }
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

    template <typename Count>
    static void resize_incidence_rows(
        std::vector<std::vector<BroadPhase::Cache::VertexPairEntry>>& rows,
        Count* counts, const std::size_t nv, const int num_workers) {
        // Scan contiguous blocks from each worker row, retaining worker order
        // at every node so the incidence lists remain sorted by pair index.
        constexpr std::size_t block_size = 256;
        const std::size_t blocks = (nv + block_size - 1) / block_size;
        #pragma omp parallel for schedule(static) if(nv >= 128)
        for (std::size_t block = 0; block < blocks; ++block) {
            const std::size_t begin = block * block_size;
            const std::size_t length = std::min(block_size, nv - begin);
            Count offsets[block_size]{};
            for (int worker = 0; worker < num_workers; ++worker) {
                Count* row = counts + static_cast<std::size_t>(worker) * nv + begin;
                for (std::size_t local = 0; local < length; ++local) {
                    const Count count = row[local];
                    row[local] = offsets[local];
                    offsets[local] += count;
                }
            }
            for (std::size_t local = 0; local < length; ++local)
                rows[begin + local].resize(offsets[local]);
        }
    }

    template <typename Count>
    static void build_solver_vertex_incidence_impl(BroadPhase::Cache& cache, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        if (mode == BroadPhase::InitializationMode::RigidSolver || cache.vertex_nt.empty())
            return;

        const std::size_t nv = cache.vertex_nt.size();
        int num_workers = 1;
        #ifdef _OPENMP
        num_workers = std::max(1, omp_get_max_threads());
        #endif
        const std::size_t count_size = static_cast<std::size_t>(num_workers) * nv;
        // Each worker initializes its own row immediately before counting.
        // A caller owns its scratch; OpenMP workers share only the captured
        // pointer. Reuse avoids repeatedly mapping a large counter matrix.
        struct Scratch {
            std::unique_ptr<Count[]> data;
            std::size_t capacity = 0;
        };
        static thread_local Scratch scratch;
        if (scratch.capacity < count_size) {
            scratch.data.reset(new Count[count_size]);
            scratch.capacity = count_size;
        }
        Count* const counts = scratch.data.get();

        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            Count* worker_counts = counts + static_cast<std::size_t>(worker) * nv;
            std::fill_n(worker_counts, nv, 0);
            const std::size_t begin = cache.nt_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.nt_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const NodeTrianglePair& pair = cache.nt_pairs[pair_index];
                if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.node) < 0) ++worker_counts[static_cast<std::size_t>(pair.node)];
                for (int role = 0; role < 3; ++role) if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.tri_v[role]) < 0) ++worker_counts[static_cast<std::size_t>(pair.tri_v[role])];
            }
        }
        resize_incidence_rows(cache.vertex_nt, counts, nv, num_workers);
        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            Count* worker_offsets = counts + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.nt_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.nt_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const NodeTrianglePair& pair = cache.nt_pairs[pair_index];
                if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.node) < 0) cache.vertex_nt[static_cast<std::size_t>(pair.node)][worker_offsets[static_cast<std::size_t>(pair.node)]++] = {pair_index, 0};
                for (int role = 0; role < 3; ++role) if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.tri_v[role]) < 0) cache.vertex_nt[static_cast<std::size_t>(pair.tri_v[role])][worker_offsets[static_cast<std::size_t>(pair.tri_v[role])]++] = {pair_index, role + 1};
            }
        }

        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            Count* worker_counts = counts + static_cast<std::size_t>(worker) * nv;
            std::fill_n(worker_counts, nv, 0);
            const std::size_t begin = cache.ss_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.ss_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const SegmentSegmentPair& pair = cache.ss_pairs[pair_index];
                for (int role = 0; role < 4; ++role) if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.v[role]) < 0) ++worker_counts[static_cast<std::size_t>(pair.v[role])];
            }
        }
        resize_incidence_rows(cache.vertex_ss, counts, nv, num_workers);
        #pragma omp parallel for schedule(static, 1)
        for (int worker = 0; worker < num_workers; ++worker) {
            Count* worker_offsets = counts + static_cast<std::size_t>(worker) * nv;
            const std::size_t begin = cache.ss_pairs.size() * static_cast<std::size_t>(worker) / static_cast<std::size_t>(num_workers);
            const std::size_t end = cache.ss_pairs.size() * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(num_workers);
            for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                const SegmentSegmentPair& pair = cache.ss_pairs[pair_index];
                for (int role = 0; role < 4; ++role) if (mode != BroadPhase::InitializationMode::GeneralSolver || rigid_owner(mesh, pair.v[role]) < 0) cache.vertex_ss[static_cast<std::size_t>(pair.v[role])][worker_offsets[static_cast<std::size_t>(pair.v[role])]++] = {pair_index, role};
            }
        }
    }

    static void build_solver_vertex_incidence(BroadPhase::Cache& cache, const RefMesh& mesh, const BroadPhase::InitializationMode mode) {
        // A vertex can receive at most four entries per pair, including
        // degenerate repeated roles. Keep the wide path for larger inputs.
        constexpr std::size_t narrow_limit =
            std::numeric_limits<std::uint32_t>::max() / 4;
        if (std::max(cache.nt_pairs.size(), cache.ss_pairs.size()) <= narrow_limit)
            build_solver_vertex_incidence_impl<std::uint32_t>(cache, mesh, mode);
        else
            build_solver_vertex_incidence_impl<std::size_t>(cache, mesh, mode);
    }

    static void materialize_solver_pairs(BroadPhase::Cache& cache, const RefMesh& mesh, const BroadPhase::InitializationMode mode, bool forward_edges_only = false, bool retain_solver_data = true) {
        const int nv = static_cast<int>(cache.node_hits.size());
        const int ne = static_cast<int>(cache.edge_hits.size());
        const auto accept_edge_pair = [&](int edge, int other) {
            // Velocity-based builds historically emit only forward edge hits;
            // box-based builds/refits retain asymmetric hits in either direction.
            if (forward_edges_only) {
                if (other <= edge) return false;
                return !share_vertex({cache.edges[edge][0], cache.edges[edge][1]},
                    {cache.edges[other][0], cache.edges[other][1]});
            }
            return accepts_ss_pair(cache, mesh, edge, other, mode);
        };

        // Each query owns one contiguous output interval. Parallel counting,
        // a deterministic prefix sum, and parallel writes preserve the exact
        // legacy pair order without a serial push_back bottleneck.
        struct PairOffsetScratch {
            std::vector<std::size_t> node_triangle, segment_segment;
        };
        // Keep both arrays in one loan: their lifetimes overlap, so borrowing
        // the same scalar-vector pool twice would discard one cached capacity.
        solver_detail::BorrowedContactBuffer<PairOffsetScratch,
            std::allocator<PairOffsetScratch>> offset_storage;
        offset_storage.resize(1);
        auto& nt_offsets = offset_storage.values()[0].node_triangle;
        nt_offsets.assign(static_cast<std::size_t>(nv) + 1, 0);
        #pragma omp parallel for schedule(dynamic, 32)
        for (int node = 0; node < nv; ++node) {
            std::size_t count = 0;
            for (const int tri_idx : cache.node_hits[node])
                count += accepts_nt_pair(mesh, node, tri_idx, mode);
            nt_offsets[static_cast<std::size_t>(node) + 1] = count;
        }
        scan_contact_offsets(nt_offsets);
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

        // Large cloth rebuilds otherwise repeat each edge-hit predicate while
        // writing pairs, including the reverse-query AABB test. Retain only
        // one acceptance bit per raw hit; raw query rows and pair order stay
        // unchanged. The forward-only predictor has a cheaper predicate and
        // keeps its existing traversal.
        struct AcceptanceScratch {
            std::vector<std::size_t> offsets;
            std::vector<std::uint64_t> words;
        };
        static thread_local AcceptanceScratch acceptance_scratch;
        const bool retain_acceptance = !forward_edges_only
            && mode == BroadPhase::InitializationMode::DeformableSolver && ne >= 4096;
        auto& acceptance_offsets = acceptance_scratch.offsets;
        auto& acceptance_words = acceptance_scratch.words;
        if (retain_acceptance) {
            acceptance_offsets.resize(static_cast<std::size_t>(ne) + 1);
            acceptance_offsets[0] = 0;
            #pragma omp parallel for schedule(static)
            for (int edge = 0; edge < ne; ++edge)
                acceptance_offsets[static_cast<std::size_t>(edge) + 1]
                    = (cache.edge_hits[edge].size() + 63) / 64;
            scan_contact_offsets(acceptance_offsets);
            acceptance_words.resize(acceptance_offsets.back());
        }

        auto& ss_offsets = offset_storage.values()[0].segment_segment;
        ss_offsets.assign(static_cast<std::size_t>(ne) + 1, 0);
        #pragma omp parallel for schedule(dynamic, 32)
        for (int edge = 0; edge < ne; ++edge) {
            std::size_t count = 0;
            const auto& hits = cache.edge_hits[edge];
            if (retain_acceptance) {
                for (std::size_t begin = 0; begin < hits.size(); begin += 64) {
                    std::uint64_t bits = 0;
                    const std::size_t length = std::min(std::size_t(64), hits.size() - begin);
                    for (std::size_t lane = 0; lane < length; ++lane) {
                        const bool accepted = accept_edge_pair(edge, hits[begin + lane]);
                        bits |= std::uint64_t(accepted) << lane;
                        count += accepted;
                    }
                    acceptance_words[acceptance_offsets[edge] + begin / 64] = bits;
                }
            } else {
                for (const int other : hits) count += accept_edge_pair(edge, other);
            }
            ss_offsets[static_cast<std::size_t>(edge) + 1] = count;
        }
        scan_contact_offsets(ss_offsets);
        cache.ss_pairs.resize(ss_offsets.back());
        cache.ss_pair_edges.resize(ss_offsets.back());
        #pragma omp parallel for schedule(dynamic, 32)
        for (int edge = 0; edge < ne; ++edge) {
            std::size_t pair_index = ss_offsets[static_cast<std::size_t>(edge)];
            const auto& hits = cache.edge_hits[edge];
            if (retain_acceptance) {
                for (std::size_t word = acceptance_offsets[edge]; word < acceptance_offsets[edge + 1]; ++word) {
                    std::uint64_t bits = acceptance_words[word];
                    const std::size_t begin = 64 * (word - acceptance_offsets[edge]);
                    while (bits) {
                        const int lane = __builtin_ctzll(bits);
                        bits &= bits - 1;
                        write_ss_pair(cache, pair_index++, edge, hits[begin + lane]);
                    }
                }
            } else {
                for (const int other : hits) {
                    if (accept_edge_pair(edge, other))
                        write_ss_pair(cache, pair_index++, edge, other);
                }
            }
        }

        if (retain_solver_data)
            build_solver_vertex_incidence(cache, mesh, mode);
    }

    // Keep arrays that the rebuild fully overwrites, avoiding serial value
    // initialization on every frame. Clear optional data that may be omitted.
    static BroadPhase::Cache take_reusable_cache(BroadPhase::Cache& old_cache, const int nv,
        const BroadPhase::InitializationMode mode, bool overwrite_incidence = true) {
        BroadPhase::Cache c = std::move(old_cache);

        c.node_bvh_nodes.clear();
        c.tri_leaf_to_node.clear();
        c.edge_leaf_to_node.clear();
        c.node_leaf_to_node.clear();
        c.tri_bvh_rigid_owner.clear();
        c.edge_bvh_rigid_owner.clear();

        c.node_root = -1;
        c.tri_root = -1;
        c.edge_root = -1;

        if (mode == BroadPhase::InitializationMode::RigidSolver) {
            c.vertex_nt.clear();
            c.vertex_ss.clear();
        } else {
            if (static_cast<int>(c.vertex_nt.size()) == nv) {
                if (!overwrite_incidence) {
                    #pragma omp parallel for schedule(static) if(nv >= 128)
                    for (int node = 0; node < nv; ++node) c.vertex_nt[node].clear();
                }
            } else {
                c.vertex_nt.assign(nv, {});
            }
            if (static_cast<int>(c.vertex_ss.size()) == nv) {
                if (!overwrite_incidence) {
                    #pragma omp parallel for schedule(static) if(nv >= 128)
                    for (int node = 0; node < nv; ++node) c.vertex_ss[node].clear();
                }
            } else {
                c.vertex_ss.assign(nv, {});
            }
        }
        return c;
    }

    static std::vector<std::vector<int>>& prepare_hit_rows(std::vector<std::vector<int>>& rows, int n) {
        if (static_cast<int>(rows.size()) == n) {
            #pragma omp parallel for schedule(static) if(n >= 128)
            for (int row = 0; row < n; ++row) rows[row].clear();
        } else {
            rows.assign(n, {});
        }
        return rows;
    }

    static void query_bvh_rows(const std::vector<BVHNode>& tree, int root,
        const std::vector<AABB>& boxes, std::vector<std::vector<int>>& hits,
        const std::vector<int>* query_nodes = nullptr) {
        if (root < 0) return;
        const int count = static_cast<int>(query_nodes ? query_nodes->size() : boxes.size());
#if defined(__AVX512F__)
        // Gather and output ownership stay outside the local SIMD transpose.
        // Four tiles per claim retain the previous 32-query dispatch grain.
        constexpr int width = static_cast<int>(bvh_query_tile_width);
        const int tiles = count / width + (count % width != 0);
        #pragma omp parallel for schedule(dynamic, 4)
        for (int tile = 0; tile < tiles; ++tile) {
            const int begin = tile * width;
            const int size = std::min(width, count - begin);
            std::array<AABB, bvh_query_tile_width> gathered;
            std::array<std::vector<int>*, bvh_query_tile_width> outputs;
            for (int lane = 0; lane < size; ++lane) {
                const int node = query_nodes ? (*query_nodes)[begin + lane] : begin + lane;
                gathered[lane] = boxes[node];
                outputs[lane] = &hits[node];
            }
            query_bvh_tile(tree, root, gathered.data(), size, outputs.data());
        }
#else
        #pragma omp parallel for schedule(dynamic, 32)
        for (int index = 0; index < count; ++index) {
            const int node = query_nodes ? (*query_nodes)[index] : index;
            query_bvh(tree, root, boxes[node], hits[node]);
        }
#endif
    }

}

// Broad phase
void BroadPhase::invalidate_mesh_topology() {
    topology_valid_ = false;
    topo_.surface_nt_query_nodes_valid = false;
    cache_.edges.clear();
    cache_.node_to_edges.clear();
    cache_.node_to_tris.clear();
}

void BroadPhase::set_mesh_topology(const RefMesh& mesh, int nv) {
    build_unique_edges_and_adjacency(mesh, nv, topo_.edges, topo_.node_to_edges, topo_.node_to_tris);
    topo_.tri_rigid_owner.resize(static_cast<std::size_t>(num_tris(mesh)));
    #pragma omp parallel for schedule(static) if(num_tris(mesh) >= 128)
    for (int tri_idx = 0; tri_idx < num_tris(mesh); ++tri_idx) {
        const int owner = common_rigid_owner(mesh, tri_vertex(mesh, tri_idx, 0), tri_vertex(mesh, tri_idx, 1));
        topo_.tri_rigid_owner[static_cast<std::size_t>(tri_idx)] = owner >= 0 && rigid_owner(mesh, tri_vertex(mesh, tri_idx, 2)) == owner ? owner : -1;
    }
    topo_.edge_rigid_owner.resize(topo_.edges.size());
    #pragma omp parallel for schedule(static) if(topo_.edges.size() >= 128)
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
    double edge_pad, const bool exclude_tet_interior_nt_queries,
    const bool retain_solver_data) {
    const int nv = static_cast<int>(x.size());
    const int nt = num_tris(mesh);
    exclude_tet_interior_nt_queries_ = exclude_tet_interior_nt_queries;

    constexpr InitializationMode mode = InitializationMode::Refittable;
    Cache c = take_reusable_cache(cache_, nv, mode, retain_solver_data);
    c.excludes_tet_interior_nt_queries = exclude_tet_interior_nt_queries;

    if (!topology_valid_) set_mesh_topology(mesh, nv);
    if (c.edges.empty()) c.edges = topo_.edges;
    if (c.node_to_edges.empty()) c.node_to_edges = topo_.node_to_edges;
    if (c.node_to_tris.empty()) c.node_to_tris = topo_.node_to_tris;
    const int ne = static_cast<int>(c.edges.size());
    const std::vector<int>* surface_query_nodes = exclude_tet_interior_nt_queries ? &surface_nt_query_nodes(mesh, nv) : nullptr;

    c.node_boxes.resize(nv);
    #pragma omp parallel for schedule(static) if(nv >= 128)
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
    #pragma omp parallel for schedule(static) if(ne >= 128)
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
        { if (retain_solver_data) c.node_root = build_bvh(c.node_boxes, c.node_bvh_nodes); }
    }

    // Parallel queries and ordered pair materialization.
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    query_bvh_rows(c.tri_bvh_nodes, c.tri_root, c.node_boxes, node_hits, surface_query_nodes);

    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
    query_bvh_rows(c.edge_bvh_nodes, c.edge_root, c.edge_boxes, edge_hits);
    materialize_solver_pairs(c, mesh, mode, /*forward_edges_only=*/true,
                             retain_solver_data);

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

void BroadPhase::initialize_node_boxes_only(const std::vector<AABB>& vertex_boxes) {
    Cache c = take_reusable_cache(
        cache_, static_cast<int>(vertex_boxes.size()),
        InitializationMode::DeformableSolver, /*overwrite_incidence=*/false);
    // This path consumes only node boxes; erase all other retained data.
    c.tri_boxes.clear();
    c.edge_boxes.clear();
    c.tri_bvh_nodes.clear();
    c.edge_bvh_nodes.clear();
    c.nt_pairs.clear();
    c.ss_pairs.clear();
    c.nt_pair_tri.clear();
    c.ss_pair_edges.clear();
    c.excludes_tet_interior_nt_queries = false;
    c.node_boxes = vertex_boxes;
    c.node_hits.clear();
    c.edge_hits.clear();
    c.red_edge_boxes.clear();
    exclude_tet_interior_nt_queries_ = false;
    cache_ = std::move(c);
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
    c.node_boxes.resize(vertex_boxes.size());
    #pragma omp parallel for schedule(static) if(nv >= 128)
    for (int node = 0; node < nv; ++node) c.node_boxes[node] = vertex_boxes[node];

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
                c.tri_root = mode == InitializationMode::DeformableSolver
                    ? build_bvh(c.tri_boxes, c.tri_bvh_nodes)
                    : build_bvh_impl(c.tri_boxes, c.tri_bvh_nodes, nullptr,
                        &topo_.tri_rigid_owner, &c.tri_bvh_rigid_owner);
            }
            #pragma omp section
            {
                c.edge_root = mode == InitializationMode::DeformableSolver
                    ? build_bvh(red_edge_boxes, c.edge_bvh_nodes)
                    : build_bvh_impl(red_edge_boxes, c.edge_bvh_nodes, nullptr,
                        &topo_.edge_rigid_owner, &c.edge_bvh_rigid_owner);
            }
        }
    }

    // NT candidates: blue node boxes queried against green triangle boxes.
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    const int num_nt_query_nodes = exclude_tet_interior_nt_queries ? static_cast<int>(surface_query_nodes->size()) : nv;
    if (mode == InitializationMode::Refittable || mode == InitializationMode::DeformableSolver) {
        query_bvh_rows(c.tri_bvh_nodes, c.tri_root, c.node_boxes, node_hits, surface_query_nodes);
    } else {
        #pragma omp parallel for schedule(dynamic, 32)
        for (int query_index = 0; query_index < num_nt_query_nodes; ++query_index) {
            const int node = exclude_tet_interior_nt_queries ? (*surface_query_nodes)[static_cast<std::size_t>(query_index)] : query_index;
            const int owner = rigid_owner(mesh, node);
            query_bvh_excluding_rigid_owner(c.tri_bvh_nodes, c.tri_bvh_rigid_owner, c.tri_root, c.node_boxes[node], owner, node_hits[node]);
        }
    }

    // SS candidates: green edge boxes queried against red edge boxes.
    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
    if (mode == InitializationMode::Refittable || mode == InitializationMode::DeformableSolver) {
        query_bvh_rows(c.edge_bvh_nodes, c.edge_root, c.edge_boxes, edge_hits);
    } else {
        #pragma omp parallel for schedule(dynamic, 32)
        for (int e = 0; e < ne; ++e) {
            const int owner = topo_.edge_rigid_owner[static_cast<std::size_t>(e)];
            query_bvh_excluding_rigid_owner(c.edge_bvh_nodes, c.edge_bvh_rigid_owner, c.edge_root, c.edge_boxes[e], owner, edge_hits[e]);
        }
    }
    materialize_solver_pairs(c, mesh, mode);

    cache_ = std::move(c);
}

void BroadPhase::refit_boxes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, const double box_pad) {
    Cache& c = cache_;
    const int nv = static_cast<int>(c.node_boxes.size());
    const int nt = num_tris(mesh);
    const int ne = static_cast<int>(c.edges.size());
    if (!topology_valid_
        || c.node_leaf_to_node.size() != c.node_boxes.size()
        || c.tri_leaf_to_node.size() != c.tri_boxes.size()
        || c.edge_leaf_to_node.size() != c.edges.size()
        || c.red_edge_boxes.size() != c.edges.size())
        throw std::logic_error("BroadPhase::refit_boxes requires box-based Refittable initialization");
    if (vertex_boxes.size() != c.node_boxes.size()
        || static_cast<std::size_t>(nt) != c.tri_boxes.size())
        throw std::invalid_argument("BroadPhase::refit_boxes requires unchanged mesh sizes");

    const Vec3 pad = Vec3::Constant(box_pad);
    #pragma omp parallel if(nv >= 128)
    {
        #pragma omp for schedule(static)
        for (int node = 0; node < nv; ++node)
            c.node_boxes[node] = vertex_boxes[node];

        #pragma omp for schedule(static)
        for (int t = 0; t < nt; ++t) {
            AABB box = c.node_boxes[tri_vertex(mesh, t, 0)];
            box.expand(c.node_boxes[tri_vertex(mesh, t, 1)]);
            box.expand(c.node_boxes[tri_vertex(mesh, t, 2)]);
            box.min -= pad;
            box.max += pad;
            c.tri_boxes[t] = box;
        }

        #pragma omp for schedule(static)
        for (int e = 0; e < ne; ++e) {
            AABB red = c.node_boxes[c.edges[e][0]];
            red.expand(c.node_boxes[c.edges[e][1]]);
            c.red_edge_boxes[e] = red;
            c.edge_boxes[e] = AABB(red.min - pad, red.max + pad);
        }

        // Each tree is independent; its reverse traversal visits children
        // before parents while retaining the existing query order.
        #pragma omp sections
        {
            #pragma omp section
            { refit_bvh(c.node_bvh_nodes, c.node_boxes); }
            #pragma omp section
            { refit_bvh(c.tri_bvh_nodes, c.tri_boxes); }
            #pragma omp section
            { refit_bvh(c.edge_bvh_nodes, c.red_edge_boxes); }
        }
    }
}

void BroadPhase::refresh_pairs(const RefMesh& mesh) {
    Cache& c = cache_;
    const int nv = static_cast<int>(c.node_boxes.size());
    const int ne = static_cast<int>(c.edges.size());

    c.nt_pairs.clear();
    c.nt_pair_tri.clear();

    c.ss_pairs.clear();
    c.ss_pair_edges.clear();

    const std::vector<int>* surface_query_nodes = exclude_tet_interior_nt_queries_ ? &surface_nt_query_nodes(mesh, nv) : nullptr;
    std::vector<std::vector<int>>& node_hits = prepare_hit_rows(c.node_hits, nv);
    const int num_nt_query_nodes = exclude_tet_interior_nt_queries_ ? static_cast<int>(surface_query_nodes->size()) : nv;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int query_index = 0; query_index < num_nt_query_nodes; ++query_index) {
        const int node = exclude_tet_interior_nt_queries_ ? (*surface_query_nodes)[static_cast<std::size_t>(query_index)] : query_index;
        if (c.tri_root < 0) continue;
        query_bvh(c.tri_bvh_nodes, c.tri_root, c.node_boxes[node], node_hits[node]);
    }

    std::vector<std::vector<int>>& edge_hits = prepare_hit_rows(c.edge_hits, ne);
    #pragma omp parallel for schedule(dynamic, 32)
    for (int e = 0; e < ne; ++e) {
        if (c.edge_root < 0) continue;
        query_bvh(c.edge_bvh_nodes, c.edge_root, c.edge_boxes[e], edge_hits[e]);
    }
    materialize_solver_pairs(c, mesh, InitializationMode::Refittable);
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
        if (e < static_cast<int>(c.red_edge_boxes.size()))
            c.red_edge_boxes[e] = red;
        c.edge_boxes[e] = AABB(red.min - pad, red.max + pad);
        refit_bvh_leaf(c.edge_bvh_nodes, c.edge_leaf_to_node, e, red);
    }
}

void BroadPhase::build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, bool retain_solver_data) {
    constexpr double epsilon_pad = 1.0e-10;  // fp tie-breaker, not a safety pad
    build(
        x, v, mesh, dt, /*node_pad=*/epsilon_pad,
        /*tri_pad=*/epsilon_pad, /*edge_pad=*/epsilon_pad,
        /*exclude_tet_interior_nt_queries=*/false, retain_solver_data);
}

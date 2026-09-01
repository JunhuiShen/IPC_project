#include "broad_phase.h"
#include "physics.h"
#include "make_shape.h"
#include "node_triangle_distance.h"
#include "safe_step.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {

    struct EdgeKey {
        int a = -1;
        int b = -1;

        EdgeKey() = default;

        EdgeKey(int i, int j) {
            if (i < j) {
                a = i;
                b = j;
            } else {
                a = j;
                b = i;
            }
        }

        bool operator<(const EdgeKey& other) const {
            return std::tie(a, b) < std::tie(other.a, other.b);
        }

        bool operator==(const EdgeKey& other) const {
            return a == other.a && b == other.b;
        }
    };

    struct NTPairKey {
        int node = -1;
        int a = -1;
        int b = -1;
        int c = -1;

        bool operator<(const NTPairKey& other) const {
            return std::tie(node, a, b, c) < std::tie(other.node, other.a, other.b, other.c);
        }

        bool operator==(const NTPairKey& other) const {
            return node == other.node && a == other.a && b == other.b && c == other.c;
        }
    };

    struct SSPairKey {
        EdgeKey e0;
        EdgeKey e1;

        bool operator<(const SSPairKey& other) const {
            return std::tie(e0.a, e0.b, e1.a, e1.b) < std::tie(other.e0.a, other.e0.b, other.e1.a, other.e1.b);
        }

        bool operator==(const SSPairKey& other) const {
            return e0 == other.e0 && e1 == other.e1;
        }
    };

    struct PairSets {
        std::set<NTPairKey> nt;
        std::set<SSPairKey> ss;
    };

    static RefMesh make_mesh(const std::vector<Vec3>& x, const std::vector<std::array<int, 3>>& tris_in) {
        RefMesh mesh;
        mesh.num_positions = x.size();

        for (const auto& t : tris_in) {
            mesh.tris.push_back(t[0]);
            mesh.tris.push_back(t[1]);
            mesh.tris.push_back(t[2]);
        }

        return mesh;
    }

    static bool share_vertex(const EdgeKey& e0, const EdgeKey& e1) {
        return e0.a == e1.a || e0.a == e1.b || e0.b == e1.a || e0.b == e1.b;
    }

    static NTPairKey make_nt_key(int node, int a, int b, int c) {
        std::array<int, 3> tri = {a, b, c};
        std::sort(tri.begin(), tri.end());
        return {node, tri[0], tri[1], tri[2]};
    }

    static SSPairKey make_ss_key(int a0, int a1, int b0, int b1) {
        EdgeKey e0(a0, a1);
        EdgeKey e1(b0, b1);
        if (e1 < e0) std::swap(e0, e1);
        return {e0, e1};
    }

    static bool nt_pair_matches(const NodeTrianglePair& p, int node, int a, int b, int c) {
        return make_nt_key(p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]) == make_nt_key(node, a, b, c);
    }

    static bool contains_nt_pair(const std::vector<NodeTrianglePair>& pairs, int node, int a, int b, int c) {
        for (const auto& p : pairs) {
            if (nt_pair_matches(p, node, a, b, c)) return true;
        }
        return false;
    }

    static bool ss_pair_matches(const SegmentSegmentPair& p, EdgeKey e0, EdgeKey e1) {
        return make_ss_key(p.v[0], p.v[1], p.v[2], p.v[3]) == make_ss_key(e0.a, e0.b, e1.a, e1.b);
    }

    static bool contains_ss_pair(const std::vector<SegmentSegmentPair>& pairs, EdgeKey e0, EdgeKey e1) {
        for (const auto& p : pairs) {
            if (ss_pair_matches(p, e0, e1)) return true;
        }
        return false;
    }

    static std::vector<EdgeKey> build_unique_edges_ref(const RefMesh& mesh) {
        std::set<EdgeKey> edge_set;
        const int nt = num_tris(mesh);
        for (int t = 0; t < nt; ++t) {
            edge_set.insert(EdgeKey(tri_vertex(mesh, t, 0), tri_vertex(mesh, t, 1)));
            edge_set.insert(EdgeKey(tri_vertex(mesh, t, 1), tri_vertex(mesh, t, 2)));
            edge_set.insert(EdgeKey(tri_vertex(mesh, t, 2), tri_vertex(mesh, t, 0)));
        }
        return std::vector<EdgeKey>(edge_set.begin(), edge_set.end());
    }

    static AABB build_node_box_ref(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int node, double dt, double pad) {
        AABB box;
        box.expand(x[node]);
        box.expand(x[node] + dt * v[node]);
        box.min.array() -= pad;
        box.max.array() += pad;
        return box;
    }

    static AABB build_triangle_box_ref(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, int c, double dt, double pad) {
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

    static AABB build_edge_box_ref(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, double dt, double pad) {
        AABB box;
        box.expand(x[a]);
        box.expand(x[b]);
        box.expand(x[a] + dt * v[a]);
        box.expand(x[b] + dt * v[b]);
        box.min.array() -= pad;
        box.max.array() += pad;
        return box;
    }

    static PairSets pair_sets_from_vectors(const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
        PairSets out;
        for (const auto& p : nt_pairs) {
            out.nt.insert(make_nt_key(p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]));
        }
        for (const auto& p : ss_pairs) {
            out.ss.insert(make_ss_key(p.v[0], p.v[1], p.v[2], p.v[3]));
        }
        return out;
    }

    static PairSets pair_sets_from_broad(const BroadPhase& broad) {
        return pair_sets_from_vectors(broad.nt_pairs(), broad.ss_pairs());
    }

    static PairSets brute_force_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad) {
        PairSets out;
        const int nv = static_cast<int>(x.size());
        const int nt = num_tris(mesh);

        std::vector<AABB> node_boxes(nv);
        for (int i = 0; i < nv; ++i) {
            node_boxes[i] = build_node_box_ref(x, v, i, dt, node_pad);
        }

        std::vector<std::array<int, 3>> tris(nt);
        std::vector<AABB> tri_boxes(nt);
        for (int t = 0; t < nt; ++t) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);
            tris[t] = {a, b, c};
            tri_boxes[t] = build_triangle_box_ref(x, v, a, b, c, dt, tri_pad);
        }

        const std::vector<EdgeKey> edges = build_unique_edges_ref(mesh);
        const int ne = static_cast<int>(edges.size());
        std::vector<AABB> edge_boxes(ne);
        for (int e = 0; e < ne; ++e) {
            edge_boxes[e] = build_edge_box_ref(x, v, edges[e].a, edges[e].b, dt, edge_pad);
        }

        for (int node = 0; node < nv; ++node) {
            for (int t = 0; t < nt; ++t) {
                const int a = tris[t][0];
                const int b = tris[t][1];
                const int c = tris[t][2];
                if (node == a || node == b || node == c) continue;
                if (!aabb_intersects(node_boxes[node], tri_boxes[t])) continue;
                out.nt.insert(make_nt_key(node, a, b, c));
            }
        }

        for (int e = 0; e < ne; ++e) {
            for (int f = e + 1; f < ne; ++f) {
                if (share_vertex(edges[e], edges[f])) continue;
                if (!aabb_intersects(edge_boxes[e], edge_boxes[f])) continue;
                out.ss.insert(make_ss_key(edges[e].a, edges[e].b, edges[f].a, edges[f].b));
            }
        }

        return out;
    }

    static PairSets exact_close_pairs(const std::vector<Vec3>& x, const RefMesh& mesh, double d_hat) {
        PairSets out;
        const int nv = static_cast<int>(x.size());
        const int nt = num_tris(mesh);
        const std::vector<EdgeKey> edges = build_unique_edges_ref(mesh);
        const int ne = static_cast<int>(edges.size());
        constexpr double eps = 1.0e-12;

        for (int node = 0; node < nv; ++node) {
            for (int t = 0; t < nt; ++t) {
                const int a = tri_vertex(mesh, t, 0);
                const int b = tri_vertex(mesh, t, 1);
                const int c = tri_vertex(mesh, t, 2);
                if (node == a || node == b || node == c) continue;
                const double d = node_triangle_distance(x[node], x[a], x[b], x[c], eps).distance;
                if (d < d_hat) out.nt.insert(make_nt_key(node, a, b, c));
            }
        }

        for (int e = 0; e < ne; ++e) {
            for (int f = e + 1; f < ne; ++f) {
                if (share_vertex(edges[e], edges[f])) continue;
                const double d = segment_segment_distance(x[edges[e].a], x[edges[e].b], x[edges[f].a], x[edges[f].b], eps).distance;
                if (d < d_hat) out.ss.insert(make_ss_key(edges[e].a, edges[e].b, edges[f].a, edges[f].b));
            }
        }

        return out;
    }

    static void build_two_sheet_scene(std::vector<Vec3>& x, std::vector<Vec3>& v, RefMesh& mesh) {
        x = {
                Vec3(0.0, 0.0, 0.00),
                Vec3(1.0, 0.0, 0.00),
                Vec3(0.0, 1.0, 0.00),
                Vec3(1.0, 1.0, 0.00),
                Vec3(0.2, 0.2, 0.35),
                Vec3(1.2, 0.2, 0.35),
                Vec3(0.2, 1.2, 0.35),
                Vec3(1.2, 1.2, 0.35),
        };

        v.assign(x.size(), Vec3::Zero());
        v[4] = Vec3(-0.05, 0.00, -0.60);
        v[5] = Vec3(-0.05, 0.00, -0.60);
        v[6] = Vec3(-0.05, 0.00, -0.60);
        v[7] = Vec3(-0.05, 0.00, -0.60);

        mesh = make_mesh(x, {
                {0, 1, 2},
                {1, 3, 2},
                {4, 5, 6},
                {5, 7, 6},
        });
    }

    static void build_three_sheet_scene(std::vector<Vec3>& x, std::vector<Vec3>& v, RefMesh& mesh) {
        x = {
                Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
                Vec3(0.1, 0.1, 0.03), Vec3(1.1, 0.1, 0.03), Vec3(0.1, 1.1, 0.03),
                Vec3(0.2, 0.2, 0.06), Vec3(1.2, 0.2, 0.06), Vec3(0.2, 1.2, 0.06),
        };
        v.assign(x.size(), Vec3::Zero());
        mesh = make_mesh(x, {
                {0, 1, 2},
                {3, 4, 5},
                {6, 7, 8},
        });
    }

    static void build_far_apart_double_scene(std::vector<Vec3>& x, std::vector<Vec3>& v, RefMesh& mesh) {
        x = {
                Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
                Vec3(0.2, 0.2, 0.03), Vec3(1.2, 0.2, 0.03), Vec3(0.2, 1.2, 0.03),
                Vec3(10.0, 0.0, 0.00), Vec3(11.0, 0.0, 0.00), Vec3(10.0, 1.0, 0.00),
                Vec3(10.2, 0.2, 0.03), Vec3(11.2, 0.2, 0.03), Vec3(10.2, 1.2, 0.03),
        };
        v.assign(x.size(), Vec3::Zero());

        mesh = make_mesh(x, {
                {0, 1, 2},
                {3, 4, 5},
                {6, 7, 8},
                {9, 10, 11},
        });
    }

    // Pair order feeds the deterministic solver schedule. Reconstruct the
    // expected first-seen order from BroadPhase's cached query results.
    static void expect_pair_order_matches_query_hits(const BroadPhase::Cache& cache,
                                                     const RefMesh& mesh) {
        std::unordered_set<std::uint64_t> seen_nt;
        std::vector<std::pair<int, int>> expected_nt;
        for (int node = 0; node < static_cast<int>(cache.node_hits.size()); ++node) {
            for (int t : cache.node_hits[node]) {
                const int a = tri_vertex(mesh, t, 0);
                const int b = tri_vertex(mesh, t, 1);
                const int c = tri_vertex(mesh, t, 2);
                if (node == a || node == b || node == c) continue;
                if (seen_nt.insert(BroadPhase::nt_key(node, t)).second)
                    expected_nt.emplace_back(node, t);
            }
        }

        ASSERT_EQ(cache.nt_pairs.size(), expected_nt.size());
        ASSERT_EQ(cache.nt_pair_tri.size(), expected_nt.size());
        for (std::size_t i = 0; i < expected_nt.size(); ++i) {
            EXPECT_EQ(cache.nt_pairs[i].node, expected_nt[i].first) << "NT pair " << i;
            EXPECT_EQ(cache.nt_pair_tri[i], expected_nt[i].second) << "NT pair " << i;
        }

        std::unordered_set<std::uint64_t> seen_ss;
        std::vector<std::array<int, 2>> expected_ss;
        for (int e = 0; e < static_cast<int>(cache.edge_hits.size()); ++e) {
            const EdgeKey e0(cache.edges[e][0], cache.edges[e][1]);
            for (int other : cache.edge_hits[e]) {
                if (other == e) continue;
                const EdgeKey e1(cache.edges[other][0], cache.edges[other][1]);
                if (share_vertex(e0, e1)) continue;
                const std::uint64_t key = BroadPhase::ss_key(e, other);
                if (seen_ss.insert(key).second)
                    expected_ss.push_back({std::min(e, other), std::max(e, other)});
            }
        }
        EXPECT_EQ(cache.ss_pair_edges, expected_ss);
    }

    static void expect_aabb_vectors_exact(
        const std::vector<AABB>& actual, const std::vector<AABB>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_EQ(actual[i].min[axis], expected[i].min[axis]) << "box " << i << ", min axis " << axis;
                EXPECT_EQ(actual[i].max[axis], expected[i].max[axis]) << "box " << i << ", max axis " << axis;
            }
        }
    }

    static void expect_bvh_vectors_exact(
        const std::vector<BVHNode>& actual,
        const std::vector<BVHNode>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_EQ(actual[i].left, expected[i].left) << "BVH node " << i;
            EXPECT_EQ(actual[i].right, expected[i].right) << "BVH node " << i;
            EXPECT_EQ(actual[i].parent, expected[i].parent) << "BVH node " << i;
            EXPECT_EQ(actual[i].leafIndex, expected[i].leafIndex) << "BVH node " << i;
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_EQ(actual[i].bbox.min[axis], expected[i].bbox.min[axis]) << "BVH node " << i << ", min axis " << axis;
                EXPECT_EQ(actual[i].bbox.max[axis], expected[i].bbox.max[axis]) << "BVH node " << i << ", max axis " << axis;
            }
        }
    }

    static void expect_vertex_pair_entries_exact(
        const std::vector<std::vector<BroadPhase::Cache::VertexPairEntry>>& actual,
        const std::vector<std::vector<BroadPhase::Cache::VertexPairEntry>>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t vertex = 0; vertex < actual.size(); ++vertex) {
            ASSERT_EQ(actual[vertex].size(), expected[vertex].size()) << "vertex " << vertex;
            for (std::size_t entry = 0; entry < actual[vertex].size(); ++entry) {
                EXPECT_EQ(actual[vertex][entry].pair_index, expected[vertex][entry].pair_index) << "vertex " << vertex << ", entry " << entry;
                EXPECT_EQ(actual[vertex][entry].dof, expected[vertex][entry].dof) << "vertex " << vertex << ", entry " << entry;
            }
        }
    }

    static void expect_cache_geometry_and_topology_exact(
        const BroadPhase::Cache& actual, const BroadPhase::Cache& expected) {
        expect_aabb_vectors_exact(actual.node_boxes, expected.node_boxes);
        expect_aabb_vectors_exact(actual.tri_boxes, expected.tri_boxes);
        expect_aabb_vectors_exact(actual.edge_boxes, expected.edge_boxes);
        expect_aabb_vectors_exact(actual.red_edge_boxes, expected.red_edge_boxes);
        expect_bvh_vectors_exact(actual.node_bvh_nodes, expected.node_bvh_nodes);
        expect_bvh_vectors_exact(actual.tri_bvh_nodes, expected.tri_bvh_nodes);
        expect_bvh_vectors_exact(actual.edge_bvh_nodes, expected.edge_bvh_nodes);
        EXPECT_EQ(actual.node_leaf_to_node, expected.node_leaf_to_node);
        EXPECT_EQ(actual.tri_leaf_to_node, expected.tri_leaf_to_node);
        EXPECT_EQ(actual.edge_leaf_to_node, expected.edge_leaf_to_node);
        EXPECT_EQ(actual.node_root, expected.node_root);
        EXPECT_EQ(actual.tri_root, expected.tri_root);
        EXPECT_EQ(actual.edge_root, expected.edge_root);
        EXPECT_EQ(actual.edges, expected.edges);
        EXPECT_EQ(actual.node_to_tris, expected.node_to_tris);
        EXPECT_EQ(actual.node_to_edges, expected.node_to_edges);
    }

    static void expect_cache_exact(
        const BroadPhase::Cache& actual, const BroadPhase::Cache& expected) {
        expect_cache_geometry_and_topology_exact(actual, expected);
        ASSERT_EQ(actual.nt_pairs.size(), expected.nt_pairs.size());
        for (std::size_t i = 0; i < actual.nt_pairs.size(); ++i) {
            EXPECT_EQ(actual.nt_pairs[i].node, expected.nt_pairs[i].node) << "NT pair " << i;
            for (int local = 0; local < 3; ++local) {
                EXPECT_EQ(actual.nt_pairs[i].tri_v[local], expected.nt_pairs[i].tri_v[local]) << "NT pair " << i << ", triangle vertex " << local;
            }
        }
        ASSERT_EQ(actual.ss_pairs.size(), expected.ss_pairs.size());
        for (std::size_t i = 0; i < actual.ss_pairs.size(); ++i) {
            for (int local = 0; local < 4; ++local) {
                EXPECT_EQ(actual.ss_pairs[i].v[local], expected.ss_pairs[i].v[local]) << "SS pair " << i << ", vertex " << local;
            }
        }
        EXPECT_EQ(actual.nt_pair_tri, expected.nt_pair_tri);
        EXPECT_EQ(actual.ss_pair_edges, expected.ss_pair_edges);
        expect_vertex_pair_entries_exact(actual.vertex_nt, expected.vertex_nt);
        expect_vertex_pair_entries_exact(actual.vertex_ss, expected.vertex_ss);
        EXPECT_EQ(actual.node_hits, expected.node_hits);
        EXPECT_EQ(actual.edge_hits, expected.edge_hits);
    }

} // namespace

TEST(AABBTest, DefaultConstructorStartsEmpty) {
AABB box;
EXPECT_GT(box.min.x(), box.max.x());
EXPECT_GT(box.min.y(), box.max.y());
EXPECT_GT(box.min.z(), box.max.z());
}

TEST(AABBTest, ExpandPointWorks) {
AABB box;
box.expand(Vec3(1.0, 2.0, 3.0));
box.expand(Vec3(-1.0, 4.0, 0.5));

EXPECT_DOUBLE_EQ(box.min.x(), -1.0);
EXPECT_DOUBLE_EQ(box.min.y(), 2.0);
EXPECT_DOUBLE_EQ(box.min.z(), 0.5);
EXPECT_DOUBLE_EQ(box.max.x(), 1.0);
EXPECT_DOUBLE_EQ(box.max.y(), 4.0);
EXPECT_DOUBLE_EQ(box.max.z(), 3.0);
}

TEST(AABBTest, IntersectionWorks) {
const AABB a(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
const AABB b(Vec3(0.5, 0.5, 0.5), Vec3(2.0, 2.0, 2.0));
const AABB c(Vec3(2.1, 2.1, 2.1), Vec3(3.0, 3.0, 3.0));
EXPECT_TRUE(aabb_intersects(a, b));
EXPECT_FALSE(aabb_intersects(a, c));
}

TEST(AABBTest, TouchingAtBoundaryIntersects) {
const AABB a(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
const AABB b(Vec3(1.0, 1.0, 1.0), Vec3(2.0, 2.0, 2.0));
EXPECT_TRUE(aabb_intersects(a, b));
}

TEST(BVH3Test, QueryReturnsExpectedHits) {
std::vector<AABB> boxes;
boxes.emplace_back(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
boxes.emplace_back(Vec3(2.0, 2.0, 2.0), Vec3(3.0, 3.0, 3.0));
boxes.emplace_back(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5));

std::vector<BVHNode> nodes;
const int root = build_bvh(boxes, nodes);

std::vector<int> hits;
query_bvh(nodes, root, AABB(Vec3(0.75, 0.75, 0.75), Vec3(0.8, 0.8, 0.8)), hits);
std::sort(hits.begin(), hits.end());

ASSERT_EQ(hits.size(), 2u);
EXPECT_EQ(hits[0], 0);
EXPECT_EQ(hits[1], 2);
}

TEST(BVH3Test, EmptyBuildReturnsInvalidRoot) {
std::vector<AABB> boxes;
std::vector<BVHNode> nodes;
const int root = build_bvh(boxes, nodes);
EXPECT_EQ(root, -1);
EXPECT_TRUE(nodes.empty());

std::vector<int> hits;
query_bvh(nodes, root, AABB(Vec3::Zero(), Vec3::Zero()), hits);
EXPECT_TRUE(hits.empty());
}

TEST(BVH3Test, RefitUpdatesQueryResults) {
std::vector<AABB> boxes = {
        AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0)),
        AABB(Vec3(3.0, 3.0, 3.0), Vec3(4.0, 4.0, 4.0)),
};

std::vector<BVHNode> nodes;
const int root = build_bvh(boxes, nodes);

{
std::vector<int> hits;
query_bvh(nodes, root, AABB(Vec3(0.2, 0.2, 0.2), Vec3(0.8, 0.8, 0.8)), hits);
ASSERT_EQ(hits.size(), 1u);
EXPECT_EQ(hits[0], 0);
}

boxes[0] = AABB(Vec3(10.0, 10.0, 10.0), Vec3(11.0, 11.0, 11.0));
refit_bvh(nodes, boxes);

{
std::vector<int> hits;
query_bvh(nodes, root, AABB(Vec3(0.2, 0.2, 0.2), Vec3(0.8, 0.8, 0.8)), hits);
EXPECT_TRUE(hits.empty());
}

{
std::vector<int> hits;
query_bvh(nodes, root, AABB(Vec3(10.2, 10.2, 10.2), Vec3(10.8, 10.8, 10.8)), hits);
ASSERT_EQ(hits.size(), 1u);
EXPECT_EQ(hits[0], 0);
}
}

TEST(BroadPhaseTest, SingleTriangleProducesNoIncidentPairs) {
const std::vector<Vec3> x = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.1);
EXPECT_TRUE(broad.nt_pairs().empty());
EXPECT_TRUE(broad.ss_pairs().empty());
}

TEST(BroadPhaseTest, DetectsNodeTrianglePairFromOverlappingBoxes) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.2, 0.2, 0.02), Vec3(1.2, 0.2, 0.02), Vec3(0.2, 1.2, 0.02),
};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.05);
const auto& nt = broad.nt_pairs();

EXPECT_TRUE(contains_nt_pair(nt, 3, 0, 1, 2) || contains_nt_pair(nt, 4, 0, 1, 2) || contains_nt_pair(nt, 5, 0, 1, 2));
}

TEST(BroadPhaseTest,
     SurfaceNodeInitializationExcludesTetInteriorQueriesWithoutChangingAllNodePath) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.2, 0.2, 0.2),
    };
    const std::vector<Vec3> v(x.size(), Vec3::Zero());
    RefMesh mesh = make_mesh(x, {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}});
    // Four tets subdivide the outer tet around interior node 4. The collision
    // triangles above are exactly the extracted outer boundary.
    mesh.tets = {
        4, 1, 2, 3,
        0, 4, 2, 3,
        0, 1, 4, 3,
        0, 1, 2, 4,
    };
    mesh.tet_nodes = {4, 1, 2, 3, 0};
    mesh.surface_nodes = {1, 2, 3, 0};

    // The large padding forces every nonincident primitive AABB to overlap,
    // so node 4 is excluded only by the new surface-query policy.
    constexpr double d_hat = 2.0;
    BroadPhase all_nodes;
    all_nodes.initialize(x, v, mesh, 1.0, d_hat);
    BroadPhase surface_nodes;
    surface_nodes.initialize_surface_nodes(x, v, mesh, 1.0, d_hat);

    EXPECT_EQ(all_nodes.nt_pairs().size(), 8U);
    EXPECT_EQ(surface_nodes.nt_pairs().size(), 4U);
    EXPECT_TRUE(std::any_of(
        all_nodes.nt_pairs().begin(), all_nodes.nt_pairs().end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));
    EXPECT_TRUE(std::none_of(
        surface_nodes.nt_pairs().begin(), surface_nodes.nt_pairs().end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));

    // The initialization policy persists when the existing BVHs are queried
    // again. The original path remains all-node, while the solid-aware path
    // continues to exclude node 4.
    all_nodes.refresh_pairs(mesh);
    surface_nodes.refresh_pairs(mesh);
    EXPECT_EQ(all_nodes.nt_pairs().size(), 8U);
    EXPECT_EQ(surface_nodes.nt_pairs().size(), 4U);
    EXPECT_TRUE(std::any_of(
        all_nodes.nt_pairs().begin(), all_nodes.nt_pairs().end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));
    EXPECT_TRUE(std::none_of(
        surface_nodes.nt_pairs().begin(), surface_nodes.nt_pairs().end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));

    const BroadPhase::Cache& all_cache = all_nodes.cache();
    const BroadPhase::Cache& surface_cache = surface_nodes.cache();
    ASSERT_EQ(surface_cache.node_to_tris.size(), x.size());
    EXPECT_TRUE(surface_cache.node_to_tris[4].empty());
    EXPECT_EQ(all_cache.vertex_nt[4].size(), 4U);
    EXPECT_TRUE(surface_cache.vertex_nt[4].empty());
    EXPECT_TRUE(surface_cache.vertex_ss[4].empty());

    // Edge construction already uses only boundary triangles, so both paths
    // retain the same six surface edges and three opposite-edge candidates.
    EXPECT_EQ(all_cache.edges.size(), 6U);
    EXPECT_EQ(surface_cache.edges.size(), 6U);
    EXPECT_EQ(all_nodes.ss_pairs().size(), 3U);
    EXPECT_EQ(surface_nodes.ss_pairs().size(), 3U);
    EXPECT_EQ(
        pair_sets_from_broad(all_nodes).ss,
        pair_sets_from_broad(surface_nodes).ss);
    for (const std::array<int, 2>& edge : surface_cache.edges) {
        EXPECT_NE(edge[0], 4);
        EXPECT_NE(edge[1], 4);
    }
}

TEST(BroadPhaseTest, SurfaceNodeInitializationRetainsPointOnlyRigidProxy) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.2, 0.2, 0.2),
    };
    const std::vector<Vec3> v(x.size(), Vec3::Zero());
    RefMesh mesh = make_mesh(x, {
        {1, 2, 3},
        {0, 3, 2},
        {0, 1, 3},
        {0, 2, 1},
    });
    mesh.tets = {0, 1, 2, 3};
    mesh.tet_nodes = {0, 1, 2, 3};
    mesh.surface_nodes = {1, 2, 3, 0};
    mesh.node_to_rb = {-1, -1, -1, -1, 0};
    mesh.rb_nodes = {{4}};

    BroadPhase broad_phase;
    broad_phase.initialize_surface_nodes(
        x, v, mesh, 1.0, /*dhat=*/2.0);

    const BroadPhase::Cache& cache = broad_phase.cache();
    ASSERT_EQ(cache.node_to_tris.size(), x.size());
    EXPECT_TRUE(cache.node_to_tris[4].empty());
    EXPECT_EQ(cache.vertex_nt[4].size(), 4U);
    EXPECT_EQ(broad_phase.nt_pairs().size(), 8U);
    EXPECT_EQ(
        std::count_if(
            broad_phase.nt_pairs().begin(), broad_phase.nt_pairs().end(),
            [](const NodeTrianglePair& pair) { return pair.node == 4; }),
        4);
    for (const BroadPhase::Cache::VertexPairEntry& entry :
         cache.vertex_nt[4]) {
        ASSERT_LT(entry.pair_index, cache.nt_pairs.size());
        EXPECT_EQ(entry.dof, 0);
        EXPECT_EQ(cache.nt_pairs[entry.pair_index].node, 4);
    }

    broad_phase.refresh_pairs(mesh);
    EXPECT_EQ(broad_phase.cache().vertex_nt[4].size(), 4U);
    EXPECT_EQ(broad_phase.nt_pairs().size(), 8U);
}

TEST(BroadPhaseTest,
     PrebuiltSurfaceNodeBoxesExcludeTetInteriorAndRetainOtherQueries) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.2, 0.2, 0.2),
        Vec3(0.3, 0.3, 0.3),
    };
    RefMesh mesh = make_mesh(x, {
        {1, 2, 3},
        {0, 3, 2},
        {0, 1, 3},
        {0, 2, 1},
    });
    mesh.tets = {
        4, 1, 2, 3,
        0, 4, 2, 3,
        0, 1, 4, 3,
        0, 1, 2, 4,
    };
    mesh.tet_nodes = {4, 1, 2, 3, 0};
    mesh.surface_nodes = {1, 2, 3, 0};
    mesh.node_to_rb = {-1, -1, -1, -1, -1, 0};
    mesh.rb_nodes = {{5}};

    // Give both initializers the exact same conservative boxes. Every box
    // overlaps every primitive, so differences in the NT results can only
    // come from the surface-node query policy rather than box construction.
    const AABB common_box(
        Vec3::Constant(-2.0), Vec3::Constant(2.0));
    const std::vector<AABB> boxes(x.size(), common_box);

    BroadPhase all_nodes;
    all_nodes.initialize(boxes, mesh, /*d_hat=*/0.0);
    BroadPhase surface_nodes;
    surface_nodes.initialize_surface_nodes(
        boxes, mesh, /*d_hat=*/0.0);

    const auto count_point_queries = [](const BroadPhase& broad_phase,
                                        const int node) {
        return std::count_if(
            broad_phase.nt_pairs().begin(), broad_phase.nt_pairs().end(),
            [node](const NodeTrianglePair& pair) {
                return pair.node == node;
            });
    };

    // Node 4 is the only tet-interior point. The ordinary prebuilt-box path
    // retains its four queries, while the surface-aware overload removes all
    // of them.
    EXPECT_EQ(count_point_queries(all_nodes, 4), 4);
    EXPECT_EQ(count_point_queries(surface_nodes, 4), 0);
    EXPECT_EQ(all_nodes.cache().vertex_nt[4].size(), 4U);
    EXPECT_TRUE(surface_nodes.cache().vertex_nt[4].empty());

    // Boundary node 0 and point-only rigid proxy 5 remain eligible under the
    // surface policy. Each sees all nonincident boundary triangles allowed by
    // topology, exactly as in the ordinary initializer.
    EXPECT_EQ(count_point_queries(surface_nodes, 0),
              count_point_queries(all_nodes, 0));
    EXPECT_GT(count_point_queries(surface_nodes, 0), 0);
    EXPECT_EQ(count_point_queries(surface_nodes, 5), 4);
    EXPECT_EQ(count_point_queries(surface_nodes, 5),
              count_point_queries(all_nodes, 5));

    // Surface filtering changes only point queries. Both initializers retain
    // identical input node boxes and identical boundary-edge SS candidates.
    ASSERT_EQ(surface_nodes.cache().node_boxes.size(), boxes.size());
    for (std::size_t node = 0; node < boxes.size(); ++node) {
        EXPECT_TRUE(surface_nodes.cache().node_boxes[node].min.isApprox(
            boxes[node].min));
        EXPECT_TRUE(surface_nodes.cache().node_boxes[node].max.isApprox(
            boxes[node].max));
    }
    EXPECT_FALSE(all_nodes.ss_pairs().empty());
    EXPECT_EQ(
        pair_sets_from_broad(surface_nodes).ss,
        pair_sets_from_broad(all_nodes).ss);
}

TEST(BroadPhaseTest, PrebuiltSurfaceNodeOnePassMatchesLegacyRefreshResultExactly) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.2, 0.2, 0.2),  // tet-interior node
        Vec3(0.3, 0.3, 0.3),  // point-only proxy
    };
    RefMesh mesh = make_mesh(x, {
        {1, 2, 3},
        {0, 3, 2},
        {0, 1, 3},
        {0, 2, 1},
    });
    mesh.tets = {
        4, 1, 2, 3,
        0, 4, 2, 3,
        0, 1, 4, 3,
        0, 1, 2, 4,
    };
    // Deliberately unordered: query order must still follow ascending node
    // index, just as it did when the old wrapper called refresh_pairs().
    mesh.tet_nodes = {4, 2, 0, 3, 1};
    mesh.surface_nodes = {3, 0, 2, 1};
    mesh.node_to_rb = {-1, -1, -1, -1, -1, 0};
    mesh.rb_nodes = {{5}};

    std::vector<AABB> boxes;
    boxes.reserve(x.size());
    for (int node = 0; node < static_cast<int>(x.size()); ++node) {
        // Distinct centroids avoid BVH tie ambiguity, while the generous
        // extents make the filtered interior query observable.
        const Vec3 offset = Vec3::Constant(0.01 * node);
        boxes.emplace_back(Vec3::Constant(-2.0) + offset, Vec3::Constant(2.0) + offset);
    }
    constexpr double d_hat = 0.125;

    BroadPhase all_nodes;
    all_nodes.initialize(boxes, mesh, d_hat);

    BroadPhase surface_nodes;
    surface_nodes.initialize_surface_nodes(boxes, mesh, d_hat);
    const BroadPhase::Cache one_pass = surface_nodes.cache();

    // The policy affects queries only. The old wrapper first produced exactly
    // this all-node geometry/topology before replacing its pair lists.
    expect_cache_geometry_and_topology_exact(one_pass, all_nodes.cache());
    ASSERT_EQ(one_pass.node_hits.size(), x.size());
    EXPECT_TRUE(one_pass.node_hits[4].empty());
    EXPECT_FALSE(all_nodes.cache().node_hits[4].empty());
    EXPECT_TRUE(std::none_of(one_pass.nt_pairs.begin(), one_pass.nt_pairs.end(), [](const NodeTrianglePair& pair) { return pair.node == 4; }));
    EXPECT_TRUE(std::any_of(one_pass.nt_pairs.begin(), one_pass.nt_pairs.end(), [](const NodeTrianglePair& pair) { return pair.node == 5; }));

    // refresh_pairs() is the final operation performed by the former
    // two-pass wrapper. Exact equality here covers hit order, NT/SS pair
    // order, pair metadata, and per-vertex incidence indices in addition to
    // the immutable boxes and BVHs.
    surface_nodes.refresh_pairs(mesh);
    expect_cache_exact(surface_nodes.cache(), one_pass);
    expect_pair_order_matches_query_hits(surface_nodes.cache(), mesh);
}

TEST(BroadPhaseTest, DetectsSegmentSegmentPairFromOverlappingBoxes) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.0),
};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.0);
EXPECT_TRUE(contains_ss_pair(broad.ss_pairs(), EdgeKey(0, 1), EdgeKey(3, 4)));
}

TEST(BroadPhaseTest, CCDCandidatesDetectFutureNodeTriangleOverlap) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.25, 0.25, 1.0), Vec3(1.25, 0.25, 1.0), Vec3(0.25, 1.25, 1.0),
};

std::vector<Vec3> v(x.size(), Vec3::Zero());
v[3] = Vec3(0.0, 0.0, -1.0);
v[4] = Vec3(0.0, 0.0, -1.0);
v[5] = Vec3(0.0, 0.0, -1.0);

const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.build_ccd_candidates(x, v, mesh, 1.0);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2) || contains_nt_pair(broad.nt_pairs(), 4, 0, 1, 2) || contains_nt_pair(broad.nt_pairs(), 5, 0, 1, 2));
}

TEST(BroadPhaseTest, SingleMeshSelfCollisionFoldDetectsNonIncidentPairs) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00), Vec3(0.2, 0.2, 0.03),
};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {1, 3, 2}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.05);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2));
}

TEST(BroadPhaseTest, CCDCandidatesDetectFutureSegmentSegmentOverlap) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 0.5), Vec3(1.0, 0.0, 0.5), Vec3(0.0, 1.0, 0.5),
};

std::vector<Vec3> v(x.size(), Vec3::Zero());
v[3] = Vec3(0.0, 0.0, -0.5);
v[4] = Vec3(0.0, 0.0, -0.5);
v[5] = Vec3(0.0, 0.0, -0.5);

const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.build_ccd_candidates(x, v, mesh, 1.0);

EXPECT_TRUE(contains_ss_pair(broad.ss_pairs(), EdgeKey(0, 1), EdgeKey(3, 4)));
}

TEST(BroadPhaseTest, CCDCandidatesDetectTangentialSkimmingMotion) {
const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
        Vec3(-0.8, 0.25, 0.02), Vec3(-0.2, 0.25, 0.02), Vec3(-0.8, 0.85, 0.02),
};
std::vector<Vec3> v(x.size(), Vec3::Zero());
v[3] = Vec3(1.0, 0.0, -0.03);
v[4] = Vec3(1.0, 0.0, -0.03);
v[5] = Vec3(1.0, 0.0, -0.03);

const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.build_ccd_candidates(x, v, mesh, 1.0);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2) || contains_nt_pair(broad.nt_pairs(), 4, 0, 1, 2) || contains_nt_pair(broad.nt_pairs(), 5, 0, 1, 2));
}

TEST(BroadPhaseTest, InitializeMatchesBruteForceReference) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_two_sheet_scene(x, v, mesh);

const double dt = 0.75;
const double dhat = 0.06;

BroadPhase broad;
broad.initialize(x, v, mesh, dt, dhat);

const PairSets got = pair_sets_from_broad(broad);
const PairSets ref = brute_force_candidates(x, v, mesh, dt, dhat, 0.0, dhat * 0.5);
EXPECT_EQ(got.nt, ref.nt);
EXPECT_EQ(got.ss, ref.ss);
}

TEST(BroadPhaseTest, ThreeObjectsPopulateAllPairwiseInteractions) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_three_sheet_scene(x, v, mesh);

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.07);

const auto& nt = broad.nt_pairs();
EXPECT_TRUE(contains_nt_pair(nt, 3, 0, 1, 2));
EXPECT_TRUE(contains_nt_pair(nt, 6, 3, 4, 5));
EXPECT_TRUE(contains_nt_pair(nt, 6, 0, 1, 2));
}

TEST(BroadPhaseTest, CCDCandidatesMatchBruteForceReferenceAtZeroPad) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_two_sheet_scene(x, v, mesh);

const double dt = 0.75;

BroadPhase broad;
broad.build_ccd_candidates(x, v, mesh, dt);

const PairSets got = pair_sets_from_vectors(broad.nt_pairs(), broad.ss_pairs());
const PairSets ref = brute_force_candidates(x, v, mesh, dt, 0.0, 0.0, 0.0);
EXPECT_EQ(got.nt, ref.nt);
EXPECT_EQ(got.ss, ref.ss);
}

TEST(BroadPhaseTest, BroadPhaseIsConservativeForPairsCloserThanDhat) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_two_sheet_scene(x, v, mesh);
for (int i = 4; i < static_cast<int>(x.size()); ++i) x[i].z() = 0.03;
std::fill(v.begin(), v.end(), Vec3::Zero());

const double d_hat = 0.05;
const PairSets exact = exact_close_pairs(x, mesh, d_hat);
ASSERT_FALSE(exact.nt.empty());
ASSERT_FALSE(exact.ss.empty());

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, d_hat);
const PairSets got = pair_sets_from_broad(broad);

EXPECT_TRUE(std::includes(got.nt.begin(), got.nt.end(), exact.nt.begin(), exact.nt.end()));
EXPECT_TRUE(std::includes(got.ss.begin(), got.ss.end(), exact.ss.begin(), exact.ss.end()));
}

TEST(BroadPhaseTest, LargerDhatProducesSupersetOfPairs) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_two_sheet_scene(x, v, mesh);

const double dt = 0.75;

BroadPhase small_bp, large_bp;
small_bp.initialize(x, v, mesh, dt, 0.01);
large_bp.initialize(x, v, mesh, dt, 0.10);

const PairSets small = pair_sets_from_broad(small_bp);
const PairSets large = pair_sets_from_broad(large_bp);

EXPECT_TRUE(std::includes(large.nt.begin(), large.nt.end(), small.nt.begin(), small.nt.end()));
EXPECT_TRUE(std::includes(large.ss.begin(), large.ss.end(), small.ss.begin(), small.ss.end()));
}

TEST(BroadPhaseTest, PairOrderMatchesFirstQueryHitOrder) {
    std::vector<Vec3> x, v;
    RefMesh mesh;
    build_three_sheet_scene(x, v, mesh);

    const int nv = static_cast<int>(x.size());
    std::vector<AABB> boxes(nv);
    for (int i = 0; i < nv; ++i)
        boxes[i] = AABB(x[i] - Vec3::Constant(0.04), x[i] + Vec3::Constant(0.04));

    BroadPhase broad;
    broad.initialize(boxes, mesh, 0.07);
    expect_pair_order_matches_query_hits(broad.cache(), mesh);

    // Exercise the OGC refresh path, whose edge BVH leaves are incrementally
    // refitted red boxes while edge_boxes remain the padded green queries.
    x[0] += Vec3(0.01, -0.005, 0.002);
    incremental_refresh_vertex(broad.mutable_cache(), 0, x, mesh,
                               /*box_pad=*/0.07, /*node_box_radius_padded=*/0.04);
    broad.refresh_pairs(mesh);
    expect_pair_order_matches_query_hits(broad.cache(), mesh);

    broad.build_ccd_candidates(x, v, mesh, 1.0);
    expect_pair_order_matches_query_hits(broad.cache(), mesh);

    // The velocity-based build does not create edge leaf maps. Refresh must
    // still recognize the earlier directional hit and avoid duplicate pairs.
    broad.refresh_pairs(mesh);
    expect_pair_order_matches_query_hits(broad.cache(), mesh);
}

TEST(BroadPhaseTest, EmptyMeshProducesNoPairs) {
const std::vector<Vec3> x;
const std::vector<Vec3> v;
const RefMesh mesh = make_mesh(x, {});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.1);
EXPECT_TRUE(broad.nt_pairs().empty());
EXPECT_TRUE(broad.ss_pairs().empty());

broad.build_ccd_candidates(x, v, mesh, 1.0);
EXPECT_TRUE(broad.nt_pairs().empty());
EXPECT_TRUE(broad.ss_pairs().empty());
}

TEST(BroadPhaseTest, TriangleFreeMeshProducesNoPairs) {
const std::vector<Vec3> x = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 1.0);
EXPECT_TRUE(broad.nt_pairs().empty());
EXPECT_TRUE(broad.ss_pairs().empty());
}

// ====================================================================
//  incremental_refresh_vertex: after moving x[vi], the partial refit
//  must leave node/tri/edge leaf boxes equal to a fresh recomputation
//  from the new positions, and every internal BVH node must equal the
//  union of its children (the refit invariant). Pair lists are NOT
//  mutated by the helper -- they are frozen for the iteration.
// ====================================================================
namespace {

bool aabb_equal(const AABB& a, const AABB& b, double tol = 1e-12) {
    return (a.min - b.min).cwiseAbs().maxCoeff() <= tol &&
           (a.max - b.max).cwiseAbs().maxCoeff() <= tol;
}

void check_bvh_internal_invariant(const std::vector<BVHNode>& nodes) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].leafIndex >= 0) continue;
        ASSERT_GE(nodes[i].left,  0);
        ASSERT_GE(nodes[i].right, 0);
        AABB combined = nodes[nodes[i].left].bbox;
        combined.expand(nodes[nodes[i].right].bbox);
        EXPECT_TRUE(aabb_equal(nodes[i].bbox, combined))
            << "internal node " << i << " bbox is not the union of its children";
    }
}

}  // namespace

TEST(BroadPhaseTest, IncrementalRefreshMatchesFreshBoxes) {
    // Two triangles sharing edge (1,2) plus a free vertex at index 4.
    // Vertex 1 is incident to both triangles and to edges (0,1),(1,2),(1,3),
    // so refreshing it touches all three BVHs nontrivially.
    const std::vector<Vec3> x_init = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
        Vec3(0.5, 0.5, 1.0),  // free vertex (no incident triangle)
    };
    const RefMesh mesh = make_mesh(x_init, {{0, 1, 2}, {1, 3, 2}});
    const int nv = static_cast<int>(x_init.size());

    constexpr double radius = 0.1;
    constexpr double pad    = 0.02;

    auto make_node_boxes = [&](const std::vector<Vec3>& x) {
        std::vector<AABB> b(nv);
        for (int i = 0; i < nv; ++i)
            b[i] = AABB(x[i] - Vec3::Constant(radius), x[i] + Vec3::Constant(radius));
        return b;
    };

    BroadPhase bp;
    bp.initialize(make_node_boxes(x_init), mesh, pad);

    // --- Before the move: snapshot leaf bboxes for unmoved primitives. ---
    const auto cache_before = bp.cache();
    const std::size_t pair_count_nt_before = cache_before.nt_pairs.size();
    const std::size_t pair_count_ss_before = cache_before.ss_pairs.size();

    // --- Move vertex 1 by a non-trivial offset. ---
    constexpr int    vi   = 1;
    const Vec3 displacement(0.05, -0.03, 0.07);
    std::vector<Vec3> x_new = x_init;
    x_new[vi] += displacement;

    bp.mutable_cache().node_boxes[vi] = AABB(x_new[vi] - Vec3::Constant(radius),
                                              x_new[vi] + Vec3::Constant(radius));
    incremental_refresh_vertex(bp.mutable_cache(), vi, x_new, mesh, pad, radius);

    const auto& c = bp.cache();

    // --- vi's node leaf must be the new padded cube around x_new[vi]. ---
    const AABB expected_node_box(x_new[vi] - Vec3::Constant(radius),
                                  x_new[vi] + Vec3::Constant(radius));
    EXPECT_TRUE(aabb_equal(c.node_boxes[vi], expected_node_box));

    // --- Untouched node leaves are unchanged. ---
    for (int j = 0; j < nv; ++j) {
        if (j == vi) continue;
        EXPECT_TRUE(aabb_equal(c.node_boxes[j], cache_before.node_boxes[j]))
            << "untouched node " << j << " was modified";
    }

    // --- Incident tri boxes match union(node_boxes) + pad. ---
    const Vec3 padv = Vec3::Constant(pad);
    for (int t : c.node_to_tris[vi]) {
        AABB expect = c.node_boxes[mesh.tris[3 * t + 0]];
        expect.expand(c.node_boxes[mesh.tris[3 * t + 1]]);
        expect.expand(c.node_boxes[mesh.tris[3 * t + 2]]);
        expect.min -= padv;
        expect.max += padv;
        EXPECT_TRUE(aabb_equal(c.tri_boxes[t], expect))
            << "tri " << t << " box does not match fresh union+pad";
    }

    // --- Incident edge boxes match union + pad (the green box). ---
    for (int e : c.node_to_edges[vi]) {
        AABB expect = c.node_boxes[c.edges[e][0]];
        expect.expand(c.node_boxes[c.edges[e][1]]);
        expect.min -= padv;
        expect.max += padv;
        EXPECT_TRUE(aabb_equal(c.edge_boxes[e], expect))
            << "edge " << e << " box does not match fresh union+pad";
    }

    // --- Refit invariant: every internal BVH node = union of its children. ---
    check_bvh_internal_invariant(c.node_bvh_nodes);
    check_bvh_internal_invariant(c.tri_bvh_nodes);
    check_bvh_internal_invariant(c.edge_bvh_nodes);

    // --- Leaf bbox in each BVH matches the box stored in the cache vector. ---
    for (int i = 0; i < nv; ++i) {
        const int n = c.node_leaf_to_node[i];
        ASSERT_GE(n, 0);
        EXPECT_EQ(c.node_bvh_nodes[n].leafIndex, i);
        EXPECT_TRUE(aabb_equal(c.node_bvh_nodes[n].bbox, c.node_boxes[i]));
    }
    for (int t = 0; t < static_cast<int>(c.tri_boxes.size()); ++t) {
        const int n = c.tri_leaf_to_node[t];
        ASSERT_GE(n, 0);
        EXPECT_TRUE(aabb_equal(c.tri_bvh_nodes[n].bbox, c.tri_boxes[t]));
    }
    // Edge BVH is built from RED (unpadded) boxes per the asymmetric SS
    // convention -- recompute the unpadded union for comparison.
    for (int e = 0; e < static_cast<int>(c.edges.size()); ++e) {
        const int n = c.edge_leaf_to_node[e];
        ASSERT_GE(n, 0);
        AABB red = c.node_boxes[c.edges[e][0]];
        red.expand(c.node_boxes[c.edges[e][1]]);
        EXPECT_TRUE(aabb_equal(c.edge_bvh_nodes[n].bbox, red))
            << "edge BVH leaf " << e << " is not the unpadded union";
    }

    // --- Pair lists are NOT mutated (frozen for the iteration). ---
    EXPECT_EQ(c.nt_pairs.size(), pair_count_nt_before);
    EXPECT_EQ(c.ss_pairs.size(), pair_count_ss_before);
}

// ====================================================================
//  Sanity check: a no-op call (move vi to itself) leaves every leaf
//  bbox bit-identical, and a sequence of partial refits stays
//  consistent with a single full rebuild against the same final state.
// ====================================================================
TEST(BroadPhaseTest, IncrementalRefreshIsIdempotentForZeroMove) {
    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
    };
    const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {1, 3, 2}});
    const int nv = static_cast<int>(x.size());

    constexpr double radius = 0.1;
    constexpr double pad    = 0.02;

    std::vector<AABB> boxes(nv);
    for (int i = 0; i < nv; ++i)
        boxes[i] = AABB(x[i] - Vec3::Constant(radius), x[i] + Vec3::Constant(radius));

    BroadPhase bp;
    bp.initialize(boxes, mesh, pad);
    const auto cache_before = bp.cache();

    for (int vi = 0; vi < nv; ++vi)
        incremental_refresh_vertex(bp.mutable_cache(), vi, x, mesh, pad, radius);

    const auto& c = bp.cache();
    for (int i = 0; i < nv; ++i)
        EXPECT_TRUE(aabb_equal(c.node_boxes[i], cache_before.node_boxes[i]));
    for (std::size_t t = 0; t < c.tri_boxes.size(); ++t)
        EXPECT_TRUE(aabb_equal(c.tri_boxes[t], cache_before.tri_boxes[t]));
    for (std::size_t e = 0; e < c.edge_boxes.size(); ++e)
        EXPECT_TRUE(aabb_equal(c.edge_boxes[e], cache_before.edge_boxes[e]));
    check_bvh_internal_invariant(c.node_bvh_nodes);
    check_bvh_internal_invariant(c.tri_bvh_nodes);
    check_bvh_internal_invariant(c.edge_bvh_nodes);
}

TEST(BroadPhaseTest, SolverModesPreserveFilteredPairAndIncidenceOrderWithoutRefitStorage) {
    const std::vector<Vec3> x = {Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00), Vec3(1.0, 1.0, 0.00), Vec3(0.1, 0.1, 0.03), Vec3(1.1, 0.1, 0.03), Vec3(0.1, 1.1, 0.03), Vec3(1.1, 1.1, 0.03), Vec3(0.2, 0.2, 0.06), Vec3(1.2, 0.2, 0.06), Vec3(0.2, 1.2, 0.06), Vec3(1.2, 1.2, 0.06)};
    RefMesh mesh = make_mesh(x, {{0, 1, 2}, {1, 3, 2}, {4, 5, 6}, {5, 7, 6}, {8, 9, 10}, {9, 11, 10}});
    mesh.node_to_rb = {0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1};
    mesh.rb_nodes = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    const std::vector<AABB> boxes(x.size(), AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    BroadPhase reference;
    BroadPhase general;
    BroadPhase rigid;
    reference.initialize(boxes, mesh, 0.1, BroadPhase::InitializationMode::Refittable);
    general.initialize(boxes, mesh, 0.1, BroadPhase::InitializationMode::GeneralSolver);
    rigid.initialize(boxes, mesh, 0.1, BroadPhase::InitializationMode::RigidSolver);

    const auto node_owner = [&](const int node) { return mesh.node_to_rb[static_cast<std::size_t>(node)]; };
    std::vector<NodeTrianglePair> expected_nt;
    for (const NodeTrianglePair& pair : reference.nt_pairs()) {
        const int owner = node_owner(pair.node);
        if (owner < 0 || node_owner(pair.tri_v[0]) != owner || node_owner(pair.tri_v[1]) != owner || node_owner(pair.tri_v[2]) != owner) expected_nt.push_back(pair);
    }
    std::vector<SegmentSegmentPair> expected_ss;
    for (const SegmentSegmentPair& pair : reference.ss_pairs()) {
        const int owner = node_owner(pair.v[0]);
        if (owner < 0 || node_owner(pair.v[1]) != owner || node_owner(pair.v[2]) != owner || node_owner(pair.v[3]) != owner) expected_ss.push_back(pair);
    }
    const auto expect_pairs = [&](const BroadPhase& broad_phase) {
        ASSERT_EQ(broad_phase.nt_pairs().size(), expected_nt.size());
        ASSERT_EQ(broad_phase.ss_pairs().size(), expected_ss.size());
        for (std::size_t pair_index = 0; pair_index < expected_nt.size(); ++pair_index) {
            EXPECT_EQ(broad_phase.nt_pairs()[pair_index].node, expected_nt[pair_index].node);
            for (int role = 0; role < 3; ++role) EXPECT_EQ(broad_phase.nt_pairs()[pair_index].tri_v[role], expected_nt[pair_index].tri_v[role]);
        }
        for (std::size_t pair_index = 0; pair_index < expected_ss.size(); ++pair_index) for (int role = 0; role < 4; ++role) EXPECT_EQ(broad_phase.ss_pairs()[pair_index].v[role], expected_ss[pair_index].v[role]);
    };
    expect_pairs(general);
    expect_pairs(rigid);

    std::vector<std::vector<BroadPhase::Cache::VertexPairEntry>> expected_vertex_nt(x.size());
    std::vector<std::vector<BroadPhase::Cache::VertexPairEntry>> expected_vertex_ss(x.size());
    for (std::size_t pair_index = 0; pair_index < expected_nt.size(); ++pair_index) {
        const NodeTrianglePair& pair = expected_nt[pair_index];
        const int nodes[4] = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        for (int role = 0; role < 4; ++role) if (node_owner(nodes[role]) < 0) expected_vertex_nt[static_cast<std::size_t>(nodes[role])].push_back({pair_index, role});
    }
    for (std::size_t pair_index = 0; pair_index < expected_ss.size(); ++pair_index) for (int role = 0; role < 4; ++role) if (node_owner(expected_ss[pair_index].v[role]) < 0) expected_vertex_ss[static_cast<std::size_t>(expected_ss[pair_index].v[role])].push_back({pair_index, role});
    ASSERT_EQ(general.cache().vertex_nt.size(), x.size());
    ASSERT_EQ(general.cache().vertex_ss.size(), x.size());
    for (std::size_t node = 0; node < x.size(); ++node) {
        ASSERT_EQ(general.cache().vertex_nt[node].size(), expected_vertex_nt[node].size());
        ASSERT_EQ(general.cache().vertex_ss[node].size(), expected_vertex_ss[node].size());
        for (std::size_t entry = 0; entry < expected_vertex_nt[node].size(); ++entry) {
            EXPECT_EQ(general.cache().vertex_nt[node][entry].pair_index, expected_vertex_nt[node][entry].pair_index);
            EXPECT_EQ(general.cache().vertex_nt[node][entry].dof, expected_vertex_nt[node][entry].dof);
        }
        for (std::size_t entry = 0; entry < expected_vertex_ss[node].size(); ++entry) {
            EXPECT_EQ(general.cache().vertex_ss[node][entry].pair_index, expected_vertex_ss[node][entry].pair_index);
            EXPECT_EQ(general.cache().vertex_ss[node][entry].dof, expected_vertex_ss[node][entry].dof);
        }
    }
    EXPECT_TRUE(rigid.cache().vertex_nt.empty());
    EXPECT_TRUE(rigid.cache().vertex_ss.empty());
    EXPECT_TRUE(general.cache().node_bvh_nodes.empty());
    EXPECT_TRUE(general.cache().node_leaf_to_node.empty());
    EXPECT_TRUE(general.cache().tri_leaf_to_node.empty());
    EXPECT_TRUE(general.cache().edge_leaf_to_node.empty());
    EXPECT_TRUE(rigid.cache().node_bvh_nodes.empty());
    EXPECT_TRUE(rigid.cache().node_leaf_to_node.empty());
    EXPECT_TRUE(rigid.cache().tri_leaf_to_node.empty());
    EXPECT_TRUE(rigid.cache().edge_leaf_to_node.empty());
}

TEST(BroadPhaseTest, PerVertexSafeStepClampsCCDAndOGC) {
    std::vector<Vec3> x = {
        Vec3(1.5, 0.25, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
    };
    const std::vector<Vec3> target = {
        Vec3(0.75, 0.25, 0.0),  // first touches the triangle at t = 1
        x[1], x[2], x[3],
    };

    BroadPhase bp;
    auto& cache = bp.mutable_cache();
    cache.node_boxes.assign(x.size(), AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    cache.vertex_nt.resize(x.size());
    cache.vertex_ss.resize(x.size());
    NodeTrianglePair pair{};
    pair.node = 0;
    pair.tri_v[0] = 1;
    pair.tri_v[1] = 2;
    pair.tri_v[2] = 3;
    cache.nt_pairs.push_back(pair);
    cache.vertex_nt[0].push_back({/*pair_index=*/0, /*dof=*/0});

    const double ccd_weight = per_vertex_safe_step(bp, x, 0, target[0], /*safety=*/0.9, /*clip_ccd=*/true, /*use_ticcd=*/false, /*use_ogc=*/false);

    const Vec3 expected(0.825, 0.25, 0.0);
    EXPECT_DOUBLE_EQ(ccd_weight, 0.9);
    EXPECT_TRUE(x[0].isApprox(expected, 1.0e-12));

    // Exercise the production OGC path with an incident pair initially 0.5
    // apart. Its trust radius is 0.4 * 0.5 = 0.2, so a 0.25 displacement is
    // scaled to 0.8 even though CCD clipping is disabled.
    std::vector<Vec3> x_ogc = {
        Vec3(0.0, 0.5, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
    const std::vector<Vec3> target_ogc = {
        Vec3(0.0, 0.25, 0.0),
        x_ogc[1], x_ogc[2], x_ogc[3],
    };

    BroadPhase ogc_bp;
    auto& ogc_cache = ogc_bp.mutable_cache();
    ogc_cache.node_boxes.assign(x_ogc.size(), AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    ogc_cache.vertex_nt.resize(x_ogc.size());
    ogc_cache.vertex_ss.resize(x_ogc.size());
    ogc_cache.nt_pairs.push_back(pair);
    ogc_cache.vertex_nt[0].push_back({/*pair_index=*/0, /*dof=*/0});

    const double ogc_weight = per_vertex_safe_step(ogc_bp, x_ogc, 0, target_ogc[0], /*safety=*/0.9, /*clip_ccd=*/false, /*use_ticcd=*/false, /*use_ogc=*/true);

    EXPECT_DOUBLE_EQ(ogc_weight, 0.8);
    EXPECT_TRUE(x_ogc[0].isApprox(Vec3(0.0, 0.3, 0.0), 1.0e-12));
}

TEST(BroadPhaseTest, PerVertexSafeStepUpdatesOnlySelectedVertex) {
    std::vector<Vec3> x = {
        Vec3(1.5, 0.25, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
    };
    const std::vector<Vec3> initial = x;
    const std::vector<Vec3> target = {
        Vec3(0.75, 0.25, 0.0),
        x[1], x[2], x[3],
    };

    BroadPhase broad_phase;
    auto& cache = broad_phase.mutable_cache();
    cache.node_boxes.assign(
        x.size(),
        AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    cache.vertex_nt.resize(x.size());
    cache.vertex_ss.resize(x.size());
    NodeTrianglePair pair{};
    pair.node = 0;
    pair.tri_v[0] = 1;
    pair.tri_v[1] = 2;
    pair.tri_v[2] = 3;
    cache.nt_pairs.push_back(pair);
    cache.vertex_nt[0].push_back(
        {/*pair_index=*/0, /*dof=*/0});

    const double safe_weight = per_vertex_safe_step(broad_phase, x, 0, target[0], /*safety=*/0.9, /*clip_ccd=*/true, /*use_ticcd=*/false, /*use_ogc=*/false);

    const Vec3 expected(0.825, 0.25, 0.0);
    EXPECT_DOUBLE_EQ(safe_weight, 0.9);
    for (int axis = 0; axis < 3; ++axis) EXPECT_DOUBLE_EQ(x[0][axis], expected[axis]);
    for (int vi = 1; vi < static_cast<int>(x.size()); ++vi)
        EXPECT_TRUE(x[vi].isApprox(initial[vi], 0.0));
}

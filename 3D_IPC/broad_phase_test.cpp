#include "broad_phase.h"
#include "physics.h"
#include "make_shape.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <set>
#include <tuple>
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

    static bool pair_shares_vertex(const SegmentSegmentPair& p) {
        return share_vertex(EdgeKey(p.v[0], p.v[1]), EdgeKey(p.v[2], p.v[3]));
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

TEST(BroadPhaseTest, SingleTriangleProducesNoSelfNodeTrianglePairs) {
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

TEST(BroadPhaseTest, FiltersOutNodeTrianglePairsThatShareVertices) {
const std::vector<Vec3> x = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 10.0);

for (const auto& p : broad.nt_pairs()) {
EXPECT_NE(p.node, p.tri_v[0]);
EXPECT_NE(p.node, p.tri_v[1]);
EXPECT_NE(p.node, p.tri_v[2]);
}
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

TEST(BroadPhaseTest, FiltersOutSegmentPairsThatShareVertices) {
const std::vector<Vec3> x = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 10.0);
for (const auto& p : broad.ss_pairs()) {
EXPECT_FALSE(pair_shares_vertex(p));
}
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

TEST(BroadPhaseTest, OutputPairsAreUnique) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_two_sheet_scene(x, v, mesh);

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.15);

const PairSets uniq = pair_sets_from_broad(broad);
EXPECT_EQ(uniq.nt.size(), broad.nt_pairs().size());
EXPECT_EQ(uniq.ss.size(), broad.ss_pairs().size());
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

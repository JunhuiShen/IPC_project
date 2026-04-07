#include "broad_phase.h"
#include "physics.h"
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
        mesh.ref_positions.resize(x.size(), Vec2::Zero());

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

TEST(BroadPhaseTest, RefreshMatchesInitializeAfterSingleNodeMove) {
std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.2, 0.2, 0.2), Vec3(1.2, 0.2, 0.2), Vec3(0.2, 1.2, 0.2),
};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase a, b;
a.initialize(x, v, mesh, 1.0, 0.15);

x[3] = Vec3(0.1, 0.1, 0.05);
a.refresh(x, v, mesh, 3, 1.0, 0.15, 0.0, 0.075);
b.initialize(x, v, mesh, 1.0, 0.15);

EXPECT_EQ(pair_sets_from_broad(a).nt, pair_sets_from_broad(b).nt);
EXPECT_EQ(pair_sets_from_broad(a).ss, pair_sets_from_broad(b).ss);
}

TEST(BroadPhaseTest, RefreshRemovesPairsWhenGeometrySeparates) {
std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
        Vec3(0.2, 0.2, 0.02), Vec3(5.0, 0.2, 0.02), Vec3(0.2, 5.0, 0.02),
};
const std::vector<Vec3> v(x.size(), Vec3::Zero());
const RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.03);
ASSERT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2));

x[3] = Vec3(0.2, 0.2, 3.0);
broad.refresh(x, v, mesh, 3, 1.0, 0.03, 0.0, 0.015);
EXPECT_FALSE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2));
}

TEST(BroadPhaseTest, RefreshPreservesFarAwayPairsOutsideLocalNeighborhood) {
std::vector<Vec3> x, v;
RefMesh mesh;
build_far_apart_double_scene(x, v, mesh);

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.05);

ASSERT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2));
ASSERT_TRUE(contains_nt_pair(broad.nt_pairs(), 9, 6, 7, 8));

x[3] = Vec3(0.15, 0.15, 0.20);
broad.refresh(x, v, mesh, 3, 1.0, 0.05, 0.0, 0.025);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 9, 6, 7, 8));
}

TEST(BroadPhaseTest, RefreshUpdatesPairsForMovedNodeAgainstUnchangedTriangles) {
std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
        Vec3(2.0, 2.0, 1.00), Vec3(3.0, 2.0, 1.00), Vec3(2.0, 3.0, 1.00),
};
std::vector<Vec3> v(x.size(), Vec3::Zero());
RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.02);
EXPECT_TRUE(broad.nt_pairs().empty());

x[3] = Vec3(0.2, 0.2, 0.01);
broad.refresh(x, v, mesh, 3, 1.0, 0.02, 0.0, 0.01);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 3, 0, 1, 2));
}

TEST(BroadPhaseTest, RefreshUpdatesPairsForChangedTriangleAgainstOtherNodes) {
std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.00), Vec3(1.0, 0.0, 0.00), Vec3(0.0, 1.0, 0.00),
        Vec3(0.2, 0.2, 0.30), Vec3(1.2, 0.2, 0.30), Vec3(0.2, 1.2, 0.30),
        Vec3(0.25, 0.25, 0.01),
};
std::vector<Vec3> v(x.size(), Vec3::Zero());
RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.02);
EXPECT_FALSE(contains_nt_pair(broad.nt_pairs(), 6, 3, 4, 5));

x[3] = Vec3(0.2, 0.2, 0.02);
broad.refresh(x, v, mesh, 3, 1.0, 0.02, 0.0, 0.01);

EXPECT_TRUE(contains_nt_pair(broad.nt_pairs(), 6, 3, 4, 5));
}

TEST(BroadPhaseTest, RefreshUpdatesSegmentSegmentPairsForChangedIncidentEdges) {
std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(3.0, 0.0, 0.0), Vec3(4.0, 1.0, 0.0), Vec3(3.0, 1.0, 0.0),
};
std::vector<Vec3> v(x.size(), Vec3::Zero());
RefMesh mesh = make_mesh(x, {{0, 1, 2}, {3, 4, 5}});

BroadPhase broad;
broad.initialize(x, v, mesh, 1.0, 0.0);
EXPECT_FALSE(contains_ss_pair(broad.ss_pairs(), EdgeKey(0, 1), EdgeKey(3, 4)));

x[3] = Vec3(0.0, 1.0, 0.0);
x[4] = Vec3(1.0, 0.0, 0.0);
broad.refresh(x, v, mesh, 3, 1.0, 0.0, 0.0, 0.0);
broad.refresh(x, v, mesh, 4, 1.0, 0.0, 0.0, 0.0);

EXPECT_TRUE(contains_ss_pair(broad.ss_pairs(), EdgeKey(0, 1), EdgeKey(3, 4)));
}
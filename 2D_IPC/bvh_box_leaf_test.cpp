#include "broad_phase/bvh.h"
#include <algorithm>
#include <gtest/gtest.h>

TEST(BVH, BuildAndQuery) {
    std::vector<AABB> boxes = {
        AABB(Vec2(0,0), Vec2(1,1)),  // box 0
        AABB(Vec2(2,1), Vec2(5,2)),  // box 1
        AABB(Vec2(2,0), Vec2(3,3)),  // box 2
        AABB(Vec2(4,1), Vec2(5,3)),  // box 3
    };

    std::vector<BVHNode> nodes;
    int root = build_bvh(boxes, nodes);

    // Tree has 2n-1 nodes for n leaves
    EXPECT_EQ(nodes.size(), 2u * boxes.size() - 1u);

    // Root bbox must cover all boxes: x=[0,5], y=[0,3]
    EXPECT_EQ(nodes[root].bbox.min.x, 0);
    EXPECT_EQ(nodes[root].bbox.min.y, 0);
    EXPECT_EQ(nodes[root].bbox.max.x, 5);
    EXPECT_EQ(nodes[root].bbox.max.y, 3);

    // Query: box covering only the left region — should hit box 0 only
    {
        AABB q(Vec2(0,0), Vec2(1,1));
        std::vector<int> hits;
        query_bvh(nodes, root, q, hits);
        ASSERT_EQ(hits.size(), 1u);
        EXPECT_EQ(hits[0], 0);
    }

    // Query: box covering the right region — should hit boxes 1, 2, 3
    {
        AABB q(Vec2(2,0), Vec2(5,3));
        std::vector<int> hits;
        query_bvh(nodes, root, q, hits);
        std::sort(hits.begin(), hits.end());
        ASSERT_EQ(hits.size(), 3u);
        EXPECT_EQ(hits[0], 1);
        EXPECT_EQ(hits[1], 2);
        EXPECT_EQ(hits[2], 3);
    }

    // Query: box outside all — no hits
    {
        AABB q(Vec2(10,10), Vec2(20,20));
        std::vector<int> hits;
        query_bvh(nodes, root, q, hits);
        EXPECT_TRUE(hits.empty());
    }

}

TEST(BVH, NodeBoxSafeStepClipsToBox) {
    BVHBroadPhase bp;
    Vec x = {0.0, 0.0, 1.0, 0.0};
    std::vector<std::pair<int, int>> edges = {{0, 1}};
    std::vector<double> radii = {0.5, 0.5};

    bp.initialize_node_radii(x, edges, radii, 0.0);

    EXPECT_DOUBLE_EQ(bp.node_box_safe_step(0, Vec2(0.0, 0.0), Vec2(2.0, 0.0)), 0.25);
    EXPECT_DOUBLE_EQ(bp.node_box_safe_step(0, Vec2(0.0, 0.0), Vec2(0.0, -1.0)), 0.5);
    EXPECT_DOUBLE_EQ(bp.node_box_safe_step(0, Vec2(0.0, 0.0), Vec2(0.1, 0.1)), 1.0);
}

TEST(BVH, SupportsNonconsecutiveEdgeEndpoints) {
    BVHBroadPhase bp;
    Vec x = {
        -1.0, 0.0,
         0.0, 0.1,
         2.0, 2.0,
         1.0, 0.0,
    };
    std::vector<std::pair<int, int>> edges = {{0, 3}};
    std::vector<double> radii(4, 0.2);

    bp.initialize_node_radii(x, edges, radii, 0.0);

    ASSERT_EQ(bp.pairs().size(), 1u);
    EXPECT_EQ(bp.pairs()[0].node, 1);
    EXPECT_EQ(bp.pairs()[0].seg0, 0);
    EXPECT_EQ(bp.pairs()[0].seg1, 3);
}

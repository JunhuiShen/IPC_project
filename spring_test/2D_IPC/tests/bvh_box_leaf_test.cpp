// Compile: clang++ -std=c++17 -O2 bvh_box_leaf_test.cpp bvh_box_leaf.cpp -o bvh_test \
//   -I/opt/homebrew/include -L/opt/homebrew/lib -lgtest -lgtest_main && ./bvh_test

#include "bvh_box_leaf.h"
#include <algorithm>
#include <gtest/gtest.h>

TEST(BVH, BuildQueryRefit) {
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

    // Move box 0 into the right region and refit, then re-query left region
    boxes[0] = AABB(Vec2(3,1), Vec2(4,2));
    refit_bvh(nodes, boxes);

    {
        AABB q(Vec2(0,0), Vec2(1,1));
        std::vector<int> hits;
        query_bvh(nodes, root, q, hits);
        EXPECT_TRUE(hits.empty());
    }
    {
        AABB q(Vec2(3,1), Vec2(4,2));
        std::vector<int> hits;
        query_bvh(nodes, root, q, hits);
        EXPECT_NE(std::find(hits.begin(), hits.end(), 0), hits.end());
    }
}

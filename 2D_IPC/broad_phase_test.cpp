#include "broad_phase.h"

#include <gtest/gtest.h>

#include <algorithm>

TEST(BVH, BuildAndQuery) {
    std::vector<AABB> boxes = {
        AABB(Vec2(0, 0), Vec2(1, 1)),
        AABB(Vec2(2, 1), Vec2(5, 2)),
        AABB(Vec2(2, 0), Vec2(3, 3)),
        AABB(Vec2(4, 1), Vec2(5, 3)),
    };

    std::vector<BVHNode> nodes;
    const int root = build_bvh(boxes, nodes);

    EXPECT_EQ(nodes.size(), 2u * boxes.size() - 1u);
    EXPECT_EQ(nodes[root].bbox.min.x, 0);
    EXPECT_EQ(nodes[root].bbox.min.y, 0);
    EXPECT_EQ(nodes[root].bbox.max.x, 5);
    EXPECT_EQ(nodes[root].bbox.max.y, 3);

    {
        const AABB query(Vec2(0, 0), Vec2(1, 1));
        std::vector<int> hits;
        query_bvh(nodes, root, query, hits);
        ASSERT_EQ(hits.size(), 1u);
        EXPECT_EQ(hits[0], 0);
    }

    {
        const AABB query(Vec2(2, 0), Vec2(5, 3));
        std::vector<int> hits;
        query_bvh(nodes, root, query, hits);
        std::sort(hits.begin(), hits.end());
        ASSERT_EQ(hits.size(), 3u);
        EXPECT_EQ(hits[0], 1);
        EXPECT_EQ(hits[1], 2);
        EXPECT_EQ(hits[2], 3);
    }

    {
        const AABB query(Vec2(10, 10), Vec2(20, 20));
        std::vector<int> hits;
        query_bvh(nodes, root, query, hits);
        EXPECT_TRUE(hits.empty());
    }
}

TEST(BVH, IterativeBuildHandlesManyBoxes) {
    constexpr int box_count = 4096;
    std::vector<AABB> boxes;
    boxes.reserve(box_count);
    for (int i = 0; i < box_count; ++i) {
        const double x = static_cast<double>(i);
        boxes.emplace_back(Vec2(x, -1.0), Vec2(x + 0.5, 1.0));
    }

    std::vector<BVHNode> nodes;
    const int root = build_bvh(boxes, nodes);

    ASSERT_EQ(root, 0);
    ASSERT_EQ(nodes.size(), 2u * boxes.size() - 1u);

    std::vector<int> hits;
    query_bvh(
        nodes, root, AABB(Vec2(-1.0, -2.0), Vec2(box_count + 1.0, 2.0)),
        hits);
    std::sort(hits.begin(), hits.end());

    ASSERT_EQ(hits.size(), boxes.size());
    for (int i = 0; i < box_count; ++i) EXPECT_EQ(hits[i], i);
}

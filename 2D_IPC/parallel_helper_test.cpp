#include "parallel_helper.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

bool has_edge(const std::vector<std::vector<int>>& graph, int a, int b) {
    return std::find(graph[a].begin(), graph[a].end(), b) != graph[a].end();
}

BroadPhase::Cache build_active_set(
    const Vec& x, const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& radii, double d_hat) {
    std::vector<AABB> blue_boxes;
    RedBoxes red_boxes;
    GreenBoxes green_boxes;
    build_blue_boxes(x, radii, blue_boxes);
    build_red_boxes(edges, blue_boxes, red_boxes);
    build_green_boxes(red_boxes, d_hat, green_boxes);
    return register_barrier_pairs_from_blue_and_green(
        edges, blue_boxes, green_boxes);
}

std::vector<int> colors_from_groups(const std::vector<std::vector<int>>& groups,
                                    int total_nodes) {
    std::vector<int> color(total_nodes, -1);
    for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
        for (int v : groups[c]) {
            EXPECT_GE(v, 0);
            EXPECT_LT(v, total_nodes);
            if (v >= 0 && v < total_nodes) {
                EXPECT_EQ(color[v], -1);
                color[v] = c;
            }
        }
    }
    return color;
}

}

TEST(ParallelHelper, BuildsExplicitBlueRedGreenBoxes) {
    Vec x = {
        0.0, 0.0,
        2.0, 1.0,
        1.0, 3.0,
    };
    std::vector<std::pair<int, int>> edges = {{0, 1}};
    std::vector<double> radii = {0.25, 0.5, 0.1};
    std::vector<AABB> blue_boxes;
    RedBoxes red_boxes;
    GreenBoxes green_boxes;
    build_blue_boxes(x, radii, blue_boxes);
    build_red_boxes(edges, blue_boxes, red_boxes);
    build_green_boxes(red_boxes, 0.2, green_boxes);

    ASSERT_EQ(blue_boxes.size(), 3u);
    ASSERT_EQ(red_boxes.segment.size(), 1u);
    ASSERT_EQ(green_boxes.segment.size(), 1u);
    EXPECT_DOUBLE_EQ(blue_boxes[0].min.x, -0.25);
    EXPECT_DOUBLE_EQ(blue_boxes[0].max.y, 0.25);
    EXPECT_DOUBLE_EQ(red_boxes.segment[0].min.x, -0.25);
    EXPECT_DOUBLE_EQ(red_boxes.segment[0].min.y, -0.25);
    EXPECT_DOUBLE_EQ(red_boxes.segment[0].max.x, 2.5);
    EXPECT_DOUBLE_EQ(red_boxes.segment[0].max.y, 1.5);
    EXPECT_DOUBLE_EQ(green_boxes.segment[0].min.x, -0.45);
    EXPECT_DOUBLE_EQ(green_boxes.segment[0].min.y, -0.45);
    EXPECT_DOUBLE_EQ(green_boxes.segment[0].max.x, 2.7);
    EXPECT_DOUBLE_EQ(green_boxes.segment[0].max.y, 1.7);
}

TEST(ParallelHelper, RegistersBlueGreenIntersection) {
    Vec x = {
        0.0, 0.0,
        2.0, 0.0,
        1.0, 0.35,
    };
    const auto cache = build_active_set(x, {{0, 1}}, {0.0, 0.0, 0.1}, 0.3);
    ASSERT_EQ(cache.pairs.size(), 1u);
    EXPECT_EQ(cache.pairs[0].node, 2);
    EXPECT_EQ(cache.pairs[0].seg0, 0);
    EXPECT_EQ(cache.pairs[0].seg1, 1);
}

TEST(ParallelHelper, SupportsNonconsecutiveEdgeEndpoints) {
    Vec x = {
        -1.0, 0.0,
         0.0, 0.1,
         2.0, 2.0,
         1.0, 0.0,
    };
    const auto cache = build_active_set(x, {{0, 3}}, std::vector<double>(4, 0.2), 0.0);
    ASSERT_EQ(cache.pairs.size(), 1u);
    EXPECT_EQ(cache.pairs[0].node, 1);
    EXPECT_EQ(cache.pairs[0].seg0, 0);
    EXPECT_EQ(cache.pairs[0].seg1, 3);
}

TEST(ParallelHelper, RejectsSeparatedBlueGreenBoxes) {
    Vec x = {
        0.0, 0.0,
        2.0, 0.0,
        1.0, 0.41,
    };
    const auto cache = build_active_set(x, {{0, 1}}, {0.0, 0.0, 0.1}, 0.3);
    EXPECT_TRUE(cache.pairs.empty());
}

TEST(ParallelHelper, RegistersBoxesTouchingAtBoundary) {
    Vec x = {
        0.0, 0.0,
        2.0, 0.0,
        1.0, 0.5,
    };
    const auto cache = build_active_set(x, {{0, 1}}, {0.0, 0.0, 0.25}, 0.25);
    ASSERT_EQ(cache.pairs.size(), 1u);
    EXPECT_EQ(cache.pairs[0].node, 2);
}

TEST(ParallelHelper, ConflictGraphContainsSpringsAndContacts) {
    std::vector<std::pair<int, int>> edges{{0, 2}, {2, 4}, {1, 3}};
    std::vector<NodeSegmentPair> pairs{{0, 3, 4}};
    const auto graph = union_adjacency(
        build_elastic_adj(edges, 5), build_contact_adj(pairs, 5));

    EXPECT_TRUE(has_edge(graph, 0, 2));
    EXPECT_TRUE(has_edge(graph, 2, 4));
    EXPECT_TRUE(has_edge(graph, 1, 3));
    EXPECT_TRUE(has_edge(graph, 0, 3));
    EXPECT_TRUE(has_edge(graph, 0, 4));
    EXPECT_TRUE(has_edge(graph, 3, 4));

    for (int node = 0; node < static_cast<int>(graph.size()); ++node) {
        EXPECT_FALSE(has_edge(graph, node, node));
        for (int neighbor : graph[node]) {
            EXPECT_TRUE(has_edge(graph, neighbor, node));
        }
    }
}

TEST(ParallelHelper, GreedyColoringSeparatesNeighbors) {
    std::vector<std::vector<int>> graph{
        {1, 2},
        {0, 2},
        {0, 1, 3},
        {2},
    };

    auto groups = greedy_color_conflict_graph(graph);
    const std::vector<int> color = colors_from_groups(groups, static_cast<int>(graph.size()));

    for (int v = 0; v < static_cast<int>(graph.size()); ++v) {
        ASSERT_GE(color[v], 0);
        for (int u : graph[v])
            EXPECT_NE(color[v], color[u]);
    }
}

TEST(ParallelHelper, MeshAndContactConflictsSeparateParallelGroups) {
    std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}};
    std::vector<NodeSegmentPair> pairs{{3, 1, 2}};

    const auto graph = union_adjacency(
        build_elastic_adj(edges, 5), build_contact_adj(pairs, 5));
    const auto groups = greedy_color_conflict_graph(graph);
    const std::vector<int> color = colors_from_groups(groups, 5);

    // Elastic coupling.
    EXPECT_NE(color[0], color[1]);
    EXPECT_NE(color[1], color[2]);

    // Contact coupling is a clique over node 3 and segment endpoints 1, 2.
    EXPECT_NE(color[3], color[1]);
    EXPECT_NE(color[3], color[2]);
    EXPECT_NE(color[1], color[2]);

    // Every node, including isolated node 4, is scheduled exactly once.
    for (int node = 0; node < 5; ++node) {
        EXPECT_GE(color[node], 0);
    }
}

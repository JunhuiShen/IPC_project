#include "parallel_helper.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

bool has_edge(const std::vector<std::vector<int>>& graph, int a, int b) {
    return std::find(graph[a].begin(), graph[a].end(), b) != graph[a].end();
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

TEST(ParallelHelper, ConflictGraphContainsSpringsAndContacts) {
    std::vector<std::pair<int, int>> edges{{0, 2}, {2, 4}, {1, 3}};
    std::vector<NodeSegmentPair> pairs{{0, 3, 4}};
    auto graph = build_conflict_graph(edges, pairs, 5);

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
    const std::vector<int> color =
            colors_from_groups(groups, static_cast<int>(graph.size()));

    for (int v = 0; v < static_cast<int>(graph.size()); ++v) {
        ASSERT_GE(color[v], 0);
        for (int u : graph[v])
            EXPECT_NE(color[v], color[u]);
    }
}

TEST(ParallelHelper, MeshAndContactConflictsSeparateParallelGroups) {
    std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}};
    std::vector<NodeSegmentPair> pairs{{3, 1, 2}};

    const auto graph = build_conflict_graph(edges, pairs, 5);
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

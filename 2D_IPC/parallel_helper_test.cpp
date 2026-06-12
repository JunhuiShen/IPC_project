#include "parallel_helper.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

bool has_edge(const std::vector<std::vector<int>>& graph, int a, int b) {
    return std::find(graph[a].begin(), graph[a].end(), b) != graph[a].end();
}

}

TEST(ParallelHelper, ConflictGraphContainsSpringsAndContacts) {
    std::vector<std::pair<int, int>> edges{{0, 2}, {2, 4}, {1, 3}};
    std::vector<contact::NodeSegmentPair> pairs{{0, 3, 4}};
    auto graph = build_conflict_graph(edges, pairs, 5);

    EXPECT_TRUE(has_edge(graph, 0, 2));
    EXPECT_TRUE(has_edge(graph, 2, 4));
    EXPECT_TRUE(has_edge(graph, 1, 3));
    EXPECT_TRUE(has_edge(graph, 0, 3));
    EXPECT_TRUE(has_edge(graph, 0, 4));
}

TEST(ParallelHelper, GreedyColoringSeparatesNeighbors) {
    std::vector<std::vector<int>> graph{
        {1, 2},
        {0, 2},
        {0, 1, 3},
        {2},
    };

    auto groups = greedy_color_conflict_graph(graph);
    std::vector<int> color(graph.size(), -1);

    for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
        for (int v : groups[c]) color[v] = c;
    }

    for (int v = 0; v < static_cast<int>(graph.size()); ++v) {
        ASSERT_GE(color[v], 0);
        for (int u : graph[v])
            EXPECT_NE(color[v], color[u]);
    }
}

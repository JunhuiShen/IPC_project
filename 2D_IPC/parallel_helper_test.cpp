#include "parallel_helper.h"
#include <gtest/gtest.h>
#include <algorithm>

namespace {

bool has_edge(const std::vector<std::vector<int>>& graph, int a, int b) {
    return std::find(graph[a].begin(), graph[a].end(), b) != graph[a].end();
}

}

TEST(ParallelHelper, ConflictGraphContainsSpringsAndContacts) {
    std::vector<double> mass0(3, 1.0);
    std::vector<double> mass1(2, 1.0);

    std::vector<BlockView> blocks{
        BlockView{nullptr, nullptr, nullptr, &mass0, nullptr, 0, nullptr, 0},
        BlockView{nullptr, nullptr, nullptr, &mass1, nullptr, 0, nullptr, 3},
    };

    std::vector<physics::NodeSegmentPair> pairs{{0, 3, 4}};
    auto graph = build_conflict_graph(blocks, pairs, 5);

    EXPECT_TRUE(has_edge(graph, 0, 1));
    EXPECT_TRUE(has_edge(graph, 1, 2));
    EXPECT_TRUE(has_edge(graph, 3, 4));
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

TEST(ParallelHelper, GlobalToBlockLocalMap) {
    std::vector<double> mass0(2, 1.0);
    std::vector<double> mass1(3, 1.0);

    std::vector<BlockView> blocks{
        BlockView{nullptr, nullptr, nullptr, &mass0, nullptr, 0, nullptr, 0},
        BlockView{nullptr, nullptr, nullptr, &mass1, nullptr, 0, nullptr, 2},
    };

    auto map = build_global_to_block_local(blocks, 5);

    EXPECT_EQ(map[0], std::make_pair(0, 0));
    EXPECT_EQ(map[1], std::make_pair(0, 1));
    EXPECT_EQ(map[2], std::make_pair(1, 0));
    EXPECT_EQ(map[4], std::make_pair(1, 2));
}


#include "parallel_helper.h"
#include <algorithm>

namespace {

void add_graph_edge(std::vector<std::vector<int>>& graph, int a, int b) {
    if (a == b || a < 0 || b < 0 ||
        a >= static_cast<int>(graph.size()) || b >= static_cast<int>(graph.size())) {
        return;
    }
    graph[a].push_back(b);
    graph[b].push_back(a);
}

}

std::vector<std::vector<int>>
build_conflict_graph(const std::vector<BlockView>& blocks,
                     const std::vector<physics::NodeSegmentPair>& barrier_pairs,
                     int total_nodes) {
    std::vector<std::vector<int>> graph(total_nodes);

    for (const auto& b : blocks) {
        for (int i = 0; i + 1 < b.size(); ++i) {
            add_graph_edge(graph, b.offset + i, b.offset + i + 1);
        }
    }

    for (const auto& p : barrier_pairs) {
        add_graph_edge(graph, p.node, p.seg0);
        add_graph_edge(graph, p.node, p.seg1);
        add_graph_edge(graph, p.seg0, p.seg1);
    }

    for (auto& nbrs : graph) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }

    return graph;
}

std::vector<std::vector<int>>
greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph) {
    const int n = static_cast<int>(graph.size());
    std::vector<int> color(n, -1);
    std::vector<std::vector<int>> groups;

    for (int v = 0; v < n; ++v) {
        std::vector<char> used(groups.size(), 0);
        for (int u : graph[v]) {
            if (u >= 0 && u < n && color[u] >= 0)
                used[color[u]] = 1;
        }

        int c = 0;
        while (c < static_cast<int>(used.size()) && used[c]) ++c;
        if (c == static_cast<int>(groups.size()))
            groups.emplace_back();
        color[v] = c;
        groups[c].push_back(v);
    }

    return groups;
}

std::vector<std::pair<int, int>>
build_global_to_block_local(const std::vector<BlockView>& blocks, int total_nodes) {
    std::vector<std::pair<int, int>> map(total_nodes, {-1, -1});
    for (int b = 0; b < static_cast<int>(blocks.size()); ++b) {
        for (int i = 0; i < blocks[b].size(); ++i) {
            const int who = blocks[b].offset + i;
            if (who >= 0 && who < total_nodes)
                map[who] = {b, i};
        }
    }
    return map;
}


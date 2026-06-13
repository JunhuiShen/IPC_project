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

} // namespace

std::vector<std::vector<int>>
build_conflict_graph(const std::vector<std::pair<int, int>>& edges,
                     const std::vector<NodeSegmentPair>& contact_pairs,
                     int total_nodes) {
    std::vector<std::vector<int>> graph(total_nodes);

    for (const auto& [a, b] : edges) {
        add_graph_edge(graph, a, b);
    }

    for (const auto& pair : contact_pairs) {
        // A node-segment barrier couples all three participating nodes.
        add_graph_edge(graph, pair.node, pair.seg0);
        add_graph_edge(graph, pair.node, pair.seg1);
        add_graph_edge(graph, pair.seg0, pair.seg1);
    }

    for (auto& neighbors : graph) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    return graph;
}

std::vector<std::vector<int>>
greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph) {
    const int n = static_cast<int>(graph.size());
    std::vector<int> color(n, -1);
    std::vector<std::vector<int>> groups;

    for (int vertex = 0; vertex < n; ++vertex) {
        std::vector<char> used(groups.size(), 0);
        for (int neighbor : graph[vertex]) {
            if (neighbor >= 0 && neighbor < n && color[neighbor] >= 0) {
                used[color[neighbor]] = 1;
            }
        }

        int group = 0;
        while (group < static_cast<int>(used.size()) && used[group]) {
            ++group;
        }
        if (group == static_cast<int>(groups.size())) {
            groups.emplace_back();
        }
        color[vertex] = group;
        groups[group].push_back(vertex);
    }

    return groups;
}

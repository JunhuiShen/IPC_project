#include "parallel_helper.h"

#include <algorithm>
#include <cmath>
#include <iterator>

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


AABB arc_node_aabb(const Vec2& x_com, const double theta, const Vec2& X, const double eps) {
    const double R         = norm(X);
    // X is the node's fixed material-space offset from the rigid body's COM.
    // phi_local is its angle inside the body; adding theta rotates that offset
    // into world space. The blue box below encloses the node's arc over
    // [phi_world - eps, phi_world + eps].
    const double phi_local = std::atan2(X.y, X.x);
    const double phi_world = phi_local + theta;
    const double theta_a   = phi_world - eps;
    const double theta_b   = phi_world + eps;

    auto point_at = [&](double t) -> Vec2 {
        return Vec2(std::cos(t), std::sin(t)) * R + x_com;
    };

    const Vec2 p_a = point_at(theta_a);
    const Vec2 p_b = point_at(theta_b);

    double x_min = std::min(p_a.x, p_b.x);
    double x_max = std::max(p_a.x, p_b.x);
    double y_min = std::min(p_a.y, p_b.y);
    double y_max = std::max(p_a.y, p_b.y);

    static const double extreme_points[4] = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};
    static const Vec2   offsets[4]   = {{1.0,0.0},{0.0,1.0},{-1.0,0.0},{0.0,-1.0}};

    for (int k = 0; k < 4; ++k) {
        double t = std::fmod(extreme_points[k] - theta_a, 2.0 * M_PI);
        if (t < 0.0) t += 2.0 * M_PI;
        t += theta_a;
        if (t <= theta_b) {
            const Vec2 p = x_com + offsets[k] * R;
            x_min = std::min(x_min, p.x); x_max = std::max(x_max, p.x);
            y_min = std::min(y_min, p.y); y_max = std::max(y_max, p.y);
        }
    }

    return AABB({x_min, y_min}, {x_max, y_max});
}

void build_blue_boxes(
    const Vec& positions, const std::vector<double>& node_radii,
    std::vector<AABB>& blue_boxes) {
    const int total_nodes = static_cast<int>(positions.size());
    blue_boxes.resize(total_nodes);

    for (int node = 0; node < total_nodes; ++node) {
        const double radius =
            node < static_cast<int>(node_radii.size()) ? std::max(0.0, node_radii[node]) : 0.0;
        const Vec2 x = get_xi(positions, node);
        blue_boxes[node] = AABB(
            Vec2(x.x - radius, x.y - radius),
            Vec2(x.x + radius, x.y + radius));
    }
}

void build_blue_boxes_rb(const Vec& positions,
                          const Vec& x_coms,
                          const std::vector<double>& thetas,
                          double eps,
                          const std::vector<double>& com_radii,
                          const std::vector<std::vector<int>>& rb_nodes,
                          const std::vector<Vec>& ref_positions,
                          std::vector<AABB>& blue_boxes) {
    blue_boxes.resize(positions.size());

    for (int rb = 0; rb < static_cast<int>(rb_nodes.size()); ++rb) {
        const Vec2   x_com  = x_coms[rb];
        const double theta  = thetas[rb];
        const double r_com  = com_radii[rb];

        for (int i = 0; i < static_cast<int>(rb_nodes[rb].size()); ++i) {
            const int  node     = rb_nodes[rb][i];
            const Vec2 X        = get_xi(ref_positions[rb], i);
            const Vec2 node_pos = get_xi(positions, node);

            // All nodes of this rb share one translation box based on COM displacement
            blue_boxes[node] = AABB(
                Vec2(node_pos.x - r_com, node_pos.y - r_com),
                Vec2(node_pos.x + r_com, node_pos.y + r_com));

            // The COM can move anywhere inside this translation square. Since
            // translating an arc by a box reaches its axis extrema at the box
            // corners, union the same rotation arc at all four COM corners.
            const Vec2 corners[2] = {
                {x_com.x - r_com, x_com.y - r_com},
                {x_com.x + r_com, x_com.y + r_com},
            };
            for (const Vec2& c : corners)
                blue_boxes[node].expand(arc_node_aabb(c, theta, X, eps));
        }
    }
}

void build_red_boxes(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<AABB>& blue_boxes, RedBoxes& red_boxes) {
    red_boxes.segment.resize(edges.size());
    for (int edge = 0; edge < static_cast<int>(edges.size()); ++edge) {
        const auto [seg0, seg1] = edges[edge];
        AABB red_box = blue_boxes[seg0];
        red_box.expand(blue_boxes[seg1]);
        red_boxes.segment[edge] = red_box;
    }
}

void build_green_boxes(
    const RedBoxes& red_boxes, double d_hat, GreenBoxes& green_boxes) {
    green_boxes.segment.resize(red_boxes.segment.size());
    for (int edge = 0; edge < static_cast<int>(red_boxes.segment.size()); ++edge) {
        const AABB& red_box = red_boxes.segment[edge];
        green_boxes.segment[edge] = AABB(
            Vec2(red_box.min.x - d_hat, red_box.min.y - d_hat),
            Vec2(red_box.max.x + d_hat, red_box.max.y + d_hat));
    }
}

BroadPhase::Cache register_barrier_pairs_from_blue_and_green(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<AABB>& blue_boxes,
    const GreenBoxes& green_boxes) {
    BroadPhase::Cache cache;
    cache.blue_boxes = blue_boxes;
    cache.segment_leaf_edges = edges;
    cache.green_bvh_root = build_bvh(green_boxes.segment, cache.green_bvh_nodes);

    for (int node = 0; node < static_cast<int>(blue_boxes.size()); ++node) {
        std::vector<int> hits;
        query_bvh(cache.green_bvh_nodes, cache.green_bvh_root, blue_boxes[node], hits);
        for (int edge_index : hits) {
            const auto [seg0, seg1] = edges[edge_index];
            if (node == seg0 || node == seg1) continue;
            cache.pairs.push_back({node, seg0, seg1});
        }
    }

    return cache;
}

std::vector<std::vector<int>>
build_elastic_adj(const std::vector<std::pair<int, int>>& edges, int total_nodes) {
    std::vector<std::vector<int>> graph(total_nodes);
    for (const auto& [a, b] : edges) add_graph_edge(graph, a, b);
    for (auto& neighbors : graph) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    return graph;
}

std::vector<std::vector<int>>
build_contact_adj(const std::vector<NodeSegmentPair>& contact_pairs, int total_nodes) {
    std::vector<std::vector<int>> graph(total_nodes);
    for (const auto& pair : contact_pairs) {
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
union_adjacency(const std::vector<std::vector<int>>& a,
                const std::vector<std::vector<int>>& b) {
    const int total_nodes = static_cast<int>(std::max(a.size(), b.size()));
    std::vector<std::vector<int>> graph(total_nodes);
    for (int node = 0; node < total_nodes; ++node) {
        const std::vector<int> empty;
        const auto& row_a = node < static_cast<int>(a.size()) ? a[node] : empty;
        const auto& row_b = node < static_cast<int>(b.size()) ? b[node] : empty;
        graph[node].reserve(row_a.size() + row_b.size());
        std::set_union(
            row_a.begin(), row_a.end(), row_b.begin(), row_b.end(),
            std::back_inserter(graph[node]));
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

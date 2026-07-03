#include "broad_phase.h"

#include <algorithm>

namespace {

bool is_invalid_pair(int node, int seg0, int seg1) {
    return node == seg0 || node == seg1;
}

// Sweep one node through a full velocity field over dt.
AABB build_swept_node_box(const Vec& x, const Vec& v, int node, double dt) {
    const Vec2 x0 = get_xi(x, node);
    const Vec2 velocity = get_xi(v, node);
    const Vec2 x1{x0.x + dt * velocity.x, x0.y + dt * velocity.y};
    return AABB(
        Vec2(std::min(x0.x, x1.x), std::min(x0.y, x1.y)),
        Vec2(std::max(x0.x, x1.x), std::max(x0.y, x1.y)));
}

// Sweep one segment through a full velocity field over dt; both endpoints may move.
AABB build_swept_segment_box(const Vec& x, const Vec& v, int seg0, int seg1, double dt) {
    const Vec2 x0 = get_xi(x, seg0);
    const Vec2 x1 = get_xi(x, seg1);
    const Vec2 v0 = get_xi(v, seg0);
    const Vec2 v1 = get_xi(v, seg1);
    return AABB(
        Vec2(std::min({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}), std::min({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y})),
        Vec2(std::max({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}), std::max({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y})));
}

} // namespace

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    out.clear();
    if (boxes.empty()) return -1;

    std::vector<int> indices(boxes.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) indices[i] = i;

    struct BuildTask {
        int node;
        int start;
        int end;
    };

    out.reserve(2 * boxes.size() - 1);
    out.emplace_back();

    std::vector<BuildTask> stack;
    stack.push_back({0, 0, static_cast<int>(indices.size())});

    while (!stack.empty()) {
        const BuildTask task = stack.back();
        stack.pop_back();

        AABB node_box;
        for (int i = task.start; i < task.end; ++i) node_box.expand(boxes[indices[i]]);
        out[task.node].bbox = node_box;

        const int count = task.end - task.start;
        if (count == 1) {
            out[task.node].leafIndex = indices[task.start];
            continue;
        }

        const Vec2 extent = node_box.extent();
        const int axis = extent.y > extent.x ? 1 : 0;
        const int mid = task.start + count / 2;
        std::nth_element(
            indices.begin() + task.start, indices.begin() + mid, indices.begin() + task.end,
            [&](int a, int b) {
                const Vec2 ca = boxes[a].centroid();
                const Vec2 cb = boxes[b].centroid();
                return axis == 0 ? ca.x < cb.x : ca.y < cb.y;
            });

        const int left = static_cast<int>(out.size());
        out.emplace_back();
        const int right = static_cast<int>(out.size());
        out.emplace_back();
        out[task.node].left = left;
        out[task.node].right = right;

        stack.push_back({right, mid, task.end});
        stack.push_back({left, task.start, mid});
    }

    return 0;
}

void query_bvh(const std::vector<BVHNode>& nodes, int root, const AABB& query, std::vector<int>& hits) {
    if (root < 0) return;

    std::vector<int> stack{root};
    while (!stack.empty()) {
        const BVHNode& node = nodes[stack.back()];
        stack.pop_back();
        if (!aabb_intersects(node.bbox, query)) continue;

        if (node.leafIndex >= 0) {
            hits.push_back(node.leafIndex);
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }
}

std::vector<NodeSegmentPair> BroadPhase::build_ccd_candidates(const Vec& x, const Vec& v, const std::vector<std::pair<int, int>>& edges, double dt) const {
    std::vector<AABB> segment_boxes;
    segment_boxes.reserve(edges.size());
    for (const auto& [seg0, seg1] : edges) {
        segment_boxes.push_back(build_swept_segment_box(x, v, seg0, seg1, dt));
    }

    std::vector<BVHNode> bvh_nodes;
    const int root = build_bvh(segment_boxes, bvh_nodes);
    std::vector<NodeSegmentPair> pairs;
    const int total_nodes = static_cast<int>(x.size());

    for (int node = 0; node < total_nodes; ++node) {
        std::vector<int> hits;
        query_bvh(bvh_nodes, root, build_swept_node_box(x, v, node, dt), hits);
        for (int edge_index : hits) {
            const auto [seg0, seg1] = edges[edge_index];
            if (!is_invalid_pair(node, seg0, seg1)) pairs.push_back({node, seg0, seg1});
        }
    }

    return pairs;
}

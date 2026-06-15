#include "broad_phase.h"

#include <algorithm>
#include <tuple>

namespace {

bool is_invalid_pair(int node, int seg0, int seg1) {
    return node == seg0 || node == seg1;
}

AABB build_swept_node_box(const Vec& x, const Vec& v, int node, double dt, double pad) {
    const Vec2 x0 = get_xi(x, node);
    const Vec2 velocity = get_xi(v, node);
    const Vec2 x1{x0.x + dt * velocity.x, x0.y + dt * velocity.y};
    return AABB(
        Vec2(std::min(x0.x, x1.x) - pad, std::min(x0.y, x1.y) - pad),
        Vec2(std::max(x0.x, x1.x) + pad, std::max(x0.y, x1.y) + pad));
}

AABB build_swept_segment_box(
    const Vec& x, const Vec& v, int seg0, int seg1, double dt, double pad) {
    const Vec2 x0 = get_xi(x, seg0);
    const Vec2 x1 = get_xi(x, seg1);
    const Vec2 v0 = get_xi(v, seg0);
    const Vec2 v1 = get_xi(v, seg1);
    return AABB(
        Vec2(std::min({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) - pad,
             std::min({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) - pad),
        Vec2(std::max({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) + pad,
             std::max({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) + pad));
}

std::vector<NodeSegmentPair> build_swept_candidates(
    const Vec& x, const Vec& v, const std::vector<std::pair<int, int>>& edges,
    double dt, double node_pad) {
    std::vector<AABB> segment_boxes;
    segment_boxes.reserve(edges.size());
    for (const auto& [seg0, seg1] : edges) {
        segment_boxes.push_back(build_swept_segment_box(x, v, seg0, seg1, dt, 0.0));
    }

    std::vector<BVHNode> bvh_nodes;
    const int root = build_bvh(segment_boxes, bvh_nodes);
    std::vector<NodeSegmentPair> pairs;
    const int total_nodes = static_cast<int>(x.size() / 2);

    for (int node = 0; node < total_nodes; ++node) {
        std::vector<int> hits;
        query_bvh(bvh_nodes, root, build_swept_node_box(x, v, node, dt, node_pad), hits);
        for (int edge_index : hits) {
            const auto [seg0, seg1] = edges[edge_index];
            if (!is_invalid_pair(node, seg0, seg1)) pairs.push_back({node, seg0, seg1});
        }
    }

    return pairs;
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

void query_bvh(
    const std::vector<BVHNode>& nodes, int root, const AABB& query,
    std::vector<int>& hits) {
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

double BroadPhase::node_box_safe_step(
    int node, const Vec2& x0, const Vec2& displacement) const {
    if (node < 0 || node >= static_cast<int>(cache_.blue_boxes.size())) return 1.0;

    const AABB& box = cache_.blue_boxes[node];
    constexpr double eps = 1.0e-12;
    double alpha = 1.0;

    auto clip_axis = [&](double x, double d, double lo, double hi) {
        if (x < lo - eps || x > hi + eps) {
            alpha = 0.0;
        } else if (d > 0.0) {
            alpha = std::min(alpha, (hi - x) / d);
        } else if (d < 0.0) {
            alpha = std::min(alpha, (lo - x) / d);
        }
    };

    clip_axis(x0.x, displacement.x, box.min.x, box.max.x);
    clip_axis(x0.y, displacement.y, box.min.y, box.max.y);
    return std::clamp(alpha, 0.0, 1.0);
}

std::vector<NodeSegmentPair> BroadPhase::build_ccd_candidates(
    const Vec& x, const Vec& v, const std::vector<std::pair<int, int>>& edges,
    double dt) const {
    return build_swept_candidates(x, v, edges, dt, 0.0);
}

std::vector<NodeSegmentPair> BroadPhase::build_ccd_candidates_for_node(
    int who, const Vec& x, const Vec& v_newton,
    const std::vector<std::pair<int, int>>& edges, double dt) const {
    std::vector<NodeSegmentPair> result;
    if (cache_.green_bvh_root < 0) return result;

    std::vector<int> hits;
    query_bvh(
        cache_.green_bvh_nodes, cache_.green_bvh_root,
        build_swept_node_box(x, v_newton, who, dt, 0.0), hits);

    for (int edge_index : hits) {
        const auto [seg0, seg1] = cache_.segment_leaf_edges[edge_index];
        if (!is_invalid_pair(who, seg0, seg1)) result.push_back({who, seg0, seg1});
    }

    const int total_nodes = static_cast<int>(x.size() / 2);
    for (const auto& [seg0, seg1] : edges) {
        if (seg0 != who && seg1 != who) continue;
        const AABB segment_box = build_swept_segment_box(x, v_newton, seg0, seg1, dt, 0.0);
        for (int node = 0; node < total_nodes; ++node) {
            if (is_invalid_pair(node, seg0, seg1)) continue;
            if (aabb_intersects(build_swept_node_box(x, v_newton, node, dt, 0.0), segment_box)) {
                result.push_back({node, seg0, seg1});
            }
        }
    }

    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return std::tie(a.node, a.seg0, a.seg1) < std::tie(b.node, b.seg0, b.seg1);
    });
    result.erase(std::unique(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.node == b.node && a.seg0 == b.seg0 && a.seg1 == b.seg1;
    }), result.end());
    return result;
}

std::vector<NodeSegmentPair> BroadPhase::build_trust_region_candidates(
    const Vec& x, const Vec& v, const std::vector<std::pair<int, int>>& edges,
    double dt, double motion_pad) const {
    return build_swept_candidates(x, v, edges, dt, motion_pad);
}

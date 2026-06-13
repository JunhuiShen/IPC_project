#include "bvh.h"
#include <algorithm>

using Cache = BVHBroadPhase::Cache;
using NSP   = contact::NodeSegmentPair;

// ======================================================
// BVH tree — build and query
// ======================================================

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    out.clear();
    if (boxes.empty()) return -1;

    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
        indices[i] = static_cast<int>(i);

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
        for (int i = task.start; i < task.end; ++i)
            node_box.expand(boxes[indices[i]]);
        out[task.node].bbox = node_box;

        const int count = task.end - task.start;
        if (count == 1) {
            out[task.node].leafIndex = indices[task.start];
            continue;
        }

        const Vec2 extent = node_box.extent();
        const int axis = (extent.y > extent.x) ? 1 : 0;
        const int mid = task.start + count / 2;
        auto compare_centroid = [&](int a, int b) {
            const Vec2 ca = boxes[a].centroid();
            const Vec2 cb = boxes[b].centroid();
            return axis == 0 ? ca.x < cb.x : ca.y < cb.y;
        };
        std::nth_element(
                indices.begin() + task.start,
                indices.begin() + mid,
                indices.begin() + task.end,
                compare_centroid);

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

void query_bvh(const std::vector<BVHNode>& nodes, int nodeIdx,
               const AABB& query, std::vector<int>& hits) {
    if (nodeIdx < 0) return;

    std::vector<int> stack;
    stack.push_back(nodeIdx);

    while (!stack.empty()) {
        const BVHNode& n = nodes[stack.back()];
        stack.pop_back();
        if (!aabb_intersects(n.bbox, query)) continue;

        if (n.leafIndex >= 0) {
            hits.push_back(n.leafIndex);
            continue;
        }

        stack.push_back(n.left);
        stack.push_back(n.right);
    }
}

// ======================================================
// Internal helpers (file-local, operate on Cache)
// ======================================================

static inline bool is_invalid_pair(int node, int seg0, int seg1) {
    return (node == seg0) || (node == seg1);
}

static inline AABB build_node_box(const Vec& x, const Vec& v, int i, double dt, double pad) {
    Vec2 x0 = get_xi(x, i);
    Vec2 vi = get_xi(v, i);
    Vec2 x1{x0.x + dt * vi.x, x0.y + dt * vi.y};

    return AABB(
            Vec2(std::min(x0.x, x1.x) - pad, std::min(x0.y, x1.y) - pad),
            Vec2(std::max(x0.x, x1.x) + pad, std::max(x0.y, x1.y) + pad)
    );
}

static inline AABB build_node_box_from_radius(const Vec& x, int i, double r) {
    Vec2 xi = get_xi(x, i);
    return AABB(Vec2(xi.x - r, xi.y - r), Vec2(xi.x + r, xi.y + r));
}

static inline AABB build_segment_box(const Vec& x, const Vec& v,
                                     int seg0, int seg1, double dt, double pad) {
    Vec2 x0 = get_xi(x, seg0);
    Vec2 x1 = get_xi(x, seg1);
    Vec2 v0 = get_xi(v, seg0);
    Vec2 v1 = get_xi(v, seg1);

    return AABB(
            Vec2(std::min({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) - pad,
                 std::min({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) - pad),
            Vec2(std::max({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) + pad,
                 std::max({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) + pad)
    );
}

static inline AABB build_segment_red_box(const std::vector<AABB>& blue_boxes,
                                         int seg0, int seg1) {
    AABB red_box = blue_boxes[seg0];
    red_box.expand(blue_boxes[seg1]);
    return red_box;
}

static inline AABB augment_box(const AABB& box, double pad) {
    return AABB(
            Vec2(box.min.x - pad, box.min.y - pad),
            Vec2(box.max.x + pad, box.max.y + pad));
}

static inline void query_node(Cache& c, int node) {
    if (c.green_bvh_root < 0) return;

    std::vector<int> hits;
    query_bvh(c.green_bvh_nodes, c.green_bvh_root, c.blue_boxes[node], hits);

    for (int leaf_k : hits) {
        const auto [seg0, seg1] = c.segment_leaf_edges[leaf_k];
        if (is_invalid_pair(node, seg0, seg1)) continue;
        c.pairs.push_back({node, seg0, seg1});
    }
}

// ======================================================
// BVHBroadPhase implementation
// ======================================================

void BVHBroadPhase::build(const Vec& x, const Vec& v,
                          const std::vector<std::pair<int, int>>& edges,
                          double dt, double node_pad, double seg_pad) {
    Cache c;
    const int total = static_cast<int>(x.size() / 2);

    c.blue_boxes.resize(total);
    c.red_segment_boxes.reserve(edges.size());
    c.green_segment_boxes.reserve(edges.size());

    for (int i = 0; i < total; ++i)
        c.blue_boxes[i] = build_node_box(x, v, i, dt, node_pad);

    for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
        const auto [seg0, seg1] = edges[e];
        c.segment_leaf_edges.push_back(edges[e]);

        const AABB red_box = build_segment_box(x, v, seg0, seg1, dt, 0.0);
        c.red_segment_boxes.push_back(red_box);
        c.green_segment_boxes.push_back(augment_box(red_box, seg_pad));
    }

    c.green_bvh_root = build_bvh(c.green_segment_boxes, c.green_bvh_nodes);

    for (int i = 0; i < total; ++i)
        query_node(c, i);

    cache_ = std::move(c);
}

void BVHBroadPhase::initialize_node_radii(
        const Vec& x,
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<double>& node_radii,
        double d_hat) {
    Cache c;
    const int total = static_cast<int>(x.size() / 2);

    c.blue_boxes.resize(total);
    c.red_segment_boxes.reserve(edges.size());
    c.green_segment_boxes.reserve(edges.size());

    for (int i = 0; i < total; ++i) {
        const double r = (i < static_cast<int>(node_radii.size())) ? node_radii[i] : 0.0;
        c.blue_boxes[i] = build_node_box_from_radius(x, i, r);
    }

    for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
        const auto [seg0, seg1] = edges[e];
        c.segment_leaf_edges.push_back(edges[e]);

        const AABB red_box = build_segment_red_box(c.blue_boxes, seg0, seg1);
        c.red_segment_boxes.push_back(red_box);
        c.green_segment_boxes.push_back(augment_box(red_box, d_hat));
    }

    c.green_bvh_root = build_bvh(c.green_segment_boxes, c.green_bvh_nodes);

    for (int i = 0; i < total; ++i)
        query_node(c, i);

    cache_ = std::move(c);
}

const std::vector<NSP>& BVHBroadPhase::pairs() const {
    return cache_.pairs;
}

double BVHBroadPhase::node_box_safe_step(int node, const Vec2& x0, const Vec2& displacement) const {
    if (node < 0 || node >= static_cast<int>(cache_.blue_boxes.size()))
        return 1.0;

    const AABB& box = cache_.blue_boxes[node];
    constexpr double eps = 1.0e-12;
    double alpha = 1.0;

    auto clip_axis = [&](double x, double d, double lo, double hi) {
        if (x < lo - eps || x > hi + eps) {
            alpha = 0.0;
            return;
        }
        if (d > 0.0) {
            alpha = std::min(alpha, (hi - x) / d);
        } else if (d < 0.0) {
            alpha = std::min(alpha, (lo - x) / d);
        }
    };

    clip_axis(x0.x, displacement.x, box.min.x, box.max.x);
    clip_axis(x0.y, displacement.y, box.min.y, box.max.y);

    return std::clamp(alpha, 0.0, 1.0);
}

std::vector<NSP> BVHBroadPhase::build_ccd_candidates(const Vec& x, const Vec& v,
                                                     const std::vector<std::pair<int, int>>& edges,
                                                     double dt) {
    BVHBroadPhase tmp;
    tmp.build(x, v, edges, dt, /*node_pad=*/0.0, /*seg_pad=*/0.0);
    return tmp.cache_.pairs;
}

std::vector<NSP> BVHBroadPhase::build_ccd_candidates_for_node(
        int who, const Vec& x, const Vec& v_newton,
        const std::vector<std::pair<int, int>>& edges, double dt) {

    std::vector<NSP> result;
    if (cache_.green_bvh_root < 0) return result;

    const int total = static_cast<int>(x.size() / 2);

    // 1. Node 'who' sweeps — query segment BVH for intersecting segments
    AABB node_box = build_node_box(x, v_newton, who, dt, 0.0);
    std::vector<int> hits;
    query_bvh(cache_.green_bvh_nodes, cache_.green_bvh_root, node_box, hits);

    for (int leaf_k : hits) {
        const auto [seg0, seg1] = cache_.segment_leaf_edges[leaf_k];
        if (is_invalid_pair(who, seg0, seg1)) continue;
        result.push_back({who, seg0, seg1});
    }

    // 2. Edges containing 'who' also sweep — scan all nodes against them.
    auto check_segment = [&](int seg0, int seg1) {
        AABB seg_box = build_segment_box(x, v_newton, seg0, seg1, dt, 0.0);
        for (int node = 0; node < total; ++node) {
            if (is_invalid_pair(node, seg0, seg1)) continue;
            if (aabb_intersects(build_node_box(x, v_newton, node, dt, 0.0), seg_box))
                result.push_back({node, seg0, seg1});
        }
    };

    for (const auto& [seg0, seg1] : edges) {
        if (seg0 == who || seg1 == who) check_segment(seg0, seg1);
    }

    // Deduplicate
    std::sort(result.begin(), result.end(), [](const NSP& a, const NSP& b) {
        return std::tie(a.node, a.seg0, a.seg1) < std::tie(b.node, b.seg0, b.seg1);
    });
    result.erase(std::unique(result.begin(), result.end(), [](const NSP& a, const NSP& b) {
        return a.node == b.node && a.seg0 == b.seg0 && a.seg1 == b.seg1;
    }), result.end());

    return result;
}

std::vector<NSP> BVHBroadPhase::build_trust_region_candidates(const Vec& x, const Vec& v,
                                                              const std::vector<std::pair<int, int>>& edges,
                                                              double dt, double motion_pad) {
    BVHBroadPhase tmp;
    tmp.build(x, v, edges, dt, /*node_pad=*/motion_pad, /*seg_pad=*/0.0);
    return tmp.cache_.pairs;
}

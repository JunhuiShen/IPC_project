#pragma once

#include "ipc_math.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

struct NodeSegmentPair {
    int node;
    int seg0;
    int seg1;
};

struct AABB {
    Vec2 min, max;

    AABB() {
        const double inf = std::numeric_limits<double>::infinity();
        min = Vec2(inf, inf);
        max = Vec2(-inf, -inf);
    }

    AABB(const Vec2& a, const Vec2& b) : min(a), max(b) {}

    void expand(const AABB& box) {
        min.x = std::min(min.x, box.min.x);
        min.y = std::min(min.y, box.min.y);
        max.x = std::max(max.x, box.max.x);
        max.y = std::max(max.y, box.max.y);
    }

    Vec2 centroid() const { return (min + max) * 0.5; }
    Vec2 extent() const { return max - min; }
};

inline bool aabb_intersects(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y;
}

struct BVHNode {
    AABB bbox;
    int left = -1;
    int right = -1;
    int leafIndex = -1;
};

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out);
void query_bvh(
    const std::vector<BVHNode>& nodes, int root, const AABB& query,
    std::vector<int>& hits);

class BroadPhase {
public:
    struct Cache {
        std::vector<AABB> blue_boxes;

        std::vector<std::pair<int, int>> segment_leaf_edges;
        std::vector<BVHNode> green_bvh_nodes;
        int green_bvh_root = -1;

        std::vector<NodeSegmentPair> pairs;
    };

    Cache& mutable_cache() { return cache_; }
    const Cache& cache() const { return cache_; }
    const std::vector<NodeSegmentPair>& pairs() const { return cache_.pairs; }

    std::vector<NodeSegmentPair> build_ccd_candidates(const Vec& x, const Vec& v, const std::vector<std::pair<int, int>>& edges, double dt) const;

private:
    Cache cache_;
};

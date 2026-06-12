#pragma once

#include "broad_phase.h"
#include <vector>
#include <limits>
#include <algorithm>

// ======================================================
// AABB — axis-aligned bounding box
// ======================================================

struct AABB {
    Vec2 min, max;

    AABB() {
        double inf = std::numeric_limits<double>::infinity();
        min = Vec2(inf, inf);
        max = Vec2(-inf, -inf);
    }
    AABB(const Vec2& a, const Vec2& b) : min(a), max(b) {}

    void expand(const AABB& o) {
        min.x = std::min(min.x, o.min.x); min.y = std::min(min.y, o.min.y);
        max.x = std::max(max.x, o.max.x); max.y = std::max(max.y, o.max.y);
    }
    Vec2 centroid() const { return (min + max) * 0.5; }
    Vec2 extent()   const { return max - min; }
};

inline bool aabb_intersects(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y;
}

// ======================================================
// BVH — flat-array median-split tree over AABBs
// ======================================================

struct BVHNode {
    AABB bbox;
    int left = -1, right = -1;
    int leafIndex = -1;
};

int  build_bvh  (const std::vector<AABB>& boxes, std::vector<BVHNode>& out);
void query_bvh  (const std::vector<BVHNode>& nodes, int nodeIdx,
                 const AABB& query, std::vector<int>& hits);

// ======================================================
// BVHBroadPhase — BVH-based implementation of BroadPhase
//
// Builds BVHs over segment AABBs and supports O(log n)
// node queries.
// ======================================================

class BVHBroadPhase : public BroadPhase {
public:
    void initialize_node_radii(const Vec& x,
                               const std::vector<std::pair<int, int>>& edges,
                               const std::vector<double>& node_radii,
                               double d_hat) override;

    const std::vector<contact::NodeSegmentPair>& pairs() const override;

    double node_box_safe_step(int node, const Vec2& x0, const Vec2& displacement) const override;

    std::vector<contact::NodeSegmentPair>
    build_ccd_candidates(const Vec& x, const Vec& v,
                         const std::vector<std::pair<int, int>>& edges,
                         double dt) override;

    std::vector<contact::NodeSegmentPair>
    build_ccd_candidates_for_node(int who, const Vec& x, const Vec& v_newton,
                                  const std::vector<std::pair<int, int>>& edges,
                                  double dt) override;

    std::vector<contact::NodeSegmentPair>
    build_trust_region_candidates(const Vec& x, const Vec& v,
                                  const std::vector<std::pair<int, int>>& edges,
                                  double dt, double motion_pad) override;

    struct Cache {
        std::vector<AABB> node_boxes;

        std::vector<std::pair<int, int>> seg_leaf_edges;
        std::vector<BVHNode> seg_bvh_nodes;
        int                  seg_bvh_root = -1;

        std::vector<contact::NodeSegmentPair> pairs;
    };

private:
    Cache cache_;

    void build(const Vec& x, const Vec& v,
               const std::vector<std::pair<int, int>>& edges,
               double dt, double node_pad, double seg_pad);

    void build_from_node_radii(const Vec& x,
                               const std::vector<std::pair<int, int>>& edges,
                               const std::vector<double>& node_radii,
                               double d_hat);
};

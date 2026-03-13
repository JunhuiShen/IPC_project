#pragma once

#include "broad_phase.h"
#include <unordered_map>
#include <cstdint>
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
    void expand(const Vec2& p) {
        min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y);
        max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y);
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
void refit_bvh  (std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes);
void query_bvh  (const std::vector<BVHNode>& nodes, int nodeIdx,
                 const AABB& query, std::vector<int>& hits);

// ======================================================
// BVHBroadPhase — BVH-based implementation of BroadPhase
//
// Builds a BVH over swept segment AABBs at initialization.
// Supports O(log n) node queries and incremental refit
// after a single node move.
// ======================================================

class BVHBroadPhase : public BroadPhase {
public:
    void initialize(const Vec& x, const Vec& v,
                    int N_left, int N_right, double dt, double dhat) override;

    void refresh(const Vec& x, const Vec& v,
                 int moved_node, int N_left, int N_right,
                 double dt, double node_pad, double seg_pad) override;

    const std::vector<physics::NodeSegmentPair>& pairs() const override;

    std::vector<physics::NodeSegmentPair>
        build_ccd_candidates(const Vec& x, const Vec& v,
                             int N_left, int N_right, double dt) override;

    std::vector<physics::NodeSegmentPair>
        build_trust_region_candidates(const Vec& x, const Vec& v,
                                      int N_left, int N_right,
                                      double dt, double motion_pad) override;

public:
    struct Cache {
        std::vector<AABB> node_boxes;
        std::vector<AABB> segment_boxes;
        std::vector<char> segment_valid;

        std::vector<int>     seg_leaf_to_seg0;
        std::vector<int>     seg0_to_leaf;
        std::vector<AABB>    seg_bvh_boxes;
        std::vector<BVHNode> seg_bvh_nodes;
        int                  seg_bvh_root = -1;

        std::vector<physics::NodeSegmentPair>          pairs;
        std::unordered_map<std::uint64_t, std::size_t> pair_index;
    };

    Cache cache_;
    int   N_left_  = 0;
    int   N_right_ = 0;

    void build(const Vec& x, const Vec& v,
               int N_left, int N_right, double dt,
               double node_pad, double seg_pad);
};

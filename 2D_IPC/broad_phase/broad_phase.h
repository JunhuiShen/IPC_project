#pragma once

#include "../ipc_math.h"

#include <utility>
#include <vector>

struct NodeSegmentPair {
    int node;
    int seg0;
    int seg1;
};

// ======================================================
// BroadPhase — base class for broad-phase collision detection
//
// Manages the barrier active set and provides one-shot
// candidate builds for step filtering.
// Swap implementations by swapping the subclass.
// ======================================================

class BroadPhase {
public:
    virtual void initialize_node_radii(const Vec& x,
                                       const std::vector<std::pair<int, int>>& edges,
                                       const std::vector<double>& node_radii,
                                       double d_hat) = 0;

    // Current barrier active-set pairs
    virtual const std::vector<NodeSegmentPair>& pairs() const = 0;

    virtual double node_box_safe_step(int node, const Vec2& x0, const Vec2& displacement) const = 0;

    // One-shot candidate builds for step filtering
    virtual std::vector<NodeSegmentPair>
    build_ccd_candidates(const Vec& x, const Vec& v,
                         const std::vector<std::pair<int, int>>& edges,
                         double dt) = 0;

    // Efficient single-node CCD candidates using the existing BVH (no rebuild)
    virtual std::vector<NodeSegmentPair>
    build_ccd_candidates_for_node(int who, const Vec& x, const Vec& v_newton,
                                  const std::vector<std::pair<int, int>>& edges,
                                  double dt) = 0;

    virtual std::vector<NodeSegmentPair>
    build_trust_region_candidates(const Vec& x, const Vec& v,
                                  const std::vector<std::pair<int, int>>& edges,
                                  double dt, double motion_pad) = 0;

    virtual ~BroadPhase() = default;
};

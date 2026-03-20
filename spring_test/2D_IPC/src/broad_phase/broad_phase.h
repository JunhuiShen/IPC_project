#pragma once

#include "../ipc_math.h"
#include "../physics.h"
#include <vector>

// ======================================================
// BroadPhase — base class for broad-phase collision detection
//
// Manages a persistent barrier pair cache and provides
// one-shot candidate builds for step filtering.
// Swap implementations by swapping the subclass.
// ======================================================

class BroadPhase {
public:
    // Set up the persistent barrier pair cache (called once per timestep)
    virtual void initialize(const Vec& x, const Vec& v,
                            const std::vector<char>& segment_valid,
                            double dt, double dhat) = 0;

    // Incremental update after one node moved (called per Newton step)
    virtual void refresh(const Vec& x, const Vec& v,
                         int moved_node,
                         double dt, double node_pad, double seg_pad) = 0;

    // Current barrier active-set pairs
    virtual const std::vector<physics::NodeSegmentPair>& pairs() const = 0;

    // One-shot candidate builds for step filtering
    virtual std::vector<physics::NodeSegmentPair>
    build_ccd_candidates(const Vec& x, const Vec& v,
                         const std::vector<char>& segment_valid,
                         double dt) = 0;

    // Efficient single-node CCD candidates using the existing BVH (no rebuild)
    // Default falls back to full rebuild — override for performance.
    virtual std::vector<physics::NodeSegmentPair>
    build_ccd_candidates_for_node(int who, const Vec& x, const Vec& v_newton,
                                  const std::vector<char>& segment_valid,
                                  double dt) {
        return build_ccd_candidates(x, v_newton, segment_valid, dt);
    }

    virtual std::vector<physics::NodeSegmentPair>
    build_trust_region_candidates(const Vec& x, const Vec& v,
                                  const std::vector<char>& segment_valid,
                                  double dt, double motion_pad) = 0;

    virtual ~BroadPhase() = default;
};
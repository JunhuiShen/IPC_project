#pragma once

#include "../ipc_math.h"
#include "../physics.h"
#include <vector>

// ======================================================
// StepFilter — base class for Newton step filtering
//
// Subclasses return a safe fraction omega in [0,1]
// such that x - omega*dx remains collision-free.
// ======================================================

class StepFilter {
public:
    virtual double compute_safe_step(int who_global, const Vec2& dx,
                                     const Vec& x_global,
                                     const std::vector<physics::NodeSegmentPair>& candidates,
                                     double eta) = 0;
    virtual ~StepFilter() = default;
};

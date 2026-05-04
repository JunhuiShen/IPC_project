#pragma once

#include "step_filter.h"

// ======================================================
// Trust-region step filter
//
// Limits step size based on current separation:
//   w = clamp(eta * d0 / |dx_total|, 0, 1)
// Cheaper than CCD, more conservative for large motions.
// ======================================================

namespace step_filter::trust_region {

    // Low-level free function (reusable without class instantiation)
    double weight(const Vec2& xi, const Vec2& dxi,
                  const Vec2& xj, const Vec2& dxj,
                  const Vec2& xk, const Vec2& dxk,
                  double eta = 0.4);

} // namespace step_filter::trust_region

class TrustRegionFilter : public StepFilter {
public:
    double compute_safe_step(int who_global, const Vec2& dx,
                             const Vec& x_global,
                             const std::vector<physics::NodeSegmentPair>& candidates,
                             double eta) override;
};

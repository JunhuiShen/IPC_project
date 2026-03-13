#pragma once

#include "initial_guess.h"
#include "../physics.h"
#include <vector>

// ======================================================
// TrustRegionGuess
//
// Limits the explicit step based on current separation:
//   alpha = min over all pairs of eta * d0 / |dx_total|
//   xnew  = x + alpha * dt * v
// ======================================================

namespace initial_guess::trust_region {
    double global_safe_step(const Vec& x, const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt, double eta = 0.4);
} // namespace initial_guess::trust_region

class TrustRegionGuess : public InitialGuess {
public:
    void apply(Chain& left, Chain& right,
               Vec& xnew_left, Vec& xnew_right,
               Vec& x_combined, Vec& v_combined,
               double dt, double eta) override;
};

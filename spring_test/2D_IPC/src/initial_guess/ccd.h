#pragma once

#include "initial_guess.h"
#include "../physics.h"
#include <vector>

// ======================================================
// CCDGuess
//
// Globally safe explicit step using CCD:
//   omega = min over all pairs of eta * t_hit
//   xnew  = x + omega * dt * v
// ======================================================

namespace initial_guess::ccd {
    double global_safe_step(const Vec& x, const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt, double eta = 0.9);
} // namespace initial_guess::ccd

class CCDGuess : public InitialGuess {
public:
    void apply(Chain& left, Chain& right,
               Vec& xnew_left, Vec& xnew_right,
               Vec& x_combined, Vec& v_combined,
               double dt, double eta) override;
};

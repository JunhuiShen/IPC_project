#pragma once

#include "initial_guess.h"
#include "../physics.h"
#include <vector>

// ======================================================
// CCD initial guess
//
// Globally safe explicit step using CCD over an arbitrary
// list of blocks/chains:
//
//   omega = min over all candidate pairs of eta * t_hit
//   xnew  = x + omega * dt * v
//
// This matches the original monolithic code.
// ======================================================

namespace initial_guess::ccd {

    double global_safe_step(const Vec& x,
                            const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt,
                            double eta = 0.9);

    void apply(const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined,
               const std::vector<char>& segment_valid,
               double dt,
               double eta);

} // namespace initial_guess::ccd
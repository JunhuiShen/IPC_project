#pragma once

#include "initial_guess.h"
#include "../physics.h"
#include <vector>

// ======================================================
// Trust-region projected initial guess
//
// Matches the original monolithic implementation:
// - operates on arbitrary blocks
// - builds global x/v over all blocks
// - uses global segment_valid
// - preserves pinned nodes
// ======================================================

namespace initial_guess::trust_region {

    double global_safe_step(const Vec& x,
                            const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt,
                            double eta = 0.4);

    void apply(const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined,
               const std::vector<char>& segment_valid,
               double dt,
               double eta);

} // namespace initial_guess::trust_region
#pragma once

#include "initial_guess.h"

namespace initial_guess::trivial {

    // xnew = current positions, v_combined = current velocities
    void apply(const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined);

} // namespace initial_guess::trivial

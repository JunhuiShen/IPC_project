#pragma once

#include "initial_guess.h"

namespace initial_guess::trivial {

    void apply(const State2D& state, Vec& xnew, Vec& solver_velocity);

} // namespace initial_guess::trivial

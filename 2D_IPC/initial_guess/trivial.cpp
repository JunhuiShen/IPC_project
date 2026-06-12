#include "trivial.h"

namespace initial_guess::trivial {

    void apply(const State2D& state, Vec& xnew, Vec& solver_velocity) {
        xnew = state.x;
        solver_velocity = state.v;
    }

} // namespace initial_guess::trivial

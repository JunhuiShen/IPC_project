#include "trivial.h"

namespace initial_guess::trivial {

    void apply(const State2D& state, Vec& xnew) {
        xnew = state.x;
    }

} // namespace initial_guess::trivial

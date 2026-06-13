#include "trivial.h"

void apply_trivial_initial_guess(const State2D& state, Vec& xnew) {
    xnew = state.x;
}

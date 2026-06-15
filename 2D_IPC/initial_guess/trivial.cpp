#include "trivial.h"

void apply_trivial_initial_guess(const DeformedState& state, Vec& xnew) {
    xnew = state.deformed_positions;
}

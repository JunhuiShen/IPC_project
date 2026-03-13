#include "trivial.h"

void TrivialGuess::apply(Chain& left, Chain& right,
                         Vec& xnew_left, Vec& xnew_right,
                         Vec& x_combined, Vec& v_combined,
                         double dt, double eta) {
    xnew_left  = left.x;
    xnew_right = right.x;
    for (int i = 0; i < left.N;  ++i) set_xi(v_combined, i,          get_xi(left.v,  i));
    for (int i = 0; i < right.N; ++i) set_xi(v_combined, left.N + i, get_xi(right.v, i));
    combine_positions(x_combined, left.x, right.x, left.N, right.N);
}

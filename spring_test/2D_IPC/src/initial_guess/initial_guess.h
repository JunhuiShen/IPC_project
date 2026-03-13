#pragma once

#include "../chain.h"

// ======================================================
// InitialGuess — base class for timestep predictors
//
// Computes a collision-free starting point for the
// nonlinear solve at the beginning of each timestep.
// Swap strategies by swapping the subclass.
// ======================================================

class InitialGuess {
public:
    virtual void apply(Chain& left, Chain& right,
                       Vec& xnew_left, Vec& xnew_right,
                       Vec& x_combined, Vec& v_combined,
                       double dt, double eta) = 0;

    virtual ~InitialGuess() = default;
};

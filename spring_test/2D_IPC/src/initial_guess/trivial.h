#pragma once

#include "initial_guess.h"

// xnew = x (current positions), v_combined = current velocities
class TrivialGuess : public InitialGuess {
public:
    void apply(Chain& left, Chain& right,
               Vec& xnew_left, Vec& xnew_right,
               Vec& x_combined, Vec& v_combined,
               double dt, double eta) override;
};

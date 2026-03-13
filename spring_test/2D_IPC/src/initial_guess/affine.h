#pragma once

#include "initial_guess.h"

// ======================================================
// AffineGuess
//
// Fits a global rigid-body velocity field
//   V(x) = vhat + omega * R * (x - xcom)
// to the current velocities in a least-squares sense,
// then predicts: xnew = x + dt * V(x).
// ======================================================

namespace initial_guess::affine {

    struct Params {
        double omega;
        Vec2   vhat;
        Vec2   xcom;
    };

    Params compute(const Chain& A, const Chain& B);
    Vec2   velocity_at(const Params& ap, const Vec2& x);
    void   apply_to_chain(const Params& ap, const Chain& c, Vec& xnew, double dt);

} // namespace initial_guess::affine

class AffineGuess : public InitialGuess {
public:
    void apply(Chain& left, Chain& right,
               Vec& xnew_left, Vec& xnew_right,
               Vec& x_combined, Vec& v_combined,
               double dt, double eta) override;
};

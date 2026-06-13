#pragma once

#include "initial_guess.h"

// ======================================================
// Affine initial guess
//
// Fits a rigid-body velocity field
//   V(x) = vhat + omega * R * (x - xcom)
// to the current velocities in a least-squares sense,
// excluding pinned nodes from the fit.
// Then predicts: xnew = x + dt * V(x),
// while keeping pinned nodes fixed.
// ======================================================

namespace initial_guess::affine {

    struct Params {
        double omega;
        Vec2   vhat;
        Vec2   xcom;
    };

    Params compute_affine_params(const State2D& state);
    Vec2   velocity_at(const Params& ap, const Vec2& x);
    void   apply(const Params& ap, const State2D& state,
                 Vec& xnew, double dt);

} // namespace initial_guess::affine

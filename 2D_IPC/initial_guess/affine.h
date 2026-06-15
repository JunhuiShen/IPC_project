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

struct AffineInitialGuessParams {
    double omega;
    Vec2   vhat;
    Vec2   xcom;
};

AffineInitialGuessParams compute_affine_initial_guess_params(
    const DeformedState& state, const RefMesh& ref_mesh, const std::vector<Pin>& pins);
Vec2 affine_initial_guess_velocity(const AffineInitialGuessParams& params, const Vec2& x);
void apply_affine_initial_guess(const AffineInitialGuessParams& params,
                                const DeformedState& state, const std::vector<Pin>& pins,
                                Vec& xnew, double dt);

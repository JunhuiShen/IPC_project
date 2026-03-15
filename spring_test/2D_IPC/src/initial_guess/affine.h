#pragma once

#include "initial_guess.h"

// ======================================================
// Affine initial guess
//
// Fits a global rigid-body velocity field
//   V(x) = vhat + omega * R * (x - xcom)
// to the current velocities in a least-squares sense,
// excluding pinned nodes from the fit.
// Then predicts: xnew = x + dt * V(x),
// while keeping pinned nodes fixed.
//
// This version matches the original monolithic code:
// it works on an arbitrary list of contiguous blocks.
// ======================================================

namespace initial_guess::affine {

    struct Params {
        double omega;
        Vec2   vhat;
        Vec2   xcom;
    };

    Params compute_affine_params_global(const std::vector<BlockRef>& blocks);
    Vec2   velocity_at(const Params& ap, const Vec2& x);
    void   apply_to_block(const Params& ap, const BlockRef& b, double dt);
    void   build_v_combined_from_affine(Vec& v_combined,
                                        const std::vector<BlockRef>& blocks,
                                        const Params& ap);

} // namespace initial_guess::affine

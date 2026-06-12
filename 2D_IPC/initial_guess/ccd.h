#pragma once

#include "../broad_phase/broad_phase.h"
#include "initial_guess.h"

#include <vector>

// ======================================================
// CCD initial guess
//
// Globally safe explicit step using CCD over explicit edges:
//
//   omega = min over all candidate pairs of eta * t_hit
//   xnew  = x + omega * dt * v
//
// ======================================================

namespace initial_guess::ccd {

    double global_safe_step(const Vec& x,
                            const Vec& v,
                            const std::vector<contact::NodeSegmentPair>& pairs,
                            double dt,
                            double eta = 0.9);

    void apply(const State2D& state, const RefMesh& ref_mesh,
               Vec& xnew, Vec& solver_velocity,
               double dt,
               double eta);

} // namespace initial_guess::ccd

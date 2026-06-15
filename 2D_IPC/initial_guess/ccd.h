#pragma once

#include "../broad_phase.h"
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

double ccd_initial_guess_safe_step(const Vec& x, const Vec& v,
                                   const std::vector<NodeSegmentPair>& pairs,
                                   double dt, double eta = 0.9);

void apply_ccd_initial_guess(const State2D& state, const RefMesh& ref_mesh,
                             Vec& xnew, double dt, double eta);

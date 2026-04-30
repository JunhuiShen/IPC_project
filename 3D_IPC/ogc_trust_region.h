#pragma once

#include "IPC_math.h"
#include "broad_phase.h"

#include <vector>

// Trust-region narrow phase

struct TrustRegionResult {
    double omega = 1.0;   // safe scale in [0, 1]
    double d0    = 0.0;   // initial separation distance
    double M     = 0.0;   // total displacement magnitude
};

// Per-vertex conservative bound 
//  b_v = gamma_p * min d0 across all NT/SS pairs incident to vi
double compute_trust_region_bound_for_vertex(int vi,
                                             const std::vector<Vec3>& x,
                                             const BroadPhase::Cache& bp_cache,
                                             double gamma_p);

// Gauss-Seidel substep: only one vertex of the pair moves, with
// displacement delta. Caller is responsible for tracking which vertex the delta belongs to.
TrustRegionResult trust_region_vertex_triangle_gauss_seidel(
        const Vec3& x,  const Vec3& x1, const Vec3& x2, const Vec3& x3,
        const Vec3& delta,
        double eta = 0.4);

TrustRegionResult trust_region_edge_edge_gauss_seidel(
        const Vec3& a1, const Vec3& a2, const Vec3& b1, const Vec3& b2,
        const Vec3& delta,
        double eta = 0.4);

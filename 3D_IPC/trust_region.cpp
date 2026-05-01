#include "trust_region.h"

#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <limits>

namespace {

constexpr double kMotionEps = 1.0e-30;
constexpr double kEtaLo     = 1.0e-12;
constexpr double kEtaHi     = 0.5 - 1.0e-12;

double clamp_eta(double eta) {
    return std::min(std::max(eta, kEtaLo), kEtaHi);
}

TrustRegionResult make_result(double d0, double M, double eta) {
    TrustRegionResult r;
    r.d0 = d0;
    r.M  = M;
    if (d0 <= 0.0) {
        r.omega = 0.0; // already in contact: trust region degenerate
    } else if (M <= kMotionEps) {
        r.omega = 1.0;  // nothing moves
    } else {
        r.omega = std::min(1.0, eta * d0 / M);
    }
    return r;
}

}  // namespace

TrustRegionResult trust_region_vertex_triangle_gauss_seidel(const Vec3& x,  const Vec3& x1,
    const Vec3& x2, const Vec3& x3, const Vec3& delta, double eta) {
    const double d0 = node_triangle_distance(x, x1, x2, x3).distance;
    return make_result(d0, delta.norm(), clamp_eta(eta));
}

TrustRegionResult trust_region_edge_edge_gauss_seidel(const Vec3& a1, const Vec3& a2, const Vec3& b1, const Vec3& b2,
    const Vec3& delta, double eta) {
    const double d0 = segment_segment_distance(a1, a2, b1, b2).distance;
    return make_result(d0, delta.norm(), clamp_eta(eta));
}

double compute_trust_region_bound_for_vertex(int vi,
                                             const std::vector<Vec3>& x,
                                             const BroadPhase::Cache& bp_cache,
                                             double gamma_p) {
    double d0_min = std::numeric_limits<double>::infinity();

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_nt.size())) {
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            const double d0 = node_triangle_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]).distance;
            if (d0 < d0_min) d0_min = d0;
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_ss.size())) {
        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            const double d0 = segment_segment_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]).distance;
            if (d0 < d0_min) d0_min = d0;
        }
    }

    return gamma_p * d0_min;  // +inf when vi has no incident pairs
}

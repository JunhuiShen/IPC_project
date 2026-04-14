#include "trust_region.h"

#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <algorithm>

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
        r.omega = 0.0;          // already in contact: trust region degenerate
    } else if (M <= kMotionEps) {
        r.omega = 1.0;          // nothing moves
    } else {
        r.omega = std::min(1.0, eta * d0 / M);
    }
    return r;
}

}  // namespace

TrustRegionResult trust_region_vertex_triangle(const Vec3& x,  const Vec3& dx, const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3, double eta) {
    const double d0 = node_triangle_distance(x, x1, x2, x3).distance;
    const double M  = dx.norm() + dx1.norm() + dx2.norm() + dx3.norm();
    return make_result(d0, M, clamp_eta(eta));
}

TrustRegionResult trust_region_edge_edge(const Vec3& a1, const Vec3& da1, const Vec3& a2, const Vec3& da2,
        const Vec3& b1, const Vec3& db1, const Vec3& b2, const Vec3& db2, double eta) {
    const double d0 = segment_segment_distance(a1, a2, b1, b2).distance;
    const double M  = da1.norm() + da2.norm() + db1.norm() + db2.norm();
    return make_result(d0, M, clamp_eta(eta));
}

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

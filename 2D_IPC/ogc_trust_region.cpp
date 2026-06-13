#include "ogc_trust_region.h"

#include "node_segment_distance.h"

#include <algorithm>

namespace {

constexpr double kMotionEps = 1.0e-12;
constexpr double kEtaLo = 1.0e-12;
constexpr double kEtaHi = 0.5 - 1.0e-12;

double clamp_eta(double eta) {
    return std::min(std::max(eta, kEtaLo), kEtaHi);
}

TrustRegionResult2D make_result(double d0, double M, double eta) {
    TrustRegionResult2D r;
    r.d0 = d0;
    r.M = M;
    if (d0 <= 0.0) {
        r.omega = 0.0;
    } else if (M <= kMotionEps) {
        r.omega = 1.0;
    } else {
        r.omega = std::min(1.0, eta * d0 / M);
    }
    return r;
}

} // namespace

TrustRegionResult2D trust_region_node_segment_gauss_seidel(
        const Vec2& xi, const Vec2& dxi,
        const Vec2& xj, const Vec2& dxj,
        const Vec2& xk, const Vec2& dxk,
        double eta) {
    double s = 0.0;
    Vec2 p{}, r{};
    double d0 = node_segment_distance(xi, xj, xk, s, p, r);
    d0 = std::max(d0, 1.0e-12);

    const double M = norm(dxi) + norm(dxj) + norm(dxk);
    return make_result(d0, M, clamp_eta(eta));
}

#include "trust_region.h"
#include <algorithm>

namespace step_filter::trust_region {

    using namespace math;

    double weight(const Vec2& xi, const Vec2& dxi,
                  const Vec2& xj, const Vec2& dxj,
                  const Vec2& xk, const Vec2& dxk,
                  double eta) {
        double s;
        Vec2 p{}, r{};
        double d0 = physics::node_segment_distance(xi, xj, xk, s, p, r);

        constexpr double eps = 1e-12;
        d0 = std::max(d0, eps);

        const double M = norm(dxi) + norm(dxj) + norm(dxk);
        if (M <= eps) return 1.0;

        return std::max(0.0, std::min(1.0, eta * d0 / M));
    }

} // namespace step_filter::trust_region

double TrustRegionFilter::compute_safe_step(int who_global, const Vec2& dx,
                                            const Vec& x_global,
                                            const std::vector<physics::NodeSegmentPair>& candidates,
                                            double eta) {
    using namespace math;
    double omega = 1.0;
    Vec2 full{-dx.x, -dx.y};

    for (const auto& c : candidates) {
        if (who_global != c.node && who_global != c.seg0 && who_global != c.seg1)
            continue;

        Vec2 xi = get_xi(x_global, c.node);
        Vec2 xj = get_xi(x_global, c.seg0);
        Vec2 xk = get_xi(x_global, c.seg1);

        Vec2 dxi{}, dxj{}, dxk{};
        if      (who_global == c.node) dxi = full;
        else if (who_global == c.seg0) dxj = full;
        else if (who_global == c.seg1) dxk = full;

        omega = std::min(omega, step_filter::trust_region::weight(xi, dxi, xj, dxj, xk, dxk, eta));
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

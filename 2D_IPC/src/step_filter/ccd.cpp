#include "ccd.h"
#include <cmath>
#include <algorithm>

namespace step_filter::ccd {

    using namespace math;

    bool point_segment_2d(const Vec2& x1, const Vec2& dx1,
                          const Vec2& x2, const Vec2& dx2,
                          const Vec2& x3, const Vec2& dx3,
                          double& t_out, double eps) {
        Vec2 x21  = sub(x1, x2),  x32  = sub(x3, x2);
        Vec2 dx21 = sub(dx1, dx2), dx32 = sub(dx3, dx2);

        double a = cross(dx32, dx21);
        double b = cross(dx32, x21) + cross(x32, dx21);
        double c = cross(x32, x21);

        double t_candidates[2];
        int num_roots = 0;

        if (std::fabs(a) < eps) {
            if (std::fabs(b) < eps) return false;
            double t = -c / b;
            if (t >= 0.0 && t <= 1.0) t_candidates[num_roots++] = t;
        } else {
            double D = b * b - 4.0 * a * c;
            if (D < 0.0) return false;

            double sqrtD = std::sqrt(std::max(D, 0.0));
            double s = (b >= 0.0) ? 1.0 : -1.0;
            double q = -0.5 * (b + s * sqrtD);

            double t1 = q / a, t2 = c / q;
            if (t1 >= 0.0 && t1 <= 1.0) t_candidates[num_roots++] = t1;
            if (t2 >= 0.0 && t2 <= 1.0) t_candidates[num_roots++] = t2;
        }

        if (num_roots == 0) return false;

        double t_star = t_candidates[0];
        if (num_roots == 2 && t_candidates[1] < t_star) t_star = t_candidates[1];

        Vec2 x1t = add(x1, scale(dx1, t_star));
        Vec2 x2t = add(x2, scale(dx2, t_star));
        Vec2 x3t = add(x3, scale(dx3, t_star));

        Vec2 seg = sub(x3t, x2t);
        Vec2 rel = sub(x1t, x2t);

        double seg_len2 = norm2(seg);
        if (seg_len2 < eps) return false;

        double s_param = dot(rel, seg) / seg_len2;
        if (s_param < 0.0 || s_param > 1.0) return false;

        t_out = t_star;
        return true;
    }

    double safe_step(const Vec2& x1, const Vec2& dx1,
                     const Vec2& x2, const Vec2& dx2,
                     const Vec2& x3, const Vec2& dx3,
                     double eta) {
        ++total_tests;

        double t_hit;
        if (!point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            return 1.0;

        ++total_collisions;
        return (t_hit <= 1e-12) ? 0.0 : eta * t_hit;
    }

} // namespace step_filter::ccd

double CCDFilter::compute_safe_step(int who_global, const Vec2& dx,
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

        omega = std::min(omega, step_filter::ccd::safe_step(xi, dxi, xj, dxj, xk, dxk, eta));
        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

#include "ccd.h"

#include <algorithm>

CCDResult node_triangle_only_node_moves(const Vec3& x,  const Vec3& dx, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps) {
    CCDResult result;

    const Vec3 n = (x2 - x1).cross(x3 - x1);
    const double d = n.dot(x - x1);
    const double c = n.dot(dx);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            result.coplanar_entire_step = true;
        } else {
            result.parallel_or_no_crossing = true;
        }
        return result;
    }

    const double t = -d / c;
    if (!in_unit_interval(t, eps)) {
        result.parallel_or_no_crossing = true;
        return result;
    }

    result.has_candidate_time = true;
    result.t = clamp_scalar(t, 0.0, 1.0);

    const Vec3 p = point_at_linear_step(x, dx, result.t);
    result.collision = point_in_triangle_on_plane(p, x1, x2, x3, eps);
    return result;
}

CCDResult segment_segment_only_x1_moves(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps) {
    CCDResult result;

    // Coplanarity condition f(t) = dot((x2-x1-t*dx1) x (x4-x3), (x3-x1-t*dx1)) = 0
    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c0 = x3 - x1;

    const double d = (a.cross(b)).dot(c0);
    const double c = -(a.cross(b)).dot(dx1) - (dx1.cross(b)).dot(c0);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            result.coplanar_entire_step = true;
        } else {
            result.parallel_or_no_crossing = true;
        }
        return result;
    }

    const double t = -d / c;
    if (!in_unit_interval(t, eps)) {
        result.parallel_or_no_crossing = true;
        return result;
    }

    result.has_candidate_time = true;
    result.t = clamp_scalar(t, 0.0, 1.0);

    const Vec3 x1_star = point_at_linear_step(x1, dx1, result.t);
    double s = 0.0;
    double u = 0.0;
    if (!segment_segment_parameters_if_not_parallel(x1_star, x2, x3, x4, s, u, eps)) {
        return result;
    }

    result.collision = in_unit_interval(s, eps) && in_unit_interval(u, eps);
    return result;
}

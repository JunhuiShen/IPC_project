#include "node_segment_distance.h"
#include <cmath>

double node_segment_distance(const Vec2& xi, const Vec2& xj, const Vec2& xjp1,
                             double& t, Vec2& p, Vec2& r) {
    Vec2 seg = {xjp1.x - xj.x, xjp1.y - xj.y};
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    if (seg_len2 < 1e-14) {
        t = 0.0;
        p = xj;
        r = {xi.x - p.x, xi.y - p.y};
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    Vec2 q  = {xi.x - xj.x, xi.y - xj.y};
    double dot = q.x * seg.x + q.y * seg.y;
    t = dot / seg_len2;
    t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

    p = {xj.x + t * seg.x, xj.y + t * seg.y};
    r = {xi.x - p.x, xi.y - p.y};
    return std::sqrt(r.x * r.x + r.y * r.y);
}

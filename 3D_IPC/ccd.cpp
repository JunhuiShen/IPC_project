// Two CCD backends behind the public dispatchers in ccd.h:
//   - Linear   : closed-form, exact when one of the four vertices moves
//                over the step (Gauss-Seidel safe-step query).
//   - TICCD    : Tight-Inclusion CCD library [Wang et al. 2021] for the
//                general case where multiple vertices move.
// `node_triangle_general_ccd` / `segment_segment_general_ccd` are TICCD-only.
// `*_only_one_node_moves` route to one of the two via the `use_ticcd` arg.

#include "ccd.h"

#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

// -----------------------------------------------------------------------------
// Internal helpers (linear backend + TICCD config + result translation)
// -----------------------------------------------------------------------------
namespace {

bool is_effectively_zero(double value, double magnitude_bound, double relative_eps) {
    return magnitude_bound == 0.0
        ? value == 0.0
        : std::abs(value) <= relative_eps * magnitude_bound;
}

bool in_unit_interval(double t, double eps) {
    return t >= -eps && t <= 1.0 + eps;
}

Vec2 project_drop_axis(const Vec3& v, int drop_axis) {
    if (drop_axis == 0) return Vec2(v.y(), v.z());
    if (drop_axis == 1) return Vec2(v.x(), v.z());
    return Vec2(v.x(), v.y());
}

int dominant_drop_axis(const Vec3& n0, const Vec3& n1) {
    Vec3 n(std::fabs(n0.x()) + std::fabs(n1.x()),
           std::fabs(n0.y()) + std::fabs(n1.y()),
           std::fabs(n0.z()) + std::fabs(n1.z()));
    if (n.x() >= n.y() && n.x() >= n.z()) return 0;
    if (n.y() >= n.x() && n.y() >= n.z()) return 1;
    return 2;
}

// -----------------------------------------------------------------------------
// One-moving-vertex linear CCD
// -----------------------------------------------------------------------------

CCDResult node_triangle_linear_ccd(const Vec3& x,  const Vec3& dx,
                                   const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   double eps) {
    CCDResult result;
    const double geometry_relative_eps = std::max(eps, 1.0e-8);

    const auto point_in_triangle_at = [&](double t) {
        const Vec3 point = x + dx * t;
        const Vec3 a = x1 + dx1 * t;
        const Vec3 edge1 = x2 + dx2 * t - a;
        const Vec3 edge2 = x3 + dx3 * t - a;
        const Vec3 offset = point - a;
        const Vec3 normal = edge1.cross(edge2);
        const double normal2 = normal.squaredNorm();
        const double e11 = edge1.dot(edge1);
        const double e12 = edge1.dot(edge2);
        const double e22 = edge2.dot(edge2);

        // Degenerate triangles are outside the supported input domain.
        const double area_scale2 = e11 * e22;
        if (area_scale2 == 0.0
            || normal2 <= geometry_relative_eps * geometry_relative_eps * area_scale2) {
            return false;
        }

        const double local_length = std::max({std::sqrt(e11), std::sqrt(e22), offset.norm()});
        if (std::fabs(normal.dot(offset))
            > geometry_relative_eps * std::sqrt(normal2) * local_length) {
            return false;
        }

        const double b1 = edge1.dot(offset);
        const double b2 = edge2.dot(offset);
        const double lambda2 = (e22 * b1 - e12 * b2) / normal2;
        const double lambda3 = (e11 * b2 - e12 * b1) / normal2;
        const double lambda1 = 1.0 - lambda2 - lambda3;
        return lambda1 >= -1.0e-8 && lambda2 >= -1.0e-8 && lambda3 >= -1.0e-8;
    };

    if (point_in_triangle_at(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }

    // Ordinary case: f(t) = d + c t because at most one vertex moves.
    const Vec3 p = x2 - x1, dp = dx2 - dx1;
    const Vec3 q = x3 - x1, dq = dx3 - dx1;
    const Vec3 r = x  - x1, dr = dx  - dx1;
    const Vec3 pxq = p.cross(q);
    const double d = pxq.dot(r);
    const double c = dp.cross(q).dot(r) + p.cross(dq).dot(r) + pxq.dot(dr);
    const double d_scale = p.norm() * q.norm() * r.norm();
    const double c_scale =
        dp.norm() * q.norm() * r.norm()
        + p.norm() * dq.norm() * r.norm()
        + p.norm() * q.norm() * dr.norm();
    const bool coplanar_for_entire_step =
        is_effectively_zero(c, c_scale, eps) && is_effectively_zero(d, d_scale, eps);

    if (c != 0.0) {
        const double t = -d / c;
        if (in_unit_interval(t, eps)) {
            const double candidate_t = std::clamp(t, 0.0, 1.0);
            if (point_in_triangle_at(candidate_t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }

    if (!coplanar_for_entire_step) return result;

    // Special case: the node and triangle remain coplanar. Project to 2D;
    // contact can begin only when the node crosses a triangle edge.
    const double projected_relative_eps = std::max(eps, 1.0e-10);
    const Vec3 normal0 = (x2 - x1).cross(x3 - x1);
    const Vec3 normal1 = (x2 + dx2 - x1 - dx1).cross(x3 + dx3 - x1 - dx1);
    const int drop_axis = dominant_drop_axis(normal0, normal1);

    const Vec2 point0 = project_drop_axis(x, drop_axis);
    const Vec2 dpoint = project_drop_axis(dx, drop_axis);
    const Vec2 a0 = project_drop_axis(x1, drop_axis);
    const Vec2 da = project_drop_axis(dx1, drop_axis);
    const Vec2 b0 = project_drop_axis(x2, drop_axis);
    const Vec2 db = project_drop_axis(dx2, drop_axis);
    const Vec2 c0 = project_drop_axis(x3, drop_axis);
    const Vec2 dc = project_drop_axis(dx3, drop_axis);

    const auto point_in_triangle_2d = [&](const Vec2& point, const Vec2& a,
                                          const Vec2& b, const Vec2& c) {
        const auto orientation_sign = [&](const Vec2& u, const Vec2& v,
                                           const Vec2& query) {
            const Vec2 edge = v - u;
            const Vec2 offset = query - u;
            const double orientation = cross_product_in_2d(edge, offset);
            const double tolerance = projected_relative_eps * edge.norm() * offset.norm();
            return (orientation > tolerance) - (orientation < -tolerance);
        };
        const int s0 = orientation_sign(a, b, point);
        const int s1 = orientation_sign(b, c, point);
        const int s2 = orientation_sign(c, a, point);
        return (s0 >= 0 && s1 >= 0 && s2 >= 0)
            || (s0 <= 0 && s1 <= 0 && s2 <= 0);
    };

    std::array<double, 3> roots{};
    std::size_t root_count = 0;
    const auto add_orientation_root = [&](const Vec2& u0, const Vec2& du,
                                           const Vec2& v0, const Vec2& dv,
                                           const Vec2& query0, const Vec2& dquery) {
        const Vec2 edge = v0 - u0;
        const Vec2 dedge = dv - du;
        const Vec2 offset = query0 - u0;
        const Vec2 doffset = dquery - du;
        const double intercept = cross_product_in_2d(edge, offset);
        const double slope = cross_product_in_2d(dedge, offset)
            + cross_product_in_2d(edge, doffset);
        const double scale = std::max(std::fabs(slope), std::fabs(intercept));
        if (std::fabs(slope) <= eps * scale) return;

        double t = -intercept / slope;
        if (!in_unit_interval(t, eps)) return;
        t = std::clamp(t, 0.0, 1.0);
        for (std::size_t i = 0; i < root_count; ++i) {
            if (std::fabs(roots[i] - t) <= 1.0e-9) return;
        }
        roots[root_count++] = t;
    };

    add_orientation_root(a0, da, b0, db, point0, dpoint);
    add_orientation_root(b0, db, c0, dc, point0, dpoint);
    add_orientation_root(c0, dc, a0, da, point0, dpoint);
    std::sort(roots.begin(), roots.begin() + root_count);

    double coplanar_t = 1.0;
    for (std::size_t i = 0; i < root_count; ++i) {
        const double t = roots[i];
        if (point_in_triangle_2d(
                point0 + dpoint * t, a0 + da * t, b0 + db * t, c0 + dc * t)) {
            coplanar_t = t;
            break;
        }
    }

    if (point_in_triangle_at(coplanar_t)) {
        if (!result.collision || coplanar_t < result.t) result.t = coplanar_t;
        result.collision = true;
    }
    return result;
}

CCDResult segment_segment_linear_ccd(const Vec3& x1, const Vec3& dx1,
                                     const Vec3& x2, const Vec3& x3,
                                     const Vec3& x4, double eps) {
    CCDResult result;
    const double geometry_relative_eps = std::max(eps, 1.0e-8);

    const auto segments_intersect_at = [&](double t) {
        const Vec3 a0 = x1 + dx1 * t;
        const Vec3 a1 = x2;
        const Vec3 b0 = x3;
        const Vec3 b1 = x4;
        const Vec3 u = a1 - a0;
        const Vec3 v = b1 - b0;
        const Vec3 w = b0 - a0;
        const Vec3 normal = u.cross(v);
        const double normal2 = normal.squaredNorm();
        const double u2 = u.squaredNorm();
        const double v2 = v.squaredNorm();
        if (u2 == 0.0 || v2 == 0.0) return false;

        const double local_length = std::max({std::sqrt(u2), std::sqrt(v2), w.norm()});
        const double distance_tolerance = geometry_relative_eps * local_length;
        if (normal2 <= geometry_relative_eps * geometry_relative_eps * u2 * v2) {
            if (w.cross(u).squaredNorm() > distance_tolerance * distance_tolerance * u2) {
                return false;
            }

            const double ax = std::fabs(u.x());
            const double ay = std::fabs(u.y());
            const double az = std::fabs(u.z());
            const int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
            const auto component = [&](const Vec3& point) {
                return axis == 0 ? point.x() : (axis == 1 ? point.y() : point.z());
            };

            double A = component(a0), B = component(a1);
            double C = component(b0), D = component(b1);
            if (A > B) std::swap(A, B);
            if (C > D) std::swap(C, D);
            return A <= D + distance_tolerance && C <= B + distance_tolerance;
        }

        if (std::fabs(w.dot(normal)) > distance_tolerance * std::sqrt(normal2)) return false;
        const double alpha = w.cross(v).dot(normal) / normal2;
        const double beta = w.cross(u).dot(normal) / normal2;
        const Vec3 point_a = a0 + u * alpha;
        const Vec3 point_b = b0 + v * beta;
        if ((point_a - point_b).squaredNorm() > distance_tolerance * distance_tolerance) return false;
        return alpha >= -1.0e-8 && alpha <= 1.0 + 1.0e-8
            && beta >= -1.0e-8 && beta <= 1.0 + 1.0e-8;
    };

    if (segments_intersect_at(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }

    // Ordinary case: f(t) = -N . (y + t dx1) = d + c t.
    const Vec3 h = x2 - x3;
    const Vec3 b = x4 - x3;
    const Vec3 y = x1 - x3;
    const Vec3 normal = h.cross(b);
    const double d = -normal.dot(y);
    const double c = -normal.dot(dx1);
    const double d_scale = normal.norm() * y.norm();
    const double c_scale = normal.norm() * dx1.norm();
    const bool coplanar_for_entire_step =
        is_effectively_zero(c, c_scale, eps) && is_effectively_zero(d, d_scale, eps);

    if (c != 0.0) {
        const double t = -d / c;
        if (in_unit_interval(t, eps)) {
            const double candidate_t = std::clamp(t, 0.0, 1.0);
            if (segments_intersect_at(candidate_t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }

    if (!coplanar_for_entire_step) return result;

    // Special cases: the segments remain coplanar. Handle continuously
    // parallel/collinear motion in 1D; otherwise use projected 2D edge events.
    const auto earliest_point_in_interval_time = [&](double point0, double dpoint,
                                                      double a0, double da,
                                                      double b0, double db) {
        const auto is_between = [&](double t) {
            const double ra = (point0 - a0) + t * (dpoint - da);
            const double rb = (point0 - b0) + t * (dpoint - db);
            const double local_length = std::max(std::fabs(ra), std::fabs(rb));
            return ra * rb <= eps * local_length * local_length;
        };
        if (is_between(0.0)) return 0.0;

        std::array<double, 2> roots{};
        std::size_t root_count = 0;
        const auto add_root = [&](double slope, double intercept) {
            const double scale = std::max(std::fabs(slope), std::fabs(intercept));
            if (std::fabs(slope) <= eps * scale) return;
            double t = -intercept / slope;
            if (!in_unit_interval(t, eps)) return;
            t = std::clamp(t, 0.0, 1.0);
            for (std::size_t i = 0; i < root_count; ++i) {
                if (std::fabs(roots[i] - t) <= 1.0e-9) return;
            }
            roots[root_count++] = t;
        };
        add_root(dpoint - da, point0 - a0);
        add_root(dpoint - db, point0 - b0);
        std::sort(roots.begin(), roots.begin() + root_count);
        for (std::size_t i = 0; i < root_count; ++i) {
            const double t = roots[i];
            if (is_between(t) || is_between(std::min(1.0, t + 1.0e-9))) return t;
        }
        return 1.0;
    };

    const auto earliest_collinear_time = [&]() {
        const Vec3 direction0 = x2 - x1;
        const Vec3 direction1 = x2 - x1 - dx1;
        const double ax = std::fabs(direction0.x()) + std::fabs(direction1.x());
        const double ay = std::fabs(direction0.y()) + std::fabs(direction1.y());
        const double az = std::fabs(direction0.z()) + std::fabs(direction1.z());
        const int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
        const auto component = [&](const Vec3& value) {
            return axis == 0 ? value.x() : (axis == 1 ? value.y() : value.z());
        };

        const double A0 = component(x1), dA = component(dx1);
        const double B0 = component(x2);
        const double C0 = component(x3);
        const double D0 = component(x4);
        double best = 1.0;
        best = std::min(best, earliest_point_in_interval_time(A0, dA, C0, 0.0, D0, 0.0));
        best = std::min(best, earliest_point_in_interval_time(B0, 0.0, C0, 0.0, D0, 0.0));
        best = std::min(best, earliest_point_in_interval_time(C0, 0.0, A0, dA, B0, 0.0));
        best = std::min(best, earliest_point_in_interval_time(D0, 0.0, A0, dA, B0, 0.0));
        return best;
    };

    const Vec3 direction0 = x2 - x1;
    const Vec3 direction1 = x2 - x1 - dx1;
    const Vec3 static_direction = x4 - x3;
    const Vec3 normal0 = direction0.cross(static_direction);
    const Vec3 normal1 = direction1.cross(static_direction);
    const auto relatively_parallel = [&](const Vec3& u, const Vec3& n) {
        const double scale2 = u.squaredNorm() * static_direction.squaredNorm();
        return scale2 == 0.0 || n.squaredNorm() <= eps * eps * scale2;
    };

    double coplanar_t = 1.0;
    if (relatively_parallel(direction0, normal0)
        && relatively_parallel(direction1, normal1)) {
        coplanar_t = earliest_collinear_time();
    } else {
        const double projected_relative_eps = std::max(eps, 1.0e-10);
        const int drop_axis = dominant_drop_axis(normal0, normal1);
        const Vec2 a0 = project_drop_axis(x1, drop_axis);
        const Vec2 da = project_drop_axis(dx1, drop_axis);
        const Vec2 b0 = project_drop_axis(x2, drop_axis);
        const Vec2 c0 = project_drop_axis(x3, drop_axis);
        const Vec2 d0 = project_drop_axis(x4, drop_axis);

        const auto orientation_sign = [&](const Vec2& u, const Vec2& v,
                                           const Vec2& query) {
            const Vec2 edge = v - u;
            const Vec2 offset = query - u;
            const double orientation = cross_product_in_2d(edge, offset);
            const double tolerance = projected_relative_eps * edge.norm() * offset.norm();
            return (orientation > tolerance) - (orientation < -tolerance);
        };
        const auto on_segment = [&](const Vec2& u, const Vec2& query, const Vec2& v) {
            const Vec2 qu = query - u, qv = query - v, vu = v - u;
            const double scale2 = std::max({qu.squaredNorm(), qv.squaredNorm(), vu.squaredNorm()});
            return qu.dot(qv) <= projected_relative_eps * scale2;
        };
        const auto segments_intersect_2d = [&](const Vec2& a, const Vec2& b,
                                                const Vec2& c, const Vec2& d) {
            const int s1 = orientation_sign(a, b, c);
            const int s2 = orientation_sign(a, b, d);
            const int s3 = orientation_sign(c, d, a);
            const int s4 = orientation_sign(c, d, b);
            if (s1 * s2 < 0 && s3 * s4 < 0) return true;
            if (s1 == 0 && on_segment(a, c, b)) return true;
            if (s2 == 0 && on_segment(a, d, b)) return true;
            if (s3 == 0 && on_segment(c, a, d)) return true;
            return s4 == 0 && on_segment(c, b, d);
        };

        std::array<double, 4> roots{};
        std::size_t root_count = 0;
        const auto add_orientation_root = [&](const Vec2& u0, const Vec2& du,
                                               const Vec2& v0, const Vec2& dv,
                                               const Vec2& query0, const Vec2& dquery) {
            const Vec2 edge = v0 - u0;
            const Vec2 dedge = dv - du;
            const Vec2 offset = query0 - u0;
            const Vec2 doffset = dquery - du;
            const double intercept = cross_product_in_2d(edge, offset);
            const double slope = cross_product_in_2d(dedge, offset)
                + cross_product_in_2d(edge, doffset);
            const double scale = std::max(std::fabs(slope), std::fabs(intercept));
            if (std::fabs(slope) <= eps * scale) return;
            double t = -intercept / slope;
            if (!in_unit_interval(t, eps)) return;
            t = std::clamp(t, 0.0, 1.0);
            for (std::size_t i = 0; i < root_count; ++i) {
                if (std::fabs(roots[i] - t) <= 1.0e-9) return;
            }
            roots[root_count++] = t;
        };

        const Vec2 zero = Vec2::Zero();
        add_orientation_root(a0, da, b0, zero, c0, zero);
        add_orientation_root(a0, da, b0, zero, d0, zero);
        add_orientation_root(c0, zero, d0, zero, a0, da);
        add_orientation_root(c0, zero, d0, zero, b0, zero);
        std::sort(roots.begin(), roots.begin() + root_count);
        bool found_projected_collision = false;
        for (std::size_t i = 0; i < root_count; ++i) {
            const double t = roots[i];
            if (segments_intersect_2d(a0 + da * t, b0, c0, d0)) {
                coplanar_t = t;
                found_projected_collision = true;
                break;
            }
        }
        if (!found_projected_collision) coplanar_t = earliest_collinear_time();
    }

    if (segments_intersect_at(coplanar_t)) {
        if (!result.collision || coplanar_t < result.t) result.t = coplanar_t;
        result.collision = true;
    }
    return result;
}

// TICCD configuration (see Wang et al. 2021, "A Large-Scale Benchmark and an
// Inclusion-Based Algorithm for Continuous Collision Detection", and the
// upstream library at github.com/Continuous-Collision-Detection/Tight-Inclusion).
//
// TICCD is a conservative inclusion-based root finder for the CCD polynomial:
// it returns `false` only when there is provably no collision over [0, t_max].
// A `true` return is exact unless the closest distance between the moving
// primitives is below `tolerance + ms + err`, in which case it can be a false
// positive (which is the safe direction for collision response).
//
//   ms (kTiccdMinSeparation, 1e-10)
//     Minimum separation distance. A collision is *guaranteed* to be reported
//     whenever the pair gets closer than `ms`. Setting `ms = 0` requires
//     exact contact; any positive value pads the inclusion box.
//
//   tolerance (kTiccdTolerance, 1e-6)
//     Target solver precision: the maximum edge length of the bracketing
//     inclusion box at termination. 1e-6 matches the upstream reference
//     example. Tighter values cost more iterations.
//
//   max_itr (kTiccdMaxIter, 1e6)
//     Hard cap on inclusion-tree iterations. 1e6 matches upstream; -1 disables
//     the cap entirely.
//
//   no_zero_toi (kTiccdNoZeroToi)
//     When `ms > 0`, set true to keep refining instead of letting `toi` snap
//     to 0. We use the upstream default for parity with the reference.
//
//   ccd_method (kTiccdMethod, BREADTH_FIRST_SEARCH)
//     Inclusion-tree traversal order. Upstream-recommended default.
//
//   err (computed per-call, Array3::Constant(-1.0))
//     Per-axis numerical filter compensating for floating-point error in the
//     inclusion function. (-1, -1, -1) tells TICCD to compute it from the
//     query alone; for tighter filters call `ticcd::get_numerical_error()`
//     once with the simulation's scene AABB and reuse the result.
//
//   t_max (1.0, hard-coded at the call site)
//     Upper bound on the time interval to check. We always sweep the full
//     [0, 1] step.

constexpr double kTiccdMinSeparation = 1.0e-10;
constexpr double kTiccdTolerance     = 1.0e-6;
constexpr long   kTiccdMaxIter       = 1000000;
constexpr bool   kTiccdNoZeroToi     = ticcd::DEFAULT_NO_ZERO_TOI;
constexpr ticcd::CCDRootFindingMethod kTiccdMethod = ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH;

double clamp_toi(double toi) {
    if (!std::isfinite(toi)) return 1.0;
    if (toi < 0.0) return 0.0;
    if (toi > 1.0) return 1.0;
    return toi;
}

CCDResult ccd_result_from_toi(double toi) {
    CCDResult r;
    if (toi < 1.0) { r.collision = true; r.t = toi; }
    return r;
}

}  // namespace

// -----------------------------------------------------------------------------
// TICCD-backed general (all-vertices-may-move) entry points (public).
// -----------------------------------------------------------------------------

double node_triangle_general_ccd(const Vec3& x,  const Vec3& dx,
                                 const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2,
                                 const Vec3& x3, const Vec3& dx3) {
    const ticcd::Array3 err = ticcd::Array3::Constant(-1.0);
    double toi = std::numeric_limits<double>::infinity();
    double output_tolerance = kTiccdTolerance;
    const bool collision = ticcd::vertexFaceCCD(
        x,      x1,       x2,       x3,
        x + dx, x1 + dx1, x2 + dx2, x3 + dx3,
        err, kTiccdMinSeparation, toi, kTiccdTolerance, /*t_max=*/1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);
    return collision ? clamp_toi(toi) : 1.0;
}

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   const Vec3& x4, const Vec3& dx4) {
    const ticcd::Array3 err = ticcd::Array3::Constant(-1.0);
    double toi = std::numeric_limits<double>::infinity();
    double output_tolerance = kTiccdTolerance;
    const bool collision = ticcd::edgeEdgeCCD(
        x1,       x2,       x3,       x4,
        x1 + dx1, x2 + dx2, x3 + dx3, x4 + dx4,
        err, kTiccdMinSeparation, toi, kTiccdTolerance, /*t_max=*/1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);
    return collision ? clamp_toi(toi) : 1.0;
}

// -----------------------------------------------------------------------------
// Public one-moving-node dispatchers. `use_ticcd` selects the backend:
//   true  (default) -> TICCD library (conservative, robust)
//   false           -> closed-form linear (faster; exact for one moving node)
// -----------------------------------------------------------------------------

CCDResult node_triangle_only_one_node_moves(const Vec3& x,  const Vec3& dx,
                                            const Vec3& x1, const Vec3& dx1,
                                            const Vec3& x2, const Vec3& dx2,
                                            const Vec3& x3, const Vec3& dx3,
                                            double eps, bool use_ticcd) {
    if (use_ticcd) {
        return ccd_result_from_toi(node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3));
    }
    return node_triangle_linear_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3, eps);
}

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1,
                                              const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                              double eps, bool use_ticcd) {
    if (use_ticcd) {
        const Vec3 zero = Vec3::Zero();
        return ccd_result_from_toi(segment_segment_general_ccd(x1, dx1, x2, zero, x3, zero, x4, zero));
    }
    return segment_segment_linear_ccd(x1, dx1, x2, x3, x4, eps);
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Rigid Body CCD  /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool point_segment_2d_rb_rotation(
    const Eigen::Vector2d& x, const Eigen::Vector2d& x_com, const double& theta_n,
    const double& theta_new, const Eigen::Vector2d& x0, const Eigen::Vector2d& x1,
    double& step) {
    step = 0.0;

    Eigen::Vector2d dx = x - x_com;
    double cos_n = std::cos(theta_n);
    double sin_n = std::sin(theta_n);

    Eigen::Vector2d r{cos_n * dx.x() + sin_n * dx.y(), -sin_n * dx.x() + cos_n * dx.y()};

    Eigen::Vector2d d = x1 - x0;
    double seg_len = d.norm();
    if (seg_len < 1e-14) {
        std::cerr << "Warning: degenerate segment in point_segment_2d_rotation(): seg_len = " << seg_len
                  << ", threshold = 1e-14\n";
        return false;
    }

    Eigen::Vector2d d_hat = d / seg_len;

    double A = r.x() * d_hat.y() - r.y() * d_hat.x();
    double B = -r.x() * d_hat.x() - r.y() * d_hat.y();
    double C = (x_com.x() - x0.x()) * d_hat.y() - (x_com.y() - x0.y()) * d_hat.x();

    double amplitude = std::sqrt(A * A + B * B);
    if (std::abs(C) > amplitude + 1e-14) return false;
    if (amplitude < 1e-14) {
        if (std::abs(C) > 1e-14) return false;
        const double t_star = (x - x0).dot(d) / (seg_len * seg_len);
        if (t_star < -1e-12 || t_star > 1.0 + 1e-12) return false;
        return true;
    }

    double phi = std::atan2(B, A);
    double arccos_val = std::acos(std::clamp(-C / amplitude, -1.0, 1.0)); // clamp to prevent numerical errors.

    double theta_candidates[2] = {phi + arccos_val, phi - arccos_val};

    double best_s = std::numeric_limits<double>::infinity();
    double dtheta = theta_new - theta_n;
    if (std::abs(dtheta) < 1e-14) {
        // std::cerr << "Warning: theta_new - theta_n < threshold\n";
        return false;
    }

    constexpr double two_pi = 2.0 * M_PI;

    auto consider_theta = [&](double theta_star) {
        double s = (theta_star - theta_n) / dtheta;
        if (s < -1e-12 || s > 1.0 + 1e-12) return;
        s = std::clamp(s, 0.0, 1.0);

        double theta_s = theta_star;
        Eigen::Vector2d x_s{
            std::cos(theta_s) * r.x() - std::sin(theta_s) * r.y(),
            std::sin(theta_s) * r.x() + std::cos(theta_s) * r.y()
        };
        x_s += x_com;

        double t_star = (x_s - x0).dot(d) / (seg_len * seg_len);
        if (t_star < -1e-12 || t_star > 1.0 + 1e-12) return;

        if (s < best_s) best_s = s;
    };

    for (double theta_base : theta_candidates) {
        double k = 0.0;
        if (dtheta > 0.0) {
            // Forward rotation: choose the first wrapped root at or after theta_n
            k = std::ceil((theta_n - theta_base - 1e-12) / two_pi);
        } else {
            // Backward rotation: choose the first wrapped root at or before theta_n
            k = std::floor((theta_n - theta_base + 1e-12) / two_pi);
        }
        consider_theta(theta_base + two_pi * k);
    }

    if (best_s == std::numeric_limits<double>::infinity()) return false;
    step = best_s;
    return true;
}

static void buildBasis(const Vec3& n_hat, Vec3& e1, Vec3& e2) {
    Vec3 ref = (std::abs(n_hat.x()) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    e1 = n_hat.cross(ref).normalized();
    e2 = n_hat.cross(e1);
}

static Vec2 project2D(const Vec3& p, const Vec3& x_com, const Vec3& e1, const Vec3& e2) {
    Vec3 dp = p - x_com;
    return Vec2(dp.dot(e1), dp.dot(e2));
}

bool segment_segment_rb_rotation_ccd(
    const Vec3& x0, const Vec3& x1, // moving segment world space posiitons
    const Vec3& x_com, // moving segment rigid body center of mass
    const Vec4& q_new, const Vec4& q_n, // moving segment rigid body orientations (quaternions)
    const Vec3& x2, const Vec3& x3, // stationary segment world space positions
    double& s)
{
    s = 0.0;
    constexpr double eps = 1e-10;

    // -------------------------
    // Step 1: Extract rotation axis from q_rel = q_new * q_n^(-1)
    // -------------------------
    Vec4 q_n_conj = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec4 q_rel    = Rigid_Body::ALGEBRA::QuaternionMultiply(q_new, q_n_conj);

    Vec3 v_rel(q_rel[1], q_rel[2], q_rel[3]);
    double v_rel_norm = v_rel.norm();
    if (v_rel_norm < eps) return false;

    Vec3 n_hat = v_rel / v_rel_norm;

    // -------------------------
    // Step 2: Decompose moving endpoints
    // -------------------------
    double h0 = (x0 - x_com).dot(n_hat);
    double h1 = (x1 - x_com).dot(n_hat);

    Vec3 x0_perp = (x0 - x_com) - h0 * n_hat;
    Vec3 x1_perp = (x1 - x_com) - h1 * n_hat;
    double r0    = x0_perp.norm();
    double r1    = x1_perp.norm();
    double r_max = std::max(r0, r1);

    if (r_max < eps) return false;

    // Signed dtheta from larger-radius endpoint
    const Vec3& ref_perp       = (r0 >= r1) ? x0_perp : x1_perp;
    Vec3        ref_perp_rotated = Rigid_Body::ALGEBRA::QuaternionRotate(q_rel, ref_perp);
    double dtheta = std::atan2(
        ref_perp.cross(ref_perp_rotated).dot(n_hat),
        ref_perp.dot(ref_perp_rotated));

    if (std::abs(dtheta) < eps) return false;

    Vec3 dx_perp = x1_perp - x0_perp;

    // -------------------------
    // Step 3: Decompose stationary segment
    // -------------------------
    Vec3   q2  = x2 - x_com;
    Vec3   q3  = x3 - x_com;
    Vec3   e   = q3 - q2;
    double h2  = q2.dot(n_hat);
    double en  = e.dot(n_hat);

    Vec3 q2_perp = q2 - h2 * n_hat;
    Vec3 e_perp  = e  - en * n_hat;

    // Build 2D basis
    Vec3 e1, e2;
    buildBasis(n_hat, e1, e2);

    // theta_n and theta_new in 2D frame
    Vec2   ref_2d       = Vec2(ref_perp.dot(e1), ref_perp.dot(e2));
    double theta_n_2d   = std::atan2(ref_2d.y(), ref_2d.x());
    double theta_new_2d = theta_n_2d + dtheta;

    bool   found  = false;
    double best_s = std::numeric_limits<double>::infinity();

    // -------------------------
    // Case A: General swept surface (h0 != h1)
    // -------------------------
    if (std::abs(h1 - h0) > eps) {

        double dh = h1 - h0;

        // For a stationary point at parameter u, the height is h2 + u*en
        // The corresponding t on the moving segment is t(u) = (h2 + u*en - h0) / (h1 - h0)
        // x_perp at t(u): x0_perp + t(u)*dx_perp = x0_perp' + u*beta*dx_perp
        // where x0_perp' = x0_perp + (h2-h0)/dh * dx_perp
        //       beta     = en / dh

        double beta      = en / dh;
        Vec3   x0_perp_p = x0_perp + ((h2 - h0) / dh) * dx_perp; // x_perp at t(0)

        // Quadratic: rho_stationary^2(u) - rho_moving^2(u) = 0
        // rho_stationary^2(u) = ||q2_perp + u*e_perp||^2
        // rho_moving^2(u)     = ||x0_perp' + u*beta*dx_perp||^2

        double A = e_perp.squaredNorm()   - beta * beta * dx_perp.squaredNorm();
        double B = 2.0 * (q2_perp.dot(e_perp) - beta * x0_perp_p.dot(dx_perp));
        double C = q2_perp.squaredNorm()  - x0_perp_p.squaredNorm();

        double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0) {
            double sqrt_disc = std::sqrt(std::max(0.0, disc));

            for (int sign : {-1, 1}) {
                double u;
                if (std::abs(A) < eps) {
                    if (std::abs(B) < eps) {
                        if (std::abs(C) < eps) {
                            // infinite intersections: segment lies on swept surface
                            // only process once (skip second iteration)
                            if (sign == -1) continue;
                            u = 0.0; // take earliest contact
                        } 
                        else {
                            continue; // no intersection
                        }
                    } 
                    else {
                        u = -C / B;
                    }
                } 
                else {
                    u = (-B + sign * sqrt_disc) / (2.0 * A);
                }

                if (u < -eps || u > 1.0 + eps) continue;
                u = std::clamp(u, 0.0, 1.0);

                // corresponding t on moving segment
                double t = (h2 + u * en - h0) / dh;
                if (t < -eps || t > 1.0 + eps) continue;
                t = std::clamp(t, 0.0, 1.0);

                // contact point on moving segment (perpendicular part)
                Vec3 moving_perp = x0_perp + t * dx_perp;
                if (moving_perp.norm() < eps) continue; // on axis, degenerate

                // contact point on stationary segment (perpendicular part)
                Vec3 p_star      = (1.0 - u) * x2 + u * x3;
                Vec3 p_star_perp = (p_star - x_com) - ((p_star - x_com).dot(n_hat)) * n_hat;

                // recover signed angle: from moving_perp to p_star_perp
                double theta_ref  = std::atan2(moving_perp.dot(e2), moving_perp.dot(e1));
                double theta_star = std::atan2(p_star_perp.dot(e2), p_star_perp.dot(e1));
                double dangle     = theta_star - theta_ref;

                // wrap dangle consistent with sign of dtheta
                while (dtheta > 0 && dangle < -eps) dangle += 2.0 * M_PI;
                while (dtheta < 0 && dangle >  eps) dangle -= 2.0 * M_PI;

                double s_cand = dangle / dtheta;
                if (s_cand < -eps || s_cand > 1.0 + eps) continue;
                s_cand = std::clamp(s_cand, 0.0, 1.0);

                if (s_cand < best_s) { best_s = s_cand; found = true; }
            }
        }

    } 
    else {
        // -------------------------
        // Case B: Flat annulus (h0 == h1)
        // -------------------------

        // Find true inner radius
        double r_inner, r_outer;
        r_outer = std::max(r0, r1);

        double dx_perp_sq = dx_perp.squaredNorm();
        if (dx_perp_sq < eps) {
            // both endpoints at same perpendicular position
            r_inner = r0;
        } 
        else {
            double t_star = -x0_perp.dot(dx_perp) / dx_perp_sq;
            if (t_star > 0.0 && t_star < 1.0) {
                r_inner = (x0_perp + t_star * dx_perp).norm();
            } 
            else {
                r_inner = std::min(r0, r1);
            }
        }

        if (std::abs(en) > eps) {
            // Sub-case B1: stationary segment pierces annulus plane
            double u_star = (h0 - h2) / en;
            if (u_star >= -eps && u_star <= 1.0 + eps) {
                u_star = std::clamp(u_star, 0.0, 1.0);

                Vec3   p_star      = (1.0 - u_star) * x2 + u_star * x3;
                Vec3   p_star_perp = (p_star - x_com) - ((p_star - x_com).dot(n_hat)) * n_hat;
                double rho_star    = p_star_perp.norm();

                if (rho_star >= r_inner - eps && rho_star <= r_outer + eps) {
                    Vec2 p_star_2d = project2D(p_star, x_com, e1, e2);
                    Vec2 x0_2d     = project2D(x0,     x_com, e1, e2);
                    Vec2 x1_2d     = project2D(x1,     x_com, e1, e2);
                    Vec2 x_com_2d  = Vec2::Zero();

                    double s_cand = 0.0;
                    bool hit = point_segment_2d_rb_rotation(
                        p_star_2d, x_com_2d, 0.0, -dtheta, x0_2d, x1_2d, s_cand);

                    if (hit) { best_s = s_cand; found = true; }
                }
            }

        } 
        else if (std::abs(h0 - h2) < eps) {
            // Sub-case B2: both segments in same plane
            Vec2 x0_2d    = project2D(x0, x_com, e1, e2);
            Vec2 x1_2d    = project2D(x1, x_com, e1, e2);
            Vec2 x2_2d    = project2D(x2, x_com, e1, e2);
            Vec2 x3_2d    = project2D(x3, x_com, e1, e2);
            Vec2 x_com_2d = Vec2::Zero();

            struct Check { Vec2 x; double tn, tnew; Vec2 seg0, seg1; };
            std::vector<Check> checks = {
                {x0_2d, theta_n_2d, theta_new_2d, x2_2d, x3_2d},
                {x1_2d, theta_n_2d, theta_new_2d, x2_2d, x3_2d},
                {x2_2d, 0.0,       -dtheta,       x0_2d, x1_2d},
                {x3_2d, 0.0,       -dtheta,       x0_2d, x1_2d},
            };

            for (auto& c : checks) {
                double s_cand = 0.0;
                bool hit = point_segment_2d_rb_rotation(
                    c.x, x_com_2d, c.tn, c.tnew, c.seg0, c.seg1, s_cand);
                if (hit && s_cand < best_s) { best_s = s_cand; found = true; }
            }
        }
        // else: parallel planes -> no collision
    }

    if (!found) return false;
    s = best_s;
    return true;
}
bool point_triangle_rb_rotation_ccd(
    const Vec3& x,          // world position of particle at s=0
    const Vec3& x_com,      // center of mass
    const Vec4& q_new,      // quaternion at s=1
    const Vec4& q_n,        // quaternion at s=0
    const Vec3& x2,         // triangle vertex 0
    const Vec3& x3,         // triangle vertex 1
    const Vec3& x4,         // triangle vertex 2
    double& s)
{
    s = 0.0;
    constexpr double eps = 1e-10;

    // -------------------------
    // Step 0: Extract rotation axis from q_rel = q_new * q_n^(-1)
    // -------------------------
    Vec4 q_n_conj = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec4 q_rel    = Rigid_Body::ALGEBRA::QuaternionMultiply(q_new, q_n_conj);

    Vec3 v_rel(q_rel[1], q_rel[2], q_rel[3]);
    double v_rel_norm = v_rel.norm();
    if (v_rel_norm < eps) return false; // no rotation

    Vec3 n_hat = v_rel / v_rel_norm;

    // -------------------------
    // Step 1: Decompose particle offset (world space at s=0)
    // -------------------------
    Vec3   dx = x - x_com;
    double h  = dx.dot(n_hat);

    Vec3 r_parallel = h * n_hat;
    Vec3 r_perp     = dx - r_parallel;

    if (r_perp.norm() < eps) return false; // particle on rotation axis

    // Signed rotation angle
    Vec3 r_perp_rotated = Rigid_Body::ALGEBRA::QuaternionRotate(q_rel, r_perp);
    double dtheta = std::atan2(
        r_perp.cross(r_perp_rotated).dot(n_hat),
        r_perp.dot(r_perp_rotated));

    if (std::abs(dtheta) < eps) return false; // no rotation

    Vec3 n_cross_r_perp = n_hat.cross(r_perp);

    // -------------------------
    // Step 3: Triangle plane condition
    // -------------------------
    Vec3 e1 = x3 - x2;
    Vec3 e2 = x4 - x2;

    Vec3   n_tri      = e1.cross(e2);
    double n_tri_norm = n_tri.norm();
    if (n_tri_norm < eps) {
        std::cerr << "Warning: degenerate triangle detected in point_triangle_rb_rotation_ccd() when computing surface normal\n";
        return false; // degenerate triangle
    }

    Vec3 n_tri_hat = n_tri / n_tri_norm;

    double A = r_perp.dot(n_tri_hat);
    double B = n_cross_r_perp.dot(n_tri_hat);
    double C = (x_com + r_parallel - x2).dot(n_tri_hat);

    // Precompute barycentric system (triangle is stationary)
    double a11 = e1.dot(e1);
    double a12 = e1.dot(e2);
    double a22 = e2.dot(e2);
    double det = a11 * a22 - a12 * a12;
    if (std::abs(det) < eps){
        std::cerr << "Warning: degenerate triangle detected in point_triangle_rb_rotation_ccd() because of 0 determinant\n";
        return false; // degenerate triangle
    }

    // barycentric inside-check for an arbitrary point 
    auto inside_triangle = [&](const Vec3& p) -> bool {
        Vec3   r_tri = p - x2;
        double b1    = r_tri.dot(e1);
        double b2    = r_tri.dot(e2);

        double alpha   = ( b1 * a22 - b2 * a12) / det;
        double beta    = (-b1 * a12 + b2 * a11) / det;
        double lambda1 = 1.0 - alpha - beta;

        return (alpha >= -eps && beta >= -eps && lambda1 >= -eps);
    };

    double amplitude = std::sqrt(A * A + B * B);

    // -------------------------
    // Degenerate case: circle parallel to triangle plane
    // -------------------------
    if (amplitude < eps) {
        if (std::abs(C) > eps) return false; // constant nonzero distance, never touches

        // In-plane case: particle rotates within the triangle plane.

        // If the particle starts inside the triangle, contact at s = 0
        if (inside_triangle(x)) {
            s = 0.0;
            return true;
        }

        // Otherwise: 2D CCD against the three triangle edges
        Vec3 be1, be2;
        buildBasis(n_hat, be1, be2);

        Vec2 x_2d     = project2D(x,  x_com, be1, be2);
        Vec2 x2_2d    = project2D(x2, x_com, be1, be2);
        Vec2 x3_2d    = project2D(x3, x_com, be1, be2);
        Vec2 x4_2d    = project2D(x4, x_com, be1, be2);
        Vec2 x_com_2d = Vec2::Zero();

        // theta_n is a free gauge choice; only dtheta matters
        double theta_n_2d   = std::atan2(x_2d.y(), x_2d.x());
        double theta_new_2d = theta_n_2d + dtheta;

        double best_s_planar = std::numeric_limits<double>::infinity();

        const std::array<std::pair<Vec2, Vec2>, 3> edges = {{
            {x2_2d, x3_2d},
            {x3_2d, x4_2d},
            {x4_2d, x2_2d},
        }};

        for (const auto& edge : edges) {
            double s_cand = 0.0;
            bool hit = point_segment_2d_rb_rotation(
                x_2d, x_com_2d, theta_n_2d, theta_new_2d,
                edge.first, edge.second, s_cand);
            if (hit && s_cand < best_s_planar) best_s_planar = s_cand;
        }

        if (best_s_planar == std::numeric_limits<double>::infinity()) return false;
        s = best_s_planar;
        return true;
    }

    if (std::abs(C) > amplitude + eps) return false; // circle never reaches plane so no collision

    // -------------------------
    // Step 4: Solve A*cos(theta) + B*sin(theta) + C = 0, theta(s) = s*dtheta
    // -------------------------
    double phi        = std::atan2(B, A);
    double arccos_val = std::acos(std::clamp(-C / amplitude, -1.0, 1.0));

    double theta_candidates[2] = {phi + arccos_val, phi - arccos_val};

    constexpr double two_pi = 2.0 * M_PI;
    double best_s = std::numeric_limits<double>::infinity();

    auto consider_theta = [&](double theta_star) {
        // theta^n = 0 by construction (r_perp is world-space at s=0)
        double s_cand = theta_star / dtheta;
        if (s_cand < -eps || s_cand > 1.0 + eps) return;
        s_cand = std::clamp(s_cand, 0.0, 1.0);

        // Step 5: contact position via Rodrigues
        double theta_s = s_cand * dtheta;
        Vec3 x_s = x_com + r_parallel
                         + r_perp         * std::cos(theta_s)
                         + n_cross_r_perp * std::sin(theta_s);

        // Step 6: barycentric inside-triangle check
        if (!inside_triangle(x_s)) return;

        if (s_cand < best_s) best_s = s_cand;
    };

    for (double theta_base : theta_candidates) {
        // wrap to find first root at/after 0 (dtheta > 0) or at/before 0 (dtheta < 0)
        double k;
        if (dtheta > 0.0) {
            k = std::ceil((0.0 - theta_base - eps) / two_pi);
        } else {
            k = std::floor((0.0 - theta_base + eps) / two_pi);
        }
        consider_theta(theta_base + two_pi * k);
    }

    if (best_s == std::numeric_limits<double>::infinity()) return false;
    s = best_s;
    return true;
}

// Two CCD backends behind the public dispatchers in ccd.h:
//   - Linear   : closed-form for one moving vertex (Gauss-Seidel safe-step
//                query), with independent exact arithmetic for ambiguous cases.
//   - TICCD    : Tight-Inclusion CCD library [Wang et al. 2021] for the
//                general case where multiple vertices move.
// `node_triangle_general_ccd` / `segment_segment_general_ccd` are TICCD-only.
// `*_only_one_node_moves` route to one of the two via the `use_ticcd` arg.

#include "ccd.h"
#include "quaternion_math.h"

#include <tight_inclusion/ccd.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------------
// Internal helpers (linear backend + TICCD config + result translation)
// -----------------------------------------------------------------------------
namespace {

namespace exact_linear {

using Rational = boost::multiprecision::cpp_rational;
using Point = std::array<Rational, 3>;
using Points = std::array<Point, 4>;

Point add(const Point& a, const Point& b) {
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

Point subtract(const Point& a, const Point& b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

Point multiply(const Point& a, const Rational& scale) {
    return {{a[0] * scale, a[1] * scale, a[2] * scale}};
}

Rational dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point cross(const Point& a, const Point& b) {
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

Rational point_segment_distance_squared(const Point& p, const Point& a,
                                         const Point& b) {
    const Point edge = subtract(b, a), offset = subtract(p, a);
    const Rational length_squared = dot(edge, edge);
    const Rational projection = dot(offset, edge);
    if (length_squared == 0 || projection <= 0) return dot(offset, offset);
    if (projection >= length_squared) {
        const Point endpoint_offset = subtract(p, b);
        return dot(endpoint_offset, endpoint_offset);
    }
    return dot(offset, offset) - projection * projection / length_squared;
}

Rational point_triangle_distance_squared(const Points& p) {
    const Point edge1 = subtract(p[2], p[1]), edge2 = subtract(p[3], p[1]);
    const Point offset = subtract(p[0], p[1]);
    const Point normal = cross(edge1, edge2);
    const Rational area_squared = dot(normal, normal);
    if (area_squared != 0) {
        const Rational e11 = dot(edge1, edge1), e12 = dot(edge1, edge2);
        const Rational e22 = dot(edge2, edge2);
        const Rational p1 = dot(offset, edge1), p2 = dot(offset, edge2);
        // Exact Gram numerators are safe even for arbitrarily thin triangles.
        const Rational u = e22 * p1 - e12 * p2;
        const Rational v = e11 * p2 - e12 * p1;
        if (u >= 0 && v >= 0 && u + v <= area_squared) {
            const Rational height = dot(offset, normal);
            return height * height / area_squared;
        }
    }
    // A zero-area triangle is the union of its edges, including point edges.
    return std::min({point_segment_distance_squared(p[0], p[1], p[2]),
                     point_segment_distance_squared(p[0], p[2], p[3]),
                     point_segment_distance_squared(p[0], p[3], p[1])});
}

Rational segment_segment_distance_squared(const Points& p) {
    Rational best = std::min({point_segment_distance_squared(p[0], p[2], p[3]),
                             point_segment_distance_squared(p[1], p[2], p[3]),
                             point_segment_distance_squared(p[2], p[0], p[1]),
                             point_segment_distance_squared(p[3], p[0], p[1])});
    if (best == 0) return best;
    const Point a = subtract(p[1], p[0]), b = subtract(p[3], p[2]);
    const Point offset = subtract(p[0], p[2]);
    const Rational aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
    const Rational denominator = aa * bb - ab * ab;
    if (denominator != 0) {
        const Rational ar = dot(a, offset), br = dot(b, offset);
        const Rational alpha = ab * br - bb * ar;
        const Rational beta = aa * br - ab * ar;
        if (alpha >= 0 && alpha <= denominator && beta >= 0 && beta <= denominator) {
            const Point delta = subtract(add(offset, multiply(a, alpha / denominator)),
                                         multiply(b, beta / denominator));
            const Rational interior_distance = dot(delta, delta);
            best = std::min(best, interior_distance);
        }
    }
    return best;
}

Points evaluate(const Points& positions, const Points& motion, const Rational& time) {
    Points result;
    for (int i = 0; i < 4; ++i) result[i] = add(positions[i], multiply(motion[i], time));
    return result;
}

Rational determinant(const Points& p) {
    return dot(cross(subtract(p[1], p[0]), subtract(p[2], p[0])),
               subtract(p[3], p[0]));
}

CCDResult contact_result(const Rational& time) {
    double rounded = time.convert_to<double>();
    // Conversion may round upward. Preserve a conservative representable TOI.
    if (Rational(rounded) > time) rounded = std::nextafter(rounded, 0.0);
    return {true, rounded};
}

CCDResult resolve_exact(const std::array<Vec3, 4>& positions,
                        const std::array<Vec3, 4>& motion, bool vertex_face) {
    Points x, dx;
    int moving_vertices = 0;
    for (int i = 0; i < 4; ++i) {
        assert(positions[i].allFinite() && motion[i].allFinite());
        moving_vertices += motion[i].cwiseAbs().maxCoeff() != 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            // The rational double constructor preserves every input bit. Use
            // original inputs so local-frame subtraction cannot hide error.
            x[i][axis] = Rational(positions[i][axis]);
            dx[i][axis] = Rational(motion[i][axis]);
        }
    }
    assert(moving_vertices <= 1 || (!vertex_face && dx[0] == dx[1]
        && dx[2] == Point{} && dx[3] == Point{}));

    // This is exact-contact CCD with a fixed world-space tolerance for finite
    // primitive validation at initial time and analytic candidate events.
    // It is not a sweep of offset surfaces: no time padding or subdivision is
    // performed, and membership never uses a condition-dependent tolerance.
    static const Rational separation_squared = Rational(1.0e-10) * Rational(1.0e-10);
    const auto intersects = [&](const Rational& time) {
        const Points p = evaluate(x, dx, time);
        return (vertex_face ? point_triangle_distance_squared(p)
                            : segment_segment_distance_squared(p)) <= separation_squared;
    };
    if (intersects(Rational(0))) return {true, 0.0};
    if (moving_vertices == 0) return {};

    // Under the supported motions the tetrahedron determinant is affine.
    // Exact evaluation at 0 and 1 recovers its coefficients without loss.
    const Rational intercept = determinant(x);
    const Rational slope = determinant(evaluate(x, dx, Rational(1))) - intercept;
    if (slope != 0) {
        const Rational time = -intercept / slope;
        return time >= 0 && time <= 1 && intersects(time) ? contact_result(time) : CCDResult{};
    }
    if (intercept != 0) return {};

    std::vector<Rational> events;
    events.reserve(31);
    events.emplace_back(1);
    const auto add_root = [&](const Rational& value, const Rational& velocity) {
        if (velocity == 0) return;
        const Rational time = -value / velocity;
        if (time > 0 && time <= 1) events.push_back(time);
    };

    // Coplanar membership changes at point/edge incidence, triangle collapse,
    // or a collinear endpoint crossing. All components from every triple
    // retain events when a chosen 2D projection becomes degenerate. Their
    // quadratic terms vanish for one moving vertex or a translating edge.
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const Point edge = subtract(x[j], x[i]);
            const Point velocity = subtract(dx[j], dx[i]);
            for (int axis = 0; axis < 3; ++axis) add_root(edge[axis], velocity[axis]);
            for (int k = j + 1; k < 4; ++k) {
                const Point other = subtract(x[k], x[i]);
                const Point other_velocity = subtract(dx[k], dx[i]);
                const Point value = cross(edge, other);
                const Point derivative = add(cross(velocity, other), cross(edge, other_velocity));
                for (int axis = 0; axis < 3; ++axis) add_root(value[axis], derivative[axis]);
            }
        }
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    for (const Rational& time : events) {
        if (intersects(time)) return contact_result(time);
    }
    return {};
}

} // namespace exact_linear



// FMA recovers the rounding error of the second product. This matters for
// almost parallel edges; a Gram determinant squares their condition number.
double difference_of_products(double a, double b, double c, double d) {
    const double cd = c * d;
    return std::fma(a, b, -cd) + std::fma(-c, d, cd);
}

double orient2(const Vec2& a, const Vec2& b) {
    return difference_of_products(a.x(), b.y(), a.y(), b.x());
}

Vec3 stable_cross(const Vec3& a, const Vec3& b) {
    return Vec3(difference_of_products(a.y(), b.z(), a.z(), b.y()),
                difference_of_products(a.z(), b.x(), a.x(), b.z()),
                difference_of_products(a.x(), b.y(), a.y(), b.x()));
}

constexpr double kRoundoff = 64.0 * std::numeric_limits<double>::epsilon();
constexpr double kLinearTimePrecision = 1.0e-8;
constexpr double kLinearValidationDistance = 1.0e-10;

struct LinearCCDGuard {
    bool uncertain = false;
    double separation = 0.0;
    double error = 0.0;
    double time_error = 0.0;

    double orientation(const Vec2& a, const Vec2& b) {
        const double value = orient2(a, b);
        error = kRoundoff * (std::abs(a.x() * b.y()) + std::abs(a.y() * b.x()));
        if (error > 0.0 && std::abs(value) <= error) uncertain = true;
        return value;
    }

    double triple(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& normal) {
        const double value = normal.dot(c);
        // Bound cancellation by the sum of absolute products, rather than
        // declaring a small determinant zero using the lengths of the edges.
        const double magnitude =
            (std::abs(a.y() * b.z()) + std::abs(a.z() * b.y())) * std::abs(c.x()) +
            (std::abs(a.z() * b.x()) + std::abs(a.x() * b.z())) * std::abs(c.y()) +
            (std::abs(a.x() * b.y()) + std::abs(a.y() * b.x())) * std::abs(c.z());
        error = kRoundoff * magnitude;
        if (!std::isfinite(value) || (magnitude > 0.0 && std::abs(value) <= error)) {
            uncertain = true;
        }
        // Extremely thin features require exact finite-primitive validation.
        // A small plane determinant alone is not a reliable contact test.
        if (!uncertain) {
            // A max-component bound cheaply certifies ordinary triangles.
            // ||n||_2 >= ||n||_inf and ||edge||_2 <= sqrt(3)*||edge||_inf;
            // the factor four leaves ample room for roundoff. Retain the
            // stable norms when this certificate cannot rule out thinness.
            const double edge_max = std::max(a.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff());
            if (!(normal.cwiseAbs().maxCoeff() > 4.0 * separation * edge_max)) {
                const double area = normal.stableNorm();
                if (area > 0.0 && area <= separation * std::max(a.stableNorm(), b.stableNorm())) uncertain = true;
            }
        }
        return value;
    }

    double root(double d, double c, double d_error) {
        const double t = -d / c;
        // Resolving the signs is insufficient when a small slope amplifies
        // coefficient error into a substantially different impact time.
        const double bound = d_error + std::abs(t) * error;
        if (bound > kLinearTimePrecision * std::abs(c)) uncertain = true;
        if (std::isfinite(t)) time_error = std::max(time_error, bound / std::abs(c));
        return t;
    }
};

double orientation_root(const Vec2& u, const Vec2& du, const Vec2& v, const Vec2& dv,
                        const Vec2& point, const Vec2& dp, LinearCCDGuard& guard) {
    const double intercept = guard.orientation(v - u, point - u);
    const double intercept_error = guard.error;
    // Exactly one of these three points can move. Avoid an expanded slope
    // with two large cancelling terms when the first endpoint moves.
    double slope;
    if (du.cwiseAbs().maxCoeff() != 0.0) slope = guard.orientation(du, v - point);
    else if (dv.cwiseAbs().maxCoeff() != 0.0) slope = guard.orientation(dv, point - u);
    else slope = guard.orientation(v - u, dp);
    return slope == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                        : guard.root(intercept, slope, intercept_error);
}

// Work in a local frame and rescale by an exact power of two. This both keeps
// products in range and avoids rounding x + t*dx in large world coordinates.
struct LinearQuery {
    std::array<Vec3, 4> x;
    std::array<Vec3, 4> dx;
    int exponent = 0; // original lengths = local lengths * 2^exponent
    bool requires_exact = false;
    double validation_distance = 0.0;

    LinearQuery(const std::array<Vec3, 4>& positions, const std::array<Vec3, 4>& motion)
        : x(positions), dx(motion) {
        const Vec3 origin = x[3];
        for (auto& v : x) v -= origin;
        bool finite = true;
        for (const auto& v : x) finite = finite && v.allFinite();
        if (!finite) {
            requires_exact = true;
            // Opposite finite coordinates can overflow during subtraction.
            double world_max = 0.0;
            for (const auto& v : positions) world_max = std::max(world_max, v.cwiseAbs().maxCoeff());
            std::frexp(world_max, &exponent);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 3; ++j) {
                    x[i][j] = std::scalbn(positions[i][j], -exponent) - std::scalbn(origin[j], -exponent);
                    dx[i][j] = std::scalbn(motion[i][j], -exponent);
                }
            }
        }
        double magnitude = 0.0;
        for (int i = 0; i < 4; ++i) {
            magnitude = std::max({magnitude, x[i].cwiseAbs().maxCoeff(), dx[i].cwiseAbs().maxCoeff()});
        }
        int local_exponent = 0;
        // A normal, exactly represented power of two gives the same rounded
        // values as scalbn, while avoiding 24 library calls per ordinary query.
        // Keep scalbn itself when forming that normal scale is not possible.
        bool normal_scale = false;
        double scale = 1.0;
        if constexpr (std::numeric_limits<double>::is_iec559
                      && std::numeric_limits<double>::radix == 2
                      && std::numeric_limits<double>::digits == 53
                      && std::numeric_limits<double>::max_exponent == 1024
                      && sizeof(double) == sizeof(std::uint64_t)) {
            std::uint64_t magnitude_bits;
            std::memcpy(&magnitude_bits, &magnitude, sizeof(magnitude_bits));
            const int biased_exponent = static_cast<int>((magnitude_bits >> 52) & 0x7ff);
            if (biased_exponent >= 1 && biased_exponent <= 2044) {
                // For positive normal binary64 values, frexp's exponent is
                // stored exponent - 1022. Construct its reciprocal power of
                // two directly; memcpy preserves strict aliasing rules.
                local_exponent = biased_exponent - 1022;
                const std::uint64_t scale_bits = std::uint64_t(2045 - biased_exponent) << 52;
                std::memcpy(&scale, &scale_bits, sizeof(scale));
                normal_scale = true;
            }
        }
        if (!normal_scale) {
            std::frexp(magnitude, &local_exponent);
            normal_scale = local_exponent >= -1023 && local_exponent <= 1022;
            if (normal_scale) scale = std::scalbn(1.0, -local_exponent);
        }
        exponent += local_exponent;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                x[i][j] = normal_scale ? x[i][j] * scale : std::scalbn(x[i][j], -local_exponent);
                dx[i][j] = normal_scale ? dx[i][j] * scale : std::scalbn(dx[i][j], -local_exponent);
                // Uniform scaling cannot protect an arbitrarily anisotropic
                // query. Leave exponent headroom for determinant products,
                // cancellation residuals, and their error bounds.
                constexpr double minimum_safe_component = 0x1p-200;
                if ((x[i][j] != 0.0 && std::abs(x[i][j]) < minimum_safe_component)
                    || (dx[i][j] != 0.0 && std::abs(dx[i][j]) < minimum_safe_component)) {
                    requires_exact = true;
                }
                if (motion[i][j] != 0.0 && dx[i][j] == 0.0) requires_exact = true;
                for (int k = 0; k < i; ++k) {
                    if (positions[i][j] != positions[k][j] && x[i][j] == x[k][j]) requires_exact = true;
                }
            }
        }
        validation_distance = normal_scale && exponent == local_exponent
            ? kLinearValidationDistance * scale : std::scalbn(kLinearValidationDistance, -exponent);
    }
};

// Every linearly swept primitive lies in the convex hull of its vertices at
// t=0 and t=1. A strict projection gap between these two hulls therefore
// certifies separation throughout the step, even for coplanar queries whose
// analytic predicates were uncertain. Candidate axes need not be exact:
// any finite direction is valid once all hull vertices pass the same test.
bool swept_hulls_separated(const LinearQuery& q, bool vertex_face, double separation) {
    std::array<std::array<Vec3, 4>, 2> points{{q.x, {}}};
    for (int i = 0; i < 4; ++i) points[1][i] = q.x[i] + q.dx[i];
    const int split = vertex_face ? 1 : 2;
    const auto separates = [&](Vec3 axis) {
        const double magnitude = axis.cwiseAbs().maxCoeff();
        if (!(magnitude > 0.0) || !axis.allFinite()) return false;
        axis /= magnitude;
        double first_lo = std::numeric_limits<double>::infinity();
        double first_hi = -first_lo;
        double second_lo = first_lo, second_hi = first_hi;
        for (const auto& endpoint : points) {
            for (int i = 0; i < 4; ++i) {
                const double value = axis.dot(endpoint[i]);
                if (i < split) {
                    first_lo = std::min(first_lo, value);
                    first_hi = std::max(first_hi, value);
                } else {
                    second_lo = std::min(second_lo, value);
                    second_hi = std::max(second_hi, value);
                }
            }
        }
        // Normalized starts and displacements have components below one;
        // endpoints are below two. Cover local-frame subtraction, endpoint
        // addition and dot-product rounding, plus the validation tolerance.
        // This absolute bound also covers collapsed/underflowed components.
        // Such queries still need exact predicates if no separation is proven.
        const double margin = (4.0 * kRoundoff + 2.0 * separation) * axis.lpNorm<1>();
        return std::max(second_lo - first_hi, first_lo - second_hi) > margin;
    };
    for (int axis = 0; axis < 3; ++axis)
        if (separates(Vec3::Unit(axis))) return true;

    for (const auto& p : points) {
        if (vertex_face) {
            const Vec3 normal = stable_cross(p[2] - p[1], p[3] - p[1]);
            if (separates(normal)) return true;
            for (int edge = 0; edge < 3; ++edge)
                if (separates(stable_cross(normal, p[1 + (edge + 1) % 3] - p[1 + edge])))
                    return true;
        } else {
            const Vec3 u = p[1] - p[0], v = p[3] - p[2], w = p[2] - p[0];
            if (separates(stable_cross(u, v))
                || separates(stable_cross(stable_cross(u, w), u))
                || separates(stable_cross(stable_cross(v, w), v))) return true;
            // Endpoint-to-segment directions also separate collinear segments
            // and cases where the closest points are outside the edge lines.
            for (int endpoint = 0; endpoint < 4; ++endpoint) {
                const int other = endpoint < 2 ? 2 : 0;
                const Vec3 edge = p[other + 1] - p[other];
                const Vec3 offset = p[endpoint] - p[other];
                const double length_squared = edge.squaredNorm();
                const double t = length_squared > 0.0
                    ? std::clamp(offset.dot(edge) / length_squared, 0.0, 1.0) : 0.0;
                if (separates(offset - t * edge)) return true;
            }
        }
    }
    return false;
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
    Vec3 n(std::fabs(n0.x()) + std::fabs(n1.x()), std::fabs(n0.y()) + std::fabs(n1.y()),  std::fabs(n0.z()) + std::fabs(n1.z()));
    if (n.x() >= n.y() && n.x() >= n.z()) return 0;
    if (n.y() >= n.x() && n.y() >= n.z()) return 1;
    return 2;
}

// Use the dominant 2D projection for finite-segment coordinates. Squaring
// the normal or solving the Gram system loses thin, nearly parallel contacts.
bool segments_intersect(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1,
                        double eps, LinearCCDGuard& guard, bool coplanar = false,
                        double evaluation_error = 0.0, bool known_collinear = false) {
    const Vec3 u = a1 - a0, v = b1 - b0, w = b0 - a0;
    if (u.cwiseAbs().maxCoeff() == 0.0 || v.cwiseAbs().maxCoeff() == 0.0) {
        guard.uncertain = true;
        return false;
    }
    const Vec3 normal = stable_cross(u, v);
    const int axis = dominant_drop_axis(normal, Vec3::Zero());
    if (normal[axis] == 0.0) {
        const Vec3 offset = stable_cross(w, u);
        if (!known_collinear && offset.cwiseAbs().maxCoeff() != 0.0) {
            const double scale = w.stableNorm() * u.stableNorm();
            if (offset.cwiseAbs().maxCoeff() <= kRoundoff * scale
                + (evaluation_error + 2 * guard.separation) * u.stableNorm()) guard.uncertain = true;
            return false;
        }
        int along = 0;
        u.cwiseAbs().maxCoeff(&along);
        double A = 0.0, B = u[along], C = w[along], D = w[along] + v[along];
        if (A > B) std::swap(A, B);
        if (C > D) std::swap(C, D);
        if (A > D || C > B) {
            const double gap = std::max(A - D, C - B);
            const double tolerance = eps * std::max({std::abs(B), std::abs(C), std::abs(D)});
            if (gap <= tolerance + 2 * guard.separation) guard.uncertain = true;
            return false;
        }
        return true;
    }
    if (!coplanar) {
        const double plane = guard.triple(u, v, w, normal);
        if (plane != 0.0) {
            if (std::abs(plane) <= 2 * guard.separation * normal.stableNorm()) guard.uncertain = true;
            return false;
        }
    }
    const Vec2 pu = project_drop_axis(u, axis), pv = project_drop_axis(v, axis);
    const double denom = orient2(pu, pv);
    const double denom_magnitude = std::abs(pu.x() * pv.y()) + std::abs(pu.y() * pv.x());
    if (kRoundoff * denom_magnitude > kLinearTimePrecision * std::abs(denom)) guard.uncertain = true;
    const Vec2 pw = project_drop_axis(w, axis);
    const double alpha = orient2(pw, pv) / denom;
    const double beta = orient2(pw, pu) / denom;
    if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        guard.uncertain = true;
        return false;
    }
    const double outside = std::max({-alpha, alpha - 1.0, -beta, beta - 1.0});
    if (outside > 0.0) {
        // A resolved plane root may still round across a finite endpoint.
        // Include interpolation/subtraction error before declaring a miss.
        const double coordinate_error = evaluation_error + kRoundoff * std::max({a0.cwiseAbs().maxCoeff(),
            a1.cwiseAbs().maxCoeff(), b0.cwiseAbs().maxCoeff(), b1.cwiseAbs().maxCoeff()});
        const double bound = (8 * coordinate_error + 2 * guard.separation)
            * (u.lpNorm<1>() + v.lpNorm<1>() + w.lpNorm<1>() + coordinate_error);
        if (outside * std::abs(denom) <= std::max(bound, eps * std::abs(denom))) guard.uncertain = true;
        return false;
    }
    return true;
}

bool clip_affine_to_interval(
        double value0, double slope,
        double lower, double upper,
        double& t_min, double& t_max) {
    if (slope == 0.0) return value0 >= lower && value0 <= upper;

    double lower_time = (lower - value0) / slope;
    double upper_time = (upper - value0) / slope;
    if (lower_time > upper_time) std::swap(lower_time, upper_time);
    t_min = std::max(t_min, lower_time);
    t_max = std::min(t_max, upper_time);
    return t_min <= t_max;
}

template <std::size_t N, typename Predicate>
CCDResult first_coplanar_contact(std::array<double, N>& roots, std::size_t count,
                                const Predicate& intersects, LinearCCDGuard& guard) {
    std::sort(roots.begin(), roots.begin() + count);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = roots[i];
        const bool prior_uncertainty = guard.uncertain;
        const bool boundary_hit = intersects(t);
        const bool boundary_uncertainty = guard.uncertain;
        if (boundary_hit && !boundary_uncertainty) return {true, t};

        const double next = i + 1 < count ? roots[i + 1] : 1.0;
        if (next > t) {
            // Between successive events, membership is constant. A resolved
            // interior hit proves that the left event is conservative even
            // if evaluating its boundary rounded just outside the primitive.
            // Preserve coefficient/earlier uncertainty, not this one probe's.
            guard.uncertain = prior_uncertainty;
            if (intersects(t + 0.5 * (next - t))) return {true, t};
            guard.uncertain = guard.uncertain || boundary_uncertainty;
        }
        if (boundary_hit) return {true, t};
    }
    return intersects(1.0) ? CCDResult{true, 1.0} : CCDResult{};
}

// -----------------------------------------------------------------------------
// One-moving-vertex linear CCD
// -----------------------------------------------------------------------------

CCDResult node_triangle_linear_ccd(const Vec3& x,  const Vec3& dx, const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3, double eps, LinearCCDGuard& guard) {
    CCDResult result;
    const double geometry_relative_eps = std::max(eps, kRoundoff);

    const auto point_in_triangle_at = [&](double t, bool coplanar = false) {
        const Vec3 point = x + dx * t;
        const Vec3 a = x1 + dx1 * t;
        const Vec3 edge1 = x2 + dx2 * t - a;
        const Vec3 edge2 = x3 + dx3 * t - a;
        const Vec3 offset = point - a;
        const Vec3 normal = stable_cross(edge1, edge2);
        const int axis = dominant_drop_axis(normal, Vec3::Zero());
        const Vec2 e1 = project_drop_axis(edge1, axis), e2 = project_drop_axis(edge2, axis);
        const double area = orient2(e1, e2);
        const double area_magnitude = std::abs(e1.x() * e2.y()) + std::abs(e1.y() * e2.x());
        if (area == 0.0 || kRoundoff * area_magnitude > kLinearTimePrecision * std::abs(area)) {
            guard.uncertain = true;
            return false;
        }
        if (!coplanar) {
            const double plane = guard.triple(edge1, edge2, offset, normal);
            if (plane != 0.0) {
                if (std::abs(plane) <= 2 * guard.separation * normal.stableNorm()) guard.uncertain = true;
                return false;
            }
        }
        const Vec2 q = project_drop_axis(offset, axis);
        const double lambda2 = orient2(q, e2) / area;
        const double lambda3 = orient2(e1, q) / area;
        const double lambda1 = orient2(e1 - q, e2 - q) / area;
        const double minimum = std::min({lambda1, lambda2, lambda3});
        if (!std::isfinite(minimum)) { guard.uncertain = true; return false; }
        if (minimum < 0.0) {
            const double position_scale = std::max({x.cwiseAbs().maxCoeff(), x1.cwiseAbs().maxCoeff(),
                x2.cwiseAbs().maxCoeff(), x3.cwiseAbs().maxCoeff()});
            const double motion_scale = std::max({dx.cwiseAbs().maxCoeff(), dx1.cwiseAbs().maxCoeff(),
                dx2.cwiseAbs().maxCoeff(), dx3.cwiseAbs().maxCoeff()});
            const double coordinate_error = kRoundoff * (position_scale + t * motion_scale)
                + guard.time_error * motion_scale;
            const double bound = (8 * coordinate_error + 2 * guard.separation)
                * (edge1.lpNorm<1>() + edge2.lpNorm<1>() + offset.lpNorm<1>() + coordinate_error);
            if (-minimum * std::abs(area) <= std::max(bound, geometry_relative_eps * std::abs(area))) guard.uncertain = true;
            return false;
        }
        return true;
    };

    if (point_in_triangle_at(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    // The dispatcher resolves uncertainty independently. Further floating
    // event work cannot clear uncertainty established at the initial state.
    if (guard.uncertain) return {};

    // Put the moving vertex last in a tetrahedron determinant. The other
    // three vertices are static, so the slope is a SINGLE triple product.
    // Expanding about a moving triangle corner adds cancelling large terms.
    const std::array<Vec3, 4> positions{{x, x1, x2, x3}};
    const std::array<Vec3, 4> motion{{dx, dx1, dx2, dx3}};
    int moving = 0;
    for (int i = 0; i < 4; ++i) if (motion[i].cwiseAbs().maxCoeff() != 0.0) moving = i;
    std::array<int, 3> fixed{};
    int count = 0;
    for (int i = 0; i < 4; ++i) if (i != moving) fixed[count++] = i;
    const Vec3 p = positions[fixed[1]] - positions[fixed[0]];
    const Vec3 q = positions[fixed[2]] - positions[fixed[0]];
    const Vec3 plane_normal = stable_cross(p, q);
    const double d = guard.triple(p, q, positions[moving] - positions[fixed[0]], plane_normal);
    const double d_error = guard.error;
    const double c = guard.triple(p, q, motion[moving], plane_normal);
    const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
    if (guard.uncertain) return {};

    if (c != 0.0) {
        const double t = guard.root(d, c, d_error);
        if (guard.uncertain) return {};
        if (in_unit_interval(t, eps)) {
            const double candidate_t = std::clamp(t, 0.0, 1.0);
            if (point_in_triangle_at(candidate_t, candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }

    if (!coplanar_for_entire_step) return result;

    // Fully coplanar step: project along the most stable coordinate plane.
    // Point-triangle membership can change only when the point crosses one of the three projected triangle-edge lines.
    const Vec3 normal0 = stable_cross(x2 - x1, x3 - x1);
    const Vec3 normal1 = stable_cross(x2 - x1 + dx2 - dx1, x3 - x1 + dx3 - dx1);
    const int drop_axis = dominant_drop_axis(normal0, normal1);

    const Vec2 point0 = project_drop_axis(x, drop_axis);
    const Vec2 dpoint = project_drop_axis(dx, drop_axis);
    const Vec2 a0 = project_drop_axis(x1, drop_axis);
    const Vec2 da = project_drop_axis(dx1, drop_axis);
    const Vec2 b0 = project_drop_axis(x2, drop_axis);
    const Vec2 db = project_drop_axis(dx2, drop_axis);
    const Vec2 c0 = project_drop_axis(x3, drop_axis);
    const Vec2 dc = project_drop_axis(dx3, drop_axis);

    // Keep every in-range boundary event, including duplicate or arbitrarily close roots.
    // Distinct physical events need not be well separated in time.
    std::array<double, 3> roots{};
    std::size_t root_count = 0;
    const auto add_orientation_root = [&](const Vec2& u0, const Vec2& du,  const Vec2& v0, const Vec2& dv,  const Vec2& query0, const Vec2& dquery) {

        double t = orientation_root(u0, du, v0, dv, query0, dquery, guard);
        if (!in_unit_interval(t, eps)) return;
        t = std::clamp(t, 0.0, 1.0);
        roots[root_count++] = t;
    };

    add_orientation_root(a0, da, b0, db, point0, dpoint);
    add_orientation_root(b0, db, c0, dc, point0, dpoint);
    add_orientation_root(c0, dc, a0, da, point0, dpoint);
    if (guard.uncertain) return {};
    return first_coplanar_contact(roots, root_count, point_in_triangle_at, guard);
}

CCDResult segment_segment_linear_ccd(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps, LinearCCDGuard& guard) {
    CCDResult result;
    const double geometry_relative_eps = std::max(eps, kRoundoff);

    const auto segments_intersect_at = [&](double t, bool coplanar = false) {
        const Vec3 a0 = x1 + dx1 * t;
        const Vec3 a1 = x2;
        const Vec3 b0 = x3;
        const Vec3 b1 = x4;
        return segments_intersect(a0, a1, b0, b1, geometry_relative_eps, guard, coplanar,
            (kRoundoff * t + guard.time_error) * dx1.cwiseAbs().maxCoeff());
    };

    if (segments_intersect_at(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    // The dispatcher resolves uncertainty independently. Further floating
    // event work cannot clear uncertainty established at the initial state.
    if (guard.uncertain) return {};

    // Ordinary case: f(t) = -N . (y + t dx1) = d + c t.
    const Vec3 h = x2 - x3;
    const Vec3 b = x4 - x3;
    const Vec3 y = x1 - x3;
    const Vec3 plane_normal = stable_cross(h, b);
    const double d = -guard.triple(h, b, y, plane_normal);
    const double d_error = guard.error;
    const double c = -guard.triple(h, b, dx1, plane_normal);
    const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
    if (guard.uncertain) return {};

    if (c != 0.0) {
        const double t = guard.root(d, c, d_error);
        if (guard.uncertain) return {};
        if (in_unit_interval(t, eps)) {
            const double candidate_t = std::clamp(t, 0.0, 1.0);
            if (segments_intersect_at(candidate_t, candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }

    if (!coplanar_for_entire_step) return result;

    // Fully coplanar step: handle permanently parallel/collinear motion in 1D; otherwise sweep the projected 2D boundary events below.
    const auto earliest_point_in_interval_time = [&](double point0, double dpoint,  double a0, double da, double b0, double db) {
        const auto is_between = [&](double t) {
            const double ra = (point0 - a0) + t * (dpoint - da);
            const double rb = (point0 - b0) + t * (dpoint - db);
            const double local_length = std::max(std::fabs(ra), std::fabs(rb));
            return ra * rb <= eps * local_length * local_length;
        };
        if (is_between(0.0)) return 0.0;

        // In 1D, interval membership can change only when the point meets an endpoint.
        // Keep both roots even when they coincide or are very close.
        std::array<double, 2> roots{};
        std::size_t root_count = 0;
        const auto add_root = [&](double slope, double intercept) {
            if (slope == 0.0) return;
            double t = -intercept / slope;
            if (!in_unit_interval(t, eps)) return;
            roots[root_count++] = std::clamp(t, 0.0, 1.0);
        };
        add_root(dpoint - da, point0 - a0);
        add_root(dpoint - db, point0 - b0);
        std::sort(roots.begin(), roots.begin() + root_count);

        // Test every endpoint event and one representative midpoint from each following interval; membership is constant between endpoint events.
        for (std::size_t i = 0; i < root_count; ++i) {
            const double t = roots[i];
            if (is_between(t)) return t;

            const double next_t = (i + 1 < root_count) ? roots[i + 1] : 1.0;
            if (next_t > t && is_between(t + 0.5 * (next_t - t))) return t;
        }
        if (is_between(1.0)) return 1.0;
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
    const Vec3 normal0 = stable_cross(direction0, static_direction);
    const Vec3 normal1 = stable_cross(direction1, static_direction);
    // Only truly parallel directions reduce to a one-dimensional overlap.
    if (normal0.cwiseAbs().maxCoeff() == 0.0 && normal1.cwiseAbs().maxCoeff() == 0.0) {
        const double collinear_t = earliest_collinear_time();
        if (segments_intersect_at(collinear_t)) {
            result.collision = true;
            result.t = collinear_t;
        }
        return result;
    }

    // Remaining coplanar case: project along the most stable coordinate plane.
    // Segment intersection can change only at the four endpoint/supporting-line orientation events generated below.
    const int drop_axis = dominant_drop_axis(normal0, normal1);
    const Vec2 a0 = project_drop_axis(x1, drop_axis);
    const Vec2 da = project_drop_axis(dx1, drop_axis);
    const Vec2 b0 = project_drop_axis(x2, drop_axis);
    const Vec2 c0 = project_drop_axis(x3, drop_axis);
    const Vec2 d0 = project_drop_axis(x4, drop_axis);

    // Keep all events, including duplicate or arbitrarily close roots.
    std::array<double, 4> roots{};
    std::size_t root_count = 0;
    const auto add_orientation_root = [&](const Vec2& u0, const Vec2& du, const Vec2& v0, const Vec2& dv, const Vec2& query0, const Vec2& dquery) {
        double t = orientation_root(u0, du, v0, dv, query0, dquery, guard);
        if (!in_unit_interval(t, eps)) return;
        roots[root_count++] = std::clamp(t, 0.0, 1.0);
    };

    const Vec2 zero = Vec2::Zero();
    add_orientation_root(a0, da, b0, zero, c0, zero);
    add_orientation_root(a0, da, b0, zero, d0, zero);
    add_orientation_root(c0, zero, d0, zero, a0, da);
    add_orientation_root(c0, zero, d0, zero, b0, zero);
    if (guard.uncertain) return {};
    return first_coplanar_contact(roots, root_count, segments_intersect_at, guard);
}

CCDResult translating_segment_linear_ccd(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& x4, double eps, LinearCCDGuard& guard) {
    assert((dx1.array() == dx2.array()).all() && "endpoint displacements must match");
    (void)dx2;
    const Vec3& dx = dx1;
    CCDResult result;
    const double geometry_relative_eps = std::max(eps, kRoundoff);

    const auto segments_intersect_at = [&](double t, bool coplanar = false) {
        return segments_intersect(x1 + dx * t, x2 + dx * t, x3, x4, geometry_relative_eps, guard, coplanar,
            (kRoundoff * t + guard.time_error) * dx.cwiseAbs().maxCoeff());
    };

    if (segments_intersect_at(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    // The dispatcher resolves uncertainty independently. Further floating
    // event work cannot clear uncertainty established at the initial state.
    if (guard.uncertain) return {};

    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 r = x3 - x1;
    const double a2 = a.squaredNorm();
    const double b2 = b.squaredNorm();

    if (a2 == 0.0 || b2 == 0.0) { guard.uncertain = true; return result; }

    const Vec3 normal = stable_cross(a, b);
    const int drop_axis = dominant_drop_axis(normal, Vec3::Zero());
    const double denominator = orient2(project_drop_axis(a, drop_axis), project_drop_axis(b, drop_axis));
    const bool parallel = denominator == 0.0;

    if (!parallel) {
        const double d = guard.triple(a, b, r, normal);
        const double d_error = guard.error;
        const double c = -guard.triple(a, b, dx, normal);
        const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
        if (guard.uncertain) return {};

        // When c != 0, this gives the single candidate time t = -d / c
        if (c != 0.0) {
            const double t = guard.root(d, c, d_error);
            if (guard.uncertain) return {};
            if (in_unit_interval(t, eps)) {
                const double candidate_t = std::clamp(t, 0.0, 1.0);
                if (segments_intersect_at(candidate_t, candidate_t == t)) {
                    result.collision = true;
                    result.t = candidate_t;
                    return result;
                }
            }
        }

        // Unless f(t) is effectively zero throughout the step, the isolated root above is the only possible coplanarity event
        if (!coplanar_for_entire_step) return result;

        // Nonparallel and coplanar for the entire step: alpha(t) and beta(t) are affine.
        // Clip [0, 1] against 0 <= alpha,beta <= 1 to obtain the first finite-segment overlap.
        const Vec2 pa = project_drop_axis(a, drop_axis), pb = project_drop_axis(b, drop_axis);
        const Vec2 pr = project_drop_axis(r, drop_axis), pdx = project_drop_axis(dx, drop_axis);
        const double alpha0 = orient2(pr, pb) / denominator;
        const double alpha_slope = -orient2(pdx, pb) / denominator;
        const double beta0 = orient2(pr, pa) / denominator;
        const double beta_slope = -orient2(pdx, pa) / denominator;
        double t_min = 0.0;
        double t_max = 1.0;
        if (!clip_affine_to_interval(alpha0, alpha_slope, 0.0, 1.0, t_min, t_max) || !clip_affine_to_interval(beta0, beta_slope, 0.0, 1.0, t_min, t_max)) {
            return result;
        }

        if (in_unit_interval(t_min, eps)) {
            const double candidate_t = std::clamp(t_min, 0.0, 1.0);
            if (segments_intersect_at(candidate_t, true)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
        return result;
    }

    // Parallel case: a x b = 0, so f(t) is identically zero regardless of  the separation between the supporting lines.
    // Test collinearity first, then test overlap of the finite segments.
    const Vec3 direction = a.stableNorm() >= b.stableNorm() ? a : b;
    const Vec3 transverse_offset = stable_cross(r, direction);
    const Vec3 transverse_velocity = stable_cross(dx, direction);
    int transverse_axis = 0;
    const double transverse_speed = transverse_velocity.cwiseAbs().maxCoeff(&transverse_axis);
    const bool no_transverse_motion = transverse_speed == 0.0;

    if (!no_transverse_motion) {
        const double offset_size = transverse_offset.cwiseAbs().maxCoeff();
        bool known_collinear = offset_size == 0.0;
        if (!known_collinear) {
            // All transverse components must vanish at the SAME time. An
            // unrelated large tangential velocity is not a spatial tolerance
            // for a persistent gap in another coordinate.
            const Vec3 u = transverse_offset / offset_size;
            const Vec3 v = transverse_velocity / transverse_speed;
            const Vec3 mismatch = stable_cross(u, v);
            known_collinear = mismatch.cwiseAbs().maxCoeff() == 0.0;
            if (!known_collinear) {
                if (mismatch.cwiseAbs().maxCoeff() <= kRoundoff) guard.uncertain = true;
                return result;
            }
        }
        // Solve transverse_offset - t * transverse_velocity = 0.
        // Use the dominant component, avoiding squares that can underflow
        // even after scaling an extremely long, narrowly separated pair.
        // The 3D predicate still rejects inconsistent transverse components.
        const double t = transverse_offset[transverse_axis] / transverse_velocity[transverse_axis];
        if (in_unit_interval(t, eps)) {
            const double candidate_t = std::clamp(t, 0.0, 1.0);
            if (segments_intersect(x1 + dx * candidate_t, x2 + dx * candidate_t, x3, x4,
                    geometry_relative_eps, guard, false, 0.0, known_collinear && candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
        return result;
    }

    const bool collinear_for_entire_step = transverse_offset.cwiseAbs().maxCoeff() == 0.0;
    if (!collinear_for_entire_step) return result;

    // The supporting lines remain collinear.
    //  Project onto the dominant axis and clip the translated interval against the fixed interval.
    const double ax = std::fabs(direction.x());
    const double ay = std::fabs(direction.y());
    const double az = std::fabs(direction.z());
    const int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
    const auto component = [&](const Vec3& point) {
        return axis == 0 ? point.x() : (axis == 1 ? point.y() : point.z());
    };

    double moving_min = component(x1);
    double moving_max = component(x2);
    double fixed_min = component(x3);
    double fixed_max = component(x4);
    if (moving_min > moving_max) std::swap(moving_min, moving_max);
    if (fixed_min > fixed_max) std::swap(fixed_min, fixed_max);

    const double speed = component(dx);
    double t_min = 0.0;
    double t_max = 1.0;
    if (!clip_affine_to_interval(moving_min - fixed_max, speed, -std::numeric_limits<double>::infinity(), 0.0, t_min, t_max)
    || !clip_affine_to_interval( fixed_min - moving_max, -speed, -std::numeric_limits<double>::infinity(), 0.0, t_min, t_max)) {
        return result;
    }

    if (in_unit_interval(t_min, eps)) {
        const double candidate_t = std::clamp(t_min, 0.0, 1.0);
        if (segments_intersect_at(candidate_t)) {
            result.collision = true;
            result.t = candidate_t;
        }
    }
    return result;
}

namespace linear_ccd_detail {

CCDResult node_triangle(const Vec3& x, const Vec3& dx,
                        const Vec3& x1, const Vec3& dx1,
                        const Vec3& x2, const Vec3& dx2,
                        const Vec3& x3, const Vec3& dx3, double eps) {
    const std::array<Vec3, 4> positions{{x, x1, x2, x3}}, motion{{dx, dx1, dx2, dx3}};
    const LinearQuery q(positions, motion);
    if (q.requires_exact) {
        if (swept_hulls_separated(q, true, q.validation_distance)) return {};
        return exact_linear::resolve_exact(positions, motion, true);
    }
    LinearCCDGuard guard{false, q.validation_distance};
    const auto result = node_triangle_linear_ccd(q.x[0], q.dx[0], q.x[1], q.dx[1],
        q.x[2], q.dx[2], q.x[3], q.dx[3], eps, guard);
    if (guard.uncertain && swept_hulls_separated(q, true, guard.separation)) return {};
    return guard.uncertain ? exact_linear::resolve_exact(positions, motion, true) : result;
}

CCDResult segment_segment(const Vec3& x1, const Vec3& dx1, const Vec3& x2,
                          const Vec3& x3, const Vec3& x4, double eps) {
    const Vec3 zero = Vec3::Zero();
    const std::array<Vec3, 4> positions{{x1, x2, x3, x4}}, motion{{dx1, zero, zero, zero}};
    const LinearQuery q(positions, motion);
    if (q.requires_exact) {
        if (swept_hulls_separated(q, false, q.validation_distance)) return {};
        return exact_linear::resolve_exact(positions, motion, false);
    }
    LinearCCDGuard guard{false, q.validation_distance};
    const auto result = segment_segment_linear_ccd(q.x[0], q.dx[0], q.x[1], q.x[2], q.x[3], eps, guard);
    if (guard.uncertain && swept_hulls_separated(q, false, guard.separation)) return {};
    return guard.uncertain ? exact_linear::resolve_exact(positions, motion, false) : result;
}

CCDResult translating_segment(const Vec3& x1, const Vec3& dx1, const Vec3& x2,
                              const Vec3& dx2, const Vec3& x3, const Vec3& x4, double eps) {
    assert((dx1.array() == dx2.array()).all() && "endpoint displacements must match");
    const Vec3 zero = Vec3::Zero();
    const std::array<Vec3, 4> positions{{x1, x2, x3, x4}}, motion{{dx1, dx2, zero, zero}};
    const LinearQuery q(positions, motion);
    if (q.requires_exact) {
        if (swept_hulls_separated(q, false, q.validation_distance)) return {};
        return exact_linear::resolve_exact(positions, motion, false);
    }
    LinearCCDGuard guard{false, q.validation_distance};
    const auto result = translating_segment_linear_ccd(q.x[0], q.dx[0], q.x[1], q.dx[1], q.x[2], q.x[3], eps, guard);
    if (guard.uncertain && swept_hulls_separated(q, false, guard.separation)) return {};
    return guard.uncertain ? exact_linear::resolve_exact(positions, motion, false) : result;
}

} // namespace linear_ccd_detail

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

CCDResult inclusion_ccd(const std::array<Vec3, 4>& x, const std::array<Vec3, 4>& dx,
                        bool vertex_face, double separation = kTiccdMinSeparation,
                        double tolerance = kTiccdTolerance) {
    const ticcd::Array3 err = ticcd::Array3::Constant(-1.0);
    double toi = std::numeric_limits<double>::infinity();
    double output_tolerance = tolerance;
    const auto ccd = vertex_face ? ticcd::vertexFaceCCD : ticcd::edgeEdgeCCD;
    const bool collision = ccd(x[0], x[1], x[2], x[3],
        x[0] + dx[0], x[1] + dx[1], x[2] + dx[2], x[3] + dx[3],
        err, separation, toi, tolerance, 1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);
    // Keep the collision flag: t == 1 is a valid endpoint contact.
    return collision ? CCDResult{true, clamp_toi(toi)} : CCDResult{};
}

}  // namespace

// -----------------------------------------------------------------------------
// TICCD-backed general (all-vertices-may-move) entry points (public).
// -----------------------------------------------------------------------------

double node_triangle_general_ccd(const Vec3& x,  const Vec3& dx,
                                 const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2,
                                 const Vec3& x3, const Vec3& dx3) {
    const auto r = inclusion_ccd({{x, x1, x2, x3}}, {{dx, dx1, dx2, dx3}}, true);
    return r.collision ? r.t : 1.0;
}

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   const Vec3& x4, const Vec3& dx4) {
    const auto r = inclusion_ccd({{x1, x2, x3, x4}}, {{dx1, dx2, dx3, dx4}}, false);
    return r.collision ? r.t : 1.0;
}

// -----------------------------------------------------------------------------
// Public one-moving-node dispatchers. `use_ticcd` selects the backend:
//   true  (default) -> TICCD library (conservative, robust)
//   false           -> independent closed-form linear backend
// -----------------------------------------------------------------------------

CCDResult node_triangle_only_one_node_moves(const Vec3& x,  const Vec3& dx,
                                            const Vec3& x1, const Vec3& dx1,
                                            const Vec3& x2, const Vec3& dx2,
                                            const Vec3& x3, const Vec3& dx3,
                                            double eps, bool use_ticcd) {
    if (use_ticcd) {
        return inclusion_ccd({{x, x1, x2, x3}}, {{dx, dx1, dx2, dx3}}, true);
    }
    return linear_ccd_detail::node_triangle(x, dx, x1, dx1, x2, dx2, x3, dx3, eps);
}

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1,
                                              const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                              double eps, bool use_ticcd) {
    if (use_ticcd) {
        const Vec3 zero = Vec3::Zero();
        return inclusion_ccd({{x1, x2, x3, x4}}, {{dx1, zero, zero, zero}}, false);
    }
    return linear_ccd_detail::segment_segment(x1, dx1, x2, x3, x4, eps);
}

CCDResult segment_segment_same_displacement_linear_ccd(
        const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& x4, double eps) {
    return linear_ccd_detail::translating_segment(x1, dx1, x2, dx2, x3, x4, eps);
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
    if (seg_len < 1e-14)
        return false;

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
    if (std::abs(dtheta) < 1e-14)
        return false;

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
    Vec4 q_n_conj = quaternion_conjugate(q_n);
    Vec4 q_rel = quaternion_multiply(q_new, q_n_conj);

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

    double dtheta = 2.0 * std::atan2(v_rel_norm, q_rel[0]);

    if (std::abs(dtheta) < eps) return false;

    // Larger-radius endpoint: well-conditioned reference for the 2D angle gauge
    const Vec3& ref_perp = (r0 >= r1) ? x0_perp : x1_perp;

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
                            if (sign == 1) continue;
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
    Vec4 q_n_conj = quaternion_conjugate(q_n);
    Vec4 q_rel = quaternion_multiply(q_new, q_n_conj);

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

    double dtheta = 2.0 * std::atan2(v_rel_norm, q_rel[0]);

    if (std::abs(dtheta) < eps) return false; // no rotation

    Vec3 n_cross_r_perp = n_hat.cross(r_perp);

    // -------------------------
    // Step 3: Triangle plane condition
    // -------------------------
    Vec3 e1 = x3 - x2;
    Vec3 e2 = x4 - x2;

    double a11 = e1.dot(e1);
    double a12 = e1.dot(e2);
    double a22 = e2.dot(e2);

    Vec3   n_tri      = e1.cross(e2);
    double n_tri_norm = n_tri.norm();
    if (n_tri_norm * n_tri_norm <= eps * a11 * a22)
        return false; // degenerate triangle

    Vec3 n_tri_hat = n_tri / n_tri_norm;

    double A = r_perp.dot(n_tri_hat);
    double B = n_cross_r_perp.dot(n_tri_hat);
    double C = (x_com + r_parallel - x2).dot(n_tri_hat);

    // Precompute barycentric system (triangle is stationary)
    double det = a11 * a22 - a12 * a12;
    if (std::abs(det) <= eps * a11 * a22)
        return false; // degenerate triangle

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

#include "ccd.h"
#include "ipc_args.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "node_triangle_distance.h"
#include "simulation.h"

#include <omp.h>
#include "safe_step.h"
#include "segment_segment_distance.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <gtest/gtest.h>

// Shared TOI tolerance for the basic CCD assertions.
constexpr double kTol = 1.0e-6;

namespace {
const Vec3 ZERO_DX(0.0, 0.0, 0.0);

void prepare_single_vertex_contact(BroadPhase& phase, const std::vector<Vec3>& x,
                                  bool segment, int role) {
    auto& cache = phase.mutable_cache();
    cache.vertex_nt.resize(x.size());
    cache.vertex_ss.resize(x.size());
    for (const Vec3& position : x)
        cache.node_boxes.emplace_back(position - Vec3::Constant(4.0),
                                      position + Vec3::Constant(4.0));
    if (segment) {
        cache.ss_pairs.push_back(SegmentSegmentPair{{0, 1, 2, 3}});
        cache.vertex_ss[role].push_back({0, role});
    } else {
        cache.nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
        cache.vertex_nt[role].push_back({0, role});
    }
}

std::vector<Vec3> planar_contact(bool segment) {
    if (segment)
        return {Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0), Vec3(0, 1, 0)};
    return {Vec3(.25, .25, 0), Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()};
}

double contact_distance(const std::vector<Vec3>& x, bool segment) {
    return segment ? segment_segment_distance(x[0], x[1], x[2], x[3]).distance
                   : node_triangle_distance(x[0], x[1], x[2], x[3]).distance;
}

ExactContactStepResult reference_exact_contact_step(const std::array<Vec3, 4>& start,
    const std::array<Vec3, 4>& end, bool face, double floor) {
    if (compare_contact_distance_exact(start, face, 0.0) == 0)
        return ExactContactStepResult::InitialContact;
    if (!contact_preserves_separation_exact(start, end, face, floor))
        return ExactContactStepResult::Unsafe;
    return single_vertex_ccd_between_exact(start, end, face).collision
        ? ExactContactStepResult::Unsafe : ExactContactStepResult::Safe;
}

// Independent copy of the pre-optimization exact distance algorithm. In
// particular this evaluates all four finite boundaries and reconstructs the
// interior residual; it does not use the production triple-product shortcut.
boost::multiprecision::cpp_rational reference_exact_segment_distance_squared(
    const std::array<Vec3, 4>& positions) {
    using Rational = boost::multiprecision::cpp_rational;
    using Point = std::array<Rational, 3>;
    std::array<Point, 4> p;
    for (int i = 0; i < 4; ++i)
        for (int axis = 0; axis < 3; ++axis) p[i][axis] = Rational(positions[i][axis]);
    const auto subtract = [](const Point& a, const Point& b) -> Point {
        return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
    };
    const auto dot = [](const Point& a, const Point& b) -> Rational {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const auto point_edge = [&](const Point& q, const Point& a, const Point& b) -> Rational {
        const Point edge = subtract(b, a), offset = subtract(q, a);
        const Rational length = dot(edge, edge), projection = dot(offset, edge);
        if (length == 0 || projection <= 0) return dot(offset, offset);
        if (projection >= length) {
            const Point delta = subtract(q, b);
            return dot(delta, delta);
        }
        return dot(offset, offset) - projection * projection / length;
    };
    Rational best = std::min({point_edge(p[0], p[2], p[3]), point_edge(p[1], p[2], p[3]),
        point_edge(p[2], p[0], p[1]), point_edge(p[3], p[0], p[1])});
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
            const Rational s = alpha / denominator, t = beta / denominator;
            Point delta;
            for (int axis = 0; axis < 3; ++axis)
                delta[axis] = offset[axis] + a[axis] * s - b[axis] * t;
            const Rational interior = dot(delta, delta);
            best = std::min(best, interior);
        }
    }
    return best;
}

// Rational NT reference retained independently of the integer-dyadic gap
// comparator: exact face feasibility, otherwise all three finite edges.
boost::multiprecision::cpp_rational reference_exact_triangle_distance_squared(
    const std::array<Vec3, 4>& positions) {
    using Rational = boost::multiprecision::cpp_rational;
    using Point = std::array<Rational, 3>;
    std::array<Point, 4> p;
    for (int i = 0; i < 4; ++i)
        for (int axis = 0; axis < 3; ++axis) p[i][axis] = Rational(positions[i][axis]);
    const auto subtract = [](const Point& a, const Point& b) -> Point {
        return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
    };
    const auto dot = [](const Point& a, const Point& b) -> Rational {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const auto edge_distance = [&](const Point& q, const Point& a, const Point& b) -> Rational {
        const Point edge = subtract(b, a), offset = subtract(q, a);
        const Rational length = dot(edge, edge), projection = dot(offset, edge);
        if (length == 0 || projection <= 0) return dot(offset, offset);
        if (projection >= length) {
            const Point residual = subtract(q, b);
            return dot(residual, residual);
        }
        return dot(offset, offset) - projection * projection / length;
    };
    const Point a = subtract(p[2], p[1]), b = subtract(p[3], p[1]);
    const Point offset = subtract(p[0], p[1]);
    const Point normal{{a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
    const Rational area = dot(normal, normal);
    if (area != 0) {
        const Rational aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
        const Rational ar = dot(offset, a), br = dot(offset, b);
        const Rational u = bb * ar - ab * br, v = aa * br - ab * ar;
        if (u >= 0 && v >= 0 && u + v <= area) {
            const Rational height = dot(offset, normal);
            return height * height / area;
        }
    }
    return std::min({edge_distance(p[0], p[1], p[2]),
        edge_distance(p[0], p[2], p[3]), edge_distance(p[0], p[3], p[1])});
}

// Preserve the production box/CCD prefix, but decide every represented trial
// using the old exact queries rather than any new floating certificate/cache.
double reference_single_contact_safe_step(const BroadPhase& phase, std::vector<Vec3>& x,
    int role, const Vec3& target, bool segment) {
    const auto& box = phase.cache().node_boxes[role];
    const Vec3 clipped = target.cwiseMax(box.min + Vec3::Constant(1e-10))
        .cwiseMin(box.max - Vec3::Constant(1e-10));
    const Vec3 before = x[role], dx = clipped - before;
    if ((clipped.array() == before.array()).all()) return 0.0;
    const CCDResult collision = segment
        ? safe_step_detail::segment_segment_vertex_ccd(phase.cache().ss_pairs[0], role, role, x, dx, false)
        : safe_step_detail::node_triangle_vertex_ccd(phase.cache().nt_pairs[0], role, role, x, dx, false);
    double step = collision.collision ? .9 * collision.t : 1.0;
    const std::array<Vec3, 4> start{{x[0], x[1], x[2], x[3]}};
    double magnitude = 0.0;
    for (const auto& p : start) magnitude = std::max(magnitude, p.cwiseAbs().maxCoeff());
    const double floor = std::max(1e-8, 128 * std::numeric_limits<double>::epsilon() * magnitude);
    const auto safe = [&](const Vec3& endpoint) {
        auto end = start;
        end[role] = endpoint;
        const auto verdict = reference_exact_contact_step(start, end, !segment, floor);
        if (verdict == ExactContactStepResult::InitialContact)
            throw std::runtime_error("reference safe step: exact initial contact");
        return verdict == ExactContactStepResult::Safe;
    };
    Vec3 accepted = before + step * dx;
    if (!safe(accepted)) {
        double lower = 0.0, upper = step;
        Vec3 best = before;
        for (int attempt = 0; attempt < 64; ++attempt) {
            const double middle = lower + .5 * (upper - lower);
            if (middle == lower || middle == upper) break;
            const Vec3 trial = before + middle * dx;
            if ((trial.array() == before.array()).all()) lower = middle;
            else if (safe(trial)) { lower = middle; best = trial; }
            else upper = middle;
        }
        step = lower;
        accepted = best;
    }
    x[role] = accepted;
    return (accepted.array() == before.array()).all() ? 0.0 : step;
}

void expect_safe_step_matches_exact(const std::array<Vec3, 4>& points,
    int role, const Vec3& target, bool segment,
    const safe_step_detail::VertexAabbRejections* rejections = nullptr) {
    std::vector<Vec3> actual(points.begin(), points.end()), expected = actual;
    BroadPhase phase;
    prepare_single_vertex_contact(phase, actual, segment, role);
    // Include extreme-scale targets without relying on a representable +4.
    for (auto& box : phase.mutable_cache().node_boxes) {
        box.expand(target);
        box.min -= Vec3::Constant(4.0);
        box.max += Vec3::Constant(4.0);
    }
    if (compare_contact_distance_exact(points, !segment, 0.0) == 0
        && (target.array() != points[role].array()).any()) {
        EXPECT_THROW(per_vertex_safe_step(phase, actual, role, target, .9, true, false,
            false, false, rejections), std::runtime_error);
        for (int i = 0; i < 4; ++i) EXPECT_TRUE((actual[i].array() == points[i].array()).all());
        return;
    }
    const double reference_step = reference_single_contact_safe_step(phase, expected, role, target, segment);
    const double actual_step = per_vertex_safe_step(phase, actual, role, target, .9, true, false,
        false, false, rejections);
    const auto bits = [](double value) {
        std::uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    };
    EXPECT_EQ(bits(actual_step), bits(reference_step));
    for (int i = 0; i < 4; ++i)
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_EQ(bits(actual[i][axis]), bits(expected[i][axis])) << "vertex=" << i << " axis=" << axis;
}

safe_step_detail::VertexAabbRejections single_contact_aabb_rejection(
    const std::array<Vec3, 4>& points, bool segment, double distance) {
    const int split = segment ? 2 : 1;
    Vec3 gap = Vec3::Zero();
    for (int axis = 0; axis < 3; ++axis) {
        double alo = INFINITY, ahi = -INFINITY, blo = INFINITY, bhi = -INFINITY;
        for (int i = 0; i < 4; ++i) {
            if (i < split) { alo = std::min(alo, points[i][axis]); ahi = std::max(ahi, points[i][axis]); }
            else { blo = std::min(blo, points[i][axis]); bhi = std::max(bhi, points[i][axis]); }
        }
        gap[axis] = std::max({0.0, alo - bhi, blo - ahi});
    }
    safe_step_detail::VertexAabbRejections result;
    result.distance = distance;
    result.clear = {static_cast<unsigned char>(gap.stableNorm() > distance)};
    return result;
}
}  // namespace

TEST(CCDLinearRobustness, ThinTriangleInteriorHit) {
    for (double height : {2e-7, 1e-9, 1e-13}) {
        SCOPED_TRACE(height);
        const auto r = node_triangle_only_one_node_moves(
            Vec3(0.75, height * 0.25, 1), Vec3(0, 0, -2),
            Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX,
            Vec3(1, height, 0), ZERO_DX, 1e-12, false);
        ASSERT_TRUE(r.collision);
        EXPECT_NEAR(r.t, 0.5, height < 1e-10 ? 2e-5 : 1e-12);
    }
}

TEST(CCDLinearRobustness, LargeTranslationDoesNotRoundAwayMotion) {
    const Vec3 shift = Vec3::Constant(std::ldexp(1.0, 50));
    const auto r = node_triangle_only_one_node_moves(
        shift + Vec3(0.25, 0.25, 0.25), Vec3(0, 0, -0.75),
        shift, ZERO_DX, shift + Vec3(1, 0, 0), ZERO_DX,
        shift + Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0 / 3.0, 1e-12);
}

TEST(CCDLinearRobustness, ScaleDoesNotOverflowOrUnderflow) {
    for (int exponent : {-500, -250, 0, 250, 500}) {
        SCOPED_TRACE(exponent);
        const double s = std::ldexp(1.0, exponent);
        const auto nt = node_triangle_only_one_node_moves(
            s * Vec3(0.25, 0.25, 1), s * Vec3(0, 0, -2),
            Vec3::Zero(), ZERO_DX, s * Vec3(1, 0, 0), ZERO_DX,
            s * Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto ss = segment_segment_only_one_node_moves(
            s * Vec3(0, 0, 1), s * Vec3(0, 0, -2), s * Vec3(1, 0, 0),
            s * Vec3(0.5, -1, 0), s * Vec3(0.5, 1, 0), 1e-12, false);
        const auto translation = segment_segment_same_displacement_linear_ccd(
            s * Vec3(0, 0, 1), s * Vec3(0, 0, -2), s * Vec3(1, 0, 1),
            s * Vec3(0, 0, -2), s * Vec3(0.5, -1, 0), s * Vec3(0.5, 1, 0));
        for (const auto& r : {nt, ss, translation}) {
            EXPECT_TRUE(r.collision);
            // Linear CCD distinguishes a positive represented initial gap
            // from true contact, even below the TICCD minimum separation.
            EXPECT_NEAR(r.t, 0.5, 1e-12);
        }
    }
}

TEST(CCDLinearRobustness, SmallInitialGapIsNotInitialContact) {
    const auto nt = node_triangle_only_one_node_moves(
        Vec3(0.25, 0.25, 5e-9), Vec3(0, 0, -1e-8),
        Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX,
        Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
    const auto ss = segment_segment_only_one_node_moves(
        Vec3(0, 0, 5e-9), Vec3(0, 0, -1e-8), Vec3(1, 0, 5e-9),
        Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, false);
    ASSERT_TRUE(nt.collision);
    EXPECT_NEAR(nt.t, 0.5, 1e-12);
    ASSERT_TRUE(ss.collision);
    EXPECT_NEAR(ss.t, 1.0, 1e-12);
}

TEST(CCDLinearRobustness, DegeneratePrimitivesAgreeWithTICCD) {
    for (const Vec3& b : {Vec3(1, 0, 0), Vec3::Zero().eval()}) {
        const Vec3 p(0, 0, 1), dp(0, 0, -2), a = Vec3::Zero();
        const auto linear = node_triangle_only_one_node_moves(
            p, dp, a, ZERO_DX, b, ZERO_DX, a, ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(
            p, dp, a, ZERO_DX, b, ZERO_DX, a, ZERO_DX, 1e-12, true);
        ASSERT_TRUE(reference.collision);
        ASSERT_TRUE(linear.collision);
        EXPECT_NEAR(linear.t, reference.t, 2e-6);
    }
    const Vec3 p(0, 0, 1), dp(0, 0, -2), a = Vec3::Zero();
    const auto linear = segment_segment_only_one_node_moves(p, dp, Vec3(1, 0, 0), a, a, 1e-12, false);
    const auto reference = segment_segment_only_one_node_moves(p, dp, Vec3(1, 0, 0), a, a, 1e-12, true);
    ASSERT_TRUE(reference.collision);
    ASSERT_TRUE(linear.collision);
    EXPECT_NEAR(linear.t, reference.t, 2e-6);
    const auto translated = segment_segment_same_displacement_linear_ccd(p, dp, p, dp, a, Vec3(1, 0, 0));
    ASSERT_TRUE(translated.collision);
    EXPECT_NEAR(translated.t, 0.5, 2e-6);
}

TEST(CCDLinearRobustness, SeededTOIAgreementWithTICCD) {
    std::mt19937 random(0xCCD2026);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int sample = 0; sample < 128; ++sample) {
        const Vec3 axis = Vec3(unit(random), unit(random), unit(random)).normalized();
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(3.0 * unit(random), axis).toRotationMatrix();
        const double scale = std::pow(10.0, sample % 3 - 1);
        const Vec3 offset = 100.0 * Vec3(unit(random), unit(random), unit(random));
        const double expected = 0.1 + 0.8 * (unit(random) + 1.0) / 2.0;
        const Vec3 motion = scale * rotation * Vec3(0.125, -0.25, 2.0);
        const auto transform = [&](const Vec3& p) -> Vec3 { return offset + scale * rotation * p; };
        const auto check = [&](const CCDResult& linear, const CCDResult& reference) {
            ASSERT_TRUE(reference.collision);
            ASSERT_TRUE(linear.collision);
            EXPECT_TRUE(std::isfinite(linear.t));
            EXPECT_NEAR(linear.t, expected, 2e-10);
            EXPECT_NEAR(linear.t, reference.t, 2e-5 / scale);
        };
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " NT role=" << role);
            std::array<Vec3, 4> x{{transform(Vec3(0.25, 0.25, 0)), transform(Vec3::Zero()),
                transform(Vec3(1, 0, 0)), transform(Vec3(0, 1, 0))}};
            std::array<Vec3, 4> dx{{ZERO_DX, ZERO_DX, ZERO_DX, ZERO_DX}};
            x[role] -= expected * motion;
            dx[role] = motion;
            check(node_triangle_only_one_node_moves(x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, false),
                  node_triangle_only_one_node_moves(x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, true));
        }
        for (int reversed = 0; reversed < 2; ++reversed) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " SS reversed=" << reversed);
            const Vec3 a = transform(Vec3(-1, 0, 0)), b = transform(Vec3(1, 0, 0));
            const Vec3 c = transform(Vec3(0, reversed ? 1 : -1, 0));
            const Vec3 d = transform(Vec3(0, reversed ? -1 : 1, 0));
            const Vec3 start = a - expected * motion;
            check(segment_segment_only_one_node_moves(start, motion, b, c, d, 1e-12, false),
                  segment_segment_only_one_node_moves(start, motion, b, c, d, 1e-12, true));
            const Vec3 end = b - expected * motion;
            check(segment_segment_same_displacement_linear_ccd(start, motion, end, motion, c, d),
                  CCDResult{true, segment_segment_general_ccd(start, motion, end, motion, c, ZERO_DX, d, ZERO_DX)});
        }
    }
}

TEST(CCDLinearRobustness, SeededSeparatedQueriesAgreeWithTICCD) {
    std::mt19937 random(0xCCD123);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int sample = 0; sample < 128; ++sample) {
        SCOPED_TRACE(sample);
        const Vec3 axis = Vec3(unit(random), unit(random), unit(random)).normalized();
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(unit(random), axis).toRotationMatrix();
        const auto p = [&](const Vec3& x) -> Vec3 { return rotation * x; };
        for (bool reference : {false, true}) {
            const auto nt = node_triangle_only_one_node_moves(p(Vec3(2, 2, 1)), p(Vec3(0, 0, -2)),
                Vec3::Zero(), ZERO_DX, p(Vec3(1, 0, 0)), ZERO_DX, p(Vec3(0, 1, 0)), ZERO_DX, 1e-12, reference);
            EXPECT_FALSE(nt.collision);
            EXPECT_TRUE(std::isnan(nt.t));
            const auto ss = segment_segment_only_one_node_moves(p(Vec3(0, 0, 1)), p(Vec3(0, 0, -2)),
                p(Vec3(1, 0, 0)), p(Vec3(2, -1, 0)), p(Vec3(2, 1, 0)), 1e-12, reference);
            EXPECT_FALSE(ss.collision);
            EXPECT_TRUE(std::isnan(ss.t));
        }
    }
}

TEST(CCDLinearRobustness, NearParallelCrossingsAgreeWithTICCD) {
    for (double angle : {1e-4, 1e-7, 1e-10, 1e-13}) {
        SCOPED_TRACE(angle);
        const Vec3 a(-1, -angle, 1), b(1, angle, 0), c(-1, 0, 0), d(1, 0, 0), dx(0, 0, -2);
        const auto linear = segment_segment_only_one_node_moves(a, dx, b, c, d, 1e-12, false);
        const auto reference = segment_segment_only_one_node_moves(a, dx, b, c, d, 1e-12, true);
        ASSERT_TRUE(linear.collision);
        ASSERT_TRUE(reference.collision);
        // Initially the exact finite-segment gap is angle/sqrt(1+4*angle^2),
        // strictly positive. Before t=1/2 only b lies in the fixed edge's
        // plane, and b.y()!=0; at t=1/2 the edges cross at their midpoints.
        EXPECT_NEAR(linear.t, 0.5, 1e-12);
        if (angle > 1e-9) {
            EXPECT_NEAR(linear.t, reference.t, 2e-6);
        } else {
            // These starts lie inside TICCD's unchanged 1e-10 minimum
            // separation. Its t=0 is not the exact-contact TOI oracle here.
            EXPECT_DOUBLE_EQ(reference.t, 0.0);
        }
    }
}

TEST(CCDLinearRobustness, ContactsNearTimeBoundaries) {
    for (double t : {0.0, 1e-12, 1.0 - 1e-12, 1.0}) {
        SCOPED_TRACE(t);
        const auto nt = node_triangle_only_one_node_moves(Vec3(0.25, 0.25, t), Vec3(0, 0, -1),
            Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto ss = segment_segment_only_one_node_moves(Vec3(0, 0, t), Vec3(0, 0, -1),
            Vec3(1, 0, 0), Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, false);
        ASSERT_TRUE(nt.collision);
        ASSERT_TRUE(ss.collision);
        EXPECT_NEAR(nt.t, t, 2e-12);
        EXPECT_NEAR(ss.t, t, 2e-12);
    }
}

TEST(CCDLinearRobustness, RotatedThinTriangleBoundariesAgreeWithTICCD) {
    for (double width : {1e-5, 1e-8, 1e-11}) {
        for (double angle : {0.0, 0.37, 1.41}) {
            const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Vec3(1, 2, 3).normalized()).toRotationMatrix();
            for (const Vec3& contact : {Vec3(0.75, width * 0.25, 0), Vec3(0.5, 0, 0), Vec3(1, width, 0)}) {
                SCOPED_TRACE(::testing::Message() << "width=" << width << " angle=" << angle << " contact=" << contact.transpose());
                const Vec3 p = rotation * (contact + Vec3(0, 0, 0.7));
                const Vec3 dp = rotation * Vec3(0, 0, -2);
                const Vec3 a = Vec3::Zero(), b = rotation * Vec3(1, 0, 0), c = rotation * Vec3(1, width, 0);
                const auto linear = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
                const auto reference = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, true);
                ASSERT_TRUE(reference.collision);
                ASSERT_TRUE(linear.collision);
                EXPECT_NEAR(linear.t, reference.t, 2e-5);
            }
        }
    }
}

TEST(CCDLinearRobustness, CoplanarMovingTriangleAndCollapseAgreeWithTICCD) {
    // The triangle changes orientation through a collapsed state at t=0.5.
    // The query point is first reached at t=0.625 (interior) or 0.75 (edge).
    for (double px : {-0.5, 0.0, 0.5}) {
        const Vec3 p(px, -0.25, 0), a(-1, 0, 0), b(1, 0, 0), c(0, -1, 0), dc(0, 2, 0);
        // Reverse the motion so the point is initially outside the triangle.
        const Vec3 start = c + dc;
        const auto linear = node_triangle_only_one_node_moves(p, ZERO_DX, a, ZERO_DX, b, ZERO_DX, start, -dc, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, ZERO_DX, a, ZERO_DX, b, ZERO_DX, start, -dc, 1e-12, true);
        ASSERT_TRUE(linear.collision);
        ASSERT_TRUE(reference.collision);
        EXPECT_NEAR(linear.t, px == 0.0 ? 0.625 : 0.75, 1e-12);
        // Coplanar inclusion searches can exhaust the reference iteration
        // budget and return a wider conservative bracket.
        EXPECT_NEAR(linear.t, reference.t, 5e-5);
    }
    const auto linear = segment_segment_only_one_node_moves(Vec3(0, 0, 1), Vec3(0, 0, -2), Vec3::Zero(),
        Vec3(-1, 0, 0), Vec3(1, 0, 0), 1e-12, false);
    ASSERT_TRUE(linear.collision);
    EXPECT_DOUBLE_EQ(linear.t, 0.0); // The stationary endpoint is already in contact.
}

TEST(CCDLinearRobustness, CoplanarAndParallelNearMissesStaySeparated) {
    for (double gap : {1e-5, 1e-7, 1e-9}) {
        SCOPED_TRACE(gap);
        for (bool reference : {false, true}) {
            EXPECT_FALSE(node_triangle_only_one_node_moves(Vec3(0.25, 0.25, gap), Vec3(0.1, 0, 0),
                Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, reference).collision);
            EXPECT_FALSE(segment_segment_only_one_node_moves(Vec3(0, gap, 0), Vec3(0.1, 0, 0),
                Vec3(1, gap, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), 1e-12, reference).collision);
        }
        EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(Vec3(0, gap, 0), Vec3(0.1, 0, 0),
            Vec3(1, gap, 0), Vec3(0.1, 0, 0), Vec3::Zero(), Vec3(1, 0, 0)).collision);
    }
}

TEST(CCDLinearRobustness, NearbyClothMovingAwayDoesNotFreezeSafeStep) {
    // Tilted cloth patches have overlapping swept AABBs while separated.
    // The old 1e-8 geometric padding reported contact at t=0; TICCD correctly
    // permits the entire step. Exercise the production vertex update path.
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.6, Vec3(1, 0, 1).normalized()).toRotationMatrix();
    const Vec3 normal = rotation * Vec3::UnitY();
    const Vec3 fall(0, -0.001, 0);
    for (double scale : {100.0, 10000.0}) {
        const double gap = 5e-9 * scale;
        for (bool reference : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "scale=" << scale << " TICCD=" << reference);
            std::vector<Vec3> x = {scale * rotation * Vec3(0.25, 0, 0.25) - gap * normal,
                Vec3::Zero(), scale * rotation * Vec3(1, 0, 0), scale * rotation * Vec3(0, 0, 1)};
            BroadPhase bp;
            auto& cache = bp.mutable_cache();
            cache.vertex_nt.resize(4);
            cache.vertex_ss.resize(4);
            cache.node_boxes.assign(4, AABB(Vec3::Constant(-2 * scale), Vec3::Constant(2 * scale)));
            NodeTrianglePair pair{};
            pair.node = 0;
            for (int i = 0; i < 3; ++i) pair.tri_v[i] = i + 1;
            cache.nt_pairs.push_back(pair);
            cache.vertex_nt[0].push_back({0, 0});
            const Vec3 target = x[0] + fall;
            const double step = per_vertex_safe_step(bp, x, 0, target, 0.9, true, reference);
            EXPECT_DOUBLE_EQ(step, 1.0);
            EXPECT_TRUE(x[0].isApprox(target, 1e-14));
        }
    }
}

TEST(CCDLinearRobustness, SkinnyTriangleDoesNotFreezeAnExteriorNode) {
    // Cancellation in the old Gram barycentrics put this exterior point
    // inside the triangle at t=0, although TICCD reports no contact.
    const Vec3 p(5000, 0.0002 * 0.501, 0), dx(0, 0.0002, 0);
    for (bool reference : {false, true}) {
        const auto r = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(10000, 0, 0), ZERO_DX, Vec3(10000, 0.0002, 0), ZERO_DX, 1e-12, reference);
        EXPECT_FALSE(r.collision);
        EXPECT_TRUE(std::isnan(r.t));
    }
}

TEST(CCDLinearRobustness, NearbyClothApproachingStillStopsBeforeContact) {
    const Vec3 a(0, 0, 0), b(1, 0, 0), c(0, 0, 1);
    const Vec3 p(0.25, 5e-9, 0.25), dx(0, -1e-8, 0);
    const auto r = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, 1e-12);
    EXPECT_GT((p + 0.9 * r.t * dx).y(), 0.0);
}

TEST(CCDLinearRobustness, ThinTriangleVertexHitSurvivesInterpolationRoundoff) {
    const std::array<Vec3, 3> corners{{Vec3::Zero(), Vec3(1, 0, 0), Vec3(1, 1e-8, 0)}};
    std::array<int, 3> permutation{{0, 1, 2}};
    do {
        const Vec3 a = corners[permutation[0]], b = corners[permutation[1]], c = corners[permutation[2]];
        const Vec3 p(1, -0.7, -0.7), dx(0, 1, 1);
        const auto linear = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, true);
        ASSERT_TRUE(reference.collision);
        ASSERT_TRUE(linear.collision);
        EXPECT_NEAR(linear.t, reference.t, 2e-5);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

TEST(CCDLinearRobustness, LongParallelEdgesDoNotLoseTransverseMotion) {
    const Vec3 a(0, 0, 1), b(1e170, 0, 1), c(0, 0, 0), d(1e170, 0, 0), dx(0, 0, -2);
    const auto r = segment_segment_same_displacement_linear_ccd(a, dx, b, dx, c, d);
    ASSERT_TRUE(r.collision);
    EXPECT_DOUBLE_EQ(r.t, 0.5);
}

TEST(CCDLinearRobustness, PositiveInitialGapDoesNotInheritTICCDMinimumSeparation) {
    for (double direction : {-1.0, 1.0}) {
        const Vec3 p(0.25, 0.25, 5e-11), dx(0, 0, direction * 1e-10);
        const auto linear = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, true);
        // TICCD retains its existing minimum-separation behavior. The linear
        // path must permit escape and still detect a genuine future crossing.
        ASSERT_TRUE(reference.collision);
        EXPECT_DOUBLE_EQ(reference.t, 0.0);
        if (direction < 0.0) {
            ASSERT_TRUE(linear.collision);
            EXPECT_NEAR(linear.t, 0.5, 1e-12);
        } else {
            EXPECT_FALSE(linear.collision);
            EXPECT_TRUE(std::isnan(linear.t));
        }
    }
}

TEST(CCDLinearRobustness, PositiveSubToleranceNodeTriangleGapsAcrossMovingRoles) {
    for (double gap : {5.0e-11, 1.0e-13, 1.0e-17}) {
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "gap=" << gap << " role=" << role);
            std::array<Vec3, 4> x{{Vec3(0.25, 0.25, 0.0), Vec3::Zero(),
                                  Vec3::UnitX(), Vec3::UnitY()}};
            x[role].z() = gap;
            std::array<Vec3, 4> dx{{ZERO_DX, ZERO_DX, ZERO_DX, ZERO_DX}};
            dx[role] = Vec3(0.0, 0.0, 1.0e-6);
            const auto away = node_triangle_only_one_node_moves(
                x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, false);
            EXPECT_FALSE(away.collision);
            EXPECT_TRUE(std::isnan(away.t));
            dx[role] = Vec3(0.0, 0.0, -2.0 * gap);
            const auto toward = node_triangle_only_one_node_moves(
                x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, false);
            ASSERT_TRUE(toward.collision);
            EXPECT_NEAR(toward.t, 0.5, 1e-12);
        }
    }
}

TEST(CCDLinearRobustness, PositiveSubToleranceEdgeGapsAcrossMovingRoles) {
    for (double gap : {5.0e-11, 1.0e-13, 1.0e-17}) {
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "gap=" << gap << " role=" << role);
            std::array<Vec3, 4> x{{Vec3(-1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
                                  Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0)}};
            x[role].z() = gap;
            const int partner = role ^ 1;
            const int other = role < 2 ? 2 : 0;
            const auto away = segment_segment_only_one_node_moves(
                x[role], Vec3(0.0, 0.0, 1.0e-6), x[partner], x[other], x[other + 1],
                1e-12, false);
            EXPECT_FALSE(away.collision);
            EXPECT_TRUE(std::isnan(away.t));
            const auto toward = segment_segment_only_one_node_moves(
                x[role], Vec3(0.0, 0.0, -2.0 * gap), x[partner], x[other], x[other + 1],
                1e-12, false);
            ASSERT_TRUE(toward.collision);
            EXPECT_NEAR(toward.t, 0.5, 1e-12);
        }
        const Vec3 a(-1.0, 0.0, gap), b(1.0, 0.0, gap);
        const Vec3 c(0.0, -1.0, 0.0), d(0.0, 1.0, 0.0);
        const Vec3 away_motion(0.0, 0.0, 1.0e-6), toward_motion(0.0, 0.0, -2.0 * gap);
        EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(
            a, away_motion, b, away_motion, c, d).collision);
        const auto toward = segment_segment_same_displacement_linear_ccd(
            a, toward_motion, b, toward_motion, c, d);
        ASSERT_TRUE(toward.collision);
        EXPECT_NEAR(toward.t, 0.5, 1e-12);
    }
}

TEST(CCDLinearRobustness, SubFloorSafeStepPreservesGapAndAllowsSubsequentEscape) {
    std::vector<Vec3> x{Vec3(0.2, 0.2, 1.1e-10), Vec3::Zero(),
                        Vec3::UnitX(), Vec3::UnitY()};
    BroadPhase bp;
    auto& cache = bp.mutable_cache();
    cache.node_boxes.assign(4, AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    cache.vertex_nt.resize(4);
    cache.vertex_ss.resize(4);
    cache.nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
    cache.vertex_nt[0].push_back({0, 0});
    const double approach_step = per_vertex_safe_step(
        bp, x, 0, x[0] + Vec3(0.0, 0.0, -1.0e-6), 0.9, true, false);
    EXPECT_DOUBLE_EQ(approach_step, 0.0);
    EXPECT_DOUBLE_EQ(x[0].z(), 1.1e-10);
    // Bypass the swept-AABB prefilter too: the axis-aligned triangle's AABB
    // alone can reject an outward move, concealing a broken CCD t=0 result.
    const auto direct_escape = node_triangle_only_one_node_moves(
        x[0], Vec3(0.0, 0.0, 1.0e-6), x[1], ZERO_DX,
        x[2], ZERO_DX, x[3], ZERO_DX, 1e-12, false);
    EXPECT_FALSE(direct_escape.collision);
    EXPECT_TRUE(std::isnan(direct_escape.t));
    const Vec3 escape_target = x[0] + Vec3(0.0, 0.0, 1.0e-6);
    const double escape_step = per_vertex_safe_step(
        bp, x, 0, escape_target, 0.9, true, false);
    EXPECT_DOUBLE_EQ(escape_step, 1.0);
    EXPECT_TRUE(x[0].isApprox(escape_target, 1.0e-15));
}

TEST(CCDSafeStepMargin, RepeatedInwardStepsPreserveFloorForEveryContactRole) {
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " role=" << role);
            auto x = planar_contact(segment);
            x[role].z() = .1;
            const auto initial = x;
            BroadPhase phase;
            prepare_single_vertex_contact(phase, x, segment, role);
            Vec3 target = x[role];
            target.z() = -.1;
            for (int attempt = 0; attempt < 30; ++attempt) {
                const double step = per_vertex_safe_step(phase, x, role, target, .9, true, false);
                EXPECT_GE(step, 0.0);
                EXPECT_LE(step, 1.0);
                EXPECT_GT(x[role].z(), 0.0);
                EXPECT_GE(contact_distance(x, segment), 1e-8 * (1.0 - 1e-12));
            }
            EXPECT_LT(contact_distance(x, segment), 1e-7);
            for (int stationary = 0; stationary < 4; ++stationary)
                if (stationary != role)
                    EXPECT_TRUE((x[stationary].array() == initial[stationary].array()).all());
        }
    }
}

TEST(CCDSafeStepMargin, SubFloorInwardMotionIsRejectedAndOutwardMotionIsAccepted) {
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            for (double gap : {5e-9, 5e-11, 5e-17}) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment << " role=" << role << " gap=" << gap);
                auto x = planar_contact(segment);
                x[role].z() = gap;
                const Vec3 before = x[role];
                BroadPhase phase;
                prepare_single_vertex_contact(phase, x, segment, role);
                Vec3 target = before;
                target.z() = .5 * gap;
                EXPECT_DOUBLE_EQ(per_vertex_safe_step(phase, x, role, target, .9, true, false), 0.0);
                EXPECT_TRUE((x[role].array() == before.array()).all());
                target.z() = 2.0 * gap;
                EXPECT_DOUBLE_EQ(per_vertex_safe_step(phase, x, role, target, .9, true, false), 1.0);
                EXPECT_TRUE((x[role].array() == target.array()).all());
            }
        }
    }
}

TEST(CCDSafeStepMargin, RoundedCommitCannotEraseOneUlpGapOrChangeSides) {
    for (bool segment : {false, true}) {
        for (double plane_height : {1.0, 1e8}) {
            for (int role = 0; role < 4; ++role) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment << " plane=" << plane_height << " role=" << role);
                auto x = planar_contact(segment);
                for (Vec3& p : x) p.z() = plane_height;
                x[role].z() = std::nextafter(plane_height, std::numeric_limits<double>::infinity());
                const Vec3 before = x[role];
                const double initial_distance = contact_distance(x, segment);
                ASSERT_GT(initial_distance, 0.0);
                BroadPhase phase;
                prepare_single_vertex_contact(phase, x, segment, role);
                Vec3 target = before;
                target.z() = plane_height - .5;
                per_vertex_safe_step(phase, x, role, target, .9, true, false);
                // The next inward representable height is exactly the contact
                // plane. An abstract positive TOI is insufficient to approve it.
                EXPECT_TRUE((x[role].array() == before.array()).all());
                EXPECT_GT(x[role].z(), plane_height);
                EXPECT_GE(contact_distance(x, segment), initial_distance);
                target.z() = plane_height + 1e-4;
                EXPECT_DOUBLE_EQ(per_vertex_safe_step(phase, x, role, target, .9, true, false), 1.0);
                EXPECT_TRUE((x[role].array() == target.array()).all());
            }
        }
    }
}

TEST(CCDSafeStepMargin, RepresentableSmallSeparatingMoveIsNotDiscarded) {
    auto x = planar_contact(false);
    x[0].z() = 5e-9;
    const auto before = x;
    BroadPhase phase;
    prepare_single_vertex_contact(phase, x, false, 0);
    const Vec3 target = x[0] + Vec3(0, 0, 1e-15);
    ASSERT_GT(target.z(), before[0].z());
    EXPECT_DOUBLE_EQ(per_vertex_safe_step(phase, x, 0, target, .9, true, false), 1.0);
    EXPECT_TRUE((x[0].array() == target.array()).all());
    for (int role = 1; role < 4; ++role)
        EXPECT_TRUE((x[role].array() == before[role].array()).all());
}

TEST(CCDSafeStepMargin, ObliquePlaneCertificatesPreserveExactEndpointAndPathSafety) {
    // Oblique primitive AABBs overlap even when a supporting-plane projection
    // can certify the update. Exercise every moving role and both approaches
    // and escapes; the independent exact predicate checks the actual committed
    // endpoints, not a rounded end-start reconstruction.
    Mat33 transform;
    transform << 1, 1, 1,
                -1, 1, 0,
                -1, -1, 2;
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            for (double scale : {1e-6, 1.0, 1e6}) {
                for (double target_height : {0.2, 0.05, -0.1}) {
                    SCOPED_TRACE(::testing::Message() << "segment=" << segment
                        << " role=" << role << " scale=" << scale
                        << " target_height=" << target_height);
                    auto x = planar_contact(segment);
                    x[role].z() = 0.1;
                    Vec3 target = x[role];
                    target.z() = target_height;
                    target = (scale * transform * target).eval();
                    for (auto& p : x) p = (scale * transform * p).eval();
                    const std::array<Vec3, 4> start{{x[0], x[1], x[2], x[3]}};
                    BroadPhase phase;
                    prepare_single_vertex_contact(phase, x, segment, role);
                    // Keep the requested displacement inside the test boxes
                    // even for the large-scale cases.
                    for (auto& box : phase.mutable_cache().node_boxes) {
                        box.min -= Vec3::Constant(4 * scale);
                        box.max += Vec3::Constant(4 * scale);
                    }
                    const double step = per_vertex_safe_step(
                        phase, x, role, target, 0.9, true, false);
                    EXPECT_GT(step, 0.0);
                    EXPECT_LE(step, 1.0);
                    const std::array<Vec3, 4> end{{x[0], x[1], x[2], x[3]}};
                    double coordinate_scale = 0.0;
                    for (const auto& p : start)
                        coordinate_scale = std::max(coordinate_scale, p.cwiseAbs().maxCoeff());
                    const double floor = std::max(1e-8,
                        128 * std::numeric_limits<double>::epsilon() * coordinate_scale);
                    EXPECT_EQ(reference_exact_contact_step(start, end, !segment, floor),
                        ExactContactStepResult::Safe);
                }
            }
        }
    }
}

TEST(CCDSafeStepMargin, DeterminantFilterMatchesExactBacktrackingAcrossRoles) {
    Mat33 transform;
    transform << 1, 1, 1, -1, 1, 0, -1, -1, 2;
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            const double shift = role % 2 ? 1e8 : 0.0;
            const double height = shift == 0.0 ? 5e-11 : 1e-7;
            auto local = planar_contact(segment);
            local[role].z() = height;
            std::array<Vec3, 4> start;
            for (int i = 0; i < 4; ++i)
                start[i] = transform * local[i] + Vec3::Constant(shift);
            // Both endpoints keep the same determinant sign. Escaping above
            // the floor is safe; staying on the same side is NOT sufficient
            // to approve the second, inward, sub-floor endpoint.
            for (double target_height : {1e-3, .25 * height}) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment
                    << " role=" << role << " shift=" << shift << " height=" << target_height);
                Vec3 target = local[role];
                target.z() = target_height;
                target = (transform * target + Vec3::Constant(shift)).eval();
                expect_safe_step_matches_exact(start, role, target, segment);
            }
        }
    }
}

TEST(CCDSafeStepMargin, FilteredRandomDegenerateAndExtremeQueriesMatchExact) {
    std::mt19937 random(0xCCD1A7E);
    std::uniform_int_distribution<int> integer(-8, 8);
    for (bool segment : {false, true}) {
        for (int sample = 0; sample < 12; ++sample) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " sample=" << sample);
            const int role = sample % 4;
            const double shift = sample % 3 == 0 ? 1e8 : 0.0;
            std::array<Vec3, 4> start;
            for (auto& p : start)
                for (int axis = 0; axis < 3; ++axis) p[axis] = shift + integer(random) / 4.0;
            if (sample % 4 == 0) start[3] = start[2]; // point edge / collapsed triangle
            if (sample % 4 == 1)
                for (auto& p : start) p.z() = shift; // identically coplanar determinant
            Vec3 target = start[role];
            for (int axis = 0; axis < 3; ++axis) target[axis] += integer(random) / 16.0;
            expect_safe_step_matches_exact(start, role, target, segment);
        }
        auto tiny = planar_contact(segment);
        tiny[0].z() = 1.0;
        const double scale = std::ldexp(1.0, -400);
        std::array<Vec3, 4> start;
        for (int i = 0; i < 4; ++i) start[i] = scale * tiny[i];
        Vec3 target = start[0];
        target.z() *= 2;
        // Determinant products underflow: the interval must be inconclusive,
        // not invent a sign or discard this representable separating motion.
        expect_safe_step_matches_exact(start, 0, target, segment);

        const double huge = std::ldexp(1.0, 53);
        start = segment
            ? std::array<Vec3, 4>{{Vec3(-1, 0, huge), Vec3(1, 0, 0),
                Vec3(0, -1, -.25), Vec3(0, 1, -.25)}}
            : std::array<Vec3, 4>{{Vec3(.25, .25, huge), Vec3(0, 0, -.5),
                Vec3(1, 0, -.5), Vec3(0, 1, -.5)}};
        target = start[0];
        target.z() = -1.0;
        // Endpoint subtraction loses a low term. Certify the represented
        // endpoints, not the different segment reconstructed from rounded dx.
        expect_safe_step_matches_exact(start, 0, target, segment);
    }
}

TEST(CCDSafeStepMargin, ValidAabbRejectionReuseMatchesUncachedExactReference) {
    for (bool segment : {false, true}) {
        const auto local = planar_contact(segment);
        std::array<Vec3, 4> start{{local[0], local[1], local[2], local[3]}};
        start[0].z() = .25;
        if (segment) start[1].z() = .25;
        const auto valid = single_contact_aabb_rejection(start, segment, .1);
        ASSERT_EQ(valid.clear[0], 1);
        auto wrong_size = valid;
        wrong_size.clear.push_back(1);
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " role=" << role);
            const Vec3 target = start[role] + Vec3(.002, -.002, -.003);
            expect_safe_step_matches_exact(start, role, target, segment);
            expect_safe_step_matches_exact(start, role, target, segment, &valid);
            expect_safe_step_matches_exact(start, role, target, segment, &wrong_size);
        }
    }
}

TEST(CCDSafeStepMargin, AabbRejectionCannotBypassSubnormalDistanceOrLargeCoordinateFloor) {
    for (bool segment : {false, true}) {
        for (bool translated : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " translated=" << translated);
            const double shift = translated ? 1e12 : 0.0;
            const double gap = translated ? .01 : 5e-11;
            const double distance = translated ? .004 : std::numeric_limits<double>::denorm_min();
            const auto local = planar_contact(segment);
            std::array<Vec3, 4> start;
            for (int i = 0; i < 4; ++i) start[i] = local[i] + Vec3::Constant(shift);
            start[0].z() += gap;
            if (segment) start[1].z() += gap;
            const auto valid = single_contact_aabb_rejection(start, segment, distance);
            ASSERT_EQ(valid.clear[0], 1); // genuinely valid current AABB certificate
            Vec3 target = start[0];
            target.z() = translated ? std::nextafter(target.z(), -INFINITY) : .5 * gap;
            // A tiny metadata distance (whose square underflows), or d_hat
            // below the coordinate-scaled floor, cannot authorize gap loss.
            expect_safe_step_matches_exact(start, 0, target, segment);
            expect_safe_step_matches_exact(start, 0, target, segment, &valid);
        }
    }
}

TEST(CCDSafeStepMargin, TrueInitialTouchThrowsWithoutChangingTheVertex) {
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            for (double direction : {-1.0, 1.0}) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment << " role=" << role << " direction=" << direction);
                auto x = planar_contact(segment);
                const Vec3 before = x[role];
                BroadPhase phase;
                prepare_single_vertex_contact(phase, x, segment, role);
                EXPECT_THROW(per_vertex_safe_step(phase, x, role,
                    before + Vec3(0, 0, direction * 1e-4), .9, true, false), std::runtime_error);
                EXPECT_TRUE((x[role].array() == before.array()).all());
            }
        }
    }
}

TEST(CCDSafeStepMargin, IllConditionedTriangleCannotBypassTheExistingGapFloor) {
    std::vector<Vec3> x{Vec3(333000, .0332667, 1e-11), Vec3::Zero(),
        Vec3(1e6, 0, 0), Vec3(1e6, .1, 0)};
    const Vec3 before = x[0];
    const std::array<Vec3, 4> original{{x[0], x[1], x[2], x[3]}};
    ASSERT_GT(compare_contact_distance_exact(original, true, 0.0), 0);
    ASSERT_LT(compare_contact_distance_exact(original, true, 1e-8), 0);
    BroadPhase phase;
    prepare_single_vertex_contact(phase, x, false, 0);
    Vec3 target = before;
    target.z() = 5e-12;
    // The exact projection is inside (y < 1e-7*x), while a cancellation-prone
    // Gram solve can select an edge and substantially overestimate the gap.
    EXPECT_DOUBLE_EQ(per_vertex_safe_step(phase, x, 0, target, .9, true, false), 0.0);
    EXPECT_TRUE((x[0].array() == before.array()).all());
}

TEST(CCDExactEndpoints, RoundedDisplacementCannotHideARepresentedEndpointCrossing) {
    const std::array<Vec3, 4> start{{Vec3(.25, .25, std::ldexp(1.0, 53)),
        Vec3(0, 0, -.5), Vec3(1, 0, -.5), Vec3(0, 1, -.5)}};
    auto end = start;
    end[0].z() = -1.0;
    const Vec3 rounded_motion = end[0] - start[0];
    ASSERT_EQ((start[0] + rounded_motion).z(), 0.0);
    ASSERT_GT((start[0] + rounded_motion).z(), start[1].z());
    ASSERT_LT(end[0].z(), start[1].z());
    const auto result = single_vertex_ccd_between_exact(start, end, true);
    ASSERT_TRUE(result.collision);
    EXPECT_GT(result.t, 0.0);
    EXPECT_LE(result.t, 1.0);
}

TEST(CCDExactEndpoints, SeparationFloorRequiresPositiveFiniteValue) {
    const std::array<Vec3, 4> start{{Vec3(.25, .25, 1e-9),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    EXPECT_GT(compare_contact_distance_exact(start, true, 0.0), 0);
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(contact_preserves_separation_exact(start, start, true, invalid), std::invalid_argument);
    }
}

TEST(CCDExactEndpoints, GapPreservationMatchesIndependentRationalReference) {
    using Rational = boost::multiprecision::cpp_rational;
    using Configuration = std::array<Vec3, 4>;
    std::mt19937 random(0xD1AD1C55);
    std::uniform_int_distribution<int> integer(-16, 16);
    for (bool segment : {false, true}) {
        std::vector<std::array<Configuration, 2>> cases;
        for (int role = 0; role < 4; ++role) {
            for (int exponent : {-1070, -600, 0, 600}) {
                const auto planar = planar_contact(segment);
                Configuration start{{planar[0], planar[1], planar[2], planar[3]}};
                start[role].z() = 1.0;
                const double scale = std::ldexp(1.0, exponent);
                for (auto& p : start) p *= scale;
                auto end = start;
                end[role].z() *= role % 2 ? 2.0 : .5;
                cases.push_back({{start, end}});
            }
        }
        for (int sample = 0; sample < 16; ++sample) {
            Configuration start;
            for (auto& p : start)
                for (int axis = 0; axis < 3; ++axis) p[axis] = integer(random) / 8.0;
            auto end = start;
            // This public API allows several changed vertices, unlike the
            // separate single-vertex path checker. Exercise that contract too.
            for (int role = 0; role < 4; ++role)
                if (sample % 2 || role == sample % 4)
                    for (int axis = 0; axis < 3; ++axis) end[role][axis] += integer(random) / 32.0;
            for (int role = 0; role < 4; ++role) {
                std::swap(start[role][0], start[role][sample % 3]);
                std::swap(end[role][0], end[role][sample % 3]);
            }
            cases.push_back({{start, end}});
        }
        for (int kind = 0; kind < 4; ++kind) {
            const auto planar = planar_contact(segment);
            Configuration start{{planar[0], planar[1], planar[2], planar[3]}};
            if (kind == 1) {
                start = {{Vec3(-.25, 0, 0), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}};
                if (segment) start[1] = start[0];
            } else if (kind == 2) {
                start = segment
                    ? Configuration{{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(-1, .25, 0), Vec3(1, .25, 0)}}
                    : Configuration{{Vec3(.25, .25, 0), Vec3::Zero(), Vec3::UnitX(), Vec3(2, 0, 0)}};
            } else if (kind == 3) {
                const double tiny = std::ldexp(1.0, -40), gap = std::ldexp(1.0, -45);
                start = segment
                    ? Configuration{{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(-1, -tiny, gap), Vec3(1, tiny, gap)}}
                    : Configuration{{Vec3(.5, tiny / 4, gap), Vec3::Zero(), Vec3::UnitX(), Vec3(1, tiny, 0)}};
            }
            auto end = start;
            end[0] += Vec3(.125, -.125, .125);
            if (kind % 2) end[1] += Vec3(.0625, 0, 0);
            cases.push_back({{start, end}});
        }
        for (int mix = 0; mix < 4; ++mix) {
            const auto planar = planar_contact(segment);
            Configuration start{{planar[0], planar[1], planar[2], planar[3]}};
            start[0].z() = 1.0;
            const std::array<int, 3> exponents{{500, -500, -1070}};
            for (auto& p : start)
                for (int axis = 0; axis < 3; ++axis)
                    p[axis] = mix == 3 ? p[axis] + std::ldexp(1.0, 45)
                        : std::ldexp(p[axis], exponents[(axis + mix) % 3]);
            auto end = start;
            end[0].z() = mix == 3 ? std::nextafter(start[0].z(), INFINITY) : start[0].z() * 2;
            cases.push_back({{start, end}});
        }
        ASSERT_EQ(cases.size(), 40u);
        for (std::size_t which = 0; which < cases.size(); ++which) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " case=" << which);
            const auto& start = cases[which][0];
            const auto& end = cases[which][1];
            const Rational before = segment ? reference_exact_segment_distance_squared(start)
                : reference_exact_triangle_distance_squared(start);
            const Rational after = segment ? reference_exact_segment_distance_squared(end)
                : reference_exact_triangle_distance_squared(end);
            std::vector<double> floors{std::numeric_limits<double>::denorm_min(), 1e-8, 1.0,
                std::numeric_limits<double>::max()};
            const auto add_gap_thresholds = [&](const Rational& squared) {
                if (squared == 0) return;
                // Normalize before converting: squared gaps can underflow or
                // overflow double even while the distance itself is finite.
                const auto numerator = boost::multiprecision::numerator(squared);
                const auto denominator = boost::multiprecision::denominator(squared);
                const int exponent = static_cast<int>(boost::multiprecision::msb(numerator))
                    - static_cast<int>(boost::multiprecision::msb(denominator));
                const double unit = std::ldexp(1.0, std::clamp(exponent / 2, -1074, 1023));
                const Rational normalized = squared / (Rational(unit) * Rational(unit));
                const double gap = std::sqrt(normalized.convert_to<double>()) * unit;
                for (double threshold : {gap, std::nextafter(gap, 0.0), std::nextafter(gap, INFINITY)})
                    if (threshold > 0.0 && std::isfinite(threshold)) floors.push_back(threshold);
            };
            add_gap_thresholds(before);
            add_gap_thresholds(after);
            std::sort(floors.begin(), floors.end());
            floors.erase(std::unique(floors.begin(), floors.end()), floors.end());
            for (double floor : floors) {
                const Rational floor_squared = Rational(floor) * Rational(floor);
                const bool expected = before != 0 && after >= std::min(before, floor_squared);
                EXPECT_EQ(contact_preserves_separation_exact(start, end, !segment, floor), expected)
                    << "floor=" << floor;
            }
        }
    }
}

TEST(CCDExactEndpoints, DirectGapPreservationRejectsNonfiniteCoordinates) {
    const std::array<Vec3, 4> good{{Vec3(.25, .25, 1e-9),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    for (bool face : {false, true}) {
        for (double invalid : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            auto bad = good;
            bad[3].x() = invalid;
            EXPECT_THROW(contact_preserves_separation_exact(bad, good, face, 1e-8), std::invalid_argument);
            EXPECT_THROW(contact_preserves_separation_exact(good, bad, face, 1e-8), std::invalid_argument);
        }
    }
}

TEST(CCDExactEndpoints, CombinedIntegerPathMatchesSeparateRationalQueries) {
    std::mt19937 random(0xE1AC7501);
    std::uniform_int_distribution<int> coordinate(-16, 16);
    for (bool segment : {false, true}) {
        for (int sample = 0; sample < 96; ++sample) {
            const int role = sample % 4;
            const auto planar = planar_contact(segment);
            std::array<Vec3, 4> start{{planar[0], planar[1], planar[2], planar[3]}};
            if (sample % 3 == 0) {
                for (auto& p : start)
                    for (int axis = 0; axis < 3; ++axis)
                        p[axis] = coordinate(random) / 8.0;
            } else if (sample % 3 == 1) {
                start[role].z() = coordinate(random) / 8.0;
            }
            if (sample % 11 == 0) start[3] = start[2]; // Degenerate primitive.
            auto end = start;
            end[role] += Vec3(coordinate(random), coordinate(random), coordinate(random)) / 16.0;
            if (sample % 13 == 0) end = start; // True no-op, including initial touch.
            const int exponent = sample % 7 == 0 ? -600 : (sample % 7 == 1 ? 500 : 0);
            const double scale = std::ldexp(1.0, exponent);
            for (int i = 0; i < 4; ++i) { start[i] *= scale; end[i] *= scale; }
            for (double floor : {scale * 1e-12, scale * 1e-8, scale * .125}) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment
                    << " sample=" << sample << " floor=" << floor);
                const auto expected = reference_exact_contact_step(start, end, !segment, floor);
                EXPECT_EQ(exact_contact_step_result(start, end, !segment, floor), expected);
                PreparedExactContactStep prepared(start, !segment, floor, role);
                EXPECT_EQ(prepared.test(end[role]), expected);
            }
        }
    }
}

TEST(CCDExactEndpoints, CombinedPathRetainsFarStartEventValidationBand) {
    // The exact plane crossing misses the finite triangle/segment by `gap`.
    // A far start retains the existing 1e-10 conservative event band, whereas
    // a near start uses exact membership. The endpoints themselves are safe.
    for (bool segment : {false, true}) {
        for (double gap : {5e-11, 1e-10, std::nextafter(1e-10, INFINITY), 2e-10}) {
            for (double height : {1.0, 5e-11}) {
                std::array<Vec3, 4> start = segment
                    ? std::array<Vec3, 4>{{Vec3(0, 0, height), Vec3(2, 0, 0),
                        Vec3(-gap, -1, 0), Vec3(-gap, 1, 0)}}
                    : std::array<Vec3, 4>{{Vec3(.5, -gap, height), Vec3::Zero(),
                        Vec3::UnitX(), Vec3::UnitY()}};
                auto end = start;
                end[0].z() = -height;
                const double floor = 1e-13;
                SCOPED_TRACE(::testing::Message() << "segment=" << segment
                    << " gap=" << gap << " height=" << height);
                EXPECT_EQ(exact_contact_step_result(start, end, !segment, floor),
                    reference_exact_contact_step(start, end, !segment, floor));
            }
        }
    }
}

TEST(CCDExactEndpoints, SegmentDistanceMatchesIndependentOldRationalAlgorithm) {
    using Rational = boost::multiprecision::cpp_rational;
    const double tiny = std::ldexp(1.0, -40);
    const std::vector<std::array<Vec3, 4>> fixtures{
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, .125), Vec3(0, 1, .125)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0), Vec3(0, 1, 0)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(1, 1, 1)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(2, -1, 1), Vec3(2, 1, 1)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(-1, 1, 0), Vec3(1, 1, 0)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 0), Vec3(2, 0, 0)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(3, 0, 0)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(.25, .25, .25), Vec3(.25, .25, .25)}},
        {{Vec3::Zero(), Vec3::Zero(), Vec3(.25, .5, 1), Vec3(.25, .5, 1)}},
        {{Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(-1, -tiny, tiny), Vec3(1, tiny, tiny)}}
    };
    std::vector<std::array<Vec3, 4>> cases;
    for (const auto& fixture : fixtures) {
        for (int exponent : {-600, -500, 0, 500, 600}) {
            auto scaled = fixture;
            for (auto& p : scaled) p *= std::ldexp(1.0, exponent);
            cases.push_back(scaled);
        }
        auto translated = fixture;
        for (auto& p : translated) p += Vec3::Constant(std::ldexp(1.0, 45));
        cases.push_back(translated);
    }
    std::mt19937 random(0xCCD55EED);
    std::uniform_int_distribution<int> coordinate(-32, 32);
    for (int sample = 0; sample < 128; ++sample) {
        std::array<Vec3, 4> points;
        for (auto& p : points)
            for (int axis = 0; axis < 3; ++axis) p[axis] = coordinate(random) / 16.0;
        cases.push_back(points);
    }
    const auto from_bits = [](std::uint64_t bits) {
        double value;
        static_assert(sizeof(value) == sizeof(bits), "binary64 double required");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    for (std::size_t which = 0; which < cases.size(); ++which) {
        SCOPED_TRACE(which);
        const auto& points = cases[which];
        const Rational reference = reference_exact_segment_distance_squared(points);
        ASSERT_GE(reference, 0);
        // Bracket sqrt(reference) using exact squared comparisons, avoiding
        // under/overflow when the squared distance is outside binary64 range.
        std::uint64_t lo = 0, hi = UINT64_C(0x7fefffffffffffff);
        while (lo < hi) {
            const std::uint64_t middle = lo + (hi - lo + 1) / 2;
            const Rational candidate(from_bits(middle));
            if (candidate * candidate <= reference) lo = middle;
            else hi = middle - 1;
        }
        double root = from_bits(lo);
        const double upper = std::nextafter(root, std::numeric_limits<double>::infinity());
        if (std::isfinite(upper)) {
            const Rational midpoint = (Rational(root) + Rational(upper)) / 2;
            const Rational middle_squared = midpoint * midpoint;
            if (reference > middle_squared || (reference == middle_squared && (lo & 1u)))
                root = upper;
        }
        const std::array<double, 6> thresholds{{0.0, 1e-8, 1.0, root,
            std::nextafter(root, 0.0),
            std::nextafter(root, std::numeric_limits<double>::infinity())}};
        for (int permutation = 0; permutation < 4; ++permutation) {
            auto ordered = points;
            if (permutation & 1) std::swap(ordered[0], ordered[1]);
            if (permutation & 2) {
                std::swap(ordered[0], ordered[2]);
                std::swap(ordered[1], ordered[3]);
            }
            for (double threshold : thresholds) {
                if (!std::isfinite(threshold)) continue;
                const Rational squared = Rational(threshold) * Rational(threshold);
                const int expected = reference < squared ? -1 : (reference > squared ? 1 : 0);
                EXPECT_EQ(compare_contact_distance_exact(ordered, false, threshold), expected)
                    << "permutation=" << permutation << " threshold=" << threshold;
            }
        }
    }
}

TEST(CCDExactEndpoints, CombinedCheckMatchesSeparateQueriesAcrossAllRolesAndGaps) {
    for (bool segment : {false, true}) {
        for (int role = 0; role < 4; ++role) {
            for (double height : {.1, 5e-9, 5e-11, 5e-17}) {
                const auto points = planar_contact(segment);
                std::array<Vec3, 4> start{{points[0], points[1], points[2], points[3]}};
                start[role].z() = height;
                ASSERT_GT(compare_contact_distance_exact(start, !segment, 0.0), 0);
                const PreparedExactContactStep prepared(start, !segment, 1e-8, role);
                for (double factor : {2.0, 1.0, .5, 0.0, -1.0}) {
                    SCOPED_TRACE(::testing::Message() << "segment=" << segment
                        << " role=" << role << " height=" << height << " factor=" << factor);
                    auto end = start;
                    end[role].z() = factor * height;
                    const auto reference = reference_exact_contact_step(start, end, !segment, 1e-8);
                    EXPECT_EQ(exact_contact_step_result(start, end, !segment, 1e-8), reference);
                    EXPECT_EQ(prepared.test(end[role]), reference);
                    if (factor >= 1.0)
                        EXPECT_EQ(reference, ExactContactStepResult::Safe);
                    if (factor <= 0.0 || (factor < 1.0 && height < 1e-8))
                        EXPECT_EQ(reference, ExactContactStepResult::Unsafe);
                }
            }
        }
    }
}

TEST(CCDExactEndpoints, CombinedCheckMatchesLargeTranslationsAndCancellation) {
    for (bool segment : {false, true}) {
        for (double shift : {1e8, std::ldexp(1.0, 45)}) {
            for (int role = 0; role < 4; ++role) {
                const auto points = planar_contact(segment);
                std::array<Vec3, 4> start{{points[0], points[1], points[2], points[3]}};
                for (auto& point : start) point += Vec3::Constant(shift);
                start[role].z() += .125;
                const PreparedExactContactStep small_floor(start, !segment, 1e-8, role);
                const PreparedExactContactStep large_floor(start, !segment, 1.0, role);
                for (double height : {.25, .125, .0625, 0.0, -.125}) {
                    SCOPED_TRACE(::testing::Message() << "segment=" << segment
                        << " shift=" << shift << " role=" << role << " height=" << height);
                    auto end = start;
                    end[role].z() = shift + height;
                    for (double floor : {1e-8, 1.0})
                        EXPECT_EQ(exact_contact_step_result(start, end, !segment, floor),
                            reference_exact_contact_step(start, end, !segment, floor));
                    EXPECT_EQ(small_floor.test(end[role]),
                        reference_exact_contact_step(start, end, !segment, 1e-8));
                    EXPECT_EQ(large_floor.test(end[role]),
                        reference_exact_contact_step(start, end, !segment, 1.0));
                }
            }
        }

        // The represented endpoint crosses, but subtraction loses the -1 low
        // term: start + round(end-start) has z=0 instead of z=-1.
        const double huge = std::ldexp(1.0, 53);
        const std::array<Vec3, 4> start = segment
            ? std::array<Vec3, 4>{{Vec3(-1, 0, huge), Vec3(1, 0, 0),
                Vec3(0, -1, -.25), Vec3(0, 1, -.25)}}
            : std::array<Vec3, 4>{{Vec3(.25, .25, huge), Vec3(0, 0, -.5),
                Vec3(1, 0, -.5), Vec3(0, 1, -.5)}};
        auto end = start;
        end[0].z() = -1.0;
        const Vec3 rounded_motion = end[0] - start[0];
        ASSERT_EQ((start[0] + rounded_motion).z(), 0.0);
        ASSERT_TRUE(contact_preserves_separation_exact(start, end, !segment, 1e-8));
        ASSERT_TRUE(single_vertex_ccd_between_exact(start, end, !segment).collision);
        EXPECT_EQ(reference_exact_contact_step(start, end, !segment, 1e-8),
            ExactContactStepResult::Unsafe);
        EXPECT_EQ(exact_contact_step_result(start, end, !segment, 1e-8),
            ExactContactStepResult::Unsafe);
        const PreparedExactContactStep prepared(start, !segment, 1e-8, 0);
        EXPECT_EQ(prepared.test(end[0]), ExactContactStepResult::Unsafe);
    }
}

TEST(CCDExactEndpoints, CombinedCheckMatchesCoplanarAndDegenerateQueries) {
    constexpr double gap = 1e-11;
    const std::array<std::array<Vec3, 4>, 4> starts{{
        {{Vec3(-gap, .25, 0), Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}},
        {{Vec3(-gap, 0, 0), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}},
        {{Vec3(-gap, 0, 0), Vec3(-1, 0, 0), Vec3::Zero(), Vec3::Zero()}},
        {{Vec3(-gap, 1, 0), Vec3(-gap, -1, 0), Vec3::Zero(), Vec3::Zero()}}
    }};
    for (std::size_t which = 0; which < starts.size(); ++which) {
        const bool face = which < 2;
        const auto& start = starts[which];
        ASSERT_GT(compare_contact_distance_exact(start, face, 0.0), 0);
        const PreparedExactContactStep prepared(start, face, 1e-8, 0);
        for (double target_x : {-1e-6, -gap, -.5 * gap, 0.0, gap, 5.0 * gap}) {
            SCOPED_TRACE(::testing::Message() << "case=" << which << " target_x=" << target_x);
            auto end = start;
            end[0].x() = target_x;
            EXPECT_EQ(exact_contact_step_result(start, end, face, 1e-8),
                reference_exact_contact_step(start, end, face, 1e-8));
            EXPECT_EQ(prepared.test(end[0]),
                reference_exact_contact_step(start, end, face, 1e-8));
        }
    }
    // A coplanar event at y=0 is not contact while x stays outside the triangle.
    const std::array<Vec3, 4> start{{Vec3(-gap, 1e-20, 0), Vec3::Zero(),
        Vec3::UnitX(), Vec3::UnitY()}};
    auto end = start;
    end[0] += Vec3(-1, -1, 0);
    EXPECT_EQ(reference_exact_contact_step(start, end, true, 1e-8), ExactContactStepResult::Safe);
    EXPECT_EQ(exact_contact_step_result(start, end, true, 1e-8), ExactContactStepResult::Safe);
    const PreparedExactContactStep prepared(start, true, 1e-8, 0);
    EXPECT_EQ(prepared.test(end[0]), ExactContactStepResult::Safe);
}

TEST(CCDExactEndpoints, CombinedCheckReportsInitialContactBeforeAnyEscape) {
    for (bool segment : {false, true}) {
        const auto points = planar_contact(segment);
        const std::array<Vec3, 4> start{{points[0], points[1], points[2], points[3]}};
        ASSERT_EQ(compare_contact_distance_exact(start, !segment, 0.0), 0);
        for (int role = 0; role < 4; ++role) {
            const PreparedExactContactStep prepared(start, !segment, 1e-8, role);
            for (double displacement : {-1e-4, 0.0, 1e-4}) {
                SCOPED_TRACE(::testing::Message() << "segment=" << segment
                    << " role=" << role << " displacement=" << displacement);
                auto end = start;
                end[role].z() += displacement;
                EXPECT_EQ(reference_exact_contact_step(start, end, !segment, 1e-8),
                    ExactContactStepResult::InitialContact);
                EXPECT_EQ(exact_contact_step_result(start, end, !segment, 1e-8),
                    ExactContactStepResult::InitialContact);
                EXPECT_EQ(prepared.test(end[role]), ExactContactStepResult::InitialContact);
            }
        }
    }
}

TEST(CCDExactEndpoints, CombinedCheckRejectsInvalidInputs) {
    const std::array<Vec3, 4> start{{Vec3(.25, .25, 1e-9),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(exact_contact_step_result(start, start, true, invalid), std::invalid_argument);
        EXPECT_THROW(contact_preserves_separation_exact(start, start, true, invalid), std::invalid_argument);
    }
    for (bool face : {false, true}) {
        for (double invalid : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            auto bad = start;
            bad[2].y() = invalid;
            EXPECT_THROW(exact_contact_step_result(bad, start, face, 1e-8), std::invalid_argument);
            EXPECT_THROW(exact_contact_step_result(start, bad, face, 1e-8), std::invalid_argument);
            EXPECT_THROW(compare_contact_distance_exact(bad, face, 0.0), std::invalid_argument);
            EXPECT_THROW(contact_preserves_separation_exact(start, bad, face, 1e-8), std::invalid_argument);
            EXPECT_THROW(single_vertex_ccd_between_exact(start, bad, face), std::invalid_argument);
        }
        auto end = start;
        end[0].z() += 1.0;
        end[1].z() += 1.0;
        EXPECT_THROW(exact_contact_step_result(start, end, face, 1e-8), std::invalid_argument);
        EXPECT_THROW(single_vertex_ccd_between_exact(start, end, face), std::invalid_argument);
    }
}

TEST(CCDExactEndpoints, CombinedCheckPreservesInputValidationPrecedence) {
    const std::array<Vec3, 4> good{{Vec3(.25, .25, 1),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    const auto expect_invalid = [](const std::array<Vec3, 4>& start,
        const std::array<Vec3, 4>& end, bool face, double floor,
        const char* message) {
        try {
            (void)exact_contact_step_result(start, end, face, floor);
            FAIL() << "Expected invalid_argument: " << message;
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), message);
        }
    };
    for (bool face : {false, true}) {
        for (double invalid : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            for (bool invalid_at_end : {false, true}) {
                auto start = good;
                auto end = good;
                end[0].z() = 2;
                end[1].z() = 2;
                // A later nonfinite coordinate must still be diagnosed before
                // the earlier two moving vertices, even if setup is reordered.
                (invalid_at_end ? end : start)[3].z() = invalid;
                expect_invalid(start, end, face, 1e-8,
                    "exact contact query: positions must be finite");
                expect_invalid(start, end, face, 0.0,
                    "exact contact query: separation floor must be finite and positive");
            }
            auto unchanged = good;
            unchanged[3].z() = invalid;
            // Equality/no-op detection must not bypass finite-input validation.
            expect_invalid(unchanged, unchanged, face, 1e-8,
                "exact contact query: positions must be finite");
        }
        auto end = good;
        end[0].z() = 2;
        end[1].z() = 2;
        expect_invalid(good, end, face, 1e-8,
            "exact endpoint CCD: at most one vertex may change");
    }
}

TEST(CCDExactEndpoints, CombinedCheckTreatsSignedZeroAsNoMotion) {
    for (bool segment : {false, true}) {
        for (bool initially_touching : {false, true}) {
            const auto planar = planar_contact(segment);
            std::array<Vec3, 4> start{{planar[0], planar[1], planar[2], planar[3]}};
            if (!initially_touching) start[0].z() = 1;
            auto end = start;
            // Change zero sign bits in every vertex without changing geometry.
            // Bitwise endpoint comparisons would falsely report many movers.
            for (int i = 0; i < 4; ++i)
                for (int axis = 0; axis < 3; ++axis)
                    if (start[i][axis] == 0.0) {
                        start[i][axis] = 0.0;
                        end[i][axis] = -0.0;
                    }
            const auto expected = initially_touching
                ? ExactContactStepResult::InitialContact : ExactContactStepResult::Safe;
            SCOPED_TRACE(::testing::Message() << "segment=" << segment
                << " initially_touching=" << initially_touching);
            EXPECT_EQ(reference_exact_contact_step(start, end, !segment, 1e-8), expected);
            EXPECT_EQ(exact_contact_step_result(start, end, !segment, 1e-8), expected);
        }
    }
}

TEST(CCDExactEndpoints, CombinedSetupHandlesEveryMovingRoleAtExtremeScales) {
    for (bool segment : {false, true}) {
        for (int exponent : {-1070, 1000}) {
            const double scale = std::ldexp(1.0, exponent);
            const double floor = std::ldexp(scale, -4);
            ASSERT_GT(floor, 0.0);
            for (int role = 0; role < 4; ++role) {
                const auto planar = planar_contact(segment);
                std::array<Vec3, 4> start{{planar[0], planar[1], planar[2], planar[3]}};
                for (auto& point : start) point *= scale;
                start[role].z() = scale;
                for (double height : {2 * scale, floor, -scale}) {
                    auto end = start;
                    end[role].z() = height;
                    SCOPED_TRACE(::testing::Message() << "segment=" << segment
                        << " exponent=" << exponent << " role=" << role
                        << " height=" << height);
                    // The smallest endpoint is subnormal in the tiny fixture;
                    // the huge fixture must not subtract/multiply in binary64.
                    EXPECT_EQ(exact_contact_step_result(start, end, !segment, floor),
                        reference_exact_contact_step(start, end, !segment, floor));
                }
            }
        }
    }
}

// Prepared queries must own their immutable start data, not borrow a trial.
TEST(CCDExactEndpoints, PreparedSnapshotIsImmutableAndMovable) {
    std::array<Vec3, 4> start{{Vec3(.25, .25, 1e-9),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    const auto original = start;
    PreparedExactContactStep first(start, true, 1e-8, 0);
    // The context owns its rational snapshot, not references into the caller.
    start[0].z() = -1.0;
    start[1].x() = 100.0;
    const std::array<Vec3, 6> endpoints{{Vec3(.25, .25, 2e-9),
        Vec3(.25, .25, 5e-10), original[0], Vec3(.25, .25, -1e-9),
        Vec3(.25, .25, 2e-9), original[0]}};
    const auto check = [&](const PreparedExactContactStep& prepared) {
        for (const auto& endpoint : endpoints) {
            auto end = original;
            end[0] = endpoint;
            EXPECT_EQ(prepared.test(endpoint),
                reference_exact_contact_step(original, end, true, 1e-8));
            EXPECT_EQ(prepared.test(endpoint), exact_contact_step_result(original, end, true, 1e-8));
        }
    };
    check(first);
    PreparedExactContactStep second(std::move(first));
    check(second);
    EXPECT_THROW(first.test(original[0]), std::logic_error);
    PreparedExactContactStep third(start, false, 1.0, 3);
    third = std::move(second);
    check(third);
    EXPECT_THROW(second.test(original[0]), std::logic_error);
}

TEST(CCDExactEndpoints, PreparedCheckRejectsInvalidConstructionAndEndpoints) {
    const std::array<Vec3, 4> start{{Vec3(.25, .25, 1e-9),
        Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()}};
    for (bool face : {false, true}) {
        for (int role : {-1, 4, std::numeric_limits<int>::max()})
            EXPECT_THROW(PreparedExactContactStep(start, face, 1e-8, role), std::invalid_argument);
        for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()})
            EXPECT_THROW(PreparedExactContactStep(start, face, invalid, 0), std::invalid_argument);
        const PreparedExactContactStep prepared(start, face, 1e-8, 0);
        for (double invalid : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            auto bad = start;
            bad[2].y() = invalid;
            EXPECT_THROW(PreparedExactContactStep(bad, face, 1e-8, 0), std::invalid_argument);
            Vec3 endpoint = start[0];
            endpoint.z() = invalid;
            EXPECT_THROW(prepared.test(endpoint), std::invalid_argument);
        }
        // An invalid trial must not corrupt the reusable prepared state.
        EXPECT_EQ(prepared.test(start[0]), ExactContactStepResult::Safe);
    }
}

TEST(CCDLinearRobustness, SavedExample20FrozenSpotNodesCanSeparate) {
    // Frame 12: Spot nodes 10032 and 8668 versus Bunny triangle vertices
    // (2408,3032,2993) and (1135,1792,1104). Fixed fixtures need no output files.
    const std::array<std::array<Vec3, 4>, 2> cases{{
        {{Vec3(0.37949910048798063, 1.3907199219247681, -0.34787540965691244),
          Vec3(0.3689223015013767, 1.3850539473661643, -0.3398516939181274),
          Vec3(0.3840675201667391, 1.388152011842588, -0.34602583510090795),
          Vec3(0.36653357413466214, 1.399693710388137, -0.3549113651578841)}},
        {{Vec3(0.20158578442515432, 1.2271647396074503, 0.21919103710650073),
          Vec3(0.21988724656197503, 1.2271372046858702, 0.21800678553994507),
          Vec3(0.1995573336994322, 1.2301827941558765, 0.19979342154412652),
          Vec3(0.20139308617333468, 1.2271096846587337, 0.21956198771926114)}}
    }};
    for (std::size_t which = 0; which < cases.size(); ++which) {
        SCOPED_TRACE(which);
        const auto& original = cases[which];
        const auto dr = node_triangle_distance(original[0], original[1], original[2], original[3]);
        ASSERT_EQ(dr.region, NodeTriangleRegion::FaceInterior);
        ASSERT_GT(dr.distance, 0.0);
        ASSERT_LT(dr.distance, 1.0e-10);
        const Vec3 outward = (dr.phi > 0.0 ? 1.0 : -1.0) * dr.normal * 1.0e-6;
        const PreparedExactContactStep prepared(original, true, 1e-8, 0);
        const std::array<Vec3, 3> endpoints{{original[0], original[0] + outward, original[0] - outward}};
        for (const auto& endpoint : endpoints) {
            auto end = original;
            end[0] = endpoint;
            EXPECT_EQ(prepared.test(endpoint), exact_contact_step_result(original, end, true, 1e-8));
            EXPECT_EQ(prepared.test(endpoint), reference_exact_contact_step(original, end, true, 1e-8));
        }
        const auto away = node_triangle_only_one_node_moves(
            original[0], outward, original[1], ZERO_DX,
            original[2], ZERO_DX, original[3], ZERO_DX, 1e-12, false);
        EXPECT_FALSE(away.collision);
        const auto toward = node_triangle_only_one_node_moves(
            original[0], -outward, original[1], ZERO_DX,
            original[2], ZERO_DX, original[3], ZERO_DX, 1e-12, false);
        ASSERT_TRUE(toward.collision);
        EXPECT_GT(toward.t, 0.0);
        EXPECT_LT(toward.t, 1.0e-3);
        EXPECT_NEAR(toward.t, dr.distance / outward.norm(), 1.0e-9);

        std::vector<Vec3> x(original.begin(), original.end());
        BroadPhase bp;
        auto& cache = bp.mutable_cache();
        cache.node_boxes.assign(4, AABB(Vec3::Constant(-4.0), Vec3::Constant(4.0)));
        cache.vertex_nt.resize(4);
        cache.vertex_ss.resize(4);
        cache.nt_pairs.push_back(NodeTrianglePair{0, {1, 2, 3}});
        cache.vertex_nt[0].push_back({0, 0});
        const Vec3 target = x[0] + outward;
        EXPECT_DOUBLE_EQ(per_vertex_safe_step(bp, x, 0, target, 0.9, true, false), 1.0);
        EXPECT_TRUE(x[0].isApprox(target, 1.0e-15));
    }
}

TEST(CCDLinearRobustness, CoplanarPositiveGapRejectsUnrelatedNearZeroEvent) {
    // Crossing y=0 at t=1e-20 is not contact: x remains strictly negative.
    // Reusing the far-query membership tolerance would invent a tiny TOI.
    const auto miss = node_triangle_only_one_node_moves(
        Vec3(-1.0e-11, 1.0e-20, 0.0), Vec3(-1.0, -1.0, 0.0),
        Vec3::Zero(), ZERO_DX, Vec3::UnitX(), ZERO_DX, Vec3::UnitY(), ZERO_DX,
        1e-12, false);
    EXPECT_FALSE(miss.collision);
    EXPECT_TRUE(std::isnan(miss.t));
}

TEST(CCDLinearRobustness, CoplanarAndDegenerateSubToleranceMotionKeepsTrueEvents) {
    constexpr double gap = 1.0e-11;
    const Vec3 away(-1.0e-6, 0.0, 0.0), toward(2.0 * gap, 0.0, 0.0);
    const Vec3 point(-gap, 0.25, 0.0);
    EXPECT_FALSE(node_triangle_only_one_node_moves(point, away,
        Vec3::Zero(), ZERO_DX, Vec3::UnitX(), ZERO_DX, Vec3::UnitY(), ZERO_DX,
        1e-12, false).collision);
    const auto triangle_hit = node_triangle_only_one_node_moves(point, toward,
        Vec3::Zero(), ZERO_DX, Vec3::UnitX(), ZERO_DX, Vec3::UnitY(), ZERO_DX,
        1e-12, false);
    ASSERT_TRUE(triangle_hit.collision);
    EXPECT_NEAR(triangle_hit.t, 0.5, 1e-12);
    const Vec3 endpoint(-gap, 0.0, 0.0);
    EXPECT_FALSE(node_triangle_only_one_node_moves(endpoint, away,
        Vec3::Zero(), ZERO_DX, Vec3::Zero(), ZERO_DX, Vec3::Zero(), ZERO_DX,
        1e-12, false).collision);
    const auto degenerate_hit = node_triangle_only_one_node_moves(endpoint, toward,
        Vec3::Zero(), ZERO_DX, Vec3::Zero(), ZERO_DX, Vec3::Zero(), ZERO_DX,
        1e-12, false);
    ASSERT_TRUE(degenerate_hit.collision);
    EXPECT_NEAR(degenerate_hit.t, 0.5, 1e-12);
    EXPECT_FALSE(segment_segment_only_one_node_moves(endpoint, away,
        Vec3(-1.0, 0.0, 0.0), Vec3::Zero(), Vec3::Zero(), 1e-12, false).collision);
    const auto edge_hit = segment_segment_only_one_node_moves(endpoint, toward,
        Vec3(-1.0, 0.0, 0.0), Vec3::Zero(), Vec3::Zero(), 1e-12, false);
    ASSERT_TRUE(edge_hit.collision);
    EXPECT_NEAR(edge_hit.t, 0.5, 1e-12);
    EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(
        endpoint, away, endpoint, away, Vec3::Zero(), Vec3::Zero()).collision);
    const auto point_hit = segment_segment_same_displacement_linear_ccd(
        endpoint, toward, endpoint, toward, Vec3::Zero(), Vec3::Zero());
    ASSERT_TRUE(point_hit.collision);
    EXPECT_NEAR(point_hit.t, 0.5, 1e-12);
}

TEST(CCDLinearRobustness, TrueInitialIntersectionsStillReturnZeroAcrossMovingRoles) {
    const std::array<Vec3, 4> nt{{Vec3(0.25, 0.25, 0.0), Vec3::Zero(),
                                Vec3::UnitX(), Vec3::UnitY()}};
    const std::array<Vec3, 4> ss{{Vec3(-1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
                                Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0)}};
    for (int role = 0; role < 4; ++role) {
        for (double sign : {-1.0, 1.0}) {
            SCOPED_TRACE(::testing::Message() << "role=" << role << " sign=" << sign);
            std::array<Vec3, 4> dx{{ZERO_DX, ZERO_DX, ZERO_DX, ZERO_DX}};
            dx[role] = Vec3(0.0, 0.0, sign * 1.0e-6);
            const auto nt_hit = node_triangle_only_one_node_moves(
                nt[0], dx[0], nt[1], dx[1], nt[2], dx[2], nt[3], dx[3], 1e-12, false);
            ASSERT_TRUE(nt_hit.collision);
            EXPECT_DOUBLE_EQ(nt_hit.t, 0.0);
            const int other = role < 2 ? 2 : 0;
            const auto ss_hit = segment_segment_only_one_node_moves(
                ss[role], dx[role], ss[role ^ 1], ss[other], ss[other + 1], 1e-12, false);
            ASSERT_TRUE(ss_hit.collision);
            EXPECT_DOUBLE_EQ(ss_hit.t, 0.0);
        }
    }
    const Vec3 motion(0.0, 0.0, 1.0e-6);
    const auto translation = segment_segment_same_displacement_linear_ccd(
        ss[0], motion, ss[1], motion, ss[2], ss[3]);
    ASSERT_TRUE(translation.collision);
    EXPECT_DOUBLE_EQ(translation.t, 0.0);
}

TEST(CCDLinearRobustness, RootJustOutsideStepDoesNotCreateFalseEndpointContact) {
    for (bool after_step : {false, true}) {
        // The root is only 1e-13 outside [0,1], but that corresponds to a
        // resolvable 1e-4 spatial gap at this speed. In particular, clamping
        // the negative root to zero must not freeze separating motion.
        const double height = after_step ? 1e9 + 1e-4 : 1e-4;
        const Vec3 dx(0, 0, after_step ? -1e9 : 1e9);
        for (bool reference : {false, true}) {
            EXPECT_FALSE(node_triangle_only_one_node_moves(Vec3(0.25, 0.25, height), dx,
                Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX,
                1e-12, reference).collision);
            EXPECT_FALSE(segment_segment_only_one_node_moves(Vec3(0, 0, height), dx, Vec3(1, 0, 0),
                Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, reference).collision);
        }
        EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(Vec3(0, 0, height), dx,
            Vec3(1, 0, height), dx, Vec3(0.5, -1, 0), Vec3(0.5, 1, 0)).collision);
    }
}

TEST(CCDLinearRobustness, BarycentricPaddingDoesNotCreateInitialContact) {
    const Vec3 a = Vec3::Zero(), b(1e9, 0, 0), c(0, 1e9, 0);
    const Vec3 p(0.5e9, (0.5 + 5e-13) * 1e9, 0), dp(0, 1e9, 0);
    for (bool reference : {false, true}) {
        const auto r = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, reference);
        EXPECT_FALSE(r.collision);
        const auto ss = segment_segment_only_one_node_moves(Vec3(0, 0, 0), Vec3(-1, 0, 0),
            Vec3(1e9, 0, 0), Vec3(1e9 + 5e-4, 0, 0), Vec3(2e9, 0, 0), 1e-12, reference);
        EXPECT_FALSE(ss.collision);
    }
}

TEST(CCDLinearRobustness, FastTangentialMotionDoesNotHideTransverseSeparation) {
    const Vec3 a(0, 1e-4, 1), b(1e12, 1e-4, 1), c = Vec3::Zero(), d(1e12, 0, 0), dx(1e12, 0, -2);
    const auto linear = segment_segment_same_displacement_linear_ccd(a, dx, b, dx, c, d);
    EXPECT_FALSE(linear.collision);
    EXPECT_DOUBLE_EQ(segment_segment_general_ccd(a, dx, b, dx, c, ZERO_DX, d, ZERO_DX), 1.0);
}

TEST(CCDNodeTriangleSingleMovingNode, InteriorHit) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x(0.25, 0.25, 0.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarForEntireStepEndpointHit) {
    // The node remains in the triangle's plane and first reaches the
    // hypotenuse exactly at the end of the step.
    const Vec3 x(1.5, 0.25, 0.0);
    const Vec3 dx(-0.75, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarCloseDistinctRootsUseTrueEntry) {
    // The point first crosses the extension of edge (x1,x2) at t=0, then
    // enters the finite triangle through edge (x2,x3) 5.56e-10 later. These
    // are distinct events even though their normalized times are very close.
    const Vec3 x(1.0 + 5.0e-8, 0.0, 0.0);
    const Vec3 dx(-100.0, 10.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const double expected_t = 5.0e-8 / 90.0;

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);

    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-15);
}

TEST(CCDNodeTriangleSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, 2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

// ===========================================================================
// Linear CCD with a single moving TRIANGLE VERTEX (any of the three corners).
// Same linear function as above; only the moving DOF differs.
//
// Setup uses triangle with vertices at (0,0,0), (2,0,0), (0,2,0). When the
// chosen corner rises in +z with dz=2, geometry was hand-derived so the
// moving plane sweeps across the static node x at exactly t=0.5.
// ===========================================================================

TEST(CCDNodeTriangleSingleMovingTriVertex, V0HitsStaticNode) {
    const Vec3 x(0.5, 0.5, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V1HitsStaticNode) {
    const Vec3 x(1.0, 0.5, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0), dx2(0.0, 0.0, 2.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, dx2, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V2HitsStaticNode) {
    const Vec3 x(0.5, 1.0, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0), dx3(0.0, 0.0, 2.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, ZERO_DX, x3, dx3,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, MovingVertexCoincidesWithStaticNode) {
    // Vertex x1 moves up through the static node x; at t=0.5, x1 == x exactly.
    const Vec3 x(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, CollinearForEntireStepEndpointHit) {
    // The moving endpoint extends the first segment until it first touches
    // the second segment exactly at the end of the step.
    const Vec3 x1(1.0, 0.0, 0.0);
    const Vec3 dx1(1.0, 0.0, 0.0);
    const Vec3 x2(0.0, 0.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarForEntireStepEndpointHit) {
    // The segments stay coplanar and non-collinear. The moving endpoint first
    // reaches the static segment exactly at the end of the step.
    const Vec3 x1(0.0, -1.0, 0.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, -1.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarCloseDistinctRootsUseTrueEntry) {
    // At t=0 the static endpoint x3 lies on the moving segment's supporting
    // line but is 5e-8 beyond the finite segment. The moving endpoint reaches
    // x3 at t=5e-10; the two events must not be merged.
    const Vec3 x1(5.0e-8, 0.0, 0.0);
    const Vec3 dx1(-100.0, 300.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 4.0, 0.0);
    const double expected_t = 5.0e-10;

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);

    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-15);
}

TEST(CCDSegmentSegmentSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSingleMovingNode, NearParallelHitAtNonSampledTime) {
    // At t=0.3 the moving segment is almost parallel to the static segment,
    // but crosses it at its midpoint. The TOI is not one of the historical
    // fixed probe times, so it must come from the affine coplanarity root.
    const Vec3 x1(0.0, -2.0e-7, -3.0e-7);
    const Vec3 dx1(0.0, 0.0, 1.0e-6);
    const Vec3 x2(1.0, 2.0e-7, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.3, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, NearParallelCoplanarityWithoutOverlap) {
    // The supporting lines become coplanar at t=0.3, but the finite segments
    // are disjoint. Coplanarity alone must not produce a collision.
    const Vec3 x1(0.0, -2.0e-7, -3.0e-7);
    const Vec3 dx1(0.0, 0.0, 1.0e-6);
    const Vec3 x2(1.0, 2.0e-7, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

// ---------------------------------------------------------------------------
// Regression guard for an example-4 segment pair whose distance increases
// throughout the sampled motion. It should remain a no-collision case.
TEST(CCDSegmentSegmentSingleMovingNode, RepoExample4Frame9Sub6Iter5) {
    const Vec3 x1(-5.33363314325580457e-02, 1.57380202214873172e-01, 3.54253855534581164e-01);
    const Vec3 dx1( 3.14637561392388425e-03,-1.90288924652220982e-03, 5.87251965795404507e-03);
    const Vec3 x2(-9.03270283891810799e-02, 1.85085196018025616e-01, 3.53048101908432388e-01);
    const Vec3 x3(-7.56544663979064197e-02, 1.74344892610936164e-01, 3.42000252922963821e-01);
    const Vec3 x4(-6.78743242217503817e-02, 1.67293411642532724e-01, 3.60000372162717852e-01);

    double d_min = 1e9; double t_at_min = -1.0;
    double d_prev = -1.0;
    for (int i = 0; i <= 100; ++i) {
        double t = i / 100.0;
        Vec3 x1t = x1 + dx1 * t;
        auto d = segment_segment_distance(x1t, x2, x3, x4);
        if (d.distance < d_min) { d_min = d.distance; t_at_min = t; }
        if (i > 0) {
            EXPECT_GE(d.distance, d_prev - 1e-12) << "distance should not decrease at sample " << i;
        }
        d_prev = d.distance;
    }

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
    EXPECT_NEAR(d_min, 4.36963887902074195e-04, 1e-15);
    EXPECT_DOUBLE_EQ(t_at_min, 0.0);
}

TEST(CCDSegmentSegmentSameDisplacement, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarityWithoutFiniteOverlap) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(2.0, -1.0, 0.0);
    const Vec3 x4(2.0,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, SpatialEndpointHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(1.0, 0.0, 0.0);
    const Vec3 x4(1.0, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, StepEndpointHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -1.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CZeroDNonzero) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, CandidateOutsideStepInterval) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, 1.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarSweep) {
    const Vec3 x1(0.0, -1.0, 0.0);
    const Vec3 dx(0.0, 2.0, 0.0);
    const Vec3 x2(1.0, -1.0, 0.0);
    const Vec3 x3(0.5, -0.25, 0.0);
    const Vec3 x4(0.5,  0.25, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.375, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarSweepWithoutFiniteOverlap) {
    const Vec3 x1(-2.0, 2.0, 0.0);
    const Vec3 dx(3.0, 0.0, 0.0);
    const Vec3 x2(-1.0, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelCollinearSweep) {
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 dx(2.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelTransverseSweep) {
    const Vec3 x1(0.0, 1.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 1.0, 0.0);
    const Vec3 x3(0.5, 0.0, 0.0);
    const Vec3 x4(1.5, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelTransverseSweepWithoutFiniteOverlap) {
    const Vec3 x1(0.0, 1.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 1.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, SmallTransverseVelocityStillFindsContact) {
    // A transverse component small relative to the total displacement must
    // not be rounded to zero: the moving segment reaches x4 at t=0.6.
    const Vec3 x1(0.0, 5.4e-8, 0.0);
    const Vec3 dx(10.0, -9.0e-8, 0.0);
    const Vec3 x2(1.0, 5.4e-8, 0.0);
    const Vec3 x3(5.0, 0.0, 0.0);
    const Vec3 x4(6.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.6, 1.0e-12);
}

TEST(CCDSegmentSegmentSameDisplacement, NearParallelCoplanarSweepUsesTrueEntry) {
    // The directions differ by only 1e-8 radians at this scale. Treating them
    // as exactly parallel delays the reported contact from x2/x4 at t=0.5 to
    // x1/x3 at t=1.
    const Vec3 x1(0.0, 2.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0e8, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0e8, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, TinyAngleStillUsesNonparallelRoot) {
    // Even an angle below eps has a well-defined nonparallel endpoint root.
    // Collapsing it to the parallel branch would delay contact until t=1.
    const Vec3 x1(0.0, 2.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0e8, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0e8, 9.0e-5, 0.0);
    const double expected_t = (2.0 - 9.0e-5) / 2.0;

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-12);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Rigid Body CCD Test /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
const Vec4 kIdentityQ(1.0, 0.0, 0.0, 0.0);

Vec4 AxisAngleQuat(const Vec3& axis, double angle) {
    const Vec3 normalized_axis = axis.normalized();
    const double sin_half_angle = std::sin(0.5 * angle);
    return Vec4(std::cos(0.5 * angle), sin_half_angle * normalized_axis[0], sin_half_angle * normalized_axis[1], sin_half_angle * normalized_axis[2]);
}
}  // namespace

TEST(SegmentSegmentRBRotationCCD, NoRotationReturnsFalse) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 2.0);
    const Vec3 x2(1.0, -0.5, 1.0);
    const Vec3 x3(2.0, 0.5, 1.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, kIdentityQ, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, SegmentOnAxisReturnsFalse) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 3.0);
    const Vec3 x2(1.0, 0.0, 2.0);
    const Vec3 x3(-1.0, 0.0, 2.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_FrustumHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 2.0);
    const Vec3 x2(1.0, -0.5, 1.0);
    const Vec3 x3(2.0, 0.5, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_PiercingAnnulusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, -1.0);
    const Vec3 x3(1.5, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB2_SamePlaneHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentGeneralHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 2.0, 3.0);
    const Vec3 x2(-0.10131289848292738, 0.60967344361641129, 1.5);
    const Vec3 x3(-0.26524061172707147, 1.596145797425957, 1.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentOppositeSidesHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(1.0, 0.0, -1.0);
    const Vec3 x1(0.0, 2.0, 2.0);
    const Vec3 x2(-0.10131289848292738, 0.60967344361641129, 0.5);
    const Vec3 x3(-0.26524061172707147, 1.596145797425957, 0.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentOppositeXSidesHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.5, -1.0);
    const Vec3 x1(1.0, -0.3, 2.0);
    const Vec3 x2(0.23511410091698948, -0.32360679774997875, 0.5);
    const Vec3 x3(-0.35267115137548433, 0.48541019662496826, 0.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_ArbitraryAxisHit) {
    const Vec3 x_com(0.5, -0.3, 0.2);
    const Vec3 x0(1.5, -0.3, -0.8);
    const Vec3 x1(0.5, 1.7, 2.2);
    const Vec3 x2(1.2351048854627524, 0.30605919321519798, 0.85883592132205056);
    const Vec3 x3(0.48919814277551343, 0.9666188030347671, 0.94418305418971882);

    const Vec4 q_new = AxisAngleQuat(Vec3(1, 1, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_HourglassEndpointTouchHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(2.0, 0.0, -2.0);
    const Vec3 x1(-2.0, 0.0, 2.0);
    const Vec3 x2(0.0, 1.0, 1.0);
    const Vec3 x3(0.0, 1.0, -1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_TrueInnerRadiusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.2, 0.0);
    const Vec3 x2(0.5, 0.0, -1.0);
    const Vec3 x3(0.5, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), -M_PI / 4.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.52395952173781901, 1e-9);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_InsideTrueInnerRadiusNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.2, 0.0);
    const Vec3 x2(0.1, 0.0, -1.0);
    const Vec3 x3(0.1, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_SkewTrueInnerRadiusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.5, 0.0);
    const Vec3 x2(0.6, 0.0, -1.0);
    const Vec3 x3(0.6, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), -M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.48624588465853014, 1e-9);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_SkewInsideTrueInnerRadiusNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.5, 0.0);
    const Vec3 x2(0.2, 0.0, -1.0);
    const Vec3 x3(0.2, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseB2_ParallelPlanesNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 5.0);
    const Vec3 x3(3.0, 0.0, 5.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(PointTriangleRBRotationCCD, CoplanarTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);
    const Vec3 x4(2.0, 1.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-14);
}

TEST(PointTriangleRBRotationCCD, TiltedAxisAlignedTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, -1.0);
    const Vec3 x3(3.0, 0.0, -1.0);
    const Vec3 x4(2.0, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-14);
}

TEST(PointTriangleRBRotationCCD, SkewTiltedTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, -0.5, -0.5);
    const Vec3 x3(2.7, 0.6, 0.4);
    const Vec3 x4(1.6, 0.7, 0.8);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.51132265526877663, 1e-14);
}

TEST(PointTriangleRBRotationCCD, RotationBeyond180Hit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, -1.0);
    const Vec3 x3(3.0, 0.0, -1.0);
    const Vec3 x4(2.0, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), 1.5 * M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 1.0 / 3.0, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, RotationBeyond180Hit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), 1.5 * M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 1.0 / 3.0, 1e-10);
}

TEST(CCDLinearRobustness, ExtremeAspectRatiosDoNotUnderflowIntoInitialContact) {
    const Vec3 zero = Vec3::Zero();
    for (double length : {1e165, 1e300}) {
        SCOPED_TRACE(length);
        const Vec3 p(0.25 * length, 0.25, 1), b(length, 0, 0), c(0, 1, 0);
        const auto away = node_triangle_only_one_node_moves(p, Vec3(0, 0, 1),
            zero, zero, b, zero, c, zero, 1e-12, false);
        EXPECT_FALSE(away.collision);
        const auto toward = node_triangle_only_one_node_moves(p, Vec3(0, 0, -2),
            zero, zero, b, zero, c, zero, 1e-12, false);
        ASSERT_TRUE(toward.collision);
        EXPECT_DOUBLE_EQ(toward.t, 0.5);
    }
    // Translating by a far corner must not merge the two distinct vertices
    // (0,0,0) and (1,0,0) before an ambiguous query is resolved.
    const auto thin = node_triangle_only_one_node_moves(Vec3(0.25, 0.25, 1), Vec3(0, 0, -2),
        zero, zero, Vec3(1, 0, 0), zero, Vec3(1e18, 1e18, 0), zero, 1e-12, false);
    ASSERT_TRUE(thin.collision);
    EXPECT_DOUBLE_EQ(thin.t, 0.5);
}

TEST(CCDLinearRobustness, SweptSeparationPreservesContactsAfterNormalizationLoss) {
    const Vec3 zero = Vec3::Zero(), down(0, 0, -2);
    const double tiny = std::ldexp(1.0, -55);
    const Vec3 a = zero, b(0.5, 1, 0), c(1, 0, 0);
    // Subtracting c.x() rounds tiny to the same local coordinate as zero.
    // A clear separation is safe to reject, but a real contact must survive.
    const auto miss = node_triangle_only_one_node_moves(Vec3(tiny, 2, 1), down,
        a, zero, b, zero, c, zero, 1e-12, false);
    EXPECT_FALSE(miss.collision);
    for (double y : {tiny, -5e-11}) {
        const auto hit = node_triangle_only_one_node_moves(Vec3(tiny, y, 1), down,
            a, zero, b, zero, c, zero, 1e-12, false);
        ASSERT_TRUE(hit.collision); // Includes the fixed boundary tolerance.
        EXPECT_DOUBLE_EQ(hit.t, 0.5);
    }
    const double large = std::numeric_limits<double>::max() * 0.75;
    for (double y : {-large, 0.0}) {
        const auto r = node_triangle_only_one_node_moves(Vec3(0, y, 1), down,
            Vec3(large, 0, 0), zero, Vec3(0, large, 0), zero,
            Vec3(-large, 0, 0), zero, 1e-12, false);
        EXPECT_EQ(r.collision, y == 0.0);
        if (r.collision) EXPECT_DOUBLE_EQ(r.t, 0.5);
    }
    // A positive gap remains separated even when both the entire primitive
    // and an outward displacement are much smaller than the old contact band.
    const double scale = std::ldexp(1.0, -800);
    const auto near = node_triangle_only_one_node_moves(scale * Vec3(0.25, 0.25, 1),
        Vec3(0, 0, std::numeric_limits<double>::denorm_min()), zero, zero,
        scale * Vec3(1, 0, 0), zero, scale * Vec3(0, 1, 0), zero, 1e-12, false);
    EXPECT_FALSE(near.collision);
    EXPECT_TRUE(std::isnan(near.t));
}

TEST(CCDIntegration, TenSeparatedClothsFollowGravityAtTenFixedIterations) {
    struct RestoreOpenMPSettings {
        const int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreOpenMPSettings() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    struct ClothScene {
        RefMesh mesh;
        DeformedState state;
        std::vector<Pin> pins;
        VertexTriangleMap adjacency;
        BroadPhase broad_phase;
        SimParams params;
    };
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    constexpr int frames = 15;
    constexpr int layers = 10;
    constexpr int subdivisions = 8;
    std::array<ClothScene, 3> scenes;
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode); // Per-vertex CCD off, independent linear, TICCD.
        auto& scene = scenes[mode];
        IPCArgs3D args;
        args.substeps = 5;
        args.max_substep_iters = 10;
        args.fixed_iters = true;
        args.gy = -9.8;
        args.use_ccd = mode != 0;
        args.use_ticcd = mode == 2;
        scene.params = args.to_sim_params();
        const auto& params = scene.params;
        std::vector<Vec2> material;
        for (int layer = 0; layer < layers; ++layer) {
            build_square_mesh(scene.mesh, scene.state, material, subdivisions, subdivisions,
                1.0, 1.0, Vec3(-0.5, 0.8 + layer * 0.02, -0.5));
        }
        scene.state.velocities.assign(scene.state.deformed_positions.size(), Vec3::Zero());
        scene.mesh.node_to_rb.assign(scene.state.deformed_positions.size(), -1);
        scene.mesh.build_deformable_nodes();
        scene.mesh.build_lumped_mass(params.density, params.thickness);
        scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
        const auto initial = scene.state.deformed_positions;
        for (int frame = 1; frame <= frames; ++frame) {
            SCOPED_TRACE(frame);
            const auto result = advance_one_frame(scene.state, scene.mesh, scene.adjacency,
                scene.pins, params, scene.broad_phase, frame);
            ASSERT_TRUE(result.converged);
            EXPECT_EQ(result.iterations, 50);
            const int steps = frame * params.substeps;
            const double fall = 0.5 * steps * (steps + 1) * params.dt2() * params.gravity.y();
            const double velocity = steps * params.dt() * params.gravity.y();
            for (std::size_t i = 0; i < initial.size(); ++i) {
                // All internal forces vanish for a uniform translation. The
                // default stiffness and limited sweeps must preserve free fall.
                EXPECT_NEAR(scene.state.deformed_positions[i].y(), initial[i].y() + fall, 2e-6) << "node=" << i;
                EXPECT_NEAR(scene.state.velocities[i].y(), velocity, 2e-6) << "node=" << i;
            }
        }
        for (std::size_t i = 0; i < initial.size(); ++i) {
            EXPECT_LT((scene.state.deformed_positions[i] - scenes[0].state.deformed_positions[i]).norm(), 2e-6);
        }
    }
}

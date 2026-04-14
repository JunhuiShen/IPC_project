#include "trust_region.h"

#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace {

constexpr double kTol = 1.0e-12;
constexpr double kEta = 0.4;  // matches the default in trust_region.h

double vt_distance(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3) {
    return node_triangle_distance(x, x1, x2, x3).distance;
}

double ee_distance(const Vec3& a1, const Vec3& a2, const Vec3& b1, const Vec3& b2) {
    return segment_segment_distance(a1, a2, b1, b2).distance;
}

// Triangle in the z = 0 plane, point one unit above its interior.
struct VTConfig {
    Vec3 x  = Vec3(0.25, 0.25, 1.0);
    Vec3 x1 = Vec3(0.0, 0.0, 0.0);
    Vec3 x2 = Vec3(1.0, 0.0, 0.0);
    Vec3 x3 = Vec3(0.0, 1.0, 0.0);
    // d0 = 1
};

// Two skew perpendicular edges separated by one unit along z.
struct EEConfig {
    Vec3 a1 = Vec3(0.0,  0.0, 0.0);
    Vec3 a2 = Vec3(1.0,  0.0, 0.0);
    Vec3 b1 = Vec3(0.5, -1.0, 1.0);
    Vec3 b2 = Vec3(0.5,  1.0, 1.0);
    // d0 = 1
};

const Vec3 ZERO(0.0, 0.0, 0.0);

}  // namespace

// ===========================================================================
// Vertex-triangle, full-motion API
// ===========================================================================

TEST(TrustRegionVertexTriangle, NoMotion_OmegaIsOne) {
    VTConfig c;
    const auto r = trust_region_vertex_triangle(
            c.x, ZERO, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);
    EXPECT_NEAR(r.d0, 1.0, kTol);
    EXPECT_NEAR(r.M,  0.0, kTol);
    EXPECT_NEAR(r.omega, 1.0, kTol);
}

TEST(TrustRegionVertexTriangle, HeadOnApproach) {
    VTConfig c;
    const Vec3 dx(0.0, 0.0, -2.0);
    const auto r = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);

    EXPECT_NEAR(r.d0, 1.0, kTol);
    EXPECT_NEAR(r.M,  2.0, kTol);
    EXPECT_NEAR(r.omega, kEta * 1.0 / 2.0, kTol);

    const double d_after = vt_distance(c.x + r.omega * dx, c.x1, c.x2, c.x3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
    EXPECT_GT(d_after, 0.0);
}

// Tangential motion does not change the true distance, but the bound is
// direction-blind so omega is still clamped. The post-step guarantee holds.
TEST(TrustRegionVertexTriangle, TangentialMotion) {
    VTConfig c;
    const Vec3 dx(0.5, 0.0, 0.0);
    const auto r = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);

    EXPECT_NEAR(r.M, 0.5, kTol);
    EXPECT_NEAR(r.omega, std::min(1.0, kEta * 1.0 / 0.5), kTol);

    const double d_after = vt_distance(c.x + r.omega * dx, c.x1, c.x2, c.x3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

TEST(TrustRegionVertexTriangle, MotionAwayFromTriangle) {
    VTConfig c;
    const Vec3 dx(0.0, 0.0, 2.0);
    const auto r = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);

    EXPECT_GT(r.omega, 0.0);
    EXPECT_LE(r.omega, 1.0 + kTol);

    const double d_after = vt_distance(c.x + r.omega * dx, c.x1, c.x2, c.x3);
    EXPECT_GT(d_after, r.d0 - kTol);
}

TEST(TrustRegionVertexTriangle, BothPointAndTriangleMove) {
    VTConfig c;
    const Vec3 dx (0.0, 0.0, -1.0);
    const Vec3 dx1(0.0, 0.0,  0.5);
    const Vec3 dx2(0.0, 0.0,  0.5);
    const Vec3 dx3(0.0, 0.0,  0.5);
    const auto r = trust_region_vertex_triangle(
            c.x, dx, c.x1, dx1, c.x2, dx2, c.x3, dx3);

    const double M_expected = dx.norm() + dx1.norm() + dx2.norm() + dx3.norm();
    EXPECT_NEAR(r.M, M_expected, kTol);
    EXPECT_NEAR(r.omega, std::min(1.0, kEta * 1.0 / M_expected), kTol);

    const double d_after = vt_distance(
            c.x  + r.omega * dx,
            c.x1 + r.omega * dx1,
            c.x2 + r.omega * dx2,
            c.x3 + r.omega * dx3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

// Without scaling, the unsafe step would punch the vertex through the
// triangle plane (z = -2). With omega applied, the vertex stays on the
// original side and the post-step distance bound holds.
TEST(TrustRegionVertexTriangle, PreventsCollision) {
    VTConfig c;
    const Vec3 dx(0.0, 0.0, -3.0);
    EXPECT_LT((c.x + dx).z(), 0.0);

    const auto r = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);
    EXPECT_GT(r.omega, 0.0);
    EXPECT_LT(r.omega, 1.0);

    const Vec3 x_safe = c.x + r.omega * dx;
    EXPECT_GT(x_safe.z(), 0.0);
    const double d_after = vt_distance(x_safe, c.x1, c.x2, c.x3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

TEST(TrustRegionVertexTriangle, AlreadyInContact_OmegaIsZero) {
    VTConfig c;
    const Vec3 x_on_triangle(0.25, 0.25, 0.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const auto r = trust_region_vertex_triangle(
            x_on_triangle, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO);
    EXPECT_NEAR(r.d0, 0.0, kTol);
    EXPECT_NEAR(r.omega, 0.0, kTol);
}

// ===========================================================================
// Vertex-triangle, Gauss-Seidel API
// ===========================================================================

TEST(TrustRegionVertexTriangleGS, OnlyPointMoves) {
    VTConfig c;
    const Vec3 delta(0.0, 0.0, -2.0);
    const auto r = trust_region_vertex_triangle_gauss_seidel(
            c.x, c.x1, c.x2, c.x3, delta);

    EXPECT_NEAR(r.M, 2.0, kTol);
    EXPECT_NEAR(r.omega, kEta * 1.0 / 2.0, kTol);

    const double d_after = vt_distance(c.x + r.omega * delta, c.x1, c.x2, c.x3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

TEST(TrustRegionVertexTriangleGS, OnlyTriangleVertexMoves) {
    VTConfig c;
    const Vec3 delta(0.0, 0.0, 0.5);
    const auto r = trust_region_vertex_triangle_gauss_seidel(
            c.x, c.x1, c.x2, c.x3, delta);

    EXPECT_NEAR(r.M, 0.5, kTol);
    EXPECT_NEAR(r.omega, std::min(1.0, kEta * 1.0 / 0.5), kTol);

    const double d_after = vt_distance(c.x, c.x1, c.x2 + r.omega * delta, c.x3);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

// ===========================================================================
// Edge-edge, full-motion API
// ===========================================================================

TEST(TrustRegionEdgeEdge, NoMotion_OmegaIsOne) {
    EEConfig c;
    const auto r = trust_region_edge_edge(
            c.a1, ZERO, c.a2, ZERO, c.b1, ZERO, c.b2, ZERO);
    EXPECT_NEAR(r.d0, 1.0, kTol);
    EXPECT_NEAR(r.M, 0.0, kTol);
    EXPECT_NEAR(r.omega, 1.0, kTol);
}

TEST(TrustRegionEdgeEdge, PerpendicularApproach) {
    EEConfig c;
    const Vec3 da1(0.0, 0.0, 2.0);
    const auto r = trust_region_edge_edge(
            c.a1, da1, c.a2, ZERO, c.b1, ZERO, c.b2, ZERO);

    EXPECT_NEAR(r.M, 2.0, kTol);
    EXPECT_NEAR(r.omega, kEta * 1.0 / 2.0, kTol);

    const double d_after = ee_distance(c.a1 + r.omega * da1, c.a2, c.b1, c.b2);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

TEST(TrustRegionEdgeEdge, AllFourMove) {
    EEConfig c;
    const Vec3 da1(0.0, 0.0,  0.5);
    const Vec3 da2(0.0, 0.0,  0.5);
    const Vec3 db1(0.0, 0.0, -0.5);
    const Vec3 db2(0.0, 0.0, -0.5);
    const auto r = trust_region_edge_edge(
            c.a1, da1, c.a2, da2, c.b1, db1, c.b2, db2);

    EXPECT_NEAR(r.M, 2.0, kTol);
    EXPECT_NEAR(r.omega, kEta * 1.0 / 2.0, kTol);

    const double d_after = ee_distance(
            c.a1 + r.omega * da1,
            c.a2 + r.omega * da2,
            c.b1 + r.omega * db1,
            c.b2 + r.omega * db2);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

// Drives both endpoints of edge A through edge B; omega must clamp them
// back to the safe side.
TEST(TrustRegionEdgeEdge, PreventsCollision) {
    EEConfig c;
    const Vec3 da1(0.0, 0.0, 3.0);
    const Vec3 da2(0.0, 0.0, 3.0);

    const auto r = trust_region_edge_edge(
            c.a1, da1, c.a2, da2, c.b1, ZERO, c.b2, ZERO);
    EXPECT_GT(r.omega, 0.0);
    EXPECT_LT(r.omega, 1.0);

    const Vec3 a1n = c.a1 + r.omega * da1;
    const Vec3 a2n = c.a2 + r.omega * da2;
    EXPECT_LT(a1n.z(), 1.0);
    EXPECT_LT(a2n.z(), 1.0);
    const double d_after = ee_distance(a1n, a2n, c.b1, c.b2);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

TEST(TrustRegionEdgeEdge, AlreadyInContact_OmegaIsZero) {
    const Vec3 a1(0.0, 0.0, 0.0);
    const Vec3 a2(1.0, 0.0, 0.0);
    const Vec3 b1(0.0, 0.0, 0.0);
    const Vec3 b2(0.0, 1.0, 0.0);
    const Vec3 da1(0.0, 0.0, 1.0);
    const auto r = trust_region_edge_edge(
            a1, da1, a2, ZERO, b1, ZERO, b2, ZERO);
    EXPECT_NEAR(r.d0, 0.0, kTol);
    EXPECT_NEAR(r.omega, 0.0, kTol);
}

// ===========================================================================
// Edge-edge, Gauss-Seidel API
// ===========================================================================

TEST(TrustRegionEdgeEdgeGS, OneEndpointMoves) {
    EEConfig c;
    const Vec3 delta(0.0, 0.0, 2.0);
    const auto r = trust_region_edge_edge_gauss_seidel(
            c.a1, c.a2, c.b1, c.b2, delta);

    EXPECT_NEAR(r.M, 2.0, kTol);
    EXPECT_NEAR(r.omega, kEta * 1.0 / 2.0, kTol);

    const double d_after = ee_distance(c.a1 + r.omega * delta, c.a2, c.b1, c.b2);
    EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - kTol);
}

// ===========================================================================
// Eta clamping
// ===========================================================================

TEST(TrustRegionEtaClamp, OutOfRangeIsClampedIntoOpenInterval) {
    VTConfig c;
    const Vec3 dx(0.0, 0.0, -2.0);

    const auto r_neg  = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO, -1.0);
    const auto r_huge = trust_region_vertex_triangle(
            c.x, dx, c.x1, ZERO, c.x2, ZERO, c.x3, ZERO,  5.0);

    EXPECT_GT(r_neg.omega, 0.0);
    EXPECT_LT(r_neg.omega, 1.0);
    EXPECT_GT(r_huge.omega, 0.0);
    EXPECT_LE(r_huge.omega, 0.5 / 2.0 + kTol);
}

// ===========================================================================
// Random-direction property tests.
//
// CCD verifies the time of first contact; trust region verifies the
// post-step distance bound directly. Each sample applies omega*delta and
// asserts the resulting primitive distance stays above (1 - eta) * d0,
// without ever solving a TOI equation.
// ===========================================================================

TEST(TrustRegionVertexTriangleSweep, RandomDirectionsAlwaysSafe) {
    VTConfig c;
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<double> N(0.0, 1.0);
    constexpr int kSamples = 200;
    constexpr double kStepNorm = 5.0;  // intentional overshoot

    for (int i = 0; i < kSamples; ++i) {
        Vec3 dx (N(rng), N(rng), N(rng));
        Vec3 dx1(N(rng), N(rng), N(rng));
        Vec3 dx2(N(rng), N(rng), N(rng));
        Vec3 dx3(N(rng), N(rng), N(rng));
        const double mag = dx.norm() + dx1.norm() + dx2.norm() + dx3.norm();
        if (mag < 1e-12) continue;
        const double s = kStepNorm / mag;
        dx *= s; dx1 *= s; dx2 *= s; dx3 *= s;

        const auto r = trust_region_vertex_triangle(
                c.x, dx, c.x1, dx1, c.x2, dx2, c.x3, dx3);
        EXPECT_GT(r.omega, 0.0);
        EXPECT_LE(r.omega, 1.0 + kTol);

        const double d_after = vt_distance(
                c.x  + r.omega * dx,
                c.x1 + r.omega * dx1,
                c.x2 + r.omega * dx2,
                c.x3 + r.omega * dx3);
        EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - 1e-10) << "sample " << i;
        EXPECT_GT(d_after, 0.0)                          << "sample " << i;
    }
}

TEST(TrustRegionEdgeEdgeSweep, RandomDirectionsAlwaysSafe) {
    EEConfig c;
    std::mt19937 rng(0xBADF00D);
    std::normal_distribution<double> N(0.0, 1.0);
    constexpr int kSamples = 200;
    constexpr double kStepNorm = 5.0;

    for (int i = 0; i < kSamples; ++i) {
        Vec3 da1(N(rng), N(rng), N(rng));
        Vec3 da2(N(rng), N(rng), N(rng));
        Vec3 db1(N(rng), N(rng), N(rng));
        Vec3 db2(N(rng), N(rng), N(rng));
        const double mag = da1.norm() + da2.norm() + db1.norm() + db2.norm();
        if (mag < 1e-12) continue;
        const double s = kStepNorm / mag;
        da1 *= s; da2 *= s; db1 *= s; db2 *= s;

        const auto r = trust_region_edge_edge(
                c.a1, da1, c.a2, da2, c.b1, db1, c.b2, db2);
        EXPECT_GT(r.omega, 0.0);
        EXPECT_LE(r.omega, 1.0 + kTol);

        const double d_after = ee_distance(
                c.a1 + r.omega * da1,
                c.a2 + r.omega * da2,
                c.b1 + r.omega * db1,
                c.b2 + r.omega * db2);
        EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - 1e-10) << "sample " << i;
        EXPECT_GT(d_after, 0.0)                          << "sample " << i;
    }
}

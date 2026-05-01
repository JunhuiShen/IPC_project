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

}  // namespace

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

TEST(TrustRegionVertexTriangleGS, AlreadyInContact_OmegaIsZero) {
    VTConfig c;
    const Vec3 x_on_triangle(0.25, 0.25, 0.0);
    const Vec3 delta(1.0, 0.0, 0.0);
    const auto r = trust_region_vertex_triangle_gauss_seidel(
            x_on_triangle, c.x1, c.x2, c.x3, delta);
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

TEST(TrustRegionEdgeEdgeGS, AlreadyInContact_OmegaIsZero) {
    const Vec3 a1(0.0, 0.0, 0.0);
    const Vec3 a2(1.0, 0.0, 0.0);
    const Vec3 b1(0.0, 0.0, 0.0);
    const Vec3 b2(0.0, 1.0, 0.0);
    const Vec3 delta(0.0, 0.0, 1.0);
    const auto r = trust_region_edge_edge_gauss_seidel(
            a1, a2, b1, b2, delta);
    EXPECT_NEAR(r.d0, 0.0, kTol);
    EXPECT_NEAR(r.omega, 0.0, kTol);
}

// ===========================================================================
// Eta clamping
// ===========================================================================

TEST(TrustRegionEtaClamp, OutOfRangeIsClampedIntoOpenInterval) {
    VTConfig c;
    const Vec3 delta(0.0, 0.0, -2.0);

    const auto r_neg  = trust_region_vertex_triangle_gauss_seidel(
            c.x, c.x1, c.x2, c.x3, delta, -1.0);
    const auto r_huge = trust_region_vertex_triangle_gauss_seidel(
            c.x, c.x1, c.x2, c.x3, delta,  5.0);

    EXPECT_GT(r_neg.omega, 0.0);
    EXPECT_LT(r_neg.omega, 1.0);
    EXPECT_GT(r_huge.omega, 0.0);
    EXPECT_LE(r_huge.omega, 0.5 / 2.0 + kTol);
}

// ===========================================================================
// Random-direction property tests (Gauss-Seidel only).
//
// CCD verifies the time of first contact; trust region verifies the
// post-step distance bound directly. Each sample applies omega*delta and
// asserts the resulting primitive distance stays above (1 - eta) * d0.
// ===========================================================================

TEST(TrustRegionVertexTriangleGSSweep, RandomDirectionsAlwaysSafe) {
    VTConfig c;
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<double> N(0.0, 1.0);
    constexpr int kSamples = 200;
    constexpr double kStepNorm = 5.0;  // intentional overshoot

    for (int i = 0; i < kSamples; ++i) {
        Vec3 delta(N(rng), N(rng), N(rng));
        const double mag = delta.norm();
        if (mag < 1e-12) continue;
        delta *= kStepNorm / mag;

        const auto r = trust_region_vertex_triangle_gauss_seidel(
                c.x, c.x1, c.x2, c.x3, delta);
        EXPECT_GT(r.omega, 0.0);
        EXPECT_LE(r.omega, 1.0 + kTol);

        const double d_after = vt_distance(c.x + r.omega * delta, c.x1, c.x2, c.x3);
        EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - 1e-10) << "sample " << i;
        EXPECT_GT(d_after, 0.0)                          << "sample " << i;
    }
}

TEST(TrustRegionEdgeEdgeGSSweep, RandomDirectionsAlwaysSafe) {
    EEConfig c;
    std::mt19937 rng(0xBADF00D);
    std::normal_distribution<double> N(0.0, 1.0);
    constexpr int kSamples = 200;
    constexpr double kStepNorm = 5.0;

    for (int i = 0; i < kSamples; ++i) {
        Vec3 delta(N(rng), N(rng), N(rng));
        const double mag = delta.norm();
        if (mag < 1e-12) continue;
        delta *= kStepNorm / mag;

        const auto r = trust_region_edge_edge_gauss_seidel(
                c.a1, c.a2, c.b1, c.b2, delta);
        EXPECT_GT(r.omega, 0.0);
        EXPECT_LE(r.omega, 1.0 + kTol);

        const double d_after = ee_distance(c.a1 + r.omega * delta, c.a2, c.b1, c.b2);
        EXPECT_GE(d_after, (1.0 - kEta) * r.d0 - 1e-10) << "sample " << i;
        EXPECT_GT(d_after, 0.0)                          << "sample " << i;
    }
}

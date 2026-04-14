#include "IPC_math.h"
#include "physics.h"
#include "broad_phase.h"
#include "make_shape.h"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <stdexcept>

// ====================================================================
//  matrix3d_inverse
// ====================================================================

TEST(Matrix3dInverse, IdentityInverse) {
    Mat33 I = Mat33::Identity();
    Mat33 inv = matrix3d_inverse(I);
    EXPECT_NEAR((inv - I).lpNorm<Eigen::Infinity>(), 0.0, 1e-14);
}

TEST(Matrix3dInverse, KnownInverse) {
    Mat33 A;
    A << 2, 1, 0,
         0, 3, 1,
         1, 0, 2;
    Mat33 inv = matrix3d_inverse(A);
    Mat33 prod = A * inv;
    EXPECT_NEAR((prod - Mat33::Identity()).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

// ====================================================================
//  clamp_scalar
// ====================================================================

TEST(ClampScalar, ClampsBelowAndAbove) {
    EXPECT_DOUBLE_EQ(clamp_scalar(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(clamp_scalar(2.0, 0.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(clamp_scalar(0.5, 0.0, 1.0), 0.5);
}

// ====================================================================
//  segment_closest_point
// ====================================================================

TEST(SegmentClosestPoint, InteriorProjection) {
    Vec3 a(0,0,0), b(2,0,0), x(1,1,0);
    double t;
    Vec3 cp = segment_closest_point(x, a, b, t);
    EXPECT_NEAR(t, 0.5, 1e-14);
    EXPECT_NEAR((cp - Vec3(1,0,0)).norm(), 0.0, 1e-14);
}

TEST(SegmentClosestPoint, ClampToEndpoints) {
    Vec3 a(0,0,0), b(1,0,0);
    double t;
    segment_closest_point(Vec3(-1,0,0), a, b, t);
    EXPECT_DOUBLE_EQ(t, 0.0);
    segment_closest_point(Vec3(5,0,0), a, b, t);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(SegmentClosestPoint, DegenerateZeroLength) {
    Vec3 a(1,2,3), b(1,2,3);
    double t;
    Vec3 cp = segment_closest_point(Vec3(5,5,5), a, b, t);
    EXPECT_DOUBLE_EQ(t, 0.0);
    EXPECT_NEAR((cp - a).norm(), 0.0, 1e-14);
}

// ====================================================================
//  nearly_zero, in_unit_interval, filter_root
// ====================================================================

TEST(NearlyZero, Basics) {
    EXPECT_TRUE(nearly_zero(0.0));
    EXPECT_TRUE(nearly_zero(1e-13));
    EXPECT_FALSE(nearly_zero(1e-10));
}

TEST(InUnitInterval, BoundaryValues) {
    EXPECT_TRUE(in_unit_interval(0.0));
    EXPECT_TRUE(in_unit_interval(1.0));
    EXPECT_TRUE(in_unit_interval(-1e-13));   // within eps
    EXPECT_TRUE(in_unit_interval(1.0 + 1e-13));
    EXPECT_FALSE(in_unit_interval(-0.1));
    EXPECT_FALSE(in_unit_interval(1.1));
}

TEST(FilterRoot, AcceptsValidRoots) {
    EXPECT_NEAR(filter_root(0.5), 0.5, 1e-14);
    EXPECT_NEAR(filter_root(0.0), 0.0, 1e-14);
    EXPECT_NEAR(filter_root(1.0), 1.0, 1e-14);
}

TEST(FilterRoot, RejectsBeyondInterval) {
    EXPECT_LT(filter_root(-0.1), 0.0);
    EXPECT_LT(filter_root(1.1), 0.0);
}

TEST(FilterRoot, ClampsNearBoundary) {
    double r = filter_root(-1e-13);
    EXPECT_GE(r, 0.0);
    EXPECT_LE(r, 1e-12);
}

// ====================================================================
//  add_root / SmallRoots
// ====================================================================

TEST(AddRoot, SmallRootsBasics) {
    SmallRoots roots;
    add_root(roots, 0.5);
    add_root(roots, 0.3);
    add_root(roots, 0.5);  // duplicate — should be skipped
    EXPECT_EQ(roots.size(), 2);
}

TEST(AddRoot, SmallRootsRejectsOutOfRange) {
    SmallRoots roots;
    add_root(roots, -0.5);
    add_root(roots, 1.5);
    EXPECT_EQ(roots.size(), 0);
}

TEST(AddRoot, SmallRootsSortable) {
    SmallRoots roots;
    add_root(roots, 0.8);
    add_root(roots, 0.2);
    add_root(roots, 0.5);
    std::sort(roots.begin(), roots.end());
    EXPECT_LT(roots[0], roots[1]);
    EXPECT_LT(roots[1], roots[2]);
}

// ====================================================================
//  cross_product_in_2d
// ====================================================================

TEST(CrossProduct2D, KnownValue) {
    Vec2 a(1, 0), b(0, 1);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(a, b), 1.0);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(b, a), -1.0);
}

TEST(CrossProduct2D, ParallelVectorsGiveZero) {
    Vec2 a(2, 4), b(1, 2);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(a, b), 0.0);
}

// ====================================================================
//  triangle_plane_barycentric_coordinates
// ====================================================================

TEST(Barycentric, VerticesReturnCorrectCoords) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0);
    auto b1 = triangle_plane_barycentric_coordinates(x1, x1, x2, x3);
    EXPECT_NEAR(b1[0], 1.0, 1e-12);
    EXPECT_NEAR(b1[1], 0.0, 1e-12);
    EXPECT_NEAR(b1[2], 0.0, 1e-12);

    auto b2 = triangle_plane_barycentric_coordinates(x2, x1, x2, x3);
    EXPECT_NEAR(b2[1], 1.0, 1e-12);

    auto b3 = triangle_plane_barycentric_coordinates(x3, x1, x2, x3);
    EXPECT_NEAR(b3[2], 1.0, 1e-12);
}

TEST(Barycentric, CentroidIsOneThird) {
    Vec3 x1(0,0,0), x2(3,0,0), x3(0,3,0);
    Vec3 centroid = (x1 + x2 + x3) / 3.0;
    auto b = triangle_plane_barycentric_coordinates(centroid, x1, x2, x3);
    EXPECT_NEAR(b[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(b[1], 1.0/3.0, 1e-12);
    EXPECT_NEAR(b[2], 1.0/3.0, 1e-12);
}

TEST(Barycentric, DegenerateTriangleReturnsZero) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(2,0,0); // collinear
    auto b = triangle_plane_barycentric_coordinates(Vec3(0.5, 1.0, 0.0), x1, x2, x3);
    EXPECT_DOUBLE_EQ(b[0], 0.0);
    EXPECT_DOUBLE_EQ(b[1], 0.0);
    EXPECT_DOUBLE_EQ(b[2], 0.0);
}

// ====================================================================
//  point_in_triangle_on_plane
// ====================================================================

TEST(PointInTriangle, InteriorPoint) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0);
    EXPECT_TRUE(point_in_triangle_on_plane(Vec3(0.2, 0.2, 0.0), x1, x2, x3));
}

TEST(PointInTriangle, ExteriorPoint) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0);
    EXPECT_FALSE(point_in_triangle_on_plane(Vec3(1.0, 1.0, 0.0), x1, x2, x3));
}

// ====================================================================
//  segment_segment_parameters_if_not_parallel
// ====================================================================

TEST(SegmentSegmentParams, CrossingSegments) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0.5,-1,0), x4(0.5,1,0);
    double s, u;
    bool ok = segment_segment_parameters_if_not_parallel(x1, x2, x3, x4, s, u);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(s, 0.5, 1e-12);
    EXPECT_NEAR(u, 0.5, 1e-12);
}

TEST(SegmentSegmentParams, ParallelReturnsFalse) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0), x4(1,1,0);
    double s, u;
    EXPECT_FALSE(segment_segment_parameters_if_not_parallel(x1, x2, x3, x4, s, u));
}

// ====================================================================
//  serialize / deserialize round-trip
// ====================================================================

TEST(SerializeState, RoundTrip) {
    DeformedState original;
    original.deformed_positions = {Vec3(1,2,3), Vec3(4,5,6), Vec3(7,8,9)};
    original.velocities = {Vec3(0.1,0.2,0.3), Vec3(0.4,0.5,0.6), Vec3(0.7,0.8,0.9)};

    std::string dir = "/tmp/ipc_serialize_test";
    std::filesystem::create_directories(dir);
    serialize_state(dir, 42, original);

    DeformedState loaded;
    ASSERT_TRUE(deserialize_state(dir, 42, loaded));

    ASSERT_EQ(loaded.deformed_positions.size(), original.deformed_positions.size());
    ASSERT_EQ(loaded.velocities.size(), original.velocities.size());
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR((loaded.deformed_positions[i] - original.deformed_positions[i]).norm(), 0.0, 1e-15);
        EXPECT_NEAR((loaded.velocities[i] - original.velocities[i]).norm(), 0.0, 1e-15);
    }
    std::filesystem::remove_all(dir);
}

// ====================================================================
//  BroadPhase::set_mesh_topology reuse matches fresh build
// ====================================================================

TEST(SetMeshTopology, CachedTopologyMatchesFreshBuild) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3::Zero());

    const int nv = static_cast<int>(state.deformed_positions.size());
    std::vector<Vec3> v(nv, Vec3(0, -1, 0));
    const double dhat = 0.1;

    // Build without pre-cached topology
    BroadPhase bp_fresh;
    bp_fresh.initialize(state.deformed_positions, v, ref_mesh, 1.0/30.0, dhat);

    // Build with pre-cached topology
    BroadPhase bp_cached;
    bp_cached.set_mesh_topology(ref_mesh, nv);
    EXPECT_TRUE(bp_cached.has_topology());
    bp_cached.initialize(state.deformed_positions, v, ref_mesh, 1.0/30.0, dhat);

    // Same pairs
    EXPECT_EQ(bp_fresh.nt_pairs().size(), bp_cached.nt_pairs().size());
    EXPECT_EQ(bp_fresh.ss_pairs().size(), bp_cached.ss_pairs().size());

    // Verify topology fields match
    const auto& fc = bp_fresh.cache();
    const auto& cc = bp_cached.cache();
    EXPECT_EQ(fc.edges.size(), cc.edges.size());
    for (size_t i = 0; i < fc.edges.size(); ++i) {
        EXPECT_EQ(fc.edges[i][0], cc.edges[i][0]);
        EXPECT_EQ(fc.edges[i][1], cc.edges[i][1]);
    }
}

// ====================================================================
//  max_abs_value_among_four_numbers
// ====================================================================

TEST(MaxAbsValue, AlwaysAtLeastOne) {
    EXPECT_GE(max_abs_value_among_four_numbers(0, 0, 0, 0), 1.0);
}

TEST(MaxAbsValue, PicksLargestAbsolute) {
    EXPECT_DOUBLE_EQ(max_abs_value_among_four_numbers(-5, 3, 2, 1), 5.0);
    EXPECT_DOUBLE_EQ(max_abs_value_among_four_numbers(0.1, 0.2, 0.3, 0.4), 1.0);
}

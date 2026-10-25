#include "dat_contact.h"
#include "gpu/resident_dat_math.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

std::array<Vec3, 4> zero_displacements() {
    return {Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
}

std::array<Vec3, 4> vertex_triangle() {
    return {Vec3(0, 0, 1), Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(0, 1, 0)};
}

dat_detail::PairGeometry vertex_triangle_geometry(const std::array<Vec3, 4>& x) {
    return dat_detail::make_vertex_triangle_geometry(x, ogc_detail::node_triangle_distance(x[0], x[1], x[2], x[3]));
}

} // namespace

TEST(DatContact, CollinearTriangleRetainsDegenerateMetadata) {
    const std::array<Vec3, 4> x = {Vec3(0.5, 1, 0.2), Vec3::Zero(), Vec3(1, 0, 0), Vec3(2, 0, 0)};
    const auto dr = node_triangle_distance(x[0], x[1], x[2], x[3]);
    EXPECT_EQ(dr.region, NodeTriangleRegion::DegenerateTriangle);
    EXPECT_TRUE(dr.closest_point.isApprox(Vec3(0.5, 0, 0), 1e-12));
    EXPECT_NEAR(dr.distance, std::sqrt(1.04), 1e-12);
    EXPECT_EQ(dr.normal, Vec3::Zero());
    EXPECT_TRUE(dat_detail::make_vertex_triangle_geometry(x, dr).valid_plane);
}

TEST(DatContact, AdaptivePlaneSharesRoomProportionalToApproach) {
    const auto geometry = vertex_triangle_geometry(vertex_triangle());
    ASSERT_TRUE(geometry.valid_plane);
    auto displacements = zero_displacements();
    displacements[0] = Vec3(0, 0, -0.8);
    for (int role = 1; role < 4; ++role) displacements[role] = Vec3(0, 0, 0.2);
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    EXPECT_NEAR(dat_detail::division_plane_offset(geometry, displacements), -0.8, 1e-12);
    for (double scale : scales) EXPECT_NEAR(scale, 0.8, 1e-12);
    auto reversed = zero_displacements();
    reversed[0].z() = -0.2;
    for (int role = 1; role < 4; ++role) reversed[role].z() = 0.8;
    EXPECT_NEAR(dat_detail::division_plane_offset(geometry, reversed), -0.2, 1e-12);
    EXPECT_NEAR(dat_detail::division_plane_offset(geometry, zero_displacements()), -0.5, 1e-12);
    EXPECT_NEAR(1.0 - 0.8 * scales[0] - 0.2 * scales[1], 0.2, 1e-12);
}

TEST(DatContact, TangentialAndSeparatingDisplacementsRemainUnrestricted) {
    const auto geometry = vertex_triangle_geometry(vertex_triangle());
    auto displacements = zero_displacements();
    displacements[0] = Vec3(100, 20, 0);
    displacements[1] = Vec3(0, 0, -100);
    displacements[2] = Vec3(-100, -20, 0);
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    for (double scale : scales) EXPECT_EQ(scale, 1.0);
}

TEST(DatContact, SingleApproachingSideStillLeavesInteriorPlane) {
    const auto geometry = vertex_triangle_geometry(vertex_triangle());
    auto displacements = zero_displacements();
    displacements[0] = Vec3(0, 0, -2);
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    EXPECT_NEAR(scales[0], 0.4 * (1.0 - 1e-8), 1e-12);
    const double plane = dat_detail::division_plane_offset(geometry, displacements);
    EXPECT_GT(plane, geometry.negative_support);
    EXPECT_LT(plane, geometry.positive_support);
    for (int role = 1; role < 4; ++role) EXPECT_EQ(scales[role], 1.0);
}

TEST(DatContact, EdgeEdgeConstrainsAllFourEndpointRoles) {
    const std::array<Vec3, 4> x = {Vec3(-1, 0, 1), Vec3(1, 0, 1), Vec3(0, -1, 0), Vec3(0, 1, 0)};
    SegmentSegmentDistanceResult distance{};
    distance.separation = Vec3(0, 0, 1);
    distance.distance = 1.0;
    const auto geometry = dat_detail::make_edge_edge_geometry(x, distance);
    ASSERT_TRUE(geometry.valid_plane);
    EXPECT_EQ(geometry.positive_count, 2);
    const std::array<Vec3, 4> displacements = {Vec3(0, 0, -0.6), Vec3(0, 0, -0.6), Vec3(0, 0, 0.4), Vec3(0, 0, 0.4)};
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    for (double scale : scales) EXPECT_NEAR(scale, 0.8, 1e-12);
}

TEST(DatContact, CPUAndGPUTruncationAgreeForAsymmetricAndStationarySides) {
    const std::array<std::array<double, 2>, 7> approaches{{{{.8,.2}}, {{.2,.8}}, {{2,0}}, {{0,2}}, {{.25,0}}, {{0,.25}}, {{0,0}}}};
    for (bool vertex_face : {true, false}) {
        const auto x = vertex_face ? vertex_triangle() : std::array<Vec3, 4>{Vec3(-1,0,1), Vec3(1,0,1), Vec3(0,-1,0), Vec3(0,1,0)};
        const auto cpu = vertex_face ? vertex_triangle_geometry(x) : dat_detail::make_edge_edge_geometry(x, segment_segment_distance(x[0], x[1], x[2], x[3], 1e-12, false));
        double positions[4][3];
        for (int role = 0; role < 4; ++role) for (int k = 0; k < 3; ++k) positions[role][k] = x[role][k];
        const auto gpu = ipc_gpu::resident::dat::geometry(positions, vertex_face, 2.0);
        ASSERT_TRUE(cpu.valid_plane);
        ASSERT_TRUE(gpu.valid);
        for (const auto& approach : approaches) {
            SCOPED_TRACE(::testing::Message() << "VT=" << vertex_face << " approaches=" << approach[0] << "," << approach[1]);
            auto proposed = zero_displacements();
            double q[4][3]{};
            for (int role = 0; role < 4; ++role) q[role][2] = proposed[role].z() = role < cpu.positive_count ? -approach[0] : approach[1];
            const double cpu_plane = dat_detail::division_plane_offset(cpu, proposed);
            const double gpu_plane = ipc_gpu::resident::dat::plane(gpu, q);
            EXPECT_NEAR(gpu_plane, cpu_plane, 1e-12);
            std::array<double, 4> scales{1,1,1,1};
            dat_detail::accumulate_pair_scales(cpu, proposed, .8, scales);
            for (int role = 0; role < 4; ++role) {
                const double gpu_scale = ipc_gpu::resident::dat::pair_scale(gpu, role, q[role], gpu_plane, .8);
                EXPECT_NEAR(gpu_scale, scales[role], 1e-12);
                // For unit-gap normal translation, both approaching sides
                // advance by the same fraction of the time to contact.
                const double own_approach = role < cpu.positive_count ? approach[0] : approach[1];
                const double expected = own_approach > 0 ? std::min(1.0, .8 / (approach[0] + approach[1])) : 1.0;
                EXPECT_NEAR(scales[role], expected, 1e-8);
            }
            EXPECT_GE(1.0 - approach[0] * scales[0] - approach[1] * scales[cpu.positive_count], .2 - 1e-12);
        }
    }
}

TEST(DatContact, InvalidPlaneUsesIsotropicHalfGapAndZeroGapFreezes) {
    dat_detail::PairGeometry geometry;
    geometry.distance = 0.5;
    const std::array<Vec3, 4> displacements = {Vec3(1, 0, 0), Vec3(0, 2, 0), Vec3::Zero(), Vec3(0, 0, 0.1)};
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    EXPECT_NEAR(scales[0], 0.2, 1e-15);
    EXPECT_NEAR(scales[1], 0.1, 1e-15);
    EXPECT_EQ(scales[2], 1.0);
    EXPECT_EQ(scales[3], 1.0);

    geometry.distance = 0.0;
    scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    EXPECT_EQ(scales, (std::array<double, 4>{0, 0, 1, 0}));
}

TEST(DatContact, UnresolvedProjectionGapRejectsPlaneAndScalesNeverIncrease) {
    auto x = vertex_triangle();
    x[0].z() = 1e-12;
    for (int role = 1; role < 4; ++role) x[role].x() *= 1e8;
    const auto geometry = vertex_triangle_geometry(x);
    ASSERT_GT(geometry.distance, 0.0);
    EXPECT_FALSE(geometry.valid_plane);
    auto displacements = zero_displacements();
    displacements[0] = Vec3::UnitX();
    std::array<double, 4> scales = {1, 0.25, 0.1, 0.5};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    EXPECT_NEAR(scales[0], 4e-13, 1e-25);
    EXPECT_EQ(scales[1], 0.25);
    EXPECT_EQ(scales[2], 0.1);
    EXPECT_EQ(scales[3], 0.5);
}

TEST(DatContact, RoundedEndpointCannotConsumeOneUlpPlaneGap) {
    auto x = vertex_triangle();
    const double height = 1e12;
    const double point_height = std::nextafter(height, std::numeric_limits<double>::infinity());
    const double gap = point_height - height;
    x[0].z() = point_height;
    for (int role = 1; role < 4; ++role) x[role].z() = height;
    const auto geometry = vertex_triangle_geometry(x);
    ASSERT_TRUE(geometry.valid_plane);
    auto displacements = zero_displacements();
    displacements[0].z() = -gap;
    for (int role = 1; role < 4; ++role) displacements[role].z() = 0.25 * gap;
    const double plane = dat_detail::division_plane_offset(geometry, displacements);
    std::array<double, 4> scales = {1, 1, 1, 1};
    dat_detail::accumulate_pair_scales(geometry, displacements, 0.8, scales);
    const Vec3 candidate = x[0] + scales[0] * displacements[0];
    ASSERT_LT(scales[0], 1.0);
    ASSERT_EQ(candidate.z(), height);
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 0, plane, x[0], candidate, 0.8));
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 0, plane, x[0], x[0], 0.8));
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 0, plane, x[0], x[0] + Vec3::UnitX(), 0.8));
}

TEST(DatContact, RoundedIsotropicDisplacementMustFitRealizedRadius) {
    const Vec3 base = Vec3::Constant(1e12);
    const double next = std::nextafter(base.x(), std::numeric_limits<double>::infinity());
    const double gap = next - base.x();
    dat_detail::PairGeometry geometry;
    geometry.distance = 2.0 * gap;
    const Vec3 candidate = base + Vec3(0.8 * gap, 0, 0);
    ASSERT_EQ(candidate.x(), next);
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 0, 0.0, base, candidate, 0.8));
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 0, 0.0, base, base, 0.8));
}

TEST(DatContact, RoundedGuardPreservesPlaneSideAndRejectsInvalidInputs) {
    dat_detail::PairGeometry geometry;
    geometry.valid_plane = true;
    geometry.normal = Vec3::UnitZ();
    geometry.positive_count = 1;
    geometry.projections = {1, 0, 0, 0};
    const Vec3 base = Vec3::Zero();
    // At zero initial clearance, tangent motion and motion into the assigned
    // halfspace remain admissible, while any crossing is rejected.
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 0, 1.0, base, Vec3::UnitX(), 0.8));
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 0, 1.0, base, Vec3::UnitZ(), 0.8));
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 0, 1.0, base, -Vec3::UnitZ(), 0.8));
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 1, 0.5, base, Vec3(0, 0, 0.5), 0.8));
    EXPECT_TRUE(dat_detail::accepts_rounded_position(geometry, 1, 0.5, base, Vec3(0, 0, 0.25), 0.8));
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 0, std::numeric_limits<double>::quiet_NaN(), base, base, 0.8));
    EXPECT_FALSE(dat_detail::accepts_rounded_position(geometry, 0, 0.5, base, Vec3::Constant(std::numeric_limits<double>::infinity()), 0.8));
}

#include "parallel_helper.h"
#include "physics.h"
#include "quaternion_math.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

Vec4 axis_angle_quaternion(const Vec3& axis_input, double angle) {
    const Vec3 axis = axis_input.normalized();
    const double half_angle = 0.5 * angle;
    const double sin_half_angle = std::sin(half_angle);
    return Vec4(
        std::cos(half_angle),
        sin_half_angle * axis.x(),
        sin_half_angle * axis.y(),
        sin_half_angle * axis.z());
}

void build_contact_adj_pair_scan_reference(const BroadPhase::Cache& cache, int nv,
                                           std::vector<std::vector<int>>& out) {
    out.assign(nv, {});
    auto add_clique = [&](const int verts[4]) {
        for (int a = 0; a < 4; ++a) {
            if (verts[a] < 0 || verts[a] >= nv) continue;
            for (int b = a + 1; b < 4; ++b) {
                if (verts[b] < 0 || verts[b] >= nv) continue;
                out[verts[a]].push_back(verts[b]);
                out[verts[b]].push_back(verts[a]);
            }
        }
    };
    for (const auto& p : cache.nt_pairs) {
        const int verts[4] = {p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]};
        add_clique(verts);
    }
    for (const auto& p : cache.ss_pairs) add_clique(p.v);
    for (auto& row : out) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
}

void populate_contact_incidence(BroadPhase::Cache& cache, int nv) {
    cache.vertex_nt.assign(nv, {});
    cache.vertex_ss.assign(nv, {});
    for (std::size_t i = 0; i < cache.nt_pairs.size(); ++i) {
        const auto& p = cache.nt_pairs[i];
        const int verts[4] = {p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]};
        for (int role = 0; role < 4; ++role)
            if (verts[role] >= 0 && verts[role] < nv)
                cache.vertex_nt[verts[role]].push_back({i, role});
    }
    for (std::size_t i = 0; i < cache.ss_pairs.size(); ++i) {
        const auto& p = cache.ss_pairs[i];
        for (int role = 0; role < 4; ++role)
            if (p.v[role] >= 0 && p.v[role] < nv)
                cache.vertex_ss[p.v[role]].push_back({i, role});
    }
}

} // namespace

TEST(ParallelHelper, ContactAdjacencyMatchesPairScanExactly) {
    constexpr int nv = 6;
    BroadPhase::Cache cache;

    NodeTrianglePair nt{};
    nt.node = 0;
    nt.tri_v[0] = 1;
    nt.tri_v[1] = 2;
    nt.tri_v[2] = 3;
    cache.nt_pairs.push_back(nt);

    SegmentSegmentPair ss{};
    ss.v[0] = 1;
    ss.v[1] = 2;
    ss.v[2] = 4;
    ss.v[3] = 5;
    cache.ss_pairs.push_back(ss);

    populate_contact_incidence(cache, nv);

    std::vector<std::vector<int>> expected;
    build_contact_adj_pair_scan_reference(cache, nv, expected);

    std::vector<std::vector<int>> actual{{99}};
    build_contact_adj(cache, nv, actual);
    EXPECT_EQ(actual, expected);

    BroadPhase::Cache empty;
    empty.vertex_nt.resize(nv);
    empty.vertex_ss.resize(nv);
    build_contact_adj(empty, nv, actual);
    EXPECT_EQ(actual, std::vector<std::vector<int>>(nv));
}

TEST(ParallelHelper, RigidContactAdjacencyFiltersInternalPairsAndColorsBodies) {
    BroadPhase::Cache cache;
    const auto add_nt = [&](int node, int a, int b, int c) {
        NodeTrianglePair pair;
        pair.node = node;
        pair.tri_v[0] = a;
        pair.tri_v[1] = b;
        pair.tri_v[2] = c;
        cache.nt_pairs.push_back(pair);
    };
    const auto add_ss = [&](int a0, int a1, int b0, int b1) {
        SegmentSegmentPair pair;
        pair.v[0] = a0;
        pair.v[1] = a1;
        pair.v[2] = b0;
        pair.v[3] = b1;
        cache.ss_pairs.push_back(pair);
    };

    add_nt(0, 1, 2, 3);
    add_nt(0, 4, 5, 6);
    add_nt(4, 8, 9, 10);
    add_nt(16, 12, 13, 14);
    add_nt(16, 17, 18, 19);
    add_ss(8, 9, 10, 11);
    add_ss(0, 1, 4, 5);
    add_ss(16, 17, 12, 13);

    std::vector<int> node_to_rb(20, -1);
    for (int node = 0; node < 16; ++node)
        node_to_rb[node] = node / 4;
    std::vector<std::vector<int>> body_nt_pair_indices;
    std::vector<std::vector<int>> body_ss_pair_indices;
    std::vector<std::vector<int>> adjacency;
    build_rb_contact_adj(cache, node_to_rb, 4, body_nt_pair_indices, body_ss_pair_indices, adjacency);

    EXPECT_EQ(adjacency, (std::vector<std::vector<int>>{{1}, {0, 2}, {1}, {}}));
    EXPECT_EQ(body_nt_pair_indices, (std::vector<std::vector<int>>{{1}, {1, 2}, {2}, {3}}));
    EXPECT_EQ(body_ss_pair_indices, (std::vector<std::vector<int>>{{1}, {1}, {}, {2}}));
    EXPECT_EQ(cache.nt_pairs.size(), 5);
    EXPECT_EQ(cache.ss_pairs.size(), 3);

    std::vector<std::vector<int>> groups;
    greedy_color_conflict_graph(adjacency, groups);
    EXPECT_EQ(groups, (std::vector<std::vector<int>>{{0, 2, 3}, {1}}));
}

TEST(GreedyColorConflictGraph, DeterministicColoringAndScratchReuse) {
    const std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 2},
        {0, 1, 3},
        {2},
        {},
    };

    std::vector<std::vector<int>> groups{{99}, {98}, {97}};
    greedy_color_conflict_graph(graph, groups);
    EXPECT_EQ(groups, (std::vector<std::vector<int>>{{0, 3, 4}, {1}, {2}}));

    greedy_color_conflict_graph({}, groups);
    EXPECT_TRUE(groups.empty());
}

// ── spherical_cap_node_aabb tests ────────────────────────────────────────────

namespace {

// For testing purposes, increasing the sampling density to 250000 for both polar and azimuth with sampling_tolerance of 1.0e-9.
// This is just to ensure the code is working properly but extremely slow, almost 30 mins.
// For pipeline testing, reduce the sampling density to 1000 and sampling_tolerance to 1.0e-3
AABB brute_force_spherical_cap_aabb(const Vec3& x_com, const Vec4& q, const Vec3& X, double theta_bound, int polar_samples = 1000, int azimuth_samples = 1000) {
    const Vec4 q_current = quaternion_normalize(q);
    const Vec3 world_space_offset = quaternion_rotate(q_current, X);
    const double radius = world_space_offset.norm();
    if (radius == 0.0)
        return AABB(x_com, x_com);

    const double cap_extent = std::min(theta_bound, M_PI);
    const Vec3 direction = world_space_offset / radius;
    if (cap_extent == 0.0) {
        const Vec3 x = x_com + world_space_offset;
        return AABB(x, x);
    }

    const Vec3 tangent_0 = direction.unitOrthogonal();
    const Vec3 tangent_1 = direction.cross(tangent_0);

    AABB box;
    box.expand(x_com + world_space_offset);
    for (int polar = 1; polar <= polar_samples; ++polar) {
        const double beta = cap_extent * polar / polar_samples;
        const double cos_beta = std::cos(beta);
        const double sin_beta = std::sin(beta);
        for (int azimuth = 0; azimuth < azimuth_samples; ++azimuth) {
            const double phi = 2.0 * M_PI * azimuth / azimuth_samples;
            const Vec3 tangent_direction = std::cos(phi) * tangent_0 + std::sin(phi) * tangent_1;
            const Vec3 sampled_direction = cos_beta * direction + sin_beta * tangent_direction;
            box.expand(x_com + radius * sampled_direction);
        }
    }
    return box;
}

void check_spherical_cap_aabb(const Vec3& x_com, const Vec4& q, const Vec3& X, double theta_bound, double sampling_tolerance = 1.0e-3) {
    const AABB result = spherical_cap_node_aabb(x_com, q, X, theta_bound);
    const AABB reference = brute_force_spherical_cap_aabb(x_com, q, X, theta_bound);

    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        // The analytic box must contain every densely sampled point.
        EXPECT_LE(result.min[coordinate], reference.min[coordinate] + 1.0e-12);
        EXPECT_GE(result.max[coordinate], reference.max[coordinate] - 1.0e-12);
        EXPECT_NEAR(result.min[coordinate], reference.min[coordinate], sampling_tolerance);
        EXPECT_NEAR(result.max[coordinate], reference.max[coordinate], sampling_tolerance);
    }
}

} // namespace

TEST(SphericalCapNodeAABB, BoundsQuarterTurnSphericalCapExactly) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X(2.0, 0.0, 0.0);
    check_spherical_cap_aabb(x_com, q, X, 0.5 * M_PI);
    const AABB box = spherical_cap_node_aabb(x_com, q, X, 0.5 * M_PI);
    EXPECT_TRUE(box.min.isApprox(Vec3(1.0, -4.0, 1.0), 1.0e-14));
    EXPECT_TRUE(box.max.isApprox(Vec3(3.0, 0.0, 5.0), 1.0e-14));
}

TEST(SphericalCapNodeAABB, ContainsMotionAboutDifferentAxes) {
    const Vec3 x_com(0.3, -0.2, 0.8);
    const Vec4 q = axis_angle_quaternion(Vec3(1.0, -2.0, 0.5), 0.7);
    const Vec3 X(1.2, -0.6, 0.4);
    const double theta_bound = 0.9;
    const AABB box = spherical_cap_node_aabb(x_com, q, X, theta_bound);
    const Vec3 offset = quaternion_rotate(q, X);
    for (const Vec3& axis : {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(-0.3, 0.9, 0.2).normalized()}) {
        const Vec3 rotated = x_com + quaternion_rotate(axis_angle_quaternion(axis, theta_bound), offset);
        EXPECT_TRUE((rotated.array() >= box.min.array() - 1.0e-14).all());
        EXPECT_TRUE((rotated.array() <= box.max.array() + 1.0e-14).all());
    }
}

TEST(SphericalCapNodeAABB, WideExtentAndNormalizedOrientation) {
    const Vec3 x_com = Vec3::Zero();
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X = Vec3::UnitX();
    check_spherical_cap_aabb(x_com, q, X, 1.5 * M_PI);
    check_spherical_cap_aabb(x_com, 3.0 * q, X, 1.5 * M_PI);
    const AABB full_270 = spherical_cap_node_aabb(x_com, q, X, 1.5 * M_PI);
    const AABB scaled = spherical_cap_node_aabb(x_com, 3.0 * q, X, 1.5 * M_PI);
    EXPECT_TRUE(full_270.min.isApprox(Vec3(-1.0, -1.0, -1.0), 1.0e-14));
    EXPECT_TRUE(full_270.max.isApprox(Vec3(1.0, 1.0, 1.0), 1.0e-14));
    EXPECT_TRUE(scaled.min.isApprox(full_270.min, 1.0e-14));
    EXPECT_TRUE(scaled.max.isApprox(full_270.max, 1.0e-14));
}

TEST(SphericalCapNodeAABB, HalfTurnExtentBoundsFullSphere) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X(2.0, 0.0, 1.0);
    check_spherical_cap_aabb(x_com, q, X, M_PI);
    const AABB box = spherical_cap_node_aabb(x_com, q, X, M_PI);
    const Vec3 radius = Vec3::Constant(X.norm());
    EXPECT_TRUE(box.min.isApprox(x_com - radius, 1.0e-14));
    EXPECT_TRUE(box.max.isApprox(x_com + radius, 1.0e-14));
}

TEST(SphericalCapNodeAABB, MatchesBruteForceForArbitraryAxis) {
    const Vec3 x_com(0.4, -0.7, 1.1);
    const Vec4 q = axis_angle_quaternion(Vec3(1.0, 2.0, -1.0), 0.8);
    const Vec3 X(1.2, -0.4, 0.7);
    check_spherical_cap_aabb(x_com, q, X, 1.1);
}

TEST(SphericalCapNodeAABB, MatchesBruteForceForWideCap) {
    const Vec3 x_com(-0.2, 0.5, -0.8);
    const Vec4 q = axis_angle_quaternion(Vec3(-1.0, 0.5, 2.0), 0.4);
    const Vec3 X(0.7, 1.3, -0.2);
    check_spherical_cap_aabb(x_com, q, X, 2.2);
}

TEST(SphericalCapNodeAABB, DegeneratePointFullTurnAndInvalidQuaternions) {
    const Vec3 x_com(0.2, -0.4, 0.7);
    const Vec4 q = axis_angle_quaternion(Vec3::UnitY(), 0.6);
    const Vec3 X(0.8, -0.3, 1.1);
    const Vec3 x = x_com + quaternion_rotate(q, X);

    check_spherical_cap_aabb(x_com, q, X, 0.0);
    const AABB stationary = spherical_cap_node_aabb(x_com, q, X, 0.0);
    const AABB full_turn = spherical_cap_node_aabb(x_com, q, X, 2.0 * M_PI);
    const Vec3 full_turn_radius = Vec3::Constant(X.norm());
    EXPECT_TRUE(stationary.min.isApprox(x, 1.0e-14));
    EXPECT_TRUE(stationary.max.isApprox(x, 1.0e-14));
    EXPECT_TRUE(full_turn.min.isApprox(x_com - full_turn_radius, 1.0e-14));
    EXPECT_TRUE(full_turn.max.isApprox(x_com + full_turn_radius, 1.0e-14));

    const Vec3 zero_X = Vec3::Zero();
    const AABB zero_radius = spherical_cap_node_aabb(x_com, q, zero_X, 2.0 * M_PI);
    EXPECT_TRUE(zero_radius.min.isApprox(x_com, 1.0e-14));
    EXPECT_TRUE(zero_radius.max.isApprox(x_com, 1.0e-14));

    EXPECT_THROW(spherical_cap_node_aabb(x_com, Vec4::Zero(), X, 0.0), std::invalid_argument);
}

TEST(RigidBodyBlueBoxes, CombinesSphericalCapAndCOMTranslation) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const std::vector<Vec3> material_positions = {Vec3(2.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0)};
    const std::vector<Vec3> positions = {x_com + material_positions[0], x_com + material_positions[1]};

    RefMesh ref_mesh;
    ref_mesh.rb_nodes = {{0, 1}};
    ref_mesh.ref_positions = {material_positions};
    std::vector<AABB> boxes(positions.size());
    build_blue_boxes_rb({x_com}, {identity}, {0.5 * M_PI}, {0.24}, ref_mesh, boxes);

    ASSERT_EQ(boxes.size(), positions.size());
    constexpr double com_padding = 0.24;
    const double first_radius = material_positions[0].norm();
    const Vec3 expected_first_min = x_com + Vec3(-1.0, -first_radius, -2.0) - Vec3::Constant(com_padding);
    const Vec3 expected_first_max = x_com + Vec3::Constant(first_radius) + Vec3::Constant(com_padding);
    const Vec3 expected_second_min = x_com + Vec3(-1.0, 0.0, -1.0) - Vec3::Constant(com_padding);
    const Vec3 expected_second_max = x_com + Vec3(1.0, 1.0, 1.0) + Vec3::Constant(com_padding);
    EXPECT_TRUE(boxes[0].min.isApprox(expected_first_min, 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(expected_first_max, 1.0e-14));
    EXPECT_TRUE(boxes[1].min.isApprox(expected_second_min, 1.0e-14));
    EXPECT_TRUE(boxes[1].max.isApprox(expected_second_max, 1.0e-14));
}

TEST(RigidBodyBlueBoxes, StationaryRotationReducesToCOMBox) {
    const Vec3 x_com(-0.4, 0.7, 1.2);
    const Vec4 q = axis_angle_quaternion(Vec3(1.0, -2.0, 0.5), 0.8);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec3 X(0.3, -0.6, 1.1);
    const Vec3 x = x_com + quaternion_rotate(q, X);

    RefMesh ref_mesh;
    ref_mesh.rb_nodes = {{0}};
    ref_mesh.ref_positions = {{X}};
    std::vector<AABB> boxes(1);
    build_blue_boxes_rb({x_com}, {q}, {0.0}, {0.4}, ref_mesh, boxes);

    ASSERT_EQ(boxes.size(), 1);
    EXPECT_TRUE(boxes[0].min.isApprox(x - Vec3::Constant(0.4), 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(x + Vec3::Constant(0.4), 1.0e-14));
}

TEST(RigidBodyBlueBoxes, UsesCOMBound) {
    const Vec3 x_com = Vec3::Zero();
    const Vec3 X = Vec3::UnitX();
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    RefMesh ref_mesh;
    ref_mesh.rb_nodes = {{0}};
    ref_mesh.ref_positions = {{X}};
    std::vector<AABB> boxes(1);
    build_blue_boxes_rb({x_com}, {identity}, {0.0}, {0.3}, ref_mesh, boxes);

    ASSERT_EQ(boxes.size(), 1);
    EXPECT_TRUE(boxes[0].min.isApprox(X - Vec3::Constant(0.3), 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(X + Vec3::Constant(0.3), 1.0e-14));
}

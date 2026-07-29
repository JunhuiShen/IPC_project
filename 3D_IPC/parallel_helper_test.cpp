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

// ── arc_node_aabb tests ──────────────────────────────────────────────────────

namespace {

AABB brute_force(
    const Vec3& x_com, const Vec4& q, const Vec3& X,
    const Vec4& q_rel, int samples = 400000) {
    const Vec4 q_current = quaternion_normalize(q);
    const Vec4 q_relative = quaternion_normalize(q_rel);
    const Vec3 x = x_com + quaternion_rotate(q_current, X);
    const Vec3 vector_part = q_relative.tail<3>();
    const double sin_half_angle = vector_part.norm();

    if (sin_half_angle < 1.0e-12) {
        if (q_relative[0] >= 0.0)
            return AABB(x, x);

        const Vec3 radius = Vec3::Constant(X.norm());
        return AABB(x_com - radius, x_com + radius);
    }

    const Vec3 axis = vector_part / sin_half_angle;
    const double angular_extent =
        2.0 * std::atan2(sin_half_angle, q_relative[0]);

    AABB box;
    for (int sample = 0; sample <= samples; ++sample) {
        const double alpha =
            -1.0 + 2.0 * static_cast<double>(sample) / samples;
        const Vec4 delta =
            axis_angle_quaternion(axis, alpha * angular_extent);
        const Vec4 sampled_q = quaternion_multiply(delta, q_current);
        box.expand(x_com + quaternion_rotate(sampled_q, X));
    }
    return box;
}

void check(
    const Vec3& x_com, const Vec4& q, const Vec3& X,
    const Vec4& q_rel, double tolerance = 1.0e-9) {
    const AABB result = arc_node_aabb(x_com, q, X, q_rel);
    const AABB reference = brute_force(x_com, q, X, q_rel);

    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        // The analytic box must contain every densely sampled point.
        EXPECT_LE(result.min[coordinate], reference.min[coordinate] + 1.0e-12);
        EXPECT_GE(result.max[coordinate], reference.max[coordinate] - 1.0e-12);
        EXPECT_NEAR(result.min[coordinate], reference.min[coordinate], tolerance);
        EXPECT_NEAR(result.max[coordinate], reference.max[coordinate], tolerance);
    }
}

} // namespace

TEST(ArcNodeAABB, BoundsSymmetricQuarterTurnExactly) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X(2.0, 0.0, 1.0);
    const Vec4 q_rel =
        axis_angle_quaternion(Vec3::UnitZ(), 0.5 * M_PI);

    check(x_com, q, X, q_rel);
    const AABB box = arc_node_aabb(x_com, q, X, q_rel);
    EXPECT_TRUE(box.min.isApprox(Vec3(1.0, -4.0, 4.0), 1.0e-14));
    EXPECT_TRUE(box.max.isApprox(Vec3(3.0, 0.0, 4.0), 1.0e-14));
}

TEST(ArcNodeAABB, PreservesFullArcSignAndNormalizesQuaternionInputs) {
    const Vec3 x_com = Vec3::Zero();
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X = Vec3::UnitX();
    const Vec4 q_rel =
        axis_angle_quaternion(Vec3::UnitZ(), 1.5 * M_PI);

    check(x_com, q, X, q_rel);
    check(x_com, q, X, -q_rel);
    check(x_com, 3.0 * q, X, 5.0 * q_rel);

    const AABB full_270 = arc_node_aabb(x_com, q, X, q_rel);
    const AABB complementary_minus_90 =
        arc_node_aabb(x_com, q, X, -q_rel);
    const AABB scaled =
        arc_node_aabb(x_com, 3.0 * q, X, 5.0 * q_rel);

    EXPECT_TRUE(full_270.min.isApprox(Vec3(-1.0, -1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(full_270.max.isApprox(Vec3(1.0, 1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(complementary_minus_90.min.isApprox(
        Vec3(0.0, -1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(complementary_minus_90.max.isApprox(
        Vec3(1.0, 1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(scaled.min.isApprox(full_270.min, 1.0e-14));
    EXPECT_TRUE(scaled.max.isApprox(full_270.max, 1.0e-14));
}

TEST(ArcNodeAABB, HalfTurnExtentBoundsFullCircle) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 X(2.0, 0.0, 1.0);
    const Vec4 q_rel =
        axis_angle_quaternion(Vec3::UnitZ(), M_PI);

    check(x_com, q, X, q_rel);
    const AABB box = arc_node_aabb(x_com, q, X, q_rel);
    EXPECT_TRUE(box.min.isApprox(Vec3(-1.0, -4.0, 4.0), 1.0e-14));
    EXPECT_TRUE(box.max.isApprox(Vec3(3.0, 0.0, 4.0), 1.0e-14));
}

TEST(ArcNodeAABB, MatchesBruteForceForArbitraryAxis) {
    const Vec3 x_com(0.4, -0.7, 1.1);
    const Vec4 q =
        axis_angle_quaternion(Vec3(1.0, 2.0, -1.0), 0.8);
    const Vec3 X(1.2, -0.4, 0.7);
    const Vec4 q_rel =
        axis_angle_quaternion(Vec3(-0.3, 0.9, 0.2), 1.1);

    check(x_com, q, X, q_rel);
}

TEST(ArcNodeAABB, DegenerateArcFullTurnFallbackAndInvalidQuaternions) {
    const Vec3 x_com(0.2, -0.4, 0.7);
    const Vec4 q =
        axis_angle_quaternion(Vec3::UnitY(), 0.6);
    const Vec3 X(0.8, -0.3, 1.1);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec3 x = x_com + quaternion_rotate(q, X);

    check(x_com, q, X, identity);
    const AABB stationary = arc_node_aabb(x_com, q, X, identity);
    const AABB antipodal = arc_node_aabb(x_com, q, X, -identity);
    const Vec3 full_turn_radius = Vec3::Constant(X.norm());
    EXPECT_TRUE(stationary.min.isApprox(x, 1.0e-14));
    EXPECT_TRUE(stationary.max.isApprox(x, 1.0e-14));
    EXPECT_TRUE(antipodal.min.isApprox(
        x_com - full_turn_radius, 1.0e-14));
    EXPECT_TRUE(antipodal.max.isApprox(
        x_com + full_turn_radius, 1.0e-14));

    EXPECT_THROW(
        arc_node_aabb(x_com, Vec4::Zero(), X, identity),
        std::invalid_argument);
    EXPECT_THROW(
        arc_node_aabb(x_com, q, X, Vec4::Zero()),
        std::invalid_argument);
}

TEST(RigidBodyBlueBoxes, CombinesRotationArcAndCOMTranslation) {
    const Vec3 x_com(1.0, -2.0, 3.0);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const std::vector<Vec3> material_positions = {
        Vec3(2.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
    };
    const std::vector<Vec3> positions = {
        x_com + material_positions[0],
        x_com + material_positions[1],
    };

    RefMesh ref_mesh;
    ref_mesh.rb_nodes = {{0, 1}};
    ref_mesh.ref_positions = {material_positions};
    SimParams params = SimParams::zeros();
    params.node_box_min = 0.1;
    params.node_box_max = 1.0;

    const std::vector<AABB> boxes = build_blue_boxes_rb(positions, {x_com}, {identity}, {axis_angle_quaternion(Vec3::UnitZ(), 0.5 * M_PI)}, {0.2}, params, ref_mesh);

    ASSERT_EQ(boxes.size(), positions.size());
    EXPECT_TRUE(boxes[0].min.isApprox(Vec3(0.76, -4.24, 3.76), 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(Vec3(3.24, 0.24, 4.24), 1.0e-14));
    EXPECT_TRUE(boxes[1].min.isApprox(Vec3(-0.24, -2.24, 2.76), 1.0e-14));
    EXPECT_TRUE(boxes[1].max.isApprox(Vec3(2.24, -0.76, 3.24), 1.0e-14));
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
    SimParams params = SimParams::zeros();
    params.node_box_min = 0.4;
    params.node_box_max = 0.8;

    const std::vector<AABB> boxes = build_blue_boxes_rb({x}, {x_com}, {q}, {identity}, {0.0}, params, ref_mesh);

    ASSERT_EQ(boxes.size(), 1);
    EXPECT_TRUE(boxes[0].min.isApprox(x - Vec3::Constant(0.4), 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(x + Vec3::Constant(0.4), 1.0e-14));
}

TEST(RigidBodyBlueBoxes, COMBoundClampsToMaximum) {
    const Vec3 x_com = Vec3::Zero();
    const Vec3 X = Vec3::UnitX();
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    RefMesh ref_mesh;
    ref_mesh.rb_nodes = {{0}};
    ref_mesh.ref_positions = {{X}};
    SimParams params = SimParams::zeros();
    params.node_box_min = 0.1;
    params.node_box_max = 0.3;

    const std::vector<AABB> boxes = build_blue_boxes_rb({X}, {x_com}, {identity}, {identity}, {2.0}, params, ref_mesh);

    ASSERT_EQ(boxes.size(), 1);
    EXPECT_TRUE(boxes[0].min.isApprox(X - Vec3::Constant(0.3), 1.0e-14));
    EXPECT_TRUE(boxes[0].max.isApprox(X + Vec3::Constant(0.3), 1.0e-14));
}

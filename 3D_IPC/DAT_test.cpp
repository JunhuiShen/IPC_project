#include "DAT.h"
#include "broad_phase.h"
#include "ccd.h"
#include "mesh_utils.h"
#include "physics.h"
#include "segment_segment_distance.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double gamma = 0.9;
constexpr double query_radius = 1.0;
constexpr double tolerance = 1e-12;

struct Scene {
    RefMesh mesh;
    std::vector<Vec3> positions;
    std::vector<Vec3> update;
};

Scene point_triangle(double gap = 0.4) {
    Scene scene;
    scene.positions = {Vec3(0, 0, gap), Vec3(-3, -3, 0),
        Vec3(3, -3, 0), Vec3(0, 3, 0)};
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {1, 2, 3};
    return scene;
}

Scene crossed_edges(double gap = 0.4) {
    Scene scene;
    // Only edges (0,1) and (3,4) are within the query radius. The
    // off-plane apices keep every cross-triangle VT and other EE pair far.
    scene.positions = {Vec3(-10, 0, 0), Vec3(10, 0, 0),
        Vec3(0, -10, -10), Vec3(0, -10, gap),
        Vec3(0, 10, gap), Vec3(10, 0, 10 + gap)};
    scene.update = {Vec3(0, 0, 0.4), Vec3(0, 0, 0.4),
        Vec3(0, 0, 0.4), Vec3(0, 0, -0.1),
        Vec3(0, 0, -0.1), Vec3(0, 0, -0.1)};
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {0, 1, 2, 3, 4, 5};
    return scene;
}

Scene point_two_triangles() {
    Scene scene = point_triangle();
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.positions[vertex].z() = 0.2;
    }
    scene.positions.insert(scene.positions.end(), {
        Vec3(-3, -3, 0.1), Vec3(3, -3, 0.1), Vec3(0, 3, 0.1)});
    scene.mesh.tris.insert(scene.mesh.tris.end(), {4, 5, 6});
    scene.mesh.num_positions = scene.positions.size();
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.update[0] = Vec3(0, 0, -0.4);
    return scene;
}

// Original binary64 geometry captured at frame 126 of the README Example 1
// DAT run. The edges are strictly separated, but subtracting rounded fast
// closest points produces a direction which is not a supporting plane.
Scene benchmark_edge_plane_case() {
    Scene scene;
    scene.positions = {
        Vec3(-1.2499964352680171, .54358597764887895, -.43915608458931166),
        Vec3(-1.2722323852803259, .53028835202992686, -.36137318480164726),
        Vec3(-1.2499964600268660, .54358615217010631, -.43915583489559568),
        Vec3(-1.2499995411271900, .50375014954939679, -.47291205323261482)};
    // Independently computed from the exact finite-edge solution, oriented A->B.
    const Vec3 normal(.95038202192690158, -.20115535389100869, .23729841128654272);
    scene.positions.push_back(scene.positions[0] - .03 * normal);
    scene.positions.push_back(scene.positions[2] + .03 * normal);
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {0, 1, 4, 2, 3, 5};
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    return scene;
}

double benchmark_edge_signed_gap(const std::vector<Vec3>& positions, int offset = 0) {
    const Vec3 normal(.95038202192690158, -.20115535389100869, .23729841128654272);
    const Vec3& origin = positions[offset];
    const double left = std::max((positions[offset] - origin).dot(normal),
        (positions[offset + 1] - origin).dot(normal));
    const double right = std::min((positions[offset + 2] - origin).dot(normal),
        (positions[offset + 3] - origin).dot(normal));
    return right - left;
}

std::vector<Vec3> filter(const Scene& scene, double padding = 1.2) {
    return planar_DAT(gamma, query_radius, scene.mesh, scene.update,
        scene.positions, padding);
}

void expect_vector_near(const Vec3& actual, const Vec3& expected) {
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(actual[axis], expected[axis], tolerance);
    }
}

void expect_vectors_identical(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t vertex = 0; vertex < actual.size(); ++vertex) {
        SCOPED_TRACE(vertex);
        EXPECT_EQ(std::memcmp(actual[vertex].data(), expected[vertex].data(),
            3 * sizeof(double)), 0);
    }
}

class PlanarDAT : public testing::Test {
protected:
    int previous_threads = omp_get_max_threads();
    int previous_dynamic = omp_get_dynamic();

    void SetUp() override {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }

    void TearDown() override {
        omp_set_num_threads(previous_threads);
        omp_set_dynamic(previous_dynamic);
    }
};

TEST_F(PlanarDAT, CapsEachUnconstrainedUpdateWithoutGlobalDamping) {
    Scene scene;
    scene.positions = {Vec3::Zero(), Vec3(1, 0, 0), Vec3(2, 0, 0)};
    scene.update = {Vec3(3, 4, 0), Vec3(0.1, -0.2, 0.1), Vec3::Zero()};
    scene.mesh.num_positions = scene.positions.size();

    const auto result = filter(scene);
    ASSERT_EQ(result.size(), scene.update.size());
    expect_vector_near(result[0], Vec3(0.27, 0.36, 0));
    expect_vector_near(result[1], scene.update[1]);
    expect_vector_near(result[2], Vec3::Zero());
    EXPECT_NEAR(result[0].norm(), 0.5 * gamma * query_radius, tolerance);
}

TEST_F(PlanarDAT, CapturedSeparatedEdgePairWithNoMotionDoesNotAbort) {
    const auto scene = benchmark_edge_plane_case();
    const std::array<Vec3, 4> edges{{scene.positions[0], scene.positions[1],
        scene.positions[2], scene.positions[3]}};
    ASSERT_GT(compare_contact_distance_exact(edges, false, 0), 0);
    ASSERT_GT(benchmark_edge_signed_gap(scene.positions), 5e-10);
    ASSERT_LT(benchmark_edge_signed_gap(scene.positions), 7e-10);
    const auto accepted = planar_DAT(.9, .0075, scene.mesh, scene.update, scene.positions);
    expect_vectors_identical(accepted, scene.update);
}

TEST_F(PlanarDAT, CapturedEdgePairApproachStaysOnItsOriginalPlaneSidesAcrossThreads) {
    const auto unit = benchmark_edge_plane_case();
    const Vec3 normal(.95038202192690158, -.20115535389100869, .23729841128654272);
    Scene scene;
    constexpr int copies = 64; // Exercise the parallel pair and node phases.
    for (int copy = 0; copy < copies; ++copy) {
        const int offset = static_cast<int>(scene.positions.size());
        for (const Vec3& p : unit.positions)
            scene.positions.push_back(p + Vec3(0, 4.0 * copy, 0));
        for (int index : unit.mesh.tris) scene.mesh.tris.push_back(offset + index);
        for (int node = 0; node < 6; ++node)
            scene.update.push_back(node < 2 ? Vec3(1e-4 * normal)
                : node < 4 ? Vec3(-1e-4 * normal) : Vec3::Zero());
    }
    scene.mesh.num_positions = scene.positions.size();
    omp_set_num_threads(1);
    const auto serial = planar_DAT(.9, .0075, scene.mesh, scene.update, scene.positions);
    for (int threads : {2, 4, 8}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        const auto parallel = planar_DAT(.9, .0075, scene.mesh, scene.update, scene.positions);
        expect_vectors_identical(parallel, serial);
        auto endpoints = scene.positions;
        for (std::size_t node = 0; node < endpoints.size(); ++node) endpoints[node] += parallel[node];
        for (int copy = 0; copy < copies; ++copy) {
            SCOPED_TRACE(copy);
            const int offset = 6 * copy;
            EXPECT_GT(benchmark_edge_signed_gap(endpoints, offset), 0);
            EXPECT_LT(benchmark_edge_signed_gap(endpoints, offset),
                benchmark_edge_signed_gap(scene.positions, offset));
            EXPECT_GT(parallel[offset].norm(), 0);
            EXPECT_LT(parallel[offset].norm(), 1e-7);
            EXPECT_TRUE(parallel[offset + 4].isZero(0));
            EXPECT_TRUE(parallel[offset + 5].isZero(0));
        }
    }
}

TEST_F(PlanarDAT, CapturedEdgePairOutsideTheQueryRadiusKeepsItsSmallUpdate) {
    auto scene = benchmark_edge_plane_case();
    const Vec3 normal(.95038202192690158, -.20115535389100869, .23729841128654272);
    for (int node = 0; node < 4; ++node)
        scene.update[node] = (node < 2 ? 1e-12 : -1e-12) * normal;
    const auto accepted = planar_DAT(.9, 1e-11, scene.mesh, scene.update, scene.positions);
    expect_vectors_identical(accepted, scene.update);
}

TEST_F(PlanarDAT, GeneralFilterUsesTheSameRepairForCapturedRigidEdgePairs) {
    auto scene = benchmark_edge_plane_case();
    scene.mesh.rb_nodes = {{0, 1, 4}, {2, 3, 5}};
    scene.mesh.ref_positions = {{scene.positions[0], scene.positions[1], scene.positions[4]},
        {scene.positions[2], scene.positions[3], scene.positions[5]}};
    scene.mesh.node_to_rb = {0, 0, 1, 1, 0, 1};
    const std::vector<Vec3> zeros(2, Vec3::Zero());
    const std::vector<Vec4> identity(2, Vec4(1, 0, 0, 0));
    const auto stationary = general_planar_DAT(.9, .0075, scene.mesh, scene.update,
        scene.positions, zeros, zeros, identity, zeros, zeros, .01);
    expect_vectors_identical(stationary.positions, scene.positions);
    const Vec3 normal(.95038202192690158, -.20115535389100869, .23729841128654272);
    const std::vector<Vec3> approaching{1e-4 * normal, -1e-4 * normal};
    const auto accepted = general_planar_DAT(.9, .0075, scene.mesh, scene.update,
        scene.positions, zeros, zeros, identity, approaching, zeros, .01);
    EXPECT_GT(benchmark_edge_signed_gap(accepted.positions), 0);
    for (int body = 0; body < 2; ++body) {
        EXPECT_GT(accepted.body_factors[body], 0);
        EXPECT_LT(accepted.body_factors[body], 1);
        const auto& nodes = scene.mesh.rb_nodes[body];
        for (std::size_t a = 0; a < nodes.size(); ++a)
            for (std::size_t b = 0; b < a; ++b)
                EXPECT_NEAR((accepted.positions[nodes[a]] - accepted.positions[nodes[b]]).norm(),
                    (scene.positions[nodes[a]] - scene.positions[nodes[b]]).norm(), 1e-14);
    }
}

TEST_F(PlanarDAT, IgnoresIncidentPrimitivesOfASingleTriangle) {
    Scene scene;
    scene.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    scene.update.assign(3, Vec3(0.1, 0.2, 0.1));
    scene.mesh.num_positions = 3;
    scene.mesh.tris = {0, 1, 2};
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 3u);
    for (int vertex = 0; vertex < 3; ++vertex) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
}

TEST_F(PlanarDAT, ProportionalVertexTrianglePlaneRelaxesExactlyOnce) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.update[vertex] = Vec3(0, 0, 0.1);
    }

    const auto positions_before = scene.positions;
    const auto update_before = scene.update;
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    // The gap is 0.4 and the closing speed 0.5: alpha=0.9*0.4/0.5.
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], 0.72 * scene.update[vertex]);
    }
    const double final_gap = scene.positions[0].z() + result[0].z()
        - scene.positions[1].z() - result[1].z();
    EXPECT_NEAR(final_gap, 0.04, tolerance);
    expect_vectors_identical(scene.positions, positions_before);
    expect_vectors_identical(scene.update, update_before);
}

TEST_F(PlanarDAT, VertexTriangleTruncationIsInvariantUnderUniformScaling) {
    Scene unscaled = point_triangle();
    unscaled.update[0] = Vec3(0, 0, -0.4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        unscaled.update[vertex] = Vec3(0, 0, 0.1);
    }
    for (double scale : {1e-100, 1e80}) {
        SCOPED_TRACE(scale);
        Scene scene = unscaled;
        for (Vec3& position : scene.positions) position *= scale;
        for (Vec3& update : scene.update) update *= scale;
        const auto result = planar_DAT(gamma, query_radius * scale, scene.mesh,
            scene.update, scene.positions);
        ASSERT_EQ(result.size(), 4u);
        for (int vertex = 0; vertex < 4; ++vertex) {
            // Compare in the original units so a tiny-scale zero result
            // cannot pass an absolute world-space tolerance accidentally.
            expect_vector_near(result[vertex] / scale,
                0.72 * unscaled.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, OneSidedApproachReceivesTheWholeAvailableGap) {
    for (bool triangle_moves : {false, true}) {
        SCOPED_TRACE(triangle_moves);
        Scene scene = point_triangle();
        if (triangle_moves) {
            for (int vertex = 1; vertex < 4; ++vertex) {
                scene.update[vertex] = Vec3(0, 0, 0.4);
            }
        } else {
            scene.update[0] = Vec3(0, 0, -0.4);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), 4u);
        for (int vertex = 0; vertex < 4; ++vertex) {
            expect_vector_near(result[vertex], gamma * scene.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, TangentialAndSeparatingMotionRemainUnchanged) {
    for (bool separating : {false, true}) {
        SCOPED_TRACE(separating);
        Scene scene = point_triangle();
        scene.update[0] = Vec3(0.1, 0.2, separating ? 0.1 : 0.0);
        for (int vertex = 1; vertex < 4; ++vertex) {
            scene.update[vertex] = Vec3(-0.2, 0.1, separating ? -0.1 : 0.0);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), scene.update.size());
        for (std::size_t vertex = 0; vertex < result.size(); ++vertex) {
            expect_vector_near(result[vertex], scene.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, TriangleUsesMaximumCornerApproach) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.2);
    scene.update[1] = Vec3(0, 0, 0.4);
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    // A nearest-point barycentric velocity would underestimate this corner.
    expect_vector_near(result[0], Vec3(0, 0, -0.12));
    expect_vector_near(result[1], Vec3(0, 0, 0.24));
    expect_vector_near(result[2], Vec3::Zero());
    expect_vector_near(result[3], Vec3::Zero());
}

TEST_F(PlanarDAT, ConstructsAdaptivePlanesFromAlreadyCappedUpdates) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.update[vertex] = Vec3(0, 0, 0.1);
    }
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    const double alpha = gamma * 0.4 / (0.45 + 0.1);
    expect_vector_near(result[0], Vec3(0, 0, -0.45 * alpha));
    for (int vertex = 1; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], Vec3(0, 0, 0.1 * alpha));
    }
    EXPECT_NEAR(0.4 + result[0].z() - result[1].z(), 0.04, tolerance);
}

TEST_F(PlanarDAT, QueryBoundaryAndFarPairsNeedOnlyTheDisplacementCap) {
    for (double gap : {1.0, 1.01, 3.0}) {
        SCOPED_TRACE(gap);
        Scene scene = point_triangle(gap);
        scene.update[0] = Vec3(0, 0, -10);
        for (int vertex = 1; vertex < 4; ++vertex) {
            scene.update[vertex] = Vec3(0, 0, 10);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), 4u);
        expect_vector_near(result[0], Vec3(0, 0, -0.45));
        for (int vertex = 1; vertex < 4; ++vertex) {
            expect_vector_near(result[vertex], Vec3(0, 0, 0.45));
        }
        EXPECT_GE(gap + result[0].z() - result[1].z(), 0.1 - tolerance);
    }
}

TEST_F(PlanarDAT, SliverTriangleUsesFiniteTriangleDistanceForRadiusFiltering) {
    Scene scene;
    scene.positions = {Vec3(0.5, 1e-3, 1e-4), Vec3(0, 0, 0),
        Vec3(1, 0, 0), Vec3(1, 1e-9, 0)};
    scene.update.assign(4, Vec3::Zero());
    scene.update[0] = Vec3(0, 0, -1e-4);
    scene.mesh.num_positions = 4;
    scene.mesh.tris = {1, 2, 3};

    // The point is near the supporting plane but outside the sliver. Its
    // true finite-triangle distance exceeds r_q, even though the AABBs overlap.
    const auto result = planar_DAT(gamma, 5e-4, scene.mesh,
        scene.update, scene.positions);
    ASSERT_EQ(result.size(), 4u);
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
}

TEST_F(PlanarDAT, ObtuseTriangleTruncatesAgainstTheActualNearestEdge) {
    for (double scale : {1.0, 1e-4}) {
        const std::array<Vec3, 3> triangle{{Vec3::Zero(), Vec3(1, 0, 0), Vec3(-10, .1, 0)}};
        std::array<int, 3> order{{0, 1, 2}};
        do {
            Scene scene;
            scene.positions = {scale * Vec3(.5, -.01, .1), scale * triangle[order[0]], scale * triangle[order[1]], scale * triangle[order[2]]};
            scene.update.assign(4, Vec3::Zero());
            scene.update[0] = scale * Vec3(0, .02, -.2);
            scene.mesh.num_positions = 4;
            scene.mesh.tris = {1, 2, 3};
            // The actual gap sqrt(.0101) is inside r_q=.3. The old
            // barycentric classification returned .51 and missed this pair.
            const auto result = planar_DAT(gamma, .3 * scale, scene.mesh, scene.update, scene.positions);
            expect_vector_near(result[0] / scale, gamma * Vec3(0, .01, -.1));
            for (int vertex = 1; vertex < 4; ++vertex) expect_vector_near(result[vertex], Vec3::Zero());
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST_F(PlanarDAT, EdgeEdgeContactTruncatesItsFourEndpointsOnly) {
    const Scene scene = crossed_edges();
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 6u);
    for (int vertex : {0, 1, 3, 4}) {
        expect_vector_near(result[vertex], 0.72 * scene.update[vertex]);
    }
    for (int vertex : {2, 5}) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
    EXPECT_NEAR(0.4 + result[3].z() - result[0].z(), 0.04, tolerance);
}

TEST_F(PlanarDAT, SharedVertexTakesMinimumIndependentOfTriangleOrder) {
    Scene scene = point_two_triangles();
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), scene.positions.size());
    // The two independent limits are 0.45 and 0.675.
    expect_vector_near(result[0], Vec3(0, 0, -0.18));
    for (std::size_t vertex = 1; vertex < result.size(); ++vertex) {
        expect_vector_near(result[vertex], Vec3::Zero());
    }
    scene.mesh.tris = {4, 5, 6, 1, 2, 3};
    expect_vectors_identical(filter(scene), result);
}

TEST_F(PlanarDAT, PaddingDoesNotChangeTheFilteredResult) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.4);
    const auto result = filter(scene, 1.0);
    expect_vector_near(result[0], Vec3(0, 0, -0.36));
    expect_vectors_identical(filter(scene, 1.2), result);
    expect_vectors_identical(filter(scene, 3.0), result);
}

TEST_F(PlanarDAT, SerialAndOpenMPResultsAreBitwiseIdentical) {
    Scene scene;
    for (int copy = 0; copy < 48; ++copy) {
        const Scene component = copy % 2 ? crossed_edges() : point_two_triangles();
        const int base = static_cast<int>(scene.positions.size());
        for (const Vec3& position : component.positions) {
            scene.positions.push_back(position + Vec3(30 * copy, 0, 0));
        }
        scene.update.insert(scene.update.end(),
            component.update.begin(), component.update.end());
        for (int vertex : component.mesh.tris) {
            scene.mesh.tris.push_back(base + vertex);
        }
    }
    scene.mesh.num_positions = scene.positions.size();
    omp_set_num_threads(1);
    const auto serial = filter(scene);
    for (int threads : {2, 4, 8}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        for (int repeat = 0; repeat < 3; ++repeat) {
            SCOPED_TRACE(repeat);
            expect_vectors_identical(filter(scene), serial);
        }
    }
}

TEST_F(PlanarDAT, SolidBoundaryContactsExcludeInteriorTetVertices) {
    Scene scene;
    // Four tetrahedra share interior node 4; node 5 approaches the outer
    // bottom face. The interior update stays inside the solid throughout.
    scene.positions = {Vec3(-3, -3, 0), Vec3(3, -3, 0), Vec3(0, 3, 0),
        Vec3(0, 0, 3), Vec3(0, 0, 0.1), Vec3(0, 0, -0.2)};
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.update[4] = Vec3(0, 0, -0.095);
    scene.update[5] = Vec3(0, 0, 0.4);
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
    scene.mesh.tets = {4, 0, 2, 1, 4, 0, 1, 3, 4, 1, 2, 3, 4, 2, 0, 3};
    scene.mesh.tet_nodes = {0, 1, 2, 3, 4};
    scene.mesh.surface_nodes = {0, 1, 2, 3};

    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 6u);
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], Vec3::Zero());
    }
    expect_vector_near(result[4], scene.update[4]);
    expect_vector_near(result[5], Vec3(0, 0, 0.18));
}

TEST_F(PlanarDAT, RejectsInitiallyTouchingVertexTriangleAndEdgeEdgePairs) {
    EXPECT_THROW(filter(point_triangle(0.0)), std::domain_error);
    EXPECT_THROW(filter(crossed_edges(0.0)), std::domain_error);
}

TEST_F(PlanarDAT, RejectsInvalidScalarParameters) {
    const Scene scene = point_triangle();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double invalid : {-0.1, 0.0, 1.0, 1.1, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(planar_DAT(invalid, query_radius, scene.mesh,
            scene.update, scene.positions), std::invalid_argument);
    }
    for (double invalid : {-1.0, 0.0, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(planar_DAT(gamma, invalid, scene.mesh,
            scene.update, scene.positions), std::invalid_argument);
    }
    for (double invalid : {-1.0, 0.99, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(filter(scene, invalid), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, RejectsMismatchedSizesAndMalformedTriangleIndices) {
    const Scene valid = point_triangle();
    Scene scene = valid;
    scene.mesh.num_positions += 1;
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.positions.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.update.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.mesh.tris.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    for (int invalid : {-1, 4}) {
        scene = valid;
        scene.mesh.tris[0] = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, RejectsNonfinitePositionsAndUpdates) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double invalid : {nan, inf, -inf}) {
        SCOPED_TRACE(invalid);
        Scene scene = point_triangle();
        scene.positions[0].x() = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
        scene = point_triangle();
        scene.update[2].z() = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, EmptyMeshProducesAnEmptyUpdate) {
    EXPECT_TRUE(filter(Scene{}).empty());
}

TEST_F(PlanarDAT, PairOnlyBroadPhaseAndReusableScratchMatchStandaloneAfterMotion) {
    auto scene=point_triangle(.4);
    scene.update[0]=Vec3(0,0,-.8);
    DATWorkspace workspace;
    BroadPhase phase;
    for(int sweep=0;sweep<4;++sweep) {
        std::vector<AABB> boxes;
        for(const auto& x:scene.positions)boxes.emplace_back(x-Vec3::Constant(2),x+Vec3::Constant(2));
        phase.initialize(boxes,scene.mesh,0,BroadPhase::InitializationMode::DeformableSolver,false);
        ASSERT_TRUE(phase.cache().vertex_nt.empty());
        ASSERT_TRUE(phase.cache().vertex_ss.empty());
        const auto expected=planar_DAT(gamma,query_radius,scene.mesh,scene.update,scene.positions);
        const auto actual=planar_DAT(gamma,query_radius,scene.mesh,scene.update,
            scene.positions,1.2,&phase,&workspace);
        for(std::size_t i=0;i<actual.size();++i)
            EXPECT_EQ(std::memcmp(actual[i].data(),expected[i].data(),3*sizeof(double)),0);
        scene.positions[0]+=actual[0]*.2;
    }
}

BroadPhase iteration_candidates(const RefMesh& mesh,
    const std::vector<Vec3>& positions, double radius,
    BroadPhase::InitializationMode mode = BroadPhase::InitializationMode::DeformableSolver) {
    std::vector<AABB> boxes;
    for (const Vec3& position : positions)
        boxes.emplace_back(position - Vec3::Constant(radius),
            position + Vec3::Constant(radius));
    BroadPhase phase;
    phase.initialize(boxes, mesh, 0.0, mode, false);
    return phase;
}

TEST_F(PlanarDAT, IterationCandidatesCoverVertexTrianglePairsAcrossColorUpdates) {
    constexpr double relaxation = .999, radius = .8;
    auto scene = point_triangle(.55);
    scene.positions.emplace_back(0, 0, 1.0);
    scene.mesh.num_positions = scene.positions.size();
    const auto initial = scene.positions;
    auto phase = iteration_candidates(scene.mesh, initial, radius);
    DATWorkspace workspace;
    const auto signed_gap = [&](int node) {
        const Vec3 normal = (scene.positions[2] - scene.positions[1]).cross(
            scene.positions[3] - scene.positions[1]).normalized();
        return (scene.positions[node] - scene.positions[1]).dot(normal);
    };
    // This point starts outside the DAT query radius. The iteration envelope
    // must retain it when earlier colors move its triangle into range.
    ASSERT_GT(initial[4].z() - initial[1].z(), radius);
    ASSERT_TRUE(std::any_of(phase.cache().nt_pairs.begin(),
        phase.cache().nt_pairs.end(), [](const auto& pair) { return pair.node == 4; }));

    for (int node : {1, 2, 3, 0, 4}) {
        SCOPED_TRACE(node);
        std::vector<Vec3> proposal(scene.positions.size(), Vec3::Zero());
        proposal[node] = Vec3(0, 0, node == 0 ? -.7 : node == 4 ? -.2 : .38);
        const auto expected = planar_DAT(relaxation, radius, scene.mesh,
            proposal, scene.positions, 1.0);
        const auto accepted = planar_DAT(relaxation, radius, scene.mesh,
            proposal, scene.positions, 1.0, &phase, &workspace);
        expect_vectors_identical(accepted, expected);
        for (std::size_t i = 0; i < accepted.size(); ++i) {
            if (int(i) != node) EXPECT_TRUE(accepted[i].isZero(0));
            scene.positions[i] += accepted[i];
            EXPECT_LE((scene.positions[i] - initial[i]).norm(),
                .5 * relaxation * radius + tolerance);
        }
        if (node == 3) {
            EXPECT_LT(signed_gap(4), radius);
        }
        if (node == 0) {
            // The later point color must use the moved triangle's separation.
            EXPECT_LT(accepted[0].norm(), .5 * relaxation * radius);
            EXPECT_GT(signed_gap(0), 0);
        }
    }
    EXPECT_GT(signed_gap(0), 0);
    EXPECT_GT(signed_gap(4), 0);
}

TEST_F(PlanarDAT, IterationCandidatesCoverEdgePairsAcrossColorUpdates) {
    constexpr double relaxation = .999, radius = .8;
    for (double gap : {.55, 1.0}) {
        SCOPED_TRACE(gap);
        auto scene = crossed_edges(gap);
        const auto initial = scene.positions;
        auto phase = iteration_candidates(scene.mesh, initial, radius);
        DATWorkspace workspace;
        ASSERT_TRUE(std::any_of(phase.cache().ss_pairs.begin(),
            phase.cache().ss_pairs.end(), [](const auto& pair) {
                std::array<int, 4> nodes{{pair.v[0], pair.v[1], pair.v[2], pair.v[3]}};
                std::sort(nodes.begin(), nodes.end());
                return nodes == std::array<int, 4>{{0, 1, 3, 4}};
            }));
        for (int node : {0, 1, 3, 4}) {
            SCOPED_TRACE(node);
            std::vector<Vec3> proposal(scene.positions.size(), Vec3::Zero());
            proposal[node] = Vec3(0, 0, node < 2 ? .38 : -.7);
            const auto expected = planar_DAT(relaxation, radius, scene.mesh,
                proposal, scene.positions, 1.0);
            const auto accepted = planar_DAT(relaxation, radius, scene.mesh,
                proposal, scene.positions, 1.0, &phase, &workspace);
            expect_vectors_identical(accepted, expected);
            for (std::size_t i = 0; i < accepted.size(); ++i) {
                if (int(i) != node) EXPECT_TRUE(accepted[i].isZero(0));
                scene.positions[i] += accepted[i];
                EXPECT_LE((scene.positions[i] - initial[i]).norm(),
                    .5 * relaxation * radius + tolerance);
            }
            if (node == 1) {
                EXPECT_LT(segment_segment_distance(scene.positions[0],
                    scene.positions[1], scene.positions[3], scene.positions[4]).distance, radius);
            }
        }
        // Both edge projections still cross at their midpoints in the xy plane.
        EXPECT_GT(.5 * (scene.positions[3].z() + scene.positions[4].z()
            - scene.positions[0].z() - scene.positions[1].z()), 0);
        EXPECT_GT(segment_segment_distance(scene.positions[0], scene.positions[1],
            scene.positions[3], scene.positions[4]).distance, 0);
    }
}

TEST_F(PlanarDAT, IterationCandidatesCoverParticlesAfterARigidColorRotates) {
    constexpr double relaxation = .999, radius = .8, dt = 1.0;
    RefMesh mesh;
    DeformedState state;
    create_rigid_body({Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(0, 1, 0)},
        Vec3::Zero(), Vec4(1, 0, 0, 0), Vec3::Zero(), 3, mesh, state);
    mesh.tris = {0, 1, 2};
    auto positions = state.deformed_positions;
    positions.emplace_back(0, 0, .55);
    positions.emplace_back(0, 0, 1.0);
    mesh.num_positions = positions.size();
    mesh.node_to_rb.resize(positions.size(), -1);
    const auto initial = positions;
    auto centers = state.x_coms;
    std::vector<Vec3> angular(1, Vec3::Zero());
    auto phase = iteration_candidates(mesh, initial, radius,
        BroadPhase::InitializationMode::GeneralSolver);
    DATWorkspace workspace;
    ASSERT_TRUE(std::any_of(phase.cache().nt_pairs.begin(),
        phase.cache().nt_pairs.end(), [](const auto& pair) { return pair.node == 4; }));
    for (int color = 0; color < 3; ++color) {
        SCOPED_TRACE(color);
        std::vector<Vec3> proposal(positions.size(), Vec3::Zero());
        std::vector<Vec3> delta_centers(1, Vec3::Zero()), delta_angular(1, Vec3::Zero());
        if (color == 0) {
            delta_centers[0] = Vec3(0, 0, .3);
            delta_angular[0] = Vec3(0, .08, 0);
        } else {
            proposal[color + 2] = Vec3(0, 0, color == 1 ? -.7 : -.2);
        }
        const auto expected = general_planar_DAT(relaxation, radius, mesh,
            proposal, positions, centers, angular, state.orientations,
            delta_centers, delta_angular, dt, 1.0);
        const auto accepted = general_planar_DAT(relaxation, radius, mesh,
            proposal, positions, centers, angular, state.orientations,
            delta_centers, delta_angular, dt, 1.0, &phase, &workspace);
        expect_vectors_identical(accepted.positions, expected.positions);
        expect_vectors_identical(accepted.centers, expected.centers);
        expect_vectors_identical(accepted.omega, expected.omega);
        ASSERT_EQ(accepted.orientations.size(), expected.orientations.size());
        EXPECT_TRUE(accepted.orientations[0].isApprox(expected.orientations[0], 0));
        EXPECT_EQ(accepted.body_factors, expected.body_factors);
        positions = accepted.positions;
        centers = accepted.centers;
        angular = accepted.omega;
        if (color == 0) {
            EXPECT_GT(angular[0].norm(), .01);
            EXPECT_GT((centers[0] - state.x_coms[0]).norm(), .1);
        }
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < a; ++b)
                EXPECT_NEAR((positions[a] - positions[b]).norm(),
                    (initial[a] - initial[b]).norm(), tolerance);
        }
        for (std::size_t node = 0; node < positions.size(); ++node)
            EXPECT_LE((positions[node] - initial[node]).norm(),
                .5 * relaxation * radius + tolerance);
        const Vec3 normal = (positions[1] - positions[0]).cross(
            positions[2] - positions[0]).normalized();
        EXPECT_GT((positions[3] - positions[0]).dot(normal), 0);
        EXPECT_GT((positions[4] - positions[0]).dot(normal), 0);
        if (color == 0)
            EXPECT_LT((positions[4] - positions[0]).dot(normal), radius);
    }
}

struct OpenMPRestore {
    int threads=omp_get_max_threads(),dynamic=omp_get_dynamic();
    ~OpenMPRestore(){omp_set_num_threads(threads);omp_set_dynamic(dynamic);}
};
struct ContactCloud {
    RefMesh mesh;
    DeformedState state;
    SimParams params=SimParams::zeros();
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    explicit ContactCloud(RigidBodyUpdateMode mode,double friction) {
        create_rigid_body({Vec3(-4,0,-4),Vec3(4,0,-4),Vec3(0,0,4)},
            Vec3::Zero(),Vec4(1,0,0,0),Vec3::Zero(),3,mesh,state,RigidBodyUpdateMode::None);
        mesh.tris={0,2,1};
        std::vector<Vec3> points;
        for(int j=0;j<24;++j)for(int i=0;i<24;++i)
            points.emplace_back(-.4+.8*i/23.,.02+.003*std::sin(i+j),-.4+.8*j/23.);
        create_rigid_body(points,Vec3(.2,-.05,.1),Vec4(1,0,0,0),Vec3(.05,.07,.02),
            double(points.size()),mesh,state,mode);
        mesh.initialize(state.deformed_positions);mesh.build_deformable_nodes();
        adjacency=build_incident_triangle_map(mesh.tris);
        params.fps=100;params.substeps=1;params.fixed_iters=true;params.max_global_iters=3;
        params.damping=.5;params.d_hat=.05;params.k_barrier=30;
        params.dat_query_radius=.1;params.use_dat=true;params.friction_coefficient=friction;
    }
};
struct Result {
    std::vector<Vec3> positions,centers,omega;
    std::vector<Vec4> orientations;
    SolverResult statistics;
};
Result solve(ContactCloud& scene,int threads,bool parallel) {
    omp_set_num_threads(threads);
    SimParams params=scene.params;params.use_parallel=parallel;
    Result out{scene.state.deformed_positions,scene.state.x_coms,
        std::vector<Vec3>(scene.state.omega.size(),Vec3::Zero()),scene.state.orientations};
    BroadPhase broad;
    const auto result=global_jacobi_solver_dat_general(scene.mesh,scene.state,scene.adjacency,
        scene.pins,params,out.positions,scene.state.deformed_positions,out.centers,
        out.orientations,out.omega,broad);
    if(params.fixed_iters)EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations,params.max_global_iters);
    out.statistics=result;
    return out;
}
template<class Vector>
void expect_same(const std::vector<Vector>& actual,const std::vector<Vector>& expected) {
    ASSERT_EQ(actual.size(),expected.size());
    for(std::size_t i=0;i<actual.size();++i) {
        ASSERT_TRUE(actual[i].allFinite())<<i;
        EXPECT_EQ(std::memcmp(actual[i].data(),expected[i].data(),Vector::SizeAtCompileTime*sizeof(double)),0)<<i;
    }
}
void expect_same(const Result& actual,const Result& expected) {
    expect_same(actual.positions,expected.positions);expect_same(actual.centers,expected.centers);
    expect_same(actual.omega,expected.omega);expect_same(actual.orientations,expected.orientations);
}
void expect_same(const GeneralDATResult& actual,const GeneralDATResult& expected) {
    expect_same(actual.positions,expected.positions);expect_same(actual.centers,expected.centers);
    expect_same(actual.omega,expected.omega);expect_same(actual.orientations,expected.orientations);
    ASSERT_EQ(actual.body_factors.size(),expected.body_factors.size());
    for(std::size_t b=0;b<actual.body_factors.size();++b)
        EXPECT_EQ(std::memcmp(&actual.body_factors[b],&expected.body_factors[b],sizeof(double)),0)<<b;
}

struct RadiusCloud {
    RefMesh mesh;
    std::vector<Vec3> positions,updates;
    const std::vector<Vec3> zero{Vec3::Zero()},rotation{Vec3(0,0,1.1)};
    const std::vector<Vec4> orientation{Vec4(1,0,0,0)};
    RadiusCloud() {
        constexpr int regular_nodes=1024;
        for(int i=0;i<regular_nodes;++i) {
            const double angle=2*std::acos(-1.0)*i/regular_nodes;
            const double radius=.2+.0005*(i%7);
            positions.emplace_back(radius*std::cos(angle),radius*std::sin(angle),.1*std::sin(3*angle));
        }
        // The limiting node is in the last, incomplete work chunk.
        positions.emplace_back(6,0,.2);
        mesh.num_positions=positions.size();
        mesh.ref_positions={positions};mesh.rb_nodes.resize(1);
        mesh.node_to_rb.assign(positions.size(),0);
        for(std::size_t v=0;v<positions.size();++v)mesh.rb_nodes[0].push_back(int(v));
        updates.assign(positions.size(),Vec3::Zero());
    }
    GeneralDATResult run(int threads,const RefMesh& topology,DATWorkspace* workspace=nullptr) const {
        omp_set_num_threads(threads);
        return general_planar_DAT(.9,1,topology,updates,positions,zero,zero,
            orientation,zero,rotation,1,1.2,nullptr,workspace);
    }
};
} // namespace

TEST(DATParallel, HeavySingleBodyContactsPreserveSerialArithmeticAcrossThreadsAndModes) {
    OpenMPRestore restore;omp_set_dynamic(0);
    for(auto mode:{RigidBodyUpdateMode::TranslationAndOrientation,
                   RigidBodyUpdateMode::TranslationOnly,RigidBodyUpdateMode::OrientationOnly})
        for(double friction:{0.,.3}) {
            SCOPED_TRACE(::testing::Message()<<"mode="<<int(mode)<<" friction="<<friction);
            ContactCloud scene(mode,friction);
            const auto original=scene.state.deformed_positions;
            const auto serial=solve(scene,1,false);
            for(int threads:{2,4,8}) {
                expect_same(solve(scene,threads,true),serial);
                expect_same(scene.state.deformed_positions,original);
            }
        }
}

TEST(DATParallel, CooperativeFailureJoinsAndAllowsTheNextSolve) {
    OpenMPRestore restore;omp_set_dynamic(0);
    ContactCloud scene(RigidBodyUpdateMode::TranslationAndOrientation,.3);
    const auto original=scene.state.deformed_positions;
    scene.params.k_barrier=std::numeric_limits<double>::max();
    EXPECT_THROW((void)solve(scene,8,true),std::exception);
    expect_same(scene.state.deformed_positions,original);
    scene.params.k_barrier=30;
    const auto serial=solve(scene,1,false);
    expect_same(solve(scene,8,true),serial);
}

TEST(DATParallel, HeavyRigidInitialResidualKeepsOrderedArithmeticAcrossThreads) {
    OpenMPRestore restore;omp_set_dynamic(0);
    for(double friction:{0.,.3}) {
        ContactCloud scene(RigidBodyUpdateMode::TranslationAndOrientation,friction);
        scene.params.fixed_iters=false;scene.params.max_global_iters=0;
        const auto serial=solve(scene,1,false);
        ASSERT_TRUE(serial.statistics.has_residual);
        ASSERT_GT(serial.statistics.final_rigid_residual,0);
        for(int threads:{2,4,8}) {
            const auto parallel=solve(scene,threads,true);
            expect_same(parallel,serial);
            EXPECT_EQ(std::memcmp(&parallel.statistics.initial_residual,
                &serial.statistics.initial_residual,sizeof(double)),0);
            EXPECT_EQ(std::memcmp(&parallel.statistics.final_rigid_residual,
                &serial.statistics.final_rigid_residual,sizeof(double)),0);
        }
    }
}

TEST(DATParallel, SingleLargeBodyRadiusMinimumMatchesTheAnalyticRotationCapAcrossThreads) {
    OpenMPRestore restore;omp_set_dynamic(0);
    const RadiusCloud scene;
    const auto original=scene.positions;
    const auto serial=scene.run(1,scene.mesh);
    ASSERT_EQ(serial.body_factors.size(),1u);
    // With a fixed rotation axis, the farthest proxy moves through a chord
    // 2*r*sin(alpha*theta/2). No contact pair can reduce this radius cap.
    const double analytic=2*std::asin(.45/(2*6))/1.1;
    EXPECT_GT(serial.body_factors[0],0);
    EXPECT_LE(serial.body_factors[0],analytic);
    EXPECT_NEAR(serial.body_factors[0],analytic,5e-10);
    EXPECT_NEAR((serial.positions.back()-original.back()).norm(),.45,5e-9);
    for(int threads:{2,4,8}) {
        SCOPED_TRACE(threads);
        expect_same(scene.run(threads,scene.mesh),serial);
        expect_same(scene.positions,original);
    }
}

TEST(DATParallel, DuplicateProxyClaimsJoinAndLeaveReusableWorkspaceRecoverable) {
    OpenMPRestore restore;omp_set_dynamic(0);
    const RadiusCloud scene;
    const auto original=scene.positions;
    const auto serial=scene.run(1,scene.mesh);
    auto invalid=scene.mesh;
    // The two claimants are far apart in the flattened parallel work list.
    invalid.rb_nodes[0][768]=invalid.rb_nodes[0][0];
    for(int threads:{1,2,4,8}) {
        SCOPED_TRACE(threads);
        DATWorkspace workspace;
        EXPECT_THROW((void)scene.run(threads,invalid,&workspace),std::invalid_argument);
        expect_same(scene.positions,original);
        expect_same(scene.run(threads,scene.mesh,&workspace),serial);
        EXPECT_EQ(invalid.rb_nodes[0][768],0);
        EXPECT_EQ(scene.mesh.rb_nodes[0][768],768);
    }
}

TEST(DATParallel, ReusedEndpointConstraintsClearBeforePairRejectionAndAfterContactsDisappear) {
    OpenMPRestore restore;omp_set_dynamic(0);
    RefMesh mesh;DeformedState state;
    create_rigid_body({Vec3(-1,-1,0),Vec3(1,-1,0),Vec3(-1,1,0)},
        Vec3::Zero(),Vec4(1,0,0,0),Vec3::Zero(),3,mesh,state,RigidBodyUpdateMode::None);
    mesh.tris={0,1,2};
    auto near=state.deformed_positions;
    for(int i=0;i<300;++i)near.emplace_back(-.5+.001*(i%20),-.4+.001*(i/20),.2);
    mesh.num_positions=near.size();mesh.node_to_rb.resize(near.size(),-1);
    auto aabb_rejected=near,outside_query=near,absent=near;
    std::vector<Vec3> updates(near.size(),Vec3(.04,-.02,-.05));
    for(int v=0;v<3;++v)updates[v].setZero();
    for(std::size_t v=3;v<near.size();++v) {
        aabb_rejected[v].z()=-2;
        // Inside the inflated triangle AABB, outside the finite triangle's
        // query radius, and on the opposite side of the previous contact plane.
        outside_query[v]=Vec3(1.2+.0001*(v%20),1.2,-.2);
        absent[v]+=Vec3(100,100,-100);
    }
    auto candidates=iteration_candidates(mesh,near,5,BroadPhase::InitializationMode::GeneralSolver);
    auto no_candidates=iteration_candidates(mesh,absent,1.2,BroadPhase::InitializationMode::GeneralSolver);
    ASSERT_GE(candidates.cache().nt_pairs.size(),300u);
    ASSERT_TRUE(no_candidates.cache().nt_pairs.empty());
    const std::vector<Vec3> zeros(1,Vec3::Zero());
    const auto run=[&](const std::vector<Vec3>& positions,const BroadPhase* pairs,DATWorkspace* workspace) {
        return general_planar_DAT(.9,1,mesh,updates,positions,state.x_coms,zeros,
            state.orientations,zeros,zeros,.01,1.2,pairs,workspace);
    };
    for(int threads:{1,2,4,8}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        DATWorkspace workspace;
        const std::array<const std::vector<Vec3>*,7> snapshots{{&near,&aabb_rejected,
            &near,&outside_query,&near,&absent,&near}};
        for(const auto* positions:snapshots) {
            const BroadPhase* pairs=positions==&absent?&no_candidates:&candidates;
            const auto expected=run(*positions,pairs,nullptr);
            const auto reused=run(*positions,pairs,&workspace);
            expect_same(reused,expected);
            for(std::size_t v=3;v<positions->size();++v)
                EXPECT_GT((reused.positions[v]-(*positions)[v]).norm(),0);
        }
    }
}

TEST(DATParallel, EndpointBacktrackingKeepsRepresentedParticleStepsInsideTheCap) {
    OpenMPRestore restore;omp_set_dynamic(0);
    RefMesh mesh;DeformedState state;
    create_rigid_body({Vec3(-.125,-.125,-.125),Vec3(.125,.125,-.125),
        Vec3(.125,-.125,.125),Vec3(-.125,.125,.125)},Vec3::Zero(),
        Vec4(1,0,0,0),Vec3::Zero(),1,mesh,state,RigidBodyUpdateMode::None);
    auto positions=state.deformed_positions;
    for(int i=0;i<300;++i)positions.emplace_back(1e12+4*i,0,0);
    mesh.num_positions=positions.size();mesh.node_to_rb.resize(positions.size(),-1);
    std::vector<Vec3> updates(positions.size(),Vec3(.3,0,0));
    for(int i=0;i<4;++i)updates[i].setZero();
    const std::vector<Vec3> zeros(1,Vec3::Zero());
    const auto run=[&](int threads) {
        omp_set_num_threads(threads);
        return general_planar_DAT(.9,2./3,mesh,updates,positions,state.x_coms,
            zeros,state.orientations,zeros,zeros,.01);
    };
    ASSERT_GT((positions[4].x()+.3)-positions[4].x(),.3);
    const auto serial=run(1);
    for(int threads:{2,4,8}) {
        SCOPED_TRACE(threads);
        const auto parallel=run(threads);
        expect_same(parallel,serial);
        for(std::size_t i=4;i<positions.size();++i) {
            const double displacement=parallel.positions[i].x()-positions[i].x();
            EXPECT_GT(displacement,0);
            EXPECT_LT(displacement,.25); // A real shortened retry, not a zero update.
            EXPECT_LE(displacement,.3);
        }
    }
    expect_same(state.deformed_positions,std::vector<Vec3>(positions.begin(),positions.begin()+4));
}

TEST(DATWorkspaceReuse, InternalAndSuppliedCandidatesAlternateWhileNodeCountsChange) {
    DATWorkspace workspace;
    auto scene=point_triangle(.4);
    for(int step=0;step<3;++step) {
        // Keep the internally cached primitive topology fixed. The supplied
        // middle query adds particles, exercising scratch growth/shrinkage
        // without changing the internal broad phase's structural contract.
        const bool supplied=step==1;
        scene.positions.resize(supplied?7:4);
        for(std::size_t node=4;node<scene.positions.size();++node)
            scene.positions[node]=Vec3(.01*node,0,.5);
        scene.mesh.num_positions=scene.positions.size();
        scene.update.assign(scene.positions.size(),Vec3(0,0,-.3));
        scene.update[0]=Vec3(0,0,-.8);
        for(int node=1;node<4;++node)scene.update[node].setZero();
        auto candidates=iteration_candidates(scene.mesh,scene.positions,1.2);
        const BroadPhase* pairs=supplied?&candidates:nullptr;
        const auto expected=planar_DAT(.9,1,scene.mesh,scene.update,scene.positions,1.2,pairs);
        const auto actual=planar_DAT(.9,1,scene.mesh,scene.update,scene.positions,1.2,pairs,&workspace);
        expect_vectors_identical(actual,expected);
    }

    DATWorkspace rigid_workspace;
    RefMesh mesh;DeformedState state;
    create_rigid_body({Vec3(-1,-1,0),Vec3(1,-1,0),Vec3(-1,1,0)},
        Vec3::Zero(),Vec4(1,0,0,0),Vec3::Zero(),3,mesh,state);
    mesh.tris={0,1,2};
    auto positions=state.deformed_positions,centers=state.x_coms;
    std::vector<Vec3> angular(1,Vec3::Zero());
    const std::vector<Vec3> dc{Vec3(.01,0,.02)},dw{Vec3(0,.02,.01)};
    for(int step=0;step<3;++step) {
        const bool supplied=step==1;
        const int particles=supplied?4:1;
        positions.resize(3+particles);
        for(int particle=0;particle<particles;++particle)
            positions[3+particle]=Vec3(-.5+.01*particle,-.4,.4);
        mesh.num_positions=positions.size();mesh.node_to_rb.resize(positions.size(),-1);
        std::vector<Vec3> updates(positions.size(),Vec3(.03,0,-.02));
        for(int node=0;node<3;++node)updates[node].setZero();
        auto candidates=iteration_candidates(mesh,positions,1.2,
            BroadPhase::InitializationMode::GeneralSolver);
        const BroadPhase* pairs=supplied?&candidates:nullptr;
        const auto expected=general_planar_DAT(.9,1,mesh,updates,positions,
            centers,angular,state.orientations,dc,dw,.1,1.2,pairs);
        const auto actual=general_planar_DAT(.9,1,mesh,updates,positions,
            centers,angular,state.orientations,dc,dw,.1,1.2,pairs,&rigid_workspace);
        expect_same(actual,expected);
        positions=actual.positions;centers=actual.centers;angular=actual.omega;
    }
}

TEST(DATWorkspaceReuse, SuppliedCandidatesStillRejectExtentOverflowAndRecover) {
    for(bool rigid:{false,true}) {
        SCOPED_TRACE(rigid);
        RefMesh mesh;DeformedState state;
        if(rigid)create_rigid_body({Vec3(-.1,-.1,-.1),Vec3(.1,.1,-.1),
            Vec3(.1,-.1,.1),Vec3(-.1,.1,.1)},Vec3::Zero(),Vec4(1,0,0,0),
            Vec3::Zero(),1,mesh,state,RigidBodyUpdateMode::None);
        auto positions=state.deformed_positions;
        positions.emplace_back(3,0,0);
        mesh.num_positions=positions.size();mesh.node_to_rb.resize(positions.size(),-1);
        const auto original=positions;
        const std::vector<Vec3> updates(positions.size(),Vec3::Zero());
        const std::vector<Vec3> zeros(state.x_coms.size(),Vec3::Zero());
        auto candidates=iteration_candidates(mesh,positions,1.2,
            rigid?BroadPhase::InitializationMode::GeneralSolver
                :BroadPhase::InitializationMode::DeformableSolver);
        DATWorkspace workspace;
        const auto run=[&](const std::vector<Vec3>& current,double radius,DATWorkspace* scratch) {
            if(rigid)return general_planar_DAT(.9,radius,mesh,updates,current,
                state.x_coms,zeros,state.orientations,zeros,zeros,.1,1,
                &candidates,scratch).positions;
            return planar_DAT(.9,radius,mesh,updates,current,1,&candidates,scratch);
        };
        const auto expected=run(positions,1,nullptr);
        expect_vectors_identical(run(positions,1,&workspace),expected);
        auto overflowing=positions;
        overflowing.back().x()=std::numeric_limits<double>::max();
        // No primitives are present, so the supplied empty pair set remains
        // valid. Skipping unused box storage must retain local extent checks.
        EXPECT_THROW((void)run(overflowing,1e308,&workspace),std::invalid_argument);
        expect_vectors_identical(run(positions,1,&workspace),expected);
        expect_vectors_identical(positions,original);
    }
}

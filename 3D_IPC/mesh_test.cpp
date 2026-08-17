#include "mesh.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace {

std::array<int, 3> sorted_face(int a, int b, int c) {
    std::array<int, 3> face{a, b, c};
    std::sort(face.begin(), face.end());
    return face;
}

std::vector<Vec3> unit_tet_positions() {
    return {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
}

std::vector<Vec3> subdivided_tet_positions() {
    std::vector<Vec3> positions = unit_tet_positions();
    positions.push_back(Vec3(0.2, 0.2, 0.2));
    return positions;
}

std::vector<int> subdivided_tets() {
    return {
        4, 1, 2, 3,
        0, 4, 2, 3,
        0, 1, 4, 3,
        0, 1, 2, 4,
    };
}

} // namespace

TEST(Mesh, TetFaceMatchesTgslExactly) {
    EXPECT_EQ(TetFace(0), (std::array<int, 3>{1, 2, 3}));
    EXPECT_EQ(TetFace(1), (std::array<int, 3>{0, 3, 2}));
    EXPECT_EQ(TetFace(2), (std::array<int, 3>{0, 1, 3}));
    EXPECT_EQ(TetFace(3), (std::array<int, 3>{0, 2, 1}));
    EXPECT_EQ(TetFace(4), (std::array<int, 3>{-1, -1, -1}));
}

TEST(Mesh, SingleTetUsesTgslLocalFaceOrdering) {
    const std::vector<int> boundary =
        compute_boundary_tri_mesh({0, 1, 2, 3});

    EXPECT_EQ(boundary, (std::vector<int>{
        1, 2, 3,
        0, 3, 2,
        0, 1, 3,
        0, 2, 1,
    }));
}

TEST(Mesh, SharedTetFaceIsRemoved) {
    const std::vector<int> boundary = compute_boundary_tri_mesh({
        0, 1, 2, 3,
        0, 2, 1, 4,
    });

    ASSERT_EQ(boundary.size(), 18U);
    for (std::size_t face = 0; face < boundary.size(); face += 3) {
        EXPECT_NE(
            sorted_face(
                boundary[face], boundary[face + 1], boundary[face + 2]),
            (std::array<int, 3>{0, 1, 2}));
    }
}

TEST(Mesh, SubdividedTetBoundaryExcludesInteriorVertex) {
    const std::vector<int> boundary =
        compute_boundary_tri_mesh(subdivided_tets());

    EXPECT_EQ(boundary, (std::vector<int>{
        1, 2, 3,
        0, 3, 2,
        0, 1, 3,
        0, 2, 1,
    }));
    EXPECT_EQ(std::count(boundary.begin(), boundary.end(), 4), 0);
    EXPECT_EQ(
        compute_boundary_tri_mesh_nodes(boundary),
        (std::vector<int>{1, 2, 3, 0}));
}

TEST(Mesh, BoundaryExtractionRejectsInvalidTopology) {
    EXPECT_THROW(compute_boundary_tri_mesh({0, 1, 2}),
                 std::invalid_argument);
    EXPECT_THROW(compute_boundary_tri_mesh({0, 1, 2, -1}),
                 std::out_of_range);
    EXPECT_THROW(compute_boundary_tri_mesh({0, 1, 1, 3}),
                 std::invalid_argument);
    EXPECT_THROW(
        compute_boundary_tri_mesh({0, 1, 2, 3, 3, 2, 1, 0}),
        std::invalid_argument);
    EXPECT_THROW(
        compute_boundary_tri_mesh({
            0, 1, 2, 3,
            0, 2, 1, 4,
            0, 1, 2, 5,
        }),
        std::invalid_argument);
}

TEST(Mesh, BoundaryTriMeshNodesMatchesTgslFirstAppearanceOrder) {
    EXPECT_EQ(
        compute_boundary_tri_mesh_nodes({3, 1, 2, 3, 2, 4}),
        (std::vector<int>{3, 1, 2, 4}));
    EXPECT_THROW(compute_boundary_tri_mesh_nodes({0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(compute_boundary_tri_mesh_nodes({0, 1, -1}),
                 std::out_of_range);
}

TEST(Mesh, TetValidationChecksIndicesAndGeometry) {
    const std::vector<Vec3> positions = unit_tet_positions();
    EXPECT_NO_THROW(validate_tet_mesh({0, 1, 2, 3}, positions));
    EXPECT_THROW(validate_tet_mesh({0, 2, 1, 3}, positions),
                 std::invalid_argument);
    EXPECT_THROW(validate_tet_mesh({0, 1, 2}, positions),
                 std::invalid_argument);
    EXPECT_THROW(validate_tet_mesh({0, 1, 2, 4}, positions),
                 std::out_of_range);
    EXPECT_THROW(validate_tet_mesh({0, 1, 1, 3}, positions),
                 std::invalid_argument);

    std::vector<Vec3> coplanar = positions;
    coplanar[3] = Vec3(0.25, 0.25, 0.0);
    EXPECT_THROW(validate_tet_mesh({0, 1, 2, 3}, coplanar),
                 std::invalid_argument);

    std::vector<Vec3> non_finite = positions;
    non_finite[3].z() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(validate_tet_mesh({0, 1, 2, 3}, non_finite),
                 std::invalid_argument);
}

TEST(Mesh, TetDegeneracyValidationIsScaleAware) {
    for (const double scale : {1.0e-9, 1.0, 1.0e9}) {
        std::vector<Vec3> positions = unit_tet_positions();
        for (Vec3& position : positions)
            position *= scale;
        EXPECT_NO_THROW(validate_tet_mesh(
            {0, 1, 2, 3}, positions));
    }
}

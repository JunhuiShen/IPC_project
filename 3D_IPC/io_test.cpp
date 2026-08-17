#include "io.h"

#include "mesh.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

class TetGenIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::atomic<std::uint64_t> next_directory{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        directory_ = fs::temp_directory_path()
                     / ("ipc_tetgen_io_test_"
                        + std::to_string(timestamp) + "_"
                        + std::to_string(next_directory.fetch_add(1)));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code error;
        fs::remove_all(directory_, error);
    }

    std::string write_file(
        const std::string& filename,
        const std::string& contents) const {
        const fs::path path = directory_ / filename;
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error("could not create TetGen test file");
        output << contents;
        if (!output)
            throw std::runtime_error("could not write TetGen test file");
        return path.string();
    }

    std::string read_file(const fs::path& path) const {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("could not open TetGen test file");
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    fs::path directory_;
};

void expect_positions_equal(
    const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(actual[i].isApprox(expected[i], 0.0)) << "position " << i;
}

TEST_F(TetGenIOTest, ReadsOneBasedNodesTetsAndFaces) {
    const std::string node_file = write_file(
        "unit.node",
        "# Four points, two attributes, and one boundary marker.\n"
        "\n"
        "4 3 2 1 # point count, dimension, attributes, markers\n"
        "1 0 0 0 10.5 -1.0 101\n"
        "2 1 0 0 11.5 -2.0 102 # inline comment\n"
        "3 0 1 0 12.5 -3.0 103\n"
        "4 0 0 1 13.5 -4.0 104\n");
    const std::string ele_file = write_file(
        "unit.ele",
        "# One linear tetrahedron with two unused attributes.\n"
        "1 4 2\n"
        "1 1 2 3 4 6.25 -8.5 # attributes are ignored\n");
    const std::string face_file = write_file(
        "unit.face",
        "# The outward faces of the unit tetrahedron.\n"
        "4 1\n"
        "1 2 3 4 91\n"
        "2 1 4 3 92\n"
        "3 1 2 4 93\n"
        "4 1 3 2 94\n");

    std::vector<Vec3> positions{Vec3(99.0, 99.0, 99.0)};
    std::vector<int> tets{99};
    std::vector<int> faces{99};
    read_tetgen_nodes(node_file, positions);
    read_tetgen_tets(ele_file, tets);
    read_tetgen_faces(face_file, faces);

    expect_positions_equal(
        positions,
        {
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
        });
    EXPECT_EQ(tets, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(
        faces,
        (std::vector<int>{
            1, 2, 3,
            0, 3, 2,
            0, 1, 3,
            0, 2, 1,
        }));
}

TEST_F(TetGenIOTest, ReadsZeroBasedNodesTetsAndFaces) {
    const std::string node_file = write_file(
        "zero.node",
        "4 3 1 0\n"
        "0 0 0 0 10\n"
        "1 1 0 0 11\n"
        "2 0 1 0 12\n"
        "3 0 0 1 13\n");
    const std::string ele_file = write_file(
        "zero.ele",
        "1 4 1\n"
        "0 0 1 2 3 77\n");
    const std::string face_file = write_file(
        "zero.face",
        "2 1\n"
        "0 1 2 3 31\n"
        "1 0 3 2 32\n");

    std::vector<Vec3> positions;
    std::vector<int> tets;
    std::vector<int> faces;
    read_tetgen_nodes(node_file, positions, true);
    read_tetgen_tets(ele_file, tets, true);
    read_tetgen_faces(face_file, faces, true);

    expect_positions_equal(
        positions,
        {
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
        });
    EXPECT_EQ(tets, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(faces, (std::vector<int>{1, 2, 3, 0, 3, 2}));
}

TEST_F(TetGenIOTest, RejectsMissingFilesWithoutChangingOutputs) {
    const std::string missing = (directory_ / "missing.tetgen").string();
    const std::vector<Vec3> original_positions{Vec3(2.0, 3.0, 4.0)};
    const std::vector<int> original_indices{7, 8, 9};

    std::vector<Vec3> positions = original_positions;
    EXPECT_THROW(
        read_tetgen_nodes(missing, positions),
        std::runtime_error);
    expect_positions_equal(positions, original_positions);

    std::vector<int> tets = original_indices;
    EXPECT_THROW(read_tetgen_tets(missing, tets), std::runtime_error);
    EXPECT_EQ(tets, original_indices);

    std::vector<int> faces = original_indices;
    EXPECT_THROW(read_tetgen_faces(missing, faces), std::runtime_error);
    EXPECT_EQ(faces, original_indices);
}

TEST_F(TetGenIOTest, RejectsMalformedHeaders) {
    const std::string bad_nodes = write_file(
        "bad_dimension.node",
        "1 2 0 0\n"
        "1 0 0\n");
    const std::string bad_tets = write_file(
        "bad_cardinality.ele",
        "1 10 0\n"
        "1 1 2 3 4 5 6 7 8 9 10\n");
    const std::string bad_faces = write_file(
        "bad_marker_header.face",
        "1 not-a-marker-count\n"
        "1 1 2 3\n");

    std::vector<Vec3> positions;
    std::vector<int> tets;
    std::vector<int> faces;
    EXPECT_THROW(
        read_tetgen_nodes(bad_nodes, positions),
        std::runtime_error);
    EXPECT_THROW(read_tetgen_tets(bad_tets, tets), std::runtime_error);
    EXPECT_THROW(read_tetgen_faces(bad_faces, faces), std::runtime_error);
}

TEST_F(TetGenIOTest, RejectsRecordIdMismatches) {
    const std::string bad_nodes = write_file(
        "bad_id.node",
        "1 3 0 0\n"
        "2 0 0 0\n");
    const std::string bad_tets = write_file(
        "bad_id.ele",
        "1 4 0\n"
        "2 1 2 3 4\n");
    const std::string bad_faces = write_file(
        "bad_id.face",
        "1 0\n"
        "2 1 2 3\n");

    std::vector<Vec3> positions;
    std::vector<int> tets;
    std::vector<int> faces;
    EXPECT_THROW(
        read_tetgen_nodes(bad_nodes, positions),
        std::runtime_error);
    EXPECT_THROW(read_tetgen_tets(bad_tets, tets), std::runtime_error);
    EXPECT_THROW(read_tetgen_faces(bad_faces, faces), std::runtime_error);
}

TEST_F(TetGenIOTest, MalformedRecordsLeaveOutputsUnchanged) {
    const std::string bad_nodes = write_file(
        "truncated.node",
        "2 3 0 0\n"
        "1 0 0 0\n"
        "2 1 0\n");
    const std::string bad_tets = write_file(
        "truncated.ele",
        "2 4 0\n"
        "1 1 2 3 4\n"
        "2 1 2 3\n");
    const std::string bad_faces = write_file(
        "truncated.face",
        "2 0\n"
        "1 1 2 3\n"
        "2 1 2\n");

    const std::vector<Vec3> original_positions{Vec3(-1.0, -2.0, -3.0)};
    const std::vector<int> original_indices{20, 21, 22, 23};

    std::vector<Vec3> positions = original_positions;
    EXPECT_THROW(
        read_tetgen_nodes(bad_nodes, positions),
        std::runtime_error);
    expect_positions_equal(positions, original_positions);

    std::vector<int> tets = original_indices;
    EXPECT_THROW(read_tetgen_tets(bad_tets, tets), std::runtime_error);
    EXPECT_EQ(tets, original_indices);

    std::vector<int> faces = original_indices;
    EXPECT_THROW(read_tetgen_faces(bad_faces, faces), std::runtime_error);
    EXPECT_EQ(faces, original_indices);
}

TEST_F(TetGenIOTest, LoadedUnitTetPassesMeshValidation) {
    const std::string node_file = write_file(
        "validated.node",
        "4 3 0 0\n"
        "1 0 0 0\n"
        "2 1 0 0\n"
        "3 0 1 0\n"
        "4 0 0 1\n");
    const std::string ele_file = write_file(
        "validated.ele",
        "1 4 0\n"
        "1 1 2 3 4\n");

    std::vector<Vec3> positions;
    std::vector<int> tets;
    read_tetgen_nodes(node_file, positions);
    read_tetgen_tets(ele_file, tets);

    EXPECT_NO_THROW(validate_tet_mesh(tets, positions));
}

TEST_F(TetGenIOTest, WritesObjTetGroupsWithOutwardTetFaceWinding) {
    const std::vector<Vec3> positions{
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.25, -2.5, 3.75),
        Vec3(-4.0, 5.5, 6.0),
        Vec3(0.125, 0.25, 0.5),
        Vec3(9.0, 8.0, 7.0),
    };
    const std::vector<int> tets{
        0, 1, 2, 3,
        4, 3, 2, 1,
    };
    const fs::path output = directory_ / "two_tets.obj";

    write_tet_mesh_obj(output.string(), positions, tets);

    // Each group preserves one tetrahedral cell for Houdini.  The four local
    // faces use mesh::TetFace exactly:
    //   (1,2,3), (0,3,2), (0,1,3), (0,2,1).
    EXPECT_EQ(
        read_file(output),
        "v 0 0 0\n"
        "v 1.25 -2.5 3.75\n"
        "v -4 5.5 6\n"
        "v 0.125 0.25 0.5\n"
        "v 9 8 7\n"
        "g tet_0\n"
        "f 2 3 4\n"
        "f 1 4 3\n"
        "f 1 2 4\n"
        "f 1 3 2\n"
        "g tet_1\n"
        "f 4 3 2\n"
        "f 5 2 3\n"
        "f 5 4 2\n"
        "f 5 3 4\n");
}

TEST_F(TetGenIOTest, ObjWriterRejectsInvalidTetMeshWithoutOverwritingOutput) {
    const std::vector<Vec3> positions{
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
    const fs::path output = directory_ / "preserve.obj";
    write_file("preserve.obj", "existing mesh\n");

    EXPECT_THROW(
        write_tet_mesh_obj(
            output.string(), positions, std::vector<int>{0, 1, 2}),
        std::invalid_argument);
    EXPECT_EQ(read_file(output), "existing mesh\n");

    EXPECT_THROW(
        write_tet_mesh_obj(
            output.string(), positions, std::vector<int>{-1, 1, 2, 3}),
        std::out_of_range);
    EXPECT_EQ(read_file(output), "existing mesh\n");

    EXPECT_THROW(
        write_tet_mesh_obj(
            output.string(), positions, std::vector<int>{0, 1, 2, 4}),
        std::out_of_range);
    EXPECT_EQ(read_file(output), "existing mesh\n");
}

TEST_F(TetGenIOTest, ObjWriterRejectsNonfinitePositionsWithoutOverwritingOutput) {
    const std::vector<int> tets{0, 1, 2, 3};
    const fs::path output = directory_ / "finite.obj";
    write_file("finite.obj", "existing mesh\n");

    std::vector<Vec3> positions{
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
    positions[1].x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        write_tet_mesh_obj(output.string(), positions, tets),
        std::invalid_argument);
    EXPECT_EQ(read_file(output), "existing mesh\n");

    positions[1].x() = 1.0;
    positions[2].z() = std::numeric_limits<double>::infinity();
    EXPECT_THROW(
        write_tet_mesh_obj(output.string(), positions, tets),
        std::invalid_argument);
    EXPECT_EQ(read_file(output), "existing mesh\n");
}

TEST_F(TetGenIOTest, ObjWriterReportsOutputOpenFailure) {
    const std::vector<Vec3> positions{
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
    const std::vector<int> tets{0, 1, 2, 3};
    const fs::path output = directory_ / "missing" / "mesh.obj";

    EXPECT_THROW(
        write_tet_mesh_obj(output.string(), positions, tets),
        std::runtime_error);
    EXPECT_FALSE(fs::exists(output));
}

} // namespace

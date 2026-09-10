#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "physics.h"
#include "output.h"
#include "grid_coloring.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <cmath>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Builds a small mesh, runs broad phase, and writes AABBs to OBJ files
// for visual inspection in Houdini. This test always passes.
// Output: OUTPUT_TEST_DIR/broad_phase_{node,tri,edge}_boxes.obj
TEST(OutputTest, ExportBroadPhaseBoxes) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 4, 4, 1.0, 1.0, Vec3::Zero());

    const int nv = static_cast<int>(state.deformed_positions.size());
    state.velocities.assign(nv, Vec3(0.0, -1.0, 0.0));

    const double dt   = 1.0 / 30.0;
    const double dhat = 0.05;

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, dt, dhat);

    const std::string dir = std::string(OUTPUT_TEST_DIR) + "/output_debug";
    fs::create_directories(dir);
    const BroadPhase::Cache& c = bp.cache();
    export_aabb_list(dir + "/broad_phase_node_boxes.obj", c.node_boxes);
    export_aabb_list(dir + "/broad_phase_tri_boxes.obj",  c.tri_boxes);
    export_aabb_list(dir + "/broad_phase_edge_boxes.obj", c.edge_boxes);

    export_obj(dir + "/mesh_before.obj", state.deformed_positions, ref_mesh.tris);

    std::vector<Vec3> x_after(nv);
    for (int i = 0; i < nv; ++i)
        x_after[i] = state.deformed_positions[i] + dt * state.velocities[i];
    export_obj(dir + "/mesh_after.obj", x_after, ref_mesh.tris);

    SUCCEED();
}

// Writes one OBJ per BVH depth level for each of the three BVHs (tri, edge, node).
// Output: OUTPUT_TEST_DIR/bvh_{tri,edge,node}_level_N.obj
// Load them in Houdini and step through N to walk the tree top-down.
TEST(OutputTest, ExportBVHLevels) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 4, 4, 1.0, 1.0, Vec3::Zero());

    const int nv = static_cast<int>(state.deformed_positions.size());
    state.velocities.assign(nv, Vec3(0.0, -1.0, 0.0));

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, 1.0 / 30.0, 0.05);

    const std::string dir = std::string(OUTPUT_TEST_DIR) + "/output_debug";
    fs::create_directories(dir);
    const BroadPhase::Cache& c = bp.cache();

    struct BVHDesc { const char* name; const std::vector<BVHNode>* nodes; int root; };
    const BVHDesc bvhs[] = {
        { "tri",  &c.tri_bvh_nodes,  c.tri_root  },
        { "edge", &c.edge_bvh_nodes, c.edge_root },
        { "node", &c.node_bvh_nodes, c.node_root },
    };

    for (const auto& bvh : bvhs) {
        for (int depth = 0; depth < 16; ++depth) {
            std::ostringstream path;
            path << dir << "/bvh_" << bvh.name << "_level_" << depth << ".obj";
            const int written = export_bvh_level(path.str(), *bvh.nodes, bvh.root, depth);
            if (written == 0) break;
        }
    }

    SUCCEED();
}

TEST(OutputTest, ParallelMeshExportsPreserveEveryByte) {
    struct RestoreThreads {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreThreads() {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "parallel_formats";
    for (const char* name : {"serial", "parallel", "nested"})
        fs::create_directories(dir / name);
    const auto read = [](const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open());
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    };
    struct Format { ExportFormat value; const char* extension; };
    const Format formats[] = {
        {ExportFormat::OBJ, ".obj"}, {ExportFormat::GEO, ".geo"},
        {ExportFormat::USD, ".usda"}, {ExportFormat::PLY, ".ply"}
    };
    // Exercise both text and binary thresholds, and incomplete final ranges.
    for (int count : {0, 1, 4095, 4096, 16385, 131071, 131072, 131073}) {
        SCOPED_TRACE(count);
        std::vector<Vec3> points(count);
        std::vector<int> triangles;
        std::vector<std::vector<int>> groups(5);
        for (int i = 0; i < count; ++i) {
            points[i] = Vec3(std::sin(0.07 * i), std::cos(0.09 * i), i * 0.00001);
            groups[i % 5].push_back(i);
        }
        if (count > 5) {
            points[0].x() = -0.0;
            points[1].y() = std::numeric_limits<double>::denorm_min();
            points[2].z() = std::numeric_limits<double>::max();
            points[3].x() = std::numeric_limits<double>::infinity();
            points[4].y() = -std::numeric_limits<double>::infinity();
            points[5].z() = std::numeric_limits<double>::quiet_NaN();
        }
        for (int i = 0; i + 2 < count; ++i)
            triangles.insert(triangles.end(), {i, i + 1, i + 2});
        for (const auto& format : formats) {
            SCOPED_TRACE(format.extension);
            // PLY stores float coordinates. Avoid a finite value outside
            // float's range while still checking its largest finite value.
            if (count > 5)
                points[2].z() = format.value == ExportFormat::PLY
                    ? std::numeric_limits<float>::max()
                    : std::numeric_limits<double>::max();
            const std::string filename = std::string("frame_0000") + format.extension;
            for (bool colored : {false, true}) {
                if (colored && format.value != ExportFormat::GEO) continue;
                SCOPED_TRACE(colored);
                const auto* colors = colored ? &groups : nullptr;
                omp_set_num_threads(1);
                export_frame((dir / "serial").string(), 0, points, triangles, format.value, colors);
                const std::string expected = read(dir / "serial" / filename);
                if (count != 0 || format.value != ExportFormat::OBJ)
                    ASSERT_FALSE(expected.empty());
                for (int threads : {8, 64}) {
                    SCOPED_TRACE(threads);
                    omp_set_num_threads(threads);
                    export_frame((dir / "parallel").string(), 0, points, triangles, format.value, colors);
                    EXPECT_TRUE(read(dir / "parallel" / filename) == expected);
                }
                if (count == 131073) {
                    // Exporters may be called inside a parallel region;
                    // their serial fallback must preserve the same bytes.
#pragma omp parallel num_threads(2)
                    {
#pragma omp single
                        export_frame((dir / "nested").string(), 0, points, triangles, format.value, colors);
                    }
                    EXPECT_TRUE(read(dir / "nested" / filename) == expected);
                }
            }
        }
    }
}

namespace {

std::string read_grid_geo(const fs::path& path) {
    std::ifstream input(path);
    EXPECT_TRUE(input.is_open());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

// Read the numeric tuples of one GEO attribute, respecting their array bounds.
// GEO uses alternating JSON key/value arrays, rather than JSON objects.
std::vector<double> grid_geo_attribute(const std::string& geo, const std::string& name) {
    const auto descriptor = geo.find("\"name\",\"" + name + "\"");
    if (descriptor == std::string::npos) {
        ADD_FAILURE() << "Missing GEO attribute " << name;
        return {};
    }
    const auto tuples = geo.find("\"tuples\",", descriptor);
    if (tuples == std::string::npos) {
        ADD_FAILURE() << "Missing GEO tuples for " << name;
        return {};
    }
    const auto begin = geo.find('[', tuples);
    std::size_t end = begin;
    int nesting = 0;
    for (; end < geo.size(); ++end) {
        if (geo[end] == '[') ++nesting;
        if (geo[end] == ']' && --nesting == 0) break;
    }
    if (end == geo.size()) {
        ADD_FAILURE() << "Unterminated GEO tuples for " << name;
        return {};
    }
    std::string numbers = geo.substr(begin, end - begin + 1);
    for (char& ch : numbers)
        if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
    std::istringstream input(numbers);
    std::vector<double> values;
    for (double value; input >> value;) values.push_back(value);
    return values;
}

int grid_geo_count(const std::string& geo, const std::string& key) {
    const auto begin = geo.find("\"" + key + "\",");
    if (begin == std::string::npos) {
        ADD_FAILURE() << "Missing GEO count " << key;
        return -1;
    }
    return std::stoi(geo.substr(begin + key.size() + 3));
}

solver_detail::ClothGridSchedule example_export_grid() {
    solver_detail::ClothGridSchedule grid;
    grid.dx = 0.5;
    grid.min_index = {{-1, 0, 0}};
    grid.max_index = {{2, 1, 0}};
    solver_detail::GridCell a, b;
    a.index = {{-1, 0, 0}};
    a.bounds = AABB(Vec3(-0.5, 0, 0), Vec3(0, 0.5, 0.5));
    a.vertices = {2, 0};
    a.color_id = 1;
    a.batch_id = 5;
    b.index = {{1, 1, 0}};
    b.bounds = AABB(Vec3(0.5, 0.5, 0), Vec3(1, 1, 0.5));
    b.vertices = {1};
    b.color_id = 3;
    b.batch_id = 7;
    grid.cells = {a, b};
    return grid;
}

}  // namespace

TEST(OutputTest, ClothGridGeoIncludesEmptyCellsAndSignedCoordinates) {
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "cloth_grid_output";
    fs::create_directories(dir);
    const auto grid = example_export_grid();
    export_cloth_grid_boxes_geo((dir / "grid_boxes.geo").string(), grid);
    const auto geo = read_grid_geo(dir / "grid_boxes.geo");
    EXPECT_EQ(grid_geo_count(geo, "pointcount"), 8 * 8);
    EXPECT_EQ(grid_geo_count(geo, "vertexcount"), 8 * 24);
    EXPECT_EQ(grid_geo_count(geo, "primitivecount"), 8 * 6);
    EXPECT_NE(geo.find("\"nvertices_rle\",[4,48]"), std::string::npos);

    const auto colors = grid_geo_attribute(geo, "color_id");
    const auto cells = grid_geo_attribute(geo, "cell_id");
    const auto batches = grid_geo_attribute(geo, "batch_id");
    const auto vertices = grid_geo_attribute(geo, "vertex_count");
    const auto grid_i = grid_geo_attribute(geo, "grid_i");
    const auto grid_j = grid_geo_attribute(geo, "grid_j");
    const auto grid_k = grid_geo_attribute(geo, "grid_k");
    const auto cd = grid_geo_attribute(geo, "Cd");
    const auto positions = grid_geo_attribute(geo, "P");
    ASSERT_EQ(colors.size(), 48u);
    ASSERT_EQ(cells.size(), 48u);
    ASSERT_EQ(batches.size(), 48u);
    ASSERT_EQ(vertices.size(), 48u);
    ASSERT_EQ(grid_i.size(), 48u);
    ASSERT_EQ(grid_j.size(), 48u);
    ASSERT_EQ(grid_k.size(), 48u);
    ASSERT_EQ(cd.size(), 48u * 3);
    ASSERT_EQ(positions.size(), 64u * 3);
    const int expected_colors[] = {1, 3, 0, 2, 1, 3, 0, 2};
    const double palette[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0.5, 0}};
    for (int cell = 0; cell < 8; ++cell) {
        const int occupied = cell == 0 ? 0 : cell == 5 ? 1 : -1;
        for (int face = 0; face < 6; ++face) {
            const int p = 6 * cell + face;
            EXPECT_EQ(colors[p], expected_colors[cell]);
            EXPECT_EQ(cells[p], occupied);
            EXPECT_EQ(batches[p], occupied == 0 ? 5 : occupied == 1 ? 7 : -1);
            EXPECT_EQ(vertices[p], occupied == 0 ? 2 : occupied == 1 ? 1 : 0);
            EXPECT_EQ(grid_i[p], cell / 2 - 1);
            EXPECT_EQ(grid_j[p], cell % 2);
            EXPECT_EQ(grid_k[p], 0);
            for (int axis = 0; axis < 3; ++axis)
                EXPECT_DOUBLE_EQ(cd[p * 3 + axis], palette[expected_colors[cell]][axis]);
        }
        EXPECT_DOUBLE_EQ(positions[cell * 24], 0.5 * (cell / 2 - 1));
        EXPECT_DOUBLE_EQ(positions[cell * 24 + 1], 0.5 * (cell % 2));
        EXPECT_DOUBLE_EQ(positions[cell * 24 + 2], 0);
        EXPECT_DOUBLE_EQ(positions[cell * 24 + 3], 0.5 * (cell / 2));
    }
}

TEST(OutputTest, ClothGridVerticesGeoPreservesActualMembership) {
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "cloth_grid_output";
    fs::create_directories(dir);
    const auto grid = example_export_grid();
    const std::vector<Vec3> positions = {
        Vec3(-0.2, 0.2, 0.2), Vec3(0.7, 0.7, 0.2),
        Vec3(-0.1, 0.3, 0.3), Vec3(0, 0, 0)
    };
    export_cloth_grid_vertices_geo((dir / "grid_vertices.geo").string(), positions, grid);
    const auto geo = read_grid_geo(dir / "grid_vertices.geo");
    EXPECT_EQ(grid_geo_count(geo, "pointcount"), 4);
    EXPECT_EQ(grid_geo_count(geo, "vertexcount"), 0);
    EXPECT_EQ(grid_geo_count(geo, "primitivecount"), 0);
    EXPECT_EQ(grid_geo_attribute(geo, "color_id"), (std::vector<double>{1, 3, 1, -1}));
    EXPECT_EQ(grid_geo_attribute(geo, "cell_id"), (std::vector<double>{0, 1, 0, -1}));
    EXPECT_EQ(grid_geo_attribute(geo, "batch_id"), (std::vector<double>{5, 7, 5, -1}));
    EXPECT_EQ(grid_geo_attribute(geo, "Cd"), (std::vector<double>{
        0, 1, 0, 1, 0.5, 0, 0, 1, 0, 0.5, 0.5, 0.5}));
    const auto p = grid_geo_attribute(geo, "P");
    ASSERT_EQ(p.size(), positions.size() * 3);
    for (std::size_t i = 0; i < positions.size(); ++i)
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_DOUBLE_EQ(p[3 * i + axis], positions[i][axis]);

    auto invalid = grid;
    invalid.cells[1].vertices.push_back(0);
    EXPECT_THROW(export_cloth_grid_vertices_geo((dir / "invalid.geo").string(), positions, invalid), std::invalid_argument);
}

TEST(OutputTest, ClothGridGeoUsesEightParityColorsInThreeDimensions) {
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "cloth_grid_output";
    fs::create_directories(dir);
    solver_detail::ClothGridSchedule grid;
    grid.dx = 1;
    grid.min_index = {{0, 0, 0}};
    grid.max_index = {{1, 1, 1}};
    solver_detail::GridCell cell;
    cell.index = {{0, 0, 0}};
    cell.color_id = 0;
    cell.batch_id = 0;
    cell.vertices = {0};
    grid.cells = {cell};
    export_cloth_grid_boxes_geo((dir / "grid_3d.geo").string(), grid);
    const auto geo = read_grid_geo(dir / "grid_3d.geo");
    const auto colors = grid_geo_attribute(geo, "color_id");
    const auto cd = grid_geo_attribute(geo, "Cd");
    ASSERT_EQ(colors.size(), 48u);
    ASSERT_EQ(cd.size(), 144u);
    const int expected[] = {0, 4, 2, 6, 1, 5, 3, 7};
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(colors[6 * i], expected[i]);
        for (int j = 0; j < i; ++j) {
            EXPECT_FALSE(cd[18 * i] == cd[18 * j]
                      && cd[18 * i + 1] == cd[18 * j + 1]
                      && cd[18 * i + 2] == cd[18 * j + 2]);
        }
    }
}

TEST(OutputTest, ClothGridGeoRejectsExcessiveGeometryAndHandlesEmptyScenes) {
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "cloth_grid_output";
    fs::create_directories(dir);
    const auto path = (dir / "invalid_grid.geo").string();
    auto grid = example_export_grid();
    grid.min_index = {{std::numeric_limits<std::int64_t>::min(), 0, 0}};
    grid.max_index = {{std::numeric_limits<std::int64_t>::max(), 0, 0}};
    EXPECT_THROW(export_cloth_grid_boxes_geo(path, grid), std::length_error);
    grid = example_export_grid();
    grid.max_index = {{250000, 1, 0}};
    EXPECT_THROW(export_cloth_grid_boxes_geo(path, grid), std::length_error);
    grid.dx = 0;
    EXPECT_THROW(export_cloth_grid_boxes_geo(path, grid), std::invalid_argument);
    grid.dx = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(export_cloth_grid_boxes_geo(path, grid), std::invalid_argument);

    grid = example_export_grid();
    grid.cells.clear();
    export_cloth_grid_boxes_geo((dir / "empty_grid.geo").string(), grid);
    const auto geo = read_grid_geo(dir / "empty_grid.geo");
    EXPECT_EQ(grid_geo_count(geo, "pointcount"), 0);
    EXPECT_EQ(grid_geo_count(geo, "primitivecount"), 0);
    EXPECT_TRUE(grid_geo_attribute(geo, "P").empty());
    EXPECT_TRUE(grid_geo_attribute(geo, "color_id").empty());
}

TEST(OutputTest, ClothGridSubstepsKeepExistingFilesAndAddOptionalDiagnostics) {
    const fs::path dir = fs::path(OUTPUT_TEST_DIR) / "cloth_grid_substeps";
    const std::vector<Vec3> positions = {
        Vec3(-0.2, 0.2, 0.2), Vec3(0.7, 0.7, 0.2), Vec3(-0.1, 0.3, 0.3)
    };
    const auto grid = example_export_grid();
    const SimParams params{};
    const BroadPhase broad_phase;
    RefMesh mesh;
    mesh.tris = {0, 1, 2};
    const std::vector<std::vector<int>> groups{{0, 2}, {1}};
    for (bool with_grid : {false, true}) {
        const fs::path output = dir / (with_grid ? "enabled" : "disabled");
        write_substep_data(params, broad_phase, positions, output.string(), &mesh,
                           &groups, with_grid ? &grid : nullptr);
        bool found_substep = false;
        for (const auto& substep : fs::directory_iterator(output)) {
            if (!substep.is_directory()) continue;
            found_substep = true;
            for (const char* filename : {"mesh.geo", "nt_pairs.geo", "ss_pairs.geo",
                                         "barrier_distances.txt", "barrier_stats.txt"})
                EXPECT_TRUE(fs::exists(substep.path() / filename));
            EXPECT_EQ(fs::exists(substep.path() / "grid_boxes.geo"), with_grid);
            EXPECT_EQ(fs::exists(substep.path() / "grid_vertices.geo"), with_grid);
            EXPECT_EQ(grid_geo_attribute(read_grid_geo(substep.path() / "mesh.geo"), "group_id"),
                      (std::vector<double>{0, 1, 0}));
        }
        EXPECT_TRUE(found_substep);
    }
}

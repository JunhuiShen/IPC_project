#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "physics.h"
#include "output.h"

#include <gtest/gtest.h>
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

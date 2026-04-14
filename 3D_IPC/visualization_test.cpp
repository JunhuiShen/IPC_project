#include "broad_phase.h"
#include "make_shape.h"
#include "physics.h"
#include "visualization.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Builds a small mesh, runs broad phase, and writes AABBs to OBJ files
// for visual inspection in Houdini. This test always passes.
// Output: VIS_OUTPUT_DIR/broad_phase_{node,tri,edge}_boxes.obj
TEST(VisualizationTest, ExportBroadPhaseBoxes) {
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

    const std::string dir = std::string(VIS_OUTPUT_DIR) + "/vis_debug";
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
// Output: VIS_OUTPUT_DIR/bvh_{tri,edge,node}_level_N.obj
// Load them in Houdini and step through N to walk the tree top-down.
TEST(VisualizationTest, ExportBVHLevels) {
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

    const std::string dir = std::string(VIS_OUTPUT_DIR) + "/vis_debug";
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

#include "make_shape.h"
#include "example.h"
#include "mesh_utils.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

namespace {

void expect_flat_horizontal_polygon_prism(
    const std::vector<Vec3>& positions, int base, int side_count,
    const Vec3& center, double radius, double thickness) {
    constexpr double tolerance = 1.0e-14;
    const double bottom_y = center.y() - 0.5 * thickness;
    const double top_y = center.y() + 0.5 * thickness;

    ASSERT_GE(base, 0);
    ASSERT_LE(
        base + 2 * side_count,
        static_cast<int>(positions.size()));

    // Material vertex zero starts on +x. Keeping it there rules out yaw.
    const Vec3& first_bottom = positions[static_cast<std::size_t>(base)];
    EXPECT_NEAR(first_bottom.x(), center.x() + radius, tolerance);
    EXPECT_NEAR(first_bottom.z(), center.z(), tolerance);

    // Both cap rings must be horizontal, and their corresponding vertices
    // must differ only in world y. These checks rule out either tilt axis.
    for (int local = 0; local < side_count; ++local) {
        SCOPED_TRACE(local);
        const Vec3& bottom = positions[static_cast<std::size_t>(base + local)];
        const Vec3& top = positions[
            static_cast<std::size_t>(base + side_count + local)];
        EXPECT_NEAR(bottom.y(), bottom_y, tolerance);
        EXPECT_NEAR(top.y(), top_y, tolerance);
        EXPECT_NEAR(bottom.x(), top.x(), tolerance);
        EXPECT_NEAR(bottom.z(), top.z(), tolerance);
        EXPECT_NEAR(
            std::hypot(bottom.x() - center.x(), bottom.z() - center.z()),
            radius, tolerance);
    }
}

} // namespace

TEST(BuildIncidentTriangleMap, BasicExample) {
// [0,1,2, 1,2,5] -- two triangles
// New format: {tri_idx, local_node_index}
std::vector<int> indices = {0, 1, 2, 1, 2, 5};
auto map = build_incident_triangle_map(indices);

EXPECT_EQ(map[0], (std::vector<std::pair<int,int>>{{0, 0}}));
EXPECT_EQ(map[1], (std::vector<std::pair<int,int>>{{0, 1}, {1, 0}}));
EXPECT_EQ(map[2], (std::vector<std::pair<int,int>>{{0, 2}, {1, 1}}));
EXPECT_EQ(map[5], (std::vector<std::pair<int,int>>{{1, 2}}));
EXPECT_EQ(map.size(), 4u);
}
TEST(BuildIncidentTriangleMap, EmptyInput) {
std::vector<int> indices = {};
auto map = build_incident_triangle_map(indices);
EXPECT_TRUE(map.empty());
}

TEST(MixedExample, FiftyMixedRigidPolygonsAboveFourCornerPinnedCloth) {
    IPCArgs3D args;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_fifty_rigid_polygons_drop_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 100;
    constexpr int cloth_nz = 100;
    constexpr int cloth_vertices =
        (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int rigid_body_count = 50;
    constexpr int rigid_vertices = 750;
    constexpr int rigid_triangles = 1300;
    constexpr int total_vertices = cloth_vertices + rigid_vertices;
    constexpr int total_triangles = cloth_triangles + rigid_triangles;

    EXPECT_EQ(state.deformed_positions.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(state.velocities.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.num_positions,
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.node_to_rb.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.mass.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.tris.size(),
              static_cast<std::size_t>(3 * total_triangles));
    EXPECT_EQ(X.size(), static_cast<std::size_t>(cloth_vertices));
    EXPECT_EQ(ref_mesh.Dm_inverse.size(),
              static_cast<std::size_t>(cloth_triangles));
    EXPECT_EQ(ref_mesh.area.size(),
              static_cast<std::size_t>(cloth_triangles));
    ASSERT_EQ(pins.size(), 4u);
    EXPECT_EQ(pins[0].vertex_index, 0);
    EXPECT_EQ(pins[1].vertex_index, cloth_nx);
    EXPECT_EQ(pins[2].vertex_index,
              cloth_nz * (cloth_nx + 1));
    EXPECT_EQ(pins[3].vertex_index, cloth_vertices - 1);

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);
    EXPECT_TRUE(std::all_of(
        state.velocities.begin(), state.velocities.end(),
        [](const Vec3& velocity) { return velocity.isZero(0.0); }));
    EXPECT_TRUE(std::all_of(
        state.v_coms.begin(), state.v_coms.end(),
        [](const Vec3& velocity) { return velocity.isZero(0.0); }));

    for (int node = 0; node < cloth_vertices; ++node)
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);

    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            const int node = ref_mesh.tris[3 * triangle + corner];
            EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        }
    }

    int triangle_cursor = cloth_triangles;
    for (int rb = 0; rb < rigid_body_count; ++rb) {
        const int side_count = 3 + (rb % 10);
        const std::size_t expected_nodes =
            static_cast<std::size_t>(2 * side_count);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), expected_nodes);
        ASSERT_EQ(ref_mesh.ref_positions[rb].size(), expected_nodes);
        EXPECT_GT(ref_mesh.total_mass[rb], 0.0);
        EXPECT_NEAR(state.orientations[rb].norm(), 1.0, 1.0e-12);
        EXPECT_TRUE(state.v_coms[rb].isZero(1.0e-12));
        EXPECT_TRUE(state.omega[rb].isZero(1.0e-12));

        for (const int node : ref_mesh.rb_nodes[rb]) {
            EXPECT_EQ(ref_mesh.node_to_rb[node], static_cast<int>(rb));
            EXPECT_GT(ref_mesh.mass[node], 0.0);

            const Vec3& position = state.deformed_positions[node];
            EXPECT_GT(position.y(), 1.2 + params.d_hat);
            EXPECT_GE(position.x(), -2.0);
            EXPECT_LE(position.x(), 2.0);
            EXPECT_GE(position.z(), -2.0);
            EXPECT_LE(position.z(), 2.0);
        }

        const int body_triangles = 4 * side_count - 4;
        for (int local_triangle = 0;
             local_triangle < body_triangles;
             ++local_triangle, ++triangle_cursor) {
            for (int corner = 0; corner < 3; ++corner) {
                const int node =
                    ref_mesh.tris[3 * triangle_cursor + corner];
                EXPECT_EQ(
                    ref_mesh.node_to_rb[node], static_cast<int>(rb));
            }
        }
    }
    EXPECT_EQ(triangle_cursor, total_triangles);

    for (const Hinge& hinge : ref_mesh.hinges) {
        for (const int node : hinge.v) {
            EXPECT_GE(node, 0);
            EXPECT_LT(node, cloth_vertices);
        }
    }

    const std::vector<double> masses_before = ref_mesh.mass;
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);

    double total_cloth_mass = 0.0;
    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        total_cloth_mass += ref_mesh.mass[node];
    }
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    for (const std::vector<int>& body_nodes : ref_mesh.rb_nodes) {
        for (const int node : body_nodes)
            EXPECT_DOUBLE_EQ(ref_mesh.mass[node], masses_before[node]);
    }
}

TEST(MixedExample,
     TenSmallRigidAndTenLargerDeformablePolygonsAbovePinnedCloth) {
    IPCArgs3D args;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_twenty_rigid_deformable_polygons_drop_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 100;
    constexpr int cloth_nz = 100;
    constexpr int cloth_vertices =
        (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int rigid_body_count = 10;
    // Each class contains one polygon with every side count from 3 through
    // 12, for 75 sides per class. A solid adds one interior star vertex.
    constexpr int rigid_vertices = 150;
    constexpr int rigid_triangles = 260;
    constexpr int solid_surface_vertices = 150;
    constexpr int solid_interior_vertices = 10;
    constexpr int solid_vertices =
        solid_surface_vertices + solid_interior_vertices;
    constexpr int solid_tetrahedra = 260;
    constexpr int total_vertices =
        cloth_vertices + rigid_vertices + solid_vertices;
    constexpr int total_triangles =
        cloth_triangles + rigid_triangles + solid_tetrahedra;

    EXPECT_EQ(state.deformed_positions.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(state.velocities.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.num_positions,
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.node_to_rb.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.mass.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.tris.size(),
              static_cast<std::size_t>(3 * total_triangles));
    EXPECT_EQ(ref_mesh.tets.size(),
              static_cast<std::size_t>(4 * solid_tetrahedra));
    EXPECT_EQ(ref_mesh.tet_nodes.size(),
              static_cast<std::size_t>(solid_vertices));
    EXPECT_EQ(ref_mesh.surface_nodes.size(),
              static_cast<std::size_t>(solid_surface_vertices));
    EXPECT_EQ(ref_mesh.deformable_nodes.size(),
              static_cast<std::size_t>(cloth_vertices + solid_vertices));
    EXPECT_EQ(X.size(), static_cast<std::size_t>(cloth_vertices));
    EXPECT_EQ(ref_mesh.Dm_inverse.size(),
              static_cast<std::size_t>(cloth_triangles));
    EXPECT_EQ(ref_mesh.area.size(),
              static_cast<std::size_t>(cloth_triangles));

    ASSERT_EQ(
        pins.size(), static_cast<std::size_t>(2 * (cloth_nz + 1)));
    for (int j = 0; j <= cloth_nz; ++j) {
        SCOPED_TRACE(j);
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j)].vertex_index,
            j * (cloth_nx + 1));
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j + 1)].vertex_index,
            j * (cloth_nx + 1) + cloth_nx);
    }

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);

    constexpr double kPi = 3.14159265358979323846;
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);

    std::vector<unsigned char> is_rigid(total_vertices, 0);
    std::vector<unsigned char> is_solid(total_vertices, 0);
    std::vector<unsigned char> is_solid_surface(total_vertices, 0);
    for (int rb = 0; rb < rigid_body_count; ++rb) {
        const int side_count =
            static_cast<int>(ref_mesh.rb_nodes[rb].size()) / 2;
        EXPECT_GE(side_count, 3);
        EXPECT_LE(side_count, 12);
        EXPECT_TRUE(state.orientations[rb].isApprox(
            flat_orientation, 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
        EXPECT_TRUE(state.omega[rb].isZero(0.0));
        EXPECT_GT(ref_mesh.total_mass[rb], 0.0);
        for (const int node : ref_mesh.rb_nodes[rb]) {
            ASSERT_GE(node, cloth_vertices);
            ASSERT_LT(node, total_vertices);
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            is_rigid[node] = 1;
        }
    }
    for (const int node : ref_mesh.tet_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        is_solid[node] = 1;
    }
    for (const int node : ref_mesh.surface_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_solid[node], 1);
        is_solid_surface[node] = 1;
    }

    EXPECT_EQ(std::count(is_rigid.begin(), is_rigid.end(), 1),
              rigid_vertices);
    EXPECT_EQ(std::count(is_solid.begin(), is_solid.end(), 1),
              solid_vertices);
    int interior_count = 0;
    for (int node = 0; node < total_vertices; ++node) {
        EXPECT_FALSE(is_rigid[node] && is_solid[node]);
        if (node < cloth_vertices) {
            EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
            EXPECT_EQ(is_rigid[node], 0);
            EXPECT_EQ(is_solid[node], 0);
        } else {
            EXPECT_TRUE(is_rigid[node] != 0 || is_solid[node] != 0);
        }
        interior_count +=
            is_solid[node] != 0 && is_solid_surface[node] == 0;
    }
    EXPECT_EQ(interior_count, solid_interior_vertices);

    for (const int node : ref_mesh.tets)
        EXPECT_EQ(is_solid[node], 1);
    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner)
            EXPECT_LT(ref_mesh.tris[3 * triangle + corner], cloth_vertices);
    }
    for (int triangle = cloth_triangles;
         triangle < total_triangles; ++triangle) {
        const int v0 = ref_mesh.tris[3 * triangle];
        const bool triangle_is_rigid = is_rigid[v0] != 0;
        for (int corner = 0; corner < 3; ++corner) {
            const int node = ref_mesh.tris[3 * triangle + corner];
            EXPECT_EQ(is_rigid[node] != 0, triangle_is_rigid);
            EXPECT_EQ(is_solid[node] != 0, !triangle_is_rigid);
            if (triangle_is_rigid)
                EXPECT_EQ(ref_mesh.node_to_rb[node], ref_mesh.node_to_rb[v0]);
            else
                EXPECT_EQ(is_solid_surface[node], 1);
        }
    }

    // Reconstruct the procedural object order to verify that both classes
    // cover every side count and that the solids are visibly larger.
    std::vector<Vec3> object_centers;
    std::vector<double> object_radii;
    std::vector<bool> rigid_side_count_seen(10, false);
    std::vector<bool> solid_side_count_seen(10, false);
    int node_cursor = cloth_vertices;
    int rigid_cursor = 0;
    double maximum_rigid_radius = 0.0;
    double minimum_solid_radius = std::numeric_limits<double>::infinity();
    for (int polygon = 0; polygon < 20; ++polygon) {
        const int side_count = 3 + polygon / 2;
        const bool object_is_rigid = polygon % 2 == 0;
        const int node_count = object_is_rigid
            ? 2 * side_count : 2 * side_count + 1;
        const Vec3 center = object_is_rigid
            ? state.x_coms[rigid_cursor]
            : state.deformed_positions[node_cursor + 2 * side_count];

        const Vec3& first_bottom = state.deformed_positions[node_cursor];
        const Vec3& first_top =
            state.deformed_positions[node_cursor + side_count];
        const double cap_radius = std::hypot(
            first_bottom.x() - center.x(),
            first_bottom.z() - center.z());
        const double cap_thickness = first_top.y() - first_bottom.y();
        ASSERT_GT(cap_radius, 0.0);
        ASSERT_GT(cap_thickness, 0.0);
        expect_flat_horizontal_polygon_prism(
            state.deformed_positions, node_cursor, side_count,
            center, cap_radius, cap_thickness);

        if (object_is_rigid) {
            ASSERT_EQ(ref_mesh.rb_nodes[rigid_cursor].front(), node_cursor);
            rigid_side_count_seen[side_count - 3] = true;
            ++rigid_cursor;
        } else {
            solid_side_count_seen[side_count - 3] = true;
        }

        double object_radius = 0.0;
        for (int local_node = 0; local_node < node_count; ++local_node) {
            object_radius = std::max(
                object_radius,
                (state.deformed_positions[node_cursor + local_node]
                 - center).norm());
        }
        if (object_is_rigid)
            maximum_rigid_radius = std::max(maximum_rigid_radius, object_radius);
        else
            minimum_solid_radius = std::min(minimum_solid_radius, object_radius);
        object_centers.push_back(center);
        object_radii.push_back(object_radius);
        node_cursor += node_count;
    }
    EXPECT_EQ(node_cursor, total_vertices);
    EXPECT_EQ(rigid_cursor, rigid_body_count);
    EXPECT_TRUE(std::all_of(
        rigid_side_count_seen.begin(), rigid_side_count_seen.end(),
        [](const bool seen) { return seen; }));
    EXPECT_TRUE(std::all_of(
        solid_side_count_seen.begin(), solid_side_count_seen.end(),
        [](const bool seen) { return seen; }));
    EXPECT_GT(minimum_solid_radius, 1.5 * maximum_rigid_radius);

    ASSERT_EQ(object_centers.size(), 20U);
    ASSERT_EQ(object_radii.size(), object_centers.size());
    for (std::size_t first = 0; first < object_centers.size(); ++first) {
        EXPECT_GT(
            object_centers[first].y() - object_radii[first],
            1.2 + params.d_hat);
        for (std::size_t second = first + 1;
             second < object_centers.size(); ++second) {
            EXPECT_GT(
                (object_centers[first] - object_centers[second]).norm(),
                object_radii[first] + object_radii[second]
                    + params.d_hat);
        }
    }

    // The shell mass pass fills cloth nodes and preserves solid and rigid
    // masses that their respective object builders already assigned.
    const std::vector<double> object_masses = ref_mesh.mass;
    for (int node = cloth_vertices; node < total_vertices; ++node)
        EXPECT_GT(object_masses[node], 0.0);
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);

    double total_cloth_mass = 0.0;
    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        total_cloth_mass += ref_mesh.mass[node];
    }
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    for (int node = cloth_vertices; node < total_vertices; ++node)
        EXPECT_DOUBLE_EQ(ref_mesh.mass[node], object_masses[node]);
}

TEST(MixedExample, SingleDeformableSolidAboveOppositeEdgePinnedCloth) {
    IPCArgs3D args;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_single_deformable_solid_drop_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 30;
    constexpr int cloth_nz = 30;
    constexpr int cloth_vertices =
        (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int side_count = 8;
    constexpr int solid_vertices = 2 * side_count + 1;
    constexpr int solid_surface_vertices = 2 * side_count;
    constexpr int solid_tetrahedra = 4 * side_count - 4;
    constexpr int total_vertices = cloth_vertices + solid_vertices;
    constexpr int total_triangles =
        cloth_triangles + solid_tetrahedra;
    constexpr double cloth_height = 1.2;
    constexpr double radius = 0.30;
    constexpr double thickness = 0.20;

    EXPECT_EQ(state.deformed_positions.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(state.velocities.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.num_positions,
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.tris.size(),
              static_cast<std::size_t>(3 * total_triangles));
    EXPECT_EQ(ref_mesh.tets.size(),
              static_cast<std::size_t>(4 * solid_tetrahedra));
    EXPECT_EQ(ref_mesh.tet_rest_data.size(),
              static_cast<std::size_t>(solid_tetrahedra));
    EXPECT_EQ(ref_mesh.tet_nodes.size(),
              static_cast<std::size_t>(solid_vertices));
    EXPECT_EQ(ref_mesh.surface_nodes.size(),
              static_cast<std::size_t>(solid_surface_vertices));
    EXPECT_EQ(ref_mesh.deformable_nodes.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.node_to_rb.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.mass.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(X.size(), static_cast<std::size_t>(cloth_vertices));
    EXPECT_EQ(ref_mesh.Dm_inverse.size(),
              static_cast<std::size_t>(cloth_triangles));
    EXPECT_EQ(ref_mesh.area.size(),
              static_cast<std::size_t>(cloth_triangles));

    ASSERT_EQ(
        pins.size(), static_cast<std::size_t>(2 * (cloth_nz + 1)));
    for (int j = 0; j <= cloth_nz; ++j) {
        SCOPED_TRACE(j);
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j)].vertex_index,
            j * (cloth_nx + 1));
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j + 1)].vertex_index,
            j * (cloth_nx + 1) + cloth_nx);
    }

    EXPECT_TRUE(ref_mesh.rb_nodes.empty());
    EXPECT_TRUE(ref_mesh.ref_positions.empty());
    EXPECT_TRUE(ref_mesh.total_mass.empty());
    EXPECT_TRUE(ref_mesh.I_hat.empty());
    EXPECT_TRUE(state.x_coms.empty());
    EXPECT_TRUE(state.v_coms.empty());
    EXPECT_TRUE(state.orientations.empty());
    EXPECT_TRUE(state.omega.empty());
    EXPECT_TRUE(std::all_of(
        ref_mesh.node_to_rb.begin(), ref_mesh.node_to_rb.end(),
        [](const int owner) { return owner == -1; }));

    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_DOUBLE_EQ(state.deformed_positions[node].y(), cloth_height);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
    }
    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            EXPECT_LT(
                ref_mesh.tris[3 * triangle + corner], cloth_vertices);
        }
    }

    std::vector<unsigned char> is_solid_node(total_vertices, 0);
    for (const int node : ref_mesh.tet_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_solid_node[static_cast<std::size_t>(node)], 0);
        is_solid_node[static_cast<std::size_t>(node)] = 1;
    }
    std::vector<unsigned char> is_solid_surface(total_vertices, 0);
    for (const int node : ref_mesh.surface_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, total_vertices);
        is_solid_surface[static_cast<std::size_t>(node)] = 1;
    }
    for (int solid = 0; solid < solid_vertices; ++solid) {
        const int node = cloth_vertices + solid;
        EXPECT_EQ(is_solid_node[static_cast<std::size_t>(node)], 1);
        EXPECT_GT(ref_mesh.mass[static_cast<std::size_t>(node)], 0.0);
        EXPECT_TRUE(state.velocities[static_cast<std::size_t>(node)].isApprox(
            Vec3(0.0, -0.75, 0.0), 0.0));
        EXPECT_GT(
            state.deformed_positions[static_cast<std::size_t>(node)].y(),
            cloth_height + params.d_hat);
    }
    const int solid_interior = total_vertices - 1;
    EXPECT_TRUE(state.deformed_positions[solid_interior].isApprox(
        Vec3(0.0, 1.5, 0.0), 1.0e-14));
    EXPECT_EQ(is_solid_surface[solid_interior], 0);
    expect_flat_horizontal_polygon_prism(
        state.deformed_positions, cloth_vertices, side_count,
        Vec3(0.0, 1.5, 0.0), radius, thickness);

    for (const int node : ref_mesh.tets) {
        EXPECT_GE(node, cloth_vertices);
        EXPECT_LT(node, total_vertices);
    }
    for (int triangle = cloth_triangles;
         triangle < total_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            const int node = ref_mesh.tris[3 * triangle + corner];
            ASSERT_GE(node, cloth_vertices);
            ASSERT_LT(node, total_vertices);
            EXPECT_EQ(is_solid_surface[static_cast<std::size_t>(node)], 1);
        }
    }

    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());

    const double expected_solid_volume =
        0.5 * side_count * radius * radius
        * std::sin(2.0 * std::acos(-1.0) / side_count) * thickness;
    double total_solid_mass = 0.0;
    for (int node = cloth_vertices; node < total_vertices; ++node)
        total_solid_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
    EXPECT_NEAR(
        total_solid_mass,
        args.solid_density * expected_solid_volume, 1.0e-11);

    const std::vector<double> solid_masses(
        ref_mesh.mass.begin() + cloth_vertices, ref_mesh.mass.end());
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    const double total_cloth_mass = std::accumulate(
        ref_mesh.mass.begin(),
        ref_mesh.mass.begin() + cloth_vertices, 0.0);
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    EXPECT_TRUE(std::equal(
        solid_masses.begin(), solid_masses.end(),
        ref_mesh.mass.begin() + cloth_vertices));
}

TEST(MixedExample,
     FourLevelBunnySpotCubeGearRowsAboveOppositeEdgePinnedCloth) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_four_bunny_spot_cube_gear_rows_scene_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);

    struct WorkingDirectoryGuard {
        fs::path previous;
        fs::path temporary;

        explicit WorkingDirectoryGuard(fs::path path)
            : previous(fs::current_path()), temporary(std::move(path)) {
            fs::current_path(temporary);
        }

        ~WorkingDirectoryGuard() {
            std::error_code error;
            fs::current_path(previous, error);
            fs::remove_all(temporary, error);
        }
    } working_directory(directory);

    // Exercise the production repository-relative asset paths with compact,
    // distinct fixtures. Bunny has two tetrahedra while Spot has one, so path
    // reuse and accidental object-type changes are both observable.
    fs::create_directories("example_obj/bunny_coarse");
    fs::create_directories("example_obj/spot");
    {
        std::ofstream nodes(
            "example_obj/bunny_coarse/bunny_2000f.1.node");
        ASSERT_TRUE(nodes.good());
        nodes << "5 3 0 0\n"
              << "0 0 0 0\n"
              << "1 1 0 0\n"
              << "2 0 0.98996474061363637 0\n"
              << "3 0 0 0.01\n"
              << "4 0 0 -0.7530728972409091\n";
    }
    {
        std::ofstream elements(
            "example_obj/bunny_coarse/bunny_2000f.1.ele");
        ASSERT_TRUE(elements.good());
        elements << "2 4 0\n"
                 << "0 0 1 2 3\n"
                 << "1 0 2 1 4\n";
    }
    {
        std::ofstream nodes("example_obj/spot/spot_2000f.1.node");
        ASSERT_TRUE(nodes.good());
        nodes << "4 3 0 0\n"
              << "0 0 0 0\n"
              << "1 0.54916355263863637 0 0\n"
              << "2 0 0.98558895052954543 0\n"
              << "3 0 0 1\n";
    }
    {
        std::ofstream elements("example_obj/spot/spot_2000f.1.ele");
        ASSERT_TRUE(elements.good());
        elements << "1 4 0\n"
                 << "0 0 1 2 3\n";
    }
    {
        std::ofstream gear("example_obj/gear_z18_coarse.obj");
        ASSERT_TRUE(gear.good());
        // Closed, outward-oriented anisotropic octahedron. Its unequal AABB
        // extents expose any rotation even if the quaternion stays normalized.
        gear << "v  0.5  0  0\n"
             << "v -0.5  0  0\n"
             << "v  0  0.49536426517656252  0\n"
             << "v  0 -0.49536426517656252  0\n"
             << "v  0  0  0.10005811135\n"
             << "v  0  0 -0.10005811135\n"
             << "f 1 3 5\n"
             << "f 3 2 5\n"
             << "f 2 4 5\n"
             << "f 4 1 5\n"
             << "f 3 1 6\n"
             << "f 2 3 6\n"
             << "f 4 2 6\n"
             << "f 1 4 6\n";
    }

    IPCArgs3D args;
    args.solid_density = 731.0;
    args.rigid_density = 947.0;
    // This deliberately exceeds half the fixture's shortest normalized edge,
    // forcing the scene builder's production-mesh safety clamp to execute.
    args.d_hat = 0.1;
    args.k_barrier = 617.0;
    args.gx = 1.25;
    args.gy = -3.5;
    args.gz = 2.25;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_four_bunny_spot_cube_gear_rows_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 30;
    constexpr int cloth_nz = 30;
    constexpr int cloth_vertices = (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int rows = 4;
    constexpr int bunny_vertices = 5;
    constexpr int bunny_tetrahedra = 2;
    constexpr int bunny_surface_triangles = 6;
    constexpr int spot_vertices = 4;
    constexpr int spot_tetrahedra = 1;
    constexpr int spot_surface_triangles = 4;
    constexpr int cube_vertices = 8;
    constexpr int cube_triangles = 12;
    constexpr int gear_vertices = 6;
    constexpr int gear_triangles = 8;
    constexpr int solid_vertices =
        rows * (bunny_vertices + spot_vertices);
    constexpr int solid_tetrahedra =
        rows * (bunny_tetrahedra + spot_tetrahedra);
    constexpr int rigid_body_count = 2 * rows;
    constexpr int total_vertices = cloth_vertices + solid_vertices
        + rows * (cube_vertices + gear_vertices);
    constexpr int total_triangles = cloth_triangles
        + rows
            * (bunny_surface_triangles + spot_surface_triangles
               + cube_triangles + gear_triangles);
    constexpr double cloth_height = 1.2;
    constexpr double object_center_y = 1.75;
    constexpr double solid_max_extent = 0.44;
    constexpr double rigid_max_extent = 0.22;
    constexpr double row_spacing = 0.45;
    enum BodyType {
        Bunny = 0, Spot = 1, Cube = 2, Gear = 3, BodyTypeCount = 4};
    static constexpr int expected_row_order[rows][BodyTypeCount] = {
        {Bunny, Cube, Gear, Spot},
        {Gear, Spot, Bunny, Cube},
        {Spot, Gear, Cube, Bunny},
        {Cube, Bunny, Spot, Gear},
    };
    // Literal expected centers keep this regression independent from the
    // production packing implementation.
    static constexpr double expected_x[rows][BodyTypeCount] = {
        {-0.35581598158033034,  0.455,
         -0.015815981580330304, 0.21418401841966972},
        { 0.12581598158033028, -0.225,
          0.46581598158033027, -0.46581598158033033},
        { 0.35581598158033034, -0.455,
          0.015815981580330304, -0.21418401841966972},
        {-0.12581598158033033,  0.225,
         -0.46581598158033033,  0.46581598158033027},
    };
    const Vec3 bunny_extents(
        0.44, 0.435584485870, 0.335752074786);
    const Vec3 spot_extents(
        0.241631963161, 0.433659138233, 0.44);
    const Vec3 cube_extents = Vec3::Constant(rigid_max_extent);
    const Vec3 gear_extents(
        0.22, 0.2179602766776875, 0.044025568994);
    const Vec3 drop_velocity(0.0, -0.75, 0.0);
    struct ObjectBounds {
        Vec3 lower;
        Vec3 upper;
    };
    std::array<ObjectBounds, rows * 4> object_bounds;

    static_assert(total_vertices == 1053);
    static_assert(total_triangles == 1920);
    static_assert(solid_tetrahedra == 12);

    ASSERT_EQ(state.deformed_positions.size(), total_vertices);
    ASSERT_EQ(state.velocities.size(), total_vertices);
    EXPECT_EQ(ref_mesh.num_positions, total_vertices);
    ASSERT_EQ(ref_mesh.mass.size(), total_vertices);
    ASSERT_EQ(ref_mesh.node_to_rb.size(), total_vertices);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_EQ(ref_mesh.tets.size(), 4 * solid_tetrahedra);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), solid_tetrahedra);
    EXPECT_EQ(ref_mesh.tet_nodes.size(), solid_vertices);
    EXPECT_EQ(ref_mesh.surface_nodes.size(), solid_vertices);
    EXPECT_EQ(
        ref_mesh.deformable_nodes.size(), cloth_vertices + solid_vertices);
    EXPECT_EQ(X.size(), cloth_vertices);
    EXPECT_EQ(ref_mesh.Dm_inverse.size(), cloth_triangles);
    EXPECT_EQ(ref_mesh.area.size(), cloth_triangles);

    ASSERT_EQ(pins.size(), 2 * (cloth_nz + 1));
    for (int j = 0; j <= cloth_nz; ++j) {
        SCOPED_TRACE(j);
        const int left = j * (cloth_nx + 1);
        const int right = left + cloth_nx;
        const Pin& left_pin = pins[static_cast<std::size_t>(2 * j)];
        const Pin& right_pin = pins[static_cast<std::size_t>(2 * j + 1)];
        EXPECT_EQ(left_pin.vertex_index, left);
        EXPECT_EQ(right_pin.vertex_index, right);
        EXPECT_TRUE(left_pin.target_position.isApprox(
            state.deformed_positions[left], 0.0));
        EXPECT_TRUE(right_pin.target_position.isApprox(
            state.deformed_positions[right], 0.0));
    }

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.rb_update_modes.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);

    std::vector<unsigned char> is_tet_node(total_vertices, 0);
    std::vector<unsigned char> is_surface_node(total_vertices, 0);
    std::vector<unsigned char> is_deformable(total_vertices, 0);
    for (const int node : ref_mesh.tet_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, cloth_vertices + solid_vertices);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
        is_tet_node[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : ref_mesh.surface_nodes) {
        ASSERT_GE(node, cloth_vertices);
        ASSERT_LT(node, cloth_vertices + solid_vertices);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
        is_surface_node[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : ref_mesh.deformable_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
        is_deformable[static_cast<std::size_t>(node)] = 1;
    }

    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_DOUBLE_EQ(state.deformed_positions[node].y(), cloth_height);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
    }
    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int node = ref_mesh.tris[3 * triangle + local];
            EXPECT_GE(node, 0);
            EXPECT_LT(node, cloth_vertices);
        }
    }
    for (const Hinge& hinge : ref_mesh.hinges) {
        for (const int node : hinge.v) {
            EXPECT_GE(node, 0);
            EXPECT_LT(node, cloth_vertices);
        }
    }

    const auto row_z = [=](const int row) {
        return -1.5 * row_spacing
            + static_cast<double>(row) * row_spacing;
    };
    const auto object_center = [&](const int row, const int type) {
        return Vec3(
            expected_x[row][type], object_center_y, row_z(row));
    };
    const auto column_for_type = [](const int row, const int type) {
        for (int column = 0; column < BodyTypeCount; ++column) {
            if (expected_row_order[row][column] == type)
                return column;
        }
        return -1;
    };
    const auto check_solid =
        [&](const int node_base, const int node_count,
            const int first_tet, const int tet_count,
            const Vec3& expected_center, const Vec3& expected_extents,
            const double expected_mass) {
            const int node_end = node_base + node_count;
            Vec3 lower = state.deformed_positions[node_base];
            Vec3 upper = lower;
            double actual_mass = 0.0;
            for (int node = node_base; node < node_end; ++node) {
                lower = lower.cwiseMin(state.deformed_positions[node]);
                upper = upper.cwiseMax(state.deformed_positions[node]);
                actual_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
                EXPECT_GT(ref_mesh.mass[static_cast<std::size_t>(node)], 0.0);
                EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
                EXPECT_TRUE(state.velocities[node].isApprox(
                    drop_velocity, 0.0));
                EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 1);
                EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 1);
                EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
            }
            EXPECT_TRUE((0.5 * (lower + upper)).isApprox(
                expected_center, 1.0e-14));
            EXPECT_TRUE((upper - lower).isApprox(
                expected_extents, 1.0e-14));
            EXPECT_GT(lower.y(), cloth_height + params.d_hat);
            EXPECT_NEAR(actual_mass, expected_mass, 1.0e-12);

            for (int element = first_tet;
                 element < first_tet + tet_count; ++element) {
                EXPECT_GT(ref_mesh.tet_rest_data[element].measure, 0.0);
                for (int local = 0; local < 4; ++local) {
                    const int node = ref_mesh.tets[4 * element + local];
                    EXPECT_GE(node, node_base);
                    EXPECT_LT(node, node_end);
                }
            }
            return ObjectBounds{lower, upper};
        };

    // The two fixture tetrahedra together occupy the axis tetrahedron defined
    // by the production Bunny AABB dimensions.
    const double expected_bunny_mass =
        args.solid_density * bunny_extents.prod() / 6.0;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("bunny row " + std::to_string(row));
        const int column = column_for_type(row, Bunny);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            check_solid(
            cloth_vertices + row * bunny_vertices,
            bunny_vertices, row * bunny_tetrahedra, bunny_tetrahedra,
            object_center(row, Bunny), bunny_extents,
            expected_bunny_mass);
    }

    const double expected_spot_mass =
        args.solid_density * spot_extents.prod() / 6.0;
    constexpr int spot_node_base =
        cloth_vertices + rows * bunny_vertices;
    constexpr int spot_tet_base = rows * bunny_tetrahedra;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("spot row " + std::to_string(row));
        const int column = column_for_type(row, Spot);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            check_solid(
            spot_node_base + row * spot_vertices,
            spot_vertices, spot_tet_base + row * spot_tetrahedra,
            spot_tetrahedra, object_center(row, Spot), spot_extents,
            expected_spot_mass);
    }

    const double expected_cube_mass = args.rigid_density
        * rigid_max_extent * rigid_max_extent * rigid_max_extent;
    constexpr int first_rigid_node = cloth_vertices + solid_vertices;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("cube row " + std::to_string(row));
        const int rb = row;
        const int expected_node_base =
            first_rigid_node + row * cube_vertices;
        const Vec3 expected_center = object_center(row, Cube);
        EXPECT_TRUE(state.x_coms[rb].isApprox(expected_center, 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isApprox(drop_velocity, 0.0));
        EXPECT_TRUE(state.omega[rb].isZero(0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            Vec4(1.0, 0.0, 0.0, 0.0), 0.0));
        EXPECT_EQ(
            ref_mesh.rb_update_modes[rb],
            RigidBodyUpdateMode::TranslationAndOrientation);
        EXPECT_NEAR(ref_mesh.total_mass[rb], expected_cube_mass, 1.0e-12);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), cube_vertices);
        EXPECT_EQ(ref_mesh.rb_nodes[rb].front(), expected_node_base);

        Vec3 lower = state.deformed_positions[expected_node_base];
        Vec3 upper = lower;
        double nodal_mass = 0.0;
        for (const int node : ref_mesh.rb_nodes[rb]) {
            lower = lower.cwiseMin(state.deformed_positions[node]);
            upper = upper.cwiseMax(state.deformed_positions[node]);
            nodal_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
            EXPECT_TRUE(state.velocities[node].isApprox(drop_velocity, 0.0));
        }
        EXPECT_TRUE((0.5 * (lower + upper)).isApprox(
            expected_center, 1.0e-14));
        EXPECT_TRUE((upper - lower).isApprox(
            cube_extents, 1.0e-14));
        EXPECT_GT(lower.y(), cloth_height + params.d_hat);
        EXPECT_NEAR(nodal_mass, ref_mesh.total_mass[rb], 1.0e-12);
        const int column = column_for_type(row, Cube);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            ObjectBounds{lower, upper};
    }

    // An axis-aligned octahedron occupies one sixth of its AABB volume.
    const double expected_gear_mass =
        args.rigid_density * gear_extents.prod() / 6.0;
    constexpr int first_gear_node =
        first_rigid_node + rows * cube_vertices;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("gear row " + std::to_string(row));
        const int rb = rows + row;
        const int expected_node_base = first_gear_node + row * gear_vertices;
        const Vec3 expected_center = object_center(row, Gear);
        EXPECT_TRUE(state.x_coms[rb].isApprox(expected_center, 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isApprox(drop_velocity, 0.0));
        EXPECT_TRUE(state.omega[rb].isZero(0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            Vec4(1.0, 0.0, 0.0, 0.0), 0.0));
        EXPECT_EQ(
            ref_mesh.rb_update_modes[rb],
            RigidBodyUpdateMode::TranslationAndOrientation);
        EXPECT_NEAR(ref_mesh.total_mass[rb], expected_gear_mass, 1.0e-12);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), gear_vertices);
        EXPECT_EQ(ref_mesh.rb_nodes[rb].front(), expected_node_base);

        Vec3 lower = state.deformed_positions[expected_node_base];
        Vec3 upper = lower;
        double nodal_mass = 0.0;
        for (const int node : ref_mesh.rb_nodes[rb]) {
            lower = lower.cwiseMin(state.deformed_positions[node]);
            upper = upper.cwiseMax(state.deformed_positions[node]);
            nodal_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
            EXPECT_TRUE(state.velocities[node].isApprox(drop_velocity, 0.0));
        }
        EXPECT_TRUE((0.5 * (lower + upper)).isApprox(
            expected_center, 1.0e-14));
        // Identity orientation keeps the anisotropic source-axis extents on
        // world x:y:z; a tilted gear would permute these dimensions.
        EXPECT_TRUE((upper - lower).isApprox(
            gear_extents,
            1.0e-14));
        EXPECT_GT(lower.y(), cloth_height + params.d_hat);
        EXPECT_NEAR(nodal_mass, ref_mesh.total_mass[rb], 1.0e-12);
        const int column = column_for_type(row, Gear);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            ObjectBounds{lower, upper};
    }

    // Every row contains Bunny, Spot, cube, gear across x, and all sixteen
    // object centers share exactly one height level.
    for (int row = 0; row < rows; ++row) {
        EXPECT_DOUBLE_EQ(state.x_coms[row].y(), object_center_y);
        EXPECT_DOUBLE_EQ(state.x_coms[rows + row].y(), object_center_y);
        EXPECT_DOUBLE_EQ(state.x_coms[row].z(), row_z(row));
        EXPECT_DOUBLE_EQ(
            state.x_coms[rows + row].z(), row_z(row));

        std::array<int, BodyTypeCount> type_counts{};
        for (int column = 0; column < BodyTypeCount; ++column) {
            const int type = expected_row_order[row][column];
            ASSERT_GE(type, 0);
            ASSERT_LT(type, BodyTypeCount);
            ++type_counts[static_cast<std::size_t>(type)];
            const ObjectBounds& bounds = object_bounds[
                static_cast<std::size_t>(4 * row + column)];
            const Vec3 actual_center = 0.5 * (bounds.lower + bounds.upper);
            EXPECT_NEAR(actual_center.x(), expected_x[row][type], 1.0e-14);
            EXPECT_NEAR(actual_center.y(), object_center_y, 1.0e-14);
            EXPECT_NEAR(actual_center.z(), row_z(row), 1.0e-14);
        }
        for (const int count : type_counts)
            EXPECT_EQ(count, 1);
        for (int previous = 0; previous < row; ++previous) {
            EXPECT_FALSE(std::equal(
                expected_row_order[row],
                expected_row_order[row] + BodyTypeCount,
                expected_row_order[previous]));
        }
    }

    // The tighter placement is intentional, but every production-shaped
    // fixture AABB must still be disjoint before the first solve. Checking all
    // 120 pairs catches both a spacing regression and a wrong asset rotation.
    const auto aabb_gap = [](const ObjectBounds& first,
                             const ObjectBounds& second) {
        Vec3 axis_gap = Vec3::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            axis_gap[axis] = std::max(
                {0.0,
                 first.lower[axis] - second.upper[axis],
                 second.lower[axis] - first.upper[axis]});
        }
        return axis_gap.norm();
    };
    double minimum_object_gap = std::numeric_limits<double>::infinity();
    for (int first = 0; first < rows * 4; ++first) {
        for (int second = first + 1; second < rows * 4; ++second) {
            SCOPED_TRACE(
                "object pair " + std::to_string(first) + ","
                + std::to_string(second));
            const double gap = aabb_gap(
                object_bounds[static_cast<std::size_t>(first)],
                object_bounds[static_cast<std::size_t>(second)]);
            EXPECT_GT(gap, 0.0);
            EXPECT_GT(gap, params.d_hat);
            minimum_object_gap = std::min(minimum_object_gap, gap);
        }
    }
    constexpr double expected_tight_gap = 0.01;
    EXPECT_NEAR(minimum_object_gap, expected_tight_gap, 5.0e-13);

    // Type-aware packing leaves exactly 10 mm between each consecutive pair,
    // independent of that row's permutation.
    const std::array<double, 3> expected_column_gaps = {
        expected_tight_gap, expected_tight_gap, 0.01};
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < 3; ++column) {
            EXPECT_NEAR(
                aabb_gap(
                    object_bounds[static_cast<std::size_t>(4 * row + column)],
                    object_bounds[
                        static_cast<std::size_t>(4 * row + column + 1)]),
                expected_column_gaps[static_cast<std::size_t>(column)],
                5.0e-13);
        }
    }
    double minimum_cloth_gap = std::numeric_limits<double>::infinity();
    for (int object = 0; object < rows * 4; ++object) {
        const double gap =
            object_bounds[static_cast<std::size_t>(object)].lower.y()
            - cloth_height;
        EXPECT_GT(gap, params.d_hat);
        minimum_cloth_gap = std::min(minimum_cloth_gap, gap);
    }
    constexpr double expected_cloth_gap = 0.332207757065;
    EXPECT_NEAR(minimum_cloth_gap, expected_cloth_gap, 1.0e-13);

    // Surface topology follows the same type-major append order as the node
    // storage. Every collision triangle must stay inside its source object.
    const auto expect_triangle_range =
        [&](const int first_triangle, const int triangle_count,
            const int node_base, const int node_count) {
            for (int triangle = first_triangle;
                 triangle < first_triangle + triangle_count; ++triangle) {
                for (int local = 0; local < 3; ++local) {
                    const int node = ref_mesh.tris[3 * triangle + local];
                    EXPECT_GE(node, node_base);
                    EXPECT_LT(node, node_base + node_count);
                }
            }
        };
    int triangle_cursor = cloth_triangles;
    for (int row = 0; row < rows; ++row) {
        expect_triangle_range(
            triangle_cursor, bunny_surface_triangles,
            cloth_vertices + row * bunny_vertices, bunny_vertices);
        triangle_cursor += bunny_surface_triangles;
    }
    for (int row = 0; row < rows; ++row) {
        expect_triangle_range(
            triangle_cursor, spot_surface_triangles,
            spot_node_base + row * spot_vertices, spot_vertices);
        triangle_cursor += spot_surface_triangles;
    }
    for (int row = 0; row < rows; ++row) {
        expect_triangle_range(
            triangle_cursor, cube_triangles,
            first_rigid_node + row * cube_vertices, cube_vertices);
        triangle_cursor += cube_triangles;
    }
    for (int row = 0; row < rows; ++row) {
        expect_triangle_range(
            triangle_cursor, gear_triangles,
            first_gear_node + row * gear_vertices, gear_vertices);
        triangle_cursor += gear_triangles;
    }
    EXPECT_EQ(triangle_cursor, total_triangles);

    EXPECT_TRUE(params.gravity.isApprox(
        Vec3(args.gx, args.gy, args.gz), 0.0));
    EXPECT_DOUBLE_EQ(params.k_barrier, args.k_barrier);
    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int first = ref_mesh.tris[3 * triangle + local];
            const int second =
                ref_mesh.tris[3 * triangle + (local + 1) % 3];
            minimum_surface_edge = std::min(
                minimum_surface_edge,
                (state.deformed_positions[second]
                 - state.deformed_positions[first])
                    .norm());
        }
    }
    EXPECT_LT(params.d_hat, args.d_hat);
    EXPECT_NEAR(
        params.d_hat,
        std::min(args.d_hat, 0.45 * minimum_surface_edge), 1.0e-15);

    const std::vector<double> object_masses(
        ref_mesh.mass.begin() + cloth_vertices, ref_mesh.mass.end());
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    EXPECT_NEAR(
        std::accumulate(
            ref_mesh.mass.begin(),
            ref_mesh.mass.begin() + cloth_vertices, 0.0),
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    EXPECT_TRUE(std::equal(
        object_masses.begin(), object_masses.end(),
        ref_mesh.mass.begin() + cloth_vertices));
}

TEST(SolidExample, DensityNineHundredOctagonalPrismAboveGround) {
    IPCArgs3D args;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;

    build_single_deformable_solid_ground_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    constexpr int side_count = 8;
    constexpr int solid_nodes = 2 * side_count + 1;
    constexpr int boundary_nodes = 2 * side_count;
    constexpr int tetrahedra = 4 * side_count - 4;
    constexpr double radius = 0.22;
    constexpr double thickness = 0.16;
    constexpr double density = 900.0;

    EXPECT_EQ(state.deformed_positions.size(), solid_nodes);
    EXPECT_EQ(state.velocities.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.num_positions, solid_nodes);
    EXPECT_EQ(ref_mesh.tets.size(), 4 * tetrahedra);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), tetrahedra);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * tetrahedra);
    EXPECT_EQ(ref_mesh.tet_nodes.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.surface_nodes.size(), boundary_nodes);
    EXPECT_EQ(ref_mesh.deformable_nodes.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.mass.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.node_to_rb.size(), solid_nodes);
    EXPECT_TRUE(ref_mesh.rb_nodes.empty());
    EXPECT_TRUE(ref_mesh.total_mass.empty());
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());

    for (int node = 0; node < solid_nodes; ++node) {
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
        EXPECT_GT(state.deformed_positions[node].y(), 0.0);
    }
    EXPECT_TRUE(state.deformed_positions.back().isApprox(
        Vec3(0.0, 1.0, 0.0), 1.0e-14));
    expect_flat_horizontal_polygon_prism(
        state.deformed_positions, 0, side_count,
        Vec3(0.0, 1.0, 0.0), radius, thickness);
    EXPECT_EQ(
        std::count(
            ref_mesh.surface_nodes.begin(), ref_mesh.surface_nodes.end(),
            solid_nodes - 1),
        0);

    const double expected_volume =
        0.5 * side_count * radius * radius
        * std::sin(2.0 * std::acos(-1.0) / side_count) * thickness;
    const double total_mass = std::accumulate(
        ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0);
    EXPECT_NEAR(total_mass, density * expected_volume, 1.0e-11);
    EXPECT_DOUBLE_EQ(args.solid_density, density);

    EXPECT_DOUBLE_EQ(
        params.solid_mu,
        args.solid_E / (2.0 * (1.0 + args.solid_nu)));
    EXPECT_DOUBLE_EQ(
        params.solid_lambda,
        args.solid_E * args.solid_nu
            / ((1.0 + args.solid_nu)
               * (1.0 - 2.0 * args.solid_nu)));
    ASSERT_EQ(params.sdf_planes.size(), 1U);
    EXPECT_TRUE(params.sdf_planes[0].point.isZero(0.0));
    EXPECT_TRUE(
        params.sdf_planes[0].normal.isApprox(Vec3::UnitY(), 0.0));

    ASSERT_EQ(static_x.size(), 4U);
    EXPECT_EQ(static_tris, (std::vector<int>{0, 1, 2, 0, 2, 3}));
    for (const Vec3& vertex : static_x)
        EXPECT_DOUBLE_EQ(vertex.y(), 0.0);
}

TEST(SolidExample, UsesSolidMaterialCommandLineOverrides) {
    IPCArgs3D args;
    char program[] = "make_shape_test";
    char solid_E_key[] = "--solid_E";
    char solid_E_value[] = "24000";
    char solid_nu_key[] = "--solid_nu";
    char solid_nu_value[] = "0.2";
    char solid_density_key[] = "--solid_density";
    char solid_density_value[] = "750";
    char* argv[] = {
        program,
        solid_E_key, solid_E_value,
        solid_nu_key, solid_nu_value,
        solid_density_key, solid_density_value,
    };
    ASSERT_TRUE(args.parse(7, argv));

    // Give the cloth fields deliberately unrelated values. Example 13 must
    // use only the solid-specific material arguments below.
    args.E = 123.0;
    args.nu = 0.1;
    args.density = 17.0;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;

    build_single_deformable_solid_ground_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    constexpr int side_count = 8;
    constexpr double radius = 0.22;
    constexpr double thickness = 0.16;
    const double expected_volume =
        0.5 * side_count * radius * radius
        * std::sin(2.0 * std::acos(-1.0) / side_count) * thickness;
    const double total_mass = std::accumulate(
        ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0);
    EXPECT_NEAR(
        total_mass, args.solid_density * expected_volume, 1.0e-11);
    EXPECT_DOUBLE_EQ(
        params.solid_mu,
        args.solid_E / (2.0 * (1.0 + args.solid_nu)));
    EXPECT_DOUBLE_EQ(
        params.solid_lambda,
        args.solid_E * args.solid_nu
            / ((1.0 + args.solid_nu)
               * (1.0 - 2.0 * args.solid_nu)));
    EXPECT_NE(params.solid_mu, params.mu);
    EXPECT_NE(params.solid_lambda, params.lambda);
}

TEST(MaterialArguments, SeparateDefaultsAndDensityOverrides) {
    IPCArgs3D defaults;
    EXPECT_DOUBLE_EQ(defaults.solid_E, 5.0e4);
    EXPECT_DOUBLE_EQ(defaults.solid_nu, 0.45);
    EXPECT_DOUBLE_EQ(defaults.solid_density, 900.0);
    EXPECT_DOUBLE_EQ(
        defaults.to_sim_params().solid_mu,
        defaults.solid_E / (2.0 * (1.0 + defaults.solid_nu)));
    EXPECT_DOUBLE_EQ(
        defaults.to_sim_params().solid_lambda,
        defaults.solid_E * defaults.solid_nu
            / ((1.0 + defaults.solid_nu)
               * (1.0 - 2.0 * defaults.solid_nu)));
    EXPECT_DOUBLE_EQ(defaults.rigid_density, 900.0);
    EXPECT_DOUBLE_EQ(defaults.to_sim_params().rigid_density, 900.0);
    EXPECT_FALSE(defaults.verbose);
    EXPECT_FALSE(defaults.to_sim_params().verbose);

    IPCArgs3D args;
    char program[] = "make_shape_test";
    char rigid_density_key[] = "--rigid_density";
    char rigid_density_value[] = "1234";
    char solid_density_key[] = "--solid_density";
    char solid_density_value[] = "567";
    char shell_density_key[] = "--density";
    char shell_density_value[] = "18";
    char verbose_key[] = "--verbose";
    char* argv[] = {
        program,
        rigid_density_key, rigid_density_value,
        solid_density_key, solid_density_value,
        shell_density_key, shell_density_value,
        verbose_key,
    };
    ASSERT_TRUE(args.parse(8, argv));

    EXPECT_DOUBLE_EQ(args.rigid_density, 1234.0);
    EXPECT_DOUBLE_EQ(args.solid_density, 567.0);
    EXPECT_DOUBLE_EQ(args.density, 18.0);
    EXPECT_TRUE(args.verbose);

    const SimParams params = args.to_sim_params();
    EXPECT_DOUBLE_EQ(params.rigid_density, 1234.0);
    EXPECT_DOUBLE_EQ(params.solid_density, 567.0);
    EXPECT_DOUBLE_EQ(params.density, 18.0);
    EXPECT_TRUE(params.verbose);
}

TEST(MixedExample, RigidAndSolidDensitiesAreIndependent) {
    struct ObjectMasses {
        std::vector<double> rigid;
        double solid = 0.0;
    };

    const auto build_object_masses = [](const double rigid_density,
                                         const double solid_density) {
        IPCArgs3D args;
        args.rigid_density = rigid_density;
        args.solid_density = solid_density;

        RefMesh ref_mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params = args.to_sim_params();
        build_twenty_rigid_deformable_polygons_drop_on_pinned_cloth_example(
            args, ref_mesh, state, X, pins, params);

        ObjectMasses masses;
        masses.rigid = ref_mesh.total_mass;
        for (const int node : ref_mesh.tet_nodes)
            masses.solid += ref_mesh.mass[node];
        return masses;
    };

    const ObjectMasses baseline = build_object_masses(450.0, 600.0);
    const ObjectMasses denser_rigid =
        build_object_masses(1350.0, 600.0);
    const ObjectMasses denser_solid =
        build_object_masses(450.0, 1200.0);

    ASSERT_EQ(baseline.rigid.size(), 10U);
    ASSERT_EQ(denser_rigid.rigid.size(), baseline.rigid.size());
    ASSERT_EQ(denser_solid.rigid.size(), baseline.rigid.size());
    for (std::size_t rb = 0; rb < baseline.rigid.size(); ++rb) {
        EXPECT_GT(baseline.rigid[rb], 0.0);
        EXPECT_NEAR(
            denser_rigid.rigid[rb], 3.0 * baseline.rigid[rb], 1.0e-12);
        EXPECT_DOUBLE_EQ(denser_solid.rigid[rb], baseline.rigid[rb]);
    }

    EXPECT_GT(baseline.solid, 0.0);
    EXPECT_DOUBLE_EQ(denser_rigid.solid, baseline.solid);
    EXPECT_NEAR(denser_solid.solid, 2.0 * baseline.solid, 1.0e-12);
}

// ---------------------------------------------------------------------------
// append_deformable_polygon_prism tests
// ---------------------------------------------------------------------------

TEST(AppendDeformablePolygonPrism,
     CountsOrientationBoundaryAndVolumeForThreeThroughTwelveSides) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double radius = 0.37;
    constexpr double thickness = 0.21;
    constexpr double density = 7.3;

    for (int sides = 3; sides <= 12; ++sides) {
        SCOPED_TRACE(sides);
        RefMesh ref_mesh;
        DeformedState state;

        const int base = append_deformable_polygon_prism(
            sides, state, ref_mesh, Vec3::Zero(), radius, density,
            thickness);
        const int expected_nodes = 2 * sides + 1;
        const int expected_tets = 4 * sides - 4;

        EXPECT_EQ(base, 0);
        EXPECT_EQ(state.deformed_positions.size(),
                  static_cast<std::size_t>(expected_nodes));
        EXPECT_EQ(state.velocities.size(),
                  static_cast<std::size_t>(expected_nodes));
        EXPECT_EQ(ref_mesh.tets.size(),
                  static_cast<std::size_t>(4 * expected_tets));
        EXPECT_EQ(ref_mesh.tet_rest_data.size(),
                  static_cast<std::size_t>(expected_tets));
        EXPECT_EQ(ref_mesh.tris.size(),
                  static_cast<std::size_t>(3 * expected_tets));
        EXPECT_EQ(ref_mesh.tet_nodes.size(),
                  static_cast<std::size_t>(expected_nodes));
        EXPECT_EQ(ref_mesh.surface_nodes.size(),
                  static_cast<std::size_t>(2 * sides));

        const int interior = 2 * sides;
        EXPECT_TRUE(state.deformed_positions[interior].isZero(0.0));
        EXPECT_EQ(std::count(ref_mesh.surface_nodes.begin(),
                             ref_mesh.surface_nodes.end(), interior),
                  0);

        double volume = 0.0;
        for (const TetRestData& rest : ref_mesh.tet_rest_data) {
            EXPECT_GT(rest.measure, 0.0);
            EXPECT_TRUE(rest.Dm_inverse.allFinite());
            volume += rest.measure;
        }
        const double expected_volume =
            0.5 * static_cast<double>(sides) * radius * radius
            * std::sin(2.0 * kPi / static_cast<double>(sides))
            * thickness;
        EXPECT_NEAR(volume, expected_volume, 1.0e-13);
        EXPECT_NEAR(
            std::accumulate(
                ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0),
            density * expected_volume, 1.0e-12);

        // create_solid extracts precisely the original outward surface.
        for (int triangle = 0;
             triangle < static_cast<int>(ref_mesh.tris.size() / 3);
             ++triangle) {
            const Vec3& a = state.deformed_positions[
                ref_mesh.tris[3 * triangle]];
            const Vec3& b = state.deformed_positions[
                ref_mesh.tris[3 * triangle + 1]];
            const Vec3& c = state.deformed_positions[
                ref_mesh.tris[3 * triangle + 2]];
            const Vec3 normal = (b - a).cross(c - a);
            EXPECT_GT(normal.dot((a + b + c) / 3.0), 0.0);
        }
    }
}

TEST(AppendDeformablePolygonPrism,
     AppliesNormalizedQuaternionAndRemapsWhenAppending) {
    RefMesh ref_mesh;
    DeformedState state;
    const int first_base = append_deformable_polygon_prism(
        3, state, ref_mesh, Vec3(-2.0, 0.0, 0.0),
        0.2, 4.0, 0.1);
    const std::size_t first_tet_entries = ref_mesh.tets.size();

    const Vec3 center(1.0, -0.5, 2.0);
    const double radius = 0.4;
    const double thickness = 0.3;
    const Vec4 orientation(2.0, -1.0, 3.0, 0.5);
    const Vec4 q = quaternion_normalize(orientation);
    const int second_base = append_deformable_polygon_prism(
        7, state, ref_mesh, center, radius, 5.0, thickness,
        orientation);

    EXPECT_EQ(first_base, 0);
    EXPECT_EQ(second_base, 7);
    const Vec3 expected_bottom = center + quaternion_rotate(
        q, Vec3(radius, 0.0, -0.5 * thickness));
    const Vec3 expected_top = center + quaternion_rotate(
        q, Vec3(radius, 0.0, 0.5 * thickness));
    EXPECT_TRUE(state.deformed_positions[second_base].isApprox(
        expected_bottom, 1.0e-14));
    EXPECT_TRUE(state.deformed_positions[second_base + 7].isApprox(
        expected_top, 1.0e-14));
    EXPECT_TRUE(state.deformed_positions[second_base + 14].isApprox(
        center, 0.0));

    for (std::size_t occurrence = first_tet_entries;
         occurrence < ref_mesh.tets.size(); ++occurrence) {
        EXPECT_GE(ref_mesh.tets[occurrence], second_base);
        EXPECT_LT(ref_mesh.tets[occurrence], second_base + 15);
    }
}

TEST(AppendDeformablePolygonPrism, RejectsInvalidInputsTransactionally) {
    RefMesh ref_mesh;
    DeformedState state;
    const auto expect_empty = [&]() {
        EXPECT_TRUE(state.deformed_positions.empty());
        EXPECT_TRUE(state.velocities.empty());
        EXPECT_TRUE(ref_mesh.tets.empty());
        EXPECT_TRUE(ref_mesh.tris.empty());
        EXPECT_TRUE(ref_mesh.mass.empty());
    };

    EXPECT_THROW(
        append_deformable_polygon_prism(
            2, state, ref_mesh, Vec3::Zero(), 1.0, 1.0, 1.0),
        std::invalid_argument);
    expect_empty();
    EXPECT_THROW(
        append_deformable_polygon_prism(
            3, state, ref_mesh, Vec3::Zero(), 0.0, 1.0, 1.0),
        std::invalid_argument);
    expect_empty();
    EXPECT_THROW(
        append_deformable_polygon_prism(
            3, state, ref_mesh, Vec3::Zero(), 1.0, 0.0, 1.0),
        std::invalid_argument);
    expect_empty();
    EXPECT_THROW(
        append_deformable_polygon_prism(
            3, state, ref_mesh, Vec3::Zero(), 1.0, 1.0, 0.0),
        std::invalid_argument);
    expect_empty();
    EXPECT_THROW(
        append_deformable_polygon_prism(
            3, state, ref_mesh, Vec3::Zero(), 1.0, 1.0, 1.0,
            Vec4::Zero()),
        std::invalid_argument);
    expect_empty();
    EXPECT_THROW(
        append_deformable_polygon_prism(
            3, state, ref_mesh,
            Vec3(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
            1.0, 1.0, 1.0),
        std::invalid_argument);
    expect_empty();
}

TEST(AppendNormalizedTetGenSolid,
     RecentersUniformlyScalesAndRepairsNegativeOrientation) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_normalized_tetgen_solid_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);
    const fs::path node_file = directory / "shape.node";
    const fs::path element_file = directory / "shape.ele";
    {
        std::ofstream output(node_file);
        ASSERT_TRUE(output.good());
        output << "4 3 0 0\n"
               << "0 -2 -1 0\n"
               << "1  2 -1 0\n"
               << "2 -2  3 0\n"
               << "3 -2 -1 1\n";
    }
    {
        std::ofstream output(element_file);
        ASSERT_TRUE(output.good());
        // This is the negative ordering of the source tetrahedron. The
        // importer must flip it before create_solid validates the mesh.
        output << "1 4 0\n"
               << "0 0 2 1 3\n";
    }

    RefMesh ref_mesh;
    DeformedState state;
    const Vec3 target_center(5.0, 6.0, 7.0);
    const int base = append_normalized_tetgen_solid(
        node_file.string(), element_file.string(), state, ref_mesh,
        target_center, /*target_max_extent=*/2.0, /*density=*/6.0,
        /*zero_based_index=*/true);

    EXPECT_EQ(base, 0);
    ASSERT_EQ(state.deformed_positions.size(), 4U);
    Vec3 lower = state.deformed_positions.front();
    Vec3 upper = state.deformed_positions.front();
    for (const Vec3& position : state.deformed_positions) {
        lower = lower.cwiseMin(position);
        upper = upper.cwiseMax(position);
    }
    EXPECT_TRUE((0.5 * (lower + upper)).isApprox(target_center, 0.0));
    EXPECT_DOUBLE_EQ((upper - lower).maxCoeff(), 2.0);
    ASSERT_EQ(ref_mesh.tet_rest_data.size(), 1U);
    EXPECT_GT(ref_mesh.tet_rest_data.front().measure, 0.0);
    EXPECT_NEAR(ref_mesh.tet_rest_data.front().measure, 1.0 / 3.0, 1.0e-15);
    EXPECT_NEAR(
        std::accumulate(ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0),
        2.0, 1.0e-14);

    std::error_code error;
    fs::remove_all(directory, error);
}

TEST(AppendNormalizedObjRigidBody,
     RemovesOrphansNormalizesRotatesOffsetsTrianglesAndUsesVolumeMass) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_normalized_obj_rigid_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);
    const fs::path obj_file = directory / "tetrahedron.obj";
    {
        std::ofstream output(obj_file);
        ASSERT_TRUE(output.good());
        output << "v 0 0 0\n"
               << "v 2 0 0\n"
               << "v 0 4 0\n"
               << "v 0 0 1\n"
               // This orphan must neither alter normalization nor become a
               // collision proxy particle.
               << "v 100 100 100\n"
               << "f 2 3 4\n"
               << "f 1 4 3\n"
               << "f 1 2 4\n"
               << "f 1 3 2\n";
    }

    RefMesh ref_mesh;
    DeformedState state;
    state.deformed_positions.push_back(Vec3(-10.0, -10.0, -10.0));
    ref_mesh.tris = {0, 0, 0};
    const Vec3 target_center(5.0, 6.0, 7.0);
    const double half_angle = 0.25 * M_PI;
    const Vec4 orientation(
        std::cos(half_angle), 0.0, 0.0, std::sin(half_angle));
    const Vec3 v_com(0.1, 0.2, 0.3);
    const Vec3 omega(0.0, 0.0, 2.0);
    const int rigid_body = append_normalized_obj_rigid_body(
        obj_file.string(), state, ref_mesh, target_center,
        /*target_max_extent=*/2.0, /*density=*/12.0,
        v_com, orientation, omega);

    EXPECT_EQ(rigid_body, 0);
    ASSERT_EQ(state.deformed_positions.size(), 5U);
    ASSERT_EQ(ref_mesh.rb_nodes.size(), 1U);
    EXPECT_EQ(
        ref_mesh.rb_nodes[0], (std::vector<int>{1, 2, 3, 4}));
    ASSERT_EQ(ref_mesh.tris.size(), 15U);
    EXPECT_EQ(
        std::vector<int>(ref_mesh.tris.begin() + 3, ref_mesh.tris.end()),
        (std::vector<int>{
            2, 3, 4, 1, 4, 3, 1, 2, 4, 1, 3, 2}));

    Vec3 lower = state.deformed_positions[1];
    Vec3 upper = state.deformed_positions[1];
    for (std::size_t node = 1; node < state.deformed_positions.size(); ++node) {
        lower = lower.cwiseMin(state.deformed_positions[node]);
        upper = upper.cwiseMax(state.deformed_positions[node]);
    }
    EXPECT_TRUE((0.5 * (lower + upper)).isApprox(target_center, 1.0e-14));
    EXPECT_NEAR((upper - lower).maxCoeff(), 2.0, 1.0e-14);
    ASSERT_EQ(ref_mesh.total_mass.size(), 1U);
    // Source volume is 8/6, and normalization scales lengths by 1/2:
    // 12 * (8/6) * (1/2)^3 = 2.
    EXPECT_NEAR(ref_mesh.total_mass[0], 2.0, 1.0e-14);
    EXPECT_TRUE(state.v_coms[0].isApprox(v_com, 0.0));
    EXPECT_TRUE(state.orientations[0].isApprox(orientation, 1.0e-15));
    EXPECT_TRUE(state.omega[0].isApprox(omega, 0.0));

    std::error_code error;
    fs::remove_all(directory, error);
}

TEST(AppendNormalizedObjRigidBody, RejectsOpenSurfaceBeforeMutation) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_open_obj_rigid_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);
    const fs::path obj_file = directory / "open.obj";
    {
        std::ofstream output(obj_file);
        ASSERT_TRUE(output.good());
        output << "v 0 0 0\n"
               << "v 1 0 0\n"
               << "v 0 1 0\n"
               << "f 1 2 3\n";
    }

    RefMesh ref_mesh;
    DeformedState state;
    ref_mesh.tris = {7, 8, 9};
    ref_mesh.mass = {3.0};
    ref_mesh.node_to_rb = {-1};
    ref_mesh.num_positions = 1;
    state.deformed_positions = {Vec3(1.0, 2.0, 3.0)};
    state.velocities = {Vec3(4.0, 5.0, 6.0)};

    EXPECT_THROW(
        append_normalized_obj_rigid_body(
            obj_file.string(), state, ref_mesh, Vec3::Zero(),
            1.0, 900.0),
        std::invalid_argument);
    EXPECT_EQ(ref_mesh.tris, (std::vector<int>{7, 8, 9}));
    EXPECT_EQ(ref_mesh.mass, (std::vector<double>{3.0}));
    EXPECT_EQ(ref_mesh.node_to_rb, (std::vector<int>{-1}));
    EXPECT_EQ(ref_mesh.num_positions, 1U);
    ASSERT_EQ(state.deformed_positions.size(), 1U);
    EXPECT_TRUE(state.deformed_positions[0].isApprox(
        Vec3(1.0, 2.0, 3.0), 0.0));
    ASSERT_EQ(state.velocities.size(), 1U);
    EXPECT_TRUE(state.velocities[0].isApprox(Vec3(4.0, 5.0, 6.0), 0.0));
    EXPECT_TRUE(ref_mesh.total_mass.empty());

    std::error_code error;
    fs::remove_all(directory, error);
}

// ---------------------------------------------------------------------------
// build_sphere_mesh tests
// ---------------------------------------------------------------------------

TEST(BuildSphereMesh, VertexAndTriangleCounts) {
    // Level 2 icosphere: V = 10*4^2 + 2 = 162, F = 20*4^2 = 320.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    const int subdiv = 2;
    const double radius = 0.5;
    const Vec3 center(0.0, 0.0, 0.0);

    const int base = build_sphere_mesh(ref_mesh, state, X, subdiv, radius, center);
    EXPECT_EQ(base, 0);

    const int expected_verts = 162;
    EXPECT_EQ(static_cast<int>(state.deformed_positions.size()), expected_verts);
    EXPECT_EQ(static_cast<int>(X.size()), expected_verts);

    const int expected_tris = 320;
    EXPECT_EQ(static_cast<int>(ref_mesh.tris.size()), 3 * expected_tris);
}

TEST(BuildSphereMesh, AllVerticesAtRadiusFromCenter) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    const double radius = 0.25;
    const Vec3 center(0.1, -0.2, 0.05);

    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/3, radius, center);

    // FP-precision tolerance: base icosahedron verts compute norm as s*sqrt(1+phi^2)
    // via a sqrt and a division, which leaves a few ULPs of error at the radius.
    // Subdivided midpoints are explicitly renormalized to exactly radius.
    // FP precision: base icosahedron verts compute norm via a sqrt and division
    // leaving a few ULPs; subdivided midpoints are explicitly renormalized.
    constexpr double kTol = 1e-10;
    for (const Vec3& p : state.deformed_positions)
        EXPECT_NEAR((p - center).norm(), radius, kTol);
}

TEST(BuildSphereMesh, BaseIcosahedron) {
    // subdiv = 0 is the base icosahedron: 12 vertices, 20 triangles.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/0, /*radius=*/1.0, Vec3::Zero());

    EXPECT_EQ(static_cast<int>(state.deformed_positions.size()), 12);
    EXPECT_EQ(static_cast<int>(ref_mesh.tris.size()),            60);
}

TEST(BuildSphereMesh, ReferenceAreasNonDegenerate) {
    // Reference-space 2D triangle areas must be strictly positive so
    // ref_mesh.initialize(X) doesn't divide by zero when forming Dm_inverse.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/2, /*radius=*/0.5, Vec3::Zero());

    const int nt = static_cast<int>(ref_mesh.tris.size()) / 3;
    for (int t = 0; t < nt; ++t) {
        const Vec2& a = X[ref_mesh.tris[3*t + 0]];
        const Vec2& b = X[ref_mesh.tris[3*t + 1]];
        const Vec2& c = X[ref_mesh.tris[3*t + 2]];
        const double area2 = std::abs((b.x()-a.x())*(c.y()-a.y()) - (b.y()-a.y())*(c.x()-a.x()));
        EXPECT_GT(area2, 1e-10) << "degenerate ref triangle " << t;
    }
}

TEST(MixedExample,
     TenAlternatingRigidAndSolidPolygonsFormAFlatVerticalStackAbovePinnedCloth) {
    IPCArgs3D args;
    // Non-default, distinct values catch accidental use of the cloth density
    // or of one body class's density for the other class.
    args.rigid_density = 432.0;
    args.solid_density = 789.0;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_ten_alternating_rigid_solid_flat_stack_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 40;
    constexpr int cloth_nz = 40;
    constexpr int cloth_vertices =
        (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int object_count = 10;
    constexpr int rigid_body_count = 5;
    // Rigid sides are 3,5,7,9,11 (35 total); solid sides are
    // 4,6,8,10,12 (40 total). Every solid also has one interior node.
    constexpr int rigid_vertices = 2 * 35;
    constexpr int rigid_triangles = 4 * 35 - 4 * rigid_body_count;
    constexpr int solid_surface_vertices = 2 * 40;
    constexpr int solid_vertices = solid_surface_vertices + 5;
    constexpr int solid_tetrahedra = 4 * 40 - 4 * 5;
    constexpr int total_vertices =
        cloth_vertices + rigid_vertices + solid_vertices;
    constexpr int total_triangles =
        cloth_triangles + rigid_triangles + solid_tetrahedra;

    EXPECT_EQ(state.deformed_positions.size(), total_vertices);
    EXPECT_EQ(state.velocities.size(), total_vertices);
    EXPECT_EQ(ref_mesh.num_positions, total_vertices);
    EXPECT_EQ(ref_mesh.mass.size(), total_vertices);
    EXPECT_EQ(ref_mesh.node_to_rb.size(), total_vertices);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_EQ(ref_mesh.tets.size(), 4 * solid_tetrahedra);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), solid_tetrahedra);
    EXPECT_EQ(ref_mesh.tet_nodes.size(), solid_vertices);
    EXPECT_EQ(ref_mesh.surface_nodes.size(), solid_surface_vertices);
    EXPECT_EQ(ref_mesh.deformable_nodes.size(),
              cloth_vertices + solid_vertices);

    EXPECT_EQ(X.size(), cloth_vertices);
    EXPECT_EQ(ref_mesh.Dm_inverse.size(), cloth_triangles);
    EXPECT_EQ(ref_mesh.area.size(), cloth_triangles);

    ASSERT_EQ(
        pins.size(), static_cast<std::size_t>(2 * (cloth_nz + 1)));
    for (int j = 0; j <= cloth_nz; ++j) {
        SCOPED_TRACE(j);
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j)].vertex_index,
            j * (cloth_nx + 1));
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j + 1)].vertex_index,
            j * (cloth_nx + 1) + cloth_nx);
    }

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);

    std::vector<unsigned char> is_tet_node(total_vertices, 0);
    std::vector<unsigned char> is_surface_node(total_vertices, 0);
    std::vector<unsigned char> is_deformable(total_vertices, 0);
    for (const int node : ref_mesh.tet_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
        is_tet_node[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : ref_mesh.surface_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
        is_surface_node[static_cast<std::size_t>(node)] = 1;
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 1);
    }
    for (const int node : ref_mesh.deformable_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
        is_deformable[static_cast<std::size_t>(node)] = 1;
        EXPECT_EQ(
            is_tet_node[static_cast<std::size_t>(node)],
            node < cloth_vertices ? 0 : 1);
        EXPECT_EQ(ref_mesh.node_to_rb[static_cast<std::size_t>(node)], -1);
    }

    constexpr double cloth_height = 1.2;
    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_DOUBLE_EQ(state.deformed_positions[node].y(), cloth_height);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
    }
    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            EXPECT_LT(
                ref_mesh.tris[3 * triangle + corner], cloth_vertices);
        }
    }

    constexpr double kPi = 3.14159265358979323846;
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    const Vec3 drop_velocity(0.0, -0.75, 0.0);
    constexpr double lowest_center_y = 1.65;
    constexpr double center_spacing = 0.42;
    constexpr double expected_radius = 0.24;
    constexpr double expected_thickness = 0.16;
    int node_cursor = cloth_vertices;
    int rigid_cursor = 0;
    int tet_cursor = 0;
    double previous_top = cloth_height;
    for (int object = 0; object < object_count; ++object) {
        SCOPED_TRACE(object);
        const int side_count = 3 + object;
        const bool is_rigid = object % 2 == 0;
        const int object_vertices =
            2 * side_count + (is_rigid ? 0 : 1);
        const int object_end = node_cursor + object_vertices;
        ASSERT_LE(object_end, total_vertices);

        const Vec3 center = is_rigid
            ? state.x_coms[static_cast<std::size_t>(rigid_cursor)]
            : state.deformed_positions[static_cast<std::size_t>(
                  node_cursor + 2 * side_count)];
        EXPECT_NEAR(center.x(), 0.0, 1.0e-14);
        EXPECT_NEAR(center.z(), 0.0, 1.0e-14);
        EXPECT_NEAR(
            center.y(), lowest_center_y + center_spacing * object,
            1.0e-14);

        const double bottom_y =
            state.deformed_positions[static_cast<std::size_t>(node_cursor)].y();
        const double top_y = state.deformed_positions[static_cast<std::size_t>(
            node_cursor + side_count)].y();
        ASSERT_GT(top_y, bottom_y);
        EXPECT_NEAR(center.y(), 0.5 * (bottom_y + top_y), 1.0e-14);
        EXPECT_GT(bottom_y, previous_top);
        if (object == 0)
            EXPECT_GT(bottom_y, cloth_height + params.d_hat);
        previous_top = top_y;

        const Vec3& first_bottom =
            state.deformed_positions[static_cast<std::size_t>(node_cursor)];
        const double radius = std::hypot(
            first_bottom.x() - center.x(),
            first_bottom.z() - center.z());
        const double thickness = top_y - bottom_y;
        ASSERT_GT(radius, 0.0);
        ASSERT_GT(thickness, 0.0);
        EXPECT_NEAR(radius, expected_radius, 1.0e-14);
        EXPECT_NEAR(thickness, expected_thickness, 1.0e-14);
        // With exactly Rx(-pi/2), material vertex (radius,0,z) still points
        // along world +x. This catches both tilt and an unrequested yaw.
        EXPECT_NEAR(first_bottom.x(), center.x() + radius, 1.0e-14);
        EXPECT_NEAR(first_bottom.z(), center.z(), 1.0e-14);

        for (int local = 0; local < side_count; ++local) {
            const Vec3& bottom = state.deformed_positions[
                static_cast<std::size_t>(node_cursor + local)];
            const Vec3& top = state.deformed_positions[
                static_cast<std::size_t>(node_cursor + side_count + local)];
            EXPECT_NEAR(bottom.y(), bottom_y, 1.0e-14);
            EXPECT_NEAR(top.y(), top_y, 1.0e-14);
            EXPECT_NEAR(bottom.x(), top.x(), 1.0e-14);
            EXPECT_NEAR(bottom.z(), top.z(), 1.0e-14);
            EXPECT_NEAR(
                std::hypot(bottom.x() - center.x(),
                           bottom.z() - center.z()),
                radius, 1.0e-14);
        }

        const double polygon_area =
            0.5 * static_cast<double>(side_count) * radius * radius
            * std::sin(2.0 * kPi / static_cast<double>(side_count));
        const double volume = polygon_area * thickness;
        if (is_rigid) {
            ASSERT_EQ(
                ref_mesh.rb_nodes[static_cast<std::size_t>(rigid_cursor)].size(),
                static_cast<std::size_t>(2 * side_count));
            EXPECT_EQ(
                ref_mesh.rb_nodes[static_cast<std::size_t>(rigid_cursor)].front(),
                node_cursor);
            EXPECT_TRUE(state.orientations[static_cast<std::size_t>(rigid_cursor)]
                            .isApprox(flat_orientation, 1.0e-14));
            EXPECT_TRUE(state.omega[static_cast<std::size_t>(rigid_cursor)]
                            .isZero(0.0));
            EXPECT_TRUE(state.v_coms[static_cast<std::size_t>(rigid_cursor)]
                            .isApprox(drop_velocity, 0.0));
            EXPECT_NEAR(
                ref_mesh.total_mass[static_cast<std::size_t>(rigid_cursor)],
                args.rigid_density * volume,
                1.0e-10 * std::max(1.0, args.rigid_density * volume));

            double nodal_mass = 0.0;
            for (int node = node_cursor; node < object_end; ++node) {
                EXPECT_EQ(
                    ref_mesh.node_to_rb[static_cast<std::size_t>(node)],
                    rigid_cursor);
                EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
                EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
                EXPECT_TRUE(
                    state.velocities[static_cast<std::size_t>(node)]
                        .isApprox(drop_velocity, 0.0));
                nodal_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
            }
            EXPECT_NEAR(
                nodal_mass,
                ref_mesh.total_mass[static_cast<std::size_t>(rigid_cursor)],
                1.0e-12);
            ++rigid_cursor;
        } else {
            double nodal_mass = 0.0;
            for (int node = node_cursor; node < object_end; ++node) {
                EXPECT_EQ(ref_mesh.node_to_rb[static_cast<std::size_t>(node)], -1);
                EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 1);
                EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
                EXPECT_TRUE(
                    state.velocities[static_cast<std::size_t>(node)]
                        .isApprox(drop_velocity, 0.0));
                nodal_mass += ref_mesh.mass[static_cast<std::size_t>(node)];
            }
            for (int local = 0; local < 2 * side_count; ++local) {
                EXPECT_EQ(
                    is_surface_node[static_cast<std::size_t>(node_cursor + local)],
                    1);
            }
            EXPECT_EQ(
                is_surface_node[static_cast<std::size_t>(object_end - 1)], 0);
            EXPECT_NEAR(
                nodal_mass, args.solid_density * volume,
                1.0e-10 * std::max(1.0, args.solid_density * volume));

            const int object_tetrahedra = 4 * side_count - 4;
            for (int local_tet = 0; local_tet < object_tetrahedra;
                 ++local_tet, ++tet_cursor) {
                for (int local = 0; local < 4; ++local) {
                    const int node = ref_mesh.tets[
                        static_cast<std::size_t>(4 * tet_cursor + local)];
                    EXPECT_GE(node, node_cursor);
                    EXPECT_LT(node, object_end);
                }
            }
        }
        node_cursor = object_end;
    }
    EXPECT_EQ(node_cursor, total_vertices);
    EXPECT_EQ(rigid_cursor, rigid_body_count);
    EXPECT_EQ(tet_cursor, solid_tetrahedra);

    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());

    // Shell lumping fills the cloth masses without changing the solid or
    // rigid-body nodal masses already constructed by their object builders.
    const std::vector<double> object_masses(
        ref_mesh.mass.begin() + cloth_vertices, ref_mesh.mass.end());
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    const double total_cloth_mass = std::accumulate(
        ref_mesh.mass.begin(),
        ref_mesh.mass.begin() + cloth_vertices, 0.0);
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    EXPECT_TRUE(std::equal(
        object_masses.begin(), object_masses.end(),
        ref_mesh.mass.begin() + cloth_vertices));
}

TEST(MixedExample,
     TwoBunnySpotCubeGearCyclesFormOneVerticalStackAbovePinnedCloth) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_bunny_spot_cube_gear_scene_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);

    struct WorkingDirectoryGuard {
        fs::path previous;
        fs::path temporary;

        explicit WorkingDirectoryGuard(fs::path path)
            : previous(fs::current_path()), temporary(std::move(path)) {
            fs::current_path(temporary);
        }

        ~WorkingDirectoryGuard() {
            std::error_code error;
            fs::current_path(previous, error);
            fs::remove_all(temporary, error);
        }
    } working_directory(directory);

    // Exercise the production scene's fixed repository-relative paths using
    // tiny, distinct meshes. The Bunny fixture has two tetrahedra while Spot
    // has one, so the test catches accidental path reuse between solid types.
    fs::create_directories("example_obj/bunny_coarse");
    fs::create_directories("example_obj/spot");
    {
        std::ofstream nodes(
            "example_obj/bunny_coarse/bunny_2000f.1.node");
        ASSERT_TRUE(nodes.good());
        nodes << "5 3 0 0\n"
              << "0 0 0 0\n"
              << "1 8 0 0\n"
              << "2 0 2 0\n"
              << "3 0 0 1\n"
              << "4 0 0 -3\n";
    }
    {
        std::ofstream elements(
            "example_obj/bunny_coarse/bunny_2000f.1.ele");
        ASSERT_TRUE(elements.good());
        elements << "2 4 0\n"
                 << "0 0 1 2 3\n"
                 << "1 0 2 1 4\n";
    }
    {
        std::ofstream nodes("example_obj/spot/spot_2000f.1.node");
        ASSERT_TRUE(nodes.good());
        nodes << "4 3 0 0\n"
              << "0 0 0 0\n"
              << "1 10 0 0\n"
              << "2 0 4 0\n"
              << "3 0 0 2\n";
    }
    {
        std::ofstream elements("example_obj/spot/spot_2000f.1.ele");
        ASSERT_TRUE(elements.good());
        elements << "1 4 0\n"
                 << "0 0 1 2 3\n";
    }
    {
        std::ofstream gear("example_obj/gear_z18_coarse.obj");
        ASSERT_TRUE(gear.good());
        // Closed, outward-oriented anisotropic octahedron with source AABB
        // 4 x 2 x 1 and volume 4/3.
        gear << "v  2  0    0\n"
             << "v -2  0    0\n"
             << "v  0  1    0\n"
             << "v  0 -1    0\n"
             << "v  0  0  0.5\n"
             << "v  0  0 -0.5\n"
             << "f 1 3 5\n"
             << "f 3 2 5\n"
             << "f 2 4 5\n"
             << "f 4 1 5\n"
             << "f 3 1 6\n"
             << "f 2 3 6\n"
             << "f 4 2 6\n"
             << "f 1 4 6\n";
    }

    IPCArgs3D args;
    args.solid_density = 731.0;
    args.rigid_density = 947.0;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_two_bunny_spot_cube_gear_cycles_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 30;
    constexpr int cloth_nz = 30;
    constexpr int cloth_vertices = (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr int copies = 2;
    constexpr int bunny_nodes = 5;
    constexpr int bunny_tets = 2;
    constexpr int bunny_surface_triangles = 6;
    constexpr int spot_nodes = 4;
    constexpr int spot_tets = 1;
    constexpr int spot_surface_triangles = 4;
    constexpr int cube_nodes = 8;
    constexpr int cube_triangles = 12;
    constexpr int gear_nodes = 6;
    constexpr int gear_triangles = 8;
    constexpr int solid_nodes = copies * (bunny_nodes + spot_nodes);
    constexpr int solid_tets = copies * (bunny_tets + spot_tets);
    constexpr int rigid_body_count = 2 * copies;
    constexpr int total_vertices = cloth_vertices
        + solid_nodes + copies * (cube_nodes + gear_nodes);
    constexpr int total_triangles = cloth_triangles
        + copies
            * (bunny_surface_triangles + spot_surface_triangles
               + cube_triangles + gear_triangles);

    static_assert(total_vertices == 1007);
    static_assert(total_triangles == 1860);
    static_assert(solid_tets == 6);

    ASSERT_EQ(state.deformed_positions.size(), total_vertices);
    ASSERT_EQ(state.velocities.size(), total_vertices);
    ASSERT_EQ(ref_mesh.num_positions, total_vertices);
    ASSERT_EQ(ref_mesh.mass.size(), total_vertices);
    ASSERT_EQ(ref_mesh.node_to_rb.size(), total_vertices);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_EQ(ref_mesh.tets.size(), 4 * solid_tets);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), solid_tets);
    EXPECT_EQ(ref_mesh.tet_nodes.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.surface_nodes.size(), solid_nodes);
    EXPECT_EQ(
        ref_mesh.deformable_nodes.size(), cloth_vertices + solid_nodes);
    EXPECT_EQ(X.size(), cloth_vertices);
    EXPECT_EQ(ref_mesh.Dm_inverse.size(), cloth_triangles);
    EXPECT_EQ(ref_mesh.area.size(), cloth_triangles);

    ASSERT_EQ(pins.size(), 2 * (cloth_nz + 1));
    for (int j = 0; j <= cloth_nz; ++j) {
        const int left = j * (cloth_nx + 1);
        const int right = left + cloth_nx;
        EXPECT_EQ(pins[static_cast<std::size_t>(2 * j)].vertex_index, left);
        EXPECT_EQ(
            pins[static_cast<std::size_t>(2 * j + 1)].vertex_index,
            right);
        EXPECT_TRUE(pins[static_cast<std::size_t>(2 * j)].target_position
                        .isApprox(state.deformed_positions[left], 0.0));
        EXPECT_TRUE(pins[static_cast<std::size_t>(2 * j + 1)].target_position
                        .isApprox(state.deformed_positions[right], 0.0));
    }

    std::vector<unsigned char> is_tet_node(total_vertices, 0);
    std::vector<unsigned char> is_surface_node(total_vertices, 0);
    std::vector<unsigned char> is_deformable(total_vertices, 0);
    for (const int node : ref_mesh.tet_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        is_tet_node[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : ref_mesh.surface_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        is_surface_node[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : ref_mesh.deformable_nodes) {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, total_vertices);
        is_deformable[static_cast<std::size_t>(node)] = 1;
    }

    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_DOUBLE_EQ(state.deformed_positions[node].y(), 1.2);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
    }

    constexpr double solid_max_extent = 0.26;
    constexpr double rigid_max_extent = 0.14;
    constexpr double initial_cloth_clearance = 0.02;
    constexpr double first_center_y =
        1.2 + 0.5 * solid_max_extent + initial_cloth_clearance;
    constexpr double vertical_spacing = 0.34;
    EXPECT_NEAR(first_center_y, 1.35, 1.0e-15);
    EXPECT_GT(
        first_center_y - 0.5 * solid_max_extent,
        1.2 + params.d_hat);
    const auto check_solid =
        [&](const int node_base, const int node_count,
            const int first_tet, const int tet_count,
            const Vec3& expected_center, const double expected_mass) {
            const int node_end = node_base + node_count;
            Vec3 lower = state.deformed_positions[node_base];
            Vec3 upper = lower;
            double mass = 0.0;
            for (int node = node_base; node < node_end; ++node) {
                lower = lower.cwiseMin(state.deformed_positions[node]);
                upper = upper.cwiseMax(state.deformed_positions[node]);
                mass += ref_mesh.mass[static_cast<std::size_t>(node)];
                EXPECT_TRUE(state.velocities[node].isZero(0.0));
                EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
                EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 1);
                EXPECT_EQ(
                    is_surface_node[static_cast<std::size_t>(node)], 1);
                EXPECT_EQ(
                    is_deformable[static_cast<std::size_t>(node)], 1);
            }
            const Vec3 actual_center = 0.5 * (lower + upper);
            EXPECT_TRUE(actual_center.isApprox(expected_center, 1.0e-14));
            EXPECT_NEAR(
                (upper - lower).maxCoeff(), solid_max_extent, 1.0e-14);
            EXPECT_NEAR(mass, expected_mass, 1.0e-12);

            for (int element = first_tet;
                 element < first_tet + tet_count; ++element) {
                for (int local = 0; local < 4; ++local) {
                    const int node = ref_mesh.tets[4 * element + local];
                    EXPECT_GE(node, node_base);
                    EXPECT_LT(node, node_end);
                }
                EXPECT_GT(ref_mesh.tet_rest_data[element].measure, 0.0);
            }
            return actual_center;
        };

    // Bunny source AABB has max extent 8 and the two source tetrahedra have
    // combined volume 32/3.
    constexpr double bunny_source_volume = 32.0 / 3.0;
    constexpr double bunny_scale = solid_max_extent / 8.0;
    const double expected_bunny_mass = args.solid_density
        * bunny_source_volume * bunny_scale * bunny_scale * bunny_scale;
    std::array<Vec3, copies> bunny_centers;
    for (int copy = 0; copy < copies; ++copy) {
        SCOPED_TRACE("bunny " + std::to_string(copy));
        const int node_base = cloth_vertices + copy * bunny_nodes;
        bunny_centers[static_cast<std::size_t>(copy)] = check_solid(
            node_base, bunny_nodes, copy * bunny_tets, bunny_tets,
            Vec3(
                0.0,
                first_center_y + (4 * copy) * vertical_spacing,
                0.0),
            expected_bunny_mass);
    }

    // Spot source AABB has max extent 10 and source volume 40/3.
    constexpr double spot_source_volume = 40.0 / 3.0;
    constexpr double spot_scale = solid_max_extent / 10.0;
    const double expected_spot_mass = args.solid_density
        * spot_source_volume * spot_scale * spot_scale * spot_scale;
    constexpr int spot_node_base =
        cloth_vertices + copies * bunny_nodes;
    constexpr int spot_tet_base = copies * bunny_tets;
    std::array<Vec3, copies> spot_centers;
    for (int copy = 0; copy < copies; ++copy) {
        SCOPED_TRACE("spot " + std::to_string(copy));
        spot_centers[static_cast<std::size_t>(copy)] = check_solid(
            spot_node_base + copy * spot_nodes, spot_nodes,
            spot_tet_base + copy * spot_tets, spot_tets,
            Vec3(
                0.0,
                first_center_y + (4 * copy + 1) * vertical_spacing,
                0.0),
            expected_spot_mass);
    }

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);

    const double expected_cube_mass = args.rigid_density
        * rigid_max_extent * rigid_max_extent * rigid_max_extent;
    for (int copy = 0; copy < copies; ++copy) {
        SCOPED_TRACE("cube " + std::to_string(copy));
        const int rb = copy;
        const Vec3 expected_center(
            0.0,
            first_center_y + (4 * copy + 2) * vertical_spacing,
            0.0);
        EXPECT_TRUE(
            state.x_coms[rb].isApprox(expected_center, 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
        EXPECT_TRUE(state.omega[rb].isZero(0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            Vec4(1.0, 0.0, 0.0, 0.0), 0.0));
        EXPECT_NEAR(
            ref_mesh.total_mass[rb], expected_cube_mass, 1.0e-12);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), cube_nodes);

        Vec3 lower =
            state.deformed_positions[ref_mesh.rb_nodes[rb].front()];
        Vec3 upper = lower;
        for (const int node : ref_mesh.rb_nodes[rb]) {
            lower = lower.cwiseMin(state.deformed_positions[node]);
            upper = upper.cwiseMax(state.deformed_positions[node]);
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
            EXPECT_TRUE(state.velocities[node].isZero(0.0));
        }
        EXPECT_TRUE(
            (0.5 * (lower + upper)).isApprox(expected_center, 1.0e-14));
        EXPECT_TRUE((upper - lower).isApprox(
            Vec3::Constant(rigid_max_extent), 1.0e-14));
    }

    constexpr double gear_source_volume = 4.0 / 3.0;
    constexpr double gear_scale = rigid_max_extent / 4.0;
    const double expected_gear_mass = args.rigid_density
        * gear_source_volume * gear_scale * gear_scale * gear_scale;
    constexpr double kPi = 3.14159265358979323846;
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    for (int copy = 0; copy < copies; ++copy) {
        SCOPED_TRACE("gear " + std::to_string(copy));
        const int rb = copies + copy;
        const Vec3 expected_center(
            0.0,
            first_center_y + (4 * copy + 3) * vertical_spacing,
            0.0);
        const double yaw = static_cast<double>(copy) * kPi / 8.0;
        const Vec4 yaw_orientation(
            std::cos(0.5 * yaw), 0.0, std::sin(0.5 * yaw), 0.0);
        const Vec4 expected_orientation = quaternion_normalize(
            quaternion_multiply(yaw_orientation, flat_orientation));

        EXPECT_TRUE(
            state.x_coms[rb].isApprox(expected_center, 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
        EXPECT_TRUE(state.omega[rb].isZero(0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            expected_orientation, 1.0e-14));
        EXPECT_NEAR(
            ref_mesh.total_mass[rb], expected_gear_mass, 1.0e-12);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), gear_nodes);

        Vec3 lower = ref_mesh.ref_positions[rb].front();
        Vec3 upper = lower;
        for (std::size_t local = 0;
             local < ref_mesh.rb_nodes[rb].size(); ++local) {
            const int node = ref_mesh.rb_nodes[rb][local];
            lower = lower.cwiseMin(ref_mesh.ref_positions[rb][local]);
            upper = upper.cwiseMax(ref_mesh.ref_positions[rb][local]);
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
            EXPECT_TRUE(state.velocities[node].isZero(0.0));
        }
        EXPECT_TRUE((0.5 * (lower + upper)).isZero(1.0e-14));
        EXPECT_NEAR(
            (upper - lower).maxCoeff(), rigid_max_extent, 1.0e-14);
    }

    // Verify the actual bottom-to-top order across both cycles.
    EXPECT_GT(vertical_spacing, solid_max_extent);
    for (int copy = 0; copy < copies; ++copy) {
        const int cube_rb = copy;
        const int gear_rb = copies + copy;
        EXPECT_NEAR(
            spot_centers[static_cast<std::size_t>(copy)].y()
                - bunny_centers[static_cast<std::size_t>(copy)].y(),
            vertical_spacing, 1.0e-14);
        EXPECT_NEAR(
            state.x_coms[cube_rb].y()
                - spot_centers[static_cast<std::size_t>(copy)].y(),
            vertical_spacing, 1.0e-14);
        EXPECT_NEAR(
            state.x_coms[gear_rb].y() - state.x_coms[cube_rb].y(),
            vertical_spacing, 1.0e-14);
        if (copy + 1 < copies) {
            EXPECT_NEAR(
                bunny_centers[static_cast<std::size_t>(copy + 1)].y()
                    - state.x_coms[gear_rb].y(),
                vertical_spacing, 1.0e-14);
        }
    }

    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (std::size_t triangle = 0; triangle < ref_mesh.tris.size() / 3;
         ++triangle) {
        const int* tri = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            minimum_surface_edge = std::min(
                minimum_surface_edge,
                (state.deformed_positions[tri[(local + 1) % 3]]
                 - state.deformed_positions[tri[local]])
                    .norm());
        }
    }
    EXPECT_NEAR(
        params.d_hat,
        std::min(args.d_hat, 0.45 * minimum_surface_edge), 1.0e-15);

    const std::vector<double> object_masses(
        ref_mesh.mass.begin() + cloth_vertices, ref_mesh.mass.end());
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    EXPECT_NEAR(
        std::accumulate(
            ref_mesh.mass.begin(),
            ref_mesh.mass.begin() + cloth_vertices, 0.0),
        params.density * params.thickness * 4.0 * 4.0,
        1.0e-10);
    EXPECT_TRUE(std::equal(
        object_masses.begin(), object_masses.end(),
        ref_mesh.mass.begin() + cloth_vertices));
}

TEST(RigidExample,
     DynamicBoltStartsDeepInsideFullyFixedNutWithSharedAssetScale) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_fixed_nut_dynamic_bolt_scene_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);

    struct WorkingDirectoryGuard {
        fs::path previous;
        fs::path temporary;

        explicit WorkingDirectoryGuard(fs::path path)
            : previous(fs::current_path()), temporary(std::move(path)) {
            fs::current_path(temporary);
        }

        ~WorkingDirectoryGuard() {
            std::error_code error;
            fs::current_path(previous, error);
            fs::remove_all(temporary, error);
        }
    } working_directory(directory);

    fs::create_directories("example_obj/bolt_and_nut");
    const auto write_octahedron = [](
        const char* filename, const double half_x,
        const double half_y, const double half_z) {
        std::ofstream obj(filename);
        ASSERT_TRUE(obj.good());
        obj.precision(17);
        obj << "v " << half_x << " 0 0\n"
            << "v " << -half_x << " 0 0\n"
            << "v 0 " << half_y << " 0\n"
            << "v 0 " << -half_y << " 0\n"
            << "v 0 0 " << half_z << "\n"
            << "v 0 0 " << -half_z << "\n"
            << "f 1 3 5\n"
            << "f 3 2 5\n"
            << "f 2 4 5\n"
            << "f 4 1 5\n"
            << "f 3 1 6\n"
            << "f 2 3 6\n"
            << "f 4 2 6\n"
            << "f 1 4 6\n";
        ASSERT_TRUE(obj.good());
    };

    // These symmetric closed fixtures have the exact source AABB dimensions
    // used by the production assets. Their simple topology keeps this scene
    // construction test fast while still detecting independent normalization.
    constexpr double source_half_x = 15.0;
    constexpr double source_half_y = 17.32051;
    constexpr double bolt_source_half_z = 29.0;
    constexpr double nut_source_half_z = 8.0;
    write_octahedron(
        "example_obj/bolt_and_nut/bolt_coarse_bolt.obj",
        source_half_x, source_half_y, bolt_source_half_z);
    write_octahedron(
        "example_obj/bolt_and_nut/bolt_coarse_nut.obj",
        source_half_x, source_half_y, nut_source_half_z);

    IPCArgs3D args;
    args.rigid_density = 947.0;
    // Force the scene's mesh-resolution clamp to be active.
    args.d_hat = 1.0;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_dynamic_bolt_into_fixed_nut_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int nodes_per_body = 6;
    constexpr int triangles_per_body = 8;
    constexpr int rigid_body_count = 2;
    constexpr int total_nodes = rigid_body_count * nodes_per_body;
    constexpr int total_triangles =
        rigid_body_count * triangles_per_body;

    EXPECT_EQ(state.deformed_positions.size(), total_nodes);
    EXPECT_EQ(state.velocities.size(), total_nodes);
    EXPECT_EQ(ref_mesh.num_positions, total_nodes);
    EXPECT_EQ(ref_mesh.mass.size(), total_nodes);
    EXPECT_EQ(ref_mesh.node_to_rb.size(), total_nodes);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.tet_rest_data.empty());
    EXPECT_TRUE(ref_mesh.tet_nodes.empty());
    EXPECT_TRUE(ref_mesh.surface_nodes.empty());
    EXPECT_TRUE(ref_mesh.deformable_nodes.empty());
    EXPECT_TRUE(ref_mesh.Dm_inverse.empty());
    EXPECT_TRUE(ref_mesh.area.empty());
    EXPECT_TRUE(ref_mesh.hinges.empty());
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());

    ASSERT_EQ(ref_mesh.rb_nodes.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), rigid_body_count);
    ASSERT_EQ(ref_mesh.rb_update_modes.size(), rigid_body_count);
    ASSERT_EQ(state.x_coms.size(), rigid_body_count);
    ASSERT_EQ(state.v_coms.size(), rigid_body_count);
    ASSERT_EQ(state.orientations.size(), rigid_body_count);
    ASSERT_EQ(state.omega.size(), rigid_body_count);
    EXPECT_EQ(
        ref_mesh.rb_update_modes[0], RigidBodyUpdateMode::None);
    EXPECT_EQ(
        ref_mesh.rb_update_modes[1],
        RigidBodyUpdateMode::TranslationAndOrientation);

    constexpr double source_to_world_scale = 0.30 / 58.0;
    constexpr double nut_center_y = 0.35;
    constexpr double initial_insertion_source = 14.0;
    constexpr double bolt_center_y =
        nut_center_y
        + source_to_world_scale * (37.0 - initial_insertion_source);
    const std::array<Vec3, rigid_body_count> expected_centers{
        Vec3(0.0, nut_center_y, 0.0),
        Vec3(0.0, bolt_center_y, 0.0)};
    const std::array<Vec3, rigid_body_count> expected_material_extents{
        Vec3(30.0, 34.64102, 16.0) * source_to_world_scale,
        Vec3(30.0, 34.64102, 58.0) * source_to_world_scale};
    constexpr double kPi = 3.14159265358979323846;
    const Vec4 upright_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    const std::array<Vec4, rigid_body_count> expected_orientations{
        upright_orientation, upright_orientation};
    const std::array<Vec3, rigid_body_count> expected_omegas{
        Vec3::Zero(), Vec3::Zero()};
    // Both assets map material +z to world +y and retain one common uniform
    // scale. Their relative axial offset is four complete thread pitches, so
    // no additional yaw is needed to preserve the authored helical phase.
    const std::array<Vec3, rigid_body_count> expected_world_extents{
        Vec3(30.0, 16.0, 34.64102) * source_to_world_scale,
        Vec3(30.0, 58.0, 34.64102) * source_to_world_scale};

    std::array<Vec3, rigid_body_count> world_lower;
    std::array<Vec3, rigid_body_count> world_upper;
    for (int rb = 0; rb < rigid_body_count; ++rb) {
        SCOPED_TRACE(rb);
        EXPECT_TRUE(state.x_coms[rb].isApprox(
            expected_centers[static_cast<std::size_t>(rb)], 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
        EXPECT_TRUE(state.omega[rb].isApprox(
            expected_omegas[static_cast<std::size_t>(rb)], 0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            expected_orientations[static_cast<std::size_t>(rb)],
            1.0e-14));
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), nodes_per_body);
        ASSERT_EQ(ref_mesh.ref_positions[rb].size(), nodes_per_body);

        Vec3 material_lower = ref_mesh.ref_positions[rb].front();
        Vec3 material_upper = material_lower;
        world_lower[static_cast<std::size_t>(rb)] =
            state.deformed_positions[ref_mesh.rb_nodes[rb].front()];
        world_upper[static_cast<std::size_t>(rb)] =
            world_lower[static_cast<std::size_t>(rb)];
        double nodal_mass = 0.0;
        for (std::size_t local = 0;
             local < ref_mesh.rb_nodes[rb].size(); ++local) {
            const int expected_node = rb * nodes_per_body
                + static_cast<int>(local);
            const int node = ref_mesh.rb_nodes[rb][local];
            EXPECT_EQ(node, expected_node);
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            const Vec3 world_offset =
                state.deformed_positions[node] - state.x_coms[rb];
            EXPECT_TRUE(state.velocities[node].isApprox(
                expected_omegas[static_cast<std::size_t>(rb)].cross(
                    world_offset),
                1.0e-14));
            EXPECT_GT(ref_mesh.mass[node], 0.0);
            nodal_mass += ref_mesh.mass[node];
            material_lower = material_lower.cwiseMin(
                ref_mesh.ref_positions[rb][local]);
            material_upper = material_upper.cwiseMax(
                ref_mesh.ref_positions[rb][local]);
            world_lower[static_cast<std::size_t>(rb)] =
                world_lower[static_cast<std::size_t>(rb)].cwiseMin(
                    state.deformed_positions[node]);
            world_upper[static_cast<std::size_t>(rb)] =
                world_upper[static_cast<std::size_t>(rb)].cwiseMax(
                    state.deformed_positions[node]);
        }

        EXPECT_TRUE((0.5 * (material_lower + material_upper))
                        .isZero(1.0e-14));
        EXPECT_TRUE((material_upper - material_lower).isApprox(
            expected_material_extents[static_cast<std::size_t>(rb)],
            1.0e-14));
        EXPECT_TRUE((0.5
                     * (world_lower[static_cast<std::size_t>(rb)]
                        + world_upper[static_cast<std::size_t>(rb)]))
                        .isApprox(
                            expected_centers[static_cast<std::size_t>(rb)],
                            1.0e-14));
        EXPECT_TRUE((world_upper[static_cast<std::size_t>(rb)]
                     - world_lower[static_cast<std::size_t>(rb)])
                        .isApprox(
                            expected_world_extents[
                                static_cast<std::size_t>(rb)],
                            1.0e-14));
        EXPECT_NEAR(
            nodal_mass, ref_mesh.total_mass[rb],
            1.0e-14 * ref_mesh.total_mass[rb]);
    }

    // The octahedron volume is 4*a*b*c/3. Both mass checks use the same
    // source-to-world scale even though the source maximum extents differ.
    constexpr double nut_source_volume =
        (4.0 / 3.0) * source_half_x * source_half_y * nut_source_half_z;
    constexpr double bolt_source_volume =
        (4.0 / 3.0) * source_half_x * source_half_y * bolt_source_half_z;
    const double scaled_volume_factor =
        source_to_world_scale * source_to_world_scale
        * source_to_world_scale;
    EXPECT_NEAR(
        ref_mesh.total_mass[0],
        args.rigid_density * nut_source_volume * scaled_volume_factor,
        1.0e-12);
    EXPECT_NEAR(
        ref_mesh.total_mass[1],
        args.rigid_density * bolt_source_volume * scaled_volume_factor,
        1.0e-12);

    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        const int expected_rb = triangle / triangles_per_body;
        for (int local = 0; local < 3; ++local) {
            const int node = ref_mesh.tris[3 * triangle + local];
            EXPECT_EQ(ref_mesh.node_to_rb[node], expected_rb);
        }
    }

    const double initial_insertion_depth =
        world_upper[0].y() - world_lower[1].y();
    EXPECT_NEAR(
        initial_insertion_depth,
        initial_insertion_source * source_to_world_scale,
        1.0e-14);
    EXPECT_GT(initial_insertion_depth, 0.0);

    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int first = ref_mesh.tris[3 * triangle + local];
            const int second =
                ref_mesh.tris[3 * triangle + (local + 1) % 3];
            minimum_surface_edge = std::min(
                minimum_surface_edge,
                (state.deformed_positions[second]
                 - state.deformed_positions[first])
                    .norm());
        }
    }
    EXPECT_LT(params.d_hat, args.d_hat);
    EXPECT_NEAR(params.d_hat, 0.45 * minimum_surface_edge, 1.0e-15);
}

TEST(MixedExample,
     CrushersUseConfiguredInitialAngularVelocities) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_armadillo_crusher_scene_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);

    struct WorkingDirectoryGuard {
        fs::path previous;
        fs::path temporary;

        explicit WorkingDirectoryGuard(fs::path path)
            : previous(fs::current_path()), temporary(std::move(path)) {
            fs::current_path(temporary);
        }

        ~WorkingDirectoryGuard() {
            std::error_code error;
            fs::current_path(previous, error);
            fs::remove_all(temporary, error);
        }
    } working_directory(directory);

    fs::create_directories("example_obj/armadillo_coarse");
    fs::create_directories("example_obj/crusher");
    {
        std::ofstream nodes(
            "example_obj/armadillo_coarse/armadillo_5000f.1.node");
        ASSERT_TRUE(nodes.good());
        // One positive tetrahedron whose 151.19197-unit vertical extent
        // makes the production target use the intended common 0.001 scale.
        nodes.precision(17);
        nodes << "4 3 0 0\n"
              << "0 0 0 0\n"
              << "1 100 0 0\n"
              << "2 0 151.19197 0\n"
              << "3 0 0 50\n";
        ASSERT_TRUE(nodes.good());
    }
    {
        std::ofstream elements(
            "example_obj/armadillo_coarse/armadillo_5000f.1.ele");
        ASSERT_TRUE(elements.good());
        elements << "1 4 0\n"
                 << "0 0 1 2 3\n";
        ASSERT_TRUE(elements.good());
    }

    const auto write_octahedron = [](
        const char* filename, const double half_x,
        const double half_y, const double half_z) {
        std::ofstream obj(filename);
        ASSERT_TRUE(obj.good());
        obj.precision(17);
        obj << "v " << half_x << " 0 0\n"
            << "v " << -half_x << " 0 0\n"
            << "v 0 " << half_y << " 0\n"
            << "v 0 " << -half_y << " 0\n"
            << "v 0 0 " << half_z << "\n"
            << "v 0 0 " << -half_z << "\n"
            << "f 1 3 5\n"
            << "f 3 2 5\n"
            << "f 2 4 5\n"
            << "f 4 1 5\n"
            << "f 3 1 6\n"
            << "f 2 3 6\n"
            << "f 4 2 6\n"
            << "f 1 4 6\n";
        ASSERT_TRUE(obj.good());
    };
    // Both crusher fixtures are 400 source units along +z and 170 across
    // the tooth-tip direction, but distinct x radii catch accidental path
    // reuse. Their shared maximum extent again implies a 0.001 scale.
    constexpr double left_half_x = 68.0;
    constexpr double right_half_x = 80.0;
    constexpr double crusher_half_y = 85.0;
    constexpr double crusher_half_z = 200.0;
    write_octahedron(
        "example_obj/crusher/crusher_coarse_left.obj",
        left_half_x, crusher_half_y, crusher_half_z);
    write_octahedron(
        "example_obj/crusher/crusher_coarse_right.obj",
        right_half_x, crusher_half_y, crusher_half_z);

    IPCArgs3D args;
    args.solid_density = 731.0;
    args.rigid_density = 947.0;
    args.crusher_angular_speed = 17.25;
    args.d_hat = 1.0;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_armadillo_through_gear_crushers_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int solid_nodes = 4;
    constexpr int solid_tets = 1;
    constexpr int solid_boundary_triangles = 4;
    constexpr int nodes_per_crusher = 6;
    constexpr int triangles_per_crusher = 8;
    constexpr int crusher_count = 2;
    constexpr int total_nodes =
        solid_nodes + crusher_count * nodes_per_crusher;
    constexpr int total_triangles =
        solid_boundary_triangles
        + crusher_count * triangles_per_crusher;

    EXPECT_EQ(state.deformed_positions.size(), total_nodes);
    EXPECT_EQ(state.velocities.size(), total_nodes);
    EXPECT_EQ(ref_mesh.num_positions, total_nodes);
    EXPECT_EQ(ref_mesh.mass.size(), total_nodes);
    EXPECT_EQ(ref_mesh.node_to_rb.size(), total_nodes);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_EQ(ref_mesh.tets.size(), 4 * solid_tets);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), solid_tets);
    EXPECT_EQ(ref_mesh.tet_nodes.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.surface_nodes.size(), solid_nodes);
    EXPECT_EQ(ref_mesh.deformable_nodes.size(), solid_nodes);
    EXPECT_TRUE(ref_mesh.Dm_inverse.empty());
    EXPECT_TRUE(ref_mesh.area.empty());
    EXPECT_TRUE(ref_mesh.hinges.empty());
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());

    std::vector<unsigned char> is_tet_node(total_nodes, 0);
    std::vector<unsigned char> is_surface_node(total_nodes, 0);
    std::vector<unsigned char> is_deformable(total_nodes, 0);
    for (const int node : ref_mesh.tet_nodes)
        is_tet_node[static_cast<std::size_t>(node)] = 1;
    for (const int node : ref_mesh.surface_nodes)
        is_surface_node[static_cast<std::size_t>(node)] = 1;
    for (const int node : ref_mesh.deformable_nodes)
        is_deformable[static_cast<std::size_t>(node)] = 1;

    Vec3 armadillo_lower = state.deformed_positions[0];
    Vec3 armadillo_upper = armadillo_lower;
    double armadillo_mass = 0.0;
    for (int node = 0; node < solid_nodes; ++node) {
        armadillo_lower = armadillo_lower.cwiseMin(
            state.deformed_positions[node]);
        armadillo_upper = armadillo_upper.cwiseMax(
            state.deformed_positions[node]);
        armadillo_mass += ref_mesh.mass[node];
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 1);
        EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 1);
        EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 1);
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
    }
    // The builder centers the physical tetrahedral volume in x/z, not this
    // deliberately asymmetric fixture's bounding box.
    const Vec3 armadillo_aabb_center(0.025, 0.65025, 0.0125);
    EXPECT_TRUE((0.5 * (armadillo_lower + armadillo_upper)).isApprox(
        armadillo_aabb_center, 1.0e-14));
    EXPECT_TRUE((armadillo_upper - armadillo_lower).isApprox(
        Vec3(0.1, 0.15119197, 0.05), 1.0e-14));
    constexpr double source_to_world_scale = 0.001;
    constexpr double armadillo_source_volume =
        100.0 * 151.19197 * 50.0 / 6.0;
    EXPECT_NEAR(
        armadillo_mass,
        args.solid_density * armadillo_source_volume
            * source_to_world_scale * source_to_world_scale
            * source_to_world_scale,
        1.0e-12);
    ASSERT_EQ(ref_mesh.tets, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_GT(ref_mesh.tet_rest_data[0].measure, 0.0);
    Vec3 armadillo_volume_centroid = Vec3::Zero();
    for (const int node : ref_mesh.tets) {
        armadillo_volume_centroid += state.deformed_positions[node];
    }
    armadillo_volume_centroid *= 0.25;
    EXPECT_NEAR(armadillo_volume_centroid.x(), 0.0, 1.0e-14);
    EXPECT_NEAR(armadillo_volume_centroid.z(), 0.0, 1.0e-14);

    ASSERT_EQ(ref_mesh.rb_nodes.size(), crusher_count);
    ASSERT_EQ(ref_mesh.ref_positions.size(), crusher_count);
    ASSERT_EQ(ref_mesh.total_mass.size(), crusher_count);
    ASSERT_EQ(ref_mesh.I_hat.size(), crusher_count);
    ASSERT_EQ(ref_mesh.rb_update_modes.size(), crusher_count);
    ASSERT_EQ(state.x_coms.size(), crusher_count);
    ASSERT_EQ(state.v_coms.size(), crusher_count);
    ASSERT_EQ(state.orientations.size(), crusher_count);
    ASSERT_EQ(state.omega.size(), crusher_count);
    const std::array<Vec3, crusher_count> expected_centers{
        Vec3(-0.100, 0.5, 0.0), Vec3(0.100, 0.5, 0.0)};
    // Preserve the scene's initial signs: positive world z on the left and
    // negative world z on the right.
    const std::array<Vec3, crusher_count> expected_omegas{
        Vec3(0.0, 0.0, args.crusher_angular_speed),
        Vec3(0.0, 0.0, -args.crusher_angular_speed)};
    const std::array<double, crusher_count> source_half_x{
        left_half_x, right_half_x};
    for (int rb = 0; rb < crusher_count; ++rb) {
        SCOPED_TRACE(rb);
        EXPECT_EQ(ref_mesh.rb_update_modes[rb],
            RigidBodyUpdateMode::OrientationOnly);
        EXPECT_TRUE(state.x_coms[rb].isApprox(
            expected_centers[static_cast<std::size_t>(rb)], 1.0e-14));
        EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
        EXPECT_TRUE(state.orientations[rb].isApprox(
            Vec4(1.0, 0.0, 0.0, 0.0), 0.0));
        EXPECT_TRUE(state.omega[rb].isApprox(
            expected_omegas[static_cast<std::size_t>(rb)], 0.0));
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), nodes_per_crusher);

        Vec3 lower =
            state.deformed_positions[ref_mesh.rb_nodes[rb].front()];
        Vec3 upper = lower;
        double nodal_mass = 0.0;
        for (int local = 0;
             local < static_cast<int>(ref_mesh.rb_nodes[rb].size());
             ++local) {
            const int node = ref_mesh.rb_nodes[rb][local];
            lower = lower.cwiseMin(state.deformed_positions[node]);
            upper = upper.cwiseMax(state.deformed_positions[node]);
            nodal_mass += ref_mesh.mass[node];
            EXPECT_EQ(ref_mesh.node_to_rb[node], rb);
            EXPECT_EQ(is_tet_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_surface_node[static_cast<std::size_t>(node)], 0);
            EXPECT_EQ(is_deformable[static_cast<std::size_t>(node)], 0);
            const Vec3 world_offset =
                state.deformed_positions[node] - state.x_coms[rb];
            EXPECT_TRUE(state.velocities[node].isApprox(
                expected_omegas[static_cast<std::size_t>(rb)].cross(
                    world_offset),
                1.0e-14));
        }

        EXPECT_TRUE((0.5 * (lower + upper)).isApprox(
            expected_centers[static_cast<std::size_t>(rb)], 1.0e-14));
        EXPECT_TRUE((upper - lower).isApprox(
            Vec3(
                2.0 * source_half_x[static_cast<std::size_t>(rb)]
                    * source_to_world_scale,
                2.0 * crusher_half_y * source_to_world_scale,
                2.0 * crusher_half_z * source_to_world_scale),
            1.0e-14));
        EXPECT_NEAR(nodal_mass, ref_mesh.total_mass[rb], 1.0e-14);

        const double source_volume = (4.0 / 3.0)
            * source_half_x[static_cast<std::size_t>(rb)]
            * crusher_half_y * crusher_half_z;
        EXPECT_NEAR(
            ref_mesh.total_mass[rb],
            args.rigid_density * source_volume
                * source_to_world_scale * source_to_world_scale
                * source_to_world_scale,
            1.0e-12);
    }

    // Its AABB starts inside the tip-radius envelope; the production meshes'
    // actual fluted surfaces remain separated at this placement.
    EXPECT_NEAR(armadillo_lower.y(), 0.574654015, 1.0e-14);
    EXPECT_NEAR(0.585 - armadillo_lower.y(), 0.010345985, 1.0e-14);

    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        const int expected_owner = triangle < solid_boundary_triangles
            ? -1
            : (triangle - solid_boundary_triangles)
                    / triangles_per_crusher;
        for (int local = 0; local < 3; ++local) {
            EXPECT_EQ(
                ref_mesh.node_to_rb[ref_mesh.tris[3 * triangle + local]],
                expected_owner);
        }
    }

    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    // The 50-source-unit Armadillo edge is the shortest fixture edge after
    // the shared scale, so the scene clamps d_hat to 0.45 * 0.05.
    EXPECT_LT(params.d_hat, args.d_hat);
    EXPECT_NEAR(params.d_hat, 0.0225, 1.0e-15);
}

TEST(MixedExample, RolledClothIsPinnedAboveAFixedInclinedWedge) {
    IPCArgs3D args;
    args.density = 37.0;
    args.thickness = 0.004;
    args.rigid_density = 123.0;
    // Deliberately exceed the cloth-grid edge bound so the scene's analytic
    // activation-distance clamp and its dependent initial clearance execute.
    args.d_hat = 0.1;
    args.k_barrier = 719.0;
    args.gx = 0.25;
    args.gy = -7.5;
    args.gz = -0.75;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_cloth_unrolling_down_fixed_ramp_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 24;
    constexpr int cloth_ny = 140;
    constexpr int cloth_nodes =
        (cloth_nx + 1) * (cloth_ny + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_ny;
    constexpr int cloth_hinges =
        cloth_nx * cloth_ny
        + cloth_nx * (cloth_ny - 1)
        + (cloth_nx - 1) * cloth_ny;
    constexpr int wedge_nodes = 6;
    constexpr int wedge_triangles = 8;
    constexpr int total_nodes = cloth_nodes + wedge_nodes;
    constexpr int total_triangles =
        cloth_triangles + wedge_triangles;

    constexpr double cloth_width = 0.90;
    constexpr double ramp_width = 1.40;
    constexpr double ramp_height = 2.40;
    constexpr double ramp_back_z = -1.20;
    constexpr double ramp_front_z = 1.20;
    constexpr int leader_rows = 7;
    constexpr int roll_offset_reference_rows = 8;
    constexpr double outer_radius = 0.0925;
    constexpr double inner_radius = 0.05;
    constexpr double layer_pitch = 0.0065;
    constexpr double scene_d_hat_cap = 0.005;
    constexpr double minimum_clearance = 0.002;
    constexpr double clearance_margin = 1.0e-4;
    constexpr double kPi = 3.14159265358979323846;
    const double spiral_a = layer_pitch / (2.0 * kPi);
    const auto spiral_primitive = [spiral_a](const double radius) {
        return 0.5 * (
            radius * std::hypot(radius, spiral_a)
            + spiral_a * spiral_a * std::asinh(radius / spiral_a));
    };
    const double theta_max =
        (outer_radius - inner_radius) / spiral_a;
    const double spiral_length =
        (spiral_primitive(outer_radius)
         - spiral_primitive(inner_radius))
        / spiral_a;
    const double roll_downslope_offset =
        static_cast<double>(roll_offset_reference_rows) * spiral_length
        / static_cast<double>(cloth_ny - roll_offset_reference_rows);
    const double leader_material_length =
        static_cast<double>(leader_rows) * spiral_length
        / static_cast<double>(cloth_ny - leader_rows);
    const double cloth_length = leader_material_length + spiral_length;
    const double ramp_length = std::hypot(
        ramp_height, ramp_front_z - ramp_back_z);
    const Vec3 ramp_top(0.0, ramp_height, ramp_back_z);
    const Vec3 downhill =
        Vec3(0.0, -ramp_height, ramp_front_z - ramp_back_z)
        / ramp_length;
    const Vec3 ramp_normal =
        Vec3(0.0, ramp_front_z - ramp_back_z, ramp_height)
        / ramp_length;
    EXPECT_NEAR(-downhill.y(), downhill.z(), 1.0e-15);
    EXPECT_NEAR(ramp_normal.y(), ramp_normal.z(), 1.0e-15);
    EXPECT_NEAR(
        downhill.z(), std::sqrt(0.5), 1.0e-15);
    EXPECT_NEAR(
        ramp_normal.y(), std::sqrt(0.5), 1.0e-15);

    ASSERT_EQ(state.deformed_positions.size(), total_nodes);
    ASSERT_EQ(state.velocities.size(), total_nodes);
    EXPECT_EQ(ref_mesh.num_positions, total_nodes);
    ASSERT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    ASSERT_EQ(ref_mesh.mass.size(), total_nodes);
    ASSERT_EQ(ref_mesh.node_to_rb.size(), total_nodes);
    EXPECT_EQ(X.size(), cloth_nodes);
    EXPECT_EQ(ref_mesh.Dm_inverse.size(), cloth_triangles);
    EXPECT_EQ(ref_mesh.area.size(), cloth_triangles);
    EXPECT_EQ(ref_mesh.hinges.size(), cloth_hinges);

    const double rest_dx = cloth_width / cloth_nx;
    const double rest_ds = cloth_length / cloth_ny;
    const double expected_d_hat = std::min(
        std::min(args.d_hat, scene_d_hat_cap),
        0.45 * std::min(rest_dx, rest_ds));
    EXPECT_LT(params.d_hat, args.d_hat);
    EXPECT_NEAR(params.d_hat, expected_d_hat, 1.0e-15);
    EXPECT_DOUBLE_EQ(params.d_hat, scene_d_hat_cap);
    const double clearance = std::max(
        std::max(expected_d_hat, 0.0) + clearance_margin,
        minimum_clearance);
    EXPECT_GT(clearance, params.d_hat);
    const double outer_spiral_derivative_norm =
        std::hypot(outer_radius, spiral_a);
    const double tangent_A = outer_radius
        + leader_material_length * spiral_a
            / outer_spiral_derivative_norm;
    const double tangent_G = leader_material_length * outer_radius
        / outer_spiral_derivative_norm;
    const double center_to_pin_distance_squared =
        tangent_A * tangent_A + tangent_G * tangent_G;
    const double pin_normal_delta_squared =
        center_to_pin_distance_squared
        - roll_downslope_offset * roll_downslope_offset;
    ASSERT_GT(pin_normal_delta_squared, 0.0);
    const double pin_normal_delta =
        std::sqrt(pin_normal_delta_squared);
    const double pin_clearance =
        outer_radius + clearance + pin_normal_delta;
    const double spiral_phase_cosine = (
        tangent_G * roll_downslope_offset
        - tangent_A * pin_normal_delta)
        / center_to_pin_distance_squared;
    const double spiral_phase_sine = (
        -tangent_A * roll_downslope_offset
        - tangent_G * pin_normal_delta)
        / center_to_pin_distance_squared;
    const double spiral_phase = std::atan2(
        spiral_phase_sine, spiral_phase_cosine);
    const double tangent_downslope =
        roll_downslope_offset
        + outer_radius * spiral_phase_sine;
    const double tangent_clearance =
        outer_radius + clearance
        - outer_radius * spiral_phase_cosine;

    EXPECT_TRUE(params.gravity.isApprox(
        Vec3(args.gx, args.gy, args.gz), 0.0));
    EXPECT_DOUBLE_EQ(params.k_barrier, args.k_barrier);
    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    const auto cloth_node = [](const int i, const int j) {
        return j * (cloth_nx + 1) + i;
    };

    // The constitutive metric remains the original flat rectangle even
    // though its initial world-space pose is a leader followed by a spiral.
    const double expected_triangle_area = 0.5 * rest_dx * rest_ds;
    for (int j = 0; j <= cloth_ny; ++j) {
        for (int i = 0; i <= cloth_nx; ++i) {
            const int node = cloth_node(i, j);
            EXPECT_TRUE(X[static_cast<std::size_t>(node)].isApprox(
                Vec2(i * rest_dx, j * rest_ds), 1.0e-14));
        }
    }
    for (int j = 0; j < cloth_ny; ++j) {
        for (int i = 0; i < cloth_nx; ++i) {
            const int triangle = 2 * (j * cloth_nx + i);
            const int v00 = cloth_node(i, j);
            const int v10 = cloth_node(i + 1, j);
            const int v01 = cloth_node(i, j + 1);
            const int v11 = cloth_node(i + 1, j + 1);
            EXPECT_EQ(
                (std::array<int, 3>{
                    ref_mesh.tris[3 * triangle + 0],
                    ref_mesh.tris[3 * triangle + 1],
                    ref_mesh.tris[3 * triangle + 2]}),
                (std::array<int, 3>{v00, v10, v11}));
            EXPECT_EQ(
                (std::array<int, 3>{
                    ref_mesh.tris[3 * triangle + 3],
                    ref_mesh.tris[3 * triangle + 4],
                    ref_mesh.tris[3 * triangle + 5]}),
                (std::array<int, 3>{v00, v11, v01}));
        }
    }
    double rest_area = 0.0;
    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        EXPECT_NEAR(
            ref_mesh.area[static_cast<std::size_t>(triangle)],
            expected_triangle_area, 1.0e-16);
        rest_area += ref_mesh.area[static_cast<std::size_t>(triangle)];

        const int v0 = ref_mesh.tris[3 * triangle + 0];
        const int v1 = ref_mesh.tris[3 * triangle + 1];
        const int v2 = ref_mesh.tris[3 * triangle + 2];
        Mat22 Dm;
        Dm.col(0) = X[static_cast<std::size_t>(v1)]
            - X[static_cast<std::size_t>(v0)];
        Dm.col(1) = X[static_cast<std::size_t>(v2)]
            - X[static_cast<std::size_t>(v0)];
        EXPECT_TRUE((
            ref_mesh.Dm_inverse[static_cast<std::size_t>(triangle)] * Dm)
            .isApprox(Mat22::Identity(), 1.0e-13));
    }
    EXPECT_NEAR(rest_area, cloth_width * cloth_length, 1.0e-12);

    double maximum_initial_bending = 0.0;
    for (const Hinge& hinge : ref_mesh.hinges) {
        EXPECT_NEAR(hinge.bar_theta, 0.0, 1.0e-14);
        EXPECT_GT(hinge.c_e, 0.0);
        HingeDef deformed_hinge;
        for (int local = 0; local < 4; ++local) {
            ASSERT_GE(hinge.v[local], 0);
            ASSERT_LT(hinge.v[local], cloth_nodes);
            deformed_hinge.x[local] = state.deformed_positions[
                static_cast<std::size_t>(hinge.v[local])];
        }
        maximum_initial_bending = std::max(
            maximum_initial_bending,
            std::abs(bending_theta(deformed_hinge) - hinge.bar_theta));
    }
    EXPECT_GT(maximum_initial_bending, 1.0e-3);

    EXPECT_NEAR(
        roll_downslope_offset, 0.17742040770396708, 1.0e-15);
    EXPECT_NEAR(
        leader_material_length, 0.154075617216603, 1.0e-15);
    EXPECT_NEAR(cloth_length, 3.08151234433206, 1.0e-15);
    EXPECT_NEAR(
        leader_rows * rest_ds, leader_material_length, 1.0e-15);
    EXPECT_NEAR(
        center_to_pin_distance_squared,
        0.03261431053512491, 1.0e-16);
    EXPECT_NEAR(pin_normal_delta, 0.03370918962661382, 1.0e-15);
    EXPECT_NEAR(pin_clearance, 0.1313091896266138, 1.0e-15);
    EXPECT_NEAR(spiral_phase_cosine, 0.7407259652988947, 1.0e-15);
    EXPECT_NEAR(spiral_phase_sine, -0.671807297022011, 1.0e-15);
    EXPECT_NEAR(spiral_phase, -0.7366459958573437, 1.0e-15);
    EXPECT_NEAR(tangent_downslope, 0.11527823272943105, 1.0e-15);
    EXPECT_NEAR(tangent_clearance, 0.02908284820985224, 1.0e-15);
    EXPECT_GT(pin_clearance, outer_radius + clearance);
    EXPECT_GT(tangent_downslope, 0.0);
    EXPECT_LT(tangent_downslope, roll_downslope_offset);
    EXPECT_GT(tangent_clearance, clearance);
    EXPECT_LT(tangent_clearance, outer_radius + clearance);
    EXPECT_GT(spiral_phase_cosine, 0.0);
    EXPECT_LT(spiral_phase_sine, 0.0);

    const Vec3 roll_center =
        ramp_top + roll_downslope_offset * downhill
        + (outer_radius + clearance) * ramp_normal;
    const auto expected_centerline = [&](const double s) -> Vec3 {
        if (s <= leader_material_length) {
            const double fraction = s / leader_material_length;
            const double normal_offset =
                (1.0 - fraction) * pin_clearance
                + fraction * tangent_clearance;
            return ramp_top
                + fraction * tangent_downslope * downhill
                + normal_offset * ramp_normal;
        }

        const double target_arc = s - leader_material_length;
        double lower_theta = 0.0;
        double upper_theta = theta_max;
        for (int iteration = 0; iteration < 100; ++iteration) {
            const double theta = 0.5 * (lower_theta + upper_theta);
            const double radius = outer_radius - spiral_a * theta;
            const double arc =
                (spiral_primitive(outer_radius)
                 - spiral_primitive(radius))
                / spiral_a;
            if (arc < target_arc)
                lower_theta = theta;
            else
                upper_theta = theta;
        }
        const double theta = 0.5 * (lower_theta + upper_theta);
        const double radius = outer_radius - spiral_a * theta;
        const double angle = spiral_phase + theta;
        return roll_center + radius * (
            std::sin(angle) * downhill
            - std::cos(angle) * ramp_normal);
    };

    double minimum_ramp_distance =
        std::numeric_limits<double>::infinity();
    std::vector<Vec3> cloth_centerlines;
    cloth_centerlines.reserve(cloth_ny + 1);
    for (int j = 0; j <= cloth_ny; ++j) {
        const double s = j * rest_ds;
        const Vec3 centerline = expected_centerline(s);
        for (int i = 0; i <= cloth_nx; ++i) {
            const int node = cloth_node(i, j);
            const Vec3 expected = centerline
                + (0.5 - static_cast<double>(i) / cloth_nx)
                    * cloth_width * Vec3::UnitX();
            EXPECT_TRUE(state.deformed_positions[
                static_cast<std::size_t>(node)].isApprox(
                    expected, 2.0e-12));
            EXPECT_TRUE(state.velocities[
                static_cast<std::size_t>(node)].isZero(0.0));
            const double ramp_distance =
                (state.deformed_positions[static_cast<std::size_t>(node)]
                 - ramp_top).dot(ramp_normal);
            EXPECT_GE(ramp_distance, clearance - 2.0e-12);
            minimum_ramp_distance = std::min(
                minimum_ramp_distance, ramp_distance);
        }
        EXPECT_NEAR(
            (state.deformed_positions[
                 static_cast<std::size_t>(cloth_node(0, j))]
             - state.deformed_positions[
                 static_cast<std::size_t>(cloth_node(cloth_nx, j))])
                .norm(),
            cloth_width, 1.0e-14);
        cloth_centerlines.push_back(0.5 * (
            state.deformed_positions[static_cast<std::size_t>(
                cloth_node(0, j))]
            + state.deformed_positions[static_cast<std::size_t>(
                cloth_node(cloth_nx, j))]));
    }
    EXPECT_GT(minimum_ramp_distance, clearance);
    EXPECT_LT(minimum_ramp_distance, clearance + layer_pitch);
    const Vec3 pin_centerline =
        ramp_top + pin_clearance * ramp_normal;
    const Vec3 expected_tangent_centerline =
        ramp_top + tangent_downslope * downhill
        + tangent_clearance * ramp_normal;
    const Vec3 expected_leader_tangent =
        ((outer_radius * spiral_phase_cosine
          - spiral_a * spiral_phase_sine)
         / outer_spiral_derivative_norm)
            * downhill
        + ((outer_radius * spiral_phase_sine
            + spiral_a * spiral_phase_cosine)
           / outer_spiral_derivative_norm)
            * ramp_normal;
    EXPECT_NEAR(expected_leader_tangent.norm(), 1.0, 1.0e-15);
    EXPECT_TRUE((
        pin_centerline
        + leader_material_length * expected_leader_tangent)
        .isApprox(expected_tangent_centerline, 2.0e-14));
    double previous_leader_offset =
        std::numeric_limits<double>::infinity();
    Vec3 previous_leader_centerline = Vec3::Zero();
    for (int j = 0; j <= leader_rows; ++j) {
        const Vec3 centerline = 0.5 * (
            state.deformed_positions[static_cast<std::size_t>(
                cloth_node(0, j))]
            + state.deformed_positions[static_cast<std::size_t>(
                cloth_node(cloth_nx, j))]);
        const double normal_offset =
            (centerline - ramp_top).dot(ramp_normal);
        const double fraction =
            static_cast<double>(j) / leader_rows;
        const double expected_offset =
            (1.0 - fraction) * pin_clearance
            + fraction * tangent_clearance;
        EXPECT_NEAR(normal_offset, expected_offset, 2.0e-14);
        EXPECT_NEAR(
            (centerline - ramp_top).dot(downhill),
            fraction * tangent_downslope, 2.0e-14);
        EXPECT_GE(normal_offset, clearance - 2.0e-14);
        EXPECT_GT(normal_offset, params.d_hat);
        const double remaining_leader_length =
            (1.0 - fraction) * leader_material_length;
        const double expected_roll_distance_squared =
            outer_radius * outer_radius
            + 2.0 * remaining_leader_length * outer_radius * spiral_a
                / outer_spiral_derivative_norm
            + remaining_leader_length * remaining_leader_length;
        EXPECT_NEAR(
            (centerline - roll_center).squaredNorm(),
            expected_roll_distance_squared, 2.0e-15);
        if (j < leader_rows)
            EXPECT_GT((centerline - roll_center).norm(), outer_radius);
        else
            EXPECT_NEAR(
                (centerline - roll_center).norm(),
                outer_radius, 2.0e-14);
        if (j > 0) {
            EXPECT_LT(normal_offset, previous_leader_offset);
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).norm(),
                rest_ds, 2.0e-14);
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).dot(downhill),
                tangent_downslope / leader_rows, 2.0e-14);
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).dot(ramp_normal),
                (tangent_clearance - pin_clearance) / leader_rows,
                2.0e-14);
            for (int i = 0; i <= cloth_nx; ++i) {
                EXPECT_NEAR(
                    (state.deformed_positions[static_cast<std::size_t>(
                         cloth_node(i, j))]
                     - state.deformed_positions[static_cast<std::size_t>(
                         cloth_node(i, j - 1))])
                        .norm(),
                    rest_ds, 2.0e-14);
            }
        }
        previous_leader_offset = normal_offset;
        previous_leader_centerline = centerline;
    }
    const Vec3 tangent_centerline = 0.5 * (
        state.deformed_positions[static_cast<std::size_t>(
            cloth_node(0, leader_rows))]
        + state.deformed_positions[static_cast<std::size_t>(
            cloth_node(cloth_nx, leader_rows))]);
    EXPECT_TRUE(tangent_centerline.isApprox(
        expected_tangent_centerline, 2.0e-14));
    EXPECT_NEAR(
        (tangent_centerline - ramp_top).dot(ramp_normal),
        tangent_clearance, 2.0e-14);
    EXPECT_NEAR(
        (tangent_centerline - ramp_top).dot(downhill),
        tangent_downslope, 2.0e-14);
    const Vec3 tangent_radius = tangent_centerline - roll_center;
    const Vec3 actual_leader_tangent =
        (tangent_centerline - pin_centerline).normalized();
    EXPECT_NEAR(tangent_radius.norm(), outer_radius, 2.0e-14);
    EXPECT_TRUE(actual_leader_tangent.isApprox(
        expected_leader_tangent, 2.0e-14));
    // The straight leader is C1 with the shrinking spiral. Its tangent has
    // the expected small inward radial component, unlike a circle tangent.
    EXPECT_NEAR(
        tangent_radius.dot(actual_leader_tangent),
        -outer_radius * spiral_a / outer_spiral_derivative_norm,
        2.0e-15);
    EXPECT_LT(tangent_radius.dot(actual_leader_tangent), 0.0);
    // The seven-row leader must not move the roll: the center retains the
    // exact downslope offset authored by the former eight-row leader.
    const double former_eight_row_leader_length =
        static_cast<double>(roll_offset_reference_rows) * spiral_length
        / static_cast<double>(cloth_ny - roll_offset_reference_rows);
    EXPECT_DOUBLE_EQ(
        roll_downslope_offset, former_eight_row_leader_length);
    const Vec3 former_roll_center =
        ramp_top + former_eight_row_leader_length * downhill
        + (outer_radius + clearance) * ramp_normal;
    EXPECT_TRUE(roll_center.isApprox(former_roll_center, 0.0));
    const Vec3 first_spiral_centerline = 0.5 * (
        state.deformed_positions[static_cast<std::size_t>(
            cloth_node(0, leader_rows + 1))]
        + state.deformed_positions[static_cast<std::size_t>(
            cloth_node(cloth_nx, leader_rows + 1))]);
    EXPECT_GT(
        (first_spiral_centerline - ramp_top).dot(ramp_normal),
        clearance);

    // All rows have the same x span, so the distance between two segments of
    // this downslope/normal centerline is the attainable surface distance at
    // equal x. Check every nonincident pair, including leader-versus-spiral
    // pairs, rather than relying only on the analytic outer-disk argument.
    double nearest_nonlocal_cloth_segment_distance =
        std::numeric_limits<double>::infinity();
    double nearest_leader_spiral_segment_distance =
        std::numeric_limits<double>::infinity();
    for (int first = 0; first < cloth_ny; ++first) {
        for (int second = first + 2; second < cloth_ny; ++second) {
            const double distance = segment_segment_distance(
                cloth_centerlines[static_cast<std::size_t>(first)],
                cloth_centerlines[static_cast<std::size_t>(first + 1)],
                cloth_centerlines[static_cast<std::size_t>(second)],
                cloth_centerlines[static_cast<std::size_t>(second + 1)])
                                        .distance;
            nearest_nonlocal_cloth_segment_distance = std::min(
                nearest_nonlocal_cloth_segment_distance, distance);
            if (first < leader_rows && second >= leader_rows) {
                nearest_leader_spiral_segment_distance = std::min(
                    nearest_leader_spiral_segment_distance, distance);
            }
        }
    }
    EXPECT_GT(nearest_nonlocal_cloth_segment_distance, params.d_hat);
    EXPECT_GT(nearest_leader_spiral_segment_distance, params.d_hat);
    EXPECT_TRUE(std::isfinite(nearest_leader_spiral_segment_distance));

    // Adjacent turns are nonlocal in mesh connectivity. Their nearest sampled
    // row centers should retain the authored radial pitch, while the true
    // piecewise-linear centerline segments must still remain outside d_hat.
    // Because every cloth row is a straight x-line with identical span, this
    // two-dimensional centerline gap is also attainable by the cloth surface
    // at equal x and is the relevant inter-layer separation.
    std::vector<Vec3> spiral_centerlines;
    spiral_centerlines.reserve(cloth_ny - leader_rows + 1);
    for (int j = leader_rows; j <= cloth_ny; ++j) {
        spiral_centerlines.push_back(0.5 * (
            state.deformed_positions[static_cast<std::size_t>(
                cloth_node(0, j))]
            + state.deformed_positions[static_cast<std::size_t>(
                cloth_node(cloth_nx, j))]));
    }

    double nearest_nonlocal_row_distance =
        std::numeric_limits<double>::infinity();
    int nearest_first_row = -1;
    int nearest_second_row = -1;
    for (int first = 0;
         first < static_cast<int>(spiral_centerlines.size()); ++first) {
        for (int second = first + 2;
             second < static_cast<int>(spiral_centerlines.size()); ++second) {
            const double distance = (
                spiral_centerlines[static_cast<std::size_t>(second)]
                - spiral_centerlines[static_cast<std::size_t>(first)])
                    .norm();
            if (distance < nearest_nonlocal_row_distance) {
                nearest_nonlocal_row_distance = distance;
                nearest_first_row = first;
                nearest_second_row = second;
            }
        }
    }
    ASSERT_GE(nearest_first_row, 0);
    ASSERT_GT(nearest_second_row, nearest_first_row + 1);
    EXPECT_GT(nearest_nonlocal_row_distance, params.d_hat);
    EXPECT_NEAR(
        nearest_nonlocal_row_distance, layer_pitch, 2.0e-5);
    const Vec3 first_turn_offset =
        spiral_centerlines[static_cast<std::size_t>(nearest_first_row)]
        - roll_center;
    const Vec3 second_turn_offset =
        spiral_centerlines[static_cast<std::size_t>(nearest_second_row)]
        - roll_center;
    EXPECT_GT(
        first_turn_offset.normalized().dot(
            second_turn_offset.normalized()),
        0.999);
    EXPECT_NEAR(
        std::abs(first_turn_offset.norm() - second_turn_offset.norm()),
        layer_pitch, 1.0e-5);

    double nearest_nonlocal_segment_distance =
        std::numeric_limits<double>::infinity();
    for (int first = 0;
         first + 1 < static_cast<int>(spiral_centerlines.size()); ++first) {
        for (int second = first + 2;
             second + 1 < static_cast<int>(spiral_centerlines.size());
             ++second) {
            nearest_nonlocal_segment_distance = std::min(
                nearest_nonlocal_segment_distance,
                segment_segment_distance(
                    spiral_centerlines[static_cast<std::size_t>(first)],
                    spiral_centerlines[static_cast<std::size_t>(first + 1)],
                    spiral_centerlines[static_cast<std::size_t>(second)],
                    spiral_centerlines[static_cast<std::size_t>(second + 1)])
                    .distance);
        }
    }
    EXPECT_GT(nearest_nonlocal_segment_distance, params.d_hat);
    EXPECT_LT(nearest_nonlocal_segment_distance, layer_pitch);

    const Vec3 final_offset = expected_centerline(cloth_length) - roll_center;
    EXPECT_NEAR(final_offset.norm(), inner_radius, 2.0e-12);

    ASSERT_EQ(pins.size(), cloth_nx + 1);
    for (int i = 0; i <= cloth_nx; ++i) {
        SCOPED_TRACE(i);
        const Pin& pin = pins[static_cast<std::size_t>(i)];
        EXPECT_EQ(pin.vertex_index, cloth_node(i, 0));
        EXPECT_TRUE(pin.target_position.isApprox(
            state.deformed_positions[
                static_cast<std::size_t>(pin.vertex_index)],
            0.0));
        EXPECT_NEAR(
            (pin.target_position - ramp_top).dot(ramp_normal),
            pin_clearance, 1.0e-14);
        EXPECT_NEAR(
            (pin.target_position - ramp_top).dot(downhill),
            0.0, 1.0e-14);
    }

    const std::array<Vec3, wedge_nodes> expected_wedge_positions{
        Vec3(-0.5 * ramp_width, 0.0, ramp_back_z),
        Vec3( 0.5 * ramp_width, 0.0, ramp_back_z),
        Vec3(-0.5 * ramp_width, ramp_height, ramp_back_z),
        Vec3( 0.5 * ramp_width, ramp_height, ramp_back_z),
        Vec3(-0.5 * ramp_width, 0.0, ramp_front_z),
        Vec3( 0.5 * ramp_width, 0.0, ramp_front_z)};
    const std::array<int, 3 * wedge_triangles> expected_wedge_tris{
        0, 1, 5, 0, 5, 4,
        0, 2, 3, 0, 3, 1,
        2, 4, 5, 2, 5, 3,
        0, 4, 2, 1, 3, 5};
    double maximum_wedge_ramp_coordinate =
        -std::numeric_limits<double>::infinity();
    for (int local = 0; local < wedge_nodes; ++local) {
        const int node = cloth_nodes + local;
        EXPECT_TRUE(state.deformed_positions[
            static_cast<std::size_t>(node)].isApprox(
                expected_wedge_positions[static_cast<std::size_t>(local)],
                0.0));
        EXPECT_TRUE(state.velocities[
            static_cast<std::size_t>(node)].isZero(0.0));
        const double ramp_coordinate =
            (state.deformed_positions[static_cast<std::size_t>(node)]
             - ramp_top).dot(ramp_normal);
        EXPECT_LE(ramp_coordinate, 1.0e-14);
        maximum_wedge_ramp_coordinate = std::max(
            maximum_wedge_ramp_coordinate, ramp_coordinate);
    }
    EXPECT_NEAR(maximum_wedge_ramp_coordinate, 0.0, 1.0e-14);
    for (int local = 0; local < 3 * wedge_triangles; ++local) {
        EXPECT_EQ(
            ref_mesh.tris[3 * cloth_triangles + local],
            cloth_nodes
                + expected_wedge_tris[static_cast<std::size_t>(local)]);
    }

    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.tet_rest_data.empty());
    EXPECT_TRUE(ref_mesh.tet_adj.empty());
    EXPECT_TRUE(ref_mesh.tet_nodes.empty());
    EXPECT_TRUE(ref_mesh.surface_nodes.empty());
    ASSERT_EQ(ref_mesh.rb_nodes.size(), 1U);
    ASSERT_EQ(ref_mesh.ref_positions.size(), 1U);
    ASSERT_EQ(ref_mesh.total_mass.size(), 1U);
    ASSERT_EQ(ref_mesh.I_hat.size(), 1U);
    ASSERT_EQ(ref_mesh.rb_update_modes.size(), 1U);
    ASSERT_EQ(state.x_coms.size(), 1U);
    ASSERT_EQ(state.v_coms.size(), 1U);
    ASSERT_EQ(state.orientations.size(), 1U);
    ASSERT_EQ(state.omega.size(), 1U);
    EXPECT_EQ(
        ref_mesh.rb_update_modes[0], RigidBodyUpdateMode::None);
    EXPECT_TRUE(state.x_coms[0].isApprox(
        Vec3(0.0, ramp_height / 3.0,
             (2.0 * ramp_back_z + ramp_front_z) / 3.0),
        1.0e-14));
    EXPECT_TRUE(state.v_coms[0].isZero(0.0));
    EXPECT_TRUE(state.orientations[0].isApprox(
        Vec4(1.0, 0.0, 0.0, 0.0), 0.0));
    EXPECT_TRUE(state.omega[0].isZero(0.0));
    ASSERT_EQ(ref_mesh.rb_nodes[0].size(), wedge_nodes);

    const double wedge_volume =
        0.5 * ramp_width * ramp_height
        * (ramp_front_z - ramp_back_z);
    EXPECT_NEAR(wedge_volume, 4.032, 1.0e-15);
    const double expected_wedge_mass =
        args.rigid_density * wedge_volume;
    EXPECT_NEAR(
        ref_mesh.total_mass[0], expected_wedge_mass, 1.0e-13);
    std::vector<double> wedge_masses;
    wedge_masses.reserve(wedge_nodes);
    for (int node = 0; node < total_nodes; ++node) {
        if (node < cloth_nodes) {
            EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
            EXPECT_DOUBLE_EQ(ref_mesh.mass[node], 0.0);
        } else {
            EXPECT_EQ(ref_mesh.node_to_rb[node], 0);
            EXPECT_DOUBLE_EQ(
                ref_mesh.mass[node], expected_wedge_mass / wedge_nodes);
            wedge_masses.push_back(ref_mesh.mass[node]);
        }
    }
    ASSERT_EQ(ref_mesh.deformable_nodes.size(), cloth_nodes);
    for (int node = 0; node < cloth_nodes; ++node)
        EXPECT_EQ(ref_mesh.deformable_nodes[node], node);

    // The mixed-scene mass pass fills only the shell entries and must leave
    // the fixed collider's already assigned proxy masses untouched.
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    double total_cloth_mass = 0.0;
    for (int node = 0; node < cloth_nodes; ++node) {
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        total_cloth_mass += ref_mesh.mass[node];
    }
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * cloth_width * cloth_length,
        1.0e-12);
    EXPECT_TRUE(std::equal(
        wedge_masses.begin(), wedge_masses.end(),
        ref_mesh.mass.begin() + cloth_nodes));

    double minimum_mesh_edge = std::numeric_limits<double>::infinity();
    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int first = ref_mesh.tris[3 * triangle + local];
            const int second =
                ref_mesh.tris[3 * triangle + (local + 1) % 3];
            minimum_mesh_edge = std::min(
                minimum_mesh_edge,
                (state.deformed_positions[static_cast<std::size_t>(second)]
                 - state.deformed_positions[static_cast<std::size_t>(first)])
                    .norm());
        }
    }
    EXPECT_LT(params.d_hat, 0.5 * minimum_mesh_edge);
}

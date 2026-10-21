#include "make_shape.h"
#include "example.h"
#include "mesh_utils.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace {

void expect_sdf_material_pose_near(
    const SDFMaterialPose& actual,
    const SDFMaterialPose& expected,
    const double tolerance = 1.0e-14) {
    EXPECT_TRUE(actual.rotation.isApprox(expected.rotation, tolerance))
        << "actual rotation:\n" << actual.rotation
        << "\nexpected rotation:\n" << expected.rotation;
    EXPECT_TRUE(actual.translation.isApprox(expected.translation, tolerance))
        << "actual translation=" << actual.translation.transpose()
        << " expected translation=" << expected.translation.transpose();
}

void expect_sdf_material_motion_near(
    const SDFMaterialMotion& actual,
    const SDFMaterialMotion& expected,
    const double tolerance = 1.0e-14) {
    expect_sdf_material_pose_near(
        actual.previous, expected.previous, tolerance);
    expect_sdf_material_pose_near(
        actual.current, expected.current, tolerance);
}

std::vector<Vec3> load_multiphysics_gear_reference_positions() {
    std::vector<Vec3> positions;
    std::vector<int> triangles;
    load_obj_mesh((std::filesystem::path(__FILE__).parent_path()
                   / "example_obj" / "gear_z18_coarse.obj").string(),
                  positions, triangles);
    return positions;
}

void expect_uniform_source_gear_geometry(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& body_positions,
    const Vec4& orientation, const double desired_world_extent,
    const double former_axial_thickness) {
    ASSERT_FALSE(source_positions.empty());
    ASSERT_EQ(body_positions.size(), source_positions.size());
    Vec3 source_lower = source_positions.front(), source_upper = source_lower;
    const auto rotated_planar = [&](const Vec3& position) {
        return quaternion_rotate(orientation, Vec3(position.x(), position.y(), 0.0));
    };
    Vec3 planar_lower = rotated_planar(source_positions.front());
    Vec3 planar_upper = planar_lower;
    Vec3 material_lower = body_positions.front(), material_upper = material_lower;
    for (std::size_t node = 0; node < source_positions.size(); ++node) {
        source_lower = source_lower.cwiseMin(source_positions[node]);
        source_upper = source_upper.cwiseMax(source_positions[node]);
        const Vec3 rotated = rotated_planar(source_positions[node]);
        planar_lower = planar_lower.cwiseMin(rotated);
        planar_upper = planar_upper.cwiseMax(rotated);
        material_lower = material_lower.cwiseMin(body_positions[node]);
        material_upper = material_upper.cwiseMax(body_positions[node]);
    }
    const Vec3 source_extent = source_upper - source_lower;
    const Vec3 planar_span = planar_upper - planar_lower;
    const Vec3 axial_direction = quaternion_rotate(orientation, Vec3::UnitZ());
    double scale = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        if (planar_span[axis] > 64.0 * std::numeric_limits<double>::epsilon()
                                   * source_extent.maxCoeff())
            scale = std::min(scale,
                (desired_world_extent - former_axial_thickness * std::abs(axial_direction[axis]))
                    / planar_span[axis]);
    }
    ASSERT_TRUE(std::isfinite(scale));
    ASSERT_GT(scale, 0.0);
    const Vec3 material_extent = material_upper - material_lower;
    EXPECT_TRUE(material_extent.isApprox(scale * source_extent, 1.0e-11));
    EXPECT_NEAR(material_extent.z() / material_extent.head<2>().maxCoeff(),
                source_extent.z() / source_extent.head<2>().maxCoeff(), 1.0e-12);
    // Retain the XY scale of the previously thickened gear; only remove its
    // axial stretch. Compare offsets rather than centers: the volume COM need
    // not coincide with the source AABB center, but every direction has the
    // same scale. This rejects axial stretching even if overall bounds match.
    for (std::size_t node = 0; node < source_positions.size(); ++node) {
        const Vec3 expected = scale * (source_positions[node] - source_positions.front());
        const Vec3 actual = body_positions[node] - body_positions.front();
        EXPECT_LE((actual - expected).norm(), 1.0e-12 * std::max(1.0, desired_world_extent));
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

TEST(BuildSquareMeshAlternatingDiagonals,
     CheckerboardWindingReflectionAndLumpedMass) {
    constexpr int nx = 24;
    constexpr int ny = 160;
    constexpr double width = 0.90;
    constexpr double height = 3.20;
    constexpr double density = 37.0;
    constexpr double thickness = 0.004;
    const Vec3 origin(-0.45, 1.25, -1.60);

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    const int base = build_square_mesh_alternating_diagonals(
        ref_mesh, state, X, nx, ny, width, height, origin);

    constexpr int node_count = (nx + 1) * (ny + 1);
    constexpr int triangle_count = 2 * nx * ny;
    ASSERT_EQ(base, 0);
    ASSERT_EQ(state.deformed_positions.size(), node_count);
    ASSERT_EQ(X.size(), node_count);
    ASSERT_EQ(ref_mesh.tris.size(), 3 * triangle_count);
    ASSERT_EQ(ref_mesh.area.size(), triangle_count);

    const auto node = [](const int i, const int j) {
        return j * (nx + 1) + i;
    };
    const auto triangle_vertices = [&ref_mesh](const int triangle) {
        return std::array<int, 3>{
            ref_mesh.tris[3 * triangle + 0],
            ref_mesh.tris[3 * triangle + 1],
            ref_mesh.tris[3 * triangle + 2]};
    };

    // Even cells retain build_square_mesh's v00--v11 diagonal; odd cells
    // use v10--v01. Both choices have positive material-space winding and
    // the same -y world-space normal.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int first_triangle = 2 * (j * nx + i);
            const int v00 = node(i, j);
            const int v10 = node(i + 1, j);
            const int v01 = node(i, j + 1);
            const int v11 = node(i + 1, j + 1);
            if ((i + j) % 2 == 0) {
                EXPECT_EQ(
                    triangle_vertices(first_triangle),
                    (std::array<int, 3>{v00, v10, v11}));
                EXPECT_EQ(
                    triangle_vertices(first_triangle + 1),
                    (std::array<int, 3>{v00, v11, v01}));
            } else {
                EXPECT_EQ(
                    triangle_vertices(first_triangle),
                    (std::array<int, 3>{v00, v10, v01}));
                EXPECT_EQ(
                    triangle_vertices(first_triangle + 1),
                    (std::array<int, 3>{v10, v11, v01}));
            }
        }
    }
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        const std::array<int, 3> vertices = triangle_vertices(triangle);
        const Vec2 material_first =
            X[static_cast<std::size_t>(vertices[1])]
            - X[static_cast<std::size_t>(vertices[0])];
        const Vec2 material_second =
            X[static_cast<std::size_t>(vertices[2])]
            - X[static_cast<std::size_t>(vertices[0])];
        EXPECT_GT(
            material_first.x() * material_second.y()
                - material_first.y() * material_second.x(),
            0.0);
        const Vec3 world_normal =
            (state.deformed_positions[static_cast<std::size_t>(vertices[1])]
             - state.deformed_positions[static_cast<std::size_t>(vertices[0])])
                .cross(
                    state.deformed_positions[static_cast<std::size_t>(vertices[2])]
                    - state.deformed_positions[static_cast<std::size_t>(vertices[0])]);
        EXPECT_LT(world_normal.y(), 0.0);
        EXPECT_NEAR(world_normal.x(), 0.0, 1.0e-15);
        EXPECT_NEAR(world_normal.z(), 0.0, 1.0e-15);
    }

    // With an even x cell count, reflecting i -> nx-i swaps both the cell
    // parity and its diagonal. The unoriented triangle and edge sets remain
    // identical to the original topology.
    const auto reflect_node = [node](const int vertex) {
        const int j = vertex / (nx + 1);
        const int i = vertex % (nx + 1);
        return node(nx - i, j);
    };
    std::set<std::array<int, 3>> triangles;
    std::set<std::array<int, 3>> reflected_triangles;
    std::set<std::array<int, 2>> edges;
    std::set<std::array<int, 2>> reflected_edges;
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        std::array<int, 3> vertices = triangle_vertices(triangle);
        std::array<int, 3> reflected = {
            reflect_node(vertices[0]),
            reflect_node(vertices[1]),
            reflect_node(vertices[2])};
        std::sort(vertices.begin(), vertices.end());
        std::sort(reflected.begin(), reflected.end());
        triangles.insert(vertices);
        reflected_triangles.insert(reflected);
        for (int local = 0; local < 3; ++local) {
            std::array<int, 2> edge = {
                vertices[local], vertices[(local + 1) % 3]};
            std::array<int, 2> reflected_edge = {
                reflect_node(edge[0]), reflect_node(edge[1])};
            std::sort(edge.begin(), edge.end());
            std::sort(reflected_edge.begin(), reflected_edge.end());
            edges.insert(edge);
            reflected_edges.insert(reflected_edge);
        }
    }
    EXPECT_EQ(triangles, reflected_triangles);
    EXPECT_EQ(edges, reflected_edges);

    constexpr int hinge_count =
        nx * ny + nx * (ny - 1) + (nx - 1) * ny;
    ASSERT_EQ(ref_mesh.hinges.size(), hinge_count);
    const auto hinge_signature = [](
                                     int edge_first, int edge_second,
                                     int apex_first, int apex_second) {
        if (edge_second < edge_first)
            std::swap(edge_first, edge_second);
        if (apex_second < apex_first)
            std::swap(apex_first, apex_second);
        return std::array<int, 4>{
            edge_first, edge_second, apex_first, apex_second};
    };
    std::map<std::array<int, 4>, double> hinge_weights;
    for (const Hinge& hinge : ref_mesh.hinges) {
        EXPECT_NEAR(hinge.bar_theta, 0.0, 1.0e-14);
        EXPECT_GT(hinge.c_e, 0.0);
        const auto [iterator, inserted] = hinge_weights.emplace(
            hinge_signature(
                hinge.v[0], hinge.v[1], hinge.v[2], hinge.v[3]),
            hinge.c_e);
        EXPECT_TRUE(inserted);
        (void)iterator;
    }
    for (const Hinge& hinge : ref_mesh.hinges) {
        const std::array<int, 4> reflected_signature = hinge_signature(
            reflect_node(hinge.v[0]), reflect_node(hinge.v[1]),
            reflect_node(hinge.v[2]), reflect_node(hinge.v[3]));
        const auto reflected_hinge = hinge_weights.find(reflected_signature);
        ASSERT_NE(reflected_hinge, hinge_weights.end());
        EXPECT_NEAR(reflected_hinge->second, hinge.c_e, 1.0e-12);
    }

    ref_mesh.build_lumped_mass(density, thickness);
    ASSERT_EQ(ref_mesh.mass.size(), node_count);
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            EXPECT_NEAR(
                ref_mesh.mass[static_cast<std::size_t>(node(i, j))],
                ref_mesh.mass[static_cast<std::size_t>(node(nx - i, j))],
                1.0e-14);
        }
    }
    const double triangle_mass = density * thickness
        * 0.5 * width / nx * height / ny;
    const double expected_corner_mass = 2.0 * triangle_mass / 3.0;
    EXPECT_NEAR(ref_mesh.mass[node(0, 0)], expected_corner_mass, 1.0e-15);
    EXPECT_NEAR(ref_mesh.mass[node(nx, 0)], expected_corner_mass, 1.0e-15);
    EXPECT_NEAR(ref_mesh.mass[node(0, ny)], expected_corner_mass, 1.0e-15);
    EXPECT_NEAR(ref_mesh.mass[node(nx, ny)], expected_corner_mass, 1.0e-15);
    EXPECT_NEAR(
        ref_mesh.mass[node(0, ny)], ref_mesh.mass[node(nx, ny)], 1.0e-15);
}

TEST(MixedExample,
     FourMixedObjectLayersAboveOppositeEdgePinnedCloth) {
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
    constexpr double solid_max_extent = 0.44;
    constexpr double rigid_max_extent = 0.22;
    enum BodyType {
        Bunny = 0, Spot = 1, Cube = 2, Gear = 3, BodyTypeCount = 4};
    static constexpr int expected_layer_order[rows][BodyTypeCount] = {
        {Cube, Spot, Gear, Bunny},
        {Gear, Bunny, Cube, Spot},
        {Bunny, Spot, Gear, Cube},
        {Gear, Cube, Bunny, Spot},
    };
    // Literal expected centers keep this regression independent from the
    // production packing implementation.
    static constexpr double expected_x[rows][BodyTypeCount] = {
        { 0.28, 0.28, -0.28, -0.28},
        { 0.28, 0.28, -0.28, -0.28},
        {-0.28, 0.28,  0.28, -0.28},
        {-0.28, 0.28,  0.28, -0.28},
    };
    static constexpr double expected_y[rows] = {1.52, 1.985, 2.45, 2.915};
    static constexpr double expected_z[rows][BodyTypeCount] = {
        { 0.27, -0.27, -0.27,  0.27},
        {-0.27,  0.27,  0.27, -0.27},
        {-0.27, -0.27,  0.27,  0.27},
        { 0.27,  0.27, -0.27, -0.27},
    };
    const Vec3 bunny_extents(
        solid_max_extent, 0.335752074786, 0.435584485870);
    const Vec3 spot_extents(
        0.433659138233, 0.241631963161, solid_max_extent);
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

    const auto object_center = [&](const int row, const int type) {
        return Vec3(
            expected_x[row][type], expected_y[row], expected_z[row][type]);
    };
    const auto column_for_type = [](const int row, const int type) {
        for (int column = 0; column < BodyTypeCount; ++column) {
            if (expected_layer_order[row][column] == type)
                return column;
        }
        return -1;
    };
    const auto check_solid =
        [&](const int node_base, const int node_count,
            const int first_tet, const int tet_count,
            const Vec3& expected_center, const Vec3& expected_extents,
            const std::array<Vec3, 3>& expected_material_edges,
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

            // Verify each animal's rotation using its material +x, +y,
            // and +z fixture edges, independently of its AABB dimensions.
            for (int axis = 0; axis < 3; ++axis) {
                const Vec3 actual_edge =
                    state.deformed_positions[node_base + axis + 1]
                    - state.deformed_positions[node_base];
                EXPECT_LT((actual_edge - expected_material_edges[
                              static_cast<std::size_t>(axis)]).norm(),
                    1.0e-14);
            }

            for (int element = first_tet;
                 element < first_tet + tet_count; ++element) {
                EXPECT_GT(ref_mesh.tet_rest_data[element].measure, 0.0);
                for (int local = 0; local < 4; ++local) {
                    const int node = ref_mesh.tets[4 * element + local];
                    EXPECT_GE(node, node_base);
                    EXPECT_LT(node, node_end);
                }
                const Vec3& x0 =
                    state.deformed_positions[ref_mesh.tets[4 * element]];
                Mat33 Ds;
                for (int axis = 0; axis < 3; ++axis) {
                    Ds.col(axis) = state.deformed_positions[
                        ref_mesh.tets[4 * element + axis + 1]] - x0;
                }
                // Rotation belongs to the rest pose, so no tetrahedron may
                // begin with elastic strain.
                EXPECT_TRUE((Ds * ref_mesh.tet_rest_data[element].Dm_inverse)
                                .isApprox(Mat33::Identity(), 1.0e-13));
            }
            return ObjectBounds{lower, upper};
        };

    // The two fixture tetrahedra together occupy the axis tetrahedron defined
    // by the production Bunny AABB dimensions.
    const double expected_bunny_mass =
        args.solid_density * bunny_extents.prod() / 6.0;
    // A +90-degree turn about X keeps Bunny's front/back axis horizontal:
    // material +x -> world +x, +y -> +z, and +z -> -y.
    const std::array<Vec3, 3> expected_bunny_edges = {
        Vec3(solid_max_extent, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.435584485870),
        Vec3(0.0, -0.01 * solid_max_extent, 0.0),
    };
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("bunny layer " + std::to_string(row));
        const int column = column_for_type(row, Bunny);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            check_solid(
            cloth_vertices + row * bunny_vertices,
            bunny_vertices, row * bunny_tetrahedra, bunny_tetrahedra,
            object_center(row, Bunny), bunny_extents,
            expected_bunny_edges, expected_bunny_mass);
    }

    const double expected_spot_mass =
        args.solid_density * spot_extents.prod() / 6.0;
    // Spot's +90-degree turn about Z maps +x -> +y and +y -> -x.
    const std::array<Vec3, 3> expected_spot_edges = {
        Vec3(0.0, 0.241631963161, 0.0),
        Vec3(-0.433659138233, 0.0, 0.0),
        Vec3(0.0, 0.0, solid_max_extent),
    };
    constexpr int spot_node_base =
        cloth_vertices + rows * bunny_vertices;
    constexpr int spot_tet_base = rows * bunny_tetrahedra;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("spot layer " + std::to_string(row));
        const int column = column_for_type(row, Spot);
        ASSERT_GE(column, 0);
        object_bounds[static_cast<std::size_t>(4 * row + column)] =
            check_solid(
            spot_node_base + row * spot_vertices,
            spot_vertices, spot_tet_base + row * spot_tetrahedra,
            spot_tetrahedra, object_center(row, Spot), spot_extents,
            expected_spot_edges, expected_spot_mass);
    }

    const double expected_cube_mass = args.rigid_density
        * rigid_max_extent * rigid_max_extent * rigid_max_extent;
    constexpr int first_rigid_node = cloth_vertices + solid_vertices;
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("cube layer " + std::to_string(row));
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
        SCOPED_TRACE("gear layer " + std::to_string(row));
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

    // Each horizontal layer has four distinct occupied columns at a common
    // height, one of every object type, and its own ordering.
    for (int row = 0; row < rows; ++row) {
        SCOPED_TRACE("horizontal layer " + std::to_string(row));
        std::array<int, BodyTypeCount> type_counts{};
        for (int column = 0; column < BodyTypeCount; ++column) {
            const int type = expected_layer_order[row][column];
            ASSERT_GE(type, 0);
            ASSERT_LT(type, BodyTypeCount);
            ++type_counts[static_cast<std::size_t>(type)];
            const ObjectBounds& bounds = object_bounds[
                static_cast<std::size_t>(4 * row + column)];
            const Vec3 actual_center = 0.5 * (bounds.lower + bounds.upper);
            EXPECT_NEAR(actual_center.x(), expected_x[row][type], 1.0e-14);
            EXPECT_NEAR(actual_center.y(), expected_y[row], 1.0e-14);
            EXPECT_NEAR(actual_center.z(), expected_z[row][type], 1.0e-14);
            EXPECT_NEAR(
                actual_center.x(), column % 2 == 0 ? -0.28 : 0.28, 1.0e-14);
            EXPECT_NEAR(
                actual_center.z(), column < 2 ? -0.27 : 0.27, 1.0e-14);
            for (int previous = 0; previous < column; ++previous) {
                const ObjectBounds& previous_bounds = object_bounds[
                    static_cast<std::size_t>(4 * row + previous)];
                const Vec3 previous_center =
                    0.5 * (previous_bounds.lower + previous_bounds.upper);
                EXPECT_NEAR(
                    actual_center.y(), previous_center.y(), 1.0e-14);
                EXPECT_GT(
                    std::hypot(actual_center.x() - previous_center.x(),
                               actual_center.z() - previous_center.z()),
                    0.5);
            }
        }
        for (const int count : type_counts)
            EXPECT_EQ(count, 1);
        for (int previous = 0; previous < row; ++previous) {
            EXPECT_FALSE(std::equal(
                expected_layer_order[row],
                expected_layer_order[row] + BodyTypeCount,
                expected_layer_order[previous]));
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
    constexpr double expected_tight_gap = 0.102207757065;
    EXPECT_NEAR(minimum_object_gap, expected_tight_gap, 5.0e-13);

    // All twelve adjacent-layer pairs are aligned for successive landings.
    // Their projected footprints overlap, with positive vertical clearance.
    int aligned_pair_count = 0;
    bool gear_above_cube = false;
    bool bunny_above_spot = false;
    bool gear_above_bunny = false;
    bool bunny_above_gear = false;
    for (int row = 1; row < rows; ++row) {
        SCOPED_TRACE("vertical pairs below layer " + std::to_string(row));
        for (int column = 0; column < BodyTypeCount; ++column) {
            SCOPED_TRACE("column " + std::to_string(column));
            const ObjectBounds& upper = object_bounds[static_cast<std::size_t>(
                4 * row + column)];
            const ObjectBounds& lower = object_bounds[static_cast<std::size_t>(
                4 * (row - 1) + column)];
            const Vec3 upper_center = 0.5 * (upper.lower + upper.upper);
            const Vec3 lower_center = 0.5 * (lower.lower + lower.upper);
            EXPECT_NEAR(upper_center.x(), lower_center.x(), 1.0e-14);
            EXPECT_NEAR(upper_center.z(), lower_center.z(), 1.0e-14);
            EXPECT_NEAR(
                upper_center.y() - lower_center.y(), 0.465, 1.0e-14);
            EXPECT_GE(upper.lower.y() - lower.upper.y(), 0.025 - 1.0e-14);
            for (const int axis : {0, 2}) {
                EXPECT_GT(
                    std::min(upper.upper[axis], lower.upper[axis])
                        - std::max(upper.lower[axis], lower.lower[axis]),
                    0.0);
            }
            ++aligned_pair_count;
            const int upper_type = expected_layer_order[row][column];
            const int lower_type = expected_layer_order[row - 1][column];
            gear_above_cube |= upper_type == Gear && lower_type == Cube;
            bunny_above_spot |= upper_type == Bunny && lower_type == Spot;
            gear_above_bunny |= upper_type == Gear && lower_type == Bunny;
            bunny_above_gear |= upper_type == Bunny && lower_type == Gear;
            EXPECT_FALSE(upper_type == Spot && lower_type == Gear);
        }
    }
    EXPECT_EQ(aligned_pair_count, 12);
    EXPECT_TRUE(gear_above_cube);
    EXPECT_TRUE(bunny_above_spot);
    EXPECT_TRUE(gear_above_bunny);
    EXPECT_TRUE(bunny_above_gear);
    double minimum_cloth_gap = std::numeric_limits<double>::infinity();
    for (int object = 0; object < rows * 4; ++object) {
        const ObjectBounds& bounds =
            object_bounds[static_cast<std::size_t>(object)];
        const double gap = bounds.lower.y() - cloth_height;
        EXPECT_GT(gap, params.d_hat);
        for (const int axis : {0, 2}) {
            EXPECT_GT(bounds.lower[axis], -2.0);
            EXPECT_LT(bounds.upper[axis], 2.0);
        }
        minimum_cloth_gap = std::min(minimum_cloth_gap, gap);
    }
    constexpr double expected_cloth_gap = 0.152123962607;
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

TEST(MaterialArguments, UsesSolidMaterialCommandLineOverrides) {
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

    // Give the cloth fields deliberately unrelated values. The solid must
    // use only the solid-specific material arguments below.
    args.E = 123.0;
    args.nu = 0.1;
    args.density = 17.0;

    RefMesh ref_mesh;
    DeformedState state;
    SimParams params = args.to_sim_params();

    constexpr int side_count = 8;
    constexpr double radius = 0.22;
    constexpr double thickness = 0.16;
    append_deformable_polygon_prism(
        side_count, state, ref_mesh, Vec3::Zero(), radius,
        params.solid_density, thickness);
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
    EXPECT_DOUBLE_EQ(defaults.friction_coefficient, 0.0);
    EXPECT_DOUBLE_EQ(defaults.friction_velocity_epsilon, 0.01);
    EXPECT_DOUBLE_EQ(defaults.to_sim_params().friction_coefficient, 0.0);
    EXPECT_DOUBLE_EQ(
        defaults.to_sim_params().friction_velocity_epsilon, 0.01);
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
    char friction_key[] = "--friction_coefficient";
    char friction_value[] = "0.37";
    char friction_epsilon_key[] = "--friction_velocity_epsilon";
    char friction_epsilon_value[] = "0.025";
    char verbose_key[] = "--verbose";
    char* argv[] = {
        program,
        rigid_density_key, rigid_density_value,
        solid_density_key, solid_density_value,
        shell_density_key, shell_density_value,
        friction_key, friction_value,
        friction_epsilon_key, friction_epsilon_value,
        verbose_key,
    };
    ASSERT_TRUE(args.parse(12, argv));

    EXPECT_DOUBLE_EQ(args.rigid_density, 1234.0);
    EXPECT_DOUBLE_EQ(args.solid_density, 567.0);
    EXPECT_DOUBLE_EQ(args.density, 18.0);
    EXPECT_DOUBLE_EQ(args.friction_coefficient, 0.37);
    EXPECT_DOUBLE_EQ(args.friction_velocity_epsilon, 0.025);
    EXPECT_TRUE(args.verbose);

    const SimParams params = args.to_sim_params();
    EXPECT_DOUBLE_EQ(params.rigid_density, 1234.0);
    EXPECT_DOUBLE_EQ(params.solid_density, 567.0);
    EXPECT_DOUBLE_EQ(params.density, 18.0);
    EXPECT_DOUBLE_EQ(params.friction_coefficient, 0.37);
    EXPECT_DOUBLE_EQ(params.friction_velocity_epsilon, 0.025);
    EXPECT_TRUE(params.verbose);

    IPCArgs3D invalid_coefficient;
    invalid_coefficient.friction_coefficient = -1.0;
    EXPECT_THROW(
        (void)invalid_coefficient.to_sim_params(), std::invalid_argument);
    invalid_coefficient.friction_coefficient =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void)invalid_coefficient.to_sim_params(), std::invalid_argument);

    IPCArgs3D invalid_epsilon;
    invalid_epsilon.friction_coefficient = 0.1;
    invalid_epsilon.friction_velocity_epsilon = 0.0;
    EXPECT_THROW(
        (void)invalid_epsilon.to_sim_params(), std::invalid_argument);
    invalid_epsilon.friction_velocity_epsilon =
        std::numeric_limits<double>::infinity();
    EXPECT_THROW(
        (void)invalid_epsilon.to_sim_params(), std::invalid_argument);

    IPCArgs3D disabled_friction;
    disabled_friction.friction_coefficient = 0.0;
    disabled_friction.friction_velocity_epsilon =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW((void)disabled_friction.to_sim_params());
}

TEST(MaterialArguments, RigidAndSolidDensitiesAreIndependent) {
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
        SimParams params = args.to_sim_params();
        for (int polygon = 0; polygon < 10; ++polygon) {
            append_rigid_polygon(
                3 + polygon, state, ref_mesh,
                Vec3(polygon, 0.0, 0.0), 0.1,
                params.rigid_density, 0.05);
            append_deformable_polygon_prism(
                3 + polygon, state, ref_mesh,
                Vec3(polygon, 1.0, 0.0), 0.2,
                params.solid_density, 0.1);
        }

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

TEST(AppendNormalizedTetGenSolid,
     RotatesRestPosePreservesMassAndRejectsInvalidOrientationsTransactionally) {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> next_directory{0};
    const fs::path directory = fs::temp_directory_path()
        / ("ipc_rotated_normalized_tetgen_solid_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + "_" + std::to_string(next_directory.fetch_add(1)));
    fs::create_directories(directory);
    struct TemporaryDirectoryGuard {
        fs::path path;
        ~TemporaryDirectoryGuard() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } directory_guard{directory};
    const fs::path node_file = directory / "shape.node";
    const fs::path element_file = directory / "shape.ele";
    {
        std::ofstream output(node_file);
        ASSERT_TRUE(output.good());
        output << "4 3 0 0\n"
               << "0 -2 -1 0\n"
               << "1  2 -1 0\n"
               << "2 -2  1 0\n"
               << "3 -2 -1 1\n";
    }
    {
        std::ofstream output(element_file);
        ASSERT_TRUE(output.good());
        output << "1 4 0\n"
               << "0 0 1 2 3\n";
    }

    RefMesh ref_mesh;
    DeformedState state;
    ASSERT_EQ(append_normalized_tetgen_solid(
        node_file.string(), element_file.string(), state, ref_mesh,
        Vec3(-3.0, -4.0, -5.0), 2.0, 6.0, true), 0);
    const Vec3 target_center(5.0, 6.0, 7.0);
    // Scaling the quaternion also verifies that the importer normalizes it.
    const double half_angle = 0.25 * std::acos(-1.0);
    const Vec4 orientation(
        3.0 * std::cos(half_angle), 0.0, 0.0,
        3.0 * std::sin(half_angle));
    const int base = append_normalized_tetgen_solid(
        node_file.string(), element_file.string(), state, ref_mesh,
        target_center, 2.0, 6.0, true, orientation);
    ASSERT_EQ(base, 4);
    ASSERT_EQ(state.deformed_positions.size(), 8U);
    const std::array<Vec3, 4> expected_offsets = {
        Vec3(0.5, -1.0, -0.25), Vec3(0.5, 1.0, -0.25),
        Vec3(-0.5, -1.0, -0.25), Vec3(0.5, -1.0, 0.25),
    };
    for (int local = 0; local < 4; ++local) {
        EXPECT_TRUE(state.deformed_positions[base + local].isApprox(
            target_center + expected_offsets[local], 1.0e-14));
        EXPECT_NEAR(ref_mesh.mass[base + local], ref_mesh.mass[local], 1.0e-14);
    }
    ASSERT_EQ(ref_mesh.tet_rest_data.size(), 2U);
    const TetRestData& rest = ref_mesh.tet_rest_data[1];
    EXPECT_GT(rest.measure, 0.0);
    EXPECT_NEAR(rest.measure, 1.0 / 6.0, 1.0e-15);
    EXPECT_NEAR(rest.measure, ref_mesh.tet_rest_data[0].measure, 1.0e-15);
    EXPECT_NEAR(
        std::accumulate(ref_mesh.mass.begin() + base, ref_mesh.mass.end(), 0.0),
        1.0, 1.0e-14);
    const Vec3& x0 = state.deformed_positions[ref_mesh.tets[4]];
    Mat33 Ds;
    for (int axis = 0; axis < 3; ++axis) {
        Ds.col(axis) =
            state.deformed_positions[ref_mesh.tets[5 + axis]] - x0;
    }
    EXPECT_GT(Ds.determinant(), 0.0);
    EXPECT_TRUE((Ds * rest.Dm_inverse).isApprox(Mat33::Identity(), 1.0e-14));

    // Rejected orientations must preserve the already populated solid state.
    const RefMesh previous_mesh = ref_mesh;
    const DeformedState previous_state = state;
    const std::array<Vec4, 3> invalid_orientations = {
        Vec4::Zero(),
        Vec4(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0),
        Vec4(1.0, 0.0, 0.0, std::numeric_limits<double>::infinity()),
    };
    for (const Vec4& invalid_orientation : invalid_orientations) {
        EXPECT_THROW(
            append_normalized_tetgen_solid(
                node_file.string(), element_file.string(), state, ref_mesh,
                target_center, 2.0, 6.0, true, invalid_orientation),
            std::invalid_argument);
        ASSERT_EQ(state.deformed_positions.size(),
                  previous_state.deformed_positions.size());
        ASSERT_EQ(state.velocities.size(), previous_state.velocities.size());
        for (std::size_t node = 0; node < state.deformed_positions.size(); ++node) {
            EXPECT_TRUE(state.deformed_positions[node].isApprox(
                previous_state.deformed_positions[node], 0.0));
            EXPECT_TRUE(state.velocities[node].isApprox(
                previous_state.velocities[node], 0.0));
        }
        EXPECT_EQ(ref_mesh.num_positions, previous_mesh.num_positions);
        EXPECT_EQ(ref_mesh.tets, previous_mesh.tets);
        EXPECT_EQ(ref_mesh.tris, previous_mesh.tris);
        EXPECT_EQ(ref_mesh.mass, previous_mesh.mass);
        EXPECT_EQ(ref_mesh.tet_adj, previous_mesh.tet_adj);
        EXPECT_EQ(ref_mesh.tet_nodes, previous_mesh.tet_nodes);
        EXPECT_EQ(ref_mesh.surface_nodes, previous_mesh.surface_nodes);
        EXPECT_EQ(ref_mesh.node_to_rb, previous_mesh.node_to_rb);
        EXPECT_EQ(ref_mesh.deformable_nodes, previous_mesh.deformable_nodes);
        ASSERT_EQ(ref_mesh.tet_rest_data.size(),
                  previous_mesh.tet_rest_data.size());
        for (std::size_t element = 0;
             element < ref_mesh.tet_rest_data.size(); ++element) {
            EXPECT_DOUBLE_EQ(ref_mesh.tet_rest_data[element].measure,
                             previous_mesh.tet_rest_data[element].measure);
            EXPECT_TRUE(ref_mesh.tet_rest_data[element].Dm_inverse.isApprox(
                previous_mesh.tet_rest_data[element].Dm_inverse, 0.0));
        }
    }
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

TEST(ClothExample, RolledClothUsesInclineAndGroundSDFs) {
    IPCArgs3D args;
    args.density = 37.0;
    args.thickness = 0.004;
    args.rigid_density = 123.0;
    // Deliberately exceed the cloth-grid edge bound so the scene's analytic
    // activation-distance clamp and its dependent initial clearance execute.
    args.d_hat = 0.1;
    args.k_barrier = 719.0;
    args.k_sdf = 840000.0;
    args.eps_sdf = 0.003;
    args.gx = 0.25;
    args.gy = -7.5;
    args.gz = -0.75;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;

    build_cloth_unrolling_down_fixed_ramp_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    constexpr int cloth_nx = 24;
    constexpr int cloth_ny = 160;
    constexpr int cloth_nodes =
        (cloth_nx + 1) * (cloth_ny + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_ny;
    constexpr int cloth_hinges =
        cloth_nx * cloth_ny
        + cloth_nx * (cloth_ny - 1)
        + (cloth_nx - 1) * cloth_ny;
    constexpr int wedge_nodes = 6;
    constexpr int wedge_triangles = 8;
    constexpr int ground_nodes = 4;
    constexpr int ground_triangles = 2;
    constexpr int total_nodes = cloth_nodes;
    constexpr int total_triangles = cloth_triangles;

    constexpr double cloth_width = 0.90;
    constexpr double ramp_width = 1.40;
    constexpr double ramp_height = 2.40;
    constexpr double ramp_back_z = -1.20;
    constexpr double ramp_front_z = 1.20;
    constexpr int leader_rows = 8;
    constexpr double outer_radius = 0.100;
    constexpr double inner_radius = 0.05;
    constexpr double layer_pitch = 0.0065;
    constexpr double scene_d_hat_cap = 0.005;
    constexpr double minimum_clearance = 0.002;
    constexpr double minimum_pin_clearance = 0.020;
    constexpr double roll_clearance_margin = 1.0e-4;
    constexpr double pin_clearance_margin = 1.0e-3;
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
    const double leader_material_length =
        static_cast<double>(leader_rows) * spiral_length
        / static_cast<double>(cloth_ny - leader_rows);
    const double cloth_length = leader_material_length + spiral_length;
    const double ramp_length = std::hypot(
        ramp_height, ramp_front_z - ramp_back_z);
    const double cloth_overhang = cloth_length - ramp_length;
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
    EXPECT_TRUE(ref_mesh.mass.empty());
    EXPECT_TRUE(ref_mesh.node_to_rb.empty());
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
    const double active_contact_range = std::max(
        std::max(expected_d_hat, 0.0),
        std::max(args.eps_sdf, 0.0));
    const double roll_clearance = std::max(
        active_contact_range + roll_clearance_margin,
        minimum_clearance);
    const double pin_clearance = std::max(
        active_contact_range + pin_clearance_margin,
        minimum_pin_clearance);
    const double leader_clearance_drop =
        pin_clearance - roll_clearance;
    const double leader_downslope_span = std::sqrt(
        leader_material_length * leader_material_length
        - leader_clearance_drop * leader_clearance_drop);
    const double roll_downslope_offset = leader_downslope_span;
    EXPECT_GT(roll_clearance, params.d_hat);
    EXPECT_GT(pin_clearance, roll_clearance);

    EXPECT_TRUE(params.gravity.isApprox(
        Vec3(args.gx, args.gy, args.gz), 0.0));
    EXPECT_DOUBLE_EQ(params.k_barrier, args.k_barrier);
    EXPECT_DOUBLE_EQ(params.k_sdf, args.k_sdf);
    EXPECT_DOUBLE_EQ(params.eps_sdf, args.eps_sdf);
    ASSERT_EQ(params.sdf_planes.size(), 2U);
    EXPECT_TRUE(params.sdf_planes[0].point.isZero(0.0));
    EXPECT_TRUE(params.sdf_planes[0].normal.isApprox(
        Vec3::UnitY(), 0.0));
    EXPECT_TRUE(params.sdf_planes[1].point.isApprox(
        ramp_top, 0.0));
    EXPECT_TRUE(params.sdf_planes[1].normal.isApprox(
        ramp_normal, 0.0));
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());

    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    const PlaneSDF& ground_sdf = params.sdf_planes[0];
    const PlaneSDF& ramp_sdf = params.sdf_planes[1];
    EXPECT_NEAR(ground_sdf.normal.norm(), 1.0, 0.0);
    EXPECT_NEAR(ramp_sdf.normal.norm(), 1.0, 1.0e-15);
    const auto minimum_sdf_evaluation = [&](const Vec3& position) {
        SDFEvaluation result = evaluate_sdf(ground_sdf, position);
        const SDFEvaluation ramp_result =
            evaluate_sdf(ramp_sdf, position);
        // Match the production minimum-phi reduction: strict comparison
        // preserves the ground-first ordering when both planes tie at the toe.
        if (ramp_result.phi < result.phi)
            result = ramp_result;
        return result;
    };

    const Vec3 ramp_midpoint =
        ramp_top + 0.5 * ramp_length * downhill;
    const Vec3 ramp_toe(0.0, 0.0, ramp_front_z);
    const Vec3 ground_surface_probe(0.0, 0.0, 2.0);
    EXPECT_NEAR(evaluate_sdf(ramp_sdf, ramp_midpoint).phi, 0.0, 1.0e-15);
    EXPECT_NEAR(
        evaluate_sdf(ground_sdf, ramp_midpoint).phi,
        0.5 * ramp_height, 1.0e-15);
    EXPECT_NEAR(evaluate_sdf(ground_sdf, ramp_toe).phi, 0.0, 0.0);
    EXPECT_NEAR(evaluate_sdf(ramp_sdf, ramp_toe).phi, 0.0, 1.0e-15);
    const SDFEvaluation toe_sdf = minimum_sdf_evaluation(ramp_toe);
    EXPECT_NEAR(toe_sdf.phi, 0.0, 0.0);
    EXPECT_TRUE(toe_sdf.grad_phi.isApprox(Vec3::UnitY(), 0.0));
    EXPECT_NEAR(
        evaluate_sdf(ground_sdf, ground_surface_probe).phi, 0.0, 0.0);
    EXPECT_GT(evaluate_sdf(ramp_sdf, ground_surface_probe).phi, 0.0);
    EXPECT_TRUE(minimum_sdf_evaluation(ground_surface_probe)
                    .grad_phi.isApprox(Vec3::UnitY(), 0.0));

    const Vec3 ramp_force_free_probe =
        ramp_midpoint + 2.0 * params.eps_sdf * ramp_normal;
    const Vec3 ground_force_free_probe(
        0.0, 2.0 * params.eps_sdf, 2.0);
    for (const Vec3& probe :
         {ramp_force_free_probe, ground_force_free_probe}) {
        const SDFEvaluation sdf = minimum_sdf_evaluation(probe);
        EXPECT_GT(sdf.phi, params.eps_sdf);
        EXPECT_DOUBLE_EQ(
            sdf_penalty_energy(sdf, params.k_sdf, params.eps_sdf), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(
            sdf, params.k_sdf, params.eps_sdf).isZero(0.0));
    }
    const SDFEvaluation inside_ramp = minimum_sdf_evaluation(
        ramp_midpoint - 0.001 * ramp_normal);
    const SDFEvaluation inside_ground = minimum_sdf_evaluation(
        ground_surface_probe - 0.001 * Vec3::UnitY());
    EXPECT_NEAR(inside_ramp.phi, -0.001, 1.0e-15);
    EXPECT_NEAR(inside_ground.phi, -0.001, 1.0e-15);
    EXPECT_GT(
        sdf_penalty_energy(
            inside_ramp, params.k_sdf, params.eps_sdf),
        0.0);
    EXPECT_GT(
        sdf_penalty_energy(
            inside_ground, params.k_sdf, params.eps_sdf),
        0.0);

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
            if ((i + j) % 2 == 0) {
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
            } else {
                EXPECT_EQ(
                    (std::array<int, 3>{
                        ref_mesh.tris[3 * triangle + 0],
                        ref_mesh.tris[3 * triangle + 1],
                        ref_mesh.tris[3 * triangle + 2]}),
                    (std::array<int, 3>{v00, v10, v01}));
                EXPECT_EQ(
                    (std::array<int, 3>{
                        ref_mesh.tris[3 * triangle + 3],
                        ref_mesh.tris[3 * triangle + 4],
                        ref_mesh.tris[3 * triangle + 5]}),
                    (std::array<int, 3>{v10, v11, v01}));
            }
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

    EXPECT_NEAR(spiral_length, 3.6252731123901647, 1.0e-15);
    EXPECT_NEAR(
        roll_downslope_offset, 0.19022118288835083, 1.0e-15);
    EXPECT_NEAR(
        leader_material_length, 0.190803848020535, 1.0e-15);
    EXPECT_NEAR(cloth_length, 3.8160769604106997, 1.0e-15);
    EXPECT_NEAR(ramp_length, 3.3941125496954281, 1.0e-15);
    EXPECT_GT(cloth_length, ramp_length);
    EXPECT_NEAR(cloth_overhang, 0.4219644107152716, 1.0e-15);
    EXPECT_NEAR(
        leader_rows * rest_ds, leader_material_length, 1.0e-15);
    EXPECT_NEAR(rest_ds, 0.023850481002566874, 1.0e-15);
    EXPECT_NEAR(roll_clearance, 0.0051, 1.0e-15);
    EXPECT_NEAR(pin_clearance, 0.020, 1.0e-15);
    EXPECT_NEAR(
        roll_clearance - params.d_hat,
        roll_clearance_margin, 1.0e-15);
    EXPECT_DOUBLE_EQ(pin_clearance, minimum_pin_clearance);
    EXPECT_GT(
        pin_clearance - params.d_hat, pin_clearance_margin);
    EXPECT_NEAR(leader_clearance_drop, 0.0149, 1.0e-15);
    EXPECT_NEAR(
        leader_material_length - leader_downslope_span,
        0.0005826651321841625, 1.0e-15);
    EXPECT_GT(leader_downslope_span, 0.99 * leader_material_length);
    // The new pin is still far below the former seven-row C1 leader.
    EXPECT_LT(pin_clearance, 0.1313091896266138);

    const Vec3 roll_center =
        ramp_top + roll_downslope_offset * downhill
        + (outer_radius + roll_clearance) * ramp_normal;
    const auto expected_centerline = [&](const double s) -> Vec3 {
        if (s <= leader_material_length) {
            const double leader_fraction = s / leader_material_length;
            return ramp_top
                + leader_fraction * leader_downslope_span * downhill
                + (pin_clearance
                    - leader_fraction * leader_clearance_drop)
                    * ramp_normal;
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
        return roll_center + radius * (
            std::sin(theta) * downhill
            - std::cos(theta) * ramp_normal);
    };

    double minimum_ramp_distance =
        std::numeric_limits<double>::infinity();
    double minimum_combined_sdf_distance =
        std::numeric_limits<double>::infinity();
    double maximum_initial_sdf_energy = 0.0;
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
            EXPECT_GE(ramp_distance, roll_clearance - 2.0e-12);
            minimum_ramp_distance = std::min(
                minimum_ramp_distance, ramp_distance);
            const Vec3& position = state.deformed_positions[
                static_cast<std::size_t>(node)];
            const SDFEvaluation ground_evaluation =
                evaluate_sdf(ground_sdf, position);
            const SDFEvaluation ramp_evaluation =
                evaluate_sdf(ramp_sdf, position);
            EXPECT_LT(ramp_evaluation.phi, ground_evaluation.phi);
            const SDFEvaluation combined_evaluation =
                minimum_sdf_evaluation(position);
            EXPECT_NEAR(
                combined_evaluation.phi,
                ramp_evaluation.phi, 1.0e-15);
            EXPECT_GE(
                combined_evaluation.phi,
                roll_clearance - 2.0e-12);
            minimum_combined_sdf_distance = std::min(
                minimum_combined_sdf_distance,
                combined_evaluation.phi);
            maximum_initial_sdf_energy = std::max(
                maximum_initial_sdf_energy,
                sdf_penalty_energy(
                    combined_evaluation,
                    params.k_sdf, params.eps_sdf));
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
    EXPECT_NEAR(minimum_ramp_distance, roll_clearance, 2.0e-12);
    EXPECT_GT(minimum_ramp_distance, params.d_hat);
    EXPECT_NEAR(
        minimum_combined_sdf_distance,
        roll_clearance, 2.0e-12);
    EXPECT_GT(minimum_combined_sdf_distance, params.eps_sdf);
    EXPECT_DOUBLE_EQ(maximum_initial_sdf_energy, 0.0);

    const Vec3 pin_centerline =
        ramp_top + pin_clearance * ramp_normal;
    const Vec3 expected_join_centerline =
        ramp_top + leader_downslope_span * downhill
        + roll_clearance * ramp_normal;
    EXPECT_TRUE((roll_center - outer_radius * ramp_normal).isApprox(
        expected_join_centerline, 2.0e-14));

    // Every edge in the sloped eight-row leader starts at its material rest
    // length: across and longitudinal grid edges, plus each cell diagonal.
    double maximum_leader_edge_strain = 0.0;
    const auto record_leader_edge_strain = [&maximum_leader_edge_strain](
                                               const Vec3& first,
                                               const Vec3& second,
                                               const double rest_length) {
        maximum_leader_edge_strain = std::max(
            maximum_leader_edge_strain,
            std::abs((second - first).norm() / rest_length - 1.0));
    };
    Vec3 previous_leader_centerline = Vec3::Zero();
    for (int j = 0; j <= leader_rows; ++j) {
        const Vec3 centerline = 0.5 * (
            state.deformed_positions[static_cast<std::size_t>(
                cloth_node(0, j))]
            + state.deformed_positions[static_cast<std::size_t>(
                cloth_node(cloth_nx, j))]);
        const double normal_offset =
            (centerline - ramp_top).dot(ramp_normal);
        const double leader_distance = j * rest_ds;
        const double leader_fraction =
            leader_distance / leader_material_length;
        const double expected_normal_offset =
            pin_clearance
            - leader_fraction * leader_clearance_drop;
        EXPECT_NEAR(normal_offset, expected_normal_offset, 2.0e-14);
        EXPECT_NEAR(
            (centerline - ramp_top).dot(downhill),
            leader_fraction * leader_downslope_span, 2.0e-14);
        EXPECT_GT(normal_offset, params.d_hat);
        const double remaining_fraction = 1.0 - leader_fraction;
        const double expected_roll_distance_squared =
            remaining_fraction * remaining_fraction
                * leader_downslope_span * leader_downslope_span
            + std::pow(
                outer_radius
                    - remaining_fraction * leader_clearance_drop,
                2.0);
        EXPECT_NEAR(
            (centerline - roll_center).squaredNorm(),
            expected_roll_distance_squared, 2.0e-15);
        if (j < leader_rows)
            EXPECT_GT((centerline - roll_center).norm(), outer_radius);
        else
            EXPECT_NEAR(
                (centerline - roll_center).norm(),
                outer_radius, 2.0e-14);

        for (int i = 0; i < cloth_nx; ++i) {
            const Vec3& first = state.deformed_positions[
                static_cast<std::size_t>(cloth_node(i, j))];
            const Vec3& second = state.deformed_positions[
                static_cast<std::size_t>(cloth_node(i + 1, j))];
            EXPECT_NEAR((second - first).norm(), rest_dx, 2.0e-14);
            record_leader_edge_strain(first, second, rest_dx);
        }
        if (j > 0) {
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).norm(),
                rest_ds, 2.0e-14);
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).dot(downhill),
                rest_ds * leader_downslope_span
                    / leader_material_length,
                2.0e-14);
            EXPECT_NEAR(
                (centerline - previous_leader_centerline).dot(ramp_normal),
                -rest_ds * leader_clearance_drop
                    / leader_material_length,
                2.0e-14);
            for (int i = 0; i <= cloth_nx; ++i) {
                const Vec3& previous = state.deformed_positions[
                    static_cast<std::size_t>(cloth_node(i, j - 1))];
                const Vec3& current = state.deformed_positions[
                    static_cast<std::size_t>(cloth_node(i, j))];
                EXPECT_NEAR(
                    (current - previous).norm(),
                    rest_ds, 2.0e-14);
                record_leader_edge_strain(previous, current, rest_ds);
            }
            const double rest_diagonal = std::hypot(rest_dx, rest_ds);
            for (int i = 0; i < cloth_nx; ++i) {
                const bool main_diagonal = (i + j - 1) % 2 == 0;
                const Vec3& diagonal_first = state.deformed_positions[
                    static_cast<std::size_t>(cloth_node(
                        main_diagonal ? i : i + 1, j - 1))];
                const Vec3& diagonal_second = state.deformed_positions[
                    static_cast<std::size_t>(cloth_node(
                        main_diagonal ? i + 1 : i, j))];
                EXPECT_NEAR(
                    (diagonal_second - diagonal_first).norm(),
                    rest_diagonal, 2.0e-14);
                record_leader_edge_strain(
                    diagonal_first, diagonal_second, rest_diagonal);
            }
        }
        previous_leader_centerline = centerline;
    }
    EXPECT_LT(maximum_leader_edge_strain, 1.0e-12);

    const Vec3 join_centerline = 0.5 * (
        state.deformed_positions[static_cast<std::size_t>(
            cloth_node(0, leader_rows))]
        + state.deformed_positions[static_cast<std::size_t>(
            cloth_node(cloth_nx, leader_rows))]);
    EXPECT_TRUE(join_centerline.isApprox(
        expected_join_centerline, 2.0e-14));
    EXPECT_NEAR(
        (join_centerline - ramp_top).dot(ramp_normal),
        roll_clearance, 2.0e-14);
    EXPECT_NEAR(
        (join_centerline - ramp_top).dot(downhill),
        leader_downslope_span, 2.0e-14);
    const Vec3 join_radius = join_centerline - roll_center;
    const Vec3 actual_leader_tangent =
        (join_centerline - pin_centerline).normalized();
    const Vec3 expected_leader_tangent =
        (leader_downslope_span * downhill
         - leader_clearance_drop * ramp_normal)
        / leader_material_length;
    EXPECT_NEAR(join_radius.norm(), outer_radius, 2.0e-14);
    EXPECT_TRUE(actual_leader_tangent.isApprox(
        expected_leader_tangent, 2.0e-14));
    EXPECT_NEAR(expected_leader_tangent.norm(), 1.0, 2.0e-15);
    EXPECT_NEAR(
        join_radius.dot(actual_leader_tangent),
        outer_radius * leader_clearance_drop
            / leader_material_length,
        2.0e-15);

    // Phase zero puts the join at the roll's lowest point. The straight
    // leader intentionally gives up exact C1 continuity; the mismatch combines
    // its gentle downward slope with the spiral's upward radial rate.
    const Vec3 expected_spiral_join_tangent =
        (outer_radius * downhill + spiral_a * ramp_normal)
        / std::hypot(outer_radius, spiral_a);
    const double leader_slope_angle = std::atan2(
        leader_clearance_drop, leader_downslope_span);
    const double spiral_tangent_angle =
        std::atan2(spiral_a, outer_radius);
    const double join_tangent_angle =
        leader_slope_angle + spiral_tangent_angle;
    EXPECT_NEAR(leader_slope_angle, 0.0781702549948662, 1.0e-15);
    EXPECT_NEAR(join_tangent_angle, 0.08851495727463299, 1.0e-15);
    EXPECT_NEAR(
        join_tangent_angle * 180.0 / kPi,
        5.071533475617274, 1.0e-13);
    EXPECT_NEAR(
        std::acos(std::clamp(
            expected_leader_tangent.dot(expected_spiral_join_tangent),
            -1.0, 1.0)),
        join_tangent_angle, 3.0e-14);

    // The roll retains its normal height while shifting only 0.583 mm
    // upslope to keep the descending leader exactly strain-free.
    EXPECT_NEAR(
        leader_material_length - roll_downslope_offset,
        0.0005826651321841625, 1.0e-15);
    EXPECT_NEAR(
        (roll_center - ramp_top).dot(ramp_normal),
        outer_radius + roll_clearance, 2.0e-14);
    const Vec3 lowest_roll_point =
        roll_center - outer_radius * ramp_normal;
    EXPECT_TRUE(lowest_roll_point.isApprox(
        expected_join_centerline, 2.0e-14));
    EXPECT_NEAR(
        (lowest_roll_point - ramp_top).dot(ramp_normal),
        roll_clearance, 2.0e-14);
    EXPECT_NEAR(
        (lowest_roll_point - ramp_top).dot(downhill),
        roll_downslope_offset, 2.0e-14);
    const Vec3 first_spiral_centerline = 0.5 * (
        state.deformed_positions[static_cast<std::size_t>(
            cloth_node(0, leader_rows + 1))]
        + state.deformed_positions[static_cast<std::size_t>(
            cloth_node(cloth_nx, leader_rows + 1))]);
    EXPECT_GT(
        (first_spiral_centerline - ramp_top).dot(ramp_normal),
        roll_clearance);

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
    EXPECT_NEAR(
        nearest_nonlocal_cloth_segment_distance,
        0.005291213099085894, 1.0e-13);
    EXPECT_GT(
        nearest_nonlocal_cloth_segment_distance - params.d_hat,
        2.5e-4);
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
    const std::array<Vec3, ground_nodes> expected_ground_positions{
        Vec3(-3.0, -0.001, -2.0),
        Vec3(-3.0, -0.001,  4.0),
        Vec3( 3.0, -0.001,  4.0),
        Vec3( 3.0, -0.001, -2.0)};
    const std::array<int, 3 * ground_triangles> expected_ground_tris{
        wedge_nodes, wedge_nodes + 1, wedge_nodes + 2,
        wedge_nodes, wedge_nodes + 2, wedge_nodes + 3};

    // The finite wedge and ground quad are one static visualization mesh.
    // Neither contributes a node, triangle, mass, or rigid body to RefMesh.
    ASSERT_EQ(static_x.size(), wedge_nodes + ground_nodes);
    ASSERT_EQ(
        static_tris.size(),
        3 * (wedge_triangles + ground_triangles));
    double maximum_wedge_ramp_coordinate =
        -std::numeric_limits<double>::infinity();
    for (int local = 0; local < wedge_nodes; ++local) {
        EXPECT_TRUE(static_x[static_cast<std::size_t>(local)].isApprox(
                expected_wedge_positions[static_cast<std::size_t>(local)],
                0.0));
        const double ramp_coordinate =
            (static_x[static_cast<std::size_t>(local)] - ramp_top)
                .dot(ramp_normal);
        EXPECT_LE(ramp_coordinate, 1.0e-14);
        maximum_wedge_ramp_coordinate = std::max(
            maximum_wedge_ramp_coordinate, ramp_coordinate);
    }
    EXPECT_NEAR(maximum_wedge_ramp_coordinate, 0.0, 1.0e-14);
    for (int local = 0; local < 3 * wedge_triangles; ++local) {
        EXPECT_EQ(
            static_tris[static_cast<std::size_t>(local)],
            expected_wedge_tris[static_cast<std::size_t>(local)]);
    }
    const Vec3 visual_ramp_normal =
        (expected_wedge_positions[4] - expected_wedge_positions[2])
            .cross(expected_wedge_positions[5]
                   - expected_wedge_positions[2])
            .normalized();
    EXPECT_TRUE(visual_ramp_normal.isApprox(ramp_normal, 1.0e-15));
    for (int local = 0; local < ground_nodes; ++local) {
        EXPECT_TRUE(static_x[static_cast<std::size_t>(wedge_nodes + local)]
            .isApprox(
                expected_ground_positions[static_cast<std::size_t>(local)],
                0.0));
    }
    for (int local = 0; local < 3 * ground_triangles; ++local) {
        EXPECT_EQ(
            static_tris[static_cast<std::size_t>(
                3 * wedge_triangles + local)],
            expected_ground_tris[static_cast<std::size_t>(local)]);
    }
    EXPECT_DOUBLE_EQ(
        expected_ground_positions[2].x()
            - expected_ground_positions[0].x(),
        6.0);
    EXPECT_DOUBLE_EQ(
        expected_ground_positions[1].z()
            - expected_ground_positions[0].z(),
        6.0);
    EXPECT_TRUE((
        (expected_ground_positions[1] - expected_ground_positions[0])
            .cross(
                expected_ground_positions[2]
                    - expected_ground_positions[0])
            .normalized()).isApprox(Vec3::UnitY(), 0.0));
    for (const int static_vertex : static_tris) {
        EXPECT_GE(static_vertex, 0);
        EXPECT_LT(static_vertex, static_cast<int>(static_x.size()));
    }

    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.tet_rest_data.empty());
    EXPECT_TRUE(ref_mesh.tet_adj.empty());
    EXPECT_TRUE(ref_mesh.tet_nodes.empty());
    EXPECT_TRUE(ref_mesh.surface_nodes.empty());
    EXPECT_TRUE(ref_mesh.rb_nodes.empty());
    EXPECT_TRUE(ref_mesh.ref_positions.empty());
    EXPECT_TRUE(ref_mesh.total_mass.empty());
    EXPECT_TRUE(ref_mesh.I_hat.empty());
    EXPECT_TRUE(ref_mesh.rb_update_modes.empty());
    EXPECT_TRUE(state.x_coms.empty());
    EXPECT_TRUE(state.v_coms.empty());
    EXPECT_TRUE(state.orientations.empty());
    EXPECT_TRUE(state.omega.empty());
    ASSERT_EQ(ref_mesh.deformable_nodes.size(), cloth_nodes);
    for (int node = 0; node < cloth_nodes; ++node)
        EXPECT_EQ(ref_mesh.deformable_nodes[node], node);

    // Match simulation startup for a pure deformable scene. The builder
    // leaves ownership and masses empty; neither visual mesh nor SDF plane
    // contributes an entry when those arrays are initialized.
    ref_mesh.node_to_rb.assign(total_nodes, -1);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    ASSERT_EQ(ref_mesh.node_to_rb.size(), total_nodes);
    ASSERT_EQ(ref_mesh.mass.size(), total_nodes);
    double total_cloth_mass = 0.0;
    for (int node = 0; node < cloth_nodes; ++node) {
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        total_cloth_mass += ref_mesh.mass[node];
    }
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * cloth_width * cloth_length,
        1.0e-12);

    double minimum_mesh_edge = std::numeric_limits<double>::infinity();
    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int first = ref_mesh.tris[3 * triangle + local];
            const int second =
                ref_mesh.tris[3 * triangle + (local + 1) % 3];
            ASSERT_GE(first, 0);
            ASSERT_GE(second, 0);
            ASSERT_LT(first, static_cast<int>(state.deformed_positions.size()));
            ASSERT_LT(second, static_cast<int>(state.deformed_positions.size()));
            minimum_mesh_edge = std::min(
                minimum_mesh_edge,
                (state.deformed_positions[static_cast<std::size_t>(second)]
                 - state.deformed_positions[static_cast<std::size_t>(first)])
                    .norm());
        }
    }
    EXPECT_LT(params.d_hat, 0.5 * minimum_mesh_edge);
}

TEST(ClothExample, RolledClothClearanceTracksLargerSDFRange) {
    IPCArgs3D args;
    args.d_hat = 0.001;
    args.k_sdf = 250000.0;
    args.eps_sdf = 0.012;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    build_cloth_unrolling_down_fixed_ramp_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    ASSERT_EQ(params.sdf_planes.size(), 2U);
    EXPECT_DOUBLE_EQ(params.k_sdf, args.k_sdf);
    EXPECT_DOUBLE_EQ(params.eps_sdf, args.eps_sdf);
    EXPECT_TRUE(ref_mesh.rb_nodes.empty());
    EXPECT_TRUE(state.x_coms.empty());

    // eps_sdf, rather than d_hat, controls this construction. Every cloth
    // node starts outside both SDF penalty ranges and therefore has zero SDF
    // energy. The roll is authored just 0.1 mm beyond that larger range.
    double minimum_union_phi = std::numeric_limits<double>::infinity();
    for (const Vec3& position : state.deformed_positions) {
        const SDFEvaluation ground =
            evaluate_sdf(params.sdf_planes[0], position);
        const SDFEvaluation incline =
            evaluate_sdf(params.sdf_planes[1], position);
        const SDFEvaluation& nearest =
            ground.phi < incline.phi ? ground : incline;
        minimum_union_phi = std::min(minimum_union_phi, nearest.phi);
        EXPECT_GT(nearest.phi, params.eps_sdf);
        EXPECT_DOUBLE_EQ(
            sdf_penalty_energy(
                nearest, params.k_sdf, params.eps_sdf),
            0.0);
    }
    EXPECT_NEAR(
        minimum_union_phi, args.eps_sdf + 1.0e-4, 2.0e-12);
}

TEST(SdfMaterialMotionExample,
     TwoCylinderUpdateUsesAbsoluteSubstepTimesAndIsRestartSafe) {
    IPCArgs3D args;
    args.fps = 20.0;
    args.substeps = 4;
    args.tcyl_n_strips = 1;
    args.tcyl_nx = 2;
    args.tcyl_ny = 8;
    args.tcyl_nu = 8;
    args.tcyl_twist_rate = 0.37;
    args.tcyl_settle_time = 0.0;
    args.tcyl_ramp_time = 0.0;
    args.tcyl_max_turn = 0.0;
    args.tcyl_untwist = false;

    const auto build_scene = [&](SimParams& params,
                                 CylinderTwistSpec& spec) {
        RefMesh ref_mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        std::vector<Vec3> static_x;
        std::vector<int> static_tris;
        params = args.to_sim_params();
        build_two_cylinder_twist_example(
            args, ref_mesh, state, X, pins, params,
            static_x, static_tris, spec);
    };

    SimParams sequential_params;
    CylinderTwistSpec sequential_spec;
    build_scene(sequential_params, sequential_spec);
    ASSERT_EQ(sequential_params.sdf_cylinders.size(), 2U);
    const double dt = sequential_params.dt();
    constexpr int target_substep = 17;
    for (int substep = 1; substep <= target_substep; ++substep) {
        update_cylinder_sdfs(
            sequential_params, sequential_spec, substep * dt);
    }

    // A restarted run rebuilds the scene and jumps directly to the next
    // absolute substep time. It must reconstruct the same previous/current
    // material poses without relying on updater call history.
    SimParams restarted_params;
    CylinderTwistSpec restarted_spec;
    build_scene(restarted_params, restarted_spec);
    update_cylinder_sdfs(
        restarted_params, restarted_spec, target_substep * dt);
    ASSERT_EQ(restarted_params.sdf_cylinders.size(), 2U);

    SimParams predecessor_params;
    CylinderTwistSpec predecessor_spec;
    build_scene(predecessor_params, predecessor_spec);
    update_cylinder_sdfs(
        predecessor_params, predecessor_spec,
        (target_substep - 1) * dt);

    for (std::size_t cylinder_index = 0; cylinder_index < 2;
         ++cylinder_index) {
        SCOPED_TRACE(cylinder_index);
        const CylinderSDF& sequential =
            sequential_params.sdf_cylinders[cylinder_index];
        const CylinderSDF& restarted =
            restarted_params.sdf_cylinders[cylinder_index];
        const CylinderSDF& predecessor =
            predecessor_params.sdf_cylinders[cylinder_index];
        expect_sdf_material_motion_near(
            restarted.material_motion, sequential.material_motion);
        expect_sdf_material_pose_near(
            restarted.material_motion.previous,
            predecessor.material_motion.current);
        EXPECT_TRUE(restarted.axis.isApprox(
            restarted.material_motion.current.rotation * Vec3::UnitX(),
            1.0e-14));

        // Rotation is about the cylinder center, so that point stays fixed.
        EXPECT_TRUE((restarted.material_motion.current.rotation
                     * restarted.point
                     + restarted.material_motion.current.translation)
                        .isApprox(restarted.point, 1.0e-14));

        // A material surface point maps to the preceding absolute-time pose.
        const Vec3 rest_surface =
            restarted.point + restarted.radius * Vec3::UnitZ();
        const Vec3 current_surface =
            restarted.material_motion.current.rotation * rest_surface
            + restarted.material_motion.current.translation;
        const Vec3 expected_previous_surface =
            restarted.material_motion.previous.rotation * rest_surface
            + restarted.material_motion.previous.translation;
        EXPECT_NEAR(evaluate_sdf(restarted, current_surface).phi,
                    0.0, 2.0e-14);
        EXPECT_TRUE(sdf_previous_material_point(
                        restarted.material_motion, current_surface)
                        .isApprox(expected_previous_surface, 2.0e-14));
    }
}

TEST(SdfMaterialMotionExample,
     TwistUntwistCylinderUpdateUsesAbsoluteTimesAndIsRestartSafe) {
    IPCArgs3D args;
    args.fps = 24.0;
    args.substeps = 3;
    args.tu_nx = 2;
    args.tu_ny = 8;
    args.tu_cyl_nu = 8;
    args.tu_twist_rate = 0.29;
    args.tu_settle_time = 0.0;
    args.tu_ramp_time = 0.0;
    args.tu_max_turn = 0.0;
    args.tu_untwist = false;

    const auto build_scene = [&](SimParams& params,
                                 TwistUntwistSpec& spec) {
        RefMesh ref_mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        std::vector<Vec3> static_x;
        std::vector<int> static_tris;
        params = args.to_sim_params();
        build_twist_untwist_example(
            args, ref_mesh, state, X, pins, params,
            static_x, static_tris, spec);
    };

    SimParams sequential_params;
    TwistUntwistSpec sequential_spec;
    build_scene(sequential_params, sequential_spec);
    ASSERT_GE(sequential_spec.cyl_sdf_index, 0);
    const double dt = sequential_params.dt();
    constexpr int target_substep = 13;
    for (int substep = 1; substep <= target_substep; ++substep) {
        update_twist_untwist_sdf(
            sequential_params, sequential_spec, substep * dt);
    }

    SimParams restarted_params;
    TwistUntwistSpec restarted_spec;
    build_scene(restarted_params, restarted_spec);
    update_twist_untwist_sdf(
        restarted_params, restarted_spec, target_substep * dt);

    SimParams predecessor_params;
    TwistUntwistSpec predecessor_spec;
    build_scene(predecessor_params, predecessor_spec);
    update_twist_untwist_sdf(
        predecessor_params, predecessor_spec,
        (target_substep - 1) * dt);

    ASSERT_EQ(restarted_spec.cyl_sdf_index,
              sequential_spec.cyl_sdf_index);
    ASSERT_EQ(predecessor_spec.cyl_sdf_index,
              sequential_spec.cyl_sdf_index);
    const std::size_t sdf_index = static_cast<std::size_t>(
        sequential_spec.cyl_sdf_index);
    const CylinderSDF& sequential =
        sequential_params.sdf_cylinders[sdf_index];
    const CylinderSDF& restarted =
        restarted_params.sdf_cylinders[sdf_index];
    const CylinderSDF& predecessor =
        predecessor_params.sdf_cylinders[sdf_index];
    expect_sdf_material_motion_near(
        restarted.material_motion, sequential.material_motion);
    expect_sdf_material_pose_near(
        restarted.material_motion.previous,
        predecessor.material_motion.current);
    EXPECT_TRUE(restarted.axis.isApprox(
        restarted.material_motion.current.rotation * Vec3::UnitX(),
        1.0e-14));
    EXPECT_TRUE((restarted.material_motion.current.rotation
                 * restarted.point
                 + restarted.material_motion.current.translation)
                    .isApprox(restarted.point, 1.0e-14));

    const Vec3 rest_surface =
        restarted.point + restarted.radius * Vec3::UnitZ();
    const Vec3 current_surface =
        restarted.material_motion.current.rotation * rest_surface
        + restarted.material_motion.current.translation;
    const Vec3 expected_previous_surface =
        restarted.material_motion.previous.rotation * rest_surface
        + restarted.material_motion.previous.translation;
    EXPECT_NEAR(evaluate_sdf(restarted, current_surface).phi,
                0.0, 2.0e-14);
    EXPECT_TRUE(sdf_previous_material_point(
                    restarted.material_motion, current_surface)
                    .isApprox(expected_previous_surface, 2.0e-14));
}

TEST(OscillatingClothLayersExample,
     DefaultSceneMatchesPaperFixedAndOppositeDrivenEdges) {
    IPCArgs3D args;
    args.E = 2.5e8;
    args.nu = 0.25;
    args.thickness = 0.001;
    args.k_barrier = 1.0e5;
    args.friction_coefficient = 0.1;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    OscillatingClothLayersSpec spec;
    build_oscillating_cloth_layers_example(
        args, ref_mesh, state, X, pins, params, spec);

    constexpr int layer_count = 3;
    constexpr int nodes_per_axis = 50;
    constexpr int cells_per_axis = 49;
    constexpr int nodes_per_layer = nodes_per_axis * nodes_per_axis;
    constexpr int triangles_per_layer =
        2 * cells_per_axis * cells_per_axis;
    constexpr int hinges_per_layer =
        3 * cells_per_axis * cells_per_axis
        - 2 * cells_per_axis;

    EXPECT_EQ(state.deformed_positions.size(), 7500U);
    EXPECT_EQ(state.velocities.size(), 7500U);
    EXPECT_EQ(X.size(), 7500U);
    EXPECT_EQ(ref_mesh.tris.size() / 3, 14406U);
    EXPECT_EQ(ref_mesh.hinges.size(),
              static_cast<std::size_t>(layer_count * hinges_per_layer));
    EXPECT_EQ(ref_mesh.deformable_nodes.size(), 7500U);
    EXPECT_EQ(pins.size(), 300U);
    EXPECT_EQ(spec.driven_pin_indices.size(), 150U);
    EXPECT_EQ(spec.driven_initial_targets.size(), 150U);

    EXPECT_NEAR(params.mu, 1.0e5, 1.0e-10);
    EXPECT_NEAR(params.lambda, 1.0e5, 1.0e-10);
    EXPECT_DOUBLE_EQ(params.k_barrier, 1.0e5);
    EXPECT_DOUBLE_EQ(params.friction_coefficient, 0.1);
    EXPECT_DOUBLE_EQ(params.d_hat, 0.002);
    EXPECT_GT(args.osc_layer_gap, params.d_hat);
    EXPECT_DOUBLE_EQ(params.k_sdf, 0.0);
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.rb_nodes.empty());

    for (const Vec3& velocity : state.velocities)
        EXPECT_TRUE(velocity.isZero(0.0));

    for (int layer = 0; layer < layer_count; ++layer) {
        const int node_begin = layer * nodes_per_layer;
        const int node_end = node_begin + nodes_per_layer;
        const double expected_y = 0.20
            + static_cast<double>(layer - 1) * args.osc_layer_gap;
        for (int local_node = 0; local_node < nodes_per_layer;
             ++local_node) {
            EXPECT_NEAR(
                state.deformed_positions[static_cast<std::size_t>(
                    node_begin + local_node)].y(),
                expected_y, 1.0e-15);
        }

        const int triangle_begin = layer * triangles_per_layer;
        const int triangle_end = triangle_begin + triangles_per_layer;
        for (int triangle = triangle_begin; triangle < triangle_end;
             ++triangle) {
            for (int local = 0; local < 3; ++local) {
                const int vertex = ref_mesh.tris[
                    static_cast<std::size_t>(3 * triangle + local)];
                EXPECT_GE(vertex, node_begin);
                EXPECT_LT(vertex, node_end);
            }
        }

        for (int j = 0; j < nodes_per_axis; ++j) {
            const int edge_entry = layer * nodes_per_axis + j;
            const int driven_pin_index = 2 * edge_entry;
            const int fixed_pin_index = driven_pin_index + 1;
            const int driven_vertex = node_begin + j * nodes_per_axis;
            const int fixed_vertex = driven_vertex + cells_per_axis;

            EXPECT_EQ(spec.driven_pin_indices[
                          static_cast<std::size_t>(edge_entry)],
                      driven_pin_index);
            EXPECT_EQ(pins[static_cast<std::size_t>(driven_pin_index)]
                          .vertex_index,
                      driven_vertex);
            EXPECT_EQ(pins[static_cast<std::size_t>(fixed_pin_index)]
                          .vertex_index,
                      fixed_vertex);
            EXPECT_TRUE(pins[static_cast<std::size_t>(driven_pin_index)]
                            .target_position.isApprox(
                                state.deformed_positions[
                                    static_cast<std::size_t>(driven_vertex)],
                                0.0));
            EXPECT_TRUE(pins[static_cast<std::size_t>(fixed_pin_index)]
                            .target_position.isApprox(
                                state.deformed_positions[
                                    static_cast<std::size_t>(fixed_vertex)],
                                0.0));
        }

        // Only the fixed and driven edges are constrained. The two lateral
        // edges are free except at their shared clamped corner vertices.
        std::vector<bool> is_pinned(
            static_cast<std::size_t>(nodes_per_layer), false);
        for (const Pin& pin : pins) {
            if (pin.vertex_index >= node_begin && pin.vertex_index < node_end) {
                is_pinned[static_cast<std::size_t>(
                    pin.vertex_index - node_begin)] = true;
            }
        }
        for (int j = 0; j < nodes_per_axis; ++j) {
            EXPECT_TRUE(is_pinned[static_cast<std::size_t>(
                j * nodes_per_axis)]);
            EXPECT_TRUE(is_pinned[static_cast<std::size_t>(
                j * nodes_per_axis + cells_per_axis)]);
        }
        for (int i = 1; i < cells_per_axis; ++i) {
            EXPECT_FALSE(is_pinned[static_cast<std::size_t>(i)]);
            EXPECT_FALSE(is_pinned[static_cast<std::size_t>(
                (nodes_per_axis - 1) * nodes_per_axis + i)]);
        }
    }
}

TEST(OscillatingClothLayersExample,
     SignedSinusoidUsesAbsoluteTimeAndLeavesFixedEdgeUnchanged) {
    IPCArgs3D args;
    args.osc_nx = 2;
    args.osc_nz = 2;
    args.osc_length = 0.60;
    args.osc_width = 0.30;
    args.osc_layer_gap = 0.02;
    args.osc_amplitude = 0.08;
    args.osc_frequency = 0.25;
    args.d_hat = 0.01;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    OscillatingClothLayersSpec spec;
    build_oscillating_cloth_layers_example(
        args, ref_mesh, state, X, pins, params, spec);

    ASSERT_EQ(pins.size(), 18U);
    ASSERT_EQ(spec.driven_pin_indices.size(), 9U);
    ASSERT_EQ(spec.driven_initial_targets.size(), 9U);
    const std::vector<Pin> initial_pins = pins;

    const double quarter_period = 0.25 / args.osc_frequency;
    update_oscillating_cloth_layer_pins(
        pins, spec, quarter_period);
    const std::vector<Pin> quarter_period_pins = pins;

    for (std::size_t driven = 0;
         driven < spec.driven_pin_indices.size(); ++driven) {
        const std::size_t pin_index = static_cast<std::size_t>(
            spec.driven_pin_indices[driven]);
        const Vec3 expected = spec.driven_initial_targets[driven]
            + args.osc_amplitude * Vec3::UnitY();
        EXPECT_TRUE(pins[pin_index].target_position.isApprox(
            expected, 1.0e-14));
    }
    for (std::size_t pin_index = 1; pin_index < pins.size();
         pin_index += 2) {
        EXPECT_TRUE(pins[pin_index].target_position.isApprox(
            initial_pins[pin_index].target_position, 0.0));
    }

    // Visiting another phase first must not change a later result: the
    // updater reconstructs each target from its immutable t=0 position.
    const double half_period = 0.5 / args.osc_frequency;
    update_oscillating_cloth_layer_pins(pins, spec, half_period);
    for (std::size_t driven = 0;
         driven < spec.driven_pin_indices.size(); ++driven) {
        const std::size_t pin_index = static_cast<std::size_t>(
            spec.driven_pin_indices[driven]);
        const Vec3& expected = spec.driven_initial_targets[driven];
        EXPECT_TRUE(pins[pin_index].target_position.isApprox(
            expected, 1.0e-14));
    }
    update_oscillating_cloth_layer_pins(
        pins, spec, quarter_period);
    for (std::size_t pin_index = 0; pin_index < pins.size(); ++pin_index) {
        EXPECT_TRUE(pins[pin_index].target_position.isApprox(
            quarter_period_pins[pin_index].target_position, 1.0e-14));
    }

    // A freshly rebuilt scene (the restart path) evaluated directly at the
    // same absolute time must recover the identical pin targets.
    RefMesh restarted_ref_mesh;
    DeformedState restarted_state;
    std::vector<Vec2> restarted_X;
    std::vector<Pin> restarted_pins;
    SimParams restarted_params = args.to_sim_params();
    OscillatingClothLayersSpec restarted_spec;
    build_oscillating_cloth_layers_example(
        args, restarted_ref_mesh, restarted_state, restarted_X,
        restarted_pins, restarted_params, restarted_spec);
    update_oscillating_cloth_layer_pins(
        restarted_pins, restarted_spec, quarter_period);
    ASSERT_EQ(restarted_pins.size(), quarter_period_pins.size());
    for (std::size_t pin_index = 0;
         pin_index < restarted_pins.size(); ++pin_index) {
        EXPECT_TRUE(restarted_pins[pin_index].target_position.isApprox(
            quarter_period_pins[pin_index].target_position, 1.0e-14));
    }

    // A full cycle samples the complete signed range while fixed targets stay
    // bitwise unchanged.
    const double period = 1.0 / args.osc_frequency;
    for (const double time : {
             0.0, 0.25 * period, 0.5 * period,
             0.75 * period, period}) {
        update_oscillating_cloth_layer_pins(pins, spec, time);
        for (std::size_t driven = 0;
             driven < spec.driven_pin_indices.size(); ++driven) {
            const std::size_t pin_index = static_cast<std::size_t>(
                spec.driven_pin_indices[driven]);
            const double lift = pins[pin_index].target_position.y()
                - spec.driven_initial_targets[driven].y();
            EXPECT_GE(lift, -args.osc_amplitude - 1.0e-14);
            EXPECT_LE(lift, args.osc_amplitude + 1.0e-14);
        }
        for (std::size_t pin_index = 1; pin_index < pins.size();
             pin_index += 2) {
            EXPECT_TRUE(pins[pin_index].target_position.isApprox(
                initial_pins[pin_index].target_position, 0.0));
        }
    }
}

TEST(CylinderMesh, RadialCapsAreClosedOutwardAndHaveFiniteReferenceTriangles) {
    constexpr int nu = 12;
    constexpr double radius = .4;
    const Vec3 center(2.0, 3.0, 4.0);
    for (int cap_rings : {1, 5}) for (double length : {.2, .4}) {
        SCOPED_TRACE(::testing::Message() << "rings=" << cap_rings << " length=" << length);
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        build_square_mesh(mesh, state, X, 1, 1, 1.0, 1.0, Vec3(10,0,0));
        const auto prior_tris = mesh.tris;
        const int base = build_cylinder_mesh(mesh, state, X, nu, radius, length, center, cap_rings);
        ASSERT_EQ(base, 4);
        const int rows = std::max(1, static_cast<int>(std::round(
            length / ((2.0 * M_PI * radius / nu) * .5 * std::sqrt(3.0)))));
        EXPECT_EQ(state.deformed_positions.size(),
            4U + nu * (rows + 1) + 2 + 2 * nu * (cap_rings - 1));
        EXPECT_EQ(mesh.tris.size() / 3, 2U + 2 * nu * rows + 2 * nu * (2 * cap_rings - 1));
        EXPECT_TRUE(std::equal(prior_tris.begin(), prior_tris.end(), mesh.tris.begin()));
        std::map<std::pair<int,int>,int> edges;
        int caps[2] = {0,0};
        for (std::size_t t = prior_tris.size(); t < mesh.tris.size(); t += 3) {
            const Vec3 a = state.deformed_positions[mesh.tris[t]] - center;
            const Vec3 b = state.deformed_positions[mesh.tris[t+1]] - center;
            const Vec3 c = state.deformed_positions[mesh.tris[t+2]] - center;
            const Vec3 normal = (b-a).cross(c-a);
            EXPECT_GT(normal.squaredNorm(), 0.0);
            if (std::abs(a.z()-b.z()) < 1e-12 && std::abs(a.z()-c.z()) < 1e-12) {
                const int end = a.z() < 0 ? 0 : 1;
                ++caps[end];
                EXPECT_NEAR(std::abs(a.z()), .5 * length, 1e-12);
                EXPECT_GT((end == 0 ? -1.0 : 1.0) * normal.z(), 0.0);
            } else {
                const Vec3 midpoint = (a+b+c)/3.0;
                EXPECT_GT(normal.head<2>().dot(midpoint.head<2>()), 0.0);
            }
            for (int j = 0; j < 3; ++j) {
                int first = mesh.tris[t+j], second = mesh.tris[t+(j+1)%3];
                if (first > second) std::swap(first, second);
                ++edges[{first,second}];
            }
            EXPECT_TRUE(mesh.Dm_inverse[t/3].allFinite());
            EXPECT_GT(mesh.area[t/3], 0.0);
        }
        for (const auto& edge : edges) EXPECT_EQ(edge.second, 2);
        EXPECT_EQ(caps[0], nu * (2 * cap_rings - 1));
        EXPECT_EQ(caps[1], nu * (2 * cap_rings - 1));
        EXPECT_THROW(build_cylinder_mesh(mesh,state,X,nu,radius,length,center,0), std::invalid_argument);
    }
}

TEST(ClothCylinderDropExample, BatchedRestDataMatchesIncrementalGridConstruction) {
    IPCArgs3D args;
    args.d_hat = 0.0048;
    args.drop_stack_count = 4;
    args.drop_cloth_nx = 5;
    args.drop_cloth_ny = 7;
    args.drop_cloth_w = 3.1;
    args.drop_cloth_h = 2.3;
    RefMesh batched, incremental;
    DeformedState state, expected;
    std::vector<Vec2> X, expected_X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    build_cloth_cylinder_drop_example(args, batched, state, X, pins,
        params, static_x, static_tris);
    for (int sheet = 0; sheet < args.drop_stack_count; ++sheet) {
        build_square_mesh(incremental, expected, expected_X,
            args.drop_cloth_nx, args.drop_cloth_ny,
            args.drop_cloth_w, args.drop_cloth_h,
            Vec3(args.drop_cx - 0.5 * args.drop_cloth_w,
                 args.drop_first_y + sheet * args.drop_spacing,
                 args.drop_cz - 0.5 * args.drop_cloth_h));
    }
    EXPECT_EQ(batched.tris, incremental.tris);
    EXPECT_EQ(batched.area, incremental.area);
    EXPECT_EQ(batched.hinge_adj, incremental.hinge_adj);
    ASSERT_EQ(batched.num_positions, incremental.num_positions);
    ASSERT_EQ(batched.hinges.size(), incremental.hinges.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
        EXPECT_TRUE((X[i].array() == expected_X[i].array()).all());
        EXPECT_TRUE((state.deformed_positions[i].array()
            == expected.deformed_positions[i].array()).all());
    }
    for (std::size_t i = 0; i < batched.Dm_inverse.size(); ++i)
        EXPECT_TRUE((batched.Dm_inverse[i].array()
            == incremental.Dm_inverse[i].array()).all());
    for (std::size_t i = 0; i < batched.hinges.size(); ++i) {
        const auto& actual = batched.hinges[i];
        const auto& reference = incremental.hinges[i];
        for (int role = 0; role < 4; ++role)
            EXPECT_EQ(actual.v[role], reference.v[role]);
        EXPECT_DOUBLE_EQ(actual.bar_theta, reference.bar_theta);
        EXPECT_DOUBLE_EQ(actual.c_e, reference.c_e);
    }
}

TEST(ClothCylinderDropExample, RejectsStackBeyondIntegerIndexLimits) {
    IPCArgs3D args;
    args.drop_stack_count = 200;
    args.drop_cloth_nx = std::numeric_limits<int>::max();
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    EXPECT_THROW(build_cloth_cylinder_drop_example(args, mesh, state, X,
        pins, params, static_x, static_tris), std::invalid_argument);
}

TEST(ClothCylinderDropExample, ContactDefaultIsSceneSpecificAndFlagsOverrideIt) {
    IPCArgs3D args;
    char program[] = "3D_sim";
    char option[] = "--example";
    char scene[] = "14";
    char distance_option[] = "--d_hat";
    char distance[] = "0.0015";
    char* argv[] = {program, option, scene, distance_option, distance};
    ASSERT_TRUE(args.parse(3, argv));
    EXPECT_DOUBLE_EQ(args.d_hat, 0.0048);
    ASSERT_TRUE(args.parse(5, argv));
    EXPECT_DOUBLE_EQ(args.d_hat, 0.0015);
    IPCArgs3D other_scene;
    ASSERT_TRUE(other_scene.parse(1, argv));
    EXPECT_DOUBLE_EQ(other_scene.d_hat, 0.005);
}

TEST(IPCArgsCCD, ExactFallbackFlagKeepsTightInclusionIndependent) {
    IPCArgs3D args;
    std::vector<std::string> values = {"3D_sim", "--use_ticcd", "true",
        "--exact_computation_fallback"};
    std::vector<char*> argv;
    for (auto& value : values) argv.push_back(value.data());
    ASSERT_TRUE(args.parse(static_cast<int>(argv.size()), argv.data()));
    EXPECT_TRUE(args.exact_computation_fallback);
    EXPECT_TRUE(args.to_sim_params().use_ticcd);
    EXPECT_FALSE(args.to_sim_params().use_original_linear_ccd());

    IPCArgs3D tight_inclusion;
    values = {"3D_sim", "--use_ticcd", "true", "--exact_computation_fallback", "false"};
    argv.clear();
    for (auto& value : values) argv.push_back(value.data());
    ASSERT_TRUE(tight_inclusion.parse(static_cast<int>(argv.size()), argv.data()));
    EXPECT_FALSE(tight_inclusion.exact_computation_fallback);
    EXPECT_TRUE(tight_inclusion.to_sim_params().use_ticcd);
}

TEST(ClothCylinderDropExample, Example15PresetPreservesScaleAndRequestedBudget) {
    IPCArgs3D args;
    char program[]="3D_sim", option[]="--example", scene[]="15";
    char* argv[]={program,option,scene};
    ASSERT_TRUE(args.parse(3,argv));
    EXPECT_EQ(args.drop_stack_count,200);
    EXPECT_EQ(args.drop_cloth_nx,140);
    EXPECT_EQ(args.drop_cloth_ny,140);
    EXPECT_EQ(args.drop_stack_count*(args.drop_cloth_nx+1)*(args.drop_cloth_ny+1),3976200);
    EXPECT_DOUBLE_EQ(args.drop_cloth_w/args.drop_cloth_nx,.03);
    EXPECT_DOUBLE_EQ(args.drop_cloth_h/args.drop_cloth_ny,.03);
    EXPECT_DOUBLE_EQ(args.drop_spacing,.005);
    EXPECT_DOUBLE_EQ(args.E,1e6);
    EXPECT_DOUBLE_EQ(args.nu,.25);
    EXPECT_DOUBLE_EQ(args.kB,.001);
    EXPECT_EQ(args.num_frames,120);
    EXPECT_EQ(args.substeps,100);
    EXPECT_EQ(args.max_substep_iters,4);
    EXPECT_EQ(args.node_box_update_count,args.max_substep_iters);
    EXPECT_TRUE(args.fixed_iters);
    EXPECT_TRUE(args.use_simd);
    EXPECT_TRUE(args.exact_computation_fallback);
    EXPECT_GT(args.drop_first_y,args.cyl_cy+args.cyl_radius+args.cyl_sdf_padding+args.eps_sdf);
}

TEST(ClothCylinderDropExample, Example15FreeFallOverridesKeepUpdateCountConsistent) {
    IPCArgs3D args;
    std::vector<std::string> values={"3D_sim","--example","15","--substeps","40",
        "--max_substep_iters","3","--d_hat","0.0025","--node_box_max","0.001",
        "--exact_computation_fallback","false","--nu","0.3"};
    std::vector<char*> argv;
    for(auto& value:values)argv.push_back(value.data());
    ASSERT_TRUE(args.parse(static_cast<int>(argv.size()),argv.data()));
    EXPECT_EQ(args.substeps,40);
    EXPECT_EQ(args.max_substep_iters,3);
    EXPECT_EQ(args.node_box_update_count,3);
    EXPECT_DOUBLE_EQ(args.d_hat,.0025);
    EXPECT_DOUBLE_EQ(args.node_box_max,.001);
    EXPECT_DOUBLE_EQ(args.nu,.3);
    EXPECT_FALSE(args.exact_computation_fallback);
    EXPECT_TRUE(args.to_sim_params().use_original_linear_ccd());
}

TEST(ClothCylinderDropExample,
     RaisedCenteredPlacementOverridesPreserveClearanceAndGroundHeight) {
    IPCArgs3D args;
    char program[] = "make_shape_test";
    char example_key[] = "--example";
    char example_value[] = "14";
    char height_key[] = "--drop_first_y";
    char height_value[] = "2.20";
    char cylinder_height_key[] = "--cyl_cy";
    char cylinder_height_value[] = "1.60";
    char spacing_key[] = "--drop_spacing";
    char spacing_value[] = "0.012";
    char cylinder_x_key[] = "--cyl_cx";
    char cylinder_x_value[] = "0.0";
    char* argv[] = {program, example_key, example_value,
                    height_key, height_value,
                    cylinder_height_key, cylinder_height_value,
                    spacing_key, spacing_value,
                    cylinder_x_key, cylinder_x_value};
    ASSERT_TRUE(args.parse(11, argv));
    EXPECT_DOUBLE_EQ(args.drop_first_y, 2.20);
    EXPECT_DOUBLE_EQ(args.cyl_cy, 1.60);
    EXPECT_DOUBLE_EQ(args.cyl_cx, args.drop_cx);
    EXPECT_DOUBLE_EQ(args.cyl_cz, args.drop_cz);
    EXPECT_DOUBLE_EQ(args.drop_spacing, 0.012);
    EXPECT_DOUBLE_EQ(args.cyl_radius, 0.35);
    EXPECT_DOUBLE_EQ(args.cyl_sdf_padding, 0.012);

    // Exercise explicit raised/centered placement without changing defaults.
    args.drop_stack_count = 2;
    args.drop_cloth_nx = 2;
    args.drop_cloth_ny = 2;
    args.cyl_ground_size = 1.0;
    args.cyl_ground_cell_size = 1.0;
    args.cyl_nu = 8;
    args.cyl_cap_rings = 1;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    ASSERT_EQ(params.sdf_cylinders.size(), 1U);
    const CylinderSDF& cylinder = params.sdf_cylinders[0];
    EXPECT_DOUBLE_EQ(cylinder.point.y(), 1.60);
    EXPECT_NEAR(cylinder.radius, 0.362, 1.0e-14);
    ASSERT_EQ(state.deformed_positions.size(), 18U);
    for (int node = 0; node < 9; ++node) {
        const double lower_y = state.deformed_positions[node].y();
        const double upper_y = state.deformed_positions[9 + node].y();
        EXPECT_DOUBLE_EQ(lower_y, 2.20);
        EXPECT_NEAR(upper_y - lower_y, 0.012, 1.0e-14);
        EXPECT_NEAR(lower_y - cylinder.point.y(), 0.60, 1.0e-14);
        EXPECT_NEAR(lower_y - cylinder.point.y() - cylinder.radius,
                    0.238, 1.0e-14);
    }
    for (int sheet = 0; sheet < args.drop_stack_count; ++sheet) {
        Vec3 center = Vec3::Zero();
        for (int node = 0; node < 9; ++node)
            center += state.deformed_positions[9 * sheet + node];
        center /= 9;
        EXPECT_NEAR(center.x(), cylinder.point.x(), 1.0e-14);
        EXPECT_NEAR(center.z(), cylinder.point.z(), 1.0e-14);
    }

    ASSERT_EQ(params.sdf_planes.size(), 1U);
    EXPECT_TRUE(params.sdf_planes[0].point.isZero(0.0));
    EXPECT_TRUE(params.sdf_planes[0].normal.isApprox(Vec3::UnitY()));
    constexpr std::size_t ground_vertices = 4;
    ASSERT_GT(static_x.size(), ground_vertices);
    for (std::size_t node = 0; node < ground_vertices; ++node)
        EXPECT_DOUBLE_EQ(static_x[node].y(), 0.0);
    double cylinder_min_y = std::numeric_limits<double>::infinity();
    double cylinder_max_y = -std::numeric_limits<double>::infinity();
    for (std::size_t node = ground_vertices; node < static_x.size(); ++node) {
        cylinder_min_y = std::min(cylinder_min_y, static_x[node].y());
        cylinder_max_y = std::max(cylinder_max_y, static_x[node].y());
    }
    EXPECT_NEAR(cylinder_min_y, 1.25, 1.0e-14);
    EXPECT_NEAR(cylinder_max_y, 1.95, 1.0e-14);
    EXPECT_NEAR(0.5 * (cylinder_min_y + cylinder_max_y), 1.60, 1.0e-14);
}

TEST(ClothCylinderDropExample, DefaultSheetsAreSeparateFreeAndClearOfColliders) {
    IPCArgs3D args;
    char program[] = "3D_sim";
    char option[] = "--example";
    char scene[] = "14";
    char* argv[] = {program, option, scene};
    ASSERT_TRUE(args.parse(3, argv));
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    ASSERT_EQ(args.drop_stack_count, 50);
    const int nodes_per_sheet = (args.drop_cloth_nx + 1)
        * (args.drop_cloth_ny + 1);
    ASSERT_EQ(nodes_per_sheet, 4900);
    ASSERT_EQ(state.deformed_positions.size(), 245000U);
    EXPECT_DOUBLE_EQ(args.drop_cloth_w / args.drop_cloth_nx, 0.03);
    EXPECT_DOUBLE_EQ(args.drop_cloth_h / args.drop_cloth_ny, 0.03);
    EXPECT_DOUBLE_EQ(args.drop_spacing, 0.005);
    EXPECT_DOUBLE_EQ(params.density, 900.0);
    EXPECT_DOUBLE_EQ(params.thickness, 0.001);
    EXPECT_DOUBLE_EQ(params.friction_coefficient, 0.0);
    ASSERT_EQ(state.velocities.size(), state.deformed_positions.size());
    EXPECT_EQ(X.size(), state.deformed_positions.size());
    EXPECT_EQ(ref_mesh.deformable_nodes.size(), state.deformed_positions.size());
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(ref_mesh.rb_nodes.empty());
    EXPECT_TRUE(ref_mesh.tets.empty());
    ASSERT_EQ(params.sdf_planes.size(), 1U);
    ASSERT_EQ(params.sdf_cylinders.size(), 1U);
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_DOUBLE_EQ(params.k_sdf, args.k_sdf);
    EXPECT_GT(params.d_hat, 0.0);
    EXPECT_GT(args.drop_spacing, params.d_hat);
    for (std::size_t node = 0; node < state.deformed_positions.size(); ++node) {
        const Vec3& position = state.deformed_positions[node];
        EXPECT_TRUE(position.allFinite());
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
        EXPECT_NEAR(position.y(), args.drop_first_y
            + (node / nodes_per_sheet) * args.drop_spacing, 1.0e-14);
        EXPECT_GT(evaluate_sdf(params.sdf_cylinders[0], position).phi,
                  params.eps_sdf);
        EXPECT_GT(evaluate_sdf(params.sdf_planes[0], position).phi,
                  params.eps_sdf);
    }

    // No triangle may stitch two sheets together; their contact must be
    // handled by the mesh barrier, with all sheets retaining free boundaries.
    ASSERT_EQ(ref_mesh.tris.size(),
              50U * 6 * args.drop_cloth_nx * args.drop_cloth_ny);
    for (std::size_t t = 0; t < ref_mesh.tris.size(); t += 3) {
        const int sheet = ref_mesh.tris[t] / nodes_per_sheet;
        EXPECT_EQ(ref_mesh.tris[t + 1] / nodes_per_sheet, sheet);
        EXPECT_EQ(ref_mesh.tris[t + 2] / nodes_per_sheet, sheet);
    }
    EXPECT_NEAR(std::accumulate(ref_mesh.area.begin(), ref_mesh.area.end(), 0.0),
                50.0 * args.drop_cloth_w * args.drop_cloth_h, 1.0e-8);

    const int ground_n = static_cast<int>(std::ceil(args.cyl_ground_size / args.cyl_ground_cell_size));
    const std::size_t ground_vertices = (ground_n + 1) * (ground_n + 1);
    const std::size_t ground_indices = 6 * ground_n * ground_n;
    ASSERT_GT(static_x.size(), ground_vertices);
    ASSERT_GT(static_tris.size(), ground_indices);
    for (std::size_t t = 0; t < ground_indices; t += 3) {
        const Vec3& a = static_x[static_tris[t]];
        const Vec3& b = static_x[static_tris[t + 1]];
        const Vec3& c = static_x[static_tris[t + 2]];
        EXPECT_GT((b - a).cross(c - a).y(), 0.0);
    }
    for (const int index : static_tris) {
        EXPECT_GE(index, 0);
        EXPECT_LT(static_cast<std::size_t>(index), static_x.size());
    }
    // The wall sits inside the padded SDF; end-cap rings fill each disk.
    const Vec3 center(args.cyl_cx, args.cyl_cy, args.cyl_cz);
    for (std::size_t node = ground_vertices; node < static_x.size(); ++node) {
        const Vec3 relative = static_x[node] - center;
        const double radial = relative.head<2>().norm();
        if (std::abs(std::abs(relative.z()) - .5 * args.cyl_length) < 1e-12)
            EXPECT_LE(radial, args.cyl_radius + 1e-12);
        else EXPECT_NEAR(radial, args.cyl_radius, 1e-12);
    }

    // Keeping vertices outside an unpadded cylinder does not keep a coarse
    // chord outside. Even a rest-length diagonal should clear the visual
    // cylinder when its endpoints sit on the padded contact surface.
    const double half_diagonal = 0.5 * std::hypot(
        args.drop_cloth_w / args.drop_cloth_nx,
        args.drop_cloth_h / args.drop_cloth_ny);
    const double contact_radius = params.sdf_cylinders[0].radius;
    ASSERT_GT(contact_radius, half_diagonal);
    const double chord_radius = std::sqrt(
        contact_radius * contact_radius - half_diagonal * half_diagonal);
    EXPECT_GT(chord_radius, args.cyl_radius);
    EXPECT_TRUE(params.sdf_planes[0].point.isZero(0.0));
    EXPECT_LT(evaluate_sdf(params.sdf_planes[0], Vec3(0.0, -0.001, 0.0)).phi, 0.0);
}

TEST(ClothCylinderDropExample, RebuildUsesIndependentClothAndCylinderLocations) {
    IPCArgs3D args;
    args.d_hat = 0.0048;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;
    build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);
    pins.push_back(Pin{0, Vec3::Zero()});
    params.sdf_spheres.push_back(SphereSDF{Vec3::Zero(), 1.0});

    args.drop_stack_count = 3;
    args.drop_cloth_nx = 4;
    args.drop_cloth_ny = 6;
    args.drop_cloth_w = 1.5;
    args.drop_cloth_h = 2.5;
    args.drop_cx = 0.75;
    args.drop_cz = 0.25;
    args.drop_first_y = 2.0;
    args.drop_spacing = 0.2;
    args.cyl_cx = 3.0;
    args.cyl_cy = 1.0;
    args.cyl_cz = -2.0;
    args.cyl_radius = 0.5;
    args.cyl_sdf_padding = 0.02;
    args.k_sdf = 2e8;
    args.cyl_length = 4.0;
    args.cyl_nu = 16;
    build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    ASSERT_EQ(state.deformed_positions.size(), 105U);
    EXPECT_EQ(ref_mesh.tris.size() / 3, 144U);
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    ASSERT_EQ(params.sdf_planes.size(), 1U);
    ASSERT_EQ(params.sdf_cylinders.size(), 1U);
    EXPECT_TRUE(params.sdf_cylinders[0].point.isApprox(Vec3(3.0, 1.0, -2.0)));
    EXPECT_TRUE(params.sdf_cylinders[0].axis.isApprox(Vec3::UnitZ()));
    EXPECT_DOUBLE_EQ(params.sdf_cylinders[0].radius, 0.52);
    EXPECT_DOUBLE_EQ(params.k_sdf, 2e8);
    for (int sheet = 0; sheet < 3; ++sheet) {
        Vec3 center = Vec3::Zero();
        for (int node = 0; node < 35; ++node)
            center += state.deformed_positions[sheet * 35 + node];
        center /= 35.0;
        EXPECT_TRUE(center.isApprox(Vec3(0.75, 2.0 + sheet * 0.2, 0.25)));
    }
    const int ground_n = static_cast<int>(std::ceil(args.cyl_ground_size / args.cyl_ground_cell_size));
    const std::size_t ground_vertices = (ground_n + 1) * (ground_n + 1);
    Vec3 lo = static_x[ground_vertices];
    Vec3 hi = lo;
    for (std::size_t node = ground_vertices; node < static_x.size(); ++node) {
        lo = lo.cwiseMin(static_x[node]);
        hi = hi.cwiseMax(static_x[node]);
    }
    EXPECT_TRUE(lo.isApprox(Vec3(2.5, 0.5, -4.0)));
    EXPECT_TRUE(hi.isApprox(Vec3(3.5, 1.5, 0.0)));

    args.cyl_ground_cell_size = 0.0;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris), std::invalid_argument);
    args.cyl_ground_cell_size = 0.25;
    args.cyl_cap_rings = 0;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris), std::invalid_argument);
    args.cyl_cap_rings = 12;

    // Reject coincident sheets and an initially intersecting cylinder.
    args.drop_spacing = 0.0;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris),
        std::invalid_argument);
    args.drop_spacing = 0.2;
    params.d_hat = 0.21;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris),
        std::invalid_argument);
    params.d_hat = args.drop_spacing;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris),
        std::invalid_argument);
    params.d_hat = args.d_hat;
    args.drop_first_y = 1.25;
    EXPECT_THROW(build_cloth_cylinder_drop_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris),
        std::invalid_argument);
}

TEST(WreckingBallExample,
     BuildsTranslatedFigureEightCompositionAndTightlyPackedWall) {
    IPCArgs3D args;
    args.d_hat = 0.01; // deliberately large, to exercise the mesh-edge clamp
    args.eps_sdf = 0.002;
    args.k_sdf = 123456.0;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    std::vector<Vec3> static_x;
    std::vector<int> static_tris;

    build_wrecking_ball_example(
        args, ref_mesh, state, X, pins, params, static_x, static_tris);

    constexpr int ordinary_links = 13;
    constexpr int chain_bodies = 14;
    constexpr int cubes = 8 * 7 * 10;
    constexpr int rigid_bodies = chain_bodies + cubes;
    constexpr int link_vertices = 220;
    constexpr int link_triangles = 440;
    constexpr int ball_vertices = 312;
    constexpr int ball_triangles = 624;
    constexpr int cube_vertices = 8;
    constexpr int cube_triangles = 12;
    constexpr double cube_edge = 1.0;
    constexpr double cube_density = 1000.0;
    constexpr double cube_gap = 0.002;
    constexpr double cube_spacing = cube_edge + cube_gap;
    constexpr double cube_ground_clearance = 0.002;
    constexpr int total_vertices =
        ordinary_links * link_vertices + ball_vertices
        + cubes * cube_vertices;
    constexpr int total_triangles =
        ordinary_links * link_triangles + ball_triangles
        + cubes * cube_triangles;

    EXPECT_EQ(state.deformed_positions.size(), total_vertices);
    EXPECT_EQ(state.velocities.size(), total_vertices);
    EXPECT_EQ(ref_mesh.num_positions, total_vertices);
    EXPECT_EQ(ref_mesh.tris.size(), 3 * total_triangles);
    EXPECT_EQ(ref_mesh.mass.size(), total_vertices);
    EXPECT_EQ(ref_mesh.node_to_rb.size(), total_vertices);
    EXPECT_EQ(ref_mesh.total_mass.size(), rigid_bodies);
    EXPECT_EQ(ref_mesh.I_hat.size(), rigid_bodies);
    EXPECT_EQ(ref_mesh.ref_positions.size(), rigid_bodies);
    EXPECT_EQ(ref_mesh.rb_nodes.size(), rigid_bodies);
    EXPECT_EQ(ref_mesh.rb_update_modes.size(), rigid_bodies);
    EXPECT_EQ(state.x_coms.size(), rigid_bodies);
    EXPECT_EQ(state.v_coms.size(), rigid_bodies);
    EXPECT_EQ(state.orientations.size(), rigid_bodies);
    EXPECT_EQ(state.omega.size(), rigid_bodies);
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(ref_mesh.Dm_inverse.empty());
    EXPECT_TRUE(ref_mesh.area.empty());
    EXPECT_TRUE(ref_mesh.hinges.empty());
    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.tet_nodes.empty());
    EXPECT_TRUE(ref_mesh.surface_nodes.empty());
    EXPECT_TRUE(ref_mesh.deformable_nodes.empty());

    ASSERT_EQ(ref_mesh.rb_nodes[0].size(), link_vertices);
    EXPECT_EQ(
        ref_mesh.rb_update_modes[0], RigidBodyUpdateMode::None);
    for (int body = 1; body < rigid_bodies; ++body) {
        SCOPED_TRACE(body);
        EXPECT_EQ(
            ref_mesh.rb_update_modes[body],
            RigidBodyUpdateMode::TranslationAndOrientation);
    }
    for (int link = 0; link < ordinary_links; ++link)
        EXPECT_EQ(ref_mesh.rb_nodes[link].size(), link_vertices);
    EXPECT_EQ(ref_mesh.rb_nodes[ordinary_links].size(), ball_vertices);
    for (int cube = 0; cube < cubes; ++cube) {
        EXPECT_EQ(
            ref_mesh.rb_nodes[chain_bodies + cube].size(),
            cube_vertices);
    }

    constexpr double kPi = 3.14159265358979323846;
    constexpr double scene_y_translation = 1.0;
    constexpr double fixture_y_offset = 12.0 + scene_y_translation;
    constexpr double ground_y = -1.0 + scene_y_translation;
    const Vec4 even_orientation(
        std::cos(kPi / 6.0), 0.0, 0.0, -std::sin(kPi / 6.0));
    const Vec4 odd_orientation = quaternion_normalize(
        quaternion_multiply(
            even_orientation,
            Vec4(std::cos(kPi / 4.0), 0.0,
                 std::sin(kPi / 4.0), 0.0)));
    for (int link = 0; link < chain_bodies; ++link) {
        SCOPED_TRACE(link);
        const Vec4& expected = link % 2 == 0
            ? even_orientation : odd_orientation;
        EXPECT_TRUE(state.orientations[link].isApprox(expected, 1.0e-14));
        EXPECT_TRUE(state.v_coms[link].isZero(0.0));
        EXPECT_TRUE(state.omega[link].isZero(0.0));
    }

    const Vec3 fixed_link_fixture_position(
        14.0 * 0.9 * std::cos(kPi / 6.0),
        fixture_y_offset + 14.0 * 0.9 * std::sin(kPi / 6.0),
        0.0);
    const Vec3 link_source_volume_center(
        -3.943106929974605e-08,
         4.6749638500182615e-17,
         3.0341234516000693e-07);
    const Vec3 expected_first_link_vertex = fixed_link_fixture_position
        + quaternion_rotate(
            even_orientation, Vec3(-0.5, 0.25, 0.0));
    EXPECT_TRUE(state.deformed_positions.front().isApprox(
        expected_first_link_vertex, 1.0e-13));
    for (int link = 0; link < ordinary_links; ++link) {
        const double chain_coordinate =
            static_cast<double>(chain_bodies - link) * 0.9;
        const Vec3 fixture_position(
            chain_coordinate * std::cos(kPi / 6.0),
            fixture_y_offset
                + chain_coordinate * std::sin(kPi / 6.0), 0.0);
        const Vec4& orientation = link % 2 == 0
            ? even_orientation : odd_orientation;
        const Vec3 expected_volume_center = fixture_position
            + quaternion_rotate(orientation, link_source_volume_center);
        SCOPED_TRACE(link);
        EXPECT_TRUE(state.x_coms[link].isApprox(
            expected_volume_center, 1.0e-11));
    }

    // The asymmetric ball OBJ is authored around a model origin at the top
    // of the sphere/link assembly. Checking the first raw vertices after the
    // fixture transforms catches accidental AABB recentering by the importer.
    const double ball_chain_coordinate = 0.9;
    const Vec3 ball_fixture_position(
        ball_chain_coordinate * std::cos(kPi / 6.0),
        fixture_y_offset
            + ball_chain_coordinate * std::sin(kPi / 6.0),
        0.0);
    const Vec3 expected_first_ball_vertex = ball_fixture_position
        + quaternion_rotate(odd_orientation, Vec3(0.0, -4.0, 0.0));
    const Vec3 ball_source_volume_center(
        -6.2641546529628083e-06,
        -1.9960208409024394,
         4.4390931976959495e-08);
    const int first_ball_vertex = ordinary_links * link_vertices;
    EXPECT_TRUE(state.deformed_positions[first_ball_vertex].isApprox(
        expected_first_ball_vertex, 1.0e-13));
    EXPECT_TRUE(state.x_coms[ordinary_links].isApprox(
        ball_fixture_position
            + quaternion_rotate(odd_orientation, ball_source_volume_center),
        1.0e-10));

    constexpr double expected_link_mass = 783.6929009084646;
    constexpr double expected_ball_mass = 249066.25120239827;
    for (int link = 0; link < ordinary_links; ++link) {
        EXPECT_NEAR(
            ref_mesh.total_mass[link], expected_link_mass, 1.0e-8);
    }
    EXPECT_NEAR(
        ref_mesh.total_mass[ordinary_links], expected_ball_mass, 1.0e-7);

    const double fixed_link_x = 14.0 * 0.9 * std::cos(kPi / 6.0);
    const Vec3 first_cube_center(
        fixed_link_x + 0.5 - 4.0,
        ground_y + 0.5 * cube_edge + cube_ground_clearance,
        0.5 - 5.0);
    for (int width = 0; width < 8; ++width) {
        for (int height = 0; height < 7; ++height) {
            for (int depth = 0; depth < 10; ++depth) {
                const int cube = (width * 7 + height) * 10 + depth;
                const int body = chain_bodies + cube;
                const Vec3 expected_center = first_cube_center
                    + cube_spacing * Vec3(
                        static_cast<double>(width),
                        static_cast<double>(height),
                        static_cast<double>(depth));
                SCOPED_TRACE(cube);
                EXPECT_TRUE(state.x_coms[body].isApprox(
                    expected_center, 1.0e-14));
                EXPECT_DOUBLE_EQ(ref_mesh.total_mass[body], cube_density);
                EXPECT_TRUE(ref_mesh.I_hat[body].isApprox(
                    (cube_density / 12.0) * Mat33::Identity(), 1.0e-11));
                EXPECT_TRUE(state.v_coms[body].isZero(0.0));
                EXPECT_TRUE(state.omega[body].isZero(0.0));
            }
        }
    }

    for (int body = 0; body < rigid_bodies; ++body) {
        for (const int node : ref_mesh.rb_nodes[body]) {
            ASSERT_GE(node, 0);
            ASSERT_LT(node, total_vertices);
            EXPECT_EQ(ref_mesh.node_to_rb[node], body);
            EXPECT_TRUE(state.velocities[node].isZero(0.0));
        }
        EXPECT_GT(ref_mesh.total_mass[body], 0.0);
        EXPECT_TRUE(ref_mesh.I_hat[body].allFinite());
    }

    EXPECT_TRUE(params.gravity.isApprox(
        Vec3(args.gx, args.gy, args.gz), 0.0));
    EXPECT_DOUBLE_EQ(params.k_sdf, args.k_sdf);
    EXPECT_DOUBLE_EQ(params.eps_sdf, args.eps_sdf);
    ASSERT_EQ(params.sdf_planes.size(), 1u);
    EXPECT_TRUE(params.sdf_planes[0].point.isApprox(
        Vec3(fixed_link_x, ground_y, 0.0), 1.0e-14));
    EXPECT_DOUBLE_EQ(params.sdf_planes[0].point.y(), 0.0);
    EXPECT_TRUE(params.sdf_planes[0].normal.isApprox(
        Vec3::UnitY(), 0.0));
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);

    ASSERT_EQ(static_x.size(), 4u);
    EXPECT_EQ(static_tris, (std::vector<int>{0, 1, 2, 0, 2, 3}));
    Vec3 visual_lower = static_x.front();
    Vec3 visual_upper = static_x.front();
    for (const Vec3& vertex : static_x) {
        visual_lower = visual_lower.cwiseMin(vertex);
        visual_upper = visual_upper.cwiseMax(vertex);
    }
    EXPECT_TRUE(visual_lower.isApprox(
        Vec3(fixed_link_x - 10.0, ground_y, -10.0), 1.0e-14));
    EXPECT_TRUE(visual_upper.isApprox(
        Vec3(fixed_link_x + 10.0, ground_y, 10.0), 1.0e-14));

    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (int triangle = 0; triangle < total_triangles; ++triangle) {
        for (int local = 0; local < 3; ++local) {
            const int first = ref_mesh.tris[3 * triangle + local];
            const int second = ref_mesh.tris[
                3 * triangle + (local + 1) % 3];
            minimum_surface_edge = std::min(
                minimum_surface_edge,
                (state.deformed_positions[second]
                 - state.deformed_positions[first]).norm());
        }
    }
    EXPECT_NEAR(
        params.d_hat,
        std::min(0.45 * minimum_surface_edge, 0.5 * cube_gap),
        1.0e-15);
    EXPECT_LT(params.d_hat, args.d_hat);

    // Every body starts outside the SDF's active penalty band. The first cube
    // layer has the same tight 2 mm clearance as adjacent cubes.
    double minimum_ground_distance = std::numeric_limits<double>::infinity();
    double minimum_world_y = std::numeric_limits<double>::infinity();
    for (const Vec3& position : state.deformed_positions) {
        minimum_ground_distance = std::min(
            minimum_ground_distance, position.y() - ground_y);
        minimum_world_y = std::min(minimum_world_y, position.y());
    }
    EXPECT_NEAR(
        minimum_ground_distance, cube_ground_clearance, 1.0e-14);
    EXPECT_GT(minimum_world_y, 0.0);
    EXPECT_NEAR(
        minimum_ground_distance, params.eps_sdf, 1.0e-14);
}

TEST(ChainNetBallExample, SquareAndRectangularNetsHaveClosedOwnedRigidMeshes) {
    for (const bool coarse_links : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "coarse_links=" << coarse_links);
        for (const auto dimensions : {std::array<int, 2>{3, 3},
                                      std::array<int, 2>{4, 5}}) {
            SCOPED_TRACE(::testing::Message()
                         << dimensions[0] << "x" << dimensions[1]);
            IPCArgs3D args;
            args.chain_net_coarse_links = coarse_links;
            args.chain_net_mixed_objects = false;
            args.chain_net_rows = dimensions[0];
            args.chain_net_cols = dimensions[1];
            args.rigid_density = 7680.0;
            if (dimensions[0] == 4) {
                args.chain_net_link_scale = 1.2;
                args.chain_net_ball_radius = 1.15;
                args.chain_net_ball_subdiv = 1;
                args.chain_net_ball_gap = 0.75;
            }
            RefMesh mesh;
            DeformedState state;
            std::vector<Vec2> X;
            std::vector<Pin> pins;
            SimParams params = args.to_sim_params();
            build_chain_net_ball_example(args, mesh, state, X, pins, params);

            const int rows = args.chain_net_rows, cols = args.chain_net_cols;
            const int base_links = rows * cols - 4;
            const int horizontal_links = (rows - 2) * (cols - 1);
            const int vertical_links = (cols - 2) * (rows - 1);
            const int links = base_links + horizontal_links + vertical_links;
            const int bodies = links + 1;
            const int sphere_factor = 1 << (2 * args.chain_net_ball_subdiv);
            const int sphere_vertices = 10 * sphere_factor + 2;
            const int sphere_triangles = 20 * sphere_factor;
            const int link_vertices = coarse_links ? 128 : 220;
            const int link_triangles = coarse_links ? 256 : 440;
            const int nodes = links * link_vertices + sphere_vertices;
            const int triangles = links * link_triangles + sphere_triangles;
            ASSERT_EQ(mesh.rb_nodes.size(), bodies);
            ASSERT_EQ(mesh.total_mass.size(), bodies);
            ASSERT_EQ(mesh.I_hat.size(), bodies);
            ASSERT_EQ(mesh.rb_update_modes.size(), bodies);
            ASSERT_EQ(mesh.ref_positions.size(), bodies);
            ASSERT_EQ(state.x_coms.size(), bodies);
            ASSERT_EQ(state.v_coms.size(), bodies);
            ASSERT_EQ(state.orientations.size(), bodies);
            ASSERT_EQ(state.omega.size(), bodies);
            ASSERT_EQ(state.deformed_positions.size(), nodes);
            ASSERT_EQ(state.velocities.size(), nodes);
            ASSERT_EQ(mesh.mass.size(), nodes);
            ASSERT_EQ(mesh.node_to_rb.size(), nodes);
            EXPECT_EQ(mesh.num_positions, nodes);
            ASSERT_EQ(mesh.tris.size(), 3 * triangles);
            EXPECT_TRUE(X.empty());
            EXPECT_TRUE(pins.empty());
            EXPECT_TRUE(mesh.Dm_inverse.empty());
            EXPECT_TRUE(mesh.area.empty());
            EXPECT_TRUE(mesh.hinges.empty());
            EXPECT_TRUE(mesh.hinge_adj.empty());
            EXPECT_TRUE(mesh.tets.empty());
            EXPECT_TRUE(mesh.tet_nodes.empty());
            EXPECT_TRUE(mesh.surface_nodes.empty());
            EXPECT_TRUE(mesh.deformable_nodes.empty());
            EXPECT_TRUE(params.sdf_planes.empty());
            EXPECT_TRUE(params.sdf_cylinders.empty());
            EXPECT_TRUE(params.sdf_spheres.empty());
            EXPECT_TRUE(params.gravity.isApprox(Vec3(args.gx, args.gy, args.gz), 0.0));

            std::vector<int> ownership(nodes, 0), body_triangles(bodies, 0);
            std::vector<std::map<std::pair<int, int>, std::pair<int, int>>>
                body_edges(bodies);
            for (int body = 0; body < bodies; ++body) {
                SCOPED_TRACE(body);
                EXPECT_EQ(mesh.rb_nodes[body].size(), body == links ? sphere_vertices : link_vertices);
                ASSERT_EQ(mesh.ref_positions[body].size(), mesh.rb_nodes[body].size());
                EXPECT_TRUE(state.x_coms[body].allFinite());
                EXPECT_TRUE(state.v_coms[body].isZero(0.0));
                EXPECT_TRUE(state.omega[body].isZero(0.0));
                EXPECT_NEAR(state.orientations[body].norm(), 1.0, 1.0e-14);
                double proxy_mass = 0.0;
                for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
                    const int node = mesh.rb_nodes[body][local];
                    ASSERT_GE(node, 0);
                    ASSERT_LT(node, nodes);
                    ++ownership[node];
                    EXPECT_EQ(mesh.node_to_rb[node], body);
                    EXPECT_TRUE(state.deformed_positions[node].allFinite());
                    EXPECT_TRUE(state.velocities[node].isZero(0.0));
                    EXPECT_GT(mesh.mass[node], 0.0);
                    proxy_mass += mesh.mass[node];
                    const Vec3 reconstructed = state.x_coms[body]
                        + quaternion_rotate(state.orientations[body],
                                            mesh.ref_positions[body][local]);
                    EXPECT_TRUE(reconstructed.isApprox(state.deformed_positions[node], 1.0e-12));
                }
                EXPECT_NEAR(proxy_mass, mesh.total_mass[body],
                            1.0e-11 * mesh.total_mass[body]);
            }
            for (const int count : ownership) EXPECT_EQ(count, 1);
            std::set<std::array<int, 3>> faces;
            for (std::size_t t = 0; t < mesh.tris.size(); t += 3) {
                std::array<int, 3> face{mesh.tris[t], mesh.tris[t + 1], mesh.tris[t + 2]};
                for (const int node : face) {
                    ASSERT_GE(node, 0);
                    ASSERT_LT(node, nodes);
                }
                const int body = mesh.node_to_rb[face[0]];
                ASSERT_GE(body, 0);
                ASSERT_LT(body, bodies);
                EXPECT_EQ(mesh.node_to_rb[face[1]], body);
                EXPECT_EQ(mesh.node_to_rb[face[2]], body);
                ++body_triangles[body];
                const Vec3 a = state.deformed_positions[face[0]];
                const Vec3 b = state.deformed_positions[face[1]];
                const Vec3 c = state.deformed_positions[face[2]];
                EXPECT_GT((b - a).cross(c - a).squaredNorm(), 0.0);
                for (int side = 0; side < 3; ++side) {
                    const int first = face[side], second = face[(side + 1) % 3];
                    auto& edge = body_edges[body][std::minmax(first, second)];
                    ++edge.first;
                    edge.second += first < second ? 1 : -1;
                }
                std::sort(face.begin(), face.end());
                EXPECT_TRUE(faces.insert(face).second);
            }
            for (int body = 0; body < bodies; ++body) {
                EXPECT_EQ(body_triangles[body], body == links ? sphere_triangles : link_triangles);
                for (const auto& edge : body_edges[body]) {
                    EXPECT_EQ(edge.second.first, 2);
                    EXPECT_EQ(edge.second.second, 0);
                }
                // Ring surfaces have genus one; the ball is a closed sphere.
                EXPECT_EQ(static_cast<int>(mesh.rb_nodes[body].size())
                              - static_cast<int>(body_edges[body].size())
                              + body_triangles[body],
                          body == links ? 2 : 0);
            }

            // Fixed bodies must be precisely the outer base links, not connectors.
            const double column_spacing = 1.65 * args.chain_net_link_scale;
            const double row_spacing = 1.6 * args.chain_net_link_scale;
            const double half_width = 0.5 * (cols - 1) * column_spacing;
            const double half_depth = 0.5 * (rows - 1) * row_spacing;
            std::set<std::pair<int, int>> fixed_locations;
            for (int body = 0; body < links; ++body) {
                const Vec3& center = state.x_coms[body];
                const bool on_boundary = std::abs(std::abs(center.x()) - half_width) < 1.0e-5
                    || std::abs(std::abs(center.z()) - half_depth) < 1.0e-5;
                EXPECT_EQ(mesh.rb_update_modes[body], on_boundary
                    ? RigidBodyUpdateMode::None
                    : RigidBodyUpdateMode::TranslationAndOrientation);
                if (on_boundary) {
                    const int col = static_cast<int>(std::lround((center.x() + half_width) / column_spacing));
                    const int row = static_cast<int>(std::lround((center.z() + half_depth) / row_spacing));
                    EXPECT_TRUE(fixed_locations.insert({row, col}).second);
                }
            }
            EXPECT_EQ(fixed_locations.size(), 2 * rows + 2 * cols - 8);
            for (int row = 0; row < rows; ++row) for (int col = 0; col < cols; ++col) {
                const bool boundary_row = row == 0 || row == rows - 1;
                const bool boundary_col = col == 0 || col == cols - 1;
                EXPECT_EQ(fixed_locations.count({row, col}),
                          static_cast<std::size_t>(boundary_row != boundary_col));
            }
            EXPECT_EQ(mesh.rb_update_modes[links], RigidBodyUpdateMode::TranslationAndOrientation);
            const double radius = args.chain_net_ball_radius == 0.0
                ? 0.5 * std::min(half_width, half_depth) : args.chain_net_ball_radius;
            double net_top = -std::numeric_limits<double>::infinity();
            double ball_bottom = std::numeric_limits<double>::infinity();
            for (int node = 0; node < nodes; ++node) {
                const Vec3& position = state.deformed_positions[node];
                if (mesh.node_to_rb[node] == links) {
                    ball_bottom = std::min(ball_bottom, position.y());
                    EXPECT_NEAR((position - state.x_coms[links]).norm(), radius, 1.0e-12);
                } else {
                    net_top = std::max(net_top, position.y());
                }
            }
            EXPECT_NEAR(state.x_coms[links].x(), 0.0, 1.0e-12);
            EXPECT_NEAR(state.x_coms[links].z(), 0.0, 1.0e-12);
            EXPECT_NEAR(ball_bottom - net_top, args.chain_net_ball_gap, 1.0e-12);
            EXPECT_GT(ball_bottom, net_top);

            // Independent degree-two tetrahedral quadrature checks closed-volume
            // mass, COM, and second moments (I_hat is not the classical inertia J).
            std::vector<double> volume(bodies, 0.0);
            std::vector<Vec3> first_moment(bodies, Vec3::Zero());
            std::vector<Mat33> second_moment(bodies, Mat33::Zero());
            const double alpha = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
            const double beta = (5.0 - std::sqrt(5.0)) / 20.0;
            for (std::size_t t = 0; t < mesh.tris.size(); t += 3) {
                const int body = mesh.node_to_rb[mesh.tris[t]];
                const Vec3 a = state.deformed_positions[mesh.tris[t]] - state.x_coms[body];
                const Vec3 b = state.deformed_positions[mesh.tris[t + 1]] - state.x_coms[body];
                const Vec3 c = state.deformed_positions[mesh.tris[t + 2]] - state.x_coms[body];
                const double tetra_volume = a.dot(b.cross(c)) / 6.0;
                volume[body] += tetra_volume;
                const std::array<Vec3, 4> corners{Vec3::Zero(), a, b, c};
                for (const Vec3& corner : corners) {
                    const Vec3 point = beta * (a + b + c) + (alpha - beta) * corner;
                    first_moment[body] += (tetra_volume / 4.0) * point;
                    second_moment[body] += (tetra_volume / 4.0) * point * point.transpose();
                }
            }
            for (int body = 0; body < bodies; ++body) {
                SCOPED_TRACE(body);
                const double density = body == links
                    ? args.chain_net_ball_density : args.rigid_density;
                EXPECT_GT(volume[body], 0.0);
                EXPECT_NEAR(mesh.total_mass[body], density * volume[body],
                            1.0e-10 * mesh.total_mass[body]);
                EXPECT_LE(first_moment[body].norm(), 1.0e-10 * volume[body]);
                Mat33 rotation;
                rotation.col(0) = quaternion_rotate(state.orientations[body], Vec3::UnitX());
                rotation.col(1) = quaternion_rotate(state.orientations[body], Vec3::UnitY());
                rotation.col(2) = quaternion_rotate(state.orientations[body], Vec3::UnitZ());
                const Mat33 world_moment = rotation * mesh.I_hat[body] * rotation.transpose();
                EXPECT_TRUE(mesh.I_hat[body].allFinite());
                EXPECT_TRUE(world_moment.isApprox(density * second_moment[body], 1.0e-10));
                EXPECT_TRUE(mesh.I_hat[body].isApprox(mesh.I_hat[body].transpose(), 1.0e-12));
                EXPECT_GT(mesh.I_hat[body](0, 0), 0.0);
                EXPECT_GT((mesh.I_hat[body].topLeftCorner<2, 2>().determinant()), 0.0);
                EXPECT_GT(mesh.I_hat[body].determinant(), 0.0);
            }
        }
    }
}

TEST(ChainNetBallExample, BothLinkMeshesStartSeparatedAndInterlocked) {
    // A 3x3 net contains one interior base link, its four connectors, and
    // four supported boundary links: the complete repeating contact motif.
    struct Triangle {
        std::array<Vec3, 3> vertices;
        Vec3 lower, upper;
    };
    const auto box_gap_squared = [](const Vec3& lower_a, const Vec3& upper_a,
                                    const Vec3& lower_b, const Vec3& upper_b) {
        return (lower_a - upper_b).cwiseMax(lower_b - upper_a)
            .cwiseMax(Vec3::Zero()).squaredNorm();
    };
    const auto edge_crosses_triangle = [](const Vec3& a, const Vec3& b,
                                          const Triangle& triangle) {
        const auto& p = triangle.vertices;
        const Vec3 normal = (p[1] - p[0]).cross(p[2] - p[0]);
        const double start = normal.dot(a - p[0]);
        const double end = normal.dot(b - p[0]);
        if ((start > 0.0 && end > 0.0) || (start < 0.0 && end < 0.0)
            || start == end) return false;
        const double t = start / (start - end);
        if (t < 0.0 || t > 1.0) return false;
        const Vec3 point = a + t * (b - a);
        // Coplanar contacts are caught by the NT/SS distances below;
        // this check additionally detects transverse edge-face crossings.
        return node_triangle_distance(point, p[0], p[1], p[2]).distance < 1.0e-11;
    };
    for (const bool coarse_links : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "coarse_links=" << coarse_links);
        IPCArgs3D args;
        args.chain_net_rows = args.chain_net_cols = 3;
        args.chain_net_coarse_links = coarse_links;
        args.chain_net_mixed_objects = false;
        args.chain_net_ball_subdiv = 0;
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params = args.to_sim_params();
        build_chain_net_ball_example(args, mesh, state, X, pins, params);
        constexpr int links = 9;
        ASSERT_EQ(mesh.rb_nodes.size(), links + 1);
        std::array<std::vector<Triangle>, links> faces;
        for (std::size_t t = 0; t < mesh.tris.size(); t += 3) {
            const int body = mesh.node_to_rb[mesh.tris[t]];
            if (body >= links) continue;
            Triangle triangle;
            for (int corner = 0; corner < 3; ++corner)
                triangle.vertices[corner] = state.deformed_positions[mesh.tris[t + corner]];
            triangle.lower = triangle.vertices[0].cwiseMin(triangle.vertices[1])
                .cwiseMin(triangle.vertices[2]);
            triangle.upper = triangle.vertices[0].cwiseMax(triangle.vertices[1])
                .cwiseMax(triangle.vertices[2]);
            faces[body].push_back(triangle);
        }
        double minimum_gap = std::numeric_limits<double>::infinity();
        for (int first = 0; first < links; ++first)
            for (int second = first + 1; second < links; ++second)
                for (const Triangle& a : faces[first])
                    for (const Triangle& b : faces[second]) {
                        const double box_gap = box_gap_squared(a.lower, a.upper, b.lower, b.upper);
                        if (box_gap >= minimum_gap * minimum_gap) continue;
                        if (box_gap == 0.0) {
                            for (int edge = 0; edge < 3; ++edge) {
                                ASSERT_FALSE(edge_crosses_triangle(
                                    a.vertices[edge], a.vertices[(edge + 1) % 3], b));
                                ASSERT_FALSE(edge_crosses_triangle(
                                    b.vertices[edge], b.vertices[(edge + 1) % 3], a));
                            }
                        }
                        for (int vertex = 0; vertex < 3; ++vertex) {
                            minimum_gap = std::min(minimum_gap, node_triangle_distance(
                                a.vertices[vertex], b.vertices[0], b.vertices[1], b.vertices[2]).distance);
                            minimum_gap = std::min(minimum_gap, node_triangle_distance(
                                b.vertices[vertex], a.vertices[0], a.vertices[1], a.vertices[2]).distance);
                            for (int other_edge = 0; other_edge < 3; ++other_edge)
                                minimum_gap = std::min(minimum_gap, segment_segment_distance(
                                    a.vertices[vertex], a.vertices[(vertex + 1) % 3],
                                    b.vertices[other_edge], b.vertices[(other_edge + 1) % 3],
                                    1.0e-12, false).distance);
                        }
                    }
        EXPECT_TRUE(std::isfinite(minimum_gap));
        EXPECT_GT(minimum_gap, 0.001);
        RecordProperty(coarse_links ? "coarse_minimum_link_gap_m" : "original_minimum_link_gap_m",
                       std::to_string(minimum_gap));

        // Both meshes wrap the same capsule-shaped core loop. Its signed
        // crossings through another link's planar spanning disk distinguish
        // genuine interlocking from merely overlapping bounding boxes.
        std::array<Mat33, links> rotations;
        std::array<Vec3, links> centers;
        for (int body = 0; body < links; ++body) {
            rotations[body].col(0) = quaternion_rotate(state.orientations[body], Vec3::UnitX());
            rotations[body].col(1) = quaternion_rotate(state.orientations[body], Vec3::UnitY());
            rotations[body].col(2) = quaternion_rotate(state.orientations[body], Vec3::UnitZ());
            Vec3 lower = Vec3::Constant(std::numeric_limits<double>::infinity());
            Vec3 upper = -lower;
            for (const Vec3& point : mesh.ref_positions[body]) {
                lower = lower.cwiseMin(point);
                upper = upper.cwiseMax(point);
            }
            centers[body] = state.x_coms[body] + rotations[body] * (0.5 * (lower + upper));
        }
        constexpr int samples = 128;
        constexpr double pi = 3.14159265358979323846;
        std::array<Vec3, samples> core;
        for (int sample = 0; sample < samples; ++sample) {
            const double angle = 2.0 * pi * (sample + 0.173) / samples;
            core[sample] = Vec3(0.4 * std::cos(angle),
                0.4 * std::sin(angle) + (std::sin(angle) > 0.0 ? 0.25 : -0.25), 0.0);
        }
        int interlocked_pairs = 0;
        std::array<int, links> neighbors{};
        for (int first = 0; first < links; ++first)
            for (int second = first + 1; second < links; ++second) {
                std::array<Vec3, samples> loop;
                for (int sample = 0; sample < samples; ++sample)
                    loop[sample] = rotations[first].transpose()
                        * (centers[second] - centers[first]
                           + args.chain_net_link_scale * rotations[second] * core[sample])
                        / args.chain_net_link_scale;
                int signed_crossings = 0;
                for (int sample = 0; sample < samples; ++sample) {
                    const Vec3& a = loop[sample];
                    const Vec3& b = loop[(sample + 1) % samples];
                    if (!((a.z() < 0.0 && b.z() > 0.0)
                          || (a.z() > 0.0 && b.z() < 0.0))) continue;
                    const Vec3 crossing = a + (a.z() / (a.z() - b.z())) * (b - a);
                    const double cap_distance = std::max(std::abs(crossing.y()) - 0.25, 0.0);
                    if (crossing.x() * crossing.x() + cap_distance * cap_distance < 0.16)
                        signed_crossings += b.z() > a.z() ? 1 : -1;
                }
                EXPECT_LE(std::abs(signed_crossings), 1);
                if (std::abs(signed_crossings) == 1) {
                    ++interlocked_pairs;
                    ++neighbors[first];
                    ++neighbors[second];
                }
            }
        EXPECT_EQ(interlocked_pairs, 8);
        EXPECT_EQ(std::count(neighbors.begin(), neighbors.end(), 1), 4);
        EXPECT_EQ(std::count(neighbors.begin(), neighbors.end(), 2), 4);
        EXPECT_EQ(std::count(neighbors.begin(), neighbors.end(), 4), 1);
    }
}

TEST(ChainNetBallExample, RebuildClearsDeformableDataPinsAndAnalyticColliders) {
    IPCArgs3D args;
    args.chain_net_mixed_objects = false;
    args.chain_net_rows = args.chain_net_cols = 3;
    args.chain_net_ball_subdiv = 0;
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    build_square_mesh(mesh, state, X, 2, 2, 1.0, 1.0, Vec3::Zero());
    pins.push_back(Pin{0, Vec3::Zero()});
    SimParams params = args.to_sim_params();
    params.sdf_planes.push_back(PlaneSDF{Vec3::Zero(), Vec3::UnitY()});
    params.sdf_spheres.push_back(SphereSDF{Vec3::Zero(), 1.0});
    params.use_ccd_guess = params.use_colored_ccd_guess = true;
    params.use_colored_ccd_guess_level_based = true;
    params.use_verlet_guess = params.use_translation_guess = true;
    params.use_ogc = params.use_ogc_solver = true;
    build_chain_net_ball_example(args, mesh, state, X, pins, params);
    EXPECT_EQ(mesh.rb_nodes.size(), 10U);
    EXPECT_EQ(state.deformed_positions.size(), 9U * 128 + 12);
    EXPECT_EQ(mesh.tris.size(), 3U * (9 * 256 + 20));
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(mesh.Dm_inverse.empty());
    EXPECT_TRUE(mesh.area.empty());
    EXPECT_TRUE(mesh.hinges.empty());
    EXPECT_TRUE(mesh.deformable_nodes.empty());
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_colored_ccd_guess);
    EXPECT_FALSE(params.use_colored_ccd_guess_level_based);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);
}

TEST(ChainNetBallExample, RejectsInvalidParametersAndOversizedScenes) {
    const auto reject = [](IPCArgs3D& args) {
        args.chain_net_mixed_objects = false;
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params;
        EXPECT_THROW(build_chain_net_ball_example(args, mesh, state, X, pins, params),
                     std::invalid_argument);
    };
    for (int IPCArgs3D::* dimension : {&IPCArgs3D::chain_net_rows, &IPCArgs3D::chain_net_cols}) {
        for (const int value : {-1, 0, 2, std::numeric_limits<int>::max()}) {
            IPCArgs3D args;
            args.*dimension = value;
            reject(args);
        }
    }
    for (double IPCArgs3D::* positive : {&IPCArgs3D::rigid_density,
            &IPCArgs3D::chain_net_link_scale,
            &IPCArgs3D::chain_net_ball_density, &IPCArgs3D::chain_net_ball_gap}) {
        for (const double value : {-1.0, 0.0, std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
            IPCArgs3D args;
            args.*positive = value;
            reject(args);
        }
    }
    for (const double value : {-1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        IPCArgs3D args;
        args.chain_net_ball_radius = value;
        reject(args);
    }
    for (const int value : {-1, 8, std::numeric_limits<int>::max()}) {
        IPCArgs3D args;
        args.chain_net_ball_subdiv = value;
        reject(args);
    }
    for (double IPCArgs3D::* component : {&IPCArgs3D::gx, &IPCArgs3D::gy, &IPCArgs3D::gz}) {
        IPCArgs3D args;
        args.*component = std::numeric_limits<double>::quiet_NaN();
        reject(args);
    }
}

TEST(ChainNetBallExample, PresetsAreSceneSpecificAndExplicitFlagsOverrideThem) {
    IPCArgs3D args;
    char program[] = "3D_sim", example_flag[] = "--example", scene[] = "16";
    char* defaults[] = {program, example_flag, scene};
    ASSERT_TRUE(args.parse(3, defaults));
    EXPECT_EQ(args.chain_net_rows, 21);
    EXPECT_EQ(args.chain_net_cols, 21);
    EXPECT_DOUBLE_EQ(args.chain_net_link_scale, 0.75);
    EXPECT_TRUE(args.chain_net_coarse_links);
    EXPECT_TRUE(args.chain_net_mixed_objects);
    EXPECT_DOUBLE_EQ(args.chain_net_object_size, 0.0);
    EXPECT_DOUBLE_EQ(args.chain_net_object_density, 1000.0);
    EXPECT_DOUBLE_EQ(args.chain_net_ball_radius, 0.0);
    EXPECT_DOUBLE_EQ(args.chain_net_ball_gap, 3.0);
    EXPECT_DOUBLE_EQ(args.chain_net_ball_density, 1000.0);
    EXPECT_EQ(args.chain_net_ball_subdiv, 3);
    EXPECT_DOUBLE_EQ(args.fps, 25.0);
    EXPECT_EQ(args.substeps, 4);
    EXPECT_EQ(args.num_frames, 250);
    EXPECT_DOUBLE_EQ(args.rigid_density, 1000.0);
    EXPECT_DOUBLE_EQ(args.to_sim_params().rigid_density, 1000.0);
    EXPECT_DOUBLE_EQ(args.d_hat, 0.005);
    EXPECT_DOUBLE_EQ(args.k_barrier, 1.0e9);
    EXPECT_EQ(args.max_substep_iters, 100);
    EXPECT_TRUE(args.fixed_iters);
    EXPECT_TRUE(args.use_basic_experimental);
    EXPECT_TRUE(args.use_simd);

    IPCArgs3D overridden;
    std::vector<std::string> values = {"3D_sim", "--example", "16",
        "--chain_net_rows", "4", "--chain_net_cols", "5", "--chain_net_link_scale", "1.2",
        "--chain_net_coarse_links", "false",
        "--chain_net_mixed_objects", "false", "--chain_net_object_size", "0.6",
        "--chain_net_object_density", "1200",
        "--chain_net_ball_radius", "1.25", "--chain_net_ball_gap", "0.75",
        "--chain_net_ball_density", "700", "--chain_net_ball_subdiv", "1",
        "--fps", "30", "--substeps", "2", "--num_frames", "7",
        "--rigid_density", "100", "--d_hat", "0.001", "--k_barrier", "1234",
        "--max_substep_iters", "3", "--fixed_iters", "false",
        "--use_basic_experimental", "false", "--use_simd", "false"};
    std::vector<char*> argv;
    for (auto& value : values) argv.push_back(value.data());
    ASSERT_TRUE(overridden.parse(static_cast<int>(argv.size()), argv.data()));
    EXPECT_EQ(overridden.chain_net_rows, 4);
    EXPECT_EQ(overridden.chain_net_cols, 5);
    EXPECT_DOUBLE_EQ(overridden.chain_net_link_scale, 1.2);
    EXPECT_FALSE(overridden.chain_net_coarse_links);
    EXPECT_FALSE(overridden.chain_net_mixed_objects);
    EXPECT_DOUBLE_EQ(overridden.chain_net_object_size, 0.6);
    EXPECT_DOUBLE_EQ(overridden.chain_net_object_density, 1200.0);
    EXPECT_DOUBLE_EQ(overridden.chain_net_ball_radius, 1.25);
    EXPECT_DOUBLE_EQ(overridden.chain_net_ball_gap, 0.75);
    EXPECT_DOUBLE_EQ(overridden.chain_net_ball_density, 700.0);
    EXPECT_EQ(overridden.chain_net_ball_subdiv, 1);
    EXPECT_DOUBLE_EQ(overridden.fps, 30.0);
    EXPECT_EQ(overridden.substeps, 2);
    EXPECT_EQ(overridden.num_frames, 7);
    EXPECT_DOUBLE_EQ(overridden.rigid_density, 100.0);
    EXPECT_DOUBLE_EQ(overridden.d_hat, 0.001);
    EXPECT_DOUBLE_EQ(overridden.k_barrier, 1234.0);
    EXPECT_EQ(overridden.max_substep_iters, 3);
    EXPECT_FALSE(overridden.fixed_iters);
    EXPECT_FALSE(overridden.use_basic_experimental);
    EXPECT_FALSE(overridden.use_simd);

    IPCArgs3D unrelated;
    ASSERT_TRUE(unrelated.parse(1, defaults));
    EXPECT_EQ(unrelated.example, 1);
    EXPECT_DOUBLE_EQ(unrelated.rigid_density, 900.0);
    EXPECT_DOUBLE_EQ(unrelated.k_barrier, 100.0);
    EXPECT_FALSE(unrelated.fixed_iters);
}

TEST(ChainNetBallExample, MixedPayloadHasFortyEightSeparateClosedRigidBodiesAboveUnchangedNet) {
    const std::vector<Vec3> gear_source = load_multiphysics_gear_reference_positions();
    ASSERT_EQ(gear_source.size(), 1721U);
    for (const auto dimensions : {std::array<int, 2>{3, 3},
                                  std::array<int, 2>{4, 4},
                                  std::array<int, 2>{4, 6}}) {
        SCOPED_TRACE(::testing::Message()
                     << dimensions[0] << "x" << dimensions[1]);
        IPCArgs3D args;
        args.chain_net_rows = dimensions[0];
        args.chain_net_cols = dimensions[1];
        args.chain_net_mixed_objects = true;
        args.chain_net_ball_subdiv = 1;
        args.chain_net_ball_gap = 0.8;
        args.rigid_density = dimensions[1] == 6 ? 7680.0 : 1000.0;
        if (dimensions[1] == 6) {
            args.chain_net_link_scale = 1.2;
            args.chain_net_object_size = 0.4;
            args.chain_net_object_density = 1375.0;
        }
        RefMesh mesh, single_mesh;
        DeformedState state, single_state;
        std::vector<Vec2> X, single_X;
        std::vector<Pin> pins, single_pins;
        SimParams params = args.to_sim_params(), single_params = params;
        build_chain_net_ball_example(args, mesh, state, X, pins, params);
        args.chain_net_mixed_objects = false;
        build_chain_net_ball_example(args, single_mesh, single_state,
                                     single_X, single_pins, single_params);

        const int rows = args.chain_net_rows, cols = args.chain_net_cols;
        const int links = rows * cols - 4 + (rows - 2) * (cols - 1)
            + (cols - 2) * (rows - 1);
        constexpr int layers = 3, slots_per_layer = 16;
        constexpr int payload_count = layers * slots_per_layer;
        const int bodies = links + payload_count;
        constexpr int sphere_vertices = 42, sphere_triangles = 80;
        const int expected_nodes = links * 128 + 12 * (1721 + 8 + sphere_vertices)
            + 4 * (6 + 10 + 16);
        const int expected_triangles = links * 256 + 12 * (3442 + 12 + sphere_triangles)
            + 4 * (8 + 16 + 28);
        ASSERT_EQ(mesh.rb_nodes.size(), bodies);
        ASSERT_EQ(mesh.rb_update_modes.size(), bodies);
        ASSERT_EQ(mesh.total_mass.size(), bodies);
        ASSERT_EQ(mesh.I_hat.size(), bodies);
        ASSERT_EQ(mesh.ref_positions.size(), bodies);
        ASSERT_EQ(state.x_coms.size(), bodies);
        ASSERT_EQ(state.v_coms.size(), bodies);
        ASSERT_EQ(state.orientations.size(), bodies);
        ASSERT_EQ(state.omega.size(), bodies);
        ASSERT_EQ(state.deformed_positions.size(), expected_nodes);
        ASSERT_EQ(state.velocities.size(), expected_nodes);
        ASSERT_EQ(mesh.node_to_rb.size(), expected_nodes);
        ASSERT_EQ(mesh.mass.size(), expected_nodes);
        EXPECT_EQ(mesh.num_positions, expected_nodes);
        ASSERT_EQ(mesh.tris.size(), 3 * expected_triangles);
        ASSERT_GE(single_mesh.tris.size(), 3 * links * 256);
        ASSERT_GE(single_state.deformed_positions.size(), links * 128);
        EXPECT_TRUE(std::equal(mesh.tris.begin(), mesh.tris.begin() + 3 * links * 256,
                               single_mesh.tris.begin()));
        for (int node = 0; node < links * 128; ++node)
            EXPECT_TRUE((state.deformed_positions[node].array()
                         == single_state.deformed_positions[node].array()).all());
        for (int body = 0; body < links; ++body) {
            EXPECT_EQ(mesh.rb_nodes[body], single_mesh.rb_nodes[body]);
            EXPECT_EQ(mesh.rb_update_modes[body], single_mesh.rb_update_modes[body]);
            EXPECT_DOUBLE_EQ(mesh.total_mass[body], single_mesh.total_mass[body]);
            EXPECT_TRUE((mesh.I_hat[body].array() == single_mesh.I_hat[body].array()).all());
            EXPECT_TRUE((state.x_coms[body].array() == single_state.x_coms[body].array()).all());
            EXPECT_TRUE((state.orientations[body].array()
                         == single_state.orientations[body].array()).all());
        }
        EXPECT_EQ(std::count(mesh.rb_update_modes.begin(), mesh.rb_update_modes.end(),
                             RigidBodyUpdateMode::None), 2 * rows + 2 * cols - 8);
        EXPECT_TRUE(X.empty());
        EXPECT_TRUE(pins.empty());
        EXPECT_TRUE(mesh.Dm_inverse.empty());
        EXPECT_TRUE(mesh.area.empty());
        EXPECT_TRUE(mesh.hinges.empty());
        EXPECT_TRUE(mesh.hinge_adj.empty());
        EXPECT_TRUE(mesh.tets.empty());
        EXPECT_TRUE(mesh.tet_rest_data.empty());
        EXPECT_TRUE(mesh.tet_adj.empty());
        EXPECT_TRUE(mesh.tet_nodes.empty());
        EXPECT_TRUE(mesh.surface_nodes.empty());
        EXPECT_TRUE(mesh.deformable_nodes.empty());
        EXPECT_TRUE(params.sdf_planes.empty());
        EXPECT_TRUE(params.sdf_spheres.empty());
        EXPECT_TRUE(params.sdf_cylinders.empty());

        std::vector<Vec3> lower(bodies, Vec3::Constant(std::numeric_limits<double>::infinity()));
        std::vector<Vec3> upper(bodies, Vec3::Constant(-std::numeric_limits<double>::infinity()));
        std::vector<int> ownership(expected_nodes, 0), triangle_counts(bodies, 0);
        for (int body = 0; body < bodies; ++body) {
            SCOPED_TRACE(body);
            ASSERT_EQ(mesh.rb_nodes[body].size(), mesh.ref_positions[body].size());
            EXPECT_TRUE(state.x_coms[body].allFinite());
            EXPECT_TRUE(state.orientations[body].allFinite());
            EXPECT_NEAR(state.orientations[body].norm(), 1.0, 1.0e-13);
            EXPECT_TRUE(state.v_coms[body].isZero(0.0));
            EXPECT_TRUE(state.omega[body].isZero(0.0));
            double proxy_mass = 0.0;
            for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
                const int node = mesh.rb_nodes[body][local];
                ASSERT_GE(node, 0);
                ASSERT_LT(node, expected_nodes);
                EXPECT_EQ(mesh.node_to_rb[node], body);
                ++ownership[node];
                const Vec3& position = state.deformed_positions[node];
                EXPECT_TRUE(position.allFinite());
                EXPECT_TRUE(state.velocities[node].isZero(0.0));
                EXPECT_GT(mesh.mass[node], 0.0);
                proxy_mass += mesh.mass[node];
                lower[body] = lower[body].cwiseMin(position);
                upper[body] = upper[body].cwiseMax(position);
                EXPECT_TRUE((state.x_coms[body]
                    + quaternion_rotate(state.orientations[body], mesh.ref_positions[body][local]))
                    .isApprox(position, 1.0e-12));
            }
            EXPECT_NEAR(proxy_mass, mesh.total_mass[body], 1.0e-10 * mesh.total_mass[body]);
        }
        for (const int count : ownership) EXPECT_EQ(count, 1);

        std::vector<double> volume(bodies, 0.0);
        std::vector<Vec3> first_moment(bodies, Vec3::Zero());
        std::vector<Mat33> second_moment(bodies, Mat33::Zero());
        std::vector<std::map<std::pair<int, int>, std::pair<int, int>>> edges(bodies);
        std::set<std::array<int, 3>> unique_faces;
        double minimum_edge = std::numeric_limits<double>::infinity();
        const double alpha = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
        const double beta = (5.0 - std::sqrt(5.0)) / 20.0;
        for (std::size_t t = 0; t < mesh.tris.size(); t += 3) {
            std::array<int, 3> face{mesh.tris[t], mesh.tris[t + 1], mesh.tris[t + 2]};
            for (const int node : face) {
                ASSERT_GE(node, 0);
                ASSERT_LT(node, expected_nodes);
            }
            const int body = mesh.node_to_rb[face[0]];
            ASSERT_GE(body, 0);
            ASSERT_LT(body, bodies);
            EXPECT_EQ(mesh.node_to_rb[face[1]], body);
            EXPECT_EQ(mesh.node_to_rb[face[2]], body);
            ++triangle_counts[body];
            const Vec3 a = state.deformed_positions[face[0]] - state.x_coms[body];
            const Vec3 b = state.deformed_positions[face[1]] - state.x_coms[body];
            const Vec3 c = state.deformed_positions[face[2]] - state.x_coms[body];
            EXPECT_GT((b - a).cross(c - a).squaredNorm(), 0.0);
            for (int side = 0; side < 3; ++side) {
                const int first = face[side], second = face[(side + 1) % 3];
                auto& edge = edges[body][std::minmax(first, second)];
                ++edge.first;
                edge.second += first < second ? 1 : -1;
                minimum_edge = std::min(minimum_edge,
                    (state.deformed_positions[first] - state.deformed_positions[second]).norm());
            }
            std::sort(face.begin(), face.end());
            EXPECT_TRUE(unique_faces.insert(face).second);
            const double tetra_volume = a.dot(b.cross(c)) / 6.0;
            volume[body] += tetra_volume;
            for (const Vec3& corner : std::array<Vec3, 4>{Vec3::Zero(), a, b, c}) {
                const Vec3 point = beta * (a + b + c) + (alpha - beta) * corner;
                first_moment[body] += (tetra_volume / 4.0) * point;
                second_moment[body] += (tetra_volume / 4.0) * point * point.transpose();
            }
        }
        for (int body = 0; body < bodies; ++body) {
            SCOPED_TRACE(body);
            for (const auto& edge : edges[body]) {
                EXPECT_EQ(edge.second.first, 2);
                EXPECT_EQ(edge.second.second, 0);
            }
            EXPECT_GT(volume[body], 0.0);
            const double density = body < links ? args.rigid_density : args.chain_net_object_density;
            EXPECT_NEAR(mesh.total_mass[body], density * volume[body],
                        1.0e-10 * mesh.total_mass[body]);
            EXPECT_LE(first_moment[body].norm(), 1.0e-10 * volume[body]);
            EXPECT_TRUE(mesh.I_hat[body].allFinite());
            Mat33 rotation;
            rotation.col(0) = quaternion_rotate(state.orientations[body], Vec3::UnitX());
            rotation.col(1) = quaternion_rotate(state.orientations[body], Vec3::UnitY());
            rotation.col(2) = quaternion_rotate(state.orientations[body], Vec3::UnitZ());
            EXPECT_TRUE((rotation * mesh.I_hat[body] * rotation.transpose())
                        .isApprox(density * second_moment[body], 1.0e-10));
            EXPECT_TRUE(mesh.I_hat[body].isApprox(mesh.I_hat[body].transpose(), 1.0e-12));
            EXPECT_GT(mesh.I_hat[body](0, 0), 0.0);
            EXPECT_GT((mesh.I_hat[body].topLeftCorner<2, 2>().determinant()), 0.0);
            EXPECT_GT(mesh.I_hat[body].determinant(), 0.0);
        }

        double net_top = -std::numeric_limits<double>::infinity();
        for (int body = 0; body < links; ++body) net_top = std::max(net_top, upper[body].y());
        // Derive the available footprint from the actual fixed-boundary
        // meshes, independently of the builder's source-mesh sizing path.
        const double half_width = 0.5 * (cols - 1) * 1.65 * args.chain_net_link_scale;
        const double half_depth = 0.5 * (rows - 1) * 1.6 * args.chain_net_link_scale;
        const double infinity = std::numeric_limits<double>::infinity();
        Vec3 inner_lower(-infinity, 0.0, -infinity);
        Vec3 inner_upper(infinity, 0.0, infinity);
        for (int body = 0; body < links; ++body) {
            if (mesh.rb_update_modes[body] != RigidBodyUpdateMode::None) continue;
            const Vec3& center = state.x_coms[body];
            if (std::abs(center.x() + half_width) < 1.0e-5)
                inner_lower.x() = std::max(inner_lower.x(), upper[body].x());
            if (std::abs(center.x() - half_width) < 1.0e-5)
                inner_upper.x() = std::min(inner_upper.x(), lower[body].x());
            if (std::abs(center.z() + half_depth) < 1.0e-5)
                inner_lower.z() = std::max(inner_lower.z(), upper[body].z());
            if (std::abs(center.z() - half_depth) < 1.0e-5)
                inner_upper.z() = std::min(inner_upper.z(), lower[body].z());
        }
        ASSERT_TRUE(inner_lower.allFinite());
        ASSERT_TRUE(inner_upper.allFinite());
        const double target_size = std::min(2.0 * half_width, 2.0 * half_depth) / 8.0;
        const double fitting_size = 0.9 * std::min({-inner_lower.x(), inner_upper.x(),
                                                   -inner_lower.z(), inner_upper.z()}) / 3.3;
        ASSERT_GT(fitting_size, 0.0);
        const double object_size = args.chain_net_object_size == 0.0
            ? std::min(target_size, fitting_size) : args.chain_net_object_size;
        if (args.chain_net_object_size == 0.0) EXPECT_LT(object_size, target_size);
        double first_payload_y = std::numeric_limits<double>::infinity();
        std::array<int, 4> type_counts{};
        std::array<int, 3> prism_counts{};
        for (int payload = 0; payload < payload_count; ++payload) {
            SCOPED_TRACE(payload);
            const int body = links + payload;
            const int layer = payload / slots_per_layer, slot = payload % slots_per_layer;
            const int type = (slot + slot / 4 + layer) % 4;
            ++type_counts[type];
            const int sides = std::array<int, 3>{3, 5, 8}[layer];
            const int expected_vertices = std::array<int, 4>{1721, 8, sphere_vertices, 2 * sides}[type];
            const int expected_faces = std::array<int, 4>{3442, 12, sphere_triangles, 4 * sides - 4}[type];
            EXPECT_EQ(mesh.rb_nodes[body].size(), expected_vertices);
            EXPECT_EQ(triangle_counts[body], expected_faces);
            EXPECT_EQ(static_cast<int>(mesh.rb_nodes[body].size())
                          - static_cast<int>(edges[body].size()) + triangle_counts[body],
                      type == 0 ? 0 : 2);
            EXPECT_EQ(mesh.rb_update_modes[body], RigidBodyUpdateMode::TranslationAndOrientation);
            const Vec3 extent = upper[body] - lower[body];
            EXPECT_LE(extent.maxCoeff(), object_size + 1.0e-12);
            if (type != 0) {
                const double x_offset = std::array<double, 3>{0.0, 0.06, -0.05}[layer];
                const double z_offset = std::array<double, 3>{0.0, -0.05, 0.06}[layer];
                EXPECT_NEAR(state.x_coms[body].x(),
                    object_size * ((slot % 4 - 1.5) * 1.8 + x_offset), 1.0e-12);
                EXPECT_NEAR(state.x_coms[body].z(),
                    object_size * ((slot / 4 - 1.5) * 1.8 + z_offset), 1.0e-12);
            }
            if (type == 0) {
                EXPECT_LE(extent.maxCoeff(), object_size * (0.90 + 0.04 * layer) + 1.0e-12);
                expect_uniform_source_gear_geometry(gear_source, mesh.ref_positions[body],
                    state.orientations[body], object_size * (0.90 + 0.04 * layer),
                    0.60 * object_size);
            } else if (type == 1) {
                const double edge = object_size * (0.80 + 0.05 * layer);
                EXPECT_TRUE(extent.isApprox(Vec3::Constant(edge), 1.0e-12));
                const double mass = args.chain_net_object_density * edge * edge * edge;
                EXPECT_NEAR(mesh.total_mass[body], mass, 1.0e-11 * mass);
                EXPECT_TRUE(mesh.I_hat[body].isApprox(
                    (mass * edge * edge / 12.0) * Mat33::Identity(), 1.0e-11));
            } else if (type == 2) {
                const double radius = 0.5 * object_size * (0.86 - 0.03 * layer);
                for (const int node : mesh.rb_nodes[body])
                    EXPECT_NEAR((state.deformed_positions[node] - state.x_coms[body]).norm(),
                                radius, 1.0e-12);
            } else {
                ++prism_counts[layer];
                const double radius = 0.5 * object_size * (0.92 - 0.04 * layer);
                const double thickness = 1.3 * radius;
                for (const Vec3& position : mesh.ref_positions[body]) {
                    EXPECT_NEAR(position.head<2>().norm(), radius, 1.0e-12);
                    EXPECT_NEAR(std::abs(position.z()), 0.5 * thickness, 1.0e-12);
                }
                constexpr double pi = 3.14159265358979323846;
                const double mass = 0.5 * sides * radius * radius
                    * std::sin(2.0 * pi / sides) * thickness * args.chain_net_object_density;
                EXPECT_NEAR(mesh.total_mass[body], mass, 1.0e-11 * mass);
            }
            EXPECT_GE(lower[body].y() - net_top, args.chain_net_ball_gap - 1.0e-12);
            EXPECT_GT(lower[body].x(), inner_lower.x());
            EXPECT_LT(upper[body].x(), inner_upper.x());
            EXPECT_GT(lower[body].z(), inner_lower.z());
            EXPECT_LT(upper[body].z(), inner_upper.z());
            first_payload_y = std::min(first_payload_y, lower[body].y());
            for (int other = 0; other < payload; ++other) {
                bool separated = false;
                for (int axis = 0; axis < 3; ++axis)
                    separated = separated || upper[body][axis] < lower[links + other][axis] - 1.0e-12
                        || upper[links + other][axis] < lower[body][axis] - 1.0e-12;
                EXPECT_TRUE(separated) << "overlapping payload AABBs: " << payload << " and " << other;
            }
        }
        EXPECT_EQ(type_counts, (std::array<int, 4>{12, 12, 12, 12}));
        EXPECT_EQ(prism_counts, (std::array<int, 3>{4, 4, 4}));
        EXPECT_NEAR(first_payload_y - net_top, args.chain_net_ball_gap, 1.0e-12);
        for (int layer = 1; layer < layers; ++layer) {
            double previous_top = -std::numeric_limits<double>::infinity();
            double layer_bottom = std::numeric_limits<double>::infinity();
            for (int slot = 0; slot < slots_per_layer; ++slot) {
                previous_top = std::max(previous_top,
                    upper[links + slots_per_layer * (layer - 1) + slot].y());
                layer_bottom = std::min(layer_bottom,
                    lower[links + slots_per_layer * layer + slot].y());
            }
            EXPECT_GT(layer_bottom, previous_top);
            EXPECT_NEAR(layer_bottom - previous_top, 0.3 * object_size, 1.0e-12);
        }
        EXPECT_NEAR(params.d_hat, std::min(args.d_hat, 0.45 * minimum_edge), 1.0e-15);
        if (rows == 3 && cols == 3)
            EXPECT_LT(params.d_hat, args.d_hat);
    }
}

TEST(ChainNetBallExample, DefaultMixedPayloadKeepsLargeObjectsAndOriginalGearAspectRatio) {
    const std::vector<Vec3> gear_source = load_multiphysics_gear_reference_positions();
    ASSERT_EQ(gear_source.size(), 1721U);
    IPCArgs3D args;
    char program[] = "3D_sim", example_flag[] = "--example", scene[] = "16";
    char* argv[] = {program, example_flag, scene};
    ASSERT_TRUE(args.parse(3, argv));
    ASSERT_EQ(args.chain_net_rows, 21);
    ASSERT_EQ(args.chain_net_cols, 21);
    ASSERT_DOUBLE_EQ(args.chain_net_link_scale, 0.75);
    ASSERT_TRUE(args.chain_net_coarse_links);
    ASSERT_TRUE(args.chain_net_mixed_objects);
    ASSERT_EQ(args.chain_net_ball_subdiv, 3);
    ASSERT_DOUBLE_EQ(args.chain_net_object_size, 0.0);
    ASSERT_DOUBLE_EQ(args.rigid_density, 1000.0);
    ASSERT_DOUBLE_EQ(args.chain_net_object_density, 1000.0);
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();
    build_chain_net_ball_example(args, mesh, state, X, pins, params);

    constexpr int links = (21 * 21 - 4) + 2 * (21 - 2) * (21 - 1);
    constexpr int payloads = 3 * 16;
    constexpr int payload_vertices = 12 * (1721 + 8 + 642) + 4 * (6 + 10 + 16);
    constexpr int payload_triangles = 12 * (3442 + 12 + 1280) + 4 * (8 + 16 + 28);
    static_assert(links == 1197 && payload_vertices == 28580 && payload_triangles == 57016);
    ASSERT_EQ(mesh.rb_nodes.size(), links + payloads);
    ASSERT_EQ(mesh.rb_update_modes.size(), links + payloads);
    ASSERT_EQ(mesh.total_mass.size(), links + payloads);
    ASSERT_EQ(mesh.I_hat.size(), links + payloads);
    ASSERT_EQ(mesh.ref_positions.size(), links + payloads);
    ASSERT_EQ(state.x_coms.size(), links + payloads);
    ASSERT_EQ(state.deformed_positions.size(), links * 128 + payload_vertices);
    EXPECT_EQ(state.deformed_positions.size(), 181796U);
    EXPECT_EQ(mesh.tris.size() / 3, links * 256 + payload_triangles);
    EXPECT_EQ(mesh.tris.size() / 3, 363448U);
    EXPECT_EQ(std::count(mesh.rb_update_modes.begin(), mesh.rb_update_modes.end(),
                         RigidBodyUpdateMode::None), 76);
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(mesh.deformable_nodes.empty());
    const double automatic_size = std::min((args.chain_net_cols - 1) * 1.65,
                                           (args.chain_net_rows - 1) * 1.6)
        * args.chain_net_link_scale / 8.0;
    EXPECT_DOUBLE_EQ(automatic_size, 3.0);
    double net_top = -std::numeric_limits<double>::infinity();
    for (int body = 0; body < links; ++body)
        for (const int node : mesh.rb_nodes[body])
            net_top = std::max(net_top, state.deformed_positions[node].y());
    std::array<int, 4> type_counts{};
    std::array<int, 3> prism_counts{};
    double lowest_payload_y = std::numeric_limits<double>::infinity();
    double minimum_non_gear_axis_extent = std::numeric_limits<double>::infinity();
    for (int payload = 0; payload < payloads; ++payload) {
        SCOPED_TRACE(payload);
        const int body = links + payload, layer = payload / 16, slot = payload % 16;
        const int type = (slot + slot / 4 + layer) % 4;
        const int sides = std::array<int, 3>{3, 5, 8}[layer];
        const int vertices = std::array<int, 4>{1721, 8, 642, 2 * sides}[type];
        ++type_counts[type];
        if (type == 3) ++prism_counts[layer];
        EXPECT_EQ(mesh.rb_nodes[body].size(), vertices);
        EXPECT_EQ(mesh.rb_update_modes[body], RigidBodyUpdateMode::TranslationAndOrientation);
        EXPECT_TRUE(state.x_coms[body].allFinite());
        EXPECT_TRUE(std::isfinite(mesh.total_mass[body]));
        EXPECT_GT(mesh.total_mass[body], 0.0);
        EXPECT_TRUE(mesh.I_hat[body].allFinite());
        Vec3 lower = Vec3::Constant(std::numeric_limits<double>::infinity());
        Vec3 upper = -lower;
        Vec3 material_lower = lower, material_upper = upper;
        for (const Vec3& position : mesh.ref_positions[body]) {
            material_lower = material_lower.cwiseMin(position);
            material_upper = material_upper.cwiseMax(position);
        }
        for (const int node : mesh.rb_nodes[body]) {
            ASSERT_GE(node, 0);
            ASSERT_LT(static_cast<std::size_t>(node), state.deformed_positions.size());
            EXPECT_TRUE(state.deformed_positions[node].allFinite());
            lower = lower.cwiseMin(state.deformed_positions[node]);
            upper = upper.cwiseMax(state.deformed_positions[node]);
            if (type == 2)
                EXPECT_NEAR((state.deformed_positions[node] - state.x_coms[body]).norm(),
                    0.5 * automatic_size * (0.86 - 0.03 * layer), 1.0e-12);
        }
        EXPECT_LE((upper - lower).maxCoeff(), automatic_size + 1.0e-12);
        // The other objects retain their enlarged dimensions; the gears use
        // the original thinner multiphysics asset proportions. These bounds
        // are geometry regressions, not a proof of retention by a deforming net.
        if (type != 0) {
            EXPECT_GT((material_upper - material_lower).minCoeff(), 1.6);
            minimum_non_gear_axis_extent = std::min(minimum_non_gear_axis_extent,
                (material_upper - material_lower).minCoeff());
        }
        if (type == 0) {
            EXPECT_LE((upper - lower).maxCoeff(),
                automatic_size * (0.90 + 0.04 * layer) + 1.0e-12);
            expect_uniform_source_gear_geometry(gear_source, mesh.ref_positions[body],
                state.orientations[body], automatic_size * (0.90 + 0.04 * layer),
                0.60 * automatic_size);
        } else if (type == 1) {
            EXPECT_TRUE((upper - lower).isApprox(
                Vec3::Constant(automatic_size * (0.80 + 0.05 * layer)), 1.0e-12));
        } else if (type == 3) {
            EXPECT_NEAR(material_upper.z() - material_lower.z(),
                0.65 * automatic_size * (0.92 - 0.04 * layer), 1.0e-12);
        }
        EXPECT_GE(lower.y() - net_top, args.chain_net_ball_gap - 1.0e-12);
        lowest_payload_y = std::min(lowest_payload_y, lower.y());
    }
    EXPECT_EQ(type_counts, (std::array<int, 4>{12, 12, 12, 12}));
    EXPECT_EQ(prism_counts, (std::array<int, 3>{4, 4, 4}));
    EXPECT_NEAR(lowest_payload_y - net_top, args.chain_net_ball_gap, 1.0e-12);
    EXPECT_NEAR(minimum_non_gear_axis_extent, 1.638, 1.0e-12);
}

TEST(ChainNetBallExample, CoarseLinksPreserveDimensionsSupportsAndPayloadGeometry) {
    const auto asset_directory = std::filesystem::path(__FILE__).parent_path()
        / "example_obj" / "wrecking_ball";
    std::vector<Vec3> coarse_positions, original_positions;
    std::vector<int> coarse_triangles, original_triangles;
    load_obj_mesh((asset_directory / "link_128.obj").string(),
                  coarse_positions, coarse_triangles);
    load_obj_mesh((asset_directory / "link.obj").string(),
                  original_positions, original_triangles);
    ASSERT_EQ(coarse_positions.size(), 128U);
    ASSERT_EQ(coarse_triangles.size(), 3U * 256);
    ASSERT_EQ(original_positions.size(), 220U);
    ASSERT_EQ(original_triangles.size(), 3U * 440);
    const auto bounds = [](const std::vector<Vec3>& positions) {
        Vec3 lower = Vec3::Constant(std::numeric_limits<double>::infinity());
        Vec3 upper = -lower;
        for (const Vec3& position : positions) {
            EXPECT_TRUE(position.allFinite());
            lower = lower.cwiseMin(position);
            upper = upper.cwiseMax(position);
        }
        return std::array<Vec3, 2>{lower, upper};
    };
    const auto coarse_bounds = bounds(coarse_positions);
    const auto original_bounds = bounds(original_positions);
    EXPECT_NEAR(coarse_bounds[1].x() - coarse_bounds[0].x(), 1.0, 1.0e-12);
    EXPECT_NEAR(coarse_bounds[1].y() - coarse_bounds[0].y(), 1.5, 1.0e-12);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(coarse_bounds[0][axis], original_bounds[0][axis], 1.0e-6);
        EXPECT_NEAR(coarse_bounds[1][axis], original_bounds[1][axis], 1.0e-6);
    }

    IPCArgs3D args;
    char program[] = "3D_sim", example_flag[] = "--example", scene[] = "16";
    char* argv[] = {program, example_flag, scene};
    ASSERT_TRUE(args.parse(3, argv));
    ASSERT_TRUE(args.chain_net_coarse_links);
    IPCArgs3D original_args = args;
    original_args.chain_net_coarse_links = false;
    RefMesh mesh, original_mesh;
    DeformedState state, original_state;
    std::vector<Vec2> X, original_X;
    std::vector<Pin> pins, original_pins;
    SimParams params = args.to_sim_params(), original_params = original_args.to_sim_params();
    build_chain_net_ball_example(args, mesh, state, X, pins, params);
    build_chain_net_ball_example(original_args, original_mesh, original_state,
                                 original_X, original_pins, original_params);
    constexpr int links = 1197, payloads = 48;
    ASSERT_EQ(mesh.rb_nodes.size(), links + payloads);
    ASSERT_EQ(original_mesh.rb_nodes.size(), links + payloads);
    EXPECT_EQ(state.deformed_positions.size(), 181796U);
    EXPECT_EQ(original_state.deformed_positions.size(), 291920U);
    EXPECT_EQ(mesh.tris.size() / 3, 363448U);
    EXPECT_EQ(original_mesh.tris.size() / 3, 583696U);
    EXPECT_EQ(mesh.rb_update_modes, original_mesh.rb_update_modes);
    EXPECT_EQ(std::count(mesh.rb_update_modes.begin(), mesh.rb_update_modes.end(),
                         RigidBodyUpdateMode::None), 76);
    for (int body = 0; body < links; ++body) {
        SCOPED_TRACE(body);
        ASSERT_EQ(mesh.rb_nodes[body].size(), 128U);
        ASSERT_EQ(original_mesh.rb_nodes[body].size(), 220U);
        EXPECT_TRUE(state.orientations[body].isApprox(original_state.orientations[body], 1.0e-14));
        const auto local_bounds = bounds(mesh.ref_positions[body]);
        const auto original_local_bounds = bounds(original_mesh.ref_positions[body]);
        // Changing tessellation changes the computed volume/COM slightly,
        // but must not alter the link's dimensions or its lattice placement.
        EXPECT_TRUE((local_bounds[1] - local_bounds[0]).isApprox(
            original_local_bounds[1] - original_local_bounds[0], 1.0e-6));
        const Vec3 midpoint = state.x_coms[body] + quaternion_rotate(
            state.orientations[body], 0.5 * (local_bounds[0] + local_bounds[1]));
        const Vec3 original_midpoint = original_state.x_coms[body] + quaternion_rotate(
            original_state.orientations[body],
            0.5 * (original_local_bounds[0] + original_local_bounds[1]));
        EXPECT_LE((midpoint - original_midpoint).norm(), 1.0e-10);
    }
    const auto net_top = [](const RefMesh& data, const DeformedState& positions) {
        double result = -std::numeric_limits<double>::infinity();
        for (int body = 0; body < links; ++body)
            for (const int node : data.rb_nodes[body])
                result = std::max(result, positions.deformed_positions[node].y());
        return result;
    };
    // The entire payload may translate vertically by the changed surface
    // sampling's top extent, but no payload may change shape, mass or x/z.
    const Vec3 payload_shift(0.0,
        net_top(mesh, state) - net_top(original_mesh, original_state), 0.0);
    for (int body = links; body < links + payloads; ++body) {
        SCOPED_TRACE(body);
        ASSERT_EQ(mesh.rb_nodes[body].size(), original_mesh.rb_nodes[body].size());
        EXPECT_TRUE(state.orientations[body].isApprox(original_state.orientations[body], 1.0e-14));
        EXPECT_TRUE(state.x_coms[body].isApprox(original_state.x_coms[body] + payload_shift, 1.0e-12));
        EXPECT_NEAR(mesh.total_mass[body], original_mesh.total_mass[body],
                    1.0e-10 * original_mesh.total_mass[body]);
        EXPECT_TRUE(mesh.I_hat[body].isApprox(original_mesh.I_hat[body], 1.0e-10));
        for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
            EXPECT_TRUE(mesh.ref_positions[body][local].isApprox(
                original_mesh.ref_positions[body][local], 1.0e-11));
            EXPECT_TRUE(state.deformed_positions[mesh.rb_nodes[body][local]].isApprox(
                original_state.deformed_positions[original_mesh.rb_nodes[body][local]]
                    + payload_shift, 1.0e-12));
        }
    }
}

TEST(ChainNetBallExample, DenserDefaultsPreserveNetSpanAndPayloadGeometry) {
    IPCArgs3D args;
    char program[] = "3D_sim", example_flag[] = "--example", scene[] = "16";
    char* argv[] = {program, example_flag, scene};
    ASSERT_TRUE(args.parse(3, argv));
    IPCArgs3D previous_args = args;
    previous_args.chain_net_rows = previous_args.chain_net_cols = 16;
    previous_args.chain_net_link_scale = 1.0;
    RefMesh mesh, previous_mesh;
    DeformedState state, previous_state;
    std::vector<Vec2> X, previous_X;
    std::vector<Pin> pins, previous_pins;
    SimParams params = args.to_sim_params(), previous_params = previous_args.to_sim_params();
    build_chain_net_ball_example(args, mesh, state, X, pins, params);
    build_chain_net_ball_example(previous_args, previous_mesh, previous_state,
                                 previous_X, previous_pins, previous_params);

    constexpr int links = 1197, previous_links = 672, payloads = 48;
    ASSERT_EQ(mesh.rb_nodes.size(), links + payloads);
    ASSERT_EQ(previous_mesh.rb_nodes.size(), previous_links + payloads);
    EXPECT_EQ(std::count(mesh.rb_update_modes.begin(), mesh.rb_update_modes.end(),
                         RigidBodyUpdateMode::None), 76);
    EXPECT_EQ(std::count(previous_mesh.rb_update_modes.begin(), previous_mesh.rb_update_modes.end(),
                         RigidBodyUpdateMode::None), 56);
    // Link centers have the same footprint, but both lattice spacings and
    // every link's geometry are 25% smaller. This preserves the normalized
    // interlocking arrangement while making each local aperture smaller.
    const auto center_span = [](const DeformedState& data, const int count) -> Vec3 {
        Vec3 lower = Vec3::Constant(std::numeric_limits<double>::infinity());
        Vec3 upper = -lower;
        for (int body = 0; body < count; ++body) {
            lower = lower.cwiseMin(data.x_coms[body]);
            upper = upper.cwiseMax(data.x_coms[body]);
        }
        return upper - lower;
    };
    const Vec3 current_span = center_span(state, links);
    const Vec3 old_span = center_span(previous_state, previous_links);
    EXPECT_NEAR(current_span.x(), 24.75, 1.0e-12);
    EXPECT_NEAR(current_span.z(), 24.0, 1.0e-12);
    EXPECT_NEAR(current_span.x(), old_span.x(), 1.0e-12);
    EXPECT_NEAR(current_span.z(), old_span.z(), 1.0e-12);
    // Tiny authored COM offsets differ between orientations and shrink too.
    EXPECT_NEAR(current_span.y(), args.chain_net_link_scale * old_span.y(), 1.0e-12);
    // Links are appended base/horizontal/vertical within each grid cell,
    // rather than all base links followed by all connectors.
    const auto index_links = [](const int rows, const int cols) {
        std::map<std::array<int, 3>, int> indices;
        int body = 0;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const bool boundary_row = row == 0 || row == rows - 1;
                const bool boundary_col = col == 0 || col == cols - 1;
                if (boundary_row && boundary_col) continue;
                indices[{row, col, 0}] = body++;
                if (!boundary_row && col < cols - 1)
                    indices[{row, col, 1}] = body++;
                if (!boundary_col && row < rows - 1)
                    indices[{row, col, 2}] = body++;
            }
        }
        return indices;
    };
    const auto current_indices = index_links(21, 21);
    const auto previous_indices = index_links(16, 16);
    ASSERT_EQ(current_indices.size(), links);
    ASSERT_EQ(previous_indices.size(), previous_links);
    const int current_origin = current_indices.at({1, 1, 0});
    const int previous_origin = previous_indices.at({1, 1, 0});
    EXPECT_NEAR(state.x_coms[current_indices.at({1, 2, 0})].x()
                    - state.x_coms[current_origin].x(), 1.2375, 1.0e-12);
    EXPECT_NEAR(previous_state.x_coms[previous_indices.at({1, 2, 0})].x()
                    - previous_state.x_coms[previous_origin].x(), 1.65, 1.0e-12);
    EXPECT_NEAR(state.x_coms[current_indices.at({2, 1, 0})].z()
                    - state.x_coms[current_origin].z(), 1.2, 1.0e-12);
    EXPECT_NEAR(previous_state.x_coms[previous_indices.at({2, 1, 0})].z()
                    - previous_state.x_coms[previous_origin].z(), 1.6, 1.0e-12);
    constexpr double scale = 0.75;
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        const int body = current_indices.at({1, 1, kind});
        const int previous_body = previous_indices.at({1, 1, kind});
        ASSERT_EQ(mesh.ref_positions[body].size(), previous_mesh.ref_positions[previous_body].size());
        EXPECT_TRUE(state.orientations[body].isApprox(previous_state.orientations[previous_body], 1.0e-14));
        for (std::size_t local = 0; local < mesh.ref_positions[body].size(); ++local)
            EXPECT_TRUE(mesh.ref_positions[body][local].isApprox(
                scale * previous_mesh.ref_positions[previous_body][local], 1.0e-11));
        const double expected_mass = std::pow(scale, 3) * previous_mesh.total_mass[previous_body];
        EXPECT_NEAR(mesh.total_mass[body], expected_mass, 1.0e-10 * expected_mass);
        EXPECT_TRUE(mesh.I_hat[body].isApprox(
            std::pow(scale, 5) * previous_mesh.I_hat[previous_body], 1.0e-10));
    }
    // Compare a complete interior repeating cell with its two neighboring
    // base links. This checks the relative interlocking placement as well as
    // each body's own shape; changing mesh size without spacing fails here.
    for (const std::array<int, 3> key : {std::array<int, 3>{1, 1, 0},
            {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {2, 1, 0}}) {
        const int body = current_indices.at(key), previous_body = previous_indices.at(key);
        ASSERT_EQ(mesh.rb_nodes[body].size(), previous_mesh.rb_nodes[previous_body].size());
        for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
            const Vec3 relative = state.deformed_positions[mesh.rb_nodes[body][local]]
                - state.x_coms[current_origin];
            const Vec3 previous_relative = previous_state.deformed_positions[
                previous_mesh.rb_nodes[previous_body][local]] - previous_state.x_coms[previous_origin];
            EXPECT_TRUE(relative.isApprox(scale * previous_relative, 1.0e-11));
        }
    }

    const auto top_of_net = [](const RefMesh& data, const DeformedState& positions,
                               const int count) {
        double top = -std::numeric_limits<double>::infinity();
        for (int body = 0; body < count; ++body)
            for (const int node : data.rb_nodes[body])
                top = std::max(top, positions.deformed_positions[node].y());
        return top;
    };
    const Vec3 expected_payload_shift(0.0,
        top_of_net(mesh, state, links) - top_of_net(previous_mesh, previous_state, previous_links), 0.0);
    for (int payload = 0; payload < payloads; ++payload) {
        SCOPED_TRACE(payload);
        const int body = links + payload, previous_body = previous_links + payload;
        ASSERT_EQ(mesh.rb_nodes[body].size(), previous_mesh.rb_nodes[previous_body].size());
        EXPECT_EQ(mesh.rb_update_modes[body], previous_mesh.rb_update_modes[previous_body]);
        EXPECT_TRUE(state.orientations[body].isApprox(previous_state.orientations[previous_body], 1.0e-14));
        EXPECT_TRUE(state.x_coms[body].isApprox(
            previous_state.x_coms[previous_body] + expected_payload_shift, 1.0e-12));
        EXPECT_NEAR(mesh.total_mass[body], previous_mesh.total_mass[previous_body],
                    1.0e-10 * previous_mesh.total_mass[previous_body]);
        EXPECT_TRUE(mesh.I_hat[body].isApprox(previous_mesh.I_hat[previous_body], 1.0e-10));
        for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
            EXPECT_TRUE(mesh.ref_positions[body][local].isApprox(
                previous_mesh.ref_positions[previous_body][local], 1.0e-11));
            EXPECT_TRUE(state.deformed_positions[mesh.rb_nodes[body][local]].isApprox(
                previous_state.deformed_positions[previous_mesh.rb_nodes[previous_body][local]]
                    + expected_payload_shift, 1.0e-12));
        }
    }
}

TEST(ChainNetBallExample, MixedPayloadRejectsInvalidSizeAndDensity) {
    // A positive explicit size that exceeds the net footprint is rejected,
    // never silently reduced like the automatic size on a small net.
    for (const double value : {-1.0, 1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        IPCArgs3D args;
        args.chain_net_mixed_objects = true;
        args.chain_net_rows = args.chain_net_cols = 4;
        args.chain_net_object_size = value;
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params = args.to_sim_params();
        EXPECT_THROW(build_chain_net_ball_example(args, mesh, state, X, pins, params),
                     std::invalid_argument);
    }
    for (const double value : {-1.0, 0.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        IPCArgs3D args;
        args.chain_net_mixed_objects = true;
        args.chain_net_rows = args.chain_net_cols = 4;
        args.chain_net_object_density = value;
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params = args.to_sim_params();
        EXPECT_THROW(build_chain_net_ball_example(args, mesh, state, X, pins, params),
                     std::invalid_argument);
    }
}

TEST(RigidObjectsInBoxExample, ThreeHundredMixedBodiesStartSeparatedAboveOneOpenRigidBox) {
    IPCArgs3D args;
    args.rigid_density = 1000.0;
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    // Replacing a previous deformable scene must not leave stale ownership,
    // elastic elements, pins, or analytic contact geometry behind.
    build_square_mesh(mesh, state, X, 2, 2, 1.0, 1.0, Vec3::Zero());
    pins.push_back(Pin{0, Vec3::Zero()});
    SimParams params = args.to_sim_params();
    params.sdf_planes.push_back(PlaneSDF{Vec3::Zero(), Vec3::UnitY()});
    params.sdf_spheres.push_back(SphereSDF{Vec3::Zero(), 1.0});
    params.use_ccd_guess = params.use_colored_ccd_guess = true;
    params.use_colored_ccd_guess_level_based = true;
    params.use_verlet_guess = params.use_translation_guess = true;
    params.use_ogc = params.use_ogc_solver = true;
    build_rigid_objects_in_box_example(args, mesh, state, X, pins, params);

    constexpr int bodies = 301;
    ASSERT_EQ(mesh.rb_nodes.size(), bodies);
    ASSERT_EQ(mesh.rb_update_modes.size(), bodies);
    ASSERT_EQ(mesh.ref_positions.size(), bodies);
    ASSERT_EQ(mesh.total_mass.size(), bodies);
    ASSERT_EQ(mesh.I_hat.size(), bodies);
    ASSERT_EQ(state.x_coms.size(), bodies);
    ASSERT_EQ(state.v_coms.size(), bodies);
    ASSERT_EQ(state.orientations.size(), bodies);
    ASSERT_EQ(state.omega.size(), bodies);
    ASSERT_EQ(mesh.num_positions, state.deformed_positions.size());
    ASSERT_EQ(mesh.node_to_rb.size(), state.deformed_positions.size());
    ASSERT_EQ(mesh.mass.size(), state.deformed_positions.size());
    ASSERT_EQ(state.velocities.size(), state.deformed_positions.size());
    EXPECT_TRUE(X.empty());
    EXPECT_TRUE(pins.empty());
    EXPECT_TRUE(mesh.Dm_inverse.empty());
    EXPECT_TRUE(mesh.area.empty());
    EXPECT_TRUE(mesh.hinges.empty());
    EXPECT_TRUE(mesh.hinge_adj.empty());
    EXPECT_TRUE(mesh.tets.empty());
    EXPECT_TRUE(mesh.tet_nodes.empty());
    EXPECT_TRUE(mesh.surface_nodes.empty());
    EXPECT_TRUE(mesh.deformable_nodes.empty());
    EXPECT_TRUE(params.sdf_planes.empty());
    EXPECT_TRUE(params.sdf_cylinders.empty());
    EXPECT_TRUE(params.sdf_spheres.empty());
    EXPECT_FALSE(params.use_ccd_guess);
    EXPECT_FALSE(params.use_colored_ccd_guess);
    EXPECT_FALSE(params.use_colored_ccd_guess_level_based);
    EXPECT_FALSE(params.use_verlet_guess);
    EXPECT_FALSE(params.use_translation_guess);
    EXPECT_FALSE(params.use_ogc);
    EXPECT_FALSE(params.use_ogc_solver);
    EXPECT_TRUE(params.gravity.isApprox(Vec3(args.gx, args.gy, args.gz), 0.0));

    std::vector<Vec3> lower(bodies), upper(bodies);
    std::vector<int> ownership(mesh.num_positions, 0), triangle_counts(bodies, 0);
    std::vector<double> volumes(bodies, 0.0);
    std::vector<std::map<std::pair<int, int>, std::pair<int, int>>> edges(bodies);
    for (int body = 0; body < bodies; ++body) {
        SCOPED_TRACE(body);
        ASSERT_FALSE(mesh.rb_nodes[body].empty());
        ASSERT_EQ(mesh.ref_positions[body].size(), mesh.rb_nodes[body].size());
        EXPECT_EQ(mesh.rb_update_modes[body], body == 0
            ? RigidBodyUpdateMode::None : RigidBodyUpdateMode::TranslationAndOrientation);
        EXPECT_TRUE(state.x_coms[body].allFinite());
        EXPECT_TRUE(state.v_coms[body].isZero(0.0));
        EXPECT_TRUE(state.omega[body].isZero(0.0));
        EXPECT_NEAR(state.orientations[body].norm(), 1.0, 1.0e-14);
        EXPECT_TRUE(std::isfinite(mesh.total_mass[body]));
        EXPECT_GT(mesh.total_mass[body], 0.0);
        EXPECT_TRUE(mesh.I_hat[body].allFinite());
        EXPECT_TRUE(mesh.I_hat[body].isApprox(mesh.I_hat[body].transpose(), 1.0e-12));
        EXPECT_GT(mesh.I_hat[body](0, 0), 0.0);
        EXPECT_GT((mesh.I_hat[body].topLeftCorner<2, 2>().determinant()), 0.0);
        EXPECT_GT(mesh.I_hat[body].determinant(), 0.0);
        lower[body] = Vec3::Constant(std::numeric_limits<double>::infinity());
        upper[body] = -lower[body];
        double proxy_mass = 0.0;
        for (std::size_t local = 0; local < mesh.rb_nodes[body].size(); ++local) {
            const int node = mesh.rb_nodes[body][local];
            ASSERT_GE(node, 0);
            ASSERT_LT(node, mesh.num_positions);
            ++ownership[node];
            EXPECT_EQ(mesh.node_to_rb[node], body);
            const Vec3& position = state.deformed_positions[node];
            EXPECT_TRUE(position.allFinite());
            EXPECT_TRUE(state.velocities[node].isZero(0.0));
            EXPECT_GT(mesh.mass[node], 0.0);
            proxy_mass += mesh.mass[node];
            lower[body] = lower[body].cwiseMin(position);
            upper[body] = upper[body].cwiseMax(position);
            const Vec3 reconstructed = state.x_coms[body]
                + quaternion_rotate(state.orientations[body], mesh.ref_positions[body][local]);
            EXPECT_TRUE(reconstructed.isApprox(position, 1.0e-12));
        }
        EXPECT_NEAR(proxy_mass, mesh.total_mass[body], 1.0e-10 * mesh.total_mass[body]);
    }
    for (const int count : ownership) EXPECT_EQ(count, 1);

    ASSERT_EQ(mesh.tris.size() % 3, 0U);
    for (std::size_t t = 0; t < mesh.tris.size(); t += 3) {
        const std::array<int, 3> face{mesh.tris[t], mesh.tris[t + 1], mesh.tris[t + 2]};
        for (const int node : face) {
            ASSERT_GE(node, 0);
            ASSERT_LT(node, mesh.num_positions);
        }
        const int body = mesh.node_to_rb[face[0]];
        ASSERT_GE(body, 0);
        ASSERT_LT(body, bodies);
        EXPECT_EQ(mesh.node_to_rb[face[1]], body);
        EXPECT_EQ(mesh.node_to_rb[face[2]], body);
        ++triangle_counts[body];
        const Vec3 a = state.deformed_positions[face[0]];
        const Vec3 b = state.deformed_positions[face[1]];
        const Vec3 c = state.deformed_positions[face[2]];
        const Vec3 normal = (b - a).cross(c - a);
        EXPECT_GT(normal.squaredNorm(), 0.0);
        const Vec3 center = state.x_coms[body];
        volumes[body] += (a - center).dot((b - center).cross(c - center)) / 6.0;
        for (int side = 0; side < 3; ++side) {
            const int first = face[side], second = face[(side + 1) % 3];
            auto& edge = edges[body][std::minmax(first, second)];
            ++edge.first;
            edge.second += first < second ? 1 : -1;
        }
        if (body == 0) {
            const Vec3 centroid = (a + b + c) / 3.0;
            const auto on_plane = [&](int axis, double coordinate) {
                return std::abs(a[axis] - coordinate) < 1.0e-12
                    && std::abs(b[axis] - coordinate) < 1.0e-12
                    && std::abs(c[axis] - coordinate) < 1.0e-12;
            };
            const bool outer_wall = on_plane(0, -6.6) || on_plane(0, 6.6)
                || on_plane(2, -4.2) || on_plane(2, 4.2);
            const bool inner_wall = on_plane(0, -6.4) || on_plane(0, 6.4)
                || on_plane(2, -4.0) || on_plane(2, 4.0);
            const bool bottom = on_plane(1, -0.2);
            const bool floor = on_plane(1, 0.0);
            const bool rim = on_plane(1, 3.0);
            EXPECT_TRUE(outer_wall || inner_wall || bottom || floor || rim);
            if (bottom) EXPECT_LT(normal.y(), 0.0);
            if (floor) EXPECT_GT(normal.y(), 0.0);
            if (rim) {
                EXPECT_GT(normal.y(), 0.0);
                // Top faces are only wall rims, never a roof over the opening.
                EXPECT_TRUE(std::abs(centroid.x()) >= 6.4 - 1.0e-12
                            || std::abs(centroid.z()) >= 4.0 - 1.0e-12);
            }
        }
    }
    for (int body = 0; body < bodies; ++body) {
        SCOPED_TRACE(body);
        for (const auto& edge : edges[body]) {
            EXPECT_EQ(edge.second.first, 2);
            EXPECT_EQ(edge.second.second, 0);
        }
        EXPECT_GT(volumes[body], 0.0);
        EXPECT_NEAR(mesh.total_mass[body], args.rigid_density * volumes[body],
                    1.0e-10 * mesh.total_mass[body]);
    }

    ASSERT_EQ(mesh.rb_nodes[0].size(), 16U);
    EXPECT_EQ(triangle_counts[0], 28);
    EXPECT_EQ(static_cast<int>(mesh.rb_nodes[0].size())
                  - static_cast<int>(edges[0].size()) + triangle_counts[0], 2);
    EXPECT_TRUE(lower[0].isApprox(Vec3(-6.6, -0.2, -4.2), 1.0e-12));
    EXPECT_TRUE(upper[0].isApprox(Vec3(6.6, 3.0, 4.2), 1.0e-12));
    EXPECT_NEAR(volumes[0], 47.616, 1.0e-10);
    // One connected shell, not independent overlapping wall solids.
    std::set<int> reachable{mesh.rb_nodes[0].front()};
    for (std::size_t pass = 0; pass < mesh.rb_nodes[0].size(); ++pass) {
        for (const auto& edge : edges[0]) {
            if (reachable.count(edge.first.first)) reachable.insert(edge.first.second);
            if (reachable.count(edge.first.second)) reachable.insert(edge.first.first);
        }
    }
    EXPECT_EQ(reachable.size(), mesh.rb_nodes[0].size());

    std::array<int, 6> families{};
    std::set<int> polygon_sides;
    for (int body = 1; body < bodies; ++body) {
        SCOPED_TRACE(body);
        const int slot = body - 1, layer = slot / 60, row = (slot / 10) % 6, col = slot % 10;
        const int family = (col + row + layer) % 6;
        ++families[family];
        const std::size_t vertices = mesh.rb_nodes[body].size();
        switch (family) {
        case 0: EXPECT_EQ(vertices, 8U); EXPECT_EQ(triangle_counts[body], 12); break;
        case 1:
            EXPECT_EQ(vertices % 2, 0U);
            EXPECT_GE(vertices, 6U);
            EXPECT_LE(vertices, 16U);
            polygon_sides.insert(static_cast<int>(vertices / 2));
            EXPECT_EQ(triangle_counts[body], 2 * static_cast<int>(vertices) - 4);
            break;
        case 2: EXPECT_EQ(vertices, 128U); EXPECT_EQ(triangle_counts[body], 256); break;
        case 3: EXPECT_EQ(vertices, 502U); EXPECT_EQ(triangle_counts[body], 1000); break;
        case 4: EXPECT_EQ(vertices, 252U); EXPECT_EQ(triangle_counts[body], 500); break;
        case 5: EXPECT_EQ(vertices, 1721U); EXPECT_EQ(triangle_counts[body], 3442); break;
        }
        const Vec3 box_center = 0.5 * (lower[body] + upper[body]);
        const Vec3 expected_center((col - 4.5) * 1.2, 4.0 + layer * 1.2, (row - 2.5) * 1.2);
        EXPECT_TRUE(box_center.isApprox(expected_center, 1.0e-12));
        const double extent = (upper[body] - lower[body]).maxCoeff();
        const double size_multiplier = family == 2 ? 0.7 : 1.0;
        const double expected_extent = size_multiplier
            * (0.8 + 0.05 * (((slot % 60) + layer) % 5));
        EXPECT_NEAR(extent, expected_extent, 1.0e-12);
        EXPECT_GT(lower[body].x(), -6.4);
        EXPECT_LT(upper[body].x(), 6.4);
        EXPECT_GT(lower[body].z(), -4.0);
        EXPECT_LT(upper[body].z(), 4.0);
        EXPECT_GT(lower[body].y(), 3.0);
    }
    for (const int count : families) EXPECT_EQ(count, 50);
    EXPECT_EQ(polygon_sides, (std::set<int>{3, 4, 5, 6, 7, 8}));
    for (int first = 1; first < bodies; ++first) {
        for (int second = first + 1; second < bodies; ++second) {
            const bool separated = (upper[first].array() < lower[second].array()).any()
                || (upper[second].array() < lower[first].array()).any();
            EXPECT_TRUE(separated) << first << " overlaps " << second;
        }
    }
}

TEST(RigidObjectsInBoxExample, RejectsInvalidDensityAndGravity) {
    const auto reject = [](const IPCArgs3D& args) {
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> X;
        std::vector<Pin> pins;
        SimParams params;
        EXPECT_THROW(build_rigid_objects_in_box_example(args, mesh, state, X, pins, params),
                     std::invalid_argument);
    };
    for (const double value : {-1.0, 0.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        IPCArgs3D args;
        args.rigid_density = value;
        reject(args);
    }
    for (double IPCArgs3D::* component : {&IPCArgs3D::gx, &IPCArgs3D::gy, &IPCArgs3D::gz}) {
        IPCArgs3D args;
        args.*component = std::numeric_limits<double>::quiet_NaN();
        reject(args);
    }
}

TEST(RigidObjectsInBoxExample, PresetsAreSceneSpecificAndExplicitFlagsOverrideThem) {
    IPCArgs3D args;
    char program[] = "3D_sim", example_flag[] = "--example", scene[] = "17";
    char* defaults[] = {program, example_flag, scene};
    ASSERT_TRUE(args.parse(3, defaults));
    EXPECT_EQ(args.example, 17);
    EXPECT_EQ(args.num_frames, 200);
    EXPECT_DOUBLE_EQ(args.fps, 30.0);
    EXPECT_EQ(args.substeps, 20);
    EXPECT_EQ(args.max_substep_iters, 50);
    EXPECT_TRUE(args.fixed_iters);
    EXPECT_TRUE(args.use_parallel);
    EXPECT_TRUE(args.use_basic_experimental);
    EXPECT_TRUE(args.use_simd);
    EXPECT_TRUE(args.exact_computation_fallback);
    EXPECT_DOUBLE_EQ(args.rigid_density, 1000.0);
    EXPECT_DOUBLE_EQ(args.to_sim_params().rigid_density, 1000.0);
    EXPECT_DOUBLE_EQ(args.d_hat, 0.001);
    EXPECT_DOUBLE_EQ(args.k_barrier, 1.0e9);
    EXPECT_DOUBLE_EQ(args.friction_coefficient, 0.1);

    IPCArgs3D overridden;
    std::vector<std::string> values = {"3D_sim", "--example", "17",
        "--fps", "25", "--substeps", "2", "--num_frames", "7",
        "--rigid_density", "1250", "--d_hat", "0.002", "--k_barrier", "1234",
        "--friction_coefficient", "0.2", "--max_substep_iters", "3",
        "--fixed_iters", "false", "--use_parallel", "false",
        "--use_basic_experimental", "false", "--use_simd", "false",
        "--exact_computation_fallback", "false"};
    std::vector<char*> argv;
    for (auto& value : values) argv.push_back(value.data());
    ASSERT_TRUE(overridden.parse(static_cast<int>(argv.size()), argv.data()));
    EXPECT_DOUBLE_EQ(overridden.fps, 25.0);
    EXPECT_EQ(overridden.substeps, 2);
    EXPECT_EQ(overridden.num_frames, 7);
    EXPECT_DOUBLE_EQ(overridden.rigid_density, 1250.0);
    EXPECT_DOUBLE_EQ(overridden.d_hat, 0.002);
    EXPECT_DOUBLE_EQ(overridden.k_barrier, 1234.0);
    EXPECT_DOUBLE_EQ(overridden.friction_coefficient, 0.2);
    EXPECT_EQ(overridden.max_substep_iters, 3);
    EXPECT_FALSE(overridden.fixed_iters);
    EXPECT_FALSE(overridden.use_parallel);
    EXPECT_FALSE(overridden.use_basic_experimental);
    EXPECT_FALSE(overridden.use_simd);
    EXPECT_FALSE(overridden.exact_computation_fallback);

    IPCArgs3D unrelated;
    ASSERT_TRUE(unrelated.parse(1, defaults));
    EXPECT_EQ(unrelated.example, 1);
    EXPECT_DOUBLE_EQ(unrelated.rigid_density, 900.0);
    EXPECT_DOUBLE_EQ(unrelated.k_barrier, 100.0);
    EXPECT_DOUBLE_EQ(unrelated.friction_coefficient, 0.0);
    EXPECT_FALSE(unrelated.fixed_iters);
}

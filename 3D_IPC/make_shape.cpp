#include "make_shape.h"
#include "io.h"
#include "rigid_body_ipc.h"
#include "solid_ipc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

int append_rigid_polygon(
    int number_of_nodes, DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double radius, double density,
    double thickness, const Vec3& v_com,
    const Vec4& orientation, const Vec3& omega) {
    if (number_of_nodes < 3)
        throw std::invalid_argument(
            "append_rigid_polygon: number_of_nodes must be at least 3");
    if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument(
            "append_rigid_polygon: radius must be positive and finite");
    if (!std::isfinite(density) || density <= 0.0)
        throw std::invalid_argument(
            "append_rigid_polygon: density must be positive and finite");
    if (!std::isfinite(thickness) || thickness <= 0.0)
        throw std::invalid_argument(
            "append_rigid_polygon: thickness must be positive and finite");

    constexpr double kPi = 3.14159265358979323846;
    const double two_pi = 2.0 * kPi;
    const double half_thickness = 0.5 * thickness;
    const Vec4 q = quaternion_normalize(orientation);
    const int base = static_cast<int>(state.deformed_positions.size());

    std::vector<Vec3> world_positions;
    world_positions.reserve(2 * number_of_nodes);
    for (int layer = 0; layer < 2; ++layer) {
        const double z = (layer == 0) ? -half_thickness : half_thickness;
        for (int i = 0; i < number_of_nodes; ++i) {
            const double angle = two_pi * static_cast<double>(i)
                / static_cast<double>(number_of_nodes);
            const Vec3 X(
                radius * std::cos(angle),
                radius * std::sin(angle), z);
            world_positions.push_back(
                center + quaternion_rotate(q, X));
        }
    }

    ref_mesh.tris.reserve(
        ref_mesh.tris.size() + 3 * (4 * number_of_nodes - 4));
    const auto bottom = [base](int i) { return base + i; };
    const auto top = [base, number_of_nodes](int i) {
        return base + number_of_nodes + i;
    };

    // Cap fans. The bottom faces -z and the top faces +z in material space.
    for (int i = 1; i + 1 < number_of_nodes; ++i) {
        ref_mesh.tris.push_back(bottom(0));
        ref_mesh.tris.push_back(bottom(i + 1));
        ref_mesh.tris.push_back(bottom(i));

        ref_mesh.tris.push_back(top(0));
        ref_mesh.tris.push_back(top(i));
        ref_mesh.tris.push_back(top(i + 1));
    }

    // Two outward-facing triangles for every rectangular side panel.
    for (int i = 0; i < number_of_nodes; ++i) {
        const int next = (i + 1) % number_of_nodes;
        ref_mesh.tris.push_back(bottom(i));
        ref_mesh.tris.push_back(bottom(next));
        ref_mesh.tris.push_back(top(next));

        ref_mesh.tris.push_back(bottom(i));
        ref_mesh.tris.push_back(top(next));
        ref_mesh.tris.push_back(top(i));
    }

    // Volume = regular-polygon area times extrusion thickness.
    const double n = static_cast<double>(number_of_nodes);
    const double area = 0.5 * n * radius * radius
        * std::sin(two_pi / n);
    const double total_mass = area * thickness * density;
    return create_rigid_body(
        world_positions, v_com, q, omega, total_mass,
        ref_mesh, state);
}

int append_deformable_polygon_prism(
    int number_of_nodes, DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double radius, double density,
    double thickness, const Vec4& orientation) {
    if (number_of_nodes < 3) {
        throw std::invalid_argument(
            "append_deformable_polygon_prism: number_of_nodes must be at least 3");
    }
    if (number_of_nodes
        > (std::numeric_limits<int>::max() - 1) / 2) {
        throw std::overflow_error(
            "append_deformable_polygon_prism: too many polygon nodes");
    }
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::invalid_argument(
            "append_deformable_polygon_prism: radius must be positive and finite");
    }
    if (!std::isfinite(density) || density <= 0.0) {
        throw std::invalid_argument(
            "append_deformable_polygon_prism: density must be positive and finite");
    }
    if (!std::isfinite(thickness) || thickness <= 0.0) {
        throw std::invalid_argument(
            "append_deformable_polygon_prism: thickness must be positive and finite");
    }

    constexpr double kPi = 3.14159265358979323846;
    const double two_pi = 2.0 * kPi;
    const double half_thickness = 0.5 * thickness;
    const Vec4 q = quaternion_normalize(orientation);
    if (state.deformed_positions.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "append_deformable_polygon_prism: global node index exceeds int range");
    }
    const int base = static_cast<int>(state.deformed_positions.size());

    // Boundary vertices use the same ordering as append_rigid_polygon:
    // bottom ring, then top ring. The final vertex is strictly interior.
    std::vector<Vec3> world_positions;
    world_positions.reserve(
        2 * static_cast<std::size_t>(number_of_nodes) + 1);
    for (int layer = 0; layer < 2; ++layer) {
        const double z = (layer == 0) ? -half_thickness : half_thickness;
        for (int i = 0; i < number_of_nodes; ++i) {
            const double angle = two_pi * static_cast<double>(i)
                / static_cast<double>(number_of_nodes);
            const Vec3 material_position(
                radius * std::cos(angle),
                radius * std::sin(angle), z);
            world_positions.push_back(
                center + quaternion_rotate(q, material_position));
        }
    }
    world_positions.push_back(center);

    const int interior = 2 * number_of_nodes;
    const auto bottom = [](int i) { return i; };
    const auto top = [number_of_nodes](int i) {
        return number_of_nodes + i;
    };

    // Every outward boundary face (a,b,c) produces the positively oriented
    // tet (interior,a,b,c). Since the prism is convex, these cones partition
    // its volume without overlaps or gaps.
    const std::size_t tet_count =
        4 * static_cast<std::size_t>(number_of_nodes) - 4;
    std::vector<int> local_tets;
    local_tets.reserve(4 * tet_count);
    const auto append_cone_tet = [&local_tets, interior](
                                     int a, int b, int c) {
        local_tets.push_back(interior);
        local_tets.push_back(a);
        local_tets.push_back(b);
        local_tets.push_back(c);
    };

    // Cap fans point outward: -z on the bottom and +z on the top.
    for (int i = 1; i + 1 < number_of_nodes; ++i) {
        append_cone_tet(bottom(0), bottom(i + 1), bottom(i));
        append_cone_tet(top(0), top(i), top(i + 1));
    }

    // Two outward triangles for each rectangular side panel.
    for (int i = 0; i < number_of_nodes; ++i) {
        const int next = (i + 1) % number_of_nodes;
        append_cone_tet(bottom(i), bottom(next), top(next));
        append_cone_tet(bottom(i), top(next), top(i));
    }

    create_solid(world_positions, local_tets, density, ref_mesh, state);
    return base;
}

int append_normalized_tetgen_solid(
    const std::string& node_filename,
    const std::string& element_filename,
    DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, const double target_max_extent,
    const double density, const bool zero_based_index) {
    if (!center.allFinite()) {
        throw std::invalid_argument(
            "append_normalized_tetgen_solid: center must be finite");
    }
    if (!std::isfinite(target_max_extent) || target_max_extent <= 0.0) {
        throw std::invalid_argument(
            "append_normalized_tetgen_solid: target extent must be positive and finite");
    }
    if (!std::isfinite(density) || density < 0.0) {
        throw std::invalid_argument(
            "append_normalized_tetgen_solid: density must be nonnegative and finite");
    }
    if (state.deformed_positions.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "append_normalized_tetgen_solid: global node index exceeds int range");
    }

    std::vector<Vec3> positions;
    std::vector<int> tets;
    read_tetgen_nodes(node_filename, positions, zero_based_index);
    read_tetgen_tets(element_filename, tets, zero_based_index);
    if (positions.empty()) {
        throw std::invalid_argument(
            "append_normalized_tetgen_solid: TetGen node file is empty");
    }

    Vec3 lower = positions.front();
    Vec3 upper = positions.front();
    for (const Vec3& position : positions) {
        lower = lower.cwiseMin(position);
        upper = upper.cwiseMax(position);
    }
    const double source_max_extent = (upper - lower).maxCoeff();
    if (!std::isfinite(source_max_extent) || source_max_extent <= 0.0) {
        throw std::invalid_argument(
            "append_normalized_tetgen_solid: source bounding box has zero extent");
    }

    const Vec3 source_center = 0.5 * (lower + upper);
    const double scale = target_max_extent / source_max_extent;
    for (Vec3& position : positions)
        position = center + scale * (position - source_center);

    // TetGen files can use either orientation convention. create_solid and
    // the TGSL face convention require positive det(Dm), so flip exactly one
    // pair for every negative element. Full range/degeneracy validation is
    // still performed transactionally by create_solid below.
    for (std::size_t element = 0; element < tets.size() / 4; ++element) {
        int* tet = tets.data() + 4 * element;
        for (int local = 0; local < 4; ++local) {
            if (tet[local] < 0
                || static_cast<std::size_t>(tet[local]) >= positions.size()) {
                throw std::out_of_range(
                    "append_normalized_tetgen_solid: tetrahedron index is out of range");
            }
        }
        const Vec3& x0 = positions[static_cast<std::size_t>(tet[0])];
        const Vec3& x1 = positions[static_cast<std::size_t>(tet[1])];
        const Vec3& x2 = positions[static_cast<std::size_t>(tet[2])];
        const Vec3& x3 = positions[static_cast<std::size_t>(tet[3])];
        const double signed_six_volume =
            (x1 - x0).dot((x2 - x0).cross(x3 - x0));
        if (signed_six_volume < 0.0)
            std::swap(tet[2], tet[3]);
    }

    const int base = static_cast<int>(state.deformed_positions.size());
    create_solid(positions, tets, density, ref_mesh, state);
    return base;
}

int append_normalized_obj_rigid_body(
    const std::string& obj_filename,
    DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, const double target_max_extent,
    const double density, const Vec3& v_com,
    const Vec4& orientation, const Vec3& omega) {
    const char* function_name = "append_normalized_obj_rigid_body";
    if (!center.allFinite() || !v_com.allFinite() || !omega.allFinite()) {
        throw std::invalid_argument(
            std::string(function_name) + ": transforms and velocities must be finite");
    }
    if (!orientation.allFinite()
        || orientation.squaredNorm() <= 1.0e-24) {
        throw std::invalid_argument(
            std::string(function_name) + ": orientation must be a nonzero finite quaternion");
    }
    if (!std::isfinite(target_max_extent) || target_max_extent <= 0.0) {
        throw std::invalid_argument(
            std::string(function_name) + ": target extent must be positive and finite");
    }
    if (!std::isfinite(density) || density <= 0.0) {
        throw std::invalid_argument(
            std::string(function_name) + ": density must be positive and finite");
    }

    // Parse into local arrays first. No RefMesh or DeformedState storage is
    // touched until all geometry, topology, and mass validation has passed.
    std::vector<Vec3> raw_positions;
    std::vector<int> raw_triangles;
    load_obj_mesh(
        obj_filename, raw_positions, raw_triangles,
        /*scale=*/1.0, Vec3::Zero());
    if (raw_positions.empty() || raw_triangles.empty()
        || raw_triangles.size() % 3 != 0) {
        throw std::invalid_argument(
            std::string(function_name) + ": OBJ must contain a triangulated surface");
    }
    if (raw_positions.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string(function_name) + ": OBJ has too many vertices");
    }
    for (const Vec3& position : raw_positions) {
        if (!position.allFinite()) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ contains a non-finite vertex");
        }
    }

    // Remove orphan vertices. Besides avoiding wasted rigid proxy particles,
    // this keeps the requested extent and vertex-lumped inertia independent of
    // unused records in the OBJ file.
    std::vector<unsigned char> used(raw_positions.size(), 0);
    for (const int node : raw_triangles) {
        if (node < 0
            || static_cast<std::size_t>(node) >= raw_positions.size()) {
            throw std::out_of_range(
                std::string(function_name) + ": OBJ face index is out of range");
        }
        used[static_cast<std::size_t>(node)] = 1;
    }
    std::vector<int> remap(raw_positions.size(), -1);
    std::vector<Vec3> positions;
    positions.reserve(raw_positions.size());
    for (std::size_t node = 0; node < raw_positions.size(); ++node) {
        if (used[node] == 0)
            continue;
        remap[node] = static_cast<int>(positions.size());
        positions.push_back(raw_positions[node]);
    }
    std::vector<int> triangles = raw_triangles;
    for (int& node : triangles)
        node = remap[static_cast<std::size_t>(node)];

    Vec3 lower = positions.front();
    Vec3 upper = positions.front();
    for (const Vec3& position : positions) {
        lower = lower.cwiseMin(position);
        upper = upper.cwiseMax(position);
    }
    const Vec3 source_center = 0.5 * (lower + upper);
    const double source_max_extent = (upper - lower).maxCoeff();
    if (!std::isfinite(source_max_extent) || source_max_extent <= 0.0) {
        throw std::invalid_argument(
            std::string(function_name) + ": OBJ bounding box has zero extent");
    }

    // A closed consistently oriented triangle manifold has two oppositely
    // directed occurrences of every undirected edge. The edge records also
    // connect triangle components so that disconnected closed shells have
    // their absolute enclosed volumes added instead of canceling each other.
    struct EdgeRecord {
        int count = 0;
        int direction_balance = 0;
        int first_triangle = -1;
    };
    std::map<std::array<int, 2>, EdgeRecord> edges;
    std::set<std::array<int, 3>> unique_triangles;
    const int triangle_count = static_cast<int>(triangles.size() / 3);
    std::vector<int> component_parent(
        static_cast<std::size_t>(triangle_count));
    std::iota(component_parent.begin(), component_parent.end(), 0);
    const auto find_component = [&component_parent](int triangle) {
        int root = triangle;
        while (component_parent[static_cast<std::size_t>(root)] != root)
            root = component_parent[static_cast<std::size_t>(root)];
        while (component_parent[static_cast<std::size_t>(triangle)]
               != triangle) {
            const int parent =
                component_parent[static_cast<std::size_t>(triangle)];
            component_parent[static_cast<std::size_t>(triangle)] = root;
            triangle = parent;
        }
        return root;
    };
    const auto unite_components =
        [&component_parent, &find_component](int first, int second) {
            first = find_component(first);
            second = find_component(second);
            if (first != second)
                component_parent[static_cast<std::size_t>(second)] = first;
        };

    const double area_tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * source_max_extent * source_max_extent;
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        const int nodes[3] = {
            triangles[3 * triangle], triangles[3 * triangle + 1],
            triangles[3 * triangle + 2]};
        if (nodes[0] == nodes[1] || nodes[1] == nodes[2]
            || nodes[2] == nodes[0]) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ contains a repeated triangle vertex");
        }
        std::array<int, 3> face{nodes[0], nodes[1], nodes[2]};
        std::sort(face.begin(), face.end());
        if (!unique_triangles.insert(face).second) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ contains a duplicate triangle");
        }
        const Vec3& x0 = positions[static_cast<std::size_t>(nodes[0])];
        const Vec3& x1 = positions[static_cast<std::size_t>(nodes[1])];
        const Vec3& x2 = positions[static_cast<std::size_t>(nodes[2])];
        if ((x1 - x0).cross(x2 - x0).norm() <= area_tolerance) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ contains a degenerate triangle");
        }

        for (int local = 0; local < 3; ++local) {
            const int first = nodes[local];
            const int second = nodes[(local + 1) % 3];
            const std::array<int, 2> edge{
                std::min(first, second), std::max(first, second)};
            EdgeRecord& record = edges[edge];
            ++record.count;
            record.direction_balance += first < second ? 1 : -1;
            if (record.first_triangle < 0) {
                record.first_triangle = triangle;
            } else {
                unite_components(record.first_triangle, triangle);
            }
        }
    }
    for (const auto& [edge, record] : edges) {
        (void)edge;
        if (record.count != 2) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ surface is not closed and edge-manifold");
        }
        if (record.direction_balance != 0) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ triangle winding is inconsistent");
        }
    }

    std::vector<double> component_six_volumes(
        static_cast<std::size_t>(triangle_count), 0.0);
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        const int root = find_component(triangle);
        const Vec3 x0 = positions[static_cast<std::size_t>(
                            triangles[3 * triangle])]
            - source_center;
        const Vec3 x1 = positions[static_cast<std::size_t>(
                            triangles[3 * triangle + 1])]
            - source_center;
        const Vec3 x2 = positions[static_cast<std::size_t>(
                            triangles[3 * triangle + 2])]
            - source_center;
        component_six_volumes[static_cast<std::size_t>(root)] +=
            x0.dot(x1.cross(x2));
    }
    const double volume_tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * source_max_extent * source_max_extent * source_max_extent;
    double source_volume = 0.0;
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        if (find_component(triangle) != triangle)
            continue;
        const double six_volume =
            component_six_volumes[static_cast<std::size_t>(triangle)];
        if (!std::isfinite(six_volume)
            || std::abs(six_volume) <= volume_tolerance) {
            throw std::invalid_argument(
                std::string(function_name) + ": OBJ encloses zero or invalid volume");
        }
        source_volume += std::abs(six_volume) / 6.0;
    }

    const double scale = target_max_extent / source_max_extent;
    const double total_mass = density * source_volume
        * scale * scale * scale;
    if (!std::isfinite(total_mass) || total_mass <= 0.0) {
        throw std::invalid_argument(
            std::string(function_name) + ": normalized mass is not positive and finite");
    }
    const Vec4 normalized_orientation = quaternion_normalize(orientation);
    for (Vec3& position : positions) {
        position = center + quaternion_rotate(
            normalized_orientation,
            scale * (position - source_center));
    }

    const std::size_t node_base = state.deformed_positions.size();
    const std::size_t max_int =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (node_base > max_int || positions.size() > max_int - node_base) {
        throw std::overflow_error(
            std::string(function_name) + ": global vertex index exceeds int range");
    }
    if (triangles.size()
        > ref_mesh.tris.max_size() - ref_mesh.tris.size()) {
        throw std::overflow_error(
            std::string(function_name) + ": triangle storage exceeds vector limits");
    }

    // Reserve before create_rigid_body mutates the particle arrays. All
    // subsequent triangle insertions are then non-allocating integer writes.
    ref_mesh.tris.reserve(ref_mesh.tris.size() + triangles.size());
    const int rigid_body = create_rigid_body(
        positions, v_com, normalized_orientation, omega, total_mass,
        ref_mesh, state);
    for (const int local_node : triangles) {
        ref_mesh.tris.push_back(
            static_cast<int>(node_base) + local_node);
    }
    return rigid_body;
}

// Total number of vertices is: (nx + 1) * (ny + 1) and total number of triangles is: 2 * nx * ny
int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nx, int ny, double width, double height, const Vec3& origin) {
    int base = static_cast<int>(state.deformed_positions.size());

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {

            // Normalize grid coordinates from 0 to 1
            double u = static_cast<double>(i) / nx;
            double v = static_cast<double>(j) / ny;

            // Scale to actual size
            double x_ref = u * width;
            double y_ref = v * height;

            // Store reference (2D) and deformed (3D) positions
            X.push_back(Vec2(x_ref, y_ref));
            state.deformed_positions.push_back(origin + Vec3(x_ref, 0.0, y_ref));
        }
    }

    // convert (col, row) -> vertex index
    auto vertex_index = [base, nx](int i, int j) {
        return base + j * (nx + 1) + i;
    };

    // Create triangles
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int v00 = vertex_index(i, j);
            int v10 = vertex_index(i + 1, j);
            int v01 = vertex_index(i, j + 1);
            int v11 = vertex_index(i + 1, j + 1);

            // Split square into two triangles
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v10); ref_mesh.tris.push_back(v11);
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v11); ref_mesh.tris.push_back(v01);
        }
    }

    ref_mesh.initialize(X, state.deformed_positions);

    return base;
}

// V = nu(n_rows + 1) + 2 and T = 2nu(n_rows + 1), including both end caps.
int build_cylinder_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
                        int nu, double radius, double length, const Vec3& center) {
    constexpr double kPi = 3.14159265358979323846;
    const int    base        = static_cast<int>(state.deformed_positions.size());
    const double two_pi      = 2.0 * kPi;
    const double theta_start = -0.5 * kPi;

    const double iso_row_h = (two_pi * radius / nu) * 0.5 * std::sqrt(3.0);
    const int    n_rows    = std::max(1, static_cast<int>(std::round(length / iso_row_h)));

    for (int j = 0; j <= n_rows; ++j) {
        const double v        = static_cast<double>(j) / n_rows;
        const double z_ref    = v * length;
        const double z_world  = center.z() - 0.5 * length + z_ref;
        const bool   shifted  = (j % 2 == 1);
        const double u_offset = shifted ? 0.5 / nu : 0.0;

        for (int i = 0; i < nu; ++i) {
            const double u     = static_cast<double>(i) / nu + u_offset;
            const double theta = theta_start + u * two_pi;

            X.push_back(Vec2(u * two_pi * radius, z_ref));
            state.deformed_positions.push_back(
                Vec3(center.x() + radius * std::cos(theta),
                     center.y() + radius * std::sin(theta),
                     z_world));
        }
    }

    auto vertex_index = [base, nu](int i, int j) {
        return base + j * nu + i;
    };

    // Row j is shifted when j is odd. Each pair (j, j+1) contributes 2*nu
    // triangles: one up-pointing (two verts on the non-shifted row, apex on
    // the shifted row) and one down-pointing (apex on the non-shifted row,
    // two verts on the shifted row). Wrap via i_next = (i+1) % nu.
    for (int j = 0; j < n_rows; ++j) {
        const bool j_shifted = (j % 2 == 1);
        for (int i = 0; i < nu; ++i) {
            const int i_next = (i + 1) % nu;
            if (!j_shifted) {
                // Non-shifted row j below, shifted row j+1 above.
                ref_mesh.tris.push_back(vertex_index(i,      j));
                ref_mesh.tris.push_back(vertex_index(i_next, j));
                ref_mesh.tris.push_back(vertex_index(i,      j + 1));

                ref_mesh.tris.push_back(vertex_index(i_next, j));
                ref_mesh.tris.push_back(vertex_index(i_next, j + 1));
                ref_mesh.tris.push_back(vertex_index(i,      j + 1));
            } else {
                // Shifted row j below, non-shifted row j+1 above.
                ref_mesh.tris.push_back(vertex_index(i,      j));
                ref_mesh.tris.push_back(vertex_index(i_next, j));
                ref_mesh.tris.push_back(vertex_index(i_next, j + 1));

                ref_mesh.tris.push_back(vertex_index(i,      j));
                ref_mesh.tris.push_back(vertex_index(i_next, j + 1));
                ref_mesh.tris.push_back(vertex_index(i,      j + 1));
            }
        }
    }

    // End caps: fan triangles around a center vertex on each circular end.
    // Cap centers sit off the unrolled strip (y=0 and y=length are occupied
    // by the boundary rings) so the fan triangles stay non-degenerate in
    // parameter space and buildCorotatedCache's Eigen decomposition succeeds.
    const double cap_offset = radius;
    const int bot_center = static_cast<int>(state.deformed_positions.size());
    X.push_back(Vec2(kPi * radius, -cap_offset));
    state.deformed_positions.push_back(
        Vec3(center.x(), center.y(), center.z() - 0.5 * length));

    const int top_center = static_cast<int>(state.deformed_positions.size());
    X.push_back(Vec2(kPi * radius, length + cap_offset));
    state.deformed_positions.push_back(
        Vec3(center.x(), center.y(), center.z() + 0.5 * length));

    for (int i = 0; i < nu; ++i) {
        const int i_next = (i + 1) % nu;
        // Bottom cap faces -z: winding (center, ring[i_next], ring[i]).
        ref_mesh.tris.push_back(bot_center);
        ref_mesh.tris.push_back(vertex_index(i_next, 0));
        ref_mesh.tris.push_back(vertex_index(i,      0));
        // Top cap faces +z: winding (center, ring[i], ring[i_next]).
        ref_mesh.tris.push_back(top_center);
        ref_mesh.tris.push_back(vertex_index(i,      n_rows));
        ref_mesh.tris.push_back(vertex_index(i_next, n_rows));
    }

    ref_mesh.initialize(X, state.deformed_positions);

    return base;
}

int build_sphere_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
                      int subdiv, double radius, const Vec3& center) {
    const int base = static_cast<int>(state.deformed_positions.size());

    // Base icosahedron: 12 vertices (at ||v|| = sqrt(1 + phi^2) with phi = golden
    // ratio), 20 triangles. See e.g. Catmull-Clark notes or any graphics text.
    constexpr double kPhi  = 1.6180339887498948482;  // (1 + sqrt(5)) / 2
    const double     kNorm = std::sqrt(1.0 + kPhi * kPhi);
    const double     s     = radius / kNorm;  // scale so |v| = radius

    // Unit-icosahedron vertices, pre-scaled to 'radius'. Rotated by a small
    // irrational angle around +x so that subdivision midpoints (e.g. the edge
    // between (0, -1, +phi) and (0, -1, -phi)) don't land exactly on the +-y
    // axis, where the stereographic projection used for the 2D ref coord
    // would otherwise become singular.
    constexpr double kTilt = 0.1;  // rad; irrational in practice, breaks ±y axis alignment
    const double ct = std::cos(kTilt), st = std::sin(kTilt);
    auto tilt = [ct, st, s](double x, double y, double z) -> Vec3 {
        const double yp = ct * y - st * z;
        const double zp = st * y + ct * z;
        return Vec3(s * x, s * yp, s * zp);
    };
    std::vector<Vec3> verts = {
        tilt(-1.0,  kPhi,  0.0), tilt( 1.0,  kPhi,  0.0),
        tilt(-1.0, -kPhi,  0.0), tilt( 1.0, -kPhi,  0.0),
        tilt( 0.0, -1.0,  kPhi), tilt( 0.0,  1.0,  kPhi),
        tilt( 0.0, -1.0, -kPhi), tilt( 0.0,  1.0, -kPhi),
        tilt( kPhi,  0.0, -1.0), tilt( kPhi,  0.0,  1.0),
        tilt(-kPhi,  0.0, -1.0), tilt(-kPhi,  0.0,  1.0),
    };

    // 20 faces of the base icosahedron (outward-normal winding when vertices
    // are placed as above).
    std::vector<std::array<int, 3>> faces = {
        {0, 11,  5}, {0,  5,  1}, {0,  1,  7}, {0,  7, 10}, {0, 10, 11},
        {1,  5,  9}, {5, 11,  4}, {11, 10, 2}, {10, 7,  6}, {7,  1,  8},
        {3,  9,  4}, {3,  4,  2}, {3,  2,  6}, {3,  6,  8}, {3,  8,  9},
        {4,  9,  5}, {2,  4, 11}, {6,  2, 10}, {8,  6,  7}, {9,  8,  1},
    };

    // Loop-subdivide: each triangle splits into 4 by inserting edge midpoints,
    // normalized to the sphere. Dedupe midpoints by canonicalized edge key.
    for (int level = 0; level < subdiv; ++level) {
        std::map<std::pair<int, int>, int> midpoint_cache;
        auto get_midpoint = [&](int a, int b) -> int {
            const auto key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            auto it = midpoint_cache.find(key);
            if (it != midpoint_cache.end()) return it->second;
            Vec3 mid = 0.5 * (verts[a] + verts[b]);
            mid *= radius / mid.norm();
            const int idx = static_cast<int>(verts.size());
            verts.push_back(mid);
            midpoint_cache.emplace(key, idx);
            return idx;
        };

        std::vector<std::array<int, 3>> next_faces;
        next_faces.reserve(faces.size() * 4);
        for (const auto& f : faces) {
            const int a = f[0], b = f[1], c = f[2];
            const int ab = get_midpoint(a, b);
            const int bc = get_midpoint(b, c);
            const int ca = get_midpoint(c, a);
            next_faces.push_back({a,  ab, ca});
            next_faces.push_back({ab,  b, bc});
            next_faces.push_back({ca, bc,  c});
            next_faces.push_back({ab, bc, ca});
        }
        faces.swap(next_faces);
    }

    // Emit vertices translated to `center`. 2D ref coord uses stereographic
    // projection from (0, -radius, 0): X = 2r * (x, z) / (y + r). This is
    // conformal so every triangle has non-degenerate 2D area. The projection
    // pole (0, -radius, 0) is never an icosphere vertex (no subdivided vertex
    // lands on the +-y axis).
    for (const Vec3& v : verts) {
        state.deformed_positions.push_back(v + center);
        const double denom = v.y() + radius;
        X.push_back(Vec2(2.0 * radius * v.x() / denom,
                         2.0 * radius * v.z() / denom));
    }

    // Emit triangles with per-batch vertex-index offset by `base`.
    for (const auto& f : faces) {
        ref_mesh.tris.push_back(base + f[0]);
        ref_mesh.tris.push_back(base + f[1]);
        ref_mesh.tris.push_back(base + f[2]);
    }

    ref_mesh.initialize(X, state.deformed_positions);

    return base;
}

int load_obj_mesh(const std::string& path, RefMesh& ref_mesh, DeformedState& state,
                  double scale, const Vec3& origin) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_obj_mesh: cannot open '" + path + "'");
    }

    const int base = static_cast<int>(state.deformed_positions.size());

    std::vector<Vec3> raw_verts;
    raw_verts.reserve(1 << 16);
    std::vector<std::array<int, 3>> raw_tris;
    raw_tris.reserve(1 << 16);

    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "v") {
            double x, y, z;
            if (iss >> x >> y >> z) raw_verts.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<int> idx;
            idx.reserve(4);
            std::string tok;
            while (iss >> tok) {
                // OBJ corners may be v / v/vt / v/vt/vn / v//vn -- take the
                // leading vertex index.
                const std::size_t slash = tok.find('/');
                const std::string vstr = (slash == std::string::npos) ? tok : tok.substr(0, slash);
                if (vstr.empty()) continue;
                int v = std::stoi(vstr);
                // Negative indices are relative to the current vertex count.
                if (v < 0) v = static_cast<int>(raw_verts.size()) + 1 + v;
                idx.push_back(v - 1);
            }
            for (int i = 1; i + 1 < static_cast<int>(idx.size()); ++i) {
                raw_tris.push_back({idx[0], idx[i], idx[i + 1]});
            }
        }
    }

    // Drop orphan vertices: simulation.cpp's residual loop calls adj.at(vi)
    // for every vertex in deformed_positions, which throws for any vertex
    // unreferenced by ref_mesh.tris.
    std::vector<char> is_used(raw_verts.size(), 0);
    for (const auto& t : raw_tris) {
        if (t[0] >= 0 && t[0] < (int)raw_verts.size()) is_used[t[0]] = 1;
        if (t[1] >= 0 && t[1] < (int)raw_verts.size()) is_used[t[1]] = 1;
        if (t[2] >= 0 && t[2] < (int)raw_verts.size()) is_used[t[2]] = 1;
    }
    std::vector<int> remap(raw_verts.size(), -1);
    int kept = 0;
    for (std::size_t i = 0; i < raw_verts.size(); ++i) {
        if (is_used[i]) remap[i] = kept++;
    }

    state.deformed_positions.reserve(state.deformed_positions.size() + kept);
    for (std::size_t i = 0; i < raw_verts.size(); ++i) {
        if (!is_used[i]) continue;
        state.deformed_positions.push_back(scale * raw_verts[i] + origin);
    }

    ref_mesh.tris.reserve(ref_mesh.tris.size() + raw_tris.size() * 3);
    for (const auto& t : raw_tris) {
        ref_mesh.tris.push_back(base + remap[t[0]]);
        ref_mesh.tris.push_back(base + remap[t[1]]);
        ref_mesh.tris.push_back(base + remap[t[2]]);
    }

    ref_mesh.initialize(state.deformed_positions);

    return base;
}

void load_obj_mesh(const std::string& path, std::vector<Vec3>& verts,
                   std::vector<int>& tris, double scale, const Vec3& origin) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("load_obj_mesh: cannot open '" + path + "'");

    const int base = static_cast<int>(verts.size());
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2 || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "v") {
            double x, y, z;
            if (iss >> x >> y >> z) verts.push_back(scale * Vec3(x, y, z) + origin);
        } else if (tag == "f") {
            std::vector<int> idx;
            std::string tok;
            while (iss >> tok) {
                const std::size_t slash = tok.find('/');
                const std::string vstr = (slash == std::string::npos) ? tok : tok.substr(0, slash);
                if (vstr.empty()) continue;
                int v = std::stoi(vstr);
                if (v < 0) v = static_cast<int>(verts.size()) - base + 1 + v;
                idx.push_back(base + v - 1);
            }
            for (int i = 1; i + 1 < static_cast<int>(idx.size()); ++i) {
                tris.push_back(idx[0]);
                tris.push_back(idx[i]);
                tris.push_back(idx[i + 1]);
            }
        }
    }
}

void rebuild_triangle_rest_isometric(RefMesh& ref_mesh,
                                     const std::vector<Vec3>& x_rest,
                                     int t_begin, int t_end) {
    const int nt = num_tris(ref_mesh);
    if (t_begin < 0 || t_end > nt || t_begin >= t_end) return;

    // Lay each rest triangle flat in 2D with X0=(0,0), X1 along +x at the
    // true 3D edge length, X2 placed so |X2-X0| / |X2-X1| match their 3D
    // counterparts. All three rest edge lengths are preserved -> corotated F
    // is identity at the rest pose. Degenerate triangles fall back to area=0
    // and Dm_inverse=I so they contribute nothing to the elastic gradient
    // (corotated_node_gradient/Hessian both scale linearly in ref_area).
    for (int t = t_begin; t < t_end; ++t) {
        const Vec3& p0 = x_rest[ref_mesh.tris[3 * t + 0]];
        const Vec3& p1 = x_rest[ref_mesh.tris[3 * t + 1]];
        const Vec3& p2 = x_rest[ref_mesh.tris[3 * t + 2]];

        const Vec3   e1     = p1 - p0;
        const double e1_len = e1.norm();
        if (e1_len <= 0.0) {
            ref_mesh.area[t]       = 0.0;
            ref_mesh.Dm_inverse[t] = Mat22::Identity();
            continue;
        }
        const Vec3   e2          = p2 - p0;
        const Vec3   e1_unit     = e1 / e1_len;
        const double dot         = e2.dot(e1_unit);
        const double e2_perp_len = (e2 - dot * e1_unit).norm();

        Mat22 Dm;
        Dm.col(0) = Vec2(e1_len, 0.0);
        Dm.col(1) = Vec2(dot,    e2_perp_len);

        const double det = Dm.determinant();
        if (det == 0.0) {
            ref_mesh.area[t]       = 0.0;
            ref_mesh.Dm_inverse[t] = Mat22::Identity();
            continue;
        }
        ref_mesh.area[t]       = 0.5 * std::abs(det);
        ref_mesh.Dm_inverse[t] = Dm.inverse();
    }
}

void rebuild_hinge_c_e_3d(RefMesh& ref_mesh,
                          const std::vector<Vec3>& x_rest,
                          int v_begin, int v_end) {
    for (Hinge& h : ref_mesh.hinges) {
        bool all_in_range = true;
        for (int k = 0; k < 4; ++k) {
            if (h.v[k] < v_begin || h.v[k] >= v_end) { all_in_range = false; break; }
        }
        if (!all_in_range) continue;

        const Vec3& p0 = x_rest[h.v[0]];
        const Vec3& p1 = x_rest[h.v[1]];
        const Vec3& p2 = x_rest[h.v[2]];
        const Vec3& p3 = x_rest[h.v[3]];
        const Vec3   e        = p1 - p0;
        const double edge_len2 = e.squaredNorm();
        const double areaA = 0.5 * (e.cross(p2 - p0)).norm();
        const double areaB = 0.5 * (e.cross(p3 - p0)).norm();
        const double area_sum = areaA + areaB;
        h.c_e = (area_sum > 0.0) ? (edge_len2 / area_sum) : 0.0;
    }
}

#include "make_shape.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& ref_mesh, int tri_idx) {
    TriangleDef def;
    def.x[0] = x[tri_vertex(ref_mesh, tri_idx, 0)];
    def.x[1] = x[tri_vertex(ref_mesh, tri_idx, 1)];
    def.x[2] = x[tri_vertex(ref_mesh, tri_idx, 2)];
    return def;
}

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt) {
    int n = static_cast<int>(x.size());
    xhat.resize(n);
    for (int i = 0; i < n; ++i) xhat[i] = x[i] + dt * v[i];
}

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt) {
    int n = static_cast<int>(xnew.size());
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = (xnew[i] - xold[i]) / dt;
}

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, std::vector<Pin>& pins) {
    X.clear();
    ref_mesh.tris.clear();
    state.deformed_positions.clear();
    state.velocities.clear();
    pins.clear();
}

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vertex_index, x[vertex_index]});
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

    ref_mesh.initialize(X);

    return base;
}

// Total number of vertices is: nu * (nv + 1) and total number of triangles is: 2 * (nu - 1) * nv.
// One wrap column is intentionally skipped, leaving a thin gap on the cylinder's underside
// (theta_start = -pi/2). Welding the seam would duplicate vertices in 3D and trip the IPC barrier.
int build_cylinder_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
                        int nu, int nv, double radius, double length, const Vec3& center) {
    constexpr double kPi = 3.14159265358979323846;
    int base = static_cast<int>(state.deformed_positions.size());
    const double two_pi      = 2.0 * kPi;
    const double theta_start = -0.5 * kPi;

    for (int j = 0; j <= nv; ++j) {

        // Normalize axial coordinate from 0 to 1
        double v = static_cast<double>(j) / nv;

        // Scale to actual length, both as 2D parameter and 3D world coord
        double z_ref   = v * length;
        double z_world = center.z() - 0.5 * length + z_ref;

        for (int i = 0; i < nu; ++i) {

            double u     = static_cast<double>(i) / nu;
            double theta = theta_start + u * two_pi;

            // Store reference (2D unrolled) and deformed (3D) positions
            X.push_back(Vec2(u * two_pi * radius, z_ref));
            state.deformed_positions.push_back(
                Vec3(center.x() + radius * std::cos(theta),
                     center.y() + radius * std::sin(theta),
                     z_world));
        }
    }

    // convert (col, row) -> vertex index
    auto vertex_index = [base, nu](int i, int j) {
        return base + j * nu + i;
    };

    // Create triangles, skipping the wrap column (i == nu-1 -> i == 0)
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu - 1; ++i) {
            int v00 = vertex_index(i,     j);
            int v10 = vertex_index(i + 1, j);
            int v01 = vertex_index(i,     j + 1);
            int v11 = vertex_index(i + 1, j + 1);

            // Split quad into two triangles
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v10); ref_mesh.tris.push_back(v11);
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v11); ref_mesh.tris.push_back(v01);
        }
    }

    ref_mesh.initialize(X);

    return base;
}

std::unordered_map<int, std::vector<int>> build_vertex_adjacency_map(const std::vector<int>& tris) {
    std::unordered_map<int, std::unordered_set<int>> adj_set;
    int nt = static_cast<int>(tris.size()) / 3;
    for (int t = 0; t < nt; ++t) {
        int a = tris[t * 3 + 0];
        int b = tris[t * 3 + 1];
        int c = tris[t * 3 + 2];
        adj_set[a].insert(b); adj_set[a].insert(c);
        adj_set[b].insert(a); adj_set[b].insert(c);
        adj_set[c].insert(a); adj_set[c].insert(b);
    }
    std::unordered_map<int, std::vector<int>> result;
    result.reserve(adj_set.size());
    for (auto& [v, neighbors] : adj_set)
        result[v] = std::vector<int>(neighbors.begin(), neighbors.end());
    return result;
}

std::vector<std::vector<int>> greedy_color(
    const std::unordered_map<int, std::vector<int>>& adj,
    int num_vertices)
{
    std::vector<int> color(num_vertices, -1);

    for (int v = 0; v < num_vertices; ++v) {
        // Collect colors already used by neighbors
        std::unordered_set<int> used;
        if (adj.count(v))
            for (int n : adj.at(v))
                if (color[n] != -1) used.insert(color[n]);

        // Pick the smallest color not in use
        int c = 0;
        while (used.count(c)) ++c;
        color[v] = c;
    }

    // Invert: color -> list of vertices
    int num_colors = *std::max_element(color.begin(), color.end()) + 1;
    std::vector<std::vector<int>> groups(num_colors);
    for (int v = 0; v < num_vertices; ++v)
        groups[color[v]].push_back(v);
    return groups;
}

// Builds VertexTriangleMap: each vertex maps to {triangle_index, local_node_index} pairs.
// Storing the local index (0,1,2) eliminates the linear search previously done by local_node().
VertexTriangleMap build_incident_triangle_map(const std::vector<int>& indices) {
    VertexTriangleMap result;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        const int vertex     = indices[i];
        const int tri_idx    = i / 3;
        const int local_node = i % 3;
        result[vertex].emplace_back(tri_idx, local_node);
    }
    return result;
}

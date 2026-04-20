#include "make_shape.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

// Total number of vertices is: nu * (nv + 1) and total number of triangles is: 2 * nu * nv.
// The surface is closed: the wrap column (i == nu-1 -> i == 0) reuses existing vertex
// indices, so no vertex is duplicated in 3D.
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

    // Create triangles, wrapping the last column back to i == 0 so the cylinder is closed.
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const int i_next = (i + 1) % nu;
            int v00 = vertex_index(i,      j);
            int v10 = vertex_index(i_next, j);
            int v01 = vertex_index(i,      j + 1);
            int v11 = vertex_index(i_next, j + 1);

            // Split quad into two triangles
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v10); ref_mesh.tris.push_back(v11);
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v11); ref_mesh.tris.push_back(v01);
        }
    }

    ref_mesh.initialize(X);

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

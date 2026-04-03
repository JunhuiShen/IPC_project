#include "make_shape.h"
#include <unordered_map>

TriangleRest make_rest_triangle(const std::vector<Vec2>& X, const RefMesh& ref_mesh, int tri_idx) {
    TriangleRest rest;
    rest.X[0] = X[tri_vertex(ref_mesh, tri_idx, 0)];
    rest.X[1] = X[tri_vertex(ref_mesh, tri_idx, 1)];
    rest.X[2] = X[tri_vertex(ref_mesh, tri_idx, 2)];
    return rest;
}

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

int build_single_triangle(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
                          const Vec2& X0, const Vec2& X1, const Vec2& X2,
                          const Vec3& x0, const Vec3& x1, const Vec3& x2) {
    int base = static_cast<int>(state.deformed_positions.size());
    X.push_back(X0); X.push_back(X1); X.push_back(X2);
    state.deformed_positions.push_back(x0);
    state.deformed_positions.push_back(x1);
    state.deformed_positions.push_back(x2);
    ref_mesh.tris.push_back(base + 0);
    ref_mesh.tris.push_back(base + 1);
    ref_mesh.tris.push_back(base + 2);
    return base;
}

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vertex_index, x[vertex_index]});
}

int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nx, int ny, double width, double height, const Vec3& origin) {
    int base = static_cast<int>(state.deformed_positions.size());

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            double u = static_cast<double>(i) / nx;
            double v = static_cast<double>(j) / ny;
            X.push_back(Vec2(u * width, v * height));
            state.deformed_positions.push_back(origin + Vec3(u * width, v * height, 0.0));
        }
    }

    auto vidx = [base, nx](int i, int j) { return base + j * (nx + 1) + i; };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int v00 = vidx(i, j), v10 = vidx(i+1, j);
            int v01 = vidx(i, j+1), v11 = vidx(i+1, j+1);
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v10); ref_mesh.tris.push_back(v11);
            ref_mesh.tris.push_back(v00); ref_mesh.tris.push_back(v11); ref_mesh.tris.push_back(v01);
        }
    }

    ref_mesh.initialize(X);
    return base;
}

std::unordered_map<int, std::vector<int>> build_incident_triangle_map(const std::vector<int>& indices) {
    std::unordered_map<int, std::vector<int>> result;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i)
        result[indices[i]].push_back(i / 3);
    return result;
}

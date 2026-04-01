#include "physics.h"
#include "make_shape.h"
#include <algorithm>
#include <cmath>

double triangle_ref_area_2d(const RefMesh& ref_mesh, const Tri& tri){
    const Vec2& X0 = ref_mesh.ref_positions[tri.v[0]];
    const Vec2& X1 = ref_mesh.ref_positions[tri.v[1]];
    const Vec2& X2 = ref_mesh.ref_positions[tri.v[2]];

    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    return 0.5 * std::abs(Dm_local.determinant());
}

LumpedMass build_lumped_mass(const RefMesh& ref_mesh, double density, double thickness){
    LumpedMass M;
    M.vertex_masses.assign(ref_mesh.ref_positions.size(), 0.0);

    for (const Tri& tri : ref_mesh.tris) {
        double A = triangle_ref_area_2d(ref_mesh, tri);
        double m = density * A * thickness;
        double mv = m / 3.0;
        for (int a : tri.v) M.vertex_masses[a] += mv;
    }

    return M;
}

VertexAdjacency build_vertex_adjacency(const RefMesh& ref_mesh){
    VertexAdjacency adj;
    adj.incident_triangle_indices.resize(ref_mesh.ref_positions.size());

    for (int t = 0; t < static_cast<int>(ref_mesh.tris.size()); ++t) {
        const Tri& tri = ref_mesh.tris[t];
        for (int a : tri.v) adj.incident_triangle_indices[a].push_back(t);
    }

    return adj;
}

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double E = 0.0, PE = 0.0, dt2 = params.dt * params.dt;

    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        Vec3 dx = x[i] - xhat[i];
        E += 0.5 * lumped_mass.vertex_masses[i] * dx.squaredNorm();
    }

    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        PE += -lumped_mass.vertex_masses[i] * params.gravity.dot(x[i]);

    for (const Pin& pin : pins) {
        Vec3 dx = x[pin.vertex_index] - pin.target_position;
        PE += 0.5 * params.kpin * dx.squaredNorm();
    }

    for (const Tri& tri : ref_mesh.tris)
        PE += corotated_energy(make_rest_triangle(ref_mesh, tri), make_def_triangle(x, tri), params.mu, params.lambda);

    return E + dt2 * PE;
}

Vec3 compute_local_gradient_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double dt2 = params.dt * params.dt;
    Vec3 g = Vec3::Zero();

    g += lumped_mass.vertex_masses[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-lumped_mass.vertex_masses[vi] * params.gravity);

    for (const Pin& pin : pins)
        if (pin.vertex_index == vi) g += dt2 * params.kpin * (x[vi] - pin.target_position);

    for (int ti : adj.incident_triangle_indices[vi]) {
        const Tri& tri = ref_mesh.tris[ti];
        auto node_g = corotated_node_gradient(make_rest_triangle(ref_mesh, tri), make_def_triangle(x, tri), params.mu, params.lambda);

        for (int a = 0; a < 3; ++a)
            if (tri.v[a] == vi) g += dt2 * node_g[a];
    }

    return g;
}

Mat33 compute_local_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x){
    double dt2 = params.dt * params.dt;
    Mat33 H = Mat33::Zero();

    H += lumped_mass.vertex_masses[vi] * Mat33::Identity();

    for (const Pin& pin : pins)
        if (pin.vertex_index == vi) H += dt2 * params.kpin * Mat33::Identity();

    for (int ti : adj.incident_triangle_indices[vi]) {
        const Tri& tri = ref_mesh.tris[ti];
        Mat99 tri_H = corotated_node_hessian(make_rest_triangle(ref_mesh, tri), make_def_triangle(x, tri), params.mu, params.lambda);

        for (int a = 0; a < 3; ++a)
            if (tri.v[a] == vi) H += dt2 * tri_H.block<3,3>(3 * a, 3 * a);
    }

    return H;
}

double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double r_inf = 0.0;

    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        Vec3 g = compute_local_gradient_no_barrier(i, ref_mesh, lumped_mass, adj, pins, params, x, xhat);
        r_inf = std::max(r_inf, std::abs(g.x()));
        r_inf = std::max(r_inf, std::abs(g.y()));
        r_inf = std::max(r_inf, std::abs(g.z()));
    }

    return r_inf;
}
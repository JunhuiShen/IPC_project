#include "physics.h"
#include "make_shape.h"
#include <algorithm>
#include <cmath>

double triangle_ref_area_2d(const RefMesh& ref_mesh, int tri_idx) {
    const Vec2& X0 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 0)];
    const Vec2& X1 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 1)];
    const Vec2& X2 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 2)];

    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    return 0.5 * std::abs(Dm_local.determinant());
}

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                     const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    double E = 0.0, PE = 0.0, dt2 = params.dt() * params.dt();

    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        Vec3 dx = x[i] - xhat[i];
        E += 0.5 * ref_mesh.mass[i] * dx.squaredNorm();
    }

    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        PE += -ref_mesh.mass[i] * params.gravity.dot(x[i]);

    for (const Pin& pin : pins) {
        Vec3 dx = x[pin.vertex_index] - pin.target_position;
        PE += 0.5 * params.kpin * dx.squaredNorm();
    }

    for (int t = 0; t < num_tris(ref_mesh); ++t)
        PE += corotated_energy(ref_mesh.area[t], ref_mesh.Dm_inverse[t], make_def_triangle(x, ref_mesh, t), params.mu, params.lambda);

    return E + dt2 * PE;
}

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double dt2 = params.dt() * params.dt();
    Vec3 g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    g += ref_mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-ref_mesh.mass[vi] * params.gravity);
    H += ref_mesh.mass[vi] * Mat33::Identity();

    for (const Pin& pin : pins) {
        if (pin.vertex_index == vi) {
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
            H += dt2 * params.kpin * Mat33::Identity();
        }
    }

    for (int ti : adj.at(vi)) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, ti);

        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
        const Mat32 F = Ds_mat * Dm_inv;
        const double A = ref_mesh.area[ti];

        CorotatedCache32 cache = buildCorotatedCache(F);

        auto node_g = corotated_node_gradient(cache, F, A, Dm_inv, params.mu, params.lambda);
        Mat99 tri_H = corotated_node_hessian(cache, F, A, Dm_inv, params.mu, params.lambda);

        for (int a = 0; a < 3; ++a) {
            if (tri_vertex(ref_mesh, ti, a) == vi) {
                g += dt2 * node_g[a];
                H += dt2 * tri_H.block<3,3>(3 * a, 3 * a);
            }
        }
    }

    return {g, H};
}

Vec3 compute_local_gradient_no_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double dt2 = params.dt() * params.dt();
    Vec3 g = Vec3::Zero();

    g += ref_mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-ref_mesh.mass[vi] * params.gravity);

    for (const Pin& pin : pins) {
        if (pin.vertex_index == vi) {
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
        }
    }

    for (int ti : adj.at(vi)) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, ti);

        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
        const Mat32 F = Ds_mat * Dm_inv;
        const double A = ref_mesh.area[ti];

        CorotatedCache32 cache = buildCorotatedCache(F);

        auto node_g = corotated_node_gradient(cache, F, A, Dm_inv, params.mu, params.lambda);

        for (int a = 0; a < 3; ++a) {
            if (tri_vertex(ref_mesh, ti, a) == vi) {
                g += dt2 * node_g[a];
            }
        }
    }

    return g;
}

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat){
    double r_inf = 0.0;

    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        Vec3 g = compute_local_gradient_no_barrier(i, ref_mesh, adj, pins, params, x, xhat);
        r_inf = std::max(r_inf, std::abs(g.x()));
        r_inf = std::max(r_inf, std::abs(g.y()));
        r_inf = std::max(r_inf, std::abs(g.z()));
    }

    return r_inf;
}

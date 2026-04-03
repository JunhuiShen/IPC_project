#include "physics.h"
#include "make_shape.h"
#include <algorithm>
#include <cmath>
#include <set>

double triangle_ref_area_2d(const RefMesh& ref_mesh, int tri_idx) {
    const Vec2& X0 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 0)];
    const Vec2& X1 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 1)];
    const Vec2& X2 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 2)];
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    return 0.5 * std::abs(Dm_local.determinant());
}

BarrierPairs build_barrier_pairs(const RefMesh& ref_mesh) {
    BarrierPairs pairs;
    const int nv = static_cast<int>(ref_mesh.num_positions);
    const int nt = num_tris(ref_mesh);

    for (int node = 0; node < nv; ++node) {
        for (int ti = 0; ti < nt; ++ti) {
            const int v0 = tri_vertex(ref_mesh, ti, 0);
            const int v1 = tri_vertex(ref_mesh, ti, 1);
            const int v2 = tri_vertex(ref_mesh, ti, 2);
            if (node == v0 || node == v1 || node == v2) continue;
            pairs.nt.push_back({node, {v0, v1, v2}});
        }
    }

    std::set<std::pair<int,int>> unique_edges;
    for (int ti = 0; ti < nt; ++ti) {
        for (int e = 0; e < 3; ++e) {
            int a = tri_vertex(ref_mesh, ti, e);
            int b = tri_vertex(ref_mesh, ti, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            unique_edges.insert({a, b});
        }
    }
    std::vector<std::pair<int,int>> edges(unique_edges.begin(), unique_edges.end());
    const int ne = static_cast<int>(edges.size());
    for (int i = 0; i < ne; ++i) {
        for (int j = i + 1; j < ne; ++j) {
            const int a0 = edges[i].first, a1 = edges[i].second;
            const int b0 = edges[j].first, b1 = edges[j].second;
            if (a0==b0 || a0==b1 || a1==b0 || a1==b1) continue;
            pairs.ss.push_back({{a0, a1, b0, b1}});
        }
    }
    return pairs;
}

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params,
                                                const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    double E = 0.0, PE = 0.0;
    const double dt2 = params.dt2();

    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        E += 0.5 * ref_mesh.mass[i] * (x[i] - xhat[i]).squaredNorm();

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

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                                                     const std::vector<Pin>& pins, const SimParams& params,
                                                                     const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    const double dt2 = params.dt2();
    Vec3  g = Vec3::Zero();
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

    for (const auto& [ti, a] : adj.at(vi)) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, ti);
        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
        const Mat32  F      = Ds_mat * Dm_inv;
        const double A      = ref_mesh.area[ti];

        // Build cache once — P, dPdF, gradN all share it
        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

        g += dt2 * corotated_node_gradient(P, A, gradN, a);
        H += dt2 * corotated_node_hessian(dPdF, A, gradN, a).template block<3, 3>(0, 3 * a);
    }

    return {g, H};
}

Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                            const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                            const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    const double dt2 = params.dt2();
    Vec3 g = Vec3::Zero();

    g += ref_mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-ref_mesh.mass[vi] * params.gravity);

    for (const Pin& pin : pins) {
        if (pin.vertex_index == vi)
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
    }

    for (const auto& [ti, a] : adj.at(vi)) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, ti);
        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
        const Mat32  F      = Ds_mat * Dm_inv;
        const double A      = ref_mesh.area[ti];

        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        g += dt2 * corotated_node_gradient(P, A, gradN, a);
    }

    if (params.d_hat > 0.0) {
        for (const auto& p : nt_pairs) {
            int dof = -1;
            if      (vi == p.node)      dof = 0;
            else if (vi == p.tri_v[0])  dof = 1;
            else if (vi == p.tri_v[1])  dof = 2;
            else if (vi == p.tri_v[2])  dof = 3;
            if (dof < 0) continue;
            g += dt2 * node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
        }

        for (const auto& p : ss_pairs) {
            int dof = -1;
            if      (vi == p.v[0]) dof = 0;
            else if (vi == p.v[1]) dof = 1;
            else if (vi == p.v[2]) dof = 2;
            else if (vi == p.v[3]) dof = 3;
            if (dof < 0) continue;
            g += dt2 * segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
        }
    }

    return g;
}

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    double r_inf = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        Vec3 g = compute_local_gradient(i, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}
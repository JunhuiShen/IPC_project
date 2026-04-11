#include "physics.h"
#include "broad_phase.h"
#include "make_shape.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

double triangle_ref_area_2d(const RefMesh& ref_mesh, int tri_idx) {
    const Vec2& X0 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 0)];
    const Vec2& X1 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 1)];
    const Vec2& X2 = ref_mesh.ref_positions[tri_vertex(ref_mesh, tri_idx, 2)];
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    return 0.5 * std::abs(Dm_local.determinant());
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

    if (params.kB > 0.0) {
        for (const Hinge& h : ref_mesh.hinges) {
            HingeDef def;
            for (int k = 0; k < 4; ++k) def.x[k] = x[h.v[k]];
            PE += bending_energy(def, params.kB, h.c_e, h.bar_theta);
        }
    }

    return E + dt2 * PE;
}

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                                                     const std::vector<Pin>& pins, const SimParams& params,
                                                                     const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                                                     const PinMap* pin_map) {
    const double dt2 = params.dt2();
    Vec3  g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    g += ref_mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-ref_mesh.mass[vi] * params.gravity);
    H += ref_mesh.mass[vi] * Mat33::Identity();

    if (pin_map) {
        const int pi = (*pin_map)[vi];
        if (pi >= 0) {
            const Pin& pin = pins[pi];
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
            H += dt2 * params.kpin * Mat33::Identity();
        }
    } else {
        for (const Pin& pin : pins) {
            if (pin.vertex_index == vi) {
                g += dt2 * params.kpin * (x[vi] - pin.target_position);
                H += dt2 * params.kpin * Mat33::Identity();
                break;
            }
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

        // Build cache once -- P, dPdF, gradN all share it
        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

        g += dt2 * corotated_node_gradient(P, A, gradN, a);
        H += dt2 * corotated_node_hessian(dPdF, A, gradN, a);
    }

    if (params.kB > 0.0) {
        auto it = ref_mesh.hinge_adj.find(vi);
        if (it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : it->second) {
                const Hinge& h = ref_mesh.hinges[hi];
                HingeDef def;
                for (int k = 0; k < 4; ++k) def.x[k] = x[h.v[k]];
                g += dt2 * bending_node_gradient(def, params.kB, h.c_e, h.bar_theta, role);
                H += dt2 * bending_node_hessian_psd(def, params.kB, h.c_e, h.bar_theta, role);
            }
        }
    }

    return {g, H};
}

Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                            const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                            const BroadPhase& broad_phase, const PinMap* pin_map) {
    const double dt2 = params.dt2();
    const auto& bp_cache = broad_phase.cache();
    Vec3 g = Vec3::Zero();

    g += ref_mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-ref_mesh.mass[vi] * params.gravity);

    if (pin_map) {
        const int pi = (*pin_map)[vi];
        if (pi >= 0) {
            g += dt2 * params.kpin * (x[vi] - pins[pi].target_position);
        }
    } else {
        for (const Pin& pin : pins) {
            if (pin.vertex_index == vi) {
                g += dt2 * params.kpin * (x[vi] - pin.target_position);
                break;
            }
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

        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        g += dt2 * corotated_node_gradient(P, A, gradN, a);
    }

    if (params.kB > 0.0) {
        auto it = ref_mesh.hinge_adj.find(vi);
        if (it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : it->second) {
                const Hinge& h = ref_mesh.hinges[hi];
                HingeDef def;
                for (int k = 0; k < 4; ++k) def.x[k] = x[h.v[k]];
                g += dt2 * bending_node_gradient(def, params.kB, h.c_e, h.bar_theta, role);
            }
        }
    }

    if (params.d_hat > 0.0) {
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            g += dt2 * node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            g += dt2 * segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
        }
    }

    return g;
}

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const BroadPhase& broad_phase, const PinMap* pin_map) {
    const int nv = static_cast<int>(x.size());
    double r_inf = 0.0;
    #pragma omp parallel for reduction(max:r_inf) schedule(static)
    for (int i = 0; i < nv; ++i) {
        Vec3 g = compute_local_gradient(i, ref_mesh, adj, pins, params, x, xhat, broad_phase, pin_map);
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

static std::string state_filename(const std::string& dir, int frame) {
    std::ostringstream ss;
    ss << dir << "/state_" << std::setw(4) << std::setfill('0') << frame << ".bin";
    return ss.str();
}

void serialize_state(const std::string& dir, int frame, const DeformedState& state) {
    std::ofstream out(state_filename(dir, frame), std::ios::binary);
    if (!out) { std::cerr << "Error: cannot write state file for frame " << frame << "\n"; return; }

    auto write_vec = [&](const std::vector<Vec3>& v) {
        uint64_t n = v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& p : v) {
            double x = p.x(), y = p.y(), z = p.z();
            out.write(reinterpret_cast<const char*>(&x), sizeof(double));
            out.write(reinterpret_cast<const char*>(&y), sizeof(double));
            out.write(reinterpret_cast<const char*>(&z), sizeof(double));
        }
    };

    write_vec(state.deformed_positions);
    write_vec(state.velocities);
}

bool deserialize_state(const std::string& dir, int frame, DeformedState& state) {
    std::ifstream in(state_filename(dir, frame), std::ios::binary);
    if (!in) { std::cerr << "Error: cannot read state file for frame " << frame << "\n"; return false; }

    auto read_vec = [&](std::vector<Vec3>& v) {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        v.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            double x, y, z;
            in.read(reinterpret_cast<char*>(&x), sizeof(double));
            in.read(reinterpret_cast<char*>(&y), sizeof(double));
            in.read(reinterpret_cast<char*>(&z), sizeof(double));
            v[i] = Vec3(x, y, z);
        }
    };

    read_vec(state.deformed_positions);
    read_vec(state.velocities);
    return in.good();
}

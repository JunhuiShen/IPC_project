#include "physics.h"
#include "broad_phase.h"
#include "mesh_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

// Union of all obstacles
static inline bool sdf_min_evaluation(const SimParams& params, const Vec3& xi, SDFEvaluation& out) {
    bool any = false;
    out.phi = std::numeric_limits<double>::infinity();
    for (const PlaneSDF& p : params.sdf_planes) {
        const SDFEvaluation s = evaluate_sdf(p, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const CylinderSDF& c : params.sdf_cylinders) {
        const SDFEvaluation s = evaluate_sdf(c, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const SphereSDF& sp : params.sdf_spheres) {
        const SDFEvaluation s = evaluate_sdf(sp, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    return any;
}

void build_frozen_residual_workspace(const RefMesh& ref_mesh, const SimParams& params, const std::vector<Vec3>& x, const BroadPhase& broad_phase, FrozenResidualWorkspace& workspace, const std::vector<ShapeGrads>* rest_shape_grads) {
    workspace.mesh = nullptr;
    workspace.params = nullptr;
    workspace.positions = nullptr;
    workspace.broad_phase = nullptr;

    const int num_cloth_triangles = static_cast<int>(ref_mesh.Dm_inverse.size());
    const int num_hinge_slots = static_cast<int>(ref_mesh.hinges.size());
    const int num_hinges = params.kB > 0.0 ? num_hinge_slots : 0;
    const int num_solid_elements = num_tets(ref_mesh);
    const BroadPhase::Cache& broad_phase_cache = broad_phase.cache();
    const bool barrier_enabled = params.d_hat > 0.0 && params.k_barrier != 0.0;
    const int num_nt_pairs = barrier_enabled ? static_cast<int>(broad_phase_cache.nt_pairs.size()) : 0;
    const int num_ss_pairs = barrier_enabled ? static_cast<int>(broad_phase_cache.ss_pairs.size()) : 0;
    const double d_hat2 = params.d_hat * params.d_hat;
    const std::size_t residual_item_count = static_cast<std::size_t>(num_cloth_triangles) + static_cast<std::size_t>(num_hinges) + static_cast<std::size_t>(num_solid_elements) + static_cast<std::size_t>(num_nt_pairs) + static_cast<std::size_t>(num_ss_pairs);
    workspace.cloth_triangle_gradients.resize(static_cast<std::size_t>(num_cloth_triangles));
    workspace.hinge_gradients.resize(static_cast<std::size_t>(num_hinge_slots));
    workspace.tet_gradients.resize(static_cast<std::size_t>(num_solid_elements));
    workspace.nt_gradients.resize(broad_phase_cache.nt_pairs.size());
    workspace.ss_gradients.resize(broad_phase_cache.ss_pairs.size());
    workspace.nt_aabb_active.assign(broad_phase_cache.nt_pairs.size(), 0);
    workspace.ss_aabb_active.assign(broad_phase_cache.ss_pairs.size(), 0);
    workspace.nt_barrier_active.assign(broad_phase_cache.nt_pairs.size(), 0);
    workspace.ss_barrier_active.assign(broad_phase_cache.ss_pairs.size(), 0);
    workspace.nt_gradient_cached.assign(broad_phase_cache.nt_pairs.size(), 0);
    workspace.ss_gradient_cached.assign(broad_phase_cache.ss_pairs.size(), 0);

    const auto build_cloth_triangle_gradient = [&](const int triangle) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, triangle);
        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[static_cast<std::size_t>(triangle)];
        const Mat32 F = Ds_mat * Dm_inv;
        const CorotatedCache32 cache = buildCorotatedCache(F);
        ShapeGrads local_gradN;
        const ShapeGrads* gradN = nullptr;
        if (rest_shape_grads != nullptr) {
            gradN = &(*rest_shape_grads)[static_cast<std::size_t>(triangle)];
        } else {
            local_gradN = shape_function_gradients(Dm_inv);
            gradN = &local_gradN;
        }
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        for (int role = 0; role < 3; ++role) workspace.cloth_triangle_gradients[static_cast<std::size_t>(triangle)][static_cast<std::size_t>(role)] = corotated_node_gradient(P, ref_mesh.area[static_cast<std::size_t>(triangle)], *gradN, role);
    };
    const auto build_hinge_gradient = [&](const int hinge_index) {
        const Hinge& hinge = ref_mesh.hinges[static_cast<std::size_t>(hinge_index)];
        HingeDef def;
        for (int role = 0; role < 4; ++role) def.x[role] = x[static_cast<std::size_t>(hinge.v[role])];
        const BendingCache cache = make_bending_cache(def);
        for (int role = 0; role < 4; ++role) {
            if (params.kB > 0.0 && !cache.degenerate) {
                const double scale = 2.0 * params.kB * hinge.c_e * (cache.theta - hinge.bar_theta);
                workspace.hinge_gradients[static_cast<std::size_t>(hinge_index)][static_cast<std::size_t>(role)] = scale * grad_theta_node(cache, def, role);
            } else {
                workspace.hinge_gradients[static_cast<std::size_t>(hinge_index)][static_cast<std::size_t>(role)] = Vec3::Zero();
            }
        }
    };
    const auto build_tet_gradient = [&](const int element_index) {
        const std::size_t element = static_cast<std::size_t>(element_index);
        const Mat33 F = ElementF(element, x, ref_mesh.tets, ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F, CorotatedCacheMode::Lean);
        const Mat33 first_piola = cache.P(F, params.solid_mu, params.solid_lambda);
        for (int role = 0; role < 4; ++role) workspace.tet_gradients[element][static_cast<std::size_t>(role)] = EFEMElementNodeEnergyGradient(cache, F, ref_mesh.tet_rest_data[element], params.solid_mu, params.solid_lambda, role, &first_piola);
    };
    const auto build_nt_gradient = [&](const int pair_index) {
        const NodeTrianglePair& pair = broad_phase_cache.nt_pairs[static_cast<std::size_t>(pair_index)];
        if (!node_triangle_aabbs_within_distance(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], d_hat2)) return;
        workspace.nt_aabb_active[static_cast<std::size_t>(pair_index)] = 1;
        const NodeTriangleDistanceResult distance = node_triangle_distance(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])]);
        workspace.nt_barrier_active[static_cast<std::size_t>(pair_index)] = distance.distance >= params.d_hat ? 0 : 1;
        if (distance.distance == 0.0) return;
        const double scalar_gradient = scalar_barrier_gradient(distance.distance, params.d_hat);
        for (int role = 0; role < 4; ++role) workspace.nt_gradients[static_cast<std::size_t>(pair_index)][static_cast<std::size_t>(role)] = node_triangle_barrier_gradient(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], params.d_hat, role, 1.0e-12, &distance, &scalar_gradient);
        workspace.nt_gradient_cached[static_cast<std::size_t>(pair_index)] = 1;
    };
    const auto build_ss_gradient = [&](const int pair_index) {
        const SegmentSegmentPair& pair = broad_phase_cache.ss_pairs[static_cast<std::size_t>(pair_index)];
        if (!segment_aabbs_within_distance(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], d_hat2)) return;
        workspace.ss_aabb_active[static_cast<std::size_t>(pair_index)] = 1;
        const SegmentSegmentDistanceResult distance = segment_segment_distance(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])]);
        workspace.ss_barrier_active[static_cast<std::size_t>(pair_index)] = distance.distance >= params.d_hat ? 0 : 1;
        if (distance.distance == 0.0) return;
        const double scalar_gradient = scalar_barrier_gradient(distance.distance, params.d_hat);
        for (int role = 0; role < 4; ++role) workspace.ss_gradients[static_cast<std::size_t>(pair_index)][static_cast<std::size_t>(role)] = segment_segment_barrier_gradient(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], params.d_hat, role, 1.0e-12, &distance, &scalar_gradient);
        workspace.ss_gradient_cached[static_cast<std::size_t>(pair_index)] = 1;
    };

    // A residual build has five independent primitive ranges. Keeping them in
    // one team avoids repeated OpenMP startup, while the item threshold keeps
    // tiny and rigid-only scenes entirely outside the OpenMP runtime.
    if (params.use_parallel && residual_item_count >= 512) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int triangle = 0; triangle < num_cloth_triangles; ++triangle) build_cloth_triangle_gradient(triangle);
            #pragma omp for schedule(static) nowait
            for (int hinge_index = 0; hinge_index < num_hinges; ++hinge_index) build_hinge_gradient(hinge_index);
            #pragma omp for schedule(static) nowait
            for (int element_index = 0; element_index < num_solid_elements; ++element_index) build_tet_gradient(element_index);
            #pragma omp for schedule(static) nowait
            for (int pair_index = 0; pair_index < num_nt_pairs; ++pair_index) build_nt_gradient(pair_index);
            #pragma omp for schedule(static) nowait
            for (int pair_index = 0; pair_index < num_ss_pairs; ++pair_index) build_ss_gradient(pair_index);
        }
    } else {
        for (int triangle = 0; triangle < num_cloth_triangles; ++triangle) build_cloth_triangle_gradient(triangle);
        for (int hinge_index = 0; hinge_index < num_hinges; ++hinge_index) build_hinge_gradient(hinge_index);
        for (int element_index = 0; element_index < num_solid_elements; ++element_index) build_tet_gradient(element_index);
        for (int pair_index = 0; pair_index < num_nt_pairs; ++pair_index) build_nt_gradient(pair_index);
        for (int pair_index = 0; pair_index < num_ss_pairs; ++pair_index) build_ss_gradient(pair_index);
    }

    workspace.mesh = &ref_mesh;
    workspace.params = &params;
    workspace.positions = &x;
    workspace.broad_phase = &broad_phase;
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

    if (params.k_sdf > 0.0) {
        for (int i = 0; i < static_cast<int>(x.size()); ++i) {
            SDFEvaluation s;
            if (sdf_min_evaluation(params, x[i], s))
                PE += sdf_penalty_energy(s, params.k_sdf, params.eps_sdf);
        }
    }

    return E + dt2 * PE;
}

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                                                     const std::vector<Pin>& pins, const SimParams& params,
                                                                     const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                                                     const PinMap* pin_map,
                                                                     const IncidentTriangles* incident_triangles,
                                                                     const std::vector<ShapeGrads>* rest_shape_grads) {
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

    const IncidentTriangles& incident = incident_triangles ? *incident_triangles : adj.at(vi);
    for (const auto& [ti, a] : incident) {
        const TriangleDef def = make_def_triangle(x, ref_mesh, ti);
        Mat32 Ds_mat;
        Ds_mat.col(0) = def.x[1] - def.x[0];
        Ds_mat.col(1) = def.x[2] - def.x[0];
        const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
        const Mat32  F      = Ds_mat * Dm_inv;
        const double A      = ref_mesh.area[ti];

        const CorotatedCache32 cache = buildCorotatedCache(F);
        ShapeGrads local_gradN;
        const ShapeGrads* gradN = nullptr;
        if (rest_shape_grads) {
            gradN = &(*rest_shape_grads)[ti];
        } else {
            local_gradN = shape_function_gradients(Dm_inv);
            gradN = &local_gradN;
        }
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

        g += dt2 * corotated_node_gradient(P, A, *gradN, a);
        H += dt2 * corotated_node_hessian(dPdF, A, *gradN, a);
    }

    if (params.kB > 0.0) {
        auto it = ref_mesh.hinge_adj.find(vi);
        if (it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : it->second) {
                const Hinge& h = ref_mesh.hinges[hi];
                HingeDef def;
                for (int k = 0; k < 4; ++k) def.x[k] = x[h.v[k]];
                auto [bg, bH] = bending_node_gradient_hessian_psd(def, params.kB, h.c_e, h.bar_theta, role);
                g += dt2 * bg;
                H += dt2 * bH;
            }
        }
    }

    if (params.k_sdf > 0.0) {
        SDFEvaluation s;
        if (sdf_min_evaluation(params, x[vi], s)) {
            g += dt2 * sdf_penalty_gradient(s, params.k_sdf, params.eps_sdf);
            H += dt2 * sdf_penalty_hessian (s, params.k_sdf, params.eps_sdf, /*include_curvature=*/false);
        }
    }

    return {g, H};
}

static Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                   const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                   const BroadPhase& broad_phase, const PinMap* pin_map,
                                   const IncidentTriangles* incident_triangles,
                                   const std::vector<ShapeGrads>* rest_shape_grads,
                                   const FrozenResidualWorkspace* frozen_workspace) {
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

    const IncidentTriangles& incident = incident_triangles ? *incident_triangles : adj.at(vi);
    for (const auto& [ti, a] : incident) {
        if (frozen_workspace != nullptr) {
            g += dt2 * frozen_workspace->cloth_triangle_gradients[static_cast<std::size_t>(ti)][static_cast<std::size_t>(a)];
        } else {
            const TriangleDef def = make_def_triangle(x, ref_mesh, ti);
            Mat32 Ds_mat;
            Ds_mat.col(0) = def.x[1] - def.x[0];
            Ds_mat.col(1) = def.x[2] - def.x[0];
            const Mat22& Dm_inv = ref_mesh.Dm_inverse[ti];
            const Mat32 F = Ds_mat * Dm_inv;
            const double A = ref_mesh.area[ti];
            const CorotatedCache32 cache = buildCorotatedCache(F);
            ShapeGrads local_gradN;
            const ShapeGrads* gradN = nullptr;
            if (rest_shape_grads) {
                gradN = &(*rest_shape_grads)[ti];
            } else {
                local_gradN = shape_function_gradients(Dm_inv);
                gradN = &local_gradN;
            }
            const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
            g += dt2 * corotated_node_gradient(P, A, *gradN, a);
        }
    }

    if (params.kB > 0.0) {
        auto it = ref_mesh.hinge_adj.find(vi);
        if (it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : it->second) {
                if (frozen_workspace != nullptr) {
                    g += dt2 * frozen_workspace->hinge_gradients[static_cast<std::size_t>(hi)][static_cast<std::size_t>(role)];
                } else {
                    const Hinge& h = ref_mesh.hinges[hi];
                    HingeDef def;
                    for (int k = 0; k < 4; ++k) def.x[k] = x[h.v[k]];
                    g += dt2 * bending_node_gradient(def, params.kB, h.c_e, h.bar_theta, role);
                }
            }
        }
    }

    if (params.d_hat > 0.0 && params.k_barrier != 0.0) {
        const double dt2k = dt2 * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (frozen_workspace != nullptr && frozen_workspace->nt_aabb_active[entry.pair_index] == 0) {
                continue;
            }
            if (frozen_workspace == nullptr && !node_triangle_aabbs_within_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], d_hat2)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->nt_gradient_cached[entry.pair_index] != 0) {
                g += dt2k * frozen_workspace->nt_gradients[entry.pair_index][static_cast<std::size_t>(entry.dof)];
            } else {
                g += dt2k * node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            }
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (frozen_workspace != nullptr && frozen_workspace->ss_aabb_active[entry.pair_index] == 0) {
                continue;
            }
            if (frozen_workspace == nullptr && !segment_aabbs_within_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], d_hat2)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->ss_gradient_cached[entry.pair_index] != 0) {
                g += dt2k * frozen_workspace->ss_gradients[entry.pair_index][static_cast<std::size_t>(entry.dof)];
            } else {
                g += dt2k * segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            }
        }
    }

    if (params.k_sdf > 0.0) {
        SDFEvaluation s;
        if (sdf_min_evaluation(params, x[vi], s))
            g += dt2 * sdf_penalty_gradient(s, params.k_sdf, params.eps_sdf);
    }

    return g;
}

double compute_global_deformable_residual(
    const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    const std::vector<Pin>& pins, const SimParams& params,
    const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<int>& deformable_nodes, const PinMap* pin_map,
    const std::vector<IncidentTriangles>* incident_triangles,
    const std::vector<ShapeGrads>* rest_shape_grads,
    const FrozenResidualWorkspace* frozen_workspace) {
    (void)params.dt2();
    if (frozen_workspace != nullptr && !frozen_workspace->matches(ref_mesh, params, x, broad_phase)) throw std::invalid_argument("compute_global_deformable_residual: frozen workspace does not match its inputs");
    double r_inf = 0.0;
    const int num_deformable = static_cast<int>(deformable_nodes.size());
    #pragma omp parallel for reduction(max:r_inf) schedule(static)
    for (int i = 0; i < num_deformable; ++i) {
        const int node = deformable_nodes[i];
        Vec3 g = compute_local_gradient(node, ref_mesh, adj, pins, params, x, xhat, broad_phase, pin_map, incident_triangles ? &(*incident_triangles)[node] : nullptr, rest_shape_grads, frozen_workspace);
        const double m = ref_mesh.mass[node];
        if (m > 0.0)
            g /= m;
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const BroadPhase& broad_phase, const PinMap* pin_map,
                               const std::vector<IncidentTriangles>* incident_triangles,
                               const std::vector<ShapeGrads>* rest_shape_grads) {
    const int nv = static_cast<int>(x.size());
    double r_inf = 0.0;
    #pragma omp parallel for reduction(max:r_inf) schedule(static)
    for (int i = 0; i < nv; ++i) {
        Vec3 g = compute_local_gradient(i, ref_mesh, adj, pins, params, x, xhat, broad_phase, pin_map, incident_triangles ? &(*incident_triangles)[i] : nullptr, rest_shape_grads, nullptr);
        const double m = ref_mesh.mass[i];
        if (m > 0.0) g /= m;
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

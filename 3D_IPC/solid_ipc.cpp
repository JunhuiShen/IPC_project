#include "solid_ipc.h"

#include "barrier_energy.h"
#include "broad_phase.h"
#include "mesh.h"
#include "physics.h"
#include "volumetric_corotated_energy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

bool solid_sdf_min_evaluation(
    const SimParams& params,
    const Vec3& x,
    SDFEvaluation& result) {
    bool found = false;
    result.phi = std::numeric_limits<double>::infinity();

    const auto consider = [&](const SDFEvaluation& candidate) {
        if (!found || candidate.phi < result.phi) {
            result = candidate;
            found = true;
        }
    };
    for (const PlaneSDF& plane : params.sdf_planes)
        consider(evaluate_sdf(plane, x));
    for (const CylinderSDF& cylinder : params.sdf_cylinders)
        consider(evaluate_sdf(cylinder, x));
    for (const SphereSDF& sphere : params.sdf_spheres)
        consider(evaluate_sdf(sphere, x));
    return found;
}

bool is_solid_surface_node(const RefMesh& ref_mesh, const int node) {
    return std::find(
               ref_mesh.surface_nodes.begin(),
               ref_mesh.surface_nodes.end(), node)
        != ref_mesh.surface_nodes.end();
}

void EFEMAddNodalMasses(
    const double rho,
    const std::vector<int>& mesh,
    const std::vector<TetRestData>& material_state,
    const std::vector<std::vector<std::pair<int, int>>>& incident_elements,
    const std::vector<int>& nodes,
    std::vector<double>& nodal_mass) {
    if (nodal_mass.size() != incident_elements.size()) {
        throw std::invalid_argument(
            "EFEMAddNodalMasses: nodal_mass is not sized consistently");
    }

    std::vector<double> element_masses(mesh.size(), 0.0);
#pragma omp parallel for
    for (int element = 0;
         element < static_cast<int>(mesh.size() / 4); ++element) {
        const double element_nodal_mass =
            rho * material_state[static_cast<std::size_t>(element)].measure
            / 4.0;
        for (std::size_t local = 0; local < 4; ++local) {
            element_masses[4 * static_cast<std::size_t>(element) + local] =
                element_nodal_mass;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const int node = nodes[static_cast<std::size_t>(i)];
        for (const auto& [element, local] : incident_elements[node]) {
            const std::size_t occurrence =
                4 * static_cast<std::size_t>(element)
                + static_cast<std::size_t>(local);
            nodal_mass[mesh[occurrence]] += element_masses[occurrence];
        }
    }
}

void validate_existing_storage(
    const RefMesh& ref_mesh,
    const DeformedState& state) {
    const std::size_t num_nodes = state.deformed_positions.size();

    if (ref_mesh.num_positions != 0
        && ref_mesh.num_positions != num_nodes) {
        throw std::invalid_argument(
            "create_solid: num_positions does not match particle state");
    }
    if (state.velocities.size() > num_nodes
        || ref_mesh.mass.size() > num_nodes
        || ref_mesh.node_to_rb.size() > num_nodes) {
        throw std::invalid_argument(
            "create_solid: existing nodal arrays have inconsistent sizes");
    }
    if (ref_mesh.tets.size() % 4 != 0
        || ref_mesh.tris.size() % 3 != 0
        || ref_mesh.tet_rest_data.size() != ref_mesh.tets.size() / 4) {
        throw std::invalid_argument(
            "create_solid: existing solid topology is inconsistent");
    }
    if ((!ref_mesh.tets.empty() && ref_mesh.tet_adj.size() != num_nodes)
        || (ref_mesh.tets.empty() && !ref_mesh.tet_adj.empty()
            && ref_mesh.tet_adj.size() != num_nodes)) {
        throw std::invalid_argument(
            "create_solid: existing tet incidence is inconsistent");
    }
    // Cloth elements and tetrahedral boundary faces share `tris`, but only
    // the leading cloth-triangle prefix has shell rest data.  Solids are
    // appended after that prefix, so preserving the existing shell arrays is
    // both valid and required by mixed cloth/solid scenes.
    if (ref_mesh.Dm_inverse.size() != ref_mesh.area.size()
        || ref_mesh.Dm_inverse.size() > ref_mesh.tris.size() / 3) {
        throw std::invalid_argument(
            "create_solid: existing cloth rest data is inconsistent");
    }

    for (const int node : ref_mesh.tet_nodes) {
        if (node < 0 || static_cast<std::size_t>(node) >= num_nodes) {
            throw std::invalid_argument(
                "create_solid: existing tet-node classification is invalid");
        }
    }
    for (const int node : ref_mesh.surface_nodes) {
        if (node < 0 || static_cast<std::size_t>(node) >= num_nodes) {
            throw std::invalid_argument(
                "create_solid: existing surface-node classification is invalid");
        }
    }
}

std::vector<unsigned char> make_solid_node_mask(
    const std::vector<int>& nodes,
    const std::size_t num_nodes) {
    std::vector<unsigned char> mask(num_nodes, 0);
    for (const int node : nodes) {
        if (node >= 0 && static_cast<std::size_t>(node) < num_nodes)
            mask[static_cast<std::size_t>(node)] = 1;
    }
    return mask;
}

bool include_solid_node_triangle_pair(
    const NodeTrianglePair& pair,
    const std::vector<unsigned char>& solid_nodes,
    const std::vector<unsigned char>& surface_nodes) {
    const int nodes[4] = {
        pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
    bool contains_solid_node = false;
    for (const int node : nodes) {
        contains_solid_node = contains_solid_node
            || solid_nodes[static_cast<std::size_t>(node)] != 0;
    }

    // Generic broad phase queries every particle as a point. Explicitly
    // remove the spurious case where that point is an interior tet node.
    const std::size_t point = static_cast<std::size_t>(pair.node);
    if (solid_nodes[point] != 0 && surface_nodes[point] == 0)
        return false;

    // Solid triangle vertices must come from the extracted tet boundary.
    for (const int node : pair.tri_v) {
        const std::size_t index = static_cast<std::size_t>(node);
        if (solid_nodes[index] != 0 && surface_nodes[index] == 0)
            return false;
    }
    return contains_solid_node;
}

bool include_solid_segment_segment_pair(
    const SegmentSegmentPair& pair,
    const std::vector<unsigned char>& solid_nodes,
    const std::vector<unsigned char>& surface_nodes) {
    bool contains_solid_node = false;
    for (const int node : pair.v) {
        const std::size_t index = static_cast<std::size_t>(node);
        if (solid_nodes[index] != 0 && surface_nodes[index] == 0)
            return false;
        contains_solid_node = contains_solid_node
            || solid_nodes[index] != 0;
    }
    return contains_solid_node;
}

Vec3 compute_solid_local_gradient(
    const int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>& solid_nodes,
    const std::vector<unsigned char>& surface_nodes,
    const std::vector<int>* pin_map,
    const FrozenResidualWorkspace* frozen_workspace) {
    const double dt2 = params.dt2();
    const double mass = ref_mesh.mass[static_cast<std::size_t>(node)];
    Vec3 gradient = mass
        * (x[static_cast<std::size_t>(node)]
           - xhat[static_cast<std::size_t>(node)]);
    gradient += dt2 * (-mass * params.gravity);

    if (pin_map != nullptr) {
        const int pin_index = (*pin_map)[static_cast<std::size_t>(node)];
        if (pin_index >= 0) {
            const Pin& pin = pins[static_cast<std::size_t>(pin_index)];
            gradient += dt2 * params.kpin
                * (x[static_cast<std::size_t>(node)]
                   - pin.target_position);
        }
    } else {
        for (const Pin& pin : pins) {
            if (pin.vertex_index == node) {
                gradient += dt2 * params.kpin
                    * (x[static_cast<std::size_t>(node)]
                       - pin.target_position);
            }
        }
    }

    for (const auto& [element_index, local_node] :
         ref_mesh.tet_adj[static_cast<std::size_t>(node)]) {
        const std::size_t element =
            static_cast<std::size_t>(element_index);
        if (frozen_workspace != nullptr) {
            gradient += dt2 * frozen_workspace->tet_gradients[element][static_cast<std::size_t>(local_node)];
        } else {
            const Mat33 F = ElementF(element, x, ref_mesh.tets, ref_mesh.tet_rest_data);
            CorotatedCache cache;
            cache.UpdateCache(F, CorotatedCacheMode::Lean);
            gradient += dt2 * EFEMElementNodeEnergyGradient(cache, F, ref_mesh.tet_rest_data[element], params.solid_mu, params.solid_lambda, local_node);
        }
    }

    if (params.k_sdf > 0.0
        && surface_nodes[static_cast<std::size_t>(node)] != 0) {
        SDFEvaluation sdf;
        if (solid_sdf_min_evaluation(
                params, x[static_cast<std::size_t>(node)], sdf)) {
            gradient += dt2 * sdf_penalty_gradient(
                sdf, params.k_sdf, params.eps_sdf);
        }
    }

    if (params.d_hat > 0.0 && params.k_barrier > 0.0) {
        const BroadPhase::Cache& bp_cache = broad_phase.cache();
        const double barrier_scale = dt2 * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const BroadPhase::Cache::VertexPairEntry& entry :
             bp_cache.vertex_nt[static_cast<std::size_t>(node)]) {
            const NodeTrianglePair& pair =
                bp_cache.nt_pairs[entry.pair_index];
            if (!include_solid_node_triangle_pair(
                    pair, solid_nodes, surface_nodes)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->nt_aabb_active[entry.pair_index] == 0) {
                continue;
            }
            if (frozen_workspace == nullptr && !node_triangle_aabbs_within_distance(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], d_hat2)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->nt_gradient_cached[entry.pair_index] != 0) {
                gradient += barrier_scale * frozen_workspace->nt_gradients[entry.pair_index][static_cast<std::size_t>(entry.dof)];
            } else {
                gradient += barrier_scale * node_triangle_barrier_gradient(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], params.d_hat, entry.dof);
            }
        }

        for (const BroadPhase::Cache::VertexPairEntry& entry :
             bp_cache.vertex_ss[static_cast<std::size_t>(node)]) {
            const SegmentSegmentPair& pair =
                bp_cache.ss_pairs[entry.pair_index];
            if (!include_solid_segment_segment_pair(
                    pair, solid_nodes, surface_nodes)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->ss_aabb_active[entry.pair_index] == 0) {
                continue;
            }
            if (frozen_workspace == nullptr && !segment_aabbs_within_distance(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], d_hat2)) {
                continue;
            }
            if (frozen_workspace != nullptr && frozen_workspace->ss_gradient_cached[entry.pair_index] != 0) {
                gradient += barrier_scale * frozen_workspace->ss_gradients[entry.pair_index][static_cast<std::size_t>(entry.dof)];
            } else {
                gradient += barrier_scale * segment_segment_barrier_gradient(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], params.d_hat, entry.dof);
            }
        }
    }

    return gradient;
}

} // namespace

void create_solid(
    const std::vector<Vec3>& x,
    const std::vector<int>& local_tets,
    const double density,
    RefMesh& ref_mesh,
    DeformedState& state) {
    if (x.empty())
        throw std::invalid_argument("create_solid: x cannot be empty");
    if (local_tets.empty()) {
        throw std::invalid_argument(
            "create_solid: local_tets cannot be empty");
    }
    if (!std::isfinite(density) || density < 0.0) {
        throw std::invalid_argument(
            "create_solid: density must be nonnegative and finite");
    }
    const std::size_t max_int =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (x.size() > max_int || local_tets.size() / 4 > max_int) {
        throw std::overflow_error(
            "create_solid: local mesh indices exceed int range");
    }

    validate_tet_mesh(local_tets, x);
    std::vector<TetRestData> local_rest_data =
        EFEMInitializeElasticMaterialState(x, local_tets);
    std::vector<int> local_boundary_tris =
        compute_boundary_tri_mesh(local_tets);
    std::vector<int> local_surface_nodes =
        compute_boundary_tri_mesh_nodes(local_boundary_tris);

    std::vector<std::vector<std::pair<int, int>>> local_tet_adj(x.size());
    std::vector<int> local_tet_nodes;
    local_tet_nodes.reserve(x.size());
    for (std::size_t element = 0; element < local_tets.size() / 4;
         ++element) {
        for (int local = 0; local < 4; ++local) {
            const int node = local_tets[4 * element + local];
            auto& incident = local_tet_adj[static_cast<std::size_t>(node)];
            if (incident.empty())
                local_tet_nodes.push_back(node);
            incident.emplace_back(static_cast<int>(element), local);
        }
    }
    if (local_tet_nodes.size() != x.size()) {
        throw std::invalid_argument(
            "create_solid: every input particle must belong to a tetrahedron");
    }

    std::vector<double> local_mass(x.size(), 0.0);
    EFEMAddNodalMasses(density, local_tets, local_rest_data, local_tet_adj, local_tet_nodes, local_mass);

    validate_existing_storage(ref_mesh, state);

    const std::size_t node_base = state.deformed_positions.size();
    const std::size_t tet_base = ref_mesh.tets.size() / 4;
    if (node_base > max_int || x.size() > max_int - node_base) {
        throw std::overflow_error(
            "create_solid: global particle indices exceed int range");
    }
    if (tet_base > max_int || local_tets.size() / 4 > max_int - tet_base) {
        throw std::overflow_error(
            "create_solid: global tet indices exceed int range");
    }
    const std::size_t new_num_nodes = node_base + x.size();

    std::vector<Vec3> new_positions = state.deformed_positions;
    new_positions.insert(new_positions.end(), x.begin(), x.end());

    std::vector<Vec3> new_velocities = state.velocities;
    new_velocities.resize(node_base, Vec3::Zero());
    new_velocities.resize(new_num_nodes, Vec3::Zero());

    std::vector<double> new_mass = ref_mesh.mass;
    new_mass.resize(node_base, 0.0);
    new_mass.insert(new_mass.end(), local_mass.begin(), local_mass.end());

    std::vector<int> new_node_to_rb = ref_mesh.node_to_rb;
    new_node_to_rb.resize(node_base, -1);
    new_node_to_rb.resize(new_num_nodes, -1);

    std::vector<int> new_tets = ref_mesh.tets;
    new_tets.reserve(new_tets.size() + local_tets.size());
    for (const int local_node : local_tets) {
        new_tets.push_back(
            static_cast<int>(node_base) + local_node);
    }

    std::vector<int> new_tris = ref_mesh.tris;
    new_tris.reserve(new_tris.size() + local_boundary_tris.size());
    for (const int local_node : local_boundary_tris) {
        new_tris.push_back(
            static_cast<int>(node_base) + local_node);
    }

    std::vector<TetRestData> new_tet_rest_data = ref_mesh.tet_rest_data;
    new_tet_rest_data.insert(
        new_tet_rest_data.end(),
        local_rest_data.begin(), local_rest_data.end());

    std::vector<std::vector<std::pair<int, int>>> new_tet_adj =
        ref_mesh.tet_adj;
    if (new_tet_adj.empty())
        new_tet_adj.resize(node_base);
    new_tet_adj.resize(new_num_nodes);
    for (std::size_t local_node = 0; local_node < local_tet_adj.size();
         ++local_node) {
        auto& global_incident = new_tet_adj[node_base + local_node];
        for (const auto& [element, local] : local_tet_adj[local_node]) {
            global_incident.emplace_back(
                static_cast<int>(tet_base)
                    + element,
                local);
        }
    }

    std::vector<int> new_tet_nodes = ref_mesh.tet_nodes;
    new_tet_nodes.reserve(new_tet_nodes.size() + local_tet_nodes.size());
    for (const int local_node : local_tet_nodes) {
        new_tet_nodes.push_back(
            static_cast<int>(node_base) + local_node);
    }

    std::vector<int> new_surface_nodes = ref_mesh.surface_nodes;
    new_surface_nodes.reserve(
        new_surface_nodes.size() + local_surface_nodes.size());
    for (const int local_node : local_surface_nodes) {
        new_surface_nodes.push_back(
            static_cast<int>(node_base) + local_node);
    }

    state.deformed_positions.swap(new_positions);
    state.velocities.swap(new_velocities);
    ref_mesh.mass.swap(new_mass);
    ref_mesh.node_to_rb.swap(new_node_to_rb);
    ref_mesh.tets.swap(new_tets);
    ref_mesh.tris.swap(new_tris);
    ref_mesh.tet_rest_data.swap(new_tet_rest_data);
    ref_mesh.tet_adj.swap(new_tet_adj);
    ref_mesh.tet_nodes.swap(new_tet_nodes);
    ref_mesh.surface_nodes.swap(new_surface_nodes);
    ref_mesh.num_positions = new_num_nodes;
}

double compute_solid_incremental_potential_no_barrier(
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat) {
    const double dt2 = params.dt2();
    double energy = 0.0;
    double potential_energy = 0.0;

    for (const int node : ref_mesh.tet_nodes) {
        energy += 0.5 * ref_mesh.mass[static_cast<std::size_t>(node)]
            * (x[static_cast<std::size_t>(node)]
               - xhat[static_cast<std::size_t>(node)])
                  .squaredNorm();
        potential_energy +=
            -ref_mesh.mass[static_cast<std::size_t>(node)]
            * params.gravity.dot(x[static_cast<std::size_t>(node)]);
    }

    for (const Pin& pin : pins) {
        const Vec3 dx = x[static_cast<std::size_t>(pin.vertex_index)]
            - pin.target_position;
        potential_energy += 0.5 * params.kpin * dx.squaredNorm();
    }

    for (std::size_t element = 0;
         element < ref_mesh.tets.size() / 4; ++element) {
        const Mat33 F = ElementF(element, x, ref_mesh.tets, ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F);
        potential_energy += EFEMElementInternalEnergy(
            cache, F, ref_mesh.tet_rest_data[element],
            params.solid_mu, params.solid_lambda);
    }

    if (params.k_sdf > 0.0) {
        for (const int node : ref_mesh.surface_nodes) {
            SDFEvaluation sdf;
            if (solid_sdf_min_evaluation(
                    params, x[static_cast<std::size_t>(node)], sdf)) {
                potential_energy += sdf_penalty_energy(
                    sdf, params.k_sdf, params.eps_sdf);
            }
        }
    }

    return energy + dt2 * potential_energy;
}

std::pair<Vec3, Mat33> compute_solid_local_gradient_and_pbgs_block_no_barrier(
    const int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const std::vector<unsigned char>* surface_node_mask,
    const std::vector<int>* pin_map) {
    const double dt2 = params.dt2();
    const double mass = ref_mesh.mass[static_cast<std::size_t>(node)];
    Vec3 gradient = mass
        * (x[static_cast<std::size_t>(node)]
           - xhat[static_cast<std::size_t>(node)]);
    gradient += dt2 * (-mass * params.gravity);
    Mat33 pbgs_block = mass * Mat33::Identity();

    if (pin_map != nullptr) {
        const int pin_index = (*pin_map)[static_cast<std::size_t>(node)];
        if (pin_index >= 0) {
            const Pin& pin = pins[static_cast<std::size_t>(pin_index)];
            gradient += dt2 * params.kpin
                * (x[static_cast<std::size_t>(node)] - pin.target_position);
            pbgs_block += dt2 * params.kpin * Mat33::Identity();
        }
    } else {
        for (const Pin& pin : pins) {
            if (pin.vertex_index != node)
                continue;
            gradient += dt2 * params.kpin
                * (x[static_cast<std::size_t>(node)] - pin.target_position);
            pbgs_block += dt2 * params.kpin * Mat33::Identity();
        }
    }

    for (const auto& [element_index, local_node] :
         ref_mesh.tet_adj[static_cast<std::size_t>(node)]) {
        const std::size_t element =
            static_cast<std::size_t>(element_index);
        const Mat33 F = ElementF(element, x, ref_mesh.tets, ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F, CorotatedCacheMode::Lean);
        const auto [element_gradient, element_block] = EFEMElementNodeGradientAndPBGSBlock(cache, F, ref_mesh.tet_rest_data[element], params.solid_mu, params.solid_lambda, local_node);
        gradient += dt2 * element_gradient;
        pbgs_block += dt2 * element_block;
    }

    const bool is_surface = surface_node_mask != nullptr
        ? (*surface_node_mask)[static_cast<std::size_t>(node)] != 0
        : is_solid_surface_node(ref_mesh, node);
    if (params.k_sdf > 0.0 && is_surface) {
        SDFEvaluation sdf;
        if (solid_sdf_min_evaluation(params, x[static_cast<std::size_t>(node)], sdf)) {
            gradient += dt2 * sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
            pbgs_block += dt2 * sdf_penalty_hessian(sdf, params.k_sdf, params.eps_sdf, /*include_curvature=*/false);
        }
    }

    return {gradient, pbgs_block};
}

std::pair<Vec3, Mat33> compute_solid_local_barrier_gradient_and_self_hessian(
    const int node,
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>* solid_node_mask,
    const std::vector<unsigned char>* surface_node_mask) {
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return {Vec3::Zero(), Mat33::Zero()};
    if ((solid_node_mask == nullptr) != (surface_node_mask == nullptr))
        throw std::invalid_argument("compute_solid_local_barrier_gradient_and_self_hessian: both node masks must be supplied together");

    std::vector<unsigned char> owned_solid_node_mask;
    std::vector<unsigned char> owned_surface_node_mask;
    if (solid_node_mask == nullptr) {
        owned_solid_node_mask = make_solid_node_mask(ref_mesh.tet_nodes, x.size());
        owned_surface_node_mask = make_solid_node_mask(ref_mesh.surface_nodes, x.size());
        solid_node_mask = &owned_solid_node_mask;
        surface_node_mask = &owned_surface_node_mask;
    }

    const BroadPhase::Cache& cache = broad_phase.cache();
    const double barrier_scale = params.dt2() * params.k_barrier;
    const double d_hat2 = params.d_hat * params.d_hat;
    Vec3 gradient = Vec3::Zero();
    Mat33 self_hessian = Mat33::Zero();

    for (const BroadPhase::Cache::VertexPairEntry& entry :
         cache.vertex_nt[static_cast<std::size_t>(node)]) {
        const NodeTrianglePair& pair = cache.nt_pairs[entry.pair_index];
        if (!include_solid_node_triangle_pair(pair, *solid_node_mask, *surface_node_mask)) {
            continue;
        }
        if (!node_triangle_aabbs_within_distance(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], d_hat2)) {
            continue;
        }
        const auto [pair_gradient, pair_self_hessian] = node_triangle_barrier_self_gradient_and_hessian(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], params.d_hat, entry.dof);
        gradient += barrier_scale * pair_gradient;
        self_hessian += barrier_scale * pair_self_hessian;
    }

    for (const BroadPhase::Cache::VertexPairEntry& entry :
         cache.vertex_ss[static_cast<std::size_t>(node)]) {
        const SegmentSegmentPair& pair = cache.ss_pairs[entry.pair_index];
        if (!include_solid_segment_segment_pair(pair, *solid_node_mask, *surface_node_mask)) {
            continue;
        }
        if (!segment_aabbs_within_distance(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], d_hat2)) {
            continue;
        }
        const auto [pair_gradient, pair_self_hessian] = segment_segment_barrier_self_gradient_and_hessian(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], params.d_hat, entry.dof);
        gradient += barrier_scale * pair_gradient;
        self_hessian += barrier_scale * pair_self_hessian;
    }

    return {gradient, self_hessian};
}

double compute_solid_barrier_incremental_potential(
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const BroadPhase& broad_phase) {
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return 0.0;

    const std::vector<unsigned char> solid_nodes = make_solid_node_mask(ref_mesh.tet_nodes, x.size());
    const std::vector<unsigned char> surface_nodes = make_solid_node_mask(ref_mesh.surface_nodes, x.size());
    const BroadPhase::Cache& cache = broad_phase.cache();
    const double barrier_scale = params.dt2() * params.k_barrier;
    const double d_hat2 = params.d_hat * params.d_hat;
    double energy = 0.0;

    // Pair arrays contain every contact once. Summing the per-vertex cache
    // rows here would count every four-vertex contact four times.
    for (const NodeTrianglePair& pair : cache.nt_pairs) {
        if (!include_solid_node_triangle_pair(pair, solid_nodes, surface_nodes)) {
            continue;
        }
        if (!node_triangle_aabbs_within_distance(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], d_hat2)) {
            continue;
        }
        energy += barrier_scale * node_triangle_barrier(x[static_cast<std::size_t>(pair.node)], x[static_cast<std::size_t>(pair.tri_v[0])], x[static_cast<std::size_t>(pair.tri_v[1])], x[static_cast<std::size_t>(pair.tri_v[2])], params.d_hat);
    }

    for (const SegmentSegmentPair& pair : cache.ss_pairs) {
        if (!include_solid_segment_segment_pair(pair, solid_nodes, surface_nodes)) {
            continue;
        }
        if (!segment_aabbs_within_distance(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], d_hat2)) {
            continue;
        }
        energy += barrier_scale * segment_segment_barrier(x[static_cast<std::size_t>(pair.v[0])], x[static_cast<std::size_t>(pair.v[1])], x[static_cast<std::size_t>(pair.v[2])], x[static_cast<std::size_t>(pair.v[3])], params.d_hat);
    }
    return energy;
}

double compute_solid_incremental_potential(
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase) {
    return compute_solid_incremental_potential_no_barrier(ref_mesh, pins, params, x, xhat) + compute_solid_barrier_incremental_potential(ref_mesh, params, x, broad_phase);
}

std::pair<Vec3, Mat33> compute_solid_local_gradient_and_block(
    const int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>* solid_node_mask,
    const std::vector<unsigned char>* surface_node_mask,
    const std::vector<int>* pin_map) {
    if ((solid_node_mask == nullptr) != (surface_node_mask == nullptr))
        throw std::invalid_argument("compute_solid_local_gradient_and_block: both node masks must be supplied together");

    auto [gradient, block] = compute_solid_local_gradient_and_pbgs_block_no_barrier(node, ref_mesh, pins, params, x, xhat, surface_node_mask, pin_map);
    const auto [barrier_gradient, barrier_self_hessian] = compute_solid_local_barrier_gradient_and_self_hessian(node, ref_mesh, params, x, broad_phase, solid_node_mask, surface_node_mask);
    gradient += barrier_gradient;
    block += barrier_self_hessian;
    return {gradient, block};
}

double compute_global_solid_residual(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const BroadPhase& broad_phase, const std::vector<int>* pin_map, const std::vector<unsigned char>* solid_node_mask, const std::vector<unsigned char>* surface_node_mask, const FrozenResidualWorkspace* frozen_workspace) {
    (void)params.dt2();
    if (frozen_workspace != nullptr && !frozen_workspace->matches(ref_mesh, params, x, broad_phase)) throw std::invalid_argument("compute_global_solid_residual: frozen workspace does not match its inputs");
    if ((solid_node_mask == nullptr) != (surface_node_mask == nullptr)) {
        throw std::invalid_argument("compute_global_solid_residual: both node masks must be supplied together");
    }

    std::vector<unsigned char> owned_solid_node_mask;
    std::vector<unsigned char> owned_surface_node_mask;
    if (solid_node_mask == nullptr) {
        owned_solid_node_mask = make_solid_node_mask(ref_mesh.tet_nodes, x.size());
        owned_surface_node_mask = make_solid_node_mask(ref_mesh.surface_nodes, x.size());
        solid_node_mask = &owned_solid_node_mask;
        surface_node_mask = &owned_surface_node_mask;
    }

    double r_inf = 0.0;
    const int num_solid_nodes = static_cast<int>(ref_mesh.tet_nodes.size());
    #pragma omp parallel for reduction(max:r_inf) schedule(static)
    for (int i = 0; i < num_solid_nodes; ++i) {
        const int node = ref_mesh.tet_nodes[static_cast<std::size_t>(i)];
        Vec3 gradient = compute_solid_local_gradient(node, ref_mesh, pins, params, x, xhat, broad_phase, *solid_node_mask, *surface_node_mask, pin_map, frozen_workspace);
        const double mass = ref_mesh.mass[static_cast<std::size_t>(node)];
        if (mass > 0.0)
            gradient /= mass;
        r_inf = std::max(r_inf, gradient.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

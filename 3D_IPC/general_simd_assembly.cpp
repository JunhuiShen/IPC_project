#include "general_simd_assembly.h"
#include "contact_scheduling.h"
#include "general_simd_solid.h"
#include "general_simd_cloth.h"
#include "general_simd_contact.h"
#include "barrier_energy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace solver_detail {

void prepare_general_simd_batch(
    const int* nodes, std::size_t count, const RefMesh& mesh,
    const std::vector<IncidentTriangles>& incident,
    const std::vector<ShapeGrads>& shape_gradients,
    const std::vector<Pin>& pins, const PinMap& pin_map,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& predicted, const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask,
    GeneralSimdVertexSystem* outputs,
    const GeneralSimdMaterials* materials) {
    constexpr std::size_t width = ipc_simd::tile_width;
    assert(count <= width);
    const double dt2 = params.dt2();
    std::array<ipc_simd::PointInput, width> points;
    std::array<Vec3, width> gradients;
    std::array<Mat33, width> hessians;
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes[i];
        points[i].mass = mesh.mass[node];
        points[i].position = positions[node];
        points[i].predicted_position = predicted[node];
        if (pin_map[node] >= 0) points[i].pin_target = pins[pin_map[node]].target_position;
        outputs[i] = GeneralSimdVertexSystem{};
    }
    ipc_simd::point_derivatives_tile(points.data(), count, params.gravity,
        params.kpin, dt2, gradients.data(), hessians.data());
    for (std::size_t i = 0; i < count; ++i) {
        outputs[i].gradient = gradients[i];
        outputs[i].hessian = hessians[i];
    }

    // Pack across complete independent nodes, retaining every node's incident
    // element order. Distribute each tile before reusing its private storage.
    std::array<Vec3, 4 * width> gathered;
    std::array<std::size_t, width> owners;
    std::array<double, width> measures;
    std::size_t pending = 0;
    const auto distribute = [&] {
        for (std::size_t e = 0; e < pending; ++e) {
            outputs[owners[e]].gradient += dt2 * gradients[e];
            outputs[owners[e]].hessian += dt2 * hessians[e];
        }
        pending = 0;
    };
    if (materials) {
        // Resolve rest data and topology once between solver calls. Only live
        // positions are gathered here. Singleton batches can pass a contiguous
        // incident span straight to the kernel without copying static values.
        const auto gather_positions = [&](const auto& records, std::size_t begin,
                                          std::size_t size, std::size_t offset) {
            for (std::size_t e = 0; e < size; ++e) {
                const auto& corners = records.nodes[begin + e];
                for (std::size_t corner = 0; corner < corners.size(); ++corner)
                    gathered[corners.size() * (offset + e) + corner] = positions[corners[corner]];
            }
        };
        const auto assemble_elements = [&](const auto& records, const auto& kernel) {
            using Matrix = typename std::decay_t<decltype(records.dm_inverse)>::value_type;
            using Shape = typename std::decay_t<decltype(records.shape_gradients)>::value_type;
            if (count == 1) {
                const auto end = records.node_offsets[nodes[0] + 1];
                for (auto e = records.node_offsets[nodes[0]]; e < end; e += width) {
                    pending = std::min(width, end - e);
                    gather_positions(records, e, pending, 0);
                    std::fill_n(owners.data(), pending, std::size_t(0));
                    kernel(records.dm_inverse.data() + e, records.measures.data() + e,
                        records.shape_gradients.data() + e);
                    distribute();
                }
                return;
            }
            std::array<Matrix, width> inverse;
            std::array<Shape, width> shapes;
            const auto flush = [&] {
                if (!pending) return;
                kernel(inverse.data(), measures.data(), shapes.data());
                distribute();
            };
            for (std::size_t i = 0; i < count; ++i) {
                auto e = records.node_offsets[nodes[i]];
                const auto end = records.node_offsets[nodes[i] + 1];
                while (e < end) {
                    const auto size = std::min(width - pending, end - e);
                    gather_positions(records, e, size, pending);
                    std::copy_n(records.dm_inverse.data() + e, size, inverse.data() + pending);
                    std::copy_n(records.measures.data() + e, size, measures.data() + pending);
                    std::copy_n(records.shape_gradients.data() + e, size, shapes.data() + pending);
                    std::fill_n(owners.data() + pending, size, i);
                    e += size; pending += size;
                    if (pending == width) flush();
                }
            }
            flush();
        };
        assemble_elements(materials->triangles, [&](const Mat22* inverse,
                                                   const double* area, const Vec2* shape) {
            ipc_simd::general_corotated_derivatives_tile(gathered.data(), inverse, area,
                shape, pending, params.mu, params.lambda, gradients.data(), hessians.data());
        });
        assemble_elements(materials->tets, [&](const Mat33* inverse,
                                              const double* measure, const Vec3* shape) {
            ipc_simd::solid_derivatives_tile(gathered.data(), inverse, measure,
                shape, pending, params.solid_mu, params.solid_lambda, gradients.data(), hessians.data());
        });
        if (params.kB > 0.0) {
            const auto& records = materials->hinges;
            if (count == 1) {
                const auto end = records.node_offsets[nodes[0] + 1];
                for (auto e = records.node_offsets[nodes[0]]; e < end; e += width) {
                    pending = std::min(width, end - e);
                    gather_positions(records, e, pending, 0);
                    std::fill_n(owners.data(), pending, std::size_t(0));
                    ipc_simd::bending_derivatives_tile(gathered.data(),
                        records.active_nodes.data() + e, records.coefficients.data() + e,
                        records.rest_angles.data() + e, pending, params.kB,
                        gradients.data(), hessians.data());
                    distribute();
                }
            } else {
                std::array<int, width> roles;
                std::array<double, width> angles;
                const auto flush = [&] {
                    if (!pending) return;
                    ipc_simd::bending_derivatives_tile(gathered.data(), roles.data(),
                        measures.data(), angles.data(), pending, params.kB,
                        gradients.data(), hessians.data());
                    distribute();
                };
                for (std::size_t i = 0; i < count; ++i) {
                    auto e = records.node_offsets[nodes[i]];
                    const auto end = records.node_offsets[nodes[i] + 1];
                    while (e < end) {
                        const auto size = std::min(width - pending, end - e);
                        gather_positions(records, e, size, pending);
                        std::copy_n(records.active_nodes.data() + e, size, roles.data() + pending);
                        std::copy_n(records.coefficients.data() + e, size, measures.data() + pending);
                        std::copy_n(records.rest_angles.data() + e, size, angles.data() + pending);
                        std::fill_n(owners.data() + pending, size, i);
                        e += size; pending += size;
                        if (pending == width) flush();
                    }
                }
                flush();
            }
        }
    } else {
        std::array<Mat22, width> triangle_dm;
        std::array<Vec2, width> triangle_shapes;
        const auto flush_triangles = [&] {
            if (!pending) return;
            ipc_simd::general_corotated_derivatives_tile(gathered.data(), triangle_dm.data(),
                measures.data(), triangle_shapes.data(), pending, params.mu,
                params.lambda, gradients.data(), hessians.data());
            distribute();
        };
        for (std::size_t i = 0; i < count; ++i) {
            if (solid_mask[nodes[i]]) continue;
            for (const auto& [triangle, corner] : incident[nodes[i]]) {
                if (triangle < 0 || static_cast<std::size_t>(triangle) >= mesh.Dm_inverse.size()) continue;
                for (int j = 0; j < 3; ++j)
                    gathered[3 * pending + j] = positions[mesh.tris[3 * triangle + j]];
                triangle_dm[pending] = mesh.Dm_inverse[triangle];
                triangle_shapes[pending] = shape_gradients[triangle][corner];
                measures[pending] = mesh.area[triangle];
                owners[pending++] = i;
                if (pending == width) flush_triangles();
            }
        }
        flush_triangles();

        std::array<Mat33, width> tet_dm;
        std::array<Vec3, width> tet_shapes;
        const auto flush_tets = [&] {
            if (!pending) return;
            ipc_simd::solid_derivatives_tile(gathered.data(), tet_dm.data(),
                measures.data(), tet_shapes.data(), pending, params.solid_mu,
                params.solid_lambda, gradients.data(), hessians.data());
            distribute();
        };
        for (std::size_t i = 0; i < count; ++i) {
            if (!solid_mask[nodes[i]]) continue;
            for (const auto& [tet, corner] : mesh.tet_adj[nodes[i]]) {
                const auto& rest = mesh.tet_rest_data[tet];
                for (int j = 0; j < 4; ++j)
                    gathered[4 * pending + j] = positions[mesh.tets[4 * tet + j]];
                tet_dm[pending] = rest.Dm_inverse;
                tet_shapes[pending] = rest.grad_N[corner];
                measures[pending] = rest.measure;
                owners[pending++] = i;
                if (pending == width) flush_tets();
            }
        }
        flush_tets();

        if (params.kB > 0.0) {
            std::array<int, width> roles;
            std::array<double, width> angles;
            const auto flush_hinges = [&] {
                if (!pending) return;
                ipc_simd::bending_derivatives_tile(gathered.data(), roles.data(),
                    measures.data(), angles.data(), pending, params.kB,
                    gradients.data(), hessians.data());
                distribute();
            };
            for (std::size_t i = 0; i < count; ++i) {
                if (solid_mask[nodes[i]]) continue;
                const auto found = mesh.hinge_adj.find(nodes[i]);
                if (found == mesh.hinge_adj.end()) continue;
                for (const auto& [hinge, role] : found->second) {
                    const auto& value = mesh.hinges[hinge];
                    for (int j = 0; j < 4; ++j)
                        gathered[4 * pending + j] = positions[value.v[j]];
                    roles[pending] = role;
                    measures[pending] = value.c_e;
                    angles[pending] = value.bar_theta;
                    owners[pending++] = i;
                    if (pending == width) flush_hinges();
                }
            }
            flush_hinges();
        }
    }

    if (params.k_sdf > 0.0 && (!params.sdf_planes.empty()
        || !params.sdf_cylinders.empty() || !params.sdf_spheres.empty())) {
        std::array<Vec3, width> current, prior;
        std::array<physics_detail::SdfDerivatives, width> values;
        std::size_t active = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const int node = nodes[i];
            if (solid_mask[node] && !surface_mask[node]) continue;
            owners[active] = i;
            current[active] = positions[node];
            if (params.friction_coefficient > 0.0) prior[active] = (*previous)[node];
            ++active;
        }
        physics_detail::compute_sdf_derivatives_tile(params, current.data(),
            params.friction_coefficient > 0.0 ? prior.data() : nullptr,
            active, values.data());
        for (std::size_t e = 0; e < active; ++e) {
            auto& output = outputs[owners[e]];
            output.gradient += dt2 * values[e].gradient;
            output.hessian += dt2 * values[e].hessian;
            if (solid_mask[nodes[owners[e]]]) {
                // Solid reference adds SDF friction after mesh friction.
                output.sdf_friction_gradient = values[e].friction_gradient;
                output.sdf_friction_hessian = values[e].friction_hessian;
            } else if (params.friction_coefficient > 0.0) {
                output.gradient += values[e].friction_gradient;
                output.hessian += values[e].friction_hessian;
            }
        }
    }
}

void accumulate_general_simd_contacts(
    int node, bool solid, const BroadPhase::Cache& cache,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask, bool cooperative,
    GeneralSimdVertexSystem& output,
    safe_step_detail::VertexAabbRejections* rejections,
    const std::function<void()>* leader_work) {
    if (rejections) rejections->distance = 0.0;
    const double scale = params.dt2() * params.k_barrier;
    GeneralSimdVertexSystem contact;
    // Cloth v2 evaluates positive-distance candidates even with a zero barrier
    // coefficient. Solids retain their original positive-stiffness condition.
    if (params.d_hat > 0.0 && (!solid || params.k_barrier > 0.0)) {
        constexpr std::size_t width = ipc_simd::contact_tile_width;
        const auto& nt = cache.vertex_nt[node];
        const auto& ss = cache.vertex_ss[node];
        const std::size_t total = nt.size() + ss.size();
        if (rejections) {
            rejections->clear.assign(total, 0);
            rejections->distance = params.d_hat;
        }
        const bool use_derivative_mask = std::isfinite(scale);
        const auto gather = [&](std::size_t index, ipc_simd::MeshContactInput& input) {
            std::array<int, 4> nodes;
            const bool segment = index >= nt.size();
            const auto& entry = segment ? ss[index - nt.size()] : nt[index];
            if (segment) {
                const auto& pair = cache.ss_pairs[entry.pair_index];
                nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
            } else {
                const auto& pair = cache.nt_pairs[entry.pair_index];
                nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            }
            if (solid && !cache.excludes_tet_interior_nt_queries) {
                for (int n : nodes)
                    if (solid_mask[n] && !surface_mask[n]) return false;
            }
            bool clear = false;
            const double d2 = params.d_hat * params.d_hat;
            const bool active = segment
                ? segment_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]],
                    positions[nodes[2]], positions[nodes[3]], d2, &clear)
                : node_triangle_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]],
                    positions[nodes[2]], positions[nodes[3]], d2, &clear);
            // Only the conservative distance certificate may suppress CCD.
            // A derivative mask, an interior filter or a plane-side rejection
            // alone does not prove that the proposed step is collision-free.
            if (rejections) rejections->clear[index] = clear;
            if (!active) return false;
            input.role = entry.dof;
            input.segment_segment = segment;
            for (int j = 0; j < 4; ++j) {
                input.positions[j] = positions[nodes[j]];
                if (params.friction_coefficient != 0.0)
                    input.previous_positions[j] = (*previous)[nodes[j]];
            }
            return true;
        };
        const auto accumulate_value = [&](const ipc_simd::MeshContactOutput& value) {
            auto& target = solid ? contact : output;
            target.gradient += scale * value.gradient;
            target.hessian += scale * value.hessian;
            if (params.friction_coefficient != 0.0) {
                if (solid) {
                    contact.sdf_friction_gradient += value.friction_gradient;
                    contact.sdf_friction_hessian += value.friction_hessian;
                } else {
                    output.gradient += value.friction_gradient;
                    output.hessian += value.friction_hessian;
                }
            }
        };
        if (cooperative && total >= 32 && omp_get_num_threads() > 1) {
            struct Tile {
                std::array<ipc_simd::MeshContactOutput, width> values;
                std::array<unsigned char, width> active{};
                std::size_t count = 0;
                Tile() {
                    // Returned packets may copy unused slots too.
                    for (auto& value : values) {
                        value.gradient.setZero(); value.hessian.setZero();
                        value.friction_gradient.setZero(); value.friction_hessian.setZero();
                    }
                }
            };
            const auto evaluate = [&](int batch) {
                Tile tile;
                std::array<ipc_simd::MeshContactInput, width> inputs;
                const auto end = std::min(total, (batch + 1) * width);
                for (std::size_t index = batch * width; index < end; ++index)
                    if (gather(index, inputs[tile.count])) ++tile.count;
            ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), tile.count,
                    params.d_hat, params.k_barrier, params.friction_coefficient,
                    params.dt(), params.friction_velocity_epsilon, tile.values.data(),
                    use_derivative_mask ? tile.active.data() : nullptr);
                return tile;
            };
            const auto accumulate = [&](const Tile& tile) {
                for (std::size_t e = 0; e < tile.count; ++e)
                    if (!use_derivative_mask || tile.active[e]) accumulate_value(tile.values[e]);
            };
            const int batches = static_cast<int>((total + width - 1) / width);
            // As in basic v2's split-contact sweep, the leader may prepare
            // local point/elastic/SDF terms while helpers evaluate contacts.
            // Both read the same fixed positions; reduction starts after join.
            parallel_contact_tasks(batches, evaluate, accumulate, leader_work);
        } else {
            if (leader_work) (*leader_work)();
            std::array<ipc_simd::MeshContactInput, width> inputs;
            std::array<ipc_simd::MeshContactOutput, width> values;
            std::array<unsigned char, width> active;
            std::size_t pending = 0;
            const auto flush = [&] {
                if (!pending) return;
                ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), pending,
                    params.d_hat, params.k_barrier, params.friction_coefficient,
                    params.dt(), params.friction_velocity_epsilon, values.data(),
                    use_derivative_mask ? active.data() : nullptr);
                for (std::size_t e = 0; e < pending; ++e)
                    if (!use_derivative_mask || active[e]) accumulate_value(values[e]);
                pending = 0;
            };
            // Rejected candidates do not consume lanes or force a partial
            // packet. Match basic v2's full-row active-contact compaction.
            for (std::size_t index = 0; index < total; ++index)
                if (gather(index, inputs[pending]) && ++pending == width) flush();
            flush();
        }
    } else if (leader_work) {
        (*leader_work)();
    }
    if (solid) {
        output.gradient += contact.gradient;
        output.hessian += contact.hessian;
        if (params.friction_coefficient != 0.0) {
            output.gradient += contact.sdf_friction_gradient;
            output.hessian += contact.sdf_friction_hessian;
            output.gradient += output.sdf_friction_gradient;
            output.hessian += output.sdf_friction_hessian;
        }
    }
}

} // namespace solver_detail

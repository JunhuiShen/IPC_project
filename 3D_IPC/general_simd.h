#pragma once

#include "SIMD.h"
#include "barrier_energy.h"
#include "broad_phase.h"
#include "contact_scheduling.h"
#include "physics.h"
#include "safe_step.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <vector>

// Membrane kernels.
namespace ipc_simd {

// The general membrane variant preserves the original scalar/Eigen rounding
// boundaries while retaining independent-element SIMD lanes. Bending uses
// bending_derivatives_tile from SIMD.h, shared with cloth experimental v2.
void general_corotated_derivatives_tile(
    const Vec3* positions, const Mat22* dm_inverse, const double* areas,
    const Vec2* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* hessians);

} // namespace ipc_simd

// Mesh contact kernels.
namespace ipc_simd {

// General-solver contact arithmetic follows the original general reference's
// rounding order. The existing cloth-v2 entry point keeps its own arithmetic.
void general_mesh_contact_derivatives_tile(const MeshContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, MeshContactOutput* outputs,
    unsigned char* derivative_active = nullptr);

} // namespace ipc_simd

// Solid kernels.
namespace ipc_simd {

// One gathered AoS record represents one tetrahedron and its active node.
// positions contains four consecutive vertices per record, and shape_gradients
// contains that node's TetRestData::grad_N. Only these local records are read.
// count must be at most tile_width; zero permits null input/output pointers.
// Outputs include rest measure, but not dt^2. Blocks are the scalar solid
// solver's TGSL PSD PBGS approximation, not exact energy Hessians. The caller
// owns ordered accumulation into its private per-node systems.
void solid_derivatives_tile(
    const Vec3* positions, const Mat33* dm_inverse, const double* measures,
    const Vec3* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* pbgs_blocks);

} // namespace ipc_simd

// Rigid contact kernels.
namespace ipc_simd {

// An immutable contact snapshot. Barrier derivatives include every role on
// side; body_role_mask identifies the actual owned roles for frozen friction.
// A zero mask selects the same roles as side. Kinematics is required for
// orientation modes, with second derivatives for an orientation Hessian.
struct RigidContactInput {
    std::array<Vec3, 4> positions;
    std::array<Vec3, 4> previous_positions;
    std::array<Vec3, 4> body_references;
    bool segment_segment = false;
    bool exact_computation_fallback = true;
    RigidBarrierSide side = RigidBarrierSide::FirstPrimitive;
    const QuaternionOmegaKinematics* kinematics = nullptr;
    RigidBodyUpdateMode update_mode = RigidBodyUpdateMode::TranslationAndOrientation;
    unsigned body_role_mask = 0;
};

struct RigidContactOutput {
    RigidEnergyDerivatives barrier;
    RigidEnergyDerivatives friction;
};

// Scalar closest-feature selection and exact Cartesian cross-Hessian
// preparation, followed by SIMD coupled rigid pullback and frozen friction.
// Independent contacts occupy lanes. No mesh lookup or shared accumulation is
// performed here. Barrier is unscaled; friction already includes dt^2.
void rigid_contact_derivatives_tile(const RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidContactOutput* outputs);

// Gather contacts in NT-then-SS order from one fixed body configuration,
// compute private tiles, join helpers, and distribute in original order.
// The caller retains separate COM and omega phases and applies dt^2*k_barrier
// to the barrier result. leader_work must not mutate the contact snapshot.
RigidContactOutput rigid_contact_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& cache,
    const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices,
    const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* kinematics = nullptr,
    bool cooperative = false,
    const std::function<void()>* leader_work = nullptr);

} // namespace ipc_simd

// Material records.
namespace solver_detail {

template <std::size_t Corners, class Matrix, class Shape>
struct GeneralSimdElementMaterials {
    std::vector<std::size_t> node_offsets;
    std::vector<std::array<int, Corners>> nodes;
    std::vector<Matrix> dm_inverse;
    std::vector<double> measures;
    std::vector<Shape> shape_gradients;
};

struct GeneralSimdHingeMaterials {
    std::vector<std::size_t> node_offsets;
    std::vector<std::array<int, 4>> nodes;
    std::vector<int> active_nodes;
    std::vector<double> coefficients, rest_angles;
};

// Static AoS material records in original per-node incident order. Prepare
// once before the worker team starts; positions, mass, pins and constitutive
// coefficients remain dynamic inputs and are never cached here. Value
// snapshots detect in-place topology, adjacency and rest-property edits.
class GeneralSimdMaterials {
public:
    GeneralSimdElementMaterials<3, Mat22, Vec2> triangles;
    GeneralSimdElementMaterials<4, Mat33, Vec3> tets;
    GeneralSimdHingeMaterials hinges;

    bool matches(const RefMesh& mesh, const std::vector<IncidentTriangles>& incident,
        std::size_t node_count, bool with_bending) const {
        return valid_ && mesh_ == &mesh && node_count_ == node_count
            && with_bending_ == with_bending && incident_ == incident
            && tet_adjacency_ == mesh.tet_adj && hinge_adjacency_ == mesh.hinge_adj
            && triangle_nodes_ == mesh.tris && tet_nodes_ == mesh.tets
            && solid_nodes_ == mesh.tet_nodes && rigid_owner_ == mesh.node_to_rb
            && rigid_nodes_ == mesh.rb_nodes
            && same_matrix_values(dm_inverse_, mesh.Dm_inverse)
            && same_scalar_values(areas_, mesh.area)
            && same_hinges(hinges_source_, mesh.hinges)
            && same_tet_rest(tet_rest_, mesh.tet_rest_data);
    }

    // Returns true only when records were rebuilt. Partial rebuilds are never
    // marked reusable, so a failed allocation/invalid index cannot poison the
    // next solver call's cache.
    bool prepare(const RefMesh& mesh, const std::vector<IncidentTriangles>& incident,
        std::size_t node_count, bool with_bending) {
        if (matches(mesh, incident, node_count, with_bending)) return false;
        valid_ = false;
        clear_records(triangles); clear_records(tets);
        hinges.nodes.clear(); hinges.active_nodes.clear();
        hinges.coefficients.clear(); hinges.rest_angles.clear();
        triangles.node_offsets.resize(node_count + 1);
        tets.node_offsets.resize(node_count + 1);
        hinges.node_offsets.resize(node_count + 1);
        std::vector<unsigned char> solid(node_count, 0), rigid(node_count, 0);
        for (int node : mesh.tet_nodes) solid.at(static_cast<std::size_t>(node)) = 1;
        for (const auto& body : mesh.rb_nodes)
            for (int node : body) rigid.at(static_cast<std::size_t>(node)) = 1;
        for (std::size_t node = 0; node < std::min(node_count, mesh.node_to_rb.size()); ++node)
            if (mesh.node_to_rb[node] >= 0) rigid[node] = 1;

        for (std::size_t node = 0; node < node_count; ++node) {
            triangles.node_offsets[node] = triangles.nodes.size();
            tets.node_offsets[node] = tets.nodes.size();
            hinges.node_offsets[node] = hinges.nodes.size();
            if (solid[node]) {
                for (const auto& [element, role] : mesh.tet_adj.at(node)) {
                    const auto& rest = mesh.tet_rest_data.at(element);
                    std::array<int, 4> corners;
                    for (int j = 0; j < 4; ++j) corners[j] = mesh.tets.at(4 * static_cast<std::size_t>(element) + j);
                    tets.nodes.push_back(corners);
                    tets.dm_inverse.push_back(rest.Dm_inverse);
                    tets.measures.push_back(rest.measure);
                    tets.shape_gradients.push_back(rest.grad_N.at(role));
                }
            } else if (!rigid[node]) {
                if (node < incident.size())
                    for (const auto& [element, role] : incident[node]) {
                        // Match the assembly's treatment of surface-only faces.
                        if (element < 0 || static_cast<std::size_t>(element) >= mesh.Dm_inverse.size()) continue;
                        std::array<int, 3> corners;
                        for (int j = 0; j < 3; ++j) corners[j] = mesh.tris.at(3 * static_cast<std::size_t>(element) + j);
                        const Mat22& inverse = mesh.Dm_inverse[element];
                        triangles.nodes.push_back(corners);
                        triangles.dm_inverse.push_back(inverse);
                        triangles.measures.push_back(mesh.area.at(element));
                        triangles.shape_gradients.push_back(shape_function_gradients(inverse).at(role));
                    }
                const auto found = with_bending ? mesh.hinge_adj.find(static_cast<int>(node)) : mesh.hinge_adj.end();
                if (found != mesh.hinge_adj.end())
                    for (const auto& [element, role] : found->second) {
                        const auto& hinge = mesh.hinges.at(element);
                        hinges.nodes.push_back({hinge.v[0], hinge.v[1], hinge.v[2], hinge.v[3]});
                        hinges.active_nodes.push_back(role);
                        hinges.coefficients.push_back(hinge.c_e);
                        hinges.rest_angles.push_back(hinge.bar_theta);
                    }
            }
        }
        triangles.node_offsets[node_count] = triangles.nodes.size();
        tets.node_offsets[node_count] = tets.nodes.size();
        hinges.node_offsets[node_count] = hinges.nodes.size();
        incident_ = incident;
        tet_adjacency_ = mesh.tet_adj;
        hinge_adjacency_ = mesh.hinge_adj;
        triangle_nodes_ = mesh.tris; tet_nodes_ = mesh.tets;
        solid_nodes_ = mesh.tet_nodes; rigid_owner_ = mesh.node_to_rb;
        rigid_nodes_ = mesh.rb_nodes;
        dm_inverse_ = mesh.Dm_inverse; areas_ = mesh.area;
        hinges_source_ = mesh.hinges; tet_rest_ = mesh.tet_rest_data;
        mesh_ = &mesh; node_count_ = node_count; with_bending_ = with_bending;
        valid_ = true;
        return true;
    }

private:
    template <std::size_t Corners, class Matrix, class Shape>
    static void clear_records(GeneralSimdElementMaterials<Corners, Matrix, Shape>& records) {
        records.nodes.clear(); records.dm_inverse.clear();
        records.measures.clear(); records.shape_gradients.clear();
    }
    static bool same_scalar_values(const std::vector<double>& a, const std::vector<double>& b) {
        return a.size() == b.size() && (a.empty()
            || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
    }
    template <class Matrix>
    static bool same_matrix_values(const std::vector<Matrix>& a, const std::vector<Matrix>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::memcmp(a[i].data(), b[i].data(), sizeof(double) * a[i].size()) != 0) return false;
        return true;
    }
    static bool same_hinges(const std::vector<Hinge>& a, const std::vector<Hinge>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!std::equal(a[i].v, a[i].v + 4, b[i].v)
                || std::memcmp(&a[i].bar_theta, &b[i].bar_theta, sizeof(double)) != 0
                || std::memcmp(&a[i].c_e, &b[i].c_e, sizeof(double)) != 0) return false;
        }
        return true;
    }
    static bool same_tet_rest(const std::vector<TetRestData>& a, const std::vector<TetRestData>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::memcmp(a[i].Dm_inverse.data(), b[i].Dm_inverse.data(), 9 * sizeof(double)) != 0
                || std::memcmp(&a[i].measure, &b[i].measure, sizeof(double)) != 0) return false;
            for (int role = 0; role < 4; ++role)
                if (std::memcmp(a[i].grad_N[role].data(), b[i].grad_N[role].data(), 3 * sizeof(double)) != 0) return false;
        }
        return true;
    }
    bool valid_ = false, with_bending_ = false;
    const RefMesh* mesh_ = nullptr;
    std::size_t node_count_ = 0;
    std::vector<IncidentTriangles> incident_, tet_adjacency_;
    VertexHingeMap hinge_adjacency_;
    std::vector<int> triangle_nodes_, tet_nodes_, solid_nodes_, rigid_owner_;
    std::vector<std::vector<int>> rigid_nodes_;
    std::vector<Mat22> dm_inverse_;
    std::vector<double> areas_;
    std::vector<Hinge> hinges_source_;
    std::vector<TetRestData> tet_rest_;
};

} // namespace solver_detail

// Vertex assembly.
namespace solver_detail {

// Private outputs for complete independent vertex ranges, just as in cloth
// experimental_v2. No kernel scatters into shared mesh positions.
struct GeneralSimdVertexSystem {
    Vec3 gradient = Vec3::Zero();
    Mat33 hessian = Mat33::Zero();
    Vec3 sdf_friction_gradient = Vec3::Zero();
    Mat33 sdf_friction_hessian = Mat33::Zero();
};

void prepare_general_simd_batch(
    const int* nodes, std::size_t count, const RefMesh& mesh,
    const std::vector<IncidentTriangles>& incident,
    const std::vector<ShapeGrads>& shape_gradients,
    const std::vector<Pin>& pins, const PinMap& pin_map,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& predicted,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask,
    GeneralSimdVertexSystem* outputs,
    const GeneralSimdMaterials* materials = nullptr);

// Retains cloth's per-pair accumulation and solid's separate normal/friction
// totals. Tet-interior filtering is identical to the solid scalar path.
void accumulate_general_simd_contacts(
    int node, bool solid, const BroadPhase::Cache& cache,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask, bool cooperative,
    GeneralSimdVertexSystem& output,
    safe_step_detail::VertexAabbRejections* rejections = nullptr,
    const std::function<void()>* leader_work = nullptr);

} // namespace solver_detail

// Batch scheduling.
namespace solver_detail {

struct GeneralSimdBatch {
    std::array<int, ipc_simd::tile_width> blocks{};
    std::size_t size = 0, contact_cost = 0;
};

// Only pack independent particle blocks of ONE color. A rigid COM/orientation
// update remains indivisible, and a heavy particle keeps its own helper team.
struct GeneralSimdBatches {
    std::vector<GeneralSimdBatch> batches;
    std::vector<std::vector<int>> colors;

    template<class Cost>
    void prepare(const std::vector<std::vector<int>>& groups, int rigid_begin,
                 const Cost& cost) {
        batches.clear();
        colors.assign(groups.size(), {});
        const auto team = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
        for (std::size_t c = 0; c < groups.size(); ++c) {
            GeneralSimdBatch pending;
            const std::size_t width = std::min(ipc_simd::tile_width,
                std::max(std::size_t(1), groups[c].size() / (2 * team)));
            const auto flush = [&] {
                if (!pending.size) return;
                colors[c].push_back(static_cast<int>(batches.size()));
                batches.push_back(pending);
                pending = GeneralSimdBatch{};
            };
            for (int block : groups[c]) {
                const auto weight = cost(block);
                if (block >= rigid_begin || weight >= 128) {
                    flush();
                    pending.blocks[0] = block;
                    pending.size = 1;
                    pending.contact_cost = weight;
                    flush();
                } else {
                    pending.blocks[pending.size++] = block;
                    pending.contact_cost += weight;
                    if (pending.size == width) flush();
                }
            }
            flush();
        }
    }
};

// Match the general solver's adaptive whole-work policy at batch granularity:
// cheap/uniform colors use cursor-free worker strides; uneven colors without
// helper assignments use heavy-first, adaptive dynamic claims. Remaining work
// alongside fixed helper teams still uses dynamic claims. Each rigid COM and
// orientation update stays one indivisible block under its assigned team.
// A single persistent team keeps every color barrier. On failure join helpers,
// restore the ENTIRE failed color
// (including rigid proxies/generalized coordinates), then rethrow on caller.
template<class Process, class Save, class Restore>
void run_general_simd_batches(const GeneralSimdBatches& batches, int sweeps,
    const Process& process, const Save& save, const Restore& restore) {
    const auto& groups = batches.colors;
    if (sweeps <= 0 || groups.empty()) return;
    if (omp_get_max_threads() == 1 || omp_in_parallel()) {
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (const auto& group : groups) {
                for (int item : group) save(item);
                try { for (int item : group) process(item, false); }
                catch (...) {
                    for (int item : group) restore(item);
                    throw;
                }
            }
        }
        return;
    }
    // Fixed sweeps revisit the same batches until the next contact rebuild.
    // Retain team storage and let prepare reuse an unchanged helper plan,
    // rather than allocating every color's contexts at each sweep interval.
    // Nested calls take the serial path above before borrowing this storage.
    static thread_local ColoredBlockTeams storage;
    ColoredBlockTeams& workspace = storage;
    workspace.prepare(groups, [&](int item) { return batches.batches[item].contact_cost; });
    workspace.barrier.reserve(omp_get_max_threads());
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::vector<std::atomic<bool>> failed_colors(groups.size());
    for (auto& flag : failed_colors) flag.store(false, std::memory_order_relaxed);
    const auto invoke = [&](int item, std::size_t color, bool cooperative) {
        // Save even after another batch failed: rollback covers this color.
        save(item);
        if (failed.load(std::memory_order_relaxed)) return;
        try { process(item, cooperative); }
        catch (...) {
            failed_colors[color].store(true, std::memory_order_relaxed);
            if (!failed.exchange(true, std::memory_order_relaxed))
                error = std::current_exception();
        }
    };
    #pragma omp parallel
    {
        const int worker = omp_get_thread_num(), team = omp_get_num_threads();
        const bool planned = team == workspace.team;
        #pragma omp single
        { workspace.barrier.initialize(team); }
        unsigned phase = 0;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (std::size_t c = 0; c < groups.size(); ++c) {
                const auto& whole = workspace.whole[c];
                if (!planned || whole.size() == groups[c].size()) {
                    const bool dynamic = planned && workspace.dynamic_whole[c];
                    const auto& group = dynamic ? whole : groups[c];
                    const int size = static_cast<int>(group.size());
                    if (planned && !dynamic) {
                        for (int item = worker; item < size; item += team)
                            invoke(group[item], c, false);
                    } else {
                        // Reserve the first wave without cursor contention.
                        // A runtime team-size mismatch disables helper plans
                        // and safely claims all remaining batches one at a time.
                        if (worker < size) invoke(group[worker], c, false);
                        for (;;) {
                            const int remaining = size - team
                                - workspace.next[c].load(std::memory_order_relaxed);
                            if (remaining <= 0) break;
                            const int count = dynamic
                                ? std::min(8, std::max(1, remaining / (4 * team))) : 1;
                            const int first = team + workspace.next[c].fetch_add(count, std::memory_order_relaxed);
                            for (int item = first; item < std::min(first + count, size); ++item)
                                invoke(group[item], c, false);
                        }
                    }
                } else {
                    const auto assignment = workspace.assignments[c * workspace.team + worker];
                    if (assignment.block >= 0) {
                        auto& context = workspace.contexts[assignment.context];
                        if (assignment.lane == 0) {
                            auto* previous = active_contact_task_group;
                            active_contact_task_group = assignment.lanes > 1 ? &context : nullptr;
                            invoke(assignment.block, c, assignment.lanes > 1);
                            active_contact_task_group = previous;
                            context.sequence.store(-1, std::memory_order_release);
                        } else context.help();
                    }
                    const int size = static_cast<int>(whole.size());
                    // Helper-only colors have no unassigned batches. Avoid
                    // making every worker contend on an already empty queue.
                    while (workspace.next[c].load(std::memory_order_relaxed) < size) {
                        const int item = workspace.next[c].fetch_add(1, std::memory_order_relaxed);
                        if (item < size) invoke(whole[item], c, false);
                    }
                }
                workspace.barrier.wait(worker, ++phase, [&] {
                    if (failed_colors[c].load(std::memory_order_relaxed)) {
                        for (int item : groups[c]) restore(item);
                        failed_colors[c].store(false, std::memory_order_relaxed);
                    }
                    workspace.next[c].store(0, std::memory_order_relaxed);
                    for (int context : workspace.helper_contexts[c])
                        workspace.contexts[context].sequence.store(0, std::memory_order_relaxed);
                });
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

} // namespace solver_detail

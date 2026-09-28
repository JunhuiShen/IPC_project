#pragma once

#include "IPC_math.h"
#include "volumetric_corotated_energy.h"

#include <optional>
#include <utility>
#include <vector>

struct DeformedState;
struct FrozenResidualWorkspace;
class BroadPhase;
struct Pin;
struct RefMesh;
struct SimParams;
namespace safe_step_detail { struct VertexAabbRejections; }

// Append one disconnected deformable solid.
//
// x contains this solid's initial/rest positions in local-node order, and
// local_tets contains four local indices per tetrahedron. The function remaps
// those indices into the shared global particle arrays, appends the outward
// boundary triangles for collision/rendering, initializes TGSL-style tet rest
// data, incidence, and lumped nodal masses, and assigns zero initial velocity.
void create_solid(
    const std::vector<Vec3>& x,
    const std::vector<int>& local_tets,
    double density,
    RefMesh& ref_mesh,
    DeformedState& state);

// Backward-Euler incremental potential for volumetric solids. Boundary
// triangles are collision/render geometry only and are deliberately not used
// for elasticity. SDF penalties are evaluated only on the solid boundary
// nodes stored in RefMesh::surface_nodes; IPC barriers are added separately.
double compute_solid_incremental_potential_no_barrier(
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat);

// Returns the exact positive gradient of the no-barrier incremental potential
// for one solid node and TGSL's PSD PBGS 3x3 block plus the cloth-style SDF
// normal-direction Hessian on boundary nodes (SDF curvature is omitted). The
// elastic part of the second return value is an approximate local Hessian used
// by PBGS, not the exact energy Hessian.
std::pair<Vec3, Mat33>
compute_solid_local_gradient_and_pbgs_block_no_barrier(
    int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const std::vector<unsigned char>* surface_node_mask = nullptr,
    const std::vector<int>* pin_map = nullptr);

// The already dt^2*k_barrier-scaled IPC barrier contribution. The broad-phase
// cache is fixed by the caller; each pair is counted once in the energy.
double compute_solid_barrier_incremental_potential(
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const BroadPhase& broad_phase);

// Frozen-contact Coulomb friction incremental potential. Each retained
// NT/SS pair is counted once. A previous-position vector is required only
// when friction is enabled; a zero coefficient returns exactly zero without
// inspecting the pointer.
double compute_solid_friction_incremental_potential(
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const BroadPhase& broad_phase,
    const std::vector<Vec3>* previous_positions = nullptr);

// Frozen friction against the minimum active analytic SDF at each solid
// boundary node. Prescribed SDF material motion is carried by SDFEvaluation.
double compute_solid_sdf_friction_incremental_potential(
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>* previous_positions = nullptr);

// The already dt^2*k_barrier-scaled barrier gradient and exact diagonal/self
// Hessian block for one solid node.
std::pair<Vec3, Mat33>
compute_solid_local_barrier_gradient_and_self_hessian(
    int node,
    const RefMesh& ref_mesh,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>* solid_node_mask = nullptr,
    const std::vector<unsigned char>* surface_node_mask = nullptr);

// Full solid incremental potential and local system, obtained by adding the
// barrier contributions above to the no-barrier terms. The complete local
// matrix combines TGSL's PSD elastic PBGS approximation, the SDF
// normal-direction Hessian, frozen friction PSD blocks, and exact barrier
// self Hessians and therefore is not guaranteed to be PSD.
double compute_solid_incremental_potential(
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<Vec3>* previous_positions = nullptr);

std::pair<Vec3, Mat33> compute_solid_local_gradient_and_block(
    int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>* solid_node_mask = nullptr,
    const std::vector<unsigned char>* surface_node_mask = nullptr,
    const std::vector<int>* pin_map = nullptr,
    const std::vector<Vec3>* previous_positions = nullptr);

namespace solid_ipc_detail {

// Rebuilt once per solve, so in-place rest/material changes cannot leave stale
// coefficients. Mutable geometry entries belong only to dependency-colored
// nodal updates, never parallel nodal residual evaluation.
// Both solid_mu and solid_lambda must stay fixed until the next prepare().
// Connectivity changes also require refreshing the enclosing solver's colors.
struct PreparedSolidWorkspace {
    using Geometry = volumetric_detail::PreparedTetGeometry;
    std::vector<volumetric_detail::PreparedTet> elements;
    std::vector<std::optional<Geometry>> geometry;

    // Production policy: probe initially and every sixteenth solve, deciding
    // from the complete solve so late settling contributes to the hit rate.
    // Low reuse disables only the geometry memo, retaining prepared material.
    void begin_solve(const RefMesh& mesh, const std::vector<Vec3>& positions,
        const SimParams& params);
    void end_solve();
    std::pair<Vec3, Mat33> evaluate(std::size_t element, int local_node,
        const std::vector<Vec3>& positions, const SimParams& params);

private:
    enum class Mode { Probe, Cached, Prepared };
    void prepare(const RefMesh& mesh, const std::vector<Vec3>& positions,
        const SimParams& params, bool with_geometry_cache);
    struct alignas(64) ProbeCounts { std::size_t calls = 0, hits = 0; };
    std::vector<ProbeCounts> probe_counts_;
    Mode mode_ = Mode::Cached;
    Mode selected_mode_ = Mode::Cached;
    int solves_until_probe_ = 0;
    const RefMesh* policy_mesh_ = nullptr;
    const int* policy_tets_ = nullptr;
    std::size_t policy_tet_count_ = 0, policy_position_count_ = 0;
    double policy_mu_ = 0.0, policy_lambda_ = 0.0;
};

// Solver-only fast path. The enclosing solver entry point must already have
// validated the friction parameters and previous-position array.
// Optional AABB rejections remain valid only until these contact positions or
// incidence rows change; safe-step applies its finite, short-step reuse guard.
std::pair<Vec3, Mat33> compute_solid_local_gradient_and_block_unchecked(
    int node,
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>* solid_node_mask,
    const std::vector<unsigned char>* surface_node_mask,
    const std::vector<int>* pin_map,
    const std::vector<Vec3>* previous_positions, bool cooperative = false,
    safe_step_detail::VertexAabbRejections* rejections = nullptr,
    PreparedSolidWorkspace* prepared = nullptr);

} // namespace solid_ipc_detail

// Mass-normalized infinity norm of the full solid gradient, matching the
// deformable-cloth residual convention. Only tetrahedral solid nodes are
// included.
double compute_global_solid_residual(
    const RefMesh& ref_mesh,
    const std::vector<Pin>& pins,
    const SimParams& params,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat,
    const BroadPhase& broad_phase,
    const std::vector<int>* pin_map = nullptr,
    const std::vector<unsigned char>* solid_node_mask = nullptr,
    const std::vector<unsigned char>* surface_node_mask = nullptr,
    const FrozenResidualWorkspace* frozen_workspace = nullptr,
    const std::vector<Vec3>* previous_positions = nullptr);

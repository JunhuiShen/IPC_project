#pragma once

#include "IPC_math.h"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

// Fixed reference data for one tetrahedron. TGSL stores these quantities in
// flattened arrays; this port keeps one typed record per element.
struct TetRestData {
    Mat33 Dm_inverse;
    double measure;
    std::array<Vec3, 4> grad_N;
};

enum class CorotatedCacheMode {
    Full,
    Lean
};

// Typed equivalent of TGSL::CorotatedCache. Call UpdateCache(F) before Psi,
// P, or PBGSElementNodeElasticityBlock. Lean mode computes every field used by
// those operations identically to Full mode but leaves Dinv_cache untouched.
struct CorotatedCache {
    Mat33 JFinvT_cache;
    Mat33 R_cache;
    Mat33 Dinv_cache;
    double J_cache;

    void UpdateCache(const Mat33& F, CorotatedCacheMode mode = CorotatedCacheMode::Full);
    double Psi(const Mat33& F, double mu, double lambda) const;
    Mat33 P(const Mat33& F, double mu, double lambda) const;
};

// TGSL ElasticFEM naming and global-array layout:
//   Ds = [u1-u0, u2-u0, u3-u0].
Mat33 ElementDs(
    std::size_t element,
    const std::vector<Vec3>& u,
    const std::vector<int>& mesh);

// F = ElementDs(element, x, mesh) * Dm_inverse[element].
Mat33 ElementF(
    std::size_t element,
    const std::vector<Vec3>& x,
    const std::vector<int>& mesh,
    const std::vector<TetRestData>& state);

namespace volumetric_detail {

// Populate the same fields as CorotatedCacheMode::Lean, leaving Dinv_cache
// untouched. Independent inputs share the exact batched signed QR-SVD kernel;
// determinant and cofactor arithmetic stays in this strict-rounding module.
// Zero count permits null pointers.
void update_corotated_cache_batch(const Mat33* inputs,
    CorotatedCache* caches, std::size_t count);

// Organizer-owned snapshot, refreshed whenever connectivity, rest data, or mu
// changes. Keeping derived material values here leaves TetRestData mutable.
struct PreparedTet {
    std::array<int, 4> nodes;
    TetRestData rest;
    std::array<double, 4> isotropic_diagonal;
};

// Initialized geometry fields used by every node role at exactly the same F.
// The organizer must clear this memo when material parameters change.
struct PreparedTetGeometry {
    Mat33 F;
    Mat33 first_piola;
    Mat33 cofactor;
};

inline constexpr std::size_t prepared_tet_batch_size = 32;
// Same per-element results as evaluate_prepared_tet[_cached]. Incidence must
// contain distinct valid elements and roles, count <= prepared_tet_batch_size.
// A null geometry pointer selects the uncached material-only path.
void evaluate_prepared_tet_batch(const PreparedTet* elements,
    const std::pair<int, int>* incidence, std::size_t count,
    const std::vector<Vec3>& positions, double mu, double lambda,
    std::optional<PreparedTetGeometry>* geometry,
    std::pair<Vec3, Mat33>* outputs, std::size_t* cache_hits = nullptr);

// Validate connectivity and rest storage with ElementF's error precedence,
// then prepare all four roles without retaining references to those arrays.
PreparedTet prepare_tet(
    std::size_t element,
    const std::vector<Vec3>& positions,
    const std::vector<int>& mesh,
    const std::vector<TetRestData>& rest,
    double mu);

// Keep prepared material/topology data even when geometry reuse is disabled.
// The same position/local-role/material preconditions as the cached path apply.
std::pair<Vec3, Mat33> evaluate_prepared_tet(
    const PreparedTet& element,
    const std::vector<Vec3>& positions,
    double mu,
    double lambda,
    int local_node);

// Reuse geometry only for a bitwise-equal live F. A failed refresh leaves the
// previous initialized entry intact. The organizer owns exclusive access.
// First use and role-zero misses refresh the entry; other misses evaluate the
// current geometry without replacing the saved key.
// Position storage must contain every prepared node, local_node must be in
// [0, 3], and mu must match prepare_tet. Parameters stay fixed until memo reset.
std::pair<Vec3, Mat33> evaluate_prepared_tet_cached(
    const PreparedTet& element,
    const std::vector<Vec3>& positions,
    double mu,
    double lambda,
    int local_node,
    std::optional<PreparedTetGeometry>& geometry);

// Probe-only entry: report whether this successful evaluation reused geometry.
// The normal entry above does not update probe counters or reporting state.
std::pair<Vec3, Mat33> evaluate_prepared_tet_cached_probe(
    const PreparedTet& element,
    const std::vector<Vec3>& positions,
    double mu,
    double lambda,
    int local_node,
    std::optional<PreparedTetGeometry>& geometry,
    bool& cache_hit);

} // namespace volumetric_detail

// TGSL EFEMInitializeElasticMaterialState equivalent. For every element this
// builds measure=det(Dm)/6, Dm_inverse, and all four grad_N values. TGSL PBGS
// stores only the current central node's grad_N in each duplicated one-ring;
// this compact representation stores all four once per global tet.
std::vector<TetRestData> EFEMInitializeElasticMaterialState(
    const std::vector<Vec3>& X,
    const std::vector<int>& mesh);

// grad(det(F)) = cofactor(F), called GradJ in TGSL.
Mat33 GradJ(const Mat33& F);

// One element's energy, positive energy gradient, and TGSL PBGS elasticity
// block. TGSL's EFEMAddInternalForce is the negative of the gradient below.
double EFEMElementInternalEnergy(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda);

std::array<Vec3, 4> EFEMElementEnergyGradient(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda);

Vec3 EFEMElementNodeEnergyGradient(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node,
    const Mat33* precomputed_first_piola = nullptr);

// TGSL PBGS's PSD per-element/node elasticity contribution, not the exact
// energy Hessian:
//   measure [2 mu ||grad_N||^2 I + lambda u u^T],
//   u = JFinvT grad_N.
Mat33 PBGSElementNodeElasticityBlock(
    const CorotatedCache& cache,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node);

std::pair<Vec3, Mat33> EFEMElementNodeGradientAndPBGSBlock(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node);

#pragma once

#include "IPC_math.h"

#include <array>
#include <cstddef>
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

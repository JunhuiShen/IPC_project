#include "volumetric_corotated_energy.h"
#include "batched_polar.h"

#include "third_party/tgsl/ImplicitQRSVD.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

void CheckElement(
    std::size_t element,
    const std::vector<Vec3>& u,
    const std::vector<int>& mesh) {
    if (mesh.size() % 4 != 0) {
        throw std::invalid_argument(
            "tet connectivity must contain four indices per element");
    }
    if (element >= mesh.size() / 4) {
        throw std::out_of_range("tet element index is out of range");
    }
    for (int local = 0; local < 4; ++local) {
        const int node = mesh[4 * element + local];
        if (node < 0 || static_cast<std::size_t>(node) >= u.size()) {
            throw std::out_of_range("tet node index is out of range");
        }
    }
}

void CheckLocalNode(int local_node) {
    if (local_node < 0 || local_node >= 4) {
        throw std::out_of_range("tet local node must be in [0, 3]");
    }
}

Mat33 ElementDsTrusted(std::size_t element, const std::vector<Vec3>& u, const std::vector<int>& mesh) {
    Mat33 result;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t c = 0; c < 3; ++c)
            result(c, i) = u[mesh[4 * element + i + 1]][c] - u[mesh[4 * element]][c];
    }
    return result;
}

inline Mat33 ElementFPreparedTrusted(
    const volumetric_detail::PreparedTet& element, const std::vector<Vec3>& x) {
    Mat33 Ds;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t c = 0; c < 3; ++c)
            Ds(c, i) = x[element.nodes[i + 1]][c] - x[element.nodes[0]][c];
    }
    return Ds * element.rest.Dm_inverse;
}

} // namespace

Mat33 ElementDs(
    std::size_t element,
    const std::vector<Vec3>& u,
    const std::vector<int>& mesh) {
    CheckElement(element, u, mesh);
    return ElementDsTrusted(element, u, mesh);
}

Mat33 ElementF(
    std::size_t element,
    const std::vector<Vec3>& x,
    const std::vector<int>& mesh,
    const std::vector<TetRestData>& state) {
    CheckElement(element, x, mesh);
    if (state.size() != mesh.size() / 4) {
        throw std::invalid_argument(
            "tet rest state must contain one record per element");
    }
    return ElementDsTrusted(element, x, mesh) * state[element].Dm_inverse;
}

volumetric_detail::PreparedTet volumetric_detail::prepare_tet(
    std::size_t element,
    const std::vector<Vec3>& positions,
    const std::vector<int>& mesh,
    const std::vector<TetRestData>& rest,
    double mu) {
    CheckElement(element, positions, mesh);
    if (rest.size() != mesh.size() / 4) {
        throw std::invalid_argument(
            "tet rest state must contain one record per element");
    }
    PreparedTet prepared;
    prepared.rest = rest[element];
    for (int local = 0; local < 4; ++local) {
        prepared.nodes[local] = mesh[4 * element + local];
        const Vec3& grad_Ni = prepared.rest.grad_N[local];
        const double grad_Ni_dot_grad_Ni = grad_Ni.dot(grad_Ni);
        prepared.isotropic_diagonal[local] =
            2.0 * mu * grad_Ni_dot_grad_Ni * prepared.rest.measure;
    }
    return prepared;
}

std::vector<TetRestData> EFEMInitializeElasticMaterialState(
    const std::vector<Vec3>& X,
    const std::vector<int>& mesh) {
    if (mesh.size() % 4 != 0) {
        throw std::invalid_argument(
            "tet connectivity must contain four indices per element");
    }

    std::vector<TetRestData> state(mesh.size() / 4);

    double one_over_d_factorial = 1.0;
    std::array<Vec3, 4> grad_N_hat;
    grad_N_hat[0] = Vec3::Zero();
    for (std::size_t alpha = 0; alpha < 3; ++alpha)
        grad_N_hat[0][alpha] = -1.0;
    for (std::size_t ie = 1; ie < 4; ++ie) {
        grad_N_hat[ie] = Vec3::Zero();
        grad_N_hat[ie][ie - 1] = 1.0;
    }
    for (std::size_t c = 1; c < 3; ++c)
        one_over_d_factorial /= static_cast<double>(c + 1);
    std::size_t num_degenerate = 0;

    for (std::size_t element = 0; element < mesh.size() / 4; ++element) {
        Mat33 Dm = ElementDs(element, X, mesh);
        if (!Dm.allFinite()) {
            throw std::invalid_argument(
                "tetrahedron rest positions must be finite");
        }

        state[element].measure = one_over_d_factorial * Dm.determinant();
        if (!std::isfinite(state[element].measure)) {
            throw std::invalid_argument(
                "tetrahedron rest measure must be finite");
        }
        if (state[element].measure <= 0.0) {
            state[element].measure = -state[element].measure;
            ++num_degenerate;
        }

        Mat33 Dm_inverse = Dm.inverse();
        state[element].Dm_inverse = Dm_inverse;

        for (std::size_t ie = 0; ie < 4; ++ie) {
            Vec3 g_Ni_hat;
            for (std::size_t alpha = 0; alpha < 3; ++alpha)
                g_Ni_hat(alpha) = grad_N_hat[ie][alpha];
            Vec3 g_Ni = Dm_inverse.transpose() * g_Ni_hat;
            for (std::size_t alpha = 0; alpha < 3; ++alpha)
                state[element].grad_N[ie][alpha] = g_Ni(alpha);
        }
    }

    if (num_degenerate != 0) {
        throw std::invalid_argument(
            "tetrahedron rest orientation must have positive measure");
    }
    return state;
}

Mat33 GradJ(const Mat33& F) {
    Mat33 grad_J;
    grad_J(0, 0) = F(1, 1) * F(2, 2) - F(2, 1) * F(1, 2);
    grad_J(0, 1) = F(2, 0) * F(1, 2) - F(1, 0) * F(2, 2);
    grad_J(0, 2) = F(1, 0) * F(2, 1) - F(2, 0) * F(1, 1);
    grad_J(1, 0) = F(2, 1) * F(0, 2) - F(0, 1) * F(2, 2);
    grad_J(1, 1) = F(0, 0) * F(2, 2) - F(2, 0) * F(0, 2);
    grad_J(1, 2) = F(2, 0) * F(0, 1) - F(0, 0) * F(2, 1);
    grad_J(2, 0) = F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2);
    grad_J(2, 1) = F(1, 0) * F(0, 2) - F(0, 0) * F(1, 2);
    grad_J(2, 2) = F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1);
    return grad_J;
}

namespace {

// Internal linkage lets the fused hot path specialize Lean even in a PIC,
// non-LTO build, without changing arithmetic in the public cache modes.
template <CorotatedCacheMode Mode>
inline void UpdateCorotatedCacheTrusted(CorotatedCache& cache, const Mat33& F) {
    if (!F.allFinite()) {
        throw std::invalid_argument("deformation gradient must be finite");
    }

    Mat33 R, S;
    if constexpr (Mode == CorotatedCacheMode::Full) {
        JIXIE::polarDecomposition(F, R, S);
    } else {
        Mat33 U;
        Vec3 sigma;
        Mat33 V;
        JIXIE::singularValueDecomposition(F, U, sigma, V);
        R.noalias() = U * V.transpose();
    }

    cache.JFinvT_cache <<
        F(1, 1) * F(2, 2) - F(2, 1) * F(1, 2),
        F(2, 0) * F(1, 2) - F(1, 0) * F(2, 2),
        F(1, 0) * F(2, 1) - F(2, 0) * F(1, 1),
        F(2, 1) * F(0, 2) - F(0, 1) * F(2, 2),
        F(0, 0) * F(2, 2) - F(2, 0) * F(0, 2),
        F(2, 0) * F(0, 1) - F(0, 0) * F(2, 1),
        F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2),
        F(1, 0) * F(0, 2) - F(0, 0) * F(1, 2),
        F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1);

    cache.R_cache <<
        R(0, 0), R(0, 1), R(0, 2),
        R(1, 0), R(1, 1), R(1, 2),
        R(2, 0), R(2, 1), R(2, 2);

    if constexpr (Mode == CorotatedCacheMode::Full) {
        Mat33 D = S.trace() * Mat33::Identity() - S;
        Mat33 Dinv = D.inverse();
        cache.Dinv_cache << Dinv(0, 0), Dinv(0, 1), Dinv(0, 2), Dinv(1, 0), Dinv(1, 1), Dinv(1, 2), Dinv(2, 0), Dinv(2, 1), Dinv(2, 2);
    }

    cache.J_cache = F.determinant();
}

} // namespace

void CorotatedCache::UpdateCache(const Mat33& F, CorotatedCacheMode mode) {
    if (mode == CorotatedCacheMode::Full)
        UpdateCorotatedCacheTrusted<CorotatedCacheMode::Full>(*this, F);
    else
        UpdateCorotatedCacheTrusted<CorotatedCacheMode::Lean>(*this, F);
}

void volumetric_detail::update_corotated_cache_batch(const Mat33* inputs,
    CorotatedCache* caches, std::size_t count) {
    constexpr std::size_t width = 8;
#if defined(__AVX512F__)
    std::array<Mat33, width> rotations;
#endif
    for (std::size_t first = 0; first < count; first += width) {
        const std::size_t lanes = std::min(width, count - first);
#if defined(__AVX512F__)
        if (lanes >= 4) {
            batched_signed_polar(inputs + first, rotations.data(), lanes);
            for (std::size_t i = 0; i < lanes; ++i) {
                auto& cache = caches[first + i];
                cache.R_cache = rotations[i];
                cache.JFinvT_cache = GradJ(inputs[first + i]);
                cache.J_cache = inputs[first + i].determinant();
            }
            continue;
        }
#endif
        for (std::size_t i = 0; i < lanes; ++i)
            UpdateCorotatedCacheTrusted<CorotatedCacheMode::Lean>(
                caches[first + i], inputs[first + i]);
    }
}

double CorotatedCache::Psi(
    const Mat33& F,
    double mu,
    double lambda) const {
    Mat33 R;
    R <<
        R_cache(0, 0), R_cache(0, 1), R_cache(0, 2),
        R_cache(1, 0), R_cache(1, 1), R_cache(1, 2),
        R_cache(2, 0), R_cache(2, 1), R_cache(2, 2);
    return mu * ((F - R).squaredNorm())
        + lambda * (J_cache - 1.0) * (J_cache - 1.0) / 2.0;
}

namespace {

inline Mat33 FirstPiolaTrusted(
    const CorotatedCache& cache, const Mat33& F, double mu, double lambda) {
    Mat33 R, JFinvT;
    R <<
        cache.R_cache(0, 0), cache.R_cache(0, 1), cache.R_cache(0, 2),
        cache.R_cache(1, 0), cache.R_cache(1, 1), cache.R_cache(1, 2),
        cache.R_cache(2, 0), cache.R_cache(2, 1), cache.R_cache(2, 2);
    JFinvT <<
        cache.JFinvT_cache(0, 0), cache.JFinvT_cache(0, 1), cache.JFinvT_cache(0, 2),
        cache.JFinvT_cache(1, 0), cache.JFinvT_cache(1, 1), cache.JFinvT_cache(1, 2),
        cache.JFinvT_cache(2, 0), cache.JFinvT_cache(2, 1), cache.JFinvT_cache(2, 2);
    Mat33 first_piola;
    first_piola = 2.0 * mu * (F - R)
        + lambda * (cache.J_cache - 1.0) * JFinvT;
    return first_piola;
}

} // namespace

Mat33 CorotatedCache::P(
    const Mat33& F, double mu, double lambda) const {
    return FirstPiolaTrusted(*this, F, mu, lambda);
}

double EFEMElementInternalEnergy(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda) {
    double energy = 0.0;
    energy += cache.Psi(F, mu, lambda) * state.measure;
    return energy;
}

std::array<Vec3, 4> EFEMElementEnergyGradient(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda) {
    Mat33 Pe = cache.P(F, mu, lambda);
    Mat33 g = state.measure * Pe * state.Dm_inverse.transpose();

    std::array<Vec3, 4> gradient = {
        Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
    for (std::size_t ie = 0; ie < 3; ++ie)
        for (std::size_t c = 0; c < 3; ++c)
            gradient[ie + 1][c] += g(c, ie);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t h = 0; h < 3; ++h)
            gradient[0][c] -= g(c, h);
    return gradient;
}

namespace {

inline Vec3 EFEMElementNodeEnergyGradientTrusted(const CorotatedCache& cache, const Mat33& F, const TetRestData& state, double mu, double lambda, int local_node, const Mat33* precomputed_first_piola = nullptr) {
    Mat33 owned_first_piola;
    if (precomputed_first_piola == nullptr) owned_first_piola = FirstPiolaTrusted(cache, F, mu, lambda);
    const Mat33& Pe = precomputed_first_piola == nullptr ? owned_first_piola : *precomputed_first_piola;
    Vec3 gNi;
    gNi << state.grad_N[local_node][0], state.grad_N[local_node][1], state.grad_N[local_node][2];
    Vec3 g = Pe * gNi * state.measure;
    return g;
}

Mat33 PBGSElementNodeElasticityBlockTrusted(const CorotatedCache& cache, const TetRestData& state, double mu, double lambda, int local_node) {
    Mat33 A = Mat33::Zero();

    const Vec3& grad_Ni = state.grad_N[local_node];
    double grad_Ni_dot_grad_Ni = grad_Ni.dot(grad_Ni);
    A(0, 0) += 2.0 * mu * grad_Ni_dot_grad_Ni * state.measure;
    A(1, 1) += 2.0 * mu * grad_Ni_dot_grad_Ni * state.measure;
    A(2, 2) += 2.0 * mu * grad_Ni_dot_grad_Ni * state.measure;

    Mat33 grad_Je = cache.JFinvT_cache;
    Vec3 g_Nie;
    for (std::size_t alpha = 0; alpha < 3; ++alpha)
        g_Nie(alpha) = grad_Ni[alpha];
    Vec3 ue = grad_Je * g_Nie;
    for (std::size_t alpha = 0; alpha < 3; ++alpha) {
        for (std::size_t beta = 0; beta < 3; ++beta)
            A(alpha, beta) += lambda * ue(alpha) * ue(beta) * state.measure;
    }

    return A;
}

} // namespace

Vec3 EFEMElementNodeEnergyGradient(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node,
    const Mat33* precomputed_first_piola) {
    CheckLocalNode(local_node);
    return EFEMElementNodeEnergyGradientTrusted(cache, F, state, mu, lambda, local_node, precomputed_first_piola);
}

Mat33 PBGSElementNodeElasticityBlock(
    const CorotatedCache& cache,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node) {
    CheckLocalNode(local_node);
    return PBGSElementNodeElasticityBlockTrusted(cache, state, mu, lambda, local_node);
}

std::pair<Vec3, Mat33> EFEMElementNodeGradientAndPBGSBlock(const CorotatedCache& cache, const Mat33& F, const TetRestData& state, double mu, double lambda, int local_node) {
    CheckLocalNode(local_node);
    const Vec3 gradient = EFEMElementNodeEnergyGradientTrusted(cache, F, state, mu, lambda, local_node);
    const Mat33 block = PBGSElementNodeElasticityBlockTrusted(cache, state, mu, lambda, local_node);
    return {gradient, block};
}

namespace {

inline std::pair<Vec3, Mat33> PreparedTetDerivativesTrusted(
    const CorotatedCache& cache, const Mat33& F, const volumetric_detail::PreparedTet& element,
    double mu, double lambda, int local_node,
    const Mat33* precomputed_first_piola) {
    assert(local_node >= 0 && local_node < 4);
    const TetRestData& state = element.rest;
    const Vec3 gradient = EFEMElementNodeEnergyGradientTrusted(
        cache, F, state, mu, lambda, local_node, precomputed_first_piola);
    Mat33 A = Mat33::Zero();
    A(0, 0) += element.isotropic_diagonal[local_node];
    A(1, 1) += element.isotropic_diagonal[local_node];
    A(2, 2) += element.isotropic_diagonal[local_node];

    Mat33 grad_Je = cache.JFinvT_cache;
    Vec3 g_Nie;
    for (std::size_t alpha = 0; alpha < 3; ++alpha)
        g_Nie(alpha) = state.grad_N[local_node][alpha];
    Vec3 ue = grad_Je * g_Nie;
    for (std::size_t alpha = 0; alpha < 3; ++alpha) {
        for (std::size_t beta = 0; beta < 3; ++beta)
            A(alpha, beta) += lambda * ue(alpha) * ue(beta) * state.measure;
    }
    return {gradient, A};
}

template <bool ReportHit>
inline std::pair<Vec3, Mat33> EvaluatePreparedTetCachedTrusted(
    const volumetric_detail::PreparedTet& element, const std::vector<Vec3>& positions,
    double mu, double lambda, int local_node,
    std::optional<volumetric_detail::PreparedTetGeometry>& geometry, bool* cache_hit) {
    const Mat33 F = ElementFPreparedTrusted(element, positions);
    CorotatedCache cache;
    if (geometry && std::memcmp(F.data(), geometry->F.data(), 9 * sizeof(double)) == 0) {
        cache.JFinvT_cache = geometry->cofactor;
        const auto result = PreparedTetDerivativesTrusted(
            cache, F, element, mu, lambda, local_node, &geometry->first_piola);
        if constexpr (ReportHit) *cache_hit = true;
        return result;
    }

    UpdateCorotatedCacheTrusted<CorotatedCacheMode::Lean>(cache, F);
    const Mat33 first_piola = FirstPiolaTrusted(cache, F, mu, lambda);
    const auto result = PreparedTetDerivativesTrusted(
        cache, F, element, mu, lambda, local_node, &first_piola);
    // Other roles use fresh derivatives without repeatedly overwriting a key
    // shared across colors. Only fully initialized consumer fields are saved.
    if (!geometry || local_node == 0)
        geometry = volumetric_detail::PreparedTetGeometry{F, first_piola, cache.JFinvT_cache};
    if constexpr (ReportHit) *cache_hit = false;
    return result;
}

} // namespace

std::pair<Vec3, Mat33> volumetric_detail::evaluate_prepared_tet(
    const PreparedTet& element, const std::vector<Vec3>& positions,
    double mu, double lambda, int local_node) {
    const Mat33 F = ElementFPreparedTrusted(element, positions);
    CorotatedCache cache;
    UpdateCorotatedCacheTrusted<CorotatedCacheMode::Lean>(cache, F);
    const Mat33 first_piola = FirstPiolaTrusted(cache, F, mu, lambda);
    return PreparedTetDerivativesTrusted(
        cache, F, element, mu, lambda, local_node, &first_piola);
}

std::pair<Vec3, Mat33> volumetric_detail::evaluate_prepared_tet_cached(
    const PreparedTet& element, const std::vector<Vec3>& positions,
    double mu, double lambda, int local_node,
    std::optional<PreparedTetGeometry>& geometry) {
    return EvaluatePreparedTetCachedTrusted<false>(
        element, positions, mu, lambda, local_node, geometry, nullptr);
}

std::pair<Vec3, Mat33> volumetric_detail::evaluate_prepared_tet_cached_probe(
    const PreparedTet& element, const std::vector<Vec3>& positions,
    double mu, double lambda, int local_node,
    std::optional<PreparedTetGeometry>& geometry, bool& cache_hit) {
    return EvaluatePreparedTetCachedTrusted<true>(
        element, positions, mu, lambda, local_node, geometry, &cache_hit);
}

void volumetric_detail::evaluate_prepared_tet_batch(const PreparedTet* elements,
    const std::pair<int, int>* incidence, std::size_t count,
    const std::vector<Vec3>& positions, double mu, double lambda,
    std::optional<PreparedTetGeometry>* geometry,
    std::pair<Vec3, Mat33>* outputs, std::size_t* cache_hits) {
    assert(count <= prepared_tet_batch_size);
    std::array<Mat33, prepared_tet_batch_size> gradients, rotations;
    std::array<std::size_t, prepared_tet_batch_size> misses;
    std::size_t miss_count = 0, hits = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto [element, role] = incidence[i];
        const Mat33 F = ElementFPreparedTrusted(elements[element], positions);
        if (geometry && geometry[element]
            && std::memcmp(F.data(), geometry[element]->F.data(), 9 * sizeof(double)) == 0) {
            CorotatedCache cache;
            cache.JFinvT_cache = geometry[element]->cofactor;
            outputs[i] = PreparedTetDerivativesTrusted(cache, F, elements[element],
                mu, lambda, role, &geometry[element]->first_piola);
            ++hits;
        } else {
            if (!F.allFinite()) throw std::invalid_argument("deformation gradient must be finite");
            gradients[miss_count] = F;
            misses[miss_count++] = i;
        }
    }
    batched_signed_polar(gradients.data(), rotations.data(), miss_count);
    for (std::size_t miss = 0; miss < miss_count; ++miss) {
        const std::size_t i = misses[miss];
        const auto [element, role] = incidence[i];
        const Mat33& F = gradients[miss];
        CorotatedCache cache;
        cache.R_cache = rotations[miss];
        cache.JFinvT_cache = GradJ(F);
        cache.J_cache = F.determinant();
        const Mat33 first_piola = FirstPiolaTrusted(cache, F, mu, lambda);
        outputs[i] = PreparedTetDerivativesTrusted(cache, F, elements[element],
            mu, lambda, role, &first_piola);
        if (geometry && (!geometry[element] || role == 0))
            geometry[element] = PreparedTetGeometry{F, first_piola, cache.JFinvT_cache};
    }
    if (cache_hits) *cache_hits = hits;
}

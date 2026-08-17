#include "volumetric_corotated_energy.h"

#include "third_party/tgsl/ImplicitQRSVD.h"

#include <cmath>
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

} // namespace

Mat33 ElementDs(
    std::size_t element,
    const std::vector<Vec3>& u,
    const std::vector<int>& mesh) {
    CheckElement(element, u, mesh);

    Mat33 result;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            result(c, i) = u[mesh[4 * element + i + 1]][c]
                - u[mesh[4 * element]][c];
        }
    }
    return result;
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
    return ElementDs(element, x, mesh) * state[element].Dm_inverse;
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

void CorotatedCache::UpdateCache(const Mat33& F) {
    if (!F.allFinite()) {
        throw std::invalid_argument("deformation gradient must be finite");
    }

    Mat33 R, S;
    JIXIE::polarDecomposition(F, R, S);

    JFinvT_cache <<
        F(1, 1) * F(2, 2) - F(2, 1) * F(1, 2),
        F(2, 0) * F(1, 2) - F(1, 0) * F(2, 2),
        F(1, 0) * F(2, 1) - F(2, 0) * F(1, 1),
        F(2, 1) * F(0, 2) - F(0, 1) * F(2, 2),
        F(0, 0) * F(2, 2) - F(2, 0) * F(0, 2),
        F(2, 0) * F(0, 1) - F(0, 0) * F(2, 1),
        F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2),
        F(1, 0) * F(0, 2) - F(0, 0) * F(1, 2),
        F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1);

    R_cache <<
        R(0, 0), R(0, 1), R(0, 2),
        R(1, 0), R(1, 1), R(1, 2),
        R(2, 0), R(2, 1), R(2, 2);

    Mat33 D = S.trace() * Mat33::Identity() - S;
    Mat33 Dinv = D.inverse();
    Dinv_cache <<
        Dinv(0, 0), Dinv(0, 1), Dinv(0, 2),
        Dinv(1, 0), Dinv(1, 1), Dinv(1, 2),
        Dinv(2, 0), Dinv(2, 1), Dinv(2, 2);

    J_cache = F.determinant();
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

Mat33 CorotatedCache::P(
    const Mat33& F,
    double mu,
    double lambda) const {
    Mat33 R, JFinvT;
    R <<
        R_cache(0, 0), R_cache(0, 1), R_cache(0, 2),
        R_cache(1, 0), R_cache(1, 1), R_cache(1, 2),
        R_cache(2, 0), R_cache(2, 1), R_cache(2, 2);
    JFinvT <<
        JFinvT_cache(0, 0), JFinvT_cache(0, 1), JFinvT_cache(0, 2),
        JFinvT_cache(1, 0), JFinvT_cache(1, 1), JFinvT_cache(1, 2),
        JFinvT_cache(2, 0), JFinvT_cache(2, 1), JFinvT_cache(2, 2);
    Mat33 first_piola;
    first_piola = 2.0 * mu * (F - R)
        + lambda * (J_cache - 1.0) * JFinvT;
    return first_piola;
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

Vec3 EFEMElementNodeEnergyGradient(
    const CorotatedCache& cache,
    const Mat33& F,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node) {
    CheckLocalNode(local_node);
    Mat33 Pe = cache.P(F, mu, lambda);
    Vec3 gNi;
    gNi << state.grad_N[local_node][0],
        state.grad_N[local_node][1],
        state.grad_N[local_node][2];
    Vec3 g = Pe * gNi * state.measure;
    return g;
}

Mat33 PBGSElementNodeElasticityBlock(
    const CorotatedCache& cache,
    const TetRestData& state,
    double mu,
    double lambda,
    int local_node) {
    CheckLocalNode(local_node);

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
        for (std::size_t beta = 0; beta < 3; ++beta) {
            A(alpha, beta) +=
                lambda * ue(alpha) * ue(beta) * state.measure;
        }
    }

    return A;
}

#include "corotated_energy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

inline int flatF(int a, int b) {
    return 2 * a + b;
}

Mat32 Ds(const TriangleDef& tri) {
    Mat32 out;
    out.col(0) = tri.x[1] - tri.x[0];
    out.col(1) = tri.x[2] - tri.x[0];
    return out;
}

// Shared eigendecomposition cache
CorotatedCache32 buildCorotatedCache(const Mat32& F) {
    CorotatedCache32 c;

    Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in buildCorotatedCache.");
    }

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1e-12);
    evals(1) = std::max(evals(1), 1e-12);

    Mat22 S_diag = Mat22::Zero();
    S_diag(0, 0) = std::sqrt(evals(0));
    S_diag(1, 1) = std::sqrt(evals(1));

    c.S      = U * S_diag * U.transpose();
    c.SInv   = c.S.inverse();
    c.R      = F * c.SInv;
    c.J      = S_diag(0, 0) * S_diag(1, 1);
    c.traceS = c.S.trace();
    c.FTFinv = C.inverse();

    return c;
}

// Cached Psi functions
double PsiCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda) {
    return mu * (F - cache.R).squaredNorm() + 0.5 * lambda * (cache.J - 1.0) * (cache.J - 1.0);
}

Mat32 PCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda) {
    return 2.0 * mu * (F - cache.R) + lambda * (cache.J - 1.0) * cache.J * F * cache.FTFinv;
}

void dPdFCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda, Mat66& dPdF) {
    const Mat22& SInv   = cache.SInv;
    const Mat32& R      = cache.R;
    const Mat22& FTFinv = cache.FTFinv;
    double J      = cache.J;
    double traceS = cache.traceS;

    Mat33 RRT = R * R.transpose();

    Mat32 Re;
    Re(0,0) =  R(0,1); Re(0,1) = -R(0,0);
    Re(1,0) =  R(1,1); Re(1,1) = -R(1,0);
    Re(2,0) =  R(2,1); Re(2,1) = -R(2,0);

    Eigen::Matrix<double, 6, 1> dcdF;
    dcdF << -R(0,1) / traceS,  R(0,0) / traceS,
            -R(1,1) / traceS,  R(1,0) / traceS,
            -R(2,1) / traceS,  R(2,0) / traceS;

    Mat66 dRdF = Mat66::Zero();
    std::array<int, 12> indices = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};

    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            int m = indices[2 * c1];
            int n = indices[2 * c1 + 1];
            int i = indices[2 * c2];
            int j = indices[2 * c2 + 1];

            double v = 0.0;
            if (m == i) v += SInv(j, n);
            v -= RRT(m, i) * SInv(j, n);
            v -= dcdF(c2) * Re(m, n);
            dRdF(c1, c2) = v;
        }
    }

    Mat32 FFTFInv = F * FTFinv;
    Mat33 FFTFInvFT = F * FTFinv * F.transpose();

    dPdF.setZero();

    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            int m = indices[2 * c1];
            int n = indices[2 * c1 + 1];
            int i = indices[2 * c2];
            int j = indices[2 * c2 + 1];

            double v = 0.0;
            if (m == i) v += lambda * (J - 1.0) * J * FTFinv(j, n);

            v -= lambda * (J - 1.0) * J *
                 (FFTFInv(m, j) * FFTFInv(i, n) + FFTFInvFT(m, i) * FTFinv(j, n));

            v += 0.5 * lambda * (2.0 * J - 1.0) * J *
                 (FFTFInv(i, j) * FFTFInv(m, n) + FFTFInv(i, j) * FFTFInv(m, n));

            dPdF(c1, c2) = v;
        }
    }

    dPdF += 2.0 * mu * (Mat66::Identity() - dRdF);
}

// Shape-function helper
static std::array<Vec2, 3> shape_function_grad_from_inv(const Mat22& Dm_inv) {
    std::array<Vec2, 3> grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

// Cached corotated energy functions
double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    CorotatedCache32 cache = buildCorotatedCache(F);
    return ref_area * PsiCorotated32(cache, F, mu, lambda);
}

std::array<Vec3, 3> corotated_node_gradient(const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    const Mat32 P = PCorotated32(cache, F, mu, lambda);
    const auto gradN = shape_function_grad_from_inv(Dm_inv);

    std::array<Vec3, 3> g;
    for (int i = 0; i < 3; ++i) {
        g[i].setZero();
        for (int gamma = 0; gamma < 3; ++gamma) {
            double val = 0.0;
            for (int beta = 0; beta < 2; ++beta)
                val += P(gamma, beta) * gradN[i](beta);
            g[i](gamma) = ref_area * val;
        }
    }
    return g;
}

Mat99 corotated_node_hessian(const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    const auto gradN = shape_function_grad_from_inv(Dm_inv);

    Mat66 dPdF;
    dPdFCorotated32(cache, F, mu, lambda, dPdF);

    Mat99 H = Mat99::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int gamma = 0; gamma < 3; ++gamma) {
                for (int delta = 0; delta < 3; ++delta) {
                    double value = 0.0;
                    for (int beta = 0; beta < 2; ++beta) {
                        for (int eta = 0; eta < 2; ++eta) {
                            value += dPdF(flatF(gamma, beta), flatF(delta, eta)) * gradN[i](beta) * gradN[j](eta);
                        }
                    }
                    H(3 * i + gamma, 3 * j + delta) = ref_area * value;
                }
            }
        }
    }
    return H;
}

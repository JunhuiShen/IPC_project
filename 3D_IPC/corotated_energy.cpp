#include "corotated_energy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

inline int flatF(int a, int b) { return 2 * a + b; }

Mat32 Ds(const TriangleDef& tri) {
    Mat32 out;
    out.col(0) = tri.x[1] - tri.x[0];
    out.col(1) = tri.x[2] - tri.x[0];
    return out;
}

// Build the SVD-based cache
CorotatedCache32 buildCorotatedCache(const Mat32& F) {
    CorotatedCache32 c;

    Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed in buildCorotatedCache.");

    Mat22 U     = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1e-12);
    evals(1) = std::max(evals(1), 1e-12);

    Mat22 S_diag  = Mat22::Zero();
    S_diag(0, 0)  = std::sqrt(evals(0));
    S_diag(1, 1)  = std::sqrt(evals(1));

    c.S         = U * S_diag * U.transpose();
    c.SInv      = c.S.inverse();
    c.R         = F * c.SInv;
    c.J         = S_diag(0, 0) * S_diag(1, 1);
    c.traceS    = c.S.trace();
    c.FTFinv    = C.inverse();

    // Pre-cache these two products used by dPdF
    c.FFTFInv   = F * c.FTFinv;
    c.FFTFInvFT = c.FFTFInv * F.transpose();

    return c;
}

double PsiCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda) {
    return mu * (F - cache.R).squaredNorm()
           + 0.5 * lambda * (cache.J - 1.0) * (cache.J - 1.0);
}

Mat32 PCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda) {
    return 2.0 * mu * (F - cache.R)
           + lambda * (cache.J - 1.0) * cache.J * cache.FFTFInv;
}

// dPdF
void dPdFCorotated32(const CorotatedCache32& cache, double mu, double lambda, Mat66& dPdF) {
    const Mat22& SInv      = cache.SInv;
    const Mat32& R         = cache.R;
    const Mat22& FTFinv    = cache.FTFinv;
    const Mat32& FFTFInv   = cache.FFTFInv;
    const Mat33& FFTFInvFT = cache.FFTFInvFT;
    double J      = cache.J;
    double traceS = cache.traceS;

    Mat33 RRT = R * R.transpose();

    Mat32 Re;
    Re(0,0) =  R(0,1); Re(0,1) = -R(0,0);
    Re(1,0) =  R(1,1); Re(1,1) = -R(1,0);
    Re(2,0) =  R(2,1); Re(2,1) = -R(2,0);

    Vec6 dcdF;
    dcdF << -R(0,1) / traceS,  R(0,0) / traceS,
            -R(1,1) / traceS,  R(1,0) / traceS,
            -R(2,1) / traceS,  R(2,0) / traceS;

    Mat66 dRdF = Mat66::Zero();
    static constexpr int idx[12] = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};

    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            int m = idx[2*c1],   n = idx[2*c1+1];
            int i = idx[2*c2],   j = idx[2*c2+1];
            double v = 0.0;
            if (m == i) v += SInv(j, n);
            v -= RRT(m, i) * SInv(j, n);
            v -= dcdF(c2) * Re(m, n);
            dRdF(c1, c2) = v;
        }
    }

    dPdF.setZero();
    static constexpr int idx2[12] = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};
    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            int m = idx2[2*c1],  n = idx2[2*c1+1];
            int i = idx2[2*c2],  j = idx2[2*c2+1];
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

double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    const CorotatedCache32 cache = buildCorotatedCache(F);
    return ref_area * PsiCorotated32(cache, F, mu, lambda);
}

// Compute shape-function gradients once per triangle — shared by gradient and hessian.
ShapeGrads shape_function_gradients(const Mat22& Dm_inv) {
    ShapeGrads grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

// Single-node gradient
Vec3 corotated_node_gradient(const Mat32& P, double ref_area, const ShapeGrads& gradN, int node) {
    Vec3 g = Vec3::Zero();
    for (int gamma = 0; gamma < 3; ++gamma) {
        double val = 0.0;
        for (int beta = 0; beta < 2; ++beta)
            val += P(gamma, beta) * gradN[node](beta);
        g(gamma) = ref_area * val;
    }
    return g;
}

// Single-node hessian row
Mat39 corotated_node_hessian(const Mat66& dPdF, double ref_area, const ShapeGrads& gradN, int node) {
    Mat39 H_row = Mat39::Zero();
    for (int j = 0; j < 3; ++j) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            for (int delta = 0; delta < 3; ++delta) {
                double value = 0.0;
                for (int beta = 0; beta < 2; ++beta)
                    for (int eta = 0; eta < 2; ++eta)
                        value += dPdF(flatF(gamma, beta), flatF(delta, eta)) * gradN[node](beta) * gradN[j](eta);
                H_row(gamma, 3 * j + delta) = ref_area * value;
            }
        }
    }
    return H_row;
}
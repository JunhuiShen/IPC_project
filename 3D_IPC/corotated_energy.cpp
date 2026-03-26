#include "corotated_energy.h"
#include "Corotated32.h"

#include <cmath>
#include <stdexcept>

inline int flatF(int a, int b) {
    return 2 * a + b;
}

Mat22 Dm(const TriangleRest& tri) {
    Mat22 out;
    out.col(0) = tri.X[1] - tri.X[0];
    out.col(1) = tri.X[2] - tri.X[0];
    return out;
}

Mat32 Ds(const TriangleDef& tri) {
    Mat32 out;
    out.col(0) = tri.x[1] - tri.x[0];
    out.col(1) = tri.x[2] - tri.x[0];
    return out;
}

double rest_area(const TriangleRest& tri) {
    return 0.5 * std::abs(Dm(tri).determinant());
}

Mat32 deformation_gradient(const TriangleRest& rest, const TriangleDef& def) {
    return Ds(def) * Dm(rest).inverse();
}

std::array<Vec2, 3> shape_function_grad(const TriangleRest& rest) {
    Mat22 Dm_inv = Dm(rest).inverse();
    std::array<Vec2, 3> grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

TriangleDef ZeroTriangleDef() {
    TriangleDef out;
    out.x[0].setZero();
    out.x[1].setZero();
    out.x[2].setZero();
    return out;
}

TriangleDef add_scale(const TriangleDef& a, const TriangleDef& b, double s) {
    TriangleDef out;
    for (int i = 0; i < 3; ++i) out.x[i] = a.x[i] + s * b.x[i];
    return out;
}

Vec9 flatten_def(const TriangleDef& def) {
    Vec9 out;
    for (int i = 0; i < 3; ++i) out.segment<3>(3 * i) = def.x[i];
    return out;
}

Vec9 flatten_gradient(const std::array<Vec3, 3>& g) {
    Vec9 out;
    for (int i = 0; i < 3; ++i) out.segment<3>(3 * i) = g[i];
    return out;
}

double get_dof(const TriangleDef& def, int node, int comp) {
    return def.x[node](comp);
}

void set_dof(TriangleDef& def, int node, int comp, double value) {
    def.x[node](comp) = value;
}

double corotated_energy(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda) {
    const double A = rest_area(rest);
    const Mat32 F = deformation_gradient(rest, def);
    return A * TGSL::PsiCorotated32(F, mu, lambda);
}

std::array<Vec3, 3> corotated_node_gradient(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda) {
    const double A = rest_area(rest);
    const Mat32 F = deformation_gradient(rest, def);
    const auto gradN = shape_function_grad(rest);

    Mat32 P;
    TGSL::PCorotated32(F, mu, lambda, P);

    std::array<Vec3, 3> g;
    for (int i = 0; i < 3; ++i) {
        g[i].setZero();
        for (int gamma = 0; gamma < 3; ++gamma) {
            double val = 0.0;
            for (int beta = 0; beta < 2; ++beta) {
                val += P(gamma, beta) * gradN[i](beta);
            }
            g[i](gamma) = A * val;
        }
    }
    return g;
}

Mat99 corotated_node_hessian(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda) {
    const double A = rest_area(rest);
    const Mat32 F = deformation_gradient(rest, def);
    const auto gradN = shape_function_grad(rest);

    Mat66 dPdF;
    TGSL::dPdFCorotated32(F, mu, lambda, dPdF);

    Mat99 H = Mat99::Zero();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int gamma = 0; gamma < 3; ++gamma) {
                for (int delta = 0; delta < 3; ++delta) {
                    double value = 0.0;
                    for (int beta = 0; beta < 2; ++beta) {
                        for (int eta = 0; eta < 2; ++eta) {
                            const int row = flatF(gamma, beta);
                            const int col = flatF(delta, eta);
                            value += dPdF(row, col) * gradN[i](beta) * gradN[j](eta);
                        }
                    }
                    H(3 * i + gamma, 3 * j + delta) = A * value;
                }
            }
        }
    }

    return H;
}
#include "bending_energy.h"

#include <cassert>
#include <cmath>

namespace {

// Cache of geometric quantities shared by theta, gradient, and Hessian.
//   e = x_1 - x_0,  a = x_2 - x_0,  b = x_3 - x_0
//   m_A = e x a,    m_B = b x e    (flat configuration -> m_A || m_B)
//   X = m_A . m_B  (proportional to cos theta)
//   Y = (m_A x m_B) . e_hat  (proportional to sin theta; m_A x m_B is parallel to e)
//   X^2 + Y^2 = |m_A|^2 |m_B|^2, so theta = atan2(Y, X).
struct BendingCache {
    Vec3 e;
    Vec3 a;
    Vec3 b;
    Vec3 mA;
    Vec3 mB;
    Vec3 e_hat;
    double muA2;
    double muB2;
    double ell;
    double X;
    double Y;
    double theta;
    bool degenerate;
};

BendingCache make_bending_cache(const HingeDef& hinge) {
    BendingCache c;
    c.e = hinge.x[1] - hinge.x[0];
    c.a = hinge.x[2] - hinge.x[0];
    c.b = hinge.x[3] - hinge.x[0];
    c.mA = c.e.cross(c.a);
    c.mB = c.b.cross(c.e);
    c.muA2 = c.mA.squaredNorm();
    c.muB2 = c.mB.squaredNorm();
    c.ell = c.e.norm();
    c.degenerate = (c.ell <= 0.0 || c.muA2 <= 0.0 || c.muB2 <= 0.0);

    if (!c.degenerate) {
        c.e_hat = c.e / c.ell;
        c.X = c.mA.dot(c.mB);
        c.Y = c.mA.cross(c.mB).dot(c.e_hat);
        c.theta = std::atan2(c.Y, c.X);
    } else {
        c.e_hat.setZero();
        c.X = 0.0;
        c.Y = 0.0;
        c.theta = 0.0;
    }
    return c;
}

// Partial derivative of X = m_A . m_B with respect to x_node (as a 3-vector).
// Derived by differentiating m_A = e x a and m_B = b x e through e, a, b
// while tracking each node's dependencies.
Vec3 dX_dnode(const BendingCache& c, const HingeDef& hinge, int node) {
    switch (node) {
        case 0: {
            const Vec3 cA = hinge.x[2] - hinge.x[1];  // x_2 - x_1
            const Vec3 cB = hinge.x[3] - hinge.x[1];  // x_3 - x_1
            return c.mB.cross(cA) - c.mA.cross(cB);
        }
        case 1:
            return c.mA.cross(c.b) - c.mB.cross(c.a);
        case 2:
            return c.mB.cross(c.e);
        case 3:
            return c.e.cross(c.mA);  // = -(m_A x e)
    }
    return Vec3::Zero();
}

// Partial derivative of Y = (m_A x m_B) . e_hat with respect to x_node.
// The contribution from d(e_hat) cancels because m_A x m_B is parallel
// to e, so only the d(m_A) and d(m_B) terms survive.
Vec3 dY_dnode(const BendingCache& c, const HingeDef& hinge, int node) {
    switch (node) {
        case 0: {
            const Vec3 cA = hinge.x[2] - hinge.x[1];
            const Vec3 cB = hinge.x[3] - hinge.x[1];
            const double coef = cA.dot(c.mB) + c.mA.dot(cB);
            return coef * c.e_hat - c.e_hat.dot(cA) * c.mB - c.e_hat.dot(cB) * c.mA;
        }
        case 1: {
            const double coef = -(c.a.dot(c.mB) + c.mA.dot(c.b));
            return coef * c.e_hat + c.e_hat.dot(c.a) * c.mB + c.e_hat.dot(c.b) * c.mA;
        }
        case 2:
            return -c.ell * c.mB;
        case 3:
            return -c.ell * c.mA;
    }
    return Vec3::Zero();
}

// Node-wise gradient of theta via chain rule:
//   d theta / d x_node = (X dY/dx_node - Y dX/dx_node) / (|m_A|^2 |m_B|^2)
Vec3 grad_theta_node(const BendingCache& c, const HingeDef& hinge, int node) {
    if (c.degenerate) return Vec3::Zero();
    const Vec3 dX = dX_dnode(c, hinge, node);
    const Vec3 dY = dY_dnode(c, hinge, node);
    return (c.X * dY - c.Y * dX) / (c.muA2 * c.muB2);
}

// Per-node local frame (e, a, m) used by the exact Hessian formula from
// the note, for nodes 2 and 3. For node 3 we use the mirrored parameters
// e_3 = x_0 - x_1, a_3 = x_3 - x_1, so that the same formula applies.
struct NodeFrame {
    Vec3 e;
    Vec3 m;
    double Q;
    double ell;
};

NodeFrame make_node_frame(const HingeDef& hinge, int node) {
    assert(node == 2 || node == 3);
    NodeFrame f;
    if (node == 2) {
        f.e = hinge.x[1] - hinge.x[0];
        const Vec3 a = hinge.x[2] - hinge.x[0];
        f.m = f.e.cross(a);
    } else {
        f.e = hinge.x[0] - hinge.x[1];
        const Vec3 a = hinge.x[3] - hinge.x[1];
        f.m = f.e.cross(a);
    }
    f.Q = f.m.squaredNorm();
    f.ell = f.e.norm();
    return f;
}

Mat33 cross_matrix(const Vec3& v) {
    Mat33 M;
    M <<      0.0, -v(2),  v(1),
            v(2),   0.0, -v(0),
           -v(1),  v(0),   0.0;
    return M;
}

}  // namespace

double bending_theta(const HingeDef& hinge) {
    return make_bending_cache(hinge).theta;
}

double bending_energy(const HingeDef& hinge, double k_B, double c_e, double bar_theta) {
    const double theta = bending_theta(hinge);
    const double delta = theta - bar_theta;
    return k_B * c_e * delta * delta;
}

Vec3 bending_node_gradient(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node) {
    assert(node >= 0 && node < 4);
    const BendingCache c = make_bending_cache(hinge);
    if (c.degenerate) return Vec3::Zero();
    const double delta = c.theta - bar_theta;
    const Vec3 gtheta = grad_theta_node(c, hinge, node);
    return (2.0 * k_B * c_e * delta) * gtheta;
}

Mat33 bending_node_hessian(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node) {
    // Exact self-block Hessian per the note's derivation. Only nodes 2
    // and 3 are covered; for nodes 0 and 1 use bending_node_hessian_psd().
    assert(node == 2 || node == 3);
    const NodeFrame f = make_node_frame(hinge, node);
    if (f.Q <= 0.0) return Mat33::Zero();

    const double theta = bending_theta(hinge);
    const double delta = theta - bar_theta;

    // theta_{,p}     = -ell m_p / Q                              -> gtheta
    // theta_{,pq}    = -ell [ [e]x_{pq}/Q - 2 m_p (m x e)_q / Q^2 ]
    // d^2 w / dx_p dx_q = 2 k_B c_e [ theta_{,p} theta_{,q} + delta * theta_{,pq} ]
    const Vec3 gtheta = (-f.ell / f.Q) * f.m;
    const Mat33 ex = cross_matrix(f.e);
    const Vec3 m_cross_e = f.m.cross(f.e);
    const Mat33 Htheta = -f.ell * (ex / f.Q - (2.0 / (f.Q * f.Q)) * f.m * m_cross_e.transpose());

    Mat33 H = 2.0 * k_B * c_e * (gtheta * gtheta.transpose() + delta * Htheta);
    // The math is symmetric (the antisymmetric [e]x term and the rank-1
    // outer product cancel), but only to infinite precision.
    H = 0.5 * (H + H.transpose());
    return H;
}

Mat33 bending_node_hessian_psd(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node) {
    assert(node >= 0 && node < 4);
    const BendingCache c = make_bending_cache(hinge);
    if (c.degenerate) return Mat33::Zero();
    const Vec3 gtheta = grad_theta_node(c, hinge, node);
    return (2.0 * k_B * c_e) * (gtheta * gtheta.transpose());
}

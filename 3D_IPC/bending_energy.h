#pragma once

#include "IPC_math.h"

// Discrete-shell hinge bending energy (Grinspun et al. 2003).
//
// Layout: nodes 0 and 1 are the shared edge endpoints; node 2 is the
// vertex opposite the edge in triangle A = (0, 1, 2); node 3 is the
// vertex opposite the edge in triangle B = (1, 0, 3). With this
// convention the unnormalized face normals m_A = e x a and
// m_B = b x e (with e = x_1 - x_0, a = x_2 - x_0, b = x_3 - x_0) point
// in the same direction when the hinge is flat, so the dihedral-angle
// complement theta is zero at rest.
struct HingeDef {
    Vec3 x[4];
};

// Signed dihedral-angle complement in (-pi, pi]. Zero when flat.
double bending_theta(const HingeDef& hinge);

// w_e = k_B * c_e * (theta - bar_theta)^2
double bending_energy(const HingeDef& hinge, double k_B, double c_e, double bar_theta);

// Node-wise gradient of w_e w.r.t. x_node, for any node in {0, 1, 2, 3}.
// Computed via the chain rule through the (m_A, m_B, e_hat) parameterization,
// so the formula is unified across all four nodes.
Vec3 bending_node_gradient(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node);

// Exact self-block Hessian following the note's derivation. Only
// defined for the apex nodes (2 and 3); asserts otherwise. Can be
// indefinite when delta != 0 -- use bending_node_hessian_psd() for
// the IPC solver.
Mat33 bending_node_hessian(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node);

// Gauss-Newton (PSD) self-block Hessian for any node 0..3:
//   H^{GN} = 2 k_B c_e * (grad_{x_node} theta)(grad_{x_node} theta)^T.
// Always positive semidefinite; equals the exact Hessian at delta = 0.
// This is the block used by the Gauss-Seidel IPC solver, for which a
// guaranteed-PSD Hessian is required to produce a descent direction.
Mat33 bending_node_hessian_psd(const HingeDef& hinge, double k_B, double c_e, double bar_theta, int node);

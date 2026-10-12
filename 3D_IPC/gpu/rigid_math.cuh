#pragma once

#include "rigid_types.h"

#ifdef __CUDACC__
#define IPC_RIGID_HD __host__ __device__
#else
#define IPC_RIGID_HD
#endif

namespace ipc_gpu {
namespace rigid_detail {

IPC_RIGID_HD inline void multiply(const double* a, const double* b, double* c) {
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            c[3*r+s] = (a[3*r]*b[s] + a[3*r+1]*b[3+s]) + a[3*r+2]*b[6+s];
}

IPC_RIGID_HD inline void transpose_multiply(const double* a, const double* b, double* c) {
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            c[3*r+s] = (a[r]*b[s] + a[3+r]*b[3+s]) + a[6+r]*b[6+s];
}

IPC_RIGID_HD inline void add(double* a, const double* b) {
    for (int k = 0; k < 9; ++k) a[k] += b[k];
}

IPC_RIGID_HD inline void symmetrize(double* a) {
    // Preserve the CPU reference's column-major, in-place symmetrization.
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            a[3*r+c] = 0.5 * (a[3*r+c] + a[3*c+r]);
}

IPC_RIGID_HD inline double inertia_trace(const double* a, const double* b, const double* moment) {
    double product[9];
    transpose_multiply(a, b, product);
    double diagonal[3];
    for (int i = 0; i < 3; ++i)
        diagonal[i] = (product[3*i]*moment[i] + product[3*i+1]*moment[3+i])
            + product[3*i+2]*moment[6+i];
    return (diagonal[0] + diagonal[1]) + diagonal[2];
}

IPC_RIGID_HD inline double skew_entry(int axis, int r, int c) {
    if (r == c || r == axis || c == axis) return 0.0;
    return (r == (axis+1)%3 && c == (axis+2)%3) ? -1.0 : 1.0;
}

} // namespace rigid_detail

IPC_RIGID_HD inline RigidResult evaluate_rigid(const RigidPreparedContact& p) {
    RigidResult result{};
    auto& b = result.barrier;
    const bool tg = p.flags & RigidTranslationGradient;
    const bool og = p.flags & RigidOrientationGradient;
    const bool th = p.flags & RigidTranslationHessian;
    const bool oh = p.flags & RigidOrientationHessian;
    const bool mh = p.flags & RigidMixedHessian;
    const bool upper = p.flags & RigidUpperCrossBlocks;
    // PT COM-only assembly uses its precomputed point self block. The old
    // cross-block sum was overwritten below, so do not read or sum it.
    const bool point_translation = !p.segment_segment && th && !og && !oh && !mh;
    for (int k = 0; k < 9; ++k) b.translation_translation_hessian[k] = p.translation_hessian[k];
    for (int i = 0; i < 4; ++i) {
        const double* g = p.gradients[i];
        const double* J = p.jacobians[i];
        for (int r = 0; r < 3; ++r) {
            if (tg) b.translation_gradient[r] += g[r];
            if (og) b.orientation_gradient[r] += (J[r]*g[0] + J[3+r]*g[1]) + J[6+r]*g[2];
        }
        if ((th || oh || mh) && !point_translation) {
            for (int j = upper ? i : 0; j < 4; ++j) {
                const double* H = p.hessians[i][j];
                if (th) rigid_detail::add(b.translation_translation_hessian, H);
                double temporary[9], contribution[9];
                if (mh) {
                    rigid_detail::multiply(H, p.jacobians[j], temporary);
                    rigid_detail::add(b.translation_orientation_hessian, temporary);
                }
                if (oh) {
                    rigid_detail::transpose_multiply(J, H, temporary);
                    rigid_detail::multiply(temporary, p.jacobians[j], contribution);
                    if (upper && i != j) {
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                b.orientation_orientation_hessian[3*r+c] += contribution[3*r+c] + contribution[3*c+r];
                    } else rigid_detail::add(b.orientation_orientation_hessian, contribution);
                }
            }
        }
        if (oh)
            for (int axis = 0; axis < 3; ++axis)
                for (int k = 0; k < 9; ++k)
                    b.orientation_orientation_hessian[k] += g[axis] * p.curvature[i][axis][k];
    }
    if (th && !og && !p.segment_segment)
        for (int k = 0; k < 9; ++k) b.translation_translation_hessian[k] = p.translation_hessian[k];
    if (th) rigid_detail::symmetrize(b.translation_translation_hessian);
    if (oh) rigid_detail::symmetrize(b.orientation_orientation_hessian);

    if (!(p.flags & RigidFriction)) return result;
    auto& f = result.friction;
    double weight = 0.0, FJ[9]{};
    for (int i = 0; i < 4; ++i) {
        weight += p.friction_weights[i];
        if (og)
            for (int k = 0; k < 9; ++k) FJ[k] += p.friction_weights[i] * p.jacobians[i][k];
    }
    double HJ[9];
    if (mh || oh) rigid_detail::multiply(p.friction_hessian, FJ, HJ);
    const bool translation = p.update_mask & 1u;
    const bool orientation = p.update_mask & 2u;
    if (oh && orientation) rigid_detail::transpose_multiply(FJ, HJ, f.orientation_orientation_hessian);
    for (int r = 0; r < 3; ++r) {
        if (tg && translation) f.translation_gradient[r] = weight * p.friction_gradient[r];
        if (og && orientation)
            f.orientation_gradient[r] = (FJ[r]*p.friction_gradient[0] + FJ[3+r]*p.friction_gradient[1])
                + FJ[6+r]*p.friction_gradient[2];
        for (int c = 0; c < 3; ++c) {
            if (th && translation) f.translation_translation_hessian[3*r+c] = (weight * p.friction_hessian[3*r+c]) * weight;
            if (mh && translation && orientation) f.translation_orientation_hessian[3*r+c] = weight * HJ[3*r+c];
        }
    }
    // Frozen friction intentionally has no quaternion-curvature contribution.
    return result;
}

IPC_RIGID_HD inline Derivatives evaluate_rigid_inertia(const RigidInertia& p) {
    Derivatives result{};
    if (!p.rotational) {
        for (int r = 0; r < 3; ++r) {
            result.gradient[r] = p.mass * (p.position[r] - p.previous_position[r] - p.dt*p.previous_velocity[r]);
            result.hessian[3*r+r] = p.mass;
        }
        result.gradient[1] -= p.dt * p.dt * p.mass * p.gravity;
        return result;
    }

    const double* q = p.orientation;
    const double vector_norm2 = (q[1]*q[1] + q[2]*q[2]) + q[3]*q[3];
    double D[9], G[4][9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double skew = 0.0;
            for (int l = 0; l < 3; ++l) skew += q[l+1] * rigid_detail::skew_entry(l, r, c);
            const double identity = r == c ? 1.0 : 0.0;
            D[3*r+c] = ((q[0]*q[0] - vector_norm2)*identity + 2.0*q[r+1]*q[c+1])
                + 2.0*q[0]*skew - p.rotation_predictor[3*r+c];
            G[0][3*r+c] = 2.0*q[0]*identity + 2.0*skew;
            for (int l = 0; l < 3; ++l)
                G[l+1][3*r+c] = ((-2.0*q[l+1]*identity + 2.0*(r == l)*q[c+1])
                    + 2.0*q[r+1]*(c == l)) + 2.0*q[0]*rigid_detail::skew_entry(l,r,c);
        }
    }
    double gradient_q[4], hessian_q[4][4];
    for (int alpha = 0; alpha < 4; ++alpha) {
        gradient_q[alpha] = rigid_detail::inertia_trace(D, G[alpha], p.second_moment);
        for (int beta = 0; beta < 4; ++beta) {
            double K[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (alpha == 0 && beta == 0) K[3*r+c] = 2.0*(r == c);
                    else if (alpha == 0 || beta == 0)
                        K[3*r+c] = 2.0*rigid_detail::skew_entry((alpha == 0 ? beta : alpha)-1,r,c);
                    else K[3*r+c] = -2.0*(alpha == beta)*(r == c)
                        + 2.0*(r == alpha-1)*(c == beta-1) + 2.0*(r == beta-1)*(c == alpha-1);
                }
            }
            hessian_q[alpha][beta] = rigid_detail::inertia_trace(G[alpha], G[beta], p.second_moment)
                + rigid_detail::inertia_trace(D, K, p.second_moment);
        }
    }
    for (int beta = 0; beta < 3; ++beta) {
        for (int alpha = 0; alpha < 4; ++alpha)
            result.gradient[beta] += gradient_q[alpha] * p.orientation_jacobian[alpha][beta];
        for (int kappa = 0; kappa < 3; ++kappa) {
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int delta = 0; delta < 4; ++delta)
                    result.hessian[3*beta+kappa] += p.orientation_jacobian[alpha][beta]
                        * hessian_q[alpha][delta] * p.orientation_jacobian[delta][kappa];
                result.hessian[3*beta+kappa] += gradient_q[alpha] * p.orientation_hessians[alpha][3*beta+kappa];
            }
        }
    }
    return result;
}

} // namespace ipc_gpu

#undef IPC_RIGID_HD

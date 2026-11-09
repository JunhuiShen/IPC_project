#pragma once
// Acceptance oracle: qualified evaluator definition; only public function name changed.
#include "original_contact.cuh"
#if defined(__CUDACC__)
#define IPC_ORIGINAL_CONTACT_HD __host__ __device__
#else
#define IPC_ORIGINAL_CONTACT_HD
#endif
namespace ipc_gpu::solver {
IPC_ORIGINAL_CONTACT_HD inline ContactOutput evaluate_contact_original_qualified_oracle(
        const ContactInput& in) {
    using namespace original_contact_detail;
    ContactOutput out{};

    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (!finite(in.positions[i][k]) || !finite(in.previous[i][k])) {
                out.status = ContactStatus::InvalidInput;
                return out;
            }
        }
    }
    if (!finite(in.d_hat) || !(in.d_hat > 0.0)
            || !finite(in.k_barrier) || in.k_barrier < 0.0
            || !finite(in.dt) || !(in.dt > 0.0)
            || !finite(in.friction) || in.friction < 0.0
            || !finite(in.eps_v) || !(in.eps_v > 0.0)
            || !finite(in.eps) || !(in.eps > 0.0)
            || (in.kind != ContactKind::PointTriangle
                && in.kind != ContactKind::EdgeEdge)
            || (in.hessian_blocks & ~0xffffu)) {
        out.status = ContactStatus::InvalidInput;
        return out;
    }
    // The CPU friction builder requires a finite positive dt*eps_v, and the
    // frozen friction derivatives require a finite dt^2. Preserve those
    // validations independently of whether friction is currently zero.
    const double eps_u = in.dt * in.eps_v;
    const double dt2 = in.dt * in.dt;
    if (!finite(eps_u) || !(eps_u > 0.0) || !finite(dt2)) {
        out.status = ContactStatus::InvalidInput;
        return out;
    }

    ContactGeometry geometry;
    geometry.point_triangle = in.kind == ContactKind::PointTriangle;
    if (geometry.point_triangle) {
        point_triangle_distance(in.positions, in.eps, geometry.pt);
        geometry.resolved_region = geometry.pt.region == PtDegenerateTriangle
                ? resolve_degenerate_triangle_region(in.positions)
                : geometry.pt.region;
        point_triangle_weights(in.positions, geometry.pt,
                               geometry.resolved_region, geometry.weights);
        geometry.distance = geometry.pt.distance;
        if (geometry.resolved_region == PtFaceInterior) {
            const double sign = sign_of(geometry.pt.phi);
            for (int k = 0; k < 3; ++k)
                geometry.normal[k] = sign * geometry.pt.normal[k];
        } else if (geometry.distance > 0.0) {
            for (int k = 0; k < 3; ++k)
                geometry.normal[k] = (in.positions[0][k] - geometry.pt.closest[k])
                        / geometry.distance;
        }
    } else {
        segment_segment_distance_fast(in.positions, in.eps, geometry.ee);
        geometry.resolved_region = geometry.ee.region == EeParallelSegments
                ? resolve_parallel_segment_region(geometry.ee.s, geometry.ee.t)
                : geometry.ee.region;
        segment_segment_weights(geometry.ee, geometry.resolved_region,
                                geometry.weights);
        geometry.distance = geometry.ee.distance;
        if (geometry.distance > 0.0) {
            for (int k = 0; k < 3; ++k)
                geometry.normal[k] = geometry.ee.separation[k]
                        / geometry.distance;
        }
    }

    out.distance = geometry.distance;
    for (int i = 0; i < 4; ++i) out.weights[i] = geometry.weights[i];
    for (int k = 0; k < 3; ++k) out.normal[k] = geometry.normal[k];
    for (int i = 0; i < 4; ++i) {
        if (!finite(out.weights[i])) {
            out.status = ContactStatus::NonFiniteDerivative;
            return out;
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (!finite(out.normal[k])) {
            out.status = ContactStatus::NonFiniteDerivative;
            return out;
        }
    }

    // The CPU evaluation validates the computed distance before asking for
    // barrier coefficients. A non-finite or negative legacy distance is a
    // fail-closed condition, never an inactive contact.
    if (!finite(out.distance) || out.distance < 0.0) {
        out.status = ContactStatus::NonFiniteDerivative;
        return out;
    }
    // CPU initialize_contact_barrier_data returns immediately for zero
    // stiffness, before any singular scalar derivative is requested. A
    // zero-stiffness packet is therefore an inactive contact (zero force and
    // zero friction) even when the geometry is exactly coincident. This must
    // be decided BEFORE treating distance == 0 as an error: the CPU false
    // evaluation accepts a valid zero distance when k_barrier == 0, so
    // rejecting it here would diverge from the oracle. Unused derivative and
    // friction payload stays identically zero; no force is consumed when
    // active == 0.
    if (in.k_barrier == 0.0) {
        out.active = 0;
        return out;
    }
    if (out.distance == 0.0) {
        out.status = ContactStatus::ZeroDistance;
        return out;
    }

    double bp = 0.0;
    double bpp = 0.0;
    bool barrier_active = false;
    // k_barrier is strictly positive here: the zero-stiffness packet returned
    // above, exactly like the CPU initialize_contact_barrier_data.
    if (in.d_hat > out.distance) {
        if (!scalar_barrier_derivatives(out.distance, in.d_hat, bp, bpp)) {
            out.status = ContactStatus::NonFiniteDerivative;
            return out;
        }
        barrier_active = true;
    }
    const double normal_load = barrier_active
            ? -in.k_barrier * bp : 0.0;
    if (!finite(normal_load)) {
        out.status = ContactStatus::NonFiniteDerivative;
        return out;
    }
    out.active = barrier_active ? 1u : 0u;

    if (barrier_active) {
        for (int i = 0; i < 4; ++i) {
            if (geometry.point_triangle) {
                point_triangle_gradient(in.positions, geometry.pt,
                                        geometry.resolved_region,
                                        out.distance, bp, i,
                                        out.barrier_gradient[i]);
            } else {
                segment_segment_gradient(in.positions, geometry.ee,
                                         geometry.resolved_region,
                                         out.distance, bp, i,
                                         out.barrier_gradient[i]);
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!(in.hessian_blocks & (1u << (4 * i + j)))) continue;
                double block[3][3];
                if (geometry.point_triangle) {
                    point_triangle_hessian(in.positions, geometry.pt,
                                           geometry.resolved_region,
                                           out.distance, bp, bpp, i, j, block);
                } else {
                    segment_segment_hessian(in.positions,
                                            geometry.resolved_region,
                                            out.distance, bp, bpp, i, j,
                                            block);
                }
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        out.barrier_hessian[i][j][3 * k + l] = block[k][l];
                    }
                }
            }
        }
    }

    // Frozen friction uses the CPU false-path geometry convention: the
    // tangent frame normal is the rounded closest-point separation direction,
    // not the face barrier normal. In particular the CPU's robust face-normal
    // substitution is NOT used here.
    if (barrier_active && normal_load > 0.0 && in.friction != 0.0
            && dt2 != 0.0) {
        double direction[3];
        if (geometry.point_triangle) {
            for (int k = 0; k < 3; ++k)
                direction[k] = in.positions[0][k] - geometry.pt.closest[k];
        } else {
            for (int k = 0; k < 3; ++k)
                direction[k] = geometry.ee.separation[k];
        }
        const double separation_norm = norm3(direction);
        if (!(separation_norm > 0.0) || !finite(separation_norm)) {
            out.status = ContactStatus::NonFiniteDerivative;
            out.active = 0;
            return out;
        }
        double friction_normal[3];
        for (int k = 0; k < 3; ++k)
            friction_normal[k] = direction[k] / separation_norm;

        double projector[3][3];
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                projector[k][l] = (k == l ? 1.0 : 0.0)
                        - friction_normal[k] * friction_normal[l];
            }
        }
        double displacement[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 3; ++k) {
                displacement[k] += geometry.weights[i]
                        * (in.positions[i][k] - in.previous[i][k]);
            }
        }
        if (!finite(displacement[0]) || !finite(displacement[1])
                || !finite(displacement[2])) {
            out.status = ContactStatus::NonFiniteDerivative;
            out.active = 0;
            return out;
        }
        double tangential[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l)
                tangential[k] += projector[k][l] * displacement[l];
        }
        if (!finite(tangential[0]) || !finite(tangential[1])
                || !finite(tangential[2])) {
            out.status = ContactStatus::NonFiniteDerivative;
            out.active = 0;
            return out;
        }
        const double slip = norm3(tangential);
        if (!finite(slip)) {
            out.status = ContactStatus::NonFiniteDerivative;
            out.active = 0;
            return out;
        }
        const double mollifier = slip < eps_u
                ? 2.0 / eps_u - slip / (eps_u * eps_u)
                : 1.0 / slip;
        const double scale = dt2 * in.friction * normal_load * mollifier;
        if (!finite(scale)) {
            out.status = ContactStatus::NonFiniteDerivative;
            out.active = 0;
            return out;
        }
        for (int k = 0; k < 3; ++k) {
            out.friction_relative_gradient[k] = scale * tangential[k];
            for (int l = 0; l < 3; ++l) {
                out.friction_relative_hessian[3 * k + l] =
                        scale * projector[k][l];
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (!finite(out.barrier_gradient[i][k])
                    || !finite(out.friction_relative_gradient[k])) {
                out.status = ContactStatus::NonFiniteDerivative;
            }
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 3; ++l) {
                    if (!finite(out.barrier_hessian[i][j][3 * k + l])
                            || !finite(out.friction_relative_hessian[3 * k + l])) {
                        out.status = ContactStatus::NonFiniteDerivative;
                    }
                }
            }
        }
    }
    if (out.status != ContactStatus::Ok) {
        out.active = 0;
        return out;
    }
    return out;
}
}
#undef IPC_ORIGINAL_CONTACT_HD

#pragma once

// Independent port of the ORIGINAL CPU contact mathematics used when
// exact_computation_fallback == false.
//
// This is deliberately separate from contact.cuh's ordinary evaluator and
// from contact_exact.cuh. It does not call any host geometry/barrier/friction
// routine and never substitutes the robust exact feature: the segment--segment
// fast solve is returned immediately, exactly like the CPU false path.
//
// Branch-specific CPU formulas are reproduced directly instead of being
// expressed through a generic Schur complement. In particular:
//   * point--triangle unconstrained interior includes barycentric boundaries;
//   * the CPU point--triangle exterior loop uses strict (<) first-edge ties;
//   * segment--segment inclusive 0 <= s,t <= 1 is Interior, even at an
//     endpoint, and retains both free parameters;
//   * segment--segment exterior classification has a 1e-14 endpoint tolerance;
//   * parallel/degenerate segments resolve the four boundary candidates to an
//     edge/corner only after the distance solve;
//   * the CPU point--triangle degenerate triangle and exterior edge branches
//     select features with their own tie/endpoint rules;
//   * barrier and friction formulas follow the CPU branch used for that
//     feature, including zero-weight roles with nonzero Hessian blocks.
//   * zero stiffness is the CPU no-force packet and is inactive even at an
//     exact zero distance, while positive stiffness at zero distance keeps
//     the singular-derivative failure (status ZeroDistance).
//
// All arithmetic is ordinary double precision on the device. No CPU callback,
// Eigen, Boost, host heap, or device heap is used.

#include "contact.cuh"

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_ORIGINAL_CONTACT_HD __host__ __device__
#else
#define IPC_ORIGINAL_CONTACT_HD
#endif

namespace ipc_gpu {
namespace solver {
namespace original_contact_detail {

constexpr double kFeatureTolerance = 1.0e-14;

// CPU NodeTriangleRegion values.
enum PtRegion : int {
    PtFaceInterior = 0,
    PtEdge12,
    PtEdge23,
    PtEdge31,
    PtVertex1,
    PtVertex2,
    PtVertex3,
    PtDegenerateTriangle
};

// CPU SegmentSegmentRegion values.
enum EeRegion : int {
    EeInterior = 0,
    EeEdgeS0,
    EeEdgeS1,
    EeEdgeT0,
    EeEdgeT1,
    EeCornerS0T0,
    EeCornerS0T1,
    EeCornerS1T0,
    EeCornerS1T1,
    EeParallelSegments
};

IPC_ORIGINAL_CONTACT_HD inline bool finite(double a) {
    return a <= DBL_MAX && a >= -DBL_MAX;
}

IPC_ORIGINAL_CONTACT_HD inline double absolute(double a) {
    return a < 0.0 ? -a : a;
}

IPC_ORIGINAL_CONTACT_HD inline double clamp_unit(double t) {
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

IPC_ORIGINAL_CONTACT_HD inline double dot3(const double* a, const double* b) {
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

IPC_ORIGINAL_CONTACT_HD inline double norm3(const double* a) {
    return sqrt(dot3(a, a));
}

IPC_ORIGINAL_CONTACT_HD inline void sub3(const double* a, const double* b,
                                         double* c) {
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

IPC_ORIGINAL_CONTACT_HD inline void cross3(const double* a, const double* b,
                                           double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

IPC_ORIGINAL_CONTACT_HD inline int levi_civita(int i, int j, int k) {
    if (i == j || j == k || i == k) return 0;
    return ((i == 0 && j == 1 && k == 2)
            || (i == 1 && j == 2 && k == 0)
            || (i == 2 && j == 0 && k == 1)) ? 1 : -1;
}

IPC_ORIGINAL_CONTACT_HD inline double sign_of(double value) {
    return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

// Explicit one-rounding subtract-product used by the interior-Hessian
// first derivatives. The native normal CPU oracle requires this rounding in
// the small-Gram cancellation cases. Global CUDA contraction stays disabled.
IPC_ORIGINAL_CONTACT_HD inline double fused_subtract_product(
        double factor_a, double factor_b, double addend) {
#if defined(__CUDACC__)
    return ::fma(-factor_a, factor_b, addend);
#else
    return std::fma(-factor_a, factor_b, addend);
#endif
}

// A separately rounded right product and one fused left product. Keep this
// rounding boundary local to the Hessian recomputation: its legacy internal
// parameters can differ from the public distance feature's stored s and t.
IPC_ORIGINAL_CONTACT_HD inline double hessian_product_difference(
        double left_a, double left_b, double right_a, double right_b) {
    const double right_product = right_a * right_b;
#if defined(__CUDACC__)
    return ::fma(left_a, left_b, -right_product);
#else
    return std::fma(left_a, left_b, -right_product);
#endif
}

// CPU segment_closest_point (IPC_math.cpp), including the t=0 branch for a
// zero-length segment.
IPC_ORIGINAL_CONTACT_HD inline void segment_closest_point(
        const double* x, const double* a, const double* b,
        double* closest, double& t) {
    double ab[3];
    sub3(b, a, ab);
    const double ab2 = dot3(ab, ab);
    if (ab2 <= 0.0) {
        t = 0.0;
        closest[0] = a[0];
        closest[1] = a[1];
        closest[2] = a[2];
        return;
    }
    double w[3];
    sub3(x, a, w);
    t = clamp_unit(dot3(w, ab) / ab2);
    closest[0] = a[0] + t * ab[0];
    closest[1] = a[1] + t * ab[1];
    closest[2] = a[2] + t * ab[2];
}

// CPU segment_parameter_from_closest_point (barrier_energy.cpp). Its
// per-coordinate accumulation order and clamp are preserved.
IPC_ORIGINAL_CONTACT_HD inline double segment_parameter_from_closest_point(
        const double* q, const double* a, const double* b) {
    double denom = 0.0;
    double numer = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double ab_k = b[k] - a[k];
        denom += ab_k * ab_k;
        numer += (q[k] - a[k]) * ab_k;
    }
    if (denom <= 0.0) return 0.0;
    return clamp_unit(numer / denom);
}

struct PtGeometry {
    double closest[3]{};
    double tilde[3]{};
    double normal[3]{};
    double bary[3]{};
    double phi = 0.0;
    double distance = 0.0;
    int region = PtDegenerateTriangle;  // public region
};

struct EeGeometry {
    double closest_point_1[3]{};
    double closest_point_2[3]{};
    double s = 0.0;
    double t = 0.0;
    double distance = 0.0;
    int region = EeParallelSegments;  // public region
    double separation[3]{};
};

struct ContactGeometry {
    double distance = 0.0;
    double weights[4]{};
    double normal[3]{};
    bool point_triangle = true;

    PtGeometry pt;
    EeGeometry ee;
    int resolved_region = PtDegenerateTriangle;
};

// ---------------------------------------------------------------------------
// Point--triangle distance (node_triangle_distance.cpp)
// ---------------------------------------------------------------------------

IPC_ORIGINAL_CONTACT_HD inline void point_triangle_distance(
        const double x[4][3], double eps, PtGeometry& out) {
    const double* node = x[0];
    const double* x1 = x[1];
    const double* x2 = x[2];
    const double* x3 = x[3];

    double e12[3], e13[3], n_raw[3];
    sub3(x2, x1, e12);
    sub3(x3, x1, e13);
    cross3(e12, e13, n_raw);
    const double n_norm = norm3(n_raw);

    if (n_norm <= eps) {
        double t12 = 0.0, t23 = 0.0, t31 = 0.0;
        double q12[3], q23[3], q31[3];
        segment_closest_point(node, x1, x2, q12, t12);
        segment_closest_point(node, x2, x3, q23, t23);
        segment_closest_point(node, x3, x1, q31, t31);

        double r[3];
        sub3(node, q12, r);
        const double d12 = norm3(r);
        sub3(node, q23, r);
        const double d23 = norm3(r);
        sub3(node, q31, r);
        const double d31 = norm3(r);

        if (d12 <= d23 && d12 <= d31) {
            out.closest[0] = q12[0];
            out.closest[1] = q12[1];
            out.closest[2] = q12[2];
            out.distance = d12;
        } else if (d23 <= d12 && d23 <= d31) {
            out.closest[0] = q23[0];
            out.closest[1] = q23[1];
            out.closest[2] = q23[2];
            out.distance = d23;
        } else {
            out.closest[0] = q31[0];
            out.closest[1] = q31[1];
            out.closest[2] = q31[2];
            out.distance = d31;
        }
        out.tilde[0] = out.closest[0];
        out.tilde[1] = out.closest[1];
        out.tilde[2] = out.closest[2];
        out.region = PtDegenerateTriangle;
        return;
    }

    double n[3];
    n[0] = n_raw[0] / n_norm;
    n[1] = n_raw[1] / n_norm;
    n[2] = n_raw[2] / n_norm;

    double w[3];
    sub3(node, x1, w);
    const double phi = dot3(n, w);
    for (int k = 0; k < 3; ++k) out.tilde[k] = node[k] - phi * n[k];

    out.normal[0] = n[0];
    out.normal[1] = n[1];
    out.normal[2] = n[2];
    out.phi = phi;

    const double area_squared = dot3(n_raw, n_raw);
    double r[3], cr[3];
    sub3(out.tilde, x1, r);
    cross3(r, e13, cr);
    const double l2 = dot3(cr, n_raw) / area_squared;
    cross3(e12, r, cr);
    const double l3 = dot3(cr, n_raw) / area_squared;
    const double l1 = 1.0 - l2 - l3;
    out.bary[0] = l1;
    out.bary[1] = l2;
    out.bary[2] = l3;

    if (l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0) {
        out.closest[0] = out.tilde[0];
        out.closest[1] = out.tilde[1];
        out.closest[2] = out.tilde[2];
        out.distance = absolute(phi);
        out.region = PtFaceInterior;
        return;
    }

    // Outside the face: minimize over the three closed segments. Strict <
    // preserves the CPU first-edge tie rule; parameter 0/1 selects vertices.
    const double* vertices[3] = {x1, x2, x3};
    const int edges[3] = {PtEdge12, PtEdge23, PtEdge31};
    const int corners[3] = {PtVertex1, PtVertex2, PtVertex3};
    double best_squared = DBL_MAX;
    for (int a = 0; a < 3; ++a) {
        const int b = (a + 1) % 3;
        double parameter = 0.0;
        double candidate[3];
        segment_closest_point(node, vertices[a], vertices[b], candidate,
                              parameter);
        double diff[3];
        sub3(node, candidate, diff);
        const double squared = dot3(diff, diff);
        if (squared < best_squared) {
            best_squared = squared;
            out.closest[0] = candidate[0];
            out.closest[1] = candidate[1];
            out.closest[2] = candidate[2];
            out.region = parameter <= 0.0 ? corners[a]
                    : parameter >= 1.0 ? corners[b] : edges[a];
        }
    }
    double diff[3];
    sub3(node, out.closest, diff);
    out.distance = norm3(diff);
}

// resolve_degenerate_triangle_region (barrier_energy.cpp).
IPC_ORIGINAL_CONTACT_HD inline int resolve_degenerate_triangle_region(
        const double x[4][3]) {
    const double* node = x[0];
    const double* x1 = x[1];
    const double* x2 = x[2];
    const double* x3 = x[3];

    double t12 = 0.0, t23 = 0.0, t31 = 0.0;
    double q12[3], q23[3], q31[3];
    segment_closest_point(node, x1, x2, q12, t12);
    segment_closest_point(node, x2, x3, q23, t23);
    segment_closest_point(node, x3, x1, q31, t31);

    double r[3];
    sub3(node, q12, r);
    const double d12 = norm3(r);
    sub3(node, q23, r);
    const double d23 = norm3(r);
    sub3(node, q31, r);
    const double d31 = norm3(r);

    double parameter = 0.0;
    int edge = PtEdge12;
    int vertex_a = PtVertex1;
    int vertex_b = PtVertex2;
    if (d12 <= d23 && d12 <= d31) {
        parameter = t12;
    } else if (d23 <= d12 && d23 <= d31) {
        parameter = t23;
        edge = PtEdge23;
        vertex_a = PtVertex2;
        vertex_b = PtVertex3;
    } else {
        parameter = t31;
        edge = PtEdge31;
        vertex_a = PtVertex3;
        vertex_b = PtVertex1;
    }
    if (parameter <= kFeatureTolerance) return vertex_a;
    if (parameter >= 1.0 - kFeatureTolerance) return vertex_b;
    return edge;
}

IPC_ORIGINAL_CONTACT_HD inline void point_triangle_weights(
        const double x[4][3], const PtGeometry& dr, int region,
        double weights[4]) {
    weights[0] = weights[1] = weights[2] = weights[3] = 0.0;
    switch (region) {
        case PtFaceInterior:
            weights[0] = 1.0;
            weights[1] = -dr.bary[0];
            weights[2] = -dr.bary[1];
            weights[3] = -dr.bary[2];
            break;
        case PtEdge12: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[1], x[2]);
            weights[0] = 1.0;
            weights[1] = -(1.0 - t);
            weights[2] = -t;
            break;
        }
        case PtEdge23: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[2], x[3]);
            weights[0] = 1.0;
            weights[2] = -(1.0 - t);
            weights[3] = -t;
            break;
        }
        case PtEdge31: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[3], x[1]);
            weights[0] = 1.0;
            weights[1] = -t;
            weights[3] = -(1.0 - t);
            break;
        }
        case PtVertex1:
            weights[0] = 1.0;
            weights[1] = -1.0;
            break;
        case PtVertex2:
            weights[0] = 1.0;
            weights[2] = -1.0;
            break;
        case PtVertex3:
            weights[0] = 1.0;
            weights[3] = -1.0;
            break;
        default:
            // DegenerateTriangle was resolved before this call.
            break;
    }
}

// ---------------------------------------------------------------------------
// Segment--segment distance (segment_segment_distance_fast)
// ---------------------------------------------------------------------------

IPC_ORIGINAL_CONTACT_HD inline double optimal_t_for_fixed_s(
        const double* x1, const double* a, const double* x3,
        const double* b, double s, double C, double& t_out) {
    double p[3];
    for (int k = 0; k < 3; ++k) p[k] = x1[k] + s * a[k];
    if (C <= 0.0) {
        t_out = 0.0;
    } else {
        double d[3];
        sub3(p, x3, d);
        t_out = clamp_unit(dot3(d, b) / C);
    }
    double q[3];
    for (int k = 0; k < 3; ++k) q[k] = x3[k] + t_out * b[k];
    double diff[3];
    sub3(p, q, diff);
    return norm3(diff);
}

IPC_ORIGINAL_CONTACT_HD inline double optimal_s_for_fixed_t(
        const double* x1, const double* a, const double* x3,
        const double* b, double t, double A, double& s_out) {
    double q[3];
    for (int k = 0; k < 3; ++k) q[k] = x3[k] + t * b[k];
    if (A <= 0.0) {
        s_out = 0.0;
    } else {
        double d[3];
        sub3(q, x1, d);
        s_out = clamp_unit(dot3(d, a) / A);
    }
    double p[3];
    for (int k = 0; k < 3; ++k) p[k] = x1[k] + s_out * a[k];
    double diff[3];
    sub3(p, q, diff);
    return norm3(diff);
}

IPC_ORIGINAL_CONTACT_HD inline int classify_segment_parameters(
        double s, double t) {
    const double tol = kFeatureTolerance;
    const bool s_at_0 = s <= tol;
    const bool s_at_1 = s >= 1.0 - tol;
    const bool t_at_0 = t <= tol;
    const bool t_at_1 = t >= 1.0 - tol;

    if (s_at_0 && t_at_0) return EeCornerS0T0;
    if (s_at_0 && t_at_1) return EeCornerS0T1;
    if (s_at_1 && t_at_0) return EeCornerS1T0;
    if (s_at_1 && t_at_1) return EeCornerS1T1;
    if (s_at_0) return EeEdgeS0;
    if (s_at_1) return EeEdgeS1;
    if (t_at_0) return EeEdgeT0;
    if (t_at_1) return EeEdgeT1;
    return EeInterior;
}

IPC_ORIGINAL_CONTACT_HD inline void segment_segment_distance_fast(
        const double x[4][3], double eps, EeGeometry& out) {
    const double* x1 = x[0];
    const double* x2 = x[1];
    const double* x3 = x[2];
    const double* x4 = x[3];

    double a[3], b[3], c[3];
    sub3(x2, x1, a);
    sub3(x4, x3, b);
    sub3(x1, x3, c);

    const double A = dot3(a, a);
    const double B = dot3(a, b);
    const double C = dot3(b, b);
    const double D = dot3(a, c);
    const double E = dot3(b, c);
    const double Delta = A * C - B * B;

    if (Delta > eps * eps) {
        const double s_unc = (B * E - C * D) / Delta;
        const double t_unc = (A * E - B * D) / Delta;
        if (s_unc >= 0.0 && s_unc <= 1.0
                && t_unc >= 0.0 && t_unc <= 1.0) {
            out.s = s_unc;
            out.t = t_unc;
            for (int k = 0; k < 3; ++k) {
                out.closest_point_1[k] = x1[k] + out.s * a[k];
                out.closest_point_2[k] = x3[k] + out.t * b[k];
                out.separation[k] =
                        out.closest_point_1[k] - out.closest_point_2[k];
            }
            out.distance = norm3(out.separation);
            out.region = EeInterior;
            return;
        }

        double best_dist = DBL_MAX;
        double best_s = 0.0, best_t = 0.0;
        {   // Edge s = 0
            double t_cand = 0.0;
            const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C,
                                                   t_cand);
            if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t_cand; }
        }
        {   // Edge s = 1
            double t_cand = 0.0;
            const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C,
                                                   t_cand);
            if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t_cand; }
        }
        {   // Edge t = 0
            double s_cand = 0.0;
            const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A,
                                                   s_cand);
            if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 0.0; }
        }
        {   // Edge t = 1
            double s_cand = 0.0;
            const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A,
                                                   s_cand);
            if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 1.0; }
        }

        out.s = best_s;
        out.t = best_t;
        for (int k = 0; k < 3; ++k) {
            out.closest_point_1[k] = x1[k] + out.s * a[k];
            out.closest_point_2[k] = x3[k] + out.t * b[k];
            out.separation[k] =
                    out.closest_point_1[k] - out.closest_point_2[k];
        }
        out.distance = best_dist;
        out.region = classify_segment_parameters(out.s, out.t);
        return;
    }

    // Parallel / degenerate case: still check all four boundary edges.
    out.region = EeParallelSegments;
    double best_dist = DBL_MAX;
    double best_s = 0.0, best_t = 0.0;
    {   // Edge s = 0
        double t_cand = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C, t_cand);
        if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t_cand; }
    }
    {   // Edge s = 1
        double t_cand = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C, t_cand);
        if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t_cand; }
    }
    {   // Edge t = 0
        double s_cand = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A, s_cand);
        if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 0.0; }
    }
    {   // Edge t = 1
        double s_cand = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A, s_cand);
        if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 1.0; }
    }

    out.s = best_s;
    out.t = best_t;
    for (int k = 0; k < 3; ++k) {
        out.closest_point_1[k] = x1[k] + out.s * a[k];
        out.closest_point_2[k] = x3[k] + out.t * b[k];
        out.separation[k] = out.closest_point_1[k] - out.closest_point_2[k];
    }
    out.distance = best_dist;
}

// resolve_parallel_segment_region (barrier_energy.cpp).
IPC_ORIGINAL_CONTACT_HD inline int resolve_parallel_segment_region(
        double s, double t) {
    const double tol = kFeatureTolerance;
    const bool s0 = s <= tol;
    const bool s1 = s >= 1.0 - tol;
    const bool t0 = t <= tol;
    const bool t1 = t >= 1.0 - tol;

    if (s0 && t0) return EeCornerS0T0;
    if (s0 && t1) return EeCornerS0T1;
    if (s1 && t0) return EeCornerS1T0;
    if (s1 && t1) return EeCornerS1T1;
    if (s0) return EeEdgeS0;
    if (s1) return EeEdgeS1;
    if (t0) return EeEdgeT0;
    if (t1) return EeEdgeT1;
    return EeParallelSegments;
}

IPC_ORIGINAL_CONTACT_HD inline void segment_segment_weights(
        const EeGeometry& dr, int region, double weights[4]) {
    weights[0] = weights[1] = weights[2] = weights[3] = 0.0;
    switch (region) {
        case EeInterior:
            weights[0] = 1.0 - dr.s;
            weights[1] = dr.s;
            weights[2] = -(1.0 - dr.t);
            weights[3] = -dr.t;
            break;
        case EeEdgeS0:
            weights[0] = 1.0;
            weights[2] = -(1.0 - dr.t);
            weights[3] = -dr.t;
            break;
        case EeEdgeS1:
            weights[1] = 1.0;
            weights[2] = -(1.0 - dr.t);
            weights[3] = -dr.t;
            break;
        case EeEdgeT0:
            weights[0] = 1.0 - dr.s;
            weights[1] = dr.s;
            weights[2] = -1.0;
            break;
        case EeEdgeT1:
            weights[0] = 1.0 - dr.s;
            weights[1] = dr.s;
            weights[3] = -1.0;
            break;
        case EeCornerS0T0:
            weights[0] = 1.0;
            weights[2] = -1.0;
            break;
        case EeCornerS0T1:
            weights[0] = 1.0;
            weights[3] = -1.0;
            break;
        case EeCornerS1T0:
            weights[1] = 1.0;
            weights[2] = -1.0;
            break;
        case EeCornerS1T1:
            weights[1] = 1.0;
            weights[3] = -1.0;
            break;
        default:
            // Defensive CPU fallback for an unresolved (s,t).
            weights[0] = 1.0 - dr.s;
            weights[1] = dr.s;
            weights[2] = -(1.0 - dr.t);
            weights[3] = -dr.t;
            break;
    }
}

// ---------------------------------------------------------------------------
// Barrier scalar derivatives
// ---------------------------------------------------------------------------

IPC_ORIGINAL_CONTACT_HD inline bool scalar_barrier_derivatives(
        double delta, double d_hat, double& b_prime, double& b_double_prime) {
    if (delta >= d_hat) {
        b_prime = 0.0;
        b_double_prime = 0.0;
        return true;
    }
    if (delta == 0.0) return false;
    const double log_ratio = log(delta / d_hat);
    const double s = delta - d_hat;
    b_prime = -2.0 * s * log_ratio - (s * s) / delta;
    const double ratio = d_hat / delta;
    b_double_prime = ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * log_ratio;
    return true;
}

// ---------------------------------------------------------------------------
// Point--triangle barrier derivative branches
// ---------------------------------------------------------------------------

IPC_ORIGINAL_CONTACT_HD inline void point_triangle_gradient(
        const double x[4][3], const PtGeometry& dr, int region,
        double delta, double bp, int dof, double g[3]) {
    g[0] = g[1] = g[2] = 0.0;
    if (bp == 0.0 || delta == 0.0) return;

    double u[3];
    for (int k = 0; k < 3; ++k) {
        u[k] = (x[0][k] - dr.closest[k]) / delta;
    }

    double coeff[4] = {0.0, 0.0, 0.0, 0.0};
    double face_n[3] = {0.0, 0.0, 0.0};
    bool use_normal = false;
    switch (region) {
        case PtFaceInterior: {
            const double sign = sign_of(dr.phi);
            coeff[0] = bp * sign;
            coeff[1] = -bp * sign * dr.bary[0];
            coeff[2] = -bp * sign * dr.bary[1];
            coeff[3] = -bp * sign * dr.bary[2];
            face_n[0] = dr.normal[0];
            face_n[1] = dr.normal[1];
            face_n[2] = dr.normal[2];
            use_normal = true;
            break;
        }
        case PtEdge12: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[1], x[2]);
            coeff[0] = bp;
            coeff[1] = -bp * (1.0 - t);
            coeff[2] = -bp * t;
            break;
        }
        case PtEdge23: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[2], x[3]);
            coeff[0] = bp;
            coeff[2] = -bp * (1.0 - t);
            coeff[3] = -bp * t;
            break;
        }
        case PtEdge31: {
            const double t = segment_parameter_from_closest_point(
                    dr.closest, x[3], x[1]);
            coeff[0] = bp;
            coeff[1] = -bp * t;
            coeff[3] = -bp * (1.0 - t);
            break;
        }
        case PtVertex1:
            coeff[0] = bp;
            coeff[1] = -bp;
            break;
        case PtVertex2:
            coeff[0] = bp;
            coeff[2] = -bp;
            break;
        case PtVertex3:
            coeff[0] = bp;
            coeff[3] = -bp;
            break;
        default:
            return;
    }

    if (use_normal) {
        for (int k = 0; k < 3; ++k) g[k] = coeff[dof] * face_n[k];
    } else {
        for (int k = 0; k < 3; ++k) g[k] = coeff[dof] * u[k];
    }
}

IPC_ORIGINAL_CONTACT_HD inline void point_triangle_hessian(
        const double x[4][3], const PtGeometry& dr, int region,
        double delta, double bp, double bpp, int p, int q, double H[3][3]) {
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l) H[k][l] = 0.0;
    if (bp == 0.0 && bpp == 0.0) return;
    if (delta == 0.0) return;

    const double* Y[4] = {x[0], x[1], x[2], x[3]};
    const int requested_dofs[2] = {p, q};
    const int requested_dof_count = 1 + (p != q);

    switch (region) {
        case PtVertex1:
        case PtVertex2:
        case PtVertex3: {
            const int a_idx = region == PtVertex1 ? 1
                    : region == PtVertex2 ? 2 : 3;
            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0;
            sp[a_idx] = -1.0;
            if (sp[p] == 0.0 || sp[q] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k)
                u[k] = (x[0][k] - Y[a_idx][k]) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            const double spq = sp[p] * sp[q];
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    H[k][l] = spq * (c1 * u[k] * u[l]
                                     + c2 * (dkl - u[k] * u[l]));
                }
            }
            break;
        }

        case PtEdge12:
        case PtEdge23:
        case PtEdge31: {
            int a_idx, b_idx;
            if (region == PtEdge12) { a_idx = 1; b_idx = 2; }
            else if (region == PtEdge23) { a_idx = 2; b_idx = 3; }
            else { a_idx = 3; b_idx = 1; }

            double omega[4] = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;
            omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;
            epsilon[b_idx] = 1.0;

            const double* xa = Y[a_idx];
            const double* xb = Y[b_idx];
            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xb[i] - xa[i];
                w[i] = x[0][i] - xa[i];
            }
            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta += e[i] * e[i];
            }
            const double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x[0][i] - (xa[i] + t * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3] = {};
            double r_d[4][3][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k]
                            + epsilon[pp] * w[k];
                    const double beta_pk = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta
                            - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dpa = (pp == a_idx) ? 1.0 : 0.0;
                        const double dpx = (pp == 0) ? 1.0 : 0.0;
                        const double q_d = dpa * dik + t_d[pp][k] * e[i]
                                + t * epsilon[pp] * dik;
                        r_d[pp][k][i] = dpx * dik - q_d;
                    }
                }
            }

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    const double alpha_pk = omega[p] * e[k]
                            + epsilon[p] * w[k];
                    const double alpha_ql = omega[q] * e[l]
                            + epsilon[q] * w[l];
                    const double alpha_pkql =
                            (omega[p] * epsilon[q] + epsilon[p] * omega[q])
                            * dkl;
                    const double beta_pk = 2.0 * epsilon[p] * e[k];
                    const double beta_ql = 2.0 * epsilon[q] * e[l];
                    const double beta_pkql =
                            2.0 * epsilon[p] * epsilon[q] * dkl;
                    const double t_pkql = alpha_pkql / beta
                            - (alpha_pk * beta_ql + alpha_ql * beta_pk
                               + alpha * beta_pkql) / (beta * beta)
                            + 2.0 * alpha * beta_pk * beta_ql
                                    / (beta * beta * beta);

                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[q][l][i];
                    }
                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j])
                                    * r_d[p][k][i] * r_d[q][l][j];
                        }
                    }
                    proj_term /= delta;
                    double uq_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double q_ipkql = t_pkql * e[i]
                                + t_d[p][k] * epsilon[q] * dil
                                + t_d[q][l] * epsilon[p] * dik;
                        uq_term += u[i] * q_ipkql;
                    }
                    const double d2delta = proj_term - uq_term;
                    H[k][l] = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                }
            }
            break;
        }

        case PtFaceInterior: {
            const double sig_a[4] = {0.0, -1.0, 1.0, 0.0};
            const double sig_b[4] = {0.0, -1.0, 0.0, 1.0};
            const double sig_w[4] = {1.0, -1.0, 0.0, 0.0};

            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x[2][i] - x[1][i];
                b[i] = x[3][i] - x[1][i];
                w[i] = x[0][i] - x[1][i];
            }

            double N[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i) {
                for (int m = 0; m < 3; ++m) {
                    for (int nn = 0; nn < 3; ++nn) {
                        N[i] += levi_civita(i, m, nn) * a[m] * b[nn];
                    }
                }
            }
            double eta = 0.0;
            for (int i = 0; i < 3; ++i) eta += N[i] * N[i];
            eta = sqrt(eta);
            double n[3];
            for (int i = 0; i < 3; ++i) n[i] = N[i] / eta;
            double psi = 0.0;
            for (int i = 0; i < 3; ++i) psi += N[i] * w[i];
            const double phi = psi / eta;
            const double s_sign = sign_of(phi);

            double Nd[4][3][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        double val = 0.0;
                        for (int nn = 0; nn < 3; ++nn)
                            val += sig_a[pp] * levi_civita(i, k, nn) * b[nn];
                        for (int m = 0; m < 3; ++m)
                            val += sig_b[pp] * levi_civita(i, m, k) * a[m];
                        Nd[pp][k][i] = val;
                    }
                }
            }

            double eta_d[4][3] = {};
            double psi_d[4][3] = {};
            double phi_d[4][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    double eta_pk = 0.0;
                    for (int i = 0; i < 3; ++i)
                        eta_pk += n[i] * Nd[pp][k][i];
                    eta_d[pp][k] = eta_pk;
                    double psi_pk = 0.0;
                    for (int i = 0; i < 3; ++i)
                        psi_pk += Nd[pp][k][i] * w[i];
                    psi_pk += sig_w[pp] * N[k];
                    psi_d[pp][k] = psi_pk;
                    phi_d[pp][k] = psi_pk / eta
                            - psi * eta_pk / (eta * eta);
                }
            }

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double coeff_N2 =
                            sig_a[p] * sig_b[q] - sig_b[p] * sig_a[q];
                    double nN2 = 0.0;
                    for (int i = 0; i < 3; ++i)
                        nN2 += n[i] * coeff_N2 * levi_civita(i, k, l);
                    double proj_NN = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_NN += (dij - n[i] * n[j])
                                    * Nd[p][k][i] * Nd[q][l][j];
                        }
                    }
                    const double eta_pkql = nN2 + proj_NN / eta;
                    double psi_pkql = 0.0;
                    for (int i = 0; i < 3; ++i)
                        psi_pkql += coeff_N2 * levi_civita(i, k, l) * w[i];
                    psi_pkql += sig_w[q] * Nd[p][k][l];
                    psi_pkql += sig_w[p] * Nd[q][l][k];
                    const double phi_pkql = psi_pkql / eta
                            - (psi_d[p][k] * eta_d[q][l]
                               + psi_d[q][l] * eta_d[p][k]
                               + psi * eta_pkql) / (eta * eta)
                            + 2.0 * psi * eta_d[p][k] * eta_d[q][l]
                                    / (eta * eta * eta);
                    H[k][l] = bpp * phi_d[p][k] * phi_d[q][l]
                            + s_sign * bp * phi_pkql;
                }
            }
            break;
        }

        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Segment--segment barrier derivative branches (non-robust CPU path)
// ---------------------------------------------------------------------------

IPC_ORIGINAL_CONTACT_HD inline void segment_segment_gradient(
        const double x[4][3], const EeGeometry& dr, int region,
        double delta, double bp, int dof, double g[3]) {
    g[0] = g[1] = g[2] = 0.0;
    if (bp == 0.0 || delta == 0.0) return;

    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = dr.separation[k] / delta;
    const double s = dr.s;
    const double t = dr.t;

    double mu[4] = {0.0, 0.0, 0.0, 0.0};
    switch (region) {
        case EeInterior:
            mu[0] = bp * (1.0 - s);
            mu[1] = bp * s;
            mu[2] = -bp * (1.0 - t);
            mu[3] = -bp * t;
            break;
        case EeEdgeS0:
            mu[0] = bp;
            mu[2] = -bp * (1.0 - t);
            mu[3] = -bp * t;
            break;
        case EeEdgeS1:
            mu[1] = bp;
            mu[2] = -bp * (1.0 - t);
            mu[3] = -bp * t;
            break;
        case EeEdgeT0:
            mu[0] = bp * (1.0 - s);
            mu[1] = bp * s;
            mu[2] = -bp;
            break;
        case EeEdgeT1:
            mu[0] = bp * (1.0 - s);
            mu[1] = bp * s;
            mu[3] = -bp;
            break;
        case EeCornerS0T0:
            mu[0] = bp;
            mu[2] = -bp;
            break;
        case EeCornerS0T1:
            mu[0] = bp;
            mu[3] = -bp;
            break;
        case EeCornerS1T0:
            mu[1] = bp;
            mu[2] = -bp;
            break;
        case EeCornerS1T1:
            mu[1] = bp;
            mu[3] = -bp;
            break;
        default: {
            // Defensive fallback for an invalid resolved (s,t).
            const double muf[4] = {1.0 - s, s, -(1.0 - t), -t};
            for (int k = 0; k < 3; ++k) g[k] = bp * muf[dof] * u[k];
            return;
        }
    }
    for (int k = 0; k < 3; ++k) g[k] = mu[dof] * u[k];
}

IPC_ORIGINAL_CONTACT_HD inline void segment_segment_hessian(
        const double x[4][3], int region, double delta,
        double bp, double bpp, int p, int q, double H[3][3]) {
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l) H[k][l] = 0.0;
    if (bp == 0.0 && bpp == 0.0) return;
    if (delta == 0.0) return;

    const double* Y[4] = {x[0], x[1], x[2], x[3]};
    const int requested_dofs[2] = {p, q};
    const int requested_dof_count = 1 + (p != q);

    switch (region) {
        case EeCornerS0T0:
        case EeCornerS0T1:
        case EeCornerS1T0:
        case EeCornerS1T1: {
            int a_idx, b_idx;
            if (region == EeCornerS0T0) { a_idx = 0; b_idx = 2; }
            else if (region == EeCornerS0T1) { a_idx = 0; b_idx = 3; }
            else if (region == EeCornerS1T0) { a_idx = 1; b_idx = 2; }
            else { a_idx = 1; b_idx = 3; }

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[a_idx] = 1.0;
            sp[b_idx] = -1.0;
            if (sp[p] == 0.0 || sp[q] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k)
                u[k] = (Y[a_idx][k] - Y[b_idx][k]) / delta;
            const double c1 = bpp;
            const double c2 = bp / delta;
            const double spq = sp[p] * sp[q];
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    H[k][l] = spq * (c1 * u[k] * u[l]
                                     + c2 * (dkl - u[k] * u[l]));
                }
            }
            break;
        }

        case EeEdgeS0:
        case EeEdgeS1:
        case EeEdgeT0:
        case EeEdgeT1: {
            int query_idx, ea_idx, eb_idx;
            if (region == EeEdgeS0) { query_idx = 0; ea_idx = 2; eb_idx = 3; }
            else if (region == EeEdgeS1) { query_idx = 1; ea_idx = 2; eb_idx = 3; }
            else if (region == EeEdgeT0) { query_idx = 2; ea_idx = 0; eb_idx = 1; }
            else { query_idx = 3; ea_idx = 0; eb_idx = 1; }

            const double* xq = Y[query_idx];
            const double* xea = Y[ea_idx];
            const double* xeb = Y[eb_idx];

            double omega[4] = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0;
            omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0;
            epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xeb[i] - xea[i];
                w[i] = xq[i] - xea[i];
            }
            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta += e[i] * e[i];
            }
            const double t_param = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq[i] - (xea[i] + t_param * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3] = {};
            double r_d[4][3][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k]
                            + epsilon[pp] * w[k];
                    const double beta_pk = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta
                            - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dp_ea = (pp == ea_idx) ? 1.0 : 0.0;
                        const double dp_q = (pp == query_idx) ? 1.0 : 0.0;
                        const double q_d = dp_ea * dik + t_d[pp][k] * e[i]
                                + t_param * epsilon[pp] * dik;
                        r_d[pp][k][i] = dp_q * dik - q_d;
                    }
                }
            }

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    const double alpha_pk = omega[p] * e[k]
                            + epsilon[p] * w[k];
                    const double alpha_ql = omega[q] * e[l]
                            + epsilon[q] * w[l];
                    const double alpha_pkql =
                            (omega[p] * epsilon[q] + epsilon[p] * omega[q])
                            * dkl;
                    const double beta_pk = 2.0 * epsilon[p] * e[k];
                    const double beta_ql = 2.0 * epsilon[q] * e[l];
                    const double beta_pkql =
                            2.0 * epsilon[p] * epsilon[q] * dkl;
                    const double t_pkql = alpha_pkql / beta
                            - (alpha_pk * beta_ql + alpha_ql * beta_pk
                               + alpha * beta_pkql) / (beta * beta)
                            + 2.0 * alpha * beta_pk * beta_ql
                                    / (beta * beta * beta);
                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[q][l][i];
                    }
                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j])
                                    * r_d[p][k][i] * r_d[q][l][j];
                        }
                    }
                    proj_term /= delta;
                    double uq_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double q_ipkql = t_pkql * e[i]
                                + t_d[p][k] * epsilon[q] * dil
                                + t_d[q][l] * epsilon[p] * dik;
                        uq_term += u[i] * q_ipkql;
                    }
                    H[k][l] = bpp * ddelta_pk * ddelta_ql
                            + bp * (proj_term - uq_term);
                }
            }
            break;
        }

        case EeInterior: {
            const double sig_a[4] = {-1.0, 1.0, 0.0, 0.0};
            const double sig_b[4] = {0.0, 0.0, -1.0, 1.0};
            const double sig_c[4] = {1.0, 0.0, -1.0, 0.0};

            double a[3], b[3], c[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x[1][i] - x[0][i];
                b[i] = x[3][i] - x[2][i];
                c[i] = x[0][i] - x[2][i];
            }
            double A = 0.0, B = 0.0, C = 0.0, D = 0.0, E = 0.0;
            for (int i = 0; i < 3; ++i) {
                A += a[i] * a[i];
                B += a[i] * b[i];
                C += b[i] * b[i];
                D += a[i] * c[i];
                E += b[i] * c[i];
            }
            const double Delta = A * C - B * B;
            const double nu = B * E - C * D;
            // The production CPU Hessian oracle requires this contracted
            // internal zeta. Preserve the independently computed public
            // distance/feature and its legacy delta without substitution.
            const double zeta = hessian_product_difference(A, E, B, D);
            const double s_val = nu / Delta;
            const double t_val = zeta / Delta;

            double Ad[4][3] = {}, Bd[4][3] = {}, Cd[4][3] = {};
            double Dd[4][3] = {}, Ed[4][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    Ad[pp][k] = 2.0 * sig_a[pp] * a[k];
                    Bd[pp][k] = sig_a[pp] * b[k] + sig_b[pp] * a[k];
                    Cd[pp][k] = 2.0 * sig_b[pp] * b[k];
                    Dd[pp][k] = sig_a[pp] * c[k] + sig_c[pp] * a[k];
                    Ed[pp][k] = sig_b[pp] * c[k] + sig_c[pp] * b[k];
                }
            }
            double nu_d[4][3] = {}, zeta_d[4][3] = {}, Delta_d[4][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    // The CPU source writes each of these as a sum followed
                    // by multiply-subtracts, e.g.
                    //   nu_d = Bd*E + B*Ed - Cd*D - C*Dd;
                    // With contraction those trailing products are fused
                    // into the running sum.  Only the fused form is a
                    // faithful transcription for near-parallel Delta: the
                    // separately rounded product destroys the cancellation
                    // (see fused_subtract_product).  The quotient-rule
                    // combination of nu_d/zeta_d/Delta_d is left in the
                    // original association, and the second-derivative
                    // numerators (nu_pkql/zeta_pkql/Delta_pkql) are left
                    // untouched: in this family their fused and unfused
                    // forms are bit-identical over all 16 Cartesian blocks,
                    // so no archive comparison justifies perturbing them.
                    nu_d[pp][k] = fused_subtract_product(C, Dd[pp][k],
                            fused_subtract_product(Cd[pp][k], D,
                                    Bd[pp][k] * E + B * Ed[pp][k]));
                    zeta_d[pp][k] = fused_subtract_product(B, Dd[pp][k],
                            fused_subtract_product(Bd[pp][k], D,
                                    Ad[pp][k] * E + A * Ed[pp][k]));
                    Delta_d[pp][k] = fused_subtract_product(2.0 * B,
                            Bd[pp][k],
                            Ad[pp][k] * C + A * Cd[pp][k]);
                }
            }
            double s_d[4][3] = {}, t_d[4][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    s_d[pp][k] = nu_d[pp][k] / Delta
                            - nu * Delta_d[pp][k] / (Delta * Delta);
                    t_d[pp][k] = zeta_d[pp][k] / Delta
                            - zeta * Delta_d[pp][k] / (Delta * Delta);
                }
            }
            double r_vec[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r_vec[i] = (x[0][i] + s_val * a[i])
                        - (x[2][i] + t_val * b[i]);
                u[i] = r_vec[i] / delta;
            }
            double p_d[4][3][3] = {}, q_d_arr[4][3][3] = {};
            double r_d[4][3][3] = {};
            for (int di = 0; di < requested_dof_count; ++di) {
                const int pp = requested_dofs[di];
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dp0 = (pp == 0) ? 1.0 : 0.0;
                        const double dp2 = (pp == 2) ? 1.0 : 0.0;
                        p_d[pp][k][i] = dp0 * dik + s_d[pp][k] * a[i]
                                + s_val * sig_a[pp] * dik;
                        q_d_arr[pp][k][i] = dp2 * dik + t_d[pp][k] * b[i]
                                + t_val * sig_b[pp] * dik;
                        r_d[pp][k][i] = p_d[pp][k][i] - q_d_arr[pp][k][i];
                    }
                }
            }
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    const double A_pkql = 2.0 * sig_a[p] * sig_a[q] * dkl;
                    const double B_pkql =
                            (sig_a[p] * sig_b[q] + sig_b[p] * sig_a[q]) * dkl;
                    const double C_pkql = 2.0 * sig_b[p] * sig_b[q] * dkl;
                    const double D_pkql =
                            (sig_a[p] * sig_c[q] + sig_c[p] * sig_a[q]) * dkl;
                    const double E_pkql =
                            (sig_b[p] * sig_c[q] + sig_c[p] * sig_b[q]) * dkl;
                    const double nu_pkql = B_pkql * E
                            + Bd[p][k] * Ed[q][l] + Bd[q][l] * Ed[p][k]
                            + B * E_pkql - C_pkql * D
                            - Cd[p][k] * Dd[q][l] - Cd[q][l] * Dd[p][k]
                            - C * D_pkql;
                    const double Delta_pkql = A_pkql * C
                            + Ad[p][k] * Cd[q][l] + Ad[q][l] * Cd[p][k]
                            + A * C_pkql
                            - 2.0 * (Bd[p][k] * Bd[q][l] + B * B_pkql);
                    const double zeta_pkql = A_pkql * E
                            + Ad[p][k] * Ed[q][l] + Ad[q][l] * Ed[p][k]
                            + A * E_pkql - B_pkql * D
                            - Bd[p][k] * Dd[q][l] - Bd[q][l] * Dd[p][k]
                            - B * D_pkql;
                    const double s_pkql = nu_pkql / Delta
                            - (nu_d[p][k] * Delta_d[q][l]
                               + nu_d[q][l] * Delta_d[p][k]
                               + nu * Delta_pkql) / (Delta * Delta)
                            + 2.0 * nu * Delta_d[p][k] * Delta_d[q][l]
                                    / (Delta * Delta * Delta);
                    const double t_pkql = zeta_pkql / Delta
                            - (zeta_d[p][k] * Delta_d[q][l]
                               + zeta_d[q][l] * Delta_d[p][k]
                               + zeta * Delta_pkql) / (Delta * Delta)
                            + 2.0 * zeta * Delta_d[p][k] * Delta_d[q][l]
                                    / (Delta * Delta * Delta);
                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[q][l][i];
                    }
                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j])
                                    * r_d[p][k][i] * r_d[q][l][j];
                        }
                    }
                    proj_term /= delta;
                    double ur_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double p_ipkql = s_pkql * a[i]
                                + s_d[p][k] * sig_a[q] * dil
                                + s_d[q][l] * sig_a[p] * dik;
                        const double q_ipkql = t_pkql * b[i]
                                + t_d[p][k] * sig_b[q] * dil
                                + t_d[q][l] * sig_b[p] * dik;
                        ur_term += u[i] * (p_ipkql - q_ipkql);
                    }
                    H[k][l] = bpp * ddelta_pk * ddelta_ql
                            + bp * (proj_term + ur_term);
                }
            }
            break;
        }

        default:
            break;
    }
}

}  // namespace original_contact_detail

// Original CPU (exact_computation_fallback == false) contact evaluation.
//
// Publication contract matches ContactOutput:
//   * distance/weights/normal come from the CPU legacy geometry;
//   * barrier derivatives are unscaled (caller applies dt^2*k_barrier);
//   * friction derivatives are relative and already include dt^2*friction;
//   * requested Cartesian Hessian blocks are filled, omitted blocks are zero.
// A non-Ok status must abort/roll back the update. In particular a zero gap
// with positive stiffness, a non-finite legacy result, or an invalid material
// input is never converted into an inactive contact. A zero-stiffness packet
// (k_barrier == 0) is the CPU legacy no-force case and is inactive even when
// the gap is exactly zero, matching initialize_contact_barrier_data.
IPC_ORIGINAL_CONTACT_HD inline ContactOutput evaluate_contact_original(
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

}  // namespace solver
}  // namespace ipc_gpu

#undef IPC_ORIGINAL_CONTACT_HD

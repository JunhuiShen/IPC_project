#include "original_contact.cuh"
#include "barrier_energy.h"
#include "friction_energy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(IPC_GPU_ORIGINAL_CUDA_TEST) || defined(IPC_GPU_CONTACT_CUDA_TEST)
#define IPC_ORIGINAL_TEST_CUDA 1
#endif

#ifdef IPC_ORIGINAL_TEST_CUDA
void gpu_original_contact_compute_gpu(
        const ipc_gpu::solver::ContactInput*,
        ipc_gpu::solver::ContactOutput*, std::size_t);
#endif

using ipc_gpu::solver::ContactInput;
using ipc_gpu::solver::ContactOutput;
using ipc_gpu::solver::ContactKind;
using ipc_gpu::solver::ContactStatus;

namespace {

std::size_t components = 0;
double worst_scaled_error = 0.0;
std::string worst_label;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void compare(double actual, double expected, const std::string& label) {
    require(std::isfinite(actual) && std::isfinite(expected),
            label + ": nonfinite");
    const double error = std::abs(actual - expected) / (1.0 + std::abs(expected));
    if (error > worst_scaled_error) {
        worst_scaled_error = error;
        worst_label = label;
    }
    require(error <= 2.0e-9,
            label + ": mismatch " + std::to_string(actual)
            + " vs " + std::to_string(expected)
            + " scaled error " + std::to_string(error));
    ++components;
}

double sign_of(double value) {
    return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

struct Reference {
    ContactKind kind = ContactKind::PointTriangle;
    double distance = 0.0;
    std::array<double, 4> weights{};
    Vec3 normal = Vec3::Zero();
    std::array<Vec3, 4> gradient{};
    std::array<std::array<Mat33, 4>, 4> hessian{};
    FrozenFrictionContact frozen;
    Vec3 friction_gradient = Vec3::Zero();
    Mat33 friction_hessian = Mat33::Zero();
    bool active = false;
};

// Every path below explicitly uses the CPU original solve
// (exact_computation_fallback = false). This is the oracle, never a mirror of
// the device implementation.
Reference make_reference(const ContactInput& in) {
    Reference ref;
    ref.kind = in.kind;
    std::array<Vec3, 4> x;
    std::array<Vec3, 4> previous;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            x[i][k] = in.positions[i][k];
            previous[i][k] = in.previous[i][k];
        }
    }

    if (in.kind == ContactKind::PointTriangle) {
        const auto evaluation = make_node_triangle_contact_evaluation(
                x, in.d_hat, in.k_barrier, in.eps, nullptr, false);
        ref.distance = evaluation.dr.distance;
        ref.active = evaluation.active;
        ref.weights = node_triangle_contact_weights(
                x[0], x[1], x[2], x[3], in.eps, &evaluation.dr);
        if (evaluation.dr.region == NodeTriangleRegion::FaceInterior) {
            const double sign = sign_of(evaluation.dr.phi);
            for (int k = 0; k < 3; ++k)
                ref.normal[k] = sign * evaluation.dr.normal[k];
        } else if (ref.distance > 0.0) {
            ref.normal = (x[0] - evaluation.dr.closest_point) / ref.distance;
        }
        for (int i = 0; i < 4; ++i) {
            ref.gradient[i] = node_triangle_barrier_gradient(
                    x[0], x[1], x[2], x[3], i, evaluation, in.eps);
            for (int j = 0; j < 4; ++j) {
                ref.hessian[i][j] = node_triangle_barrier_cross_hessian(
                        x[0], x[1], x[2], x[3], i, j, evaluation, in.eps);
            }
        }
        ref.frozen = make_node_triangle_frozen_friction_contact(
                x, previous, in.d_hat, in.k_barrier, in.dt, in.eps_v,
                in.eps, nullptr, false);
    } else {
        const auto evaluation = make_segment_segment_contact_evaluation(
                x, in.d_hat, in.k_barrier, in.eps, nullptr, false);
        ref.distance = evaluation.dr.distance;
        ref.active = evaluation.active;
        ref.weights = segment_segment_contact_weights(
                x[0], x[1], x[2], x[3], in.eps, &evaluation.dr, false);
        if (ref.distance > 0.0)
            ref.normal = evaluation.dr.separation / ref.distance;
        for (int i = 0; i < 4; ++i) {
            ref.gradient[i] = segment_segment_barrier_gradient(
                    x[0], x[1], x[2], x[3], i, evaluation, in.eps);
            for (int j = 0; j < 4; ++j) {
                ref.hessian[i][j] = segment_segment_barrier_cross_hessian(
                        x[0], x[1], x[2], x[3], i, j, evaluation, in.eps);
            }
        }
        ref.frozen = make_segment_segment_frozen_friction_contact(
                x, previous, in.d_hat, in.k_barrier, in.dt, in.eps_v,
                in.eps, nullptr, false);
    }
    const auto friction = frozen_friction_relative_gradient_and_hessian(
            ref.frozen, in.friction, in.dt * in.dt);
    ref.friction_gradient = friction.first;
    ref.friction_hessian = friction.second;
    return ref;
}

void reference_check(const ContactInput& in, const ContactOutput& out,
                     std::size_t index) {
    const std::string tag = "record " + std::to_string(index)
            + (in.kind == ContactKind::PointTriangle ? " PT" : " EE");
    const Reference ref = make_reference(in);
    require(out.status == ContactStatus::Ok,
            tag + ": unexpected status "
            + std::to_string(unsigned(out.status)));
    require(bool(out.active) == ref.active, tag + ": activity differs");
    compare(out.distance, ref.distance, tag + " distance");
    for (int k = 0; k < 3; ++k) compare(out.normal[k], ref.normal[k],
                                        tag + " normal");
    for (int i = 0; i < 4; ++i) {
        compare(out.weights[i], ref.weights[i], tag + " weight");
        for (int k = 0; k < 3; ++k) {
            compare(out.barrier_gradient[i][k], ref.gradient[i][k],
                    tag + " gradient " + std::to_string(i));
            for (int j = 0; j < 4; ++j) {
                const bool requested =
                        (in.hessian_blocks & (1u << (4 * i + j))) != 0;
                for (int l = 0; l < 3; ++l) {
                    const double value =
                            out.barrier_hessian[i][j][3 * k + l];
                    if (requested) {
                        compare(value, ref.hessian[i][j](k, l),
                                tag + " Hessian "
                                + std::to_string(i) + std::to_string(j)
                                + std::to_string(k) + std::to_string(l));
                    } else {
                        require(value == 0.0,
                                tag + ": omitted Hessian block is nonzero");
                    }
                }
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        compare(out.friction_relative_gradient[k], ref.friction_gradient[k],
                tag + " friction gradient");
        for (int l = 0; l < 3; ++l) {
            compare(out.friction_relative_hessian[3 * k + l],
                    ref.friction_hessian(k, l), tag + " friction Hessian");
        }
    }
    if (ref.frozen.active) {
        for (int role = 0; role < 4; ++role) {
            compare(out.weights[role], ref.frozen.weights[role],
                    tag + " active friction weight");
            const Vec3 role_gradient = frozen_friction_role_gradient(
                    ref.frozen, role, in.friction, in.dt * in.dt);
            const Mat33 role_hessian = frozen_friction_role_hessian(
                    ref.frozen, role, in.friction, in.dt * in.dt);
            for (int k = 0; k < 3; ++k) {
                compare(role_gradient[k],
                        out.weights[role] * out.friction_relative_gradient[k],
                        tag + " friction role gradient");
                for (int l = 0; l < 3; ++l) {
                    compare(role_hessian(k, l),
                            out.weights[role] * out.weights[role]
                            * out.friction_relative_hessian[3 * k + l],
                            tag + " friction role Hessian");
                }
            }
        }
    }
}

ContactInput make_input(ContactKind kind,
                        const std::array<Vec3, 4>& positions,
                        int slip_mode, double friction, unsigned mask =
                                0xffffu) {
    ContactInput in{};
    in.kind = kind;
    in.d_hat = 0.35;
    in.k_barrier = 257.0;
    in.dt = 0.017;
    in.friction = friction;
    in.eps_v = 0.003;
    in.eps = 1.0e-12;
    in.hessian_blocks = mask;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) in.positions[i][k] = positions[i][k];
    }
    double slip = 0.0;
    if (slip_mode == 1) slip = 1.0e-8;
    if (slip_mode == 2) slip = 0.5 * in.dt * in.eps_v;
    if (slip_mode == 3) slip = 2.0 * in.dt * in.eps_v;
    if (slip_mode == 4) slip = 1.0e-3;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            in.previous[i][k] = in.positions[i][k]
                    - slip * double(i + 1) * 0.07 * double(k + 1);
        }
    }
    return in;
}

std::vector<ContactInput> fixtures() {
    std::vector<ContactInput> records;
    const ContactKind pt = ContactKind::PointTriangle;
    const ContactKind ee = ContactKind::EdgeEdge;

    const std::array<Vec3, 3> tri{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    // Point--triangle: all ordinary features, including interior boundaries
    // (barycentric zero) and exterior edges/vertices.
    const std::vector<Vec3> pt_points{
            Vec3(0.25, 0.25, 0.10),   // face interior
            Vec3(0.50, 0.00, 0.12),   // face/edge boundary, inclusive
            Vec3(0.00, 0.00, 0.12),   // face/vertex boundary, inclusive
            Vec3(0.50, -0.20, 0.12),  // edge 12
            Vec3(1.40, -0.20, 0.12),  // nearest vertex 2 (not edge)
            Vec3(0.60, 0.60, 0.10),   // edge 23 (hypotenuse)
            Vec3(-0.20, 0.50, 0.12),  // edge 31
            Vec3(-1.00, -1.00, 0.12), // vertex 1
            Vec3(2.00, -0.20, 0.12),  // vertex 2
            Vec3(-0.20, 2.00, 0.12),  // vertex 3
    };
    for (const auto& point : pt_points) {
        for (int slip = 0; slip < 5; ++slip) {
            const double friction = (slip & 1) ? 0.0 : 0.23;
            records.push_back(make_input(
                    pt, {point, tri[0], tri[1], tri[2]}, slip, friction));
        }
    }

    // Obtuse triangle. The nearest edge, not the "obvious" barycentric sign,
    // must be selected (the legacy exterior loop).
    const std::array<Vec3, 3> obtuse{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
            Vec3(-10.0, 0.1, 0.0)};
    for (const Vec3& point : {Vec3(0.5, -0.01, 0.1), Vec3(0.5, 0.05, 0.1),
                              Vec3(-5.0, -0.01, 0.1)}) {
        records.push_back(make_input(pt,
                {point, obtuse[0], obtuse[1], obtuse[2]}, 2, 0.23));
    }

    // Sliver and small triangles.
    const std::array<Vec3, 3> sliver{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
            Vec3(0.5, 1.0e-3, 0.0)};
    for (const Vec3& point : {Vec3(0.5, -0.01, 0.1), Vec3(0.5, 0.2, 0.1),
                              Vec3(0.5, 0.0005, 0.1)}) {
        records.push_back(make_input(pt,
                {point, sliver[0], sliver[1], sliver[2]}, 3, 0.23));
    }
    const double small = 1.0e-4;
    records.push_back(make_input(pt,
            {Vec3(0.25 * small, 0.25 * small, 0.1 * small),
             Vec3(0.0, 0.0, 0.0), Vec3(small, 0.0, 0.0),
             Vec3(0.0, small, 0.0)}, 1, 0.23));

    // Large translation: local geometry is unchanged, so the CPU false path
    // must still agree.
    const Vec3 shift(1000.0, -2000.0, 5000.0);
    records.push_back(make_input(pt,
            {Vec3(1000.25, -1999.75, 5000.1), shift,
             shift + Vec3(1.0, 0.0, 0.0), shift + Vec3(0.0, 1.0, 0.0)},
            2, 0.23));

    // The CPU point--triangle area test is n_norm <= eps, not area^2 <= eps.
    for (double height : {1.0e-13, 1.0e-12,
                          std::nextafter(1.0e-12,
                                         std::numeric_limits<double>::infinity())}) {
        records.push_back(make_input(pt,
                {Vec3(0.25, 0.2, 0.1), Vec3(0.0, 0.0, 0.0),
                 Vec3(1.0, 0.0, 0.0), Vec3(0.0, height, 0.0)}, 2, 0.23));
    }
    // Degenerate collinear triangle, including the 1e-14 endpoint classification.
    const std::array<Vec3, 3> collinear{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0)};
    for (const Vec3& point : {Vec3(0.2, 0.2, 0.1), Vec3(1.0e-15, 0.2, 0.1),
                              Vec3(1.0 - 1.0e-15, 0.2, 0.1),
                              Vec3(1.8, 0.2, 0.1)}) {
        records.push_back(make_input(pt,
                {point, collinear[0], collinear[1], collinear[2]}, 2, 0.23));
    }
    // Strict tie between the second and third candidate: the first winner
    // (E23 here) must win, not the third equal-distance candidate.
    records.push_back(make_input(pt,
            {Vec3(-2.0, -2.0, 1.0), Vec3(-3.0, -3.0, 0.0),
             Vec3(-3.0, -1.0, 0.0), Vec3(-1.0, -2.0, 0.0)}, 2, 0.23));

    // Segment--segment: strict interior, all four inclusive endpoint
    // interiors (both free parameters survive), and near-boundary exterior.
    const std::array<Vec3, 2> s1{Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)};
    const auto add_ee = [&](Vec3 a, Vec3 b, int slip, double friction) {
        records.push_back(make_input(ee, {s1[0], s1[1], a, b}, slip, friction));
    };
    add_ee(Vec3(0.2, -0.5, 0.15), Vec3(0.2, 0.5, 0.15), 0, 0.0);
    add_ee(Vec3(0.2, -0.5, 0.15), Vec3(0.2, 0.5, 0.15), 1, 0.23);
    add_ee(Vec3(0.2, -0.5, 0.15), Vec3(0.2, 0.5, 0.15), 4, 0.23);
    add_ee(Vec3(0.0, 0.0, 0.2), Vec3(0.0, 1.0, 0.2), 2, 0.23);   // s=0,t=0
    add_ee(Vec3(1.0, 0.0, 0.2), Vec3(1.0, 1.0, 0.2), 3, 0.23);   // s=1,t=0
    add_ee(Vec3(0.0, -1.0, 0.2), Vec3(0.0, 0.0, 0.2), 2, 0.23);   // s=0,t=1
    add_ee(Vec3(1.0, -1.0, 0.2), Vec3(1.0, 0.0, 0.2), 3, 0.23);   // s=1,t=1
    add_ee(Vec3(1.0e-15, -0.5, 0.2), Vec3(1.0e-15, 0.5, 0.2), 2, 0.23);
    // Outside the parameter square, but inside the 1e-14 endpoint classifier.
    add_ee(Vec3(-1.0, -1.0e-14, 0.2), Vec3(-1.0, 1.0 - 1.0e-14, 0.2),
           2, 0.23);                                             // C00
    add_ee(Vec3(-1.0, -2.0e-14, 0.2), Vec3(-1.0, 1.0 - 2.0e-14, 0.2),
           2, 0.23);                                             // S0
    add_ee(Vec3(-1.0, -1.0 + 1.0e-14, 0.2),
           Vec3(-1.0, 1.0e-14, 0.2), 2, 0.23);                   // C01
    // Exterior edge candidates, one per boundary edge.
    add_ee(Vec3(-0.5, -0.5, 0.2), Vec3(-0.5, 0.5, 0.2), 3, 0.23); // s=0
    add_ee(Vec3(1.5, -0.5, 0.2), Vec3(1.5, 0.5, 0.2), 2, 0.23);  // s=1
    add_ee(Vec3(0.5, 0.5, 0.2), Vec3(0.5, 1.5, 0.2), 3, 0.23);   // t=0
    add_ee(Vec3(0.5, -2.0, 0.2), Vec3(0.5, -1.0, 0.2), 2, 0.23); // t=1
    // The false path never replaces a small absolute Gram determinant by the
    // exact solve. With eps=1e-6 these straddle the absolute eps^2 gate.
    for (double tilt : {1.0e-7, 1.0e-5}) {
        ContactInput gate = make_input(ee,
                {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
                 Vec3(0.0, 0.0, 0.2), Vec3(1.0, tilt, 0.2)}, 2, 0.23);
        gate.eps = 1.0e-6;
        records.push_back(gate);
    }

    // Parallel segments resolve to edge/corner only from the boundary
    // candidate. Covers C00/C01/C10, S0, S1, T0, and a strict S1/T0 tie.
    add_ee(Vec3(-1.0e-14, 0.0, 0.5), Vec3(1.0 - 1.0e-14, 0.0, 0.5),
           2, 0.23);                                             // C00
    add_ee(Vec3(-1.5, 0.2, 0.5), Vec3(-0.5, 0.2, 0.5), 2, 0.23); // C01
    add_ee(Vec3(1.0, 0.2, 0.5), Vec3(2.0, 0.2, 0.5), 2, 0.23);   // C10
    add_ee(Vec3(0.2, -2.0, 0.5), Vec3(1.0, -2.0, 0.5), 2, 0.23); // C11
    add_ee(Vec3(-0.5, 0.2, 0.5), Vec3(0.5, 0.2, 0.5), 2, 0.23);  // S0
    add_ee(Vec3(0.2, 0.2, 0.5), Vec3(1.2, 0.2, 0.5), 2, 0.23);   // S1/T0 tie
    add_ee(Vec3(0.4, 0.2, 0.5), Vec3(0.6, 0.2, 0.5), 2, 0.23);   // T0
    // Degenerate segment: the CPU edge fallback still chooses s=0.
    add_ee(Vec3(0.2, 0.2, 0.5), Vec3(0.2, 0.2, 0.5), 2, 0.23);
    // Both segments degenerate.
    records.push_back(make_input(ee,
            {Vec3(0.1, 0.2, 0.3), Vec3(0.1, 0.2, 0.3),
             Vec3(0.4, -0.1, 0.5), Vec3(0.4, -0.1, 0.5)}, 2, 0.23));
    // Sliver/translated segment geometry.
    add_ee(Vec3(0.25, -0.4, 0.2), Vec3(0.75, 0.4, 0.2), 2, 0.23);

    // Every feature fixture above must be barrier-active so its branch-specific
    // derivative formulas, not just its geometry, are exercised.
    for (auto& input : records) input.d_hat = 5.0;

    // Strict activation: distance == d_hat is inactive; the next representable
    // larger d_hat activates. Both keep the same positive gap.
    ContactInput equal_gap = make_input(pt,
            {Vec3(0.25, 0.25, 0.1), tri[0], tri[1], tri[2]}, 2, 0.23);
    equal_gap.d_hat = 0.1;
    records.push_back(equal_gap);
    ContactInput just_above = equal_gap;
    just_above.d_hat = std::nextafter(
            0.1, std::numeric_limits<double>::infinity());
    records.push_back(just_above);

    // Keep explicit inactive records too: the CPU evaluation returns zero
    // derivatives while preserving the positive distance and weights.
    ContactInput inactive_pt = records.front();
    inactive_pt.d_hat = 0.01;
    records.push_back(inactive_pt);
    ContactInput inactive_ee;
    bool have_inactive_ee = false;
    for (const auto& input : records) {
        if (input.kind == ContactKind::EdgeEdge) {
            inactive_ee = input;
            have_inactive_ee = true;
            break;
        }
    }
    if (have_inactive_ee) {
        inactive_ee.d_hat = 0.01;
        inactive_ee.friction = 0.23;
        records.push_back(inactive_ee);
    }

    return records;
}

std::vector<ContactInput> masked_variants(const ContactInput& in) {
    const unsigned masks[] = {
            0u, 1u, 1u << 5, 1u << 10, 1u << 15,
            (1u << 0) | (1u << 1) | (1u << 5),
            (1u << 5) | (1u << 6) | (1u << 7) | (1u << 10)
                    | (1u << 11) | (1u << 15),
            0xffffu};
    std::vector<ContactInput> result;
    for (unsigned mask : masks) {
        ContactInput copy = in;
        copy.hessian_blocks = mask;
        result.push_back(copy);
    }
    return result;
}

#ifdef IPC_ORIGINAL_TEST_CUDA
std::vector<ContactOutput> evaluate(const std::vector<ContactInput>& inputs) {
    std::vector<ContactOutput> outputs(inputs.size());
    gpu_original_contact_compute_gpu(inputs.data(), outputs.data(),
                                     inputs.size());
    return outputs;
}
#else
std::vector<ContactOutput> evaluate(const std::vector<ContactInput>& inputs) {
    std::vector<ContactOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs)
        outputs.push_back(ipc_gpu::solver::evaluate_contact_original(input));
    return outputs;
}
#endif

ContactInput transformed(const ContactInput& in, const Mat33& rotation,
                         const Vec3& translation, double scale) {
    ContactInput copy = in;
    for (int i = 0; i < 4; ++i) {
        Vec3 p(in.positions[i][0], in.positions[i][1], in.positions[i][2]);
        Vec3 q(in.previous[i][0], in.previous[i][1], in.previous[i][2]);
        p = scale * (rotation * p) + translation;
        q = scale * (rotation * q) + translation;
        for (int k = 0; k < 3; ++k) {
            copy.positions[i][k] = p[k];
            copy.previous[i][k] = q[k];
        }
    }
    return copy;
}

bool reference_all_finite(const Reference& ref) {
    if (!std::isfinite(ref.distance)) return false;
    for (double weight : ref.weights)
        if (!std::isfinite(weight)) return false;
    if (!ref.normal.allFinite()) return false;
    for (const auto& row : ref.gradient)
        if (!row.allFinite()) return false;
    for (const auto& row : ref.hessian)
        for (const auto& block : row)
            if (!block.allFinite()) return false;
    if (!ref.friction_gradient.allFinite()) return false;
    if (!ref.friction_hessian.allFinite()) return false;
    return true;
}

// The CPU false evaluation returns an identically zero force payload for an
// inactive contact. Nothing in the unused payload may be a fabricated
// non-finite value.
void require_no_force_payload(const ContactOutput& out,
                              const std::string& tag) {
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            require(out.barrier_gradient[i][k] == 0.0,
                    tag + ": inactive barrier gradient must be zero");
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 3; ++l) {
                    require(out.barrier_hessian[i][j][3 * k + l] == 0.0,
                            tag + ": inactive barrier Hessian must be zero");
                }
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        require(out.friction_relative_gradient[k] == 0.0,
                tag + ": inactive friction gradient must be zero");
        for (int l = 0; l < 3; ++l) {
            require(out.friction_relative_hessian[3 * k + l] == 0.0,
                    tag + ": inactive friction Hessian must be zero");
        }
    }
}

struct CoincidentCase {
    ContactKind kind;
    std::array<Vec3, 4> positions;
    const char* name;
};

std::array<CoincidentCase, 4> coincident_cases() {
    const std::array<Vec3, 4> pt_face{
            Vec3(0.2, 0.2, 0.0), Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    // Coincident with a face vertex: the CPU inclusive barycentric test still
    // reports FaceInterior with distance zero.
    const std::array<Vec3, 4> pt_vertex{
            Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const std::array<Vec3, 4> ee_cross{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
            Vec3(0.5, -0.5, 0.0), Vec3(0.5, 0.5, 0.0)};
    // Collinear overlapping segments: the CPU parallel branch reaches zero
    // distance through a boundary candidate (Edge_t0 here).
    const std::array<Vec3, 4> ee_overlap{
            Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
            Vec3(0.25, 0.0, 0.0), Vec3(0.75, 0.0, 0.0)};
    return {{
            {ContactKind::PointTriangle, pt_face, "PT face"},
            {ContactKind::PointTriangle, pt_vertex, "PT vertex"},
            {ContactKind::EdgeEdge, ee_cross, "EE crossing"},
            {ContactKind::EdgeEdge, ee_overlap, "EE collinear overlap"}}};
}

// CPU parity at exact zero gap with zero stiffness. The CPU evaluation permits
// a valid distance of zero and initialize_contact_barrier_data then returns an
// inactive packet before any singular scalar derivative is requested; frozen
// friction follows the same zero-stiffness early return. The device component
// must therefore report Ok/inactive/zero force for the very same inputs instead
// of masking this into ZeroDistance.
void zero_stiffness_zero_gap_check() {
    const double frictions[] = {0.0, 0.2};
    std::size_t index = 0;
    for (const CoincidentCase& c : coincident_cases()) {
        for (double friction : frictions) {
            ContactInput in = make_input(c.kind, c.positions, 0, friction);
            in.k_barrier = 0.0;
            in.hessian_blocks = 0xffffu;
            const std::string tag = std::string("zero-stiffness zero-gap ")
                    + c.name + " friction " + std::to_string(friction);
            const ContactOutput out = evaluate({in})[0];
            require(out.status == ContactStatus::Ok,
                    tag + ": status " + std::to_string(unsigned(out.status))
                    + " is not Ok");
            require(out.active == 0, tag + ": must be inactive");
            require(out.distance == 0.0, tag + ": distance must be zero");
            require_no_force_payload(out, tag);

            // The real CPU false oracle must agree and must not throw.
            const Reference ref = make_reference(in);
            require(!ref.active, tag + ": CPU oracle reported active");
            require(ref.distance == 0.0, tag + ": CPU oracle distance not zero");
            reference_check(in, out, 80000 + index);
            ++index;
        }
    }
    std::cout << "zero-stiffness zero-gap parity: " << index
              << " PT/EE records checked against the CPU false oracle\n";
}

// Positive stiffness at exact zero gap remains a fail-closed error: the CPU
// scalar derivative is singular, so both the CPU oracle and the device must
// reject it rather than turning it into an inactive contact.
void positive_stiffness_zero_gap_check() {
    const double frictions[] = {0.0, 0.2};
    std::size_t index = 0;
    for (const CoincidentCase& c : coincident_cases()) {
        for (double friction : frictions) {
            ContactInput in = make_input(c.kind, c.positions, 0, friction);
            in.k_barrier = 257.0;
            in.hessian_blocks = 0xffffu;
            const std::string tag = std::string("positive-stiffness zero-gap ")
                    + c.name + " friction " + std::to_string(friction);
            const ContactOutput out = evaluate({in})[0];
            require(out.status == ContactStatus::ZeroDistance,
                    tag + ": status " + std::to_string(unsigned(out.status))
                    + " is not ZeroDistance");
            require(out.active == 0, tag + ": must be inactive");
            require_no_force_payload(out, tag);

            bool cpu_rejected = false;
            try {
                (void)make_reference(in);
            } catch (const std::runtime_error&) {
                cpu_rejected = true;
            }
            require(cpu_rejected,
                    tag + ": CPU false oracle must reject this input");
            ++index;
        }
    }
    std::cout << "positive-stiffness zero-gap rejection: " << index
              << " PT/EE records checked against the CPU false oracle\n";
}

void random_transformed_check() {
    std::mt19937_64 generator(20261030u);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> positive(0.05, 0.30);
    const Mat33 rotation =
            Eigen::AngleAxisd(0.61, Vec3(0.3, -0.8, 0.5).normalized())
                    .toRotationMatrix();

    std::vector<ContactInput> records;
    for (int trial = 0; trial < 40; ++trial) {
        Vec3 a(uniform(generator), uniform(generator), uniform(generator));
        Vec3 b(uniform(generator), uniform(generator), uniform(generator));
        Vec3 c(uniform(generator), uniform(generator), uniform(generator));
        const Vec3 raw_normal = (b - a).cross(c - a);
        if (raw_normal.norm() < 0.10) continue;
        const Vec3 normal = raw_normal.normalized();
        const Vec3 centroid = (a + b + c) / 3.0;
        Vec3 point = centroid + normal * positive(generator);
        // Include exterior points as well as face interior.
        if (trial % 2) point += 0.35 * (a - centroid);
        const double scale = 0.4 + 1.6 * std::abs(uniform(generator));
        const Vec3 translation(3.0 * uniform(generator),
                               3.0 * uniform(generator),
                               3.0 * uniform(generator));
        ContactInput in = make_input(ContactKind::PointTriangle,
                {point, a, b, c}, trial % 5,
                (trial % 3 == 0) ? 0.0 : 0.23);
        in.d_hat = 0.5 * scale;
        records.push_back(transformed(in, rotation, translation, scale));
    }
    for (int trial = 0; trial < 40; ++trial) {
        const Vec3 a(uniform(generator), uniform(generator), uniform(generator));
        const Vec3 b(uniform(generator), uniform(generator), uniform(generator));
        const Vec3 c(uniform(generator), uniform(generator), uniform(generator));
        Vec3 d = c + (b - a) * (0.3 + 0.5 * std::abs(uniform(generator)))
                + Vec3(uniform(generator), uniform(generator),
                       uniform(generator)) * 0.3;
        const double scale = 0.4 + 1.6 * std::abs(uniform(generator));
        const Vec3 translation(3.0 * uniform(generator),
                               3.0 * uniform(generator),
                               3.0 * uniform(generator));
        ContactInput in = make_input(ContactKind::EdgeEdge, {a, b, c, d},
                                     trial % 5,
                                     (trial % 3 == 0) ? 0.0 : 0.23);
        in.d_hat = 0.5 * scale;
        records.push_back(transformed(in, rotation, translation, scale));
    }

    const auto outputs = evaluate(records);
    std::size_t compared = 0;
    std::size_t skipped = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        Reference reference;
        try {
            reference = make_reference(records[i]);
        } catch (const std::exception&) {
            ++skipped;
            continue;
        }
        if (reference.distance == 0.0 || !reference_all_finite(reference)
                || reference.distance < 1.0e-10) {
            ++skipped;
            continue;
        }
        require(outputs[i].status == ContactStatus::Ok,
                "random conditioned case failed on device but not on CPU");
        reference_check(records[i], outputs[i], 10000 + i);
        ++compared;
    }
    require(compared >= 40, "too few random transformed comparisons");
    std::cout << "random transformed comparisons: " << compared
              << " compared, " << skipped << " CPU-degenerate skipped\n";
}

// ---------------------------------------------------------------------------
// Additive regression: near-parallel (small-Gram) EE interior family.
// ---------------------------------------------------------------------------
// In this family A = B = 1, C = 1 + 2^(-2e), Delta = 2^(-2e), and the
// unconstrained parameters are s = offset (inclusive Interior) with t = 0, so
// the closest-point separation is exactly (0, 0, -1/8) and the distance is
// exactly 1/8.  The production CPU false evaluation contracts the trailing
// products of the Interior derivative numerators; the uncontracted
// transcription loses a quarter ulp before an O(1e15) cancellation and
// reported a spurious H00yy (measured -0.0093214623481245734 at 2^-25 with
// offset 1/16).  Both segment directions have zero z component, so sliding
// endpoint 0 along its supporting y axis cannot change the infinite-line
// separation and the exact Hessian entry in that direction is zero.
std::array<Vec3, 4> small_gram_positions(int exponent, double offset) {
    const double tilt = std::ldexp(1.0, -exponent);
    return {{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
             Vec3(0.0, offset, 0.125), Vec3(tilt, 1.0 + offset, 0.125)}};
}

// The legacy Gram product C * D is exactly representable for every family
// member except (exponent 26, offset 3/8): there it needs 54 significant
// bits, so the value of the legacy feature parameter itself depends on the
// host archive's multiply-add contraction.  That single member is reported by
// small_gram_contract_diagnostics() instead of being asserted as parity.
bool small_gram_contract_invariant(int exponent, double offset) {
    return !(exponent == 26 && offset == 3.0 / 8.0);
}

ContactInput small_gram_input(int exponent, double offset, int slip,
                              double friction) {
    return make_input(ContactKind::EdgeEdge,
                      small_gram_positions(exponent, offset), slip, friction);
}

// The CPU false oracle is the source of truth for the feature premises.  A
// robust fallback, loss of the inclusive Interior endpoint, or a different
// feature parameter invalidates the regression, so each premise is asserted
// rather than skipped.
void require_small_gram_premise(const ContactInput& in, double offset,
                                const std::string& tag) {
    std::array<Vec3, 4> x;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) x[i][k] = in.positions[i][k];
    const auto evaluation = make_segment_segment_contact_evaluation(
            x, in.d_hat, in.k_barrier, in.eps, nullptr, false);
    require(!evaluation.dr.robust,
            tag + ": CPU false solve must not use the robust fallback");
    require(evaluation.active, tag + ": CPU legacy contact must be active");
    require(evaluation.dr.region == SegmentSegmentRegion::Interior,
            tag + ": CPU legacy region is not Interior");
    require(evaluation.dr.s == offset,
            tag + ": CPU legacy s is not the requested interior offset");
    require(evaluation.dr.t == 0.0,
            tag + ": CPU legacy inclusive Interior did not keep t == 0");
    require(evaluation.dr.distance == 0.125,
            tag + ": CPU legacy distance is not exactly 1/8");
}

struct RigidTransformSpec {
    Mat33 rotation;
    Vec3 translation;
};

// Signed-axis symmetries (reflections, axis swaps, quarter turns) plus a
// dyadic translation.  These are orthogonal, so the exactly representable
// Gram products of the family stay exact and the CPU feature premises are
// unchanged; the reflected orientations reproduce the spurious H00yy.
std::vector<RigidTransformSpec> signed_axis_transforms() {
    std::vector<RigidTransformSpec> result;
    const Vec3 none = Vec3::Zero();
    Mat33 m;
    m = Mat33::Identity(); m(0, 0) = -1.0; result.push_back({m, none});
    m = Mat33::Identity(); m(1, 1) = -1.0; result.push_back({m, none});
    m = Mat33::Identity(); m(2, 2) = -1.0; result.push_back({m, none});
    m = -Mat33::Identity(); result.push_back({m, none});
    m.setZero(); m(0, 1) = 1.0; m(1, 0) = 1.0; m(2, 2) = 1.0;
    result.push_back({m, none});
    m.setZero(); m(0, 1) = -1.0; m(1, 0) = 1.0; m(2, 2) = 1.0;
    result.push_back({m, none});
    m.setZero(); m(0, 0) = 1.0; m(1, 2) = -1.0; m(2, 1) = 1.0;
    result.push_back({m, none});
    m.setZero(); m(0, 2) = 1.0; m(1, 1) = 1.0; m(2, 0) = -1.0;
    result.push_back({m, none});
    result.push_back({Mat33::Identity(), Vec3(1024.0, -2048.0, 4096.0)});
    return result;
}

ContactInput relabelled(const ContactInput& in,
                        const std::array<int, 4>& order) {
    ContactInput copy = in;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) {
            copy.positions[i][k] = in.positions[order[i]][k];
            copy.previous[i][k] = in.previous[order[i]][k];
        }
    }
    return copy;
}

void small_gram_check() {
    const int exponents[] = {20, 22, 24, 25, 26};
    const double offsets[] = {1.0 / 16.0, 1.0 / 8.0, 3.0 / 8.0};
    const unsigned masks[] = {0xffffu, 0x0001u};
    const double frictions[] = {0.0, 0.23};

    std::vector<ContactInput> requests;
    std::size_t premises = 0;

    // Base family: every slip regime of make_input(), with and without
    // friction, full and self Hessian masks.
    for (int exponent : exponents) {
        for (double offset : offsets) {
            if (!small_gram_contract_invariant(exponent, offset)) continue;
            const std::string member = "small-Gram 2^-"
                    + std::to_string(exponent) + " offset "
                    + std::to_string(offset);
            require_small_gram_premise(
                    small_gram_input(exponent, offset, 2, 0.23), offset,
                    member);
            ++premises;
            for (int slip = 0; slip < 5; ++slip) {
                for (double friction : frictions) {
                    for (unsigned mask : masks) {
                        ContactInput in = small_gram_input(
                                exponent, offset, slip, friction);
                        in.hessian_blocks = mask;
                        requests.push_back(in);
                    }
                }
            }
        }
    }

    // The measured native failure configuration, with the exact scalar
    // barrier parameters used by the three-way CPU/host/device diagnostic
    // (d_hat = 0.25, k_barrier = 37, eps = 1e-12, x3.x = 2^-25).
    for (int slip = 0; slip < 5; ++slip) {
        for (unsigned mask : masks) {
            ContactInput in = small_gram_input(25, 1.0 / 16.0, slip, 0.23);
            in.d_hat = 0.25;
            in.k_barrier = 37.0;
            in.hessian_blocks = mask;
            requests.push_back(in);
        }
    }

    // Signed-axis rigid transforms of the contraction-sensitive members.  The
    // frame still contains all four roles and every requested cross block.
    const int transformed_exponents[] = {25, 26};
    const double transformed_offsets[] = {1.0 / 16.0, 1.0 / 8.0};
    for (const RigidTransformSpec& spec : signed_axis_transforms()) {
        for (int exponent : transformed_exponents) {
            for (double offset : transformed_offsets) {
                for (int slip : {0, 3}) {
                    for (unsigned mask : masks) {
                        ContactInput in = transformed(
                                small_gram_input(exponent, offset, slip, 0.23),
                                spec.rotation, spec.translation, 1.0);
                        in.hessian_blocks = mask;
                        requests.push_back(in);
                    }
                }
            }
        }
    }

    // Segment relabelling (reversal, edge swap) on the exactly representable
    // Gram members, where the legacy distance solve is independent of the
    // host multiply-add contract.  Each relabelling keeps the inclusive
    // SegmentSegmentRegion::Interior classification but moves the surviving
    // free parameter to an endpoint (s or t exactly 0 or 1).  The segment
    // swap keeps every Gram product exact, so it is also compared at the two
    // largest exponents; reversals there would change which product rounds
    // inside the legacy distance solve and are covered as a reported
    // FP-contract diagnostic instead.
    const std::array<std::array<int, 4>, 4> orders{{
            {{1, 0, 2, 3}},
            {{0, 1, 3, 2}},
            {{2, 3, 0, 1}},
            {{1, 0, 3, 2}}}};
    for (const std::array<int, 4>& order : orders) {
        const bool swap_segments = order[0] == 2;
        for (int exponent : {20, 22, 24, 25, 26}) {
            if (!swap_segments && exponent > 24) continue;
            for (double offset : offsets) {
                if (!small_gram_contract_invariant(exponent, offset)) continue;
                for (int slip : {2, 4}) {
                    for (unsigned mask : masks) {
                        ContactInput in = relabelled(
                                small_gram_input(exponent, offset, slip, 0.23),
                                order);
                        in.hessian_blocks = mask;
                        requests.push_back(in);
                    }
                }
            }
        }
    }

    const auto outputs = evaluate(requests);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        reference_check(requests[i], outputs[i], 200000 + i);
        // The base family, its rigid transforms and the segment relabellings
        // all keep the exact supporting-line separation of 1/8.
        compare(outputs[i].distance, 0.125,
                "record " + std::to_string(200000 + i)
                + " small-Gram invariant distance");
    }
    std::cout << "small-Gram near-parallel EE: " << premises
              << " CPU feature premises asserted, " << requests.size()
              << " masked/transformed records compared\n";
}

// Independent witness for the null supporting-line Hessian direction.  It
// does not reuse the production derivative algebra: both segment directions
// have zero z component and the supporting lines lie in z = 0 and z = 1/8, so
// the exact infinite-line distance |c.(a x b)|/|a x b| is 1/8 for every slide
// of endpoint 0 along y.  The exact second difference is therefore zero and
// the barrier Hessian entry H00yy (and the matching gradient component) must
// vanish.  The measured production CPU false oracle returns exactly +0.
void supporting_line_null_witness() {
    const int exponent = 25;
    const double offset = 1.0 / 16.0;
    const double steps[] = {std::ldexp(1.0, -8), std::ldexp(1.0, -20)};
    for (double step : steps) {
        double second_difference = 0.0;
        for (int side = -1; side <= 1; ++side) {
            std::array<Vec3, 4> x = small_gram_positions(exponent, offset);
            x[0][1] += double(side) * step;
            const Vec3 a = x[1] - x[0];
            const Vec3 b = x[3] - x[2];
            const Vec3 c = x[0] - x[2];
            const Vec3 normal = a.cross(b);
            const double exact = std::abs(c.dot(normal)) / normal.norm();
            compare(exact, 0.125,
                    "supporting-line witness h=" + std::to_string(step)
                    + " slide " + std::to_string(side)
                    + " exact infinite-line distance");
            second_difference += side == 0 ? -2.0 * exact : exact;
        }
        // The exact supporting-line distance is constant, so this is an
        // absolute zero (up to a few ulps of 1/8); the curvature follows.
        compare(second_difference, 0.0,
                "supporting-line witness h=" + std::to_string(step)
                + " exact second difference");
        compare(second_difference / (step * step), 0.0,
                "supporting-line witness h=" + std::to_string(step)
                + " exact null curvature");
    }

    ContactInput in = small_gram_input(exponent, offset, 2, 0.23);
    in.hessian_blocks = 0xffffu;
    const Reference reference = make_reference(in);
    compare(reference.distance, 0.125, "supporting-line witness CPU distance");
    compare(reference.gradient[0][1], 0.0,
            "supporting-line witness CPU null gradient");
    compare(reference.hessian[0][0](1, 1), 0.0,
            "supporting-line witness CPU null Hessian");

    const ContactOutput out = evaluate({in})[0];
    require(out.status == ContactStatus::Ok,
            "supporting-line witness: device status is not Ok");
    compare(out.barrier_gradient[0][1], 0.0,
            "supporting-line witness device null gradient");
    compare(out.barrier_hessian[0][0][3 * 1 + 1], 0.0,
            "supporting-line witness device null Hessian");
    std::cout << "supporting-line null Hessian witness: exact line distance"
              << " constant to second order; CPU and device H00yy zero\n";
}

// Report, without manufacturing agreement, the two family members whose
// legacy Gram product itself rounds (C * D needs more than 53 bits).  Their
// feature parameter is sensitive to the host archive's multiply-add
// contraction, which is a separate floating-point contract from the
// derivative defect fixed here.  Divergence is printed, never asserted away.
void small_gram_contract_diagnostics() {
    struct Entry {
        int exponent;
        double offset;
        bool reverse;
        const char* label;
    };
    const Entry entries[] = {
            {26, 3.0 / 8.0, false, "2^-26 offset 3/8"},
            {25, 1.0 / 16.0, true, "2^-25 offset 1/16 reversed"}};
    for (const Entry& entry : entries) {
        std::array<Vec3, 4> x = small_gram_positions(entry.exponent,
                                                     entry.offset);
        if (entry.reverse) {
            const Vec3 first = x[0];
            x[0] = x[1];
            x[1] = first;
        }
        ContactInput in = make_input(ContactKind::EdgeEdge, x, 0, 0.23);
        in.hessian_blocks = 0xffffu;

        double cpu_s = 0.0, cpu_t = 0.0, cpu_distance = 0.0;
        int cpu_region = -1;
        try {
            const auto evaluation = make_segment_segment_contact_evaluation(
                    x, in.d_hat, in.k_barrier, in.eps, nullptr, false);
            cpu_s = evaluation.dr.s;
            cpu_t = evaluation.dr.t;
            cpu_distance = evaluation.dr.distance;
            cpu_region = int(evaluation.dr.region);
        } catch (const std::exception&) {
            cpu_region = -1;
        }

        const ContactOutput out = evaluate({in})[0];
        double max_error = std::numeric_limits<double>::infinity();
        try {
            const Reference reference = make_reference(in);
            max_error = 0.0;
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 3; ++k) {
                    max_error = std::max(max_error,
                            std::abs(out.barrier_gradient[i][k]
                                     - reference.gradient[i][k])
                            / (1.0 + std::abs(reference.gradient[i][k])));
                    for (int j = 0; j < 4; ++j) {
                        for (int l = 0; l < 3; ++l) {
                            const double expected =
                                    reference.hessian[i][j](k, l);
                            max_error = std::max(max_error,
                                    std::abs(out.barrier_hessian[i][j][3 * k + l]
                                             - expected)
                                    / (1.0 + std::abs(expected)));
                        }
                    }
                }
            }
        } catch (const std::exception&) {
        }

        std::cout << "small-Gram FP-contract diagnostic (" << entry.label
                  << "): CPU region " << cpu_region << " s=" << cpu_s
                  << " t=" << cpu_t << " distance=" << cpu_distance
                  << ", device distance=" << out.distance
                  << ", max scaled divergence " << max_error
                  << " (reported, not a parity claim)\n";
    }
}

// Report, without manufacturing agreement, the genuinely cancellation-prone
// legacy EE regime. Ordinary tests above keep the strict 2e-9 floor.
void ill_conditioned_report() {
    ContactInput in = make_input(ContactKind::EdgeEdge,
            {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
             Vec3(0.2, -0.5, 1.0e-15), Vec3(0.2, 0.5, 1.0e-15)}, 0, 0.0);
    in.d_hat = 0.5;
    in.k_barrier = 257.0;
    const auto out = evaluate({in})[0];
    try {
        const Reference reference = make_reference(in);
        if (out.status != ContactStatus::Ok) {
            std::cout << "ill-conditioned legacy EE: device status "
                      << unsigned(out.status) << " (CPU distance "
                      << reference.distance << ")\n";
            return;
        }
        double max_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 3; ++k) {
                max_error = std::max(max_error,
                        std::abs(out.barrier_gradient[i][k]
                                 - reference.gradient[i][k])
                        / (1.0 + std::abs(reference.gradient[i][k])));
                for (int j = 0; j < 4; ++j) {
                    for (int l = 0; l < 3; ++l) {
                        const double expected = reference.hessian[i][j](k, l);
                        max_error = std::max(max_error,
                                std::abs(out.barrier_hessian[i][j][3 * k + l]
                                         - expected)
                                / (1.0 + std::abs(expected)));
                    }
                }
            }
        }
        std::cout << "ill-conditioned legacy EE: gap " << reference.distance
                  << ", max scaled divergence " << max_error
                  << (max_error > 2.0e-9 ? " (real legacy divergence reported)"
                                         : " (within strict floor)") << "\n";
    } catch (const std::exception& error) {
        std::cout << "ill-conditioned legacy EE: CPU false oracle rejected ("
                  << error.what() << ")\n";
    }
}

void failure_check() {
    std::vector<ContactInput> failures;

    ContactInput pt_zero = make_input(ContactKind::PointTriangle,
            {Vec3(0.2, 0.2, 0.0), Vec3(0.0, 0.0, 0.0),
             Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)}, 0, 0.0);
    failures.push_back(pt_zero);
    ContactInput ee_zero = make_input(ContactKind::EdgeEdge,
            {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
             Vec3(0.5, -0.5, 0.0), Vec3(0.5, 0.5, 0.0)}, 0, 0.0);
    failures.push_back(ee_zero);

    ContactInput invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.dt = 0.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.d_hat = 0.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.eps = 0.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.eps_v = 0.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.k_barrier = -1.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.friction = -1.0;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.hessian_blocks = 1u << 16;
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.kind = static_cast<ContactKind>(7);
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = std::numeric_limits<double>::quiet_NaN();
    failures.push_back(invalid);
    invalid = pt_zero;
    invalid.positions[0][2] = 0.1;
    invalid.previous[0][0] = std::numeric_limits<double>::infinity();
    failures.push_back(invalid);

    const auto outputs = evaluate(failures);
    for (std::size_t i = 0; i < failures.size(); ++i) {
        require(outputs[i].status != ContactStatus::Ok,
                "failure fixture " + std::to_string(i) + " must fail closed");
        require(outputs[i].active == 0,
                "failure fixture " + std::to_string(i)
                + " must not advertise an active contact");
    }
    require(outputs[0].status == ContactStatus::ZeroDistance,
            "point-triangle zero gap must be ZeroDistance");
    require(outputs[1].status == ContactStatus::ZeroDistance,
            "segment--segment zero gap must be ZeroDistance");
    for (std::size_t i = 2; i < failures.size(); ++i) {
        require(outputs[i].status == ContactStatus::InvalidInput,
                "invalid input fixture " + std::to_string(i)
                + " must be InvalidInput");
    }

    // Confirm the corresponding CPU builders also reject these inputs so the
    // test gates are meaningful rather than device-specific conventions.
    std::array<Vec3, 4> x;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) x[i][k] = pt_zero.positions[i][k];
    }
    bool zero_rejected = false;
    try {
        make_node_triangle_contact_evaluation(
                x, pt_zero.d_hat, pt_zero.k_barrier, pt_zero.eps,
                nullptr, false);
    } catch (const std::runtime_error&) {
        zero_rejected = true;
    }
    require(zero_rejected, "CPU false point-triangle zero gap must throw");

    std::array<Vec3, 4> ex;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 3; ++k) ex[i][k] = ee_zero.positions[i][k];
    }
    bool ee_zero_rejected = false;
    try {
        make_segment_segment_contact_evaluation(
                ex, ee_zero.d_hat, ee_zero.k_barrier, ee_zero.eps,
                nullptr, false);
    } catch (const std::runtime_error&) {
        ee_zero_rejected = true;
    }
    require(ee_zero_rejected, "CPU false segment--segment zero gap must throw");
}

}  // namespace

int main() {
    try {
        const auto base = fixtures();
        std::vector<ContactInput> requests;
        for (const auto& input : base) {
            const auto variants = masked_variants(input);
            requests.insert(requests.end(), variants.begin(), variants.end());
        }
        const auto outputs = evaluate(requests);
        for (std::size_t i = 0; i < requests.size(); ++i)
            reference_check(requests[i], outputs[i], i);

        // k_barrier == 0 is a friction-only / no-force record in the CPU
        // legacy evaluation: geometry is still reported, all derivatives are zero.
        ContactInput zero_stiffness = base[0];
        zero_stiffness.k_barrier = 0.0;
        zero_stiffness.friction = 0.0;
        zero_stiffness.hessian_blocks = 0xffffu;
        reference_check(zero_stiffness, evaluate({zero_stiffness})[0], 70000);
        ContactInput zero_stiffness_friction = base[0];
        zero_stiffness_friction.k_barrier = 0.0;
        zero_stiffness_friction.friction = 0.23;
        zero_stiffness_friction.hessian_blocks = 0xffffu;
        reference_check(zero_stiffness_friction,
                        evaluate({zero_stiffness_friction})[0], 70001);

        zero_stiffness_zero_gap_check();
        positive_stiffness_zero_gap_check();
        random_transformed_check();
        small_gram_check();
        supporting_line_null_witness();
        small_gram_contract_diagnostics();
        ill_conditioned_report();
        failure_check();

        std::cout << "original contact: " << requests.size()
                  << " masked fixture records, " << components
                  << " compared components, maximum scaled error "
                  << worst_scaled_error << " at " << worst_label
#ifdef IPC_ORIGINAL_TEST_CUDA
                  << " (CUDA device execution, real kernel launch)\n";
#else
                  << " (HOST ONLY: no CUDA qualification)\n";
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

#ifdef IPC_ORIGINAL_TEST_CUDA
#undef IPC_ORIGINAL_TEST_CUDA
#endif

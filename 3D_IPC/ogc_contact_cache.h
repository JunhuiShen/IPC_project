#pragma once

#include "barrier_energy.h"
#include "broad_phase.h"
#include "friction_energy.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver_detail {

// Frozen geometry makes these operands identical for all incident vertex
// roles. Keep the AABB decision separate from barrier activation: accepted
// pairs beyond d_hat still contribute the legacy zero additions at each role.
// SoA storage costs one byte and two doubles per candidate, without padding.
struct OGCFrozenContactOperands {
    std::vector<unsigned char> nt_aabb_active, ss_aabb_active;
    std::vector<double> nt_b_prime, nt_b_double_prime;
    std::vector<double> ss_b_prime, ss_b_double_prime;

    void resize(std::size_t nt_count, std::size_t ss_count) {
        nt_aabb_active.resize(nt_count);
        ss_aabb_active.resize(ss_count);
        nt_b_prime.resize(nt_count);
        nt_b_double_prime.resize(nt_count);
        ss_b_prime.resize(ss_count);
        ss_b_double_prime.resize(ss_count);
    }

    // Each call writes one exclusive pair slot; the caller supplies the team
    // and joins preparation before any role consumes the cached operands.
    void prepare_node_triangle(std::size_t index, const NodeTrianglePair& pair,
        const std::vector<Vec3>& positions,
        const NodeTriangleDistanceResult& distance, double d_hat) {
        nt_b_prime[index] = 0.0;
        nt_b_double_prime[index] = 0.0;
        const bool accepted = d_hat > 0.0
            && node_triangle_aabbs_within_distance(positions[pair.node],
                positions[pair.tri_v[0]], positions[pair.tri_v[1]],
                positions[pair.tri_v[2]], d_hat * d_hat);
        nt_aabb_active[index] = static_cast<unsigned char>(accepted);
        if (!accepted) return;
        // The normal-only legacy path evaluates these even at k_barrier==0.
        // Do not substitute make_contact_evaluation's zero-stiffness policy.
        nt_b_prime[index] = scalar_barrier_gradient(distance.distance, d_hat);
        nt_b_double_prime[index] = scalar_barrier_hessian(distance.distance, d_hat);
    }

    void prepare_segment_segment(std::size_t index, const SegmentSegmentPair& pair,
        const std::vector<Vec3>& positions,
        const SegmentSegmentDistanceResult& distance, double d_hat) {
        ss_b_prime[index] = 0.0;
        ss_b_double_prime[index] = 0.0;
        const bool accepted = d_hat > 0.0
            && segment_aabbs_within_distance(positions[pair.v[0]],
                positions[pair.v[1]], positions[pair.v[2]],
                positions[pair.v[3]], d_hat * d_hat);
        ss_aabb_active[index] = static_cast<unsigned char>(accepted);
        if (!accepted) return;
        ss_b_prime[index] = scalar_barrier_gradient(distance.distance, d_hat);
        ss_b_double_prime[index] = scalar_barrier_hessian(distance.distance, d_hat);
    }

    std::size_t size_bytes() const {
        return (nt_aabb_active.size() + ss_aabb_active.size())
            * (sizeof(unsigned char) + 2 * sizeof(double));
    }

    std::size_t capacity_bytes() const {
        return (nt_aabb_active.capacity() + ss_aabb_active.capacity())
                * sizeof(unsigned char)
            + (nt_b_prime.capacity() + nt_b_double_prime.capacity()
                + ss_b_prime.capacity() + ss_b_double_prime.capacity())
                * sizeof(double);
    }
};

// Friction stores the larger evaluation/relative derivatives only for pairs with
// active normal barriers. Dense masks and slot maps retain constant-time
// lookup in original vertex incidence order. Preparation has three phases:
// parallel classify_*, one compact(), then parallel prepare_* and a join.
struct OGCFrozenFrictionOperands {
    template <class Evaluation>
    struct Record {
        Evaluation evaluation;
        std::array<double, 4> weights{{0.0, 0.0, 0.0, 0.0}};
        Vec3 relative_gradient = Vec3::Zero();
        Mat33 relative_hessian = Mat33::Zero();
    };
    using NodeTriangleRecord = Record<NodeTriangleContactEvaluation>;
    using SegmentSegmentRecord = Record<SegmentSegmentContactEvaluation>;
    static constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();

    std::vector<unsigned char> nt_aabb_active, ss_aabb_active;
    std::vector<std::size_t> nt_slots, ss_slots;
    std::vector<NodeTriangleRecord> nt_records;
    std::vector<SegmentSegmentRecord> ss_records;

    void resize(std::size_t nt_count, std::size_t ss_count) {
        nt_aabb_active.resize(nt_count);
        ss_aabb_active.resize(ss_count);
        nt_slots.resize(nt_count);
        ss_slots.resize(ss_count);
    }

    void classify_node_triangle(std::size_t index, const NodeTrianglePair& pair,
        const std::vector<Vec3>& positions,
        const NodeTriangleDistanceResult& distance,
        double d_hat, double k_barrier) {
        const bool accepted = d_hat > 0.0
            && node_triangle_aabbs_within_distance(positions[pair.node],
                positions[pair.tri_v[0]], positions[pair.tri_v[1]],
                positions[pair.tri_v[2]], d_hat * d_hat);
        nt_aabb_active[index] = static_cast<unsigned char>(accepted);
        nt_slots[index] = accepted && k_barrier != 0.0 && distance.distance < d_hat
            ? 0 : missing;
    }

    void classify_segment_segment(std::size_t index, const SegmentSegmentPair& pair,
        const std::vector<Vec3>& positions,
        const SegmentSegmentDistanceResult& distance,
        double d_hat, double k_barrier) {
        const bool accepted = d_hat > 0.0
            && segment_aabbs_within_distance(positions[pair.v[0]],
                positions[pair.v[1]], positions[pair.v[2]],
                positions[pair.v[3]], d_hat * d_hat);
        ss_aabb_active[index] = static_cast<unsigned char>(accepted);
        ss_slots[index] = accepted && k_barrier != 0.0 && distance.distance < d_hat
            ? 0 : missing;
    }

    void compact() {
        std::size_t nt_count = 0, ss_count = 0;
        for (auto& slot : nt_slots)
            if (slot != missing) slot = nt_count++;
        for (auto& slot : ss_slots)
            if (slot != missing) slot = ss_count++;
        nt_records.resize(nt_count);
        ss_records.resize(ss_count);
    }

    void prepare_node_triangle(std::size_t index, const NodeTrianglePair& pair,
        const std::vector<Vec3>& positions, const std::vector<Vec3>& previous,
        const NodeTriangleDistanceResult& distance, const SimParams& params) {
        if (!nt_aabb_active[index]) return;
        const std::array<Vec3, 4> current{
            positions[pair.node], positions[pair.tri_v[0]],
            positions[pair.tri_v[1]], positions[pair.tri_v[2]]};
        const std::array<Vec3, 4> prior{
            previous[pair.node], previous[pair.tri_v[0]],
            previous[pair.tri_v[1]], previous[pair.tri_v[2]]};
        const auto evaluation = make_node_triangle_contact_evaluation(current,
            params.d_hat, params.k_barrier, 1.0e-12, &distance,
            params.exact_computation_fallback);
        const auto friction = make_node_triangle_frozen_friction_contact(
            current, prior, evaluation, params.dt(), params.friction_velocity_epsilon);
        const auto derivatives = frozen_friction_relative_gradient_and_hessian(
            friction, params.friction_coefficient, params.dt2());
        // Inactive accepted pairs still execute both builders above, preserving
        // validation and zero-stiffness behavior without retaining their data.
        if (nt_slots[index] != missing)
            nt_records[nt_slots[index]] = {
                evaluation, friction.weights, derivatives.first, derivatives.second};
    }

    void prepare_segment_segment(std::size_t index, const SegmentSegmentPair& pair,
        const std::vector<Vec3>& positions, const std::vector<Vec3>& previous,
        const SegmentSegmentDistanceResult& distance, const SimParams& params) {
        if (!ss_aabb_active[index]) return;
        const std::array<Vec3, 4> current{
            positions[pair.v[0]], positions[pair.v[1]],
            positions[pair.v[2]], positions[pair.v[3]]};
        const std::array<Vec3, 4> prior{
            previous[pair.v[0]], previous[pair.v[1]],
            previous[pair.v[2]], previous[pair.v[3]]};
        const auto evaluation = make_segment_segment_contact_evaluation(current,
            params.d_hat, params.k_barrier, 1.0e-12, &distance,
            params.exact_computation_fallback);
        const auto friction = make_segment_segment_frozen_friction_contact(
            current, prior, evaluation, params.dt(), params.friction_velocity_epsilon);
        const auto derivatives = frozen_friction_relative_gradient_and_hessian(
            friction, params.friction_coefficient, params.dt2());
        if (ss_slots[index] != missing)
            ss_records[ss_slots[index]] = {
                evaluation, friction.weights, derivatives.first, derivatives.second};
    }

    const NodeTriangleContactEvaluation& node_triangle_evaluation(std::size_t index) const {
        return nt_slots[index] == missing ? inactive_nt : nt_records[nt_slots[index]].evaluation;
    }

    const SegmentSegmentContactEvaluation& segment_segment_evaluation(std::size_t index) const {
        return ss_slots[index] == missing ? inactive_ss : ss_records[ss_slots[index]].evaluation;
    }

    std::pair<Vec3, Mat33> node_triangle_role_derivatives(std::size_t index, int role) const {
        validate_role(role);
        if (nt_slots[index] == missing)
            return {Vec3::Zero(), Mat33::Zero()};
        const auto& record = nt_records[nt_slots[index]];
        const double weight = record.weights[role];
        return {weight * record.relative_gradient,
            weight * weight * record.relative_hessian};
    }

    std::pair<Vec3, Mat33> segment_segment_role_derivatives(std::size_t index, int role) const {
        validate_role(role);
        if (ss_slots[index] == missing)
            return {Vec3::Zero(), Mat33::Zero()};
        const auto& record = ss_records[ss_slots[index]];
        const double weight = record.weights[role];
        return {weight * record.relative_gradient,
            weight * weight * record.relative_hessian};
    }

    std::size_t size_bytes() const {
        return (nt_slots.size() + ss_slots.size())
                * (sizeof(std::size_t) + sizeof(unsigned char))
            + nt_records.size() * sizeof(NodeTriangleRecord)
            + ss_records.size() * sizeof(SegmentSegmentRecord);
    }

    std::size_t capacity_bytes() const {
        return (nt_aabb_active.capacity() + ss_aabb_active.capacity())
                * sizeof(unsigned char)
            + (nt_slots.capacity() + ss_slots.capacity()) * sizeof(std::size_t)
            + nt_records.capacity() * sizeof(NodeTriangleRecord)
            + ss_records.capacity() * sizeof(SegmentSegmentRecord);
    }

private:
    static void validate_role(int role) {
        if (role < 0 || role >= 4)
            throw std::invalid_argument("frozen friction: role index must be in [0, 3].");
    }

    // Coefficient/dt2 validation executes once during preparation, including
    // accepted inactive pairs. Role consumers retain their index validation.
    NodeTriangleContactEvaluation inactive_nt{};
    SegmentSegmentContactEvaluation inactive_ss{};
};

// Robust active SS contacts share expensive exact Hessian geometry across
// their four vertex roles. Only the final unscaled self blocks are retained.
// Classify every pair, compact once, prepare independent pairs, then join
// before consuming. Normal-only callers keep terms enabled at k_barrier==0;
// friction callers use their evaluation's normal-barrier activation policy.
struct OGCFrozenSegmentHessians {
    struct Record {
        // Keep records initialized even when a preceding preparation threw:
        // later vector growth must not move unwritten Eigen coefficients.
        std::array<Mat33, 4> hessians{{Mat33::Zero(), Mat33::Zero(),
            Mat33::Zero(), Mat33::Zero()}};
    };
    static constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> slots;
    std::vector<Record> records;

    void resize(std::size_t pair_count) { slots.resize(pair_count); }

    void classify(std::size_t index, const SegmentSegmentDistanceResult& distance,
        double d_hat, bool aabb_active, bool normal_terms_enabled = true) {
        slots[index] = aabb_active && normal_terms_enabled && distance.robust
                && distance.distance < d_hat ? 0 : missing;
    }

    void compact() {
        std::size_t count = 0;
        for (auto& slot : slots)
            if (slot != missing) slot = count++;
        records.resize(count);
    }

    void prepare_segment_segment(std::size_t index, const SegmentSegmentPair& pair,
        const std::vector<Vec3>& positions,
        const SegmentSegmentDistanceResult& distance, double d_hat,
        double b_prime, double b_double_prime) {
        if (slots[index] == missing) return;
        records[slots[index]].hessians = segment_segment_barrier_self_hessians(
            positions[pair.v[0]], positions[pair.v[1]], positions[pair.v[2]],
            positions[pair.v[3]], d_hat, distance, b_prime, b_double_prime);
    }

    const Mat33* self_hessian(std::size_t index, int role) const {
        if (role < 0 || role >= 4)
            throw std::invalid_argument("OGC segment Hessian cache: role index must be in [0, 3].");
        return slots[index] == missing ? nullptr : &records[slots[index]].hessians[role];
    }

    std::size_t size_bytes() const {
        return slots.size() * sizeof(std::size_t)
            + records.size() * sizeof(Record);
    }

    std::size_t capacity_bytes() const {
        return slots.capacity() * sizeof(std::size_t)
            + records.capacity() * sizeof(Record);
    }
};

} // namespace solver_detail

#include "friction_energy.h"

#include <cmath>
#include <stdexcept>

namespace {

void require_positive_finite(double value, const char* message) {
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(message);
}

void require_nonnegative_finite(double value, const char* message) {
    if (value < 0.0 || !std::isfinite(value))
        throw std::invalid_argument(message);
}

void validate_builder_parameters(
        double d_hat, double k_barrier, double dt, double eps_v, double eps) {
    require_positive_finite(d_hat, "friction contact: d_hat must be positive and finite.");
    require_nonnegative_finite(k_barrier, "friction contact: k_barrier must be nonnegative and finite.");
    require_positive_finite(dt, "friction contact: dt must be positive and finite.");
    require_positive_finite(eps_v, "friction contact: eps_v must be positive and finite.");
    require_positive_finite(eps, "friction contact: eps must be positive and finite.");
}

void validate_distance(double distance) {
    if (distance < 0.0 || !std::isfinite(distance))
        throw std::invalid_argument("friction contact: distance must be nonnegative and finite.");
}

void validate_evaluation_parameters(double friction_coefficient, double dt2) {
    require_nonnegative_finite(
            friction_coefficient,
            "frozen friction: friction coefficient must be nonnegative and finite.");
    require_nonnegative_finite(dt2, "frozen friction: dt2 must be nonnegative and finite.");
}

// Validate only the data needed to form the common friction scale. In
// particular, neither the stored normal nor the projector is used by the
// energy or relative gradient. The projector is checked separately by the
// Hessian evaluations that actually consume it.
void validate_active_contact_scale_data(const FrozenFrictionContact& contact) {
    require_positive_finite(
            contact.eps_u, "frozen friction: eps_u must be positive and finite.");
    require_positive_finite(
            contact.normal_force,
            "frozen friction: active contact normal force must be positive and finite.");
    if (!contact.tangential_displacement.allFinite()) {
        throw std::invalid_argument("frozen friction: contact data must be finite.");
    }
}

void validate_active_contact_projector(const FrozenFrictionContact& contact) {
    if (!contact.projector.allFinite())
        throw std::invalid_argument("frozen friction: contact data must be finite.");
}

double smooth_friction_potential_unchecked(double slip, double eps_u) {
    if (slip < eps_u)
        return slip * slip / eps_u
                - slip * slip * slip / (3.0 * eps_u * eps_u);
    return slip - eps_u / 3.0;
}

double smooth_friction_mollifier_over_slip_unchecked(
        double slip, double eps_u) {
    if (slip < eps_u)
        return 2.0 / eps_u - slip / (eps_u * eps_u);
    return 1.0 / slip;
}

Vec3 weighted_relative_displacement(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const std::array<double, 4>& weights) {
    Vec3 displacement = Vec3::Zero();
    for (int i = 0; i < 4; ++i)
        displacement += weights[i] * (current_positions[i] - previous_positions[i]);
    if (!displacement.allFinite())
        throw std::invalid_argument("friction contact: positions must be finite.");
    return displacement;
}

void initialize_mesh_contact_tangent_frame(
        FrozenFrictionContact& contact, const Vec3& separation) {
    const double separation_norm = separation.norm();
    if (!(separation_norm > 0.0) || !std::isfinite(separation_norm)) {
        throw std::runtime_error(
                "friction contact: separation must have nonzero finite norm.");
    }
    contact.normal = separation / separation_norm;
    contact.projector = Mat33::Identity()
            - contact.normal * contact.normal.transpose();
}

template <typename Evaluation, typename InitializeGeometry>
FrozenFrictionContact make_mesh_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const Evaluation& evaluation, double dt, double eps_v,
        InitializeGeometry initialize_geometry) {
    require_positive_finite(
            dt, "friction contact: dt must be positive and finite.");
    require_positive_finite(
            eps_v, "friction contact: eps_v must be positive and finite.");

    FrozenFrictionContact contact;
    contact.eps_u = dt * eps_v;
    require_positive_finite(
            contact.eps_u,
            "friction contact: dt * eps_v must be positive and finite.");
    contact.normal_force = evaluation.normal_load;
    if (!std::isfinite(contact.normal_force))
        throw std::runtime_error("friction contact: normal force is not finite.");
    if (!evaluation.active || !(contact.normal_force > 0.0)) {
        contact.normal_force = 0.0;
        return contact;
    }

    initialize_geometry(contact);
    const Vec3 displacement = weighted_relative_displacement(
            current_positions, previous_positions, contact.weights);
    contact.tangential_displacement = contact.projector * displacement;
    if (!contact.tangential_displacement.allFinite())
        throw std::runtime_error("friction contact: tangential displacement is not finite.");
    contact.active = true;
    return contact;
}

double friction_scale(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2) {
    validate_evaluation_parameters(friction_coefficient, dt2);
    if (!contact.active || friction_coefficient == 0.0 || dt2 == 0.0)
        return 0.0;
    validate_active_contact_scale_data(contact);

    const double slip = contact.tangential_displacement.norm();
    if (!std::isfinite(slip))
        throw std::runtime_error("frozen friction: slip is not finite.");
    const double scale = dt2 * friction_coefficient * contact.normal_force
            * smooth_friction_mollifier_over_slip_unchecked(
                    slip, contact.eps_u);
    if (!std::isfinite(scale))
        throw std::runtime_error("frozen friction: derivative scale is not finite.");
    return scale;
}

void validate_role(int role) {
    if (role < 0 || role >= 4)
        throw std::invalid_argument("frozen friction: role index must be in [0, 3].");
}

} // namespace

FrozenFrictionContact make_node_triangle_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        double d_hat, double k_barrier, double dt, double eps_v,
        double eps, const NodeTriangleDistanceResult* precomputed_dr) {
    validate_builder_parameters(d_hat, k_barrier, dt, eps_v, eps);

    const NodeTriangleDistanceResult dr = precomputed_dr
            ? *precomputed_dr
            : node_triangle_distance(
                    current_positions[0], current_positions[1],
                    current_positions[2], current_positions[3], eps);
    validate_distance(dr.distance);

    // Preserve the legacy friction-only behavior at zero stiffness: an exact
    // zero-distance contact remains inactive instead of asking the scalar
    // barrier for its singular derivative.
    if (k_barrier == 0.0) {
        FrozenFrictionContact contact;
        contact.eps_u = dt * eps_v;
        require_positive_finite(
                contact.eps_u,
                "friction contact: dt * eps_v must be positive and finite.");
        contact.weights = node_triangle_contact_weights(
                current_positions[0], current_positions[1],
                current_positions[2], current_positions[3], eps, &dr);
        return contact;
    }

    const NodeTriangleContactEvaluation evaluation =
            make_node_triangle_contact_evaluation(
                    current_positions, d_hat, k_barrier, eps, &dr);
    return make_node_triangle_frozen_friction_contact(
            current_positions, previous_positions, evaluation, dt, eps_v);
}

FrozenFrictionContact make_node_triangle_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const NodeTriangleContactEvaluation& evaluation,
        double dt, double eps_v) {
    return make_mesh_frozen_friction_contact(
            current_positions, previous_positions, evaluation, dt, eps_v,
            [&](FrozenFrictionContact& contact) {
                // The distance result is already shared with the normal
                // kernel. Form friction-only geometry once, directly in the
                // frozen contact. With a precomputed result, eps is unused.
                contact.weights = node_triangle_contact_weights(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        1.0e-12, &evaluation.dr);
                initialize_mesh_contact_tangent_frame(
                        contact,
                        current_positions[0] - evaluation.dr.closest_point);
            });
}

FrozenFrictionContact make_segment_segment_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        double d_hat, double k_barrier, double dt, double eps_v,
        double eps, const SegmentSegmentDistanceResult* precomputed_dr) {
    validate_builder_parameters(d_hat, k_barrier, dt, eps_v, eps);

    const SegmentSegmentDistanceResult dr = precomputed_dr
            ? *precomputed_dr
            : segment_segment_distance(
                    current_positions[0], current_positions[1],
                    current_positions[2], current_positions[3], eps);
    validate_distance(dr.distance);

    if (k_barrier == 0.0) {
        FrozenFrictionContact contact;
        contact.eps_u = dt * eps_v;
        require_positive_finite(
                contact.eps_u,
                "friction contact: dt * eps_v must be positive and finite.");
        contact.weights = segment_segment_contact_weights(
                current_positions[0], current_positions[1],
                current_positions[2], current_positions[3], eps, &dr);
        return contact;
    }

    const SegmentSegmentContactEvaluation evaluation =
            make_segment_segment_contact_evaluation(
                    current_positions, d_hat, k_barrier, eps, &dr);
    return make_segment_segment_frozen_friction_contact(
            current_positions, previous_positions, evaluation, dt, eps_v);
}

FrozenFrictionContact make_segment_segment_frozen_friction_contact(
        const std::array<Vec3, 4>& current_positions,
        const std::array<Vec3, 4>& previous_positions,
        const SegmentSegmentContactEvaluation& evaluation,
        double dt, double eps_v) {
    return make_mesh_frozen_friction_contact(
            current_positions, previous_positions, evaluation, dt, eps_v,
            [&](FrozenFrictionContact& contact) {
                contact.weights = segment_segment_contact_weights(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        1.0e-12, &evaluation.dr);
                initialize_mesh_contact_tangent_frame(
                        contact, evaluation.dr.closest_point_1
                                - evaluation.dr.closest_point_2);
            });
}

FrozenFrictionContact make_sdf_frozen_friction_contact(
        const Vec3& current_position,
        const Vec3& previous_position,
        const SDFEvaluation& sdf,
        double k_sdf, double eps_sdf, double dt, double eps_v,
        double eps, const Vec3* precomputed_penalty_gradient) {
    require_nonnegative_finite(
        k_sdf,
        "SDF friction contact: k_sdf must be nonnegative and finite.");
    if (!std::isfinite(eps_sdf)) {
        throw std::invalid_argument(
            "SDF friction contact: eps_sdf must be finite.");
    }
    require_positive_finite(
        dt, "SDF friction contact: dt must be positive and finite.");
    require_positive_finite(
        eps_v,
        "SDF friction contact: eps_v must be positive and finite.");
    require_positive_finite(
        eps, "SDF friction contact: eps must be positive and finite.");
    if (!current_position.allFinite() || !previous_position.allFinite()
        || !std::isfinite(sdf.phi) || !sdf.grad_phi.allFinite()
        || !sdf.surface_point.allFinite()) {
        throw std::invalid_argument(
            "SDF friction contact: positions and SDF data must be finite.");
    }

    FrozenFrictionContact contact;
    contact.weights[0] = 1.0;
    contact.eps_u = dt * eps_v;
    require_positive_finite(
        contact.eps_u,
        "SDF friction contact: dt * eps_v must be positive and finite.");
    if (k_sdf == 0.0)
        return contact;

    const double normal_norm = sdf.grad_phi.norm();
    if (!(normal_norm > eps) || !std::isfinite(normal_norm))
        return contact;
    contact.normal = sdf.grad_phi / normal_norm;

    // sdf_penalty_gradient is d psi / dx. Its component opposite the outward
    // SDF normal is the positive normal load that caps Coulomb friction.
    const Vec3 normal_gradient = precomputed_penalty_gradient
        ? *precomputed_penalty_gradient
        : sdf_penalty_gradient(sdf, k_sdf, eps_sdf);
    if (!normal_gradient.allFinite()) {
        throw std::invalid_argument(
            "SDF friction contact: penalty gradient must be finite.");
    }
    contact.normal_force = std::max(
        0.0, -normal_gradient.dot(contact.normal));
    if (!std::isfinite(contact.normal_force)) {
        throw std::runtime_error(
            "SDF friction contact: normal force is not finite.");
    }
    if (!(contact.normal_force > 0.0)) {
        contact.normal_force = 0.0;
        return contact;
    }

    contact.projector = Mat33::Identity()
        - contact.normal * contact.normal.transpose();
    const Vec3 surface_displacement = sdf.surface_point
        - sdf_previous_material_point(
            sdf.material_motion, sdf.surface_point);
    const Vec3 relative_displacement =
        (current_position - previous_position)
        - surface_displacement;
    contact.tangential_displacement =
        contact.projector * relative_displacement;
    if (!contact.tangential_displacement.allFinite()) {
        throw std::runtime_error(
            "SDF friction contact: tangential displacement is not finite.");
    }
    contact.active = true;
    return contact;
}

double smooth_friction_potential(double slip, double eps_u) {
    require_nonnegative_finite(
            slip, "smooth_friction_potential: slip must be nonnegative and finite.");
    require_positive_finite(
            eps_u, "smooth_friction_potential: eps_u must be positive and finite.");

    return smooth_friction_potential_unchecked(slip, eps_u);
}

double smooth_friction_mollifier_over_slip(double slip, double eps_u) {
    require_nonnegative_finite(
            slip,
            "smooth_friction_mollifier_over_slip: slip must be nonnegative and finite.");
    require_positive_finite(
            eps_u,
            "smooth_friction_mollifier_over_slip: eps_u must be positive and finite.");

    return smooth_friction_mollifier_over_slip_unchecked(slip, eps_u);
}

double frozen_friction_energy(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2) {
    validate_evaluation_parameters(friction_coefficient, dt2);
    if (!contact.active || friction_coefficient == 0.0 || dt2 == 0.0)
        return 0.0;
    validate_active_contact_scale_data(contact);

    const double slip = contact.tangential_displacement.norm();
    if (!std::isfinite(slip))
        throw std::runtime_error("frozen friction: slip is not finite.");
    const double energy = dt2 * friction_coefficient * contact.normal_force
            * smooth_friction_potential_unchecked(slip, contact.eps_u);
    if (!std::isfinite(energy))
        throw std::runtime_error("frozen friction: energy is not finite.");
    return energy;
}

Vec3 frozen_friction_relative_gradient(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2) {
    const double scale = friction_scale(contact, friction_coefficient, dt2);
    if (scale == 0.0)
        return Vec3::Zero();
    return scale * contact.tangential_displacement;
}

Mat33 frozen_friction_relative_hessian(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2) {
    const double scale = friction_scale(contact, friction_coefficient, dt2);
    if (scale == 0.0)
        return Mat33::Zero();
    validate_active_contact_projector(contact);
    return scale * contact.projector;
}

std::pair<Vec3, Mat33> frozen_friction_relative_gradient_and_hessian(
        const FrozenFrictionContact& contact,
        double friction_coefficient, double dt2) {
    const double scale = friction_scale(contact, friction_coefficient, dt2);
    if (scale == 0.0)
        return {Vec3::Zero(), Mat33::Zero()};
    validate_active_contact_projector(contact);
    return {
        scale * contact.tangential_displacement,
        scale * contact.projector
    };
}

Vec3 frozen_friction_role_gradient(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2) {
    validate_role(role);
    const Vec3 relative_gradient = frozen_friction_relative_gradient(
            contact, friction_coefficient, dt2);
    return contact.weights[role] * relative_gradient;
}

Mat33 frozen_friction_role_hessian(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2) {
    validate_role(role);
    const double weight = contact.weights[role];
    const Mat33 relative_hessian = frozen_friction_relative_hessian(
            contact, friction_coefficient, dt2);
    return weight * weight * relative_hessian;
}

std::pair<Vec3, Mat33> frozen_friction_role_gradient_and_hessian(
        const FrozenFrictionContact& contact, int role,
        double friction_coefficient, double dt2) {
    validate_role(role);
    const auto [relative_gradient, relative_hessian] =
        frozen_friction_relative_gradient_and_hessian(
            contact, friction_coefficient, dt2);
    const double weight = contact.weights[role];
    return {
        weight * relative_gradient,
        weight * weight * relative_hessian
    };
}

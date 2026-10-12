#include "rigid_bridge.h"

#include "friction_energy.h"
#include "parallel_helper.h"

#include <stdexcept>
#include <type_traits>

namespace ipc_gpu {
namespace {

void pack(const Vec3& source, double* target) {
    for (int k = 0; k < 3; ++k) target[k] = source[k];
}

void pack(const Mat33& source, double* target) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) target[3*r+c] = source(r,c);
}

Vec3 vector(const double* value) { return Vec3(value[0],value[1],value[2]); }

Mat33 matrix(const double* value) {
    Mat33 result;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) result(r,c) = value[3*r+c];
    return result;
}

unsigned flags_for(RigidDerivativeMode mode) {
    unsigned flags = 0;
    if (mode != RigidDerivativeMode::OrientationHessian) flags |= RigidTranslationGradient;
    if (mode != RigidDerivativeMode::TranslationHessian) flags |= RigidOrientationGradient;
    if (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::TranslationHessian) flags |= RigidTranslationHessian;
    if (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian) flags |= RigidOrientationHessian;
    if (mode == RigidDerivativeMode::Full) flags |= RigidMixedHessian;
    if (mode == RigidDerivativeMode::OrientationHessian) flags |= RigidUpperCrossBlocks;
    return flags;
}

template<class Evaluation>
void prepare(const ipc_simd::RigidContactInput& input, const Evaluation& evaluation,
    RigidDerivativeMode mode, double friction, double dt, double eps_v,
    RigidPreparedContact& output) {
    constexpr bool segment = std::is_same_v<Evaluation, SegmentSegmentContactEvaluation>;
    const auto& x = input.positions;
    FrozenFrictionContact frozen;
    if (friction != 0.0) {
        // Keep validation of dt/eps_v even for geometrically inactive contacts.
        if constexpr (segment)
            frozen = make_segment_segment_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
        else frozen = make_node_triangle_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
    }
    if (!evaluation.active) return;
    const int first = input.side == RigidBarrierSide::FirstPrimitive ? 0 : (segment ? 2 : 1);
    const int last = input.side == RigidBarrierSide::FirstPrimitive ? (segment ? 1 : 0) : 3;
    const bool og = output.flags & RigidOrientationGradient;
    const bool oh = output.flags & RigidOrientationHessian;
    if (og && (!input.kinematics || (oh && !input.kinematics->has_second_derivatives)))
        throw std::invalid_argument("GPU rigid preparation: requested orientation kinematics are missing");
    const auto gradient = [&](int i) -> Vec3 {
        if constexpr (segment) return segment_segment_barrier_gradient(x[0],x[1],x[2],x[3],i,evaluation);
        else return node_triangle_barrier_gradient(x[0],x[1],x[2],x[3],i,evaluation);
    };
    const auto hessian = [&](int i, int j) -> Mat33 {
        if constexpr (segment) return segment_segment_barrier_cross_hessian(x[0],x[1],x[2],x[3],i,j,evaluation);
        else return node_triangle_barrier_cross_hessian(x[0],x[1],x[2],x[3],i,j,evaluation);
    };
    for (int i = first; i <= last; ++i) {
        pack(gradient(i), output.gradients[i]);
        if (og) pack(dx_domega(input.body_references[i],*input.kinematics),output.jacobians[i]);
        if (oh) {
            const auto curvature = d2x_domega2(input.body_references[i],*input.kinematics);
            for (int axis = 0; axis < 3; ++axis) pack(curvature[axis],output.curvature[i][axis]);
        }
    }
    if (mode == RigidDerivativeMode::TranslationHessian) {
        if constexpr (segment) {
            pack(hessian(first,first),output.hessians[first][first]);
            const Mat33 cross = hessian(first,last);
            pack(cross,output.hessians[first][last]);
            pack(Mat33(cross.transpose()),output.hessians[last][first]);
            pack(hessian(last,last),output.hessians[last][last]);
        } else pack(hessian(0,0),output.translation_hessian);
    } else if (mode != RigidDerivativeMode::Gradient) {
        for (int i = first; i <= last; ++i)
            for (int j = mode == RigidDerivativeMode::OrientationHessian ? i : first; j <= last; ++j)
                pack(hessian(i,j),output.hessians[i][j]);
    }
    if (friction != 0.0 && frozen.active) {
        const unsigned mask = input.body_role_mask ? input.body_role_mask
            : ((1u << (last+1)) - (1u << first));
        for (int i = 0; i < 4; ++i) {
            if (!(mask & (1u << i))) continue;
            output.friction_weights[i] = frozen.weights[i];
            if (og && updates_rigid_orientation(input.update_mode) && (i < first || i > last))
                pack(dx_domega(input.body_references[i],*input.kinematics),output.jacobians[i]);
        }
        const auto derivatives = frozen_friction_relative_gradient_and_hessian(frozen,friction,dt*dt);
        pack(derivatives.first,output.friction_gradient);
        pack(derivatives.second,output.friction_hessian);
    }
}

RigidEnergyDerivatives unpack(const RigidDerivatives& value) {
    RigidEnergyDerivatives result;
    result.translation_gradient = vector(value.translation_gradient);
    result.orientation_gradient = vector(value.orientation_gradient);
    result.translation_translation_hessian = matrix(value.translation_translation_hessian);
    result.translation_orientation_hessian = matrix(value.translation_orientation_hessian);
    result.orientation_orientation_hessian = matrix(value.orientation_orientation_hessian);
    return result;
}

void add(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& value) {
    total.translation_gradient += value.translation_gradient;
    total.orientation_gradient += value.orientation_gradient;
    total.translation_translation_hessian += value.translation_translation_hessian;
    total.translation_orientation_hessian += value.translation_orientation_hessian;
    total.orientation_orientation_hessian += value.orientation_orientation_hessian;
}

} // namespace

void prepare_rigid_contacts(const ipc_simd::RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidPreparedContact* outputs) {
    if (count && (!inputs || !outputs)) throw std::invalid_argument("GPU rigid preparation: null input/output");
    for (std::size_t e = 0; e < count; ++e) {
        const auto& input = inputs[e];
        const auto& x = input.positions;
        auto& output = outputs[e];
        output = {};
        output.flags = flags_for(mode) | (friction != 0.0 ? unsigned(RigidFriction) : 0u);
        output.update_mask = (updates_rigid_translation(input.update_mode) ? 1u : 0u)
            | (updates_rigid_orientation(input.update_mode) ? 2u : 0u);
        output.segment_segment = input.segment_segment;
        if (input.segment_segment) {
            SegmentSegmentContactEvaluation evaluation;
            if (friction != 0.0)
                evaluation = make_segment_segment_contact_evaluation(x,d_hat,k_barrier,1e-12,nullptr,input.exact_computation_fallback);
            else {
                evaluation.dr = segment_segment_distance(x[0],x[1],x[2],x[3],1e-12,input.exact_computation_fallback);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance,d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance,d_hat);
                }
            }
            prepare(input,evaluation,mode,friction,dt,eps_v,output);
        } else {
            NodeTriangleContactEvaluation evaluation;
            if (friction != 0.0)
                evaluation = make_node_triangle_contact_evaluation(x,d_hat,k_barrier,1e-12,nullptr,input.exact_computation_fallback);
            else {
                evaluation.dr = node_triangle_distance(x[0],x[1],x[2],x[3]);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance,d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance,d_hat);
                }
            }
            prepare(input,evaluation,mode,friction,dt,eps_v,output);
        }
    }
}

ipc_simd::RigidContactOutput unpack_rigid(const RigidResult& value) {
    return {unpack(value.barrier),unpack(value.friction)};
}

ipc_simd::RigidContactOutput rigid_contact_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& cache, const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* kinematics, bool cooperative,
    const std::function<void()>* leader_work) {
    (void)cooperative;
    // This callback can itself launch GPU inertia. Execute before acquiring
    // the CUDA workspace mutex in compute_rigid, and exactly once per call.
    if (leader_work) (*leader_work)();
    ipc_simd::RigidContactOutput total;
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0) return total;
    const std::size_t nt_count = nt_pair_indices.size();
    const std::size_t count = nt_count + ss_pair_indices.size();
    if (!count) return total;
    QuaternionOmegaKinematics owned_kinematics;
    if (!kinematics && mode != RigidDerivativeMode::TranslationHessian) {
        owned_kinematics = quaternion_omega_kinematics(state.orientations[rb],omega_new[rb],dt,
            mode != RigidDerivativeMode::Gradient);
        kinematics = &owned_kinematics;
    }
    std::vector<ipc_simd::RigidContactInput> inputs;
    inputs.reserve(count);
    const double distance2 = params.d_hat * params.d_hat;
    for (std::size_t index = 0; index < count; ++index) {
        std::array<int,4> nodes;
        const bool segment = index >= nt_count;
        if (segment) {
            const auto& pair = cache.ss_pairs[ss_pair_indices[index-nt_count]];
            nodes = {pair.v[0],pair.v[1],pair.v[2],pair.v[3]};
        } else {
            const auto& pair = cache.nt_pairs[nt_pair_indices[index]];
            nodes = {pair.node,pair.tri_v[0],pair.tri_v[1],pair.tri_v[2]};
        }
        const int first_owner = owning_rb_for_node(ref_mesh.node_to_rb,nodes[0]);
        const int second_owner = owning_rb_for_node(ref_mesh.node_to_rb,nodes[segment ? 2 : 1]);
        if (first_owner == second_owner || (first_owner != rb && second_owner != rb)) continue;
        const bool nearby = segment
            ? segment_aabbs_within_distance(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]],distance2)
            : node_triangle_aabbs_within_distance(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]],distance2);
        if (!nearby) continue;
        ipc_simd::RigidContactInput input;
        input.segment_segment = segment;
        input.exact_computation_fallback = params.exact_computation_fallback;
        input.side = first_owner == rb ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
        input.kinematics = kinematics;
        input.update_mode = ref_mesh.rb_update_modes[rb];
        for (int role = 0; role < 4; ++role) {
            const int node = nodes[role];
            input.positions[role] = positions[node];
            input.body_references[role].setZero();
            if (owning_rb_for_node(ref_mesh.node_to_rb,node) == rb) {
                input.body_role_mask |= 1u << role;
                input.body_references[role] = ref_mesh.ref_positions[rb][node_to_rb_local[node]];
            }
            if (params.friction_coefficient != 0.0)
                input.previous_positions[role] = state.deformed_positions[node];
        }
        inputs.push_back(input);
    }
    std::vector<RigidPreparedContact> prepared(inputs.size());
    std::vector<RigidResult> results(inputs.size());
    prepare_rigid_contacts(inputs.data(),inputs.size(),params.d_hat,params.k_barrier,
        params.friction_coefficient,dt,params.friction_velocity_epsilon,mode,prepared.data());
    compute_rigid(prepared.data(),prepared.size(),results.data());
    // Match the solver's NT-then-SS accumulation without GPU atomics/reduction.
    for (const auto& result : results) {
        const auto value = unpack_rigid(result);
        add(total.barrier,value.barrier);
        if (params.friction_coefficient != 0.0) add(total.friction,value.friction);
    }
    return total;
}

std::pair<Vec3, Mat33> rigid_translation_inertia_gravity(
    const Vec3& x_com, const Vec3& x_com_n, const Vec3& v_com_n,
    double dt, double total_mass, double gravity) {
    RigidInertia input{};
    pack(x_com,input.position);
    pack(x_com_n,input.previous_position);
    pack(v_com_n,input.previous_velocity);
    input.dt = dt;
    input.mass = total_mass;
    input.gravity = gravity;
    Derivatives result;
    compute_rigid_inertia(&input,1,&result);
    return {vector(result.gradient),matrix(result.hessian)};
}

std::pair<Vec3, Mat33> rigid_rotation_inertia(
    const Vec3& omega, const Vec4& q_n, const Vec3& omega_n, double dt,
    const Mat33& I_hat, const QuaternionOmegaKinematics* kinematics,
    const Mat33* rotation_predictor) {
    QuaternionOmegaKinematics owned;
    if (!kinematics) {
        owned = quaternion_omega_kinematics(q_n,omega,dt,true);
        kinematics = &owned;
    }
    RigidInertia input{};
    input.rotational = 1;
    for (int a = 0; a < 4; ++a) {
        input.orientation[a] = kinematics->orientation[a];
        for (int b = 0; b < 3; ++b) input.orientation_jacobian[a][b] = kinematics->orientation_jacobian(a,b);
    }
    const auto curvature = kinematics->has_second_derivatives
        ? kinematics->orientation_hessians : d2q_domega2(q_n,omega,dt);
    for (int a = 0; a < 4; ++a) pack(curvature[a],input.orientation_hessians[a]);
    pack(I_hat,input.second_moment);
    pack(rotation_predictor ? *rotation_predictor : rigid_rotation_predictor(q_n,omega_n,dt),input.rotation_predictor);
    Derivatives result;
    compute_rigid_inertia(&input,1,&result);
    return {vector(result.gradient),matrix(result.hessian)};
}

} // namespace ipc_gpu

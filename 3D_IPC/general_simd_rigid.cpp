#include "general_simd_rigid.h"

#include "contact_scheduling.h"
#include "friction_energy.h"
#include "parallel_helper.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <type_traits>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace ipc_simd {
namespace {

// Every hardware lane belongs to an independent contact. These are explicit
// SIMD products across contacts, separate from Eigen's within-matrix SIMD.
struct Pack {
#if defined(__AVX512F__)
    using Native = __m512d;
    static constexpr int width = 8;
#elif defined(__AVX2__)
    using Native = __m256d;
    static constexpr int width = 4;
#elif defined(__aarch64__) || defined(_M_ARM64)
    using Native = float64x2_t;
    static constexpr int width = 2;
#elif defined(__SSE2__) || defined(_M_X64)
    using Native = __m128d;
    static constexpr int width = 2;
#else
    using Native = double;
    static constexpr int width = 1;
#endif
    Native value;
    Pack() : Pack(0.0) {}
    explicit Pack(double x) {
#if defined(__AVX512F__)
        value = _mm512_set1_pd(x);
#elif defined(__AVX2__)
        value = _mm256_set1_pd(x);
#elif defined(__aarch64__) || defined(_M_ARM64)
        value = vdupq_n_f64(x);
#elif defined(__SSE2__) || defined(_M_X64)
        value = _mm_set1_pd(x);
#else
        value = x;
#endif
    }
    static Pack raw(Native x) { Pack p; p.value = x; return p; }
    static Pack load(const double* x) {
#if defined(__AVX512F__)
        return raw(_mm512_loadu_pd(x));
#elif defined(__AVX2__)
        return raw(_mm256_loadu_pd(x));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vld1q_f64(x));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_loadu_pd(x));
#else
        return raw(*x);
#endif
    }
    void store(double* x) const {
#if defined(__AVX512F__)
        _mm512_storeu_pd(x, value);
#elif defined(__AVX2__)
        _mm256_storeu_pd(x, value);
#elif defined(__aarch64__) || defined(_M_ARM64)
        vst1q_f64(x, value);
#elif defined(__SSE2__) || defined(_M_X64)
        _mm_storeu_pd(x, value);
#else
        *x = value;
#endif
    }
    friend Pack operator+(Pack a, Pack b) {
#if defined(__AVX512F__)
        return raw(_mm512_add_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_add_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vaddq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_add_pd(a.value, b.value));
#else
        return raw(a.value + b.value);
#endif
    }
    friend Pack operator*(Pack a, Pack b) {
#if defined(__AVX512F__)
        return raw(_mm512_mul_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_mul_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vmulq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_mul_pd(a.value, b.value));
#else
        return raw(a.value * b.value);
#endif
    }
    Pack& operator+=(Pack b) { *this = *this + b; return *this; }
};

using Matrix = std::array<std::array<Pack, 3>, 3>;
using Vector = std::array<Pack, 3>;
using ReferencePacket = typename Eigen::internal::find_best_packet<double, 3>::type;
constexpr int reference_packet_width = Eigen::internal::unpacket_traits<ReferencePacket>::size;

Pack multiply_add(Pack a, Pack b, Pack c) {
#if defined(__AVX512F__)
    return Pack::raw(_mm512_fmadd_pd(a.value, b.value, c.value));
#elif defined(__AVX2__) && defined(__FMA__)
    return Pack::raw(_mm256_fmadd_pd(a.value, b.value, c.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Pack::raw(vfmaq_f64(c.value, a.value, b.value));
#else
    return a * b + c;
#endif
}

Pack select(Pack condition, Pack yes, Pack no) {
#if defined(__AVX512F__)
    return Pack::raw(_mm512_mask_blend_pd(
        _mm512_cmp_pd_mask(condition.value, _mm512_setzero_pd(), _CMP_NEQ_OQ),
        no.value, yes.value));
#elif defined(__AVX2__)
    return Pack::raw(_mm256_blendv_pd(no.value, yes.value,
        _mm256_cmp_pd(condition.value, _mm256_setzero_pd(), _CMP_NEQ_OQ)));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Pack::raw(vbslq_f64(vcgtq_f64(condition.value, vdupq_n_f64(0.0)),
        yes.value, no.value));
#elif defined(__SSE2__) || defined(_M_X64)
    const auto mask = _mm_cmpneq_pd(condition.value, _mm_setzero_pd());
    return Pack::raw(_mm_or_pd(_mm_and_pd(mask, yes.value), _mm_andnot_pd(mask, no.value)));
#else
    return condition.value != 0.0 ? yes : no;
#endif
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if constexpr (reference_packet_width > 1)
                c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j])
                    + a[i][2] * b[2][j];
            else
                c[i][j] = a[i][0] * b[0][j]
                    + (a[i][1] * b[1][j] + a[i][2] * b[2][j]);
        }
    return c;
}

Matrix multiply_column_major(const Matrix& a, const Matrix& b) {
    Matrix c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // Eigen's fixed 3x3 product evaluates its first two rows with
            // packet multiply-add, and its final scalar row with the balanced
            // three-term reduction. SIMD lanes here remain separate contacts.
            if (reference_packet_width > 1 && i < 2)
                c[i][j] = multiply_add(a[i][2], b[2][j],
                    multiply_add(a[i][1], b[1][j], a[i][0] * b[0][j]));
            else
                c[i][j] = a[i][0] * b[0][j]
                    + (a[i][1] * b[1][j] + a[i][2] * b[2][j]);
        }
    return c;
}

Matrix transpose(const Matrix& a) {
    Matrix b;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) b[i][j] = a[j][i];
    return b;
}

void add(Matrix& a, const Matrix& b) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i][j] += b[i][j];
}

void symmetrize(Matrix& a) {
    // Match the reference's in-place, column-major Eigen assignment. Using
    // a snapshot would give a slightly different result for roundoff-sized
    // asymmetry in the input matrix.
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            a[i][j] = Pack(0.5) * (a[i][j] + a[j][i]);
}

struct Modes {
    bool tg, og, th, oh, mh;
    explicit Modes(RigidDerivativeMode mode)
        : tg(mode != RigidDerivativeMode::OrientationHessian),
          og(mode != RigidDerivativeMode::TranslationHessian),
          th(mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::TranslationHessian),
          oh(mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian),
          mh(mode == RigidDerivativeMode::Full) {}
};

struct PreparedContact {
    std::array<Vec3, 4> gradients;
    std::array<Mat33, 4> jacobians;
    std::array<std::array<Mat33, 4>, 4> hessians;
    std::array<std::array<Mat33, 3>, 4> curvature;
    std::array<double, 4> friction_weights;
    Mat33 translation_hessian;
    void clear(const Modes& flags) {
        friction_weights.fill(0.0);
        translation_hessian.setZero();
        for (int i = 0; i < 4; ++i) {
            gradients[i].setZero();
            if (flags.og) jacobians[i].setZero();
            if (flags.th || flags.oh)
                for (auto& h : hessians[i]) h.setZero();
            if (flags.oh)
                for (auto& h : curvature[i]) h.setZero();
        }
    }
};

template <class Evaluation>
void prepare_contact(const RigidContactInput& input, const Evaluation& evaluation,
    RigidDerivativeMode mode, PreparedContact& prepared,
    FrozenFrictionContact* frozen, double dt, double eps_v) {
    constexpr bool segment = std::is_same_v<Evaluation, SegmentSegmentContactEvaluation>;
    const auto& x = input.positions;
    // The scalar builder validates dt/eps_v even for a geometrically inactive
    // admitted contact, but ignores its previous positions in that case.
    if (frozen) {
        if constexpr (segment)
            *frozen = make_segment_segment_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
        else
            *frozen = make_node_triangle_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
    }
    if (!evaluation.active) return;
    const int first = input.side == RigidBarrierSide::FirstPrimitive ? 0 : (segment ? 2 : 1);
    const int last = input.side == RigidBarrierSide::FirstPrimitive ? (segment ? 1 : 0) : 3;
    const Modes flags(mode);
    const auto gradient = [&](int i) {
        if constexpr (segment)
            return segment_segment_barrier_gradient(x[0], x[1], x[2], x[3], i, evaluation);
        else
            return node_triangle_barrier_gradient(x[0], x[1], x[2], x[3], i, evaluation);
    };
    const auto hessian = [&](int i, int j) {
        if constexpr (segment)
            return segment_segment_barrier_cross_hessian(x[0], x[1], x[2], x[3], i, j, evaluation);
        else
            return node_triangle_barrier_cross_hessian(x[0], x[1], x[2], x[3], i, j, evaluation);
    };
    if (flags.og && (!input.kinematics || (flags.oh && !input.kinematics->has_second_derivatives)))
        throw std::invalid_argument("SIMD rigid contact: requested orientation kinematics are missing.");
    for (int i = first; i <= last; ++i) {
        prepared.gradients[i] = gradient(i);
        if (flags.og)
            prepared.jacobians[i] = dx_domega(input.body_references[i], *input.kinematics);
        if (flags.oh)
            prepared.curvature[i] = d2x_domega2(input.body_references[i], *input.kinematics);
    }
    if (mode == RigidDerivativeMode::TranslationHessian) {
        // These are the scalar solver's exact COM shortcuts. In particular,
        // translating all three triangle vertices gives the point self block.
        if constexpr (segment) {
            prepared.hessians[first][first] = hessian(first, first);
            prepared.hessians[first][last] = hessian(first, last);
            prepared.hessians[last][first] = prepared.hessians[first][last].transpose();
            prepared.hessians[last][last] = hessian(last, last);
        } else prepared.translation_hessian = hessian(0, 0);
    } else if (mode != RigidDerivativeMode::Gradient) {
        for (int i = first; i <= last; ++i)
            for (int j = mode == RigidDerivativeMode::OrientationHessian ? i : first; j <= last; ++j)
                prepared.hessians[i][j] = hessian(i, j);
    }
    if (frozen) {
        if (frozen->active) {
            const unsigned mask = input.body_role_mask ? input.body_role_mask
                : ((1u << (last + 1)) - (1u << first));
            for (int i = 0; i < 4; ++i) {
                if (!(mask & (1u << i))) continue;
                prepared.friction_weights[i] = frozen->weights[i];
                if (flags.og && updates_rigid_orientation(input.update_mode)
                    && (i < first || i > last))
                    prepared.jacobians[i] = dx_domega(input.body_references[i], *input.kinematics);
            }
        }
        // Existing SIMD friction then computes the common relative gradient
        // and Hessian. Signed body weights are pulled back only once below.
        frozen->weights[0] = 1.0;
    }
}

template <class Getter>
Pack gather(int active, const Getter& get) {
    alignas(64) double values[Pack::width];
    for (int lane = 0; lane < Pack::width; ++lane)
        values[lane] = get(std::min(lane, active - 1));
    return Pack::load(values);
}

void add_derivatives(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& value) {
    total.translation_gradient += value.translation_gradient;
    total.orientation_gradient += value.orientation_gradient;
    total.translation_translation_hessian += value.translation_translation_hessian;
    total.translation_orientation_hessian += value.translation_orientation_hessian;
    total.orientation_orientation_hessian += value.orientation_orientation_hessian;
}

} // namespace

void rigid_contact_derivatives_tile(const RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidContactOutput* outputs) {
    assert(count <= contact_tile_width);
    if (count == 0) return;
    std::array<PreparedContact, contact_tile_width> prepared;
    std::array<FrozenFrictionContact, contact_tile_width> frozen;
    std::array<Vec3, contact_tile_width> friction_gradient;
    std::array<Mat33, contact_tile_width> friction_hessian;
    const Modes flags(mode);
    for (std::size_t e = 0; e < count; ++e) {
        // Tail packets and gradient/COM-only requests never read the unused
        // records or orientation-Hessian fields. Initialize only live work.
        prepared[e].clear(flags);
        outputs[e] = RigidContactOutput{};
        const auto& input = inputs[e];
        const auto& x = input.positions;
        auto* contact = friction != 0.0 ? &frozen[e] : nullptr;
        if (input.segment_segment) {
            SegmentSegmentContactEvaluation evaluation;
            if (contact) evaluation = make_segment_segment_contact_evaluation(x, d_hat, k_barrier);
            else {
                evaluation.dr = segment_segment_distance(x[0], x[1], x[2], x[3]);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance, d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance, d_hat);
                }
            }
            prepare_contact(input, evaluation, mode, prepared[e], contact, dt, eps_v);
        } else {
            NodeTriangleContactEvaluation evaluation;
            if (contact) evaluation = make_node_triangle_contact_evaluation(x, d_hat, k_barrier);
            else {
                evaluation.dr = node_triangle_distance(x[0], x[1], x[2], x[3]);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance, d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance, d_hat);
                }
            }
            prepare_contact(input, evaluation, mode, prepared[e], contact, dt, eps_v);
        }
    }
    if (friction != 0.0) {
        std::array<int, contact_tile_width> roles{};
        friction_derivatives_tile(frozen.data(), roles.data(), count,
            friction, dt * dt, friction_gradient.data(), friction_hessian.data());
    }

    for (std::size_t begin = 0; begin < count; begin += Pack::width) {
        const int active = static_cast<int>(std::min<std::size_t>(Pack::width, count - begin));
        const auto scalar = [&](const auto& get) {
            return gather(active, [&](int lane) { return get(prepared[begin + lane]); });
        };
        const auto matrix = [&](const auto& get) {
            Matrix result;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    result[r][c] = scalar([&](const auto& p) { return get(p)(r, c); });
            return result;
        };
        const auto store_vector = [&](const Vector& value, const auto& get) {
            for (int i = 0; i < 3; ++i) {
                alignas(64) double lanes[Pack::width];
                value[i].store(lanes);
                for (int lane = 0; lane < active; ++lane) get(outputs[begin + lane])[i] = lanes[lane];
            }
        };
        const auto store_matrix = [&](const Matrix& value, const auto& get) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    alignas(64) double lanes[Pack::width];
                    value[r][c].store(lanes);
                    for (int lane = 0; lane < active; ++lane) get(outputs[begin + lane])(r, c) = lanes[lane];
                }
        };
        std::array<Matrix, 4> J;
        for (int i = 0; i < 4; ++i)
            if (flags.og) J[i] = matrix([&](const auto& p) -> const Mat33& { return p.jacobians[i]; });
        Vector tg, og;
        Matrix tt = matrix([](const auto& p) -> const Mat33& { return p.translation_hessian; }), to, oo;
        for (int i = 0; i < 4; ++i) {
            Vector g;
            for (int r = 0; r < 3; ++r) g[r] = scalar([&](const auto& p) { return p.gradients[i][r]; });
            for (int r = 0; r < 3; ++r) {
                if (flags.tg) tg[r] += g[r];
                if (flags.og) og[r] += (J[i][0][r] * g[0] + J[i][1][r] * g[1]) + J[i][2][r] * g[2];
            }
            if (mode != RigidDerivativeMode::Gradient) {
                for (int j = mode == RigidDerivativeMode::OrientationHessian ? i : 0; j < 4; ++j) {
                    const Matrix H = matrix([&](const auto& p) -> const Mat33& { return p.hessians[i][j]; });
                    if (flags.th) add(tt, H);
                    if (flags.mh) add(to, multiply_column_major(H, J[j]));
                    if (flags.oh) {
                        const Matrix contribution = multiply_column_major(multiply(transpose(J[i]), H), J[j]);
                        if (mode == RigidDerivativeMode::OrientationHessian && i != j) {
                            Matrix pair = contribution;
                            add(pair, transpose(contribution));
                            add(oo, pair);
                        } else add(oo, contribution);
                    }
                }
            }
            if (flags.oh)
                for (int axis = 0; axis < 3; ++axis) {
                    const Matrix curvature = matrix([&](const auto& p) -> const Mat33& { return p.curvature[i][axis]; });
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) oo[r][c] += g[axis] * curvature[r][c];
                }
        }
        if (mode == RigidDerivativeMode::TranslationHessian) {
            const Pack segment = gather(active, [&](int lane) {
                return inputs[begin + lane].segment_segment ? 1.0 : 0.0;
            });
            const Matrix point_block = matrix([](const auto& p) -> const Mat33& {
                return p.translation_hessian;
            });
            // NT translation uses its self block directly in the reference.
            // Adding the unused zero cross blocks would erase negative zeros.
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    tt[r][c] = select(segment, tt[r][c], point_block[r][c]);
        }
        if (flags.th) symmetrize(tt);
        if (flags.oh) symmetrize(oo);
        store_vector(tg, [](auto& o) -> Vec3& { return o.barrier.translation_gradient; });
        store_vector(og, [](auto& o) -> Vec3& { return o.barrier.orientation_gradient; });
        store_matrix(tt, [](auto& o) -> Mat33& { return o.barrier.translation_translation_hessian; });
        store_matrix(to, [](auto& o) -> Mat33& { return o.barrier.translation_orientation_hessian; });
        store_matrix(oo, [](auto& o) -> Mat33& { return o.barrier.orientation_orientation_hessian; });

        if (friction == 0.0) continue;
        Pack weight;
        Matrix FJ;
        for (int i = 0; i < 4; ++i) {
            const Pack w = scalar([&](const auto& p) { return p.friction_weights[i]; });
            weight += w;
            if (flags.og)
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) FJ[r][c] += w * J[i][r][c];
        }
        Vector fg, ftg, fog;
        Matrix fH, ftt, fto, foo;
        for (int r = 0; r < 3; ++r) {
            fg[r] = gather(active, [&](int lane) { return friction_gradient[begin + lane][r]; });
            for (int c = 0; c < 3; ++c)
                fH[r][c] = gather(active, [&](int lane) { return friction_hessian[begin + lane](r, c); });
        }
        const Pack translation = gather(active, [&](int lane) { return updates_rigid_translation(inputs[begin + lane].update_mode) ? 1.0 : 0.0; });
        const Pack orientation = gather(active, [&](int lane) { return updates_rigid_orientation(inputs[begin + lane].update_mode) ? 1.0 : 0.0; });
        const Matrix fHj = multiply_column_major(fH, FJ);
        if (flags.oh) foo = multiply(transpose(FJ), fHj);
        for (int r = 0; r < 3; ++r) {
            if (flags.tg) ftg[r] = translation * (weight * fg[r]);
            if (flags.og) fog[r] = orientation * ((FJ[0][r] * fg[0] + FJ[1][r] * fg[1]) + FJ[2][r] * fg[2]);
            for (int c = 0; c < 3; ++c) {
                if (flags.th) ftt[r][c] = translation * ((weight * fH[r][c]) * weight);
                if (flags.mh) fto[r][c] = (translation * orientation) * (weight * fHj[r][c]);
                if (flags.oh) foo[r][c] = orientation * foo[r][c];
            }
        }
        // Frozen friction intentionally has no quaternion-curvature term.
        store_vector(ftg, [](auto& o) -> Vec3& { return o.friction.translation_gradient; });
        store_vector(fog, [](auto& o) -> Vec3& { return o.friction.orientation_gradient; });
        store_matrix(ftt, [](auto& o) -> Mat33& { return o.friction.translation_translation_hessian; });
        store_matrix(fto, [](auto& o) -> Mat33& { return o.friction.translation_orientation_hessian; });
        store_matrix(foo, [](auto& o) -> Mat33& { return o.friction.orientation_orientation_hessian; });
    }
}

RigidContactOutput rigid_contact_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& cache,
    const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices,
    const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* kinematics, bool cooperative,
    const std::function<void()>* leader_work) {
    RigidContactOutput total;
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0) {
        if (leader_work) (*leader_work)();
        return total;
    }
    const std::size_t nt_count = nt_pair_indices.size();
    const std::size_t count = nt_count + ss_pair_indices.size();
    QuaternionOmegaKinematics owned_kinematics;
    if (!kinematics && count != 0 && mode != RigidDerivativeMode::TranslationHessian) {
        owned_kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt,
            mode != RigidDerivativeMode::Gradient);
        kinematics = &owned_kinematics;
    }
    const double distance2 = params.d_hat * params.d_hat;
    const auto gather_contact = [&](std::size_t index, RigidContactInput& input) {
        std::array<int, 4> nodes;
        const bool segment = index >= nt_count;
        if (segment) {
            const auto& pair = cache.ss_pairs[ss_pair_indices[index - nt_count]];
            nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        } else {
            const auto& pair = cache.nt_pairs[nt_pair_indices[index]];
            nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        }
        const int first_owner = owning_rb_for_node(ref_mesh.node_to_rb, nodes[0]);
        const int second_owner = owning_rb_for_node(ref_mesh.node_to_rb, nodes[segment ? 2 : 1]);
        if (first_owner == second_owner || (first_owner != rb && second_owner != rb)) return false;
        const bool nearby = segment
            ? segment_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]], positions[nodes[3]], distance2)
            : node_triangle_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]], positions[nodes[3]], distance2);
        if (!nearby) return false;
        input.segment_segment = segment;
        input.side = first_owner == rb ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
        input.kinematics = kinematics;
        input.update_mode = ref_mesh.rb_update_modes[rb];
        input.body_role_mask = 0;
        for (int role = 0; role < 4; ++role) {
            const int node = nodes[role];
            input.positions[role] = positions[node];
            input.body_references[role].setZero();
            if (owning_rb_for_node(ref_mesh.node_to_rb, node) == rb) {
                input.body_role_mask |= 1u << role;
                input.body_references[role] = ref_mesh.ref_positions[rb][node_to_rb_local[node]];
            }
            if (params.friction_coefficient != 0.0)
                input.previous_positions[role] = state.deformed_positions[node];
        }
        return true;
    };
    const auto accumulate = [&](const RigidContactOutput& value) {
        add_derivatives(total.barrier, value.barrier);
        if (params.friction_coefficient != 0.0) add_derivatives(total.friction, value.friction);
    };
    const auto evaluate = [&](std::size_t begin, std::size_t end, const auto& emit) {
        std::array<RigidContactInput, contact_tile_width> inputs;
        std::array<RigidContactOutput, contact_tile_width> outputs;
        std::array<std::size_t, contact_tile_width> indices;
        std::size_t active = 0;
        const auto flush = [&] {
            if (active == 0) return;
            rigid_contact_derivatives_tile(inputs.data(), active, params.d_hat,
                params.k_barrier, params.friction_coefficient, dt,
                params.friction_velocity_epsilon, mode, outputs.data());
            for (std::size_t e = 0; e < active; ++e) emit(indices[e], outputs[e]);
            active = 0;
        };
        for (std::size_t index = begin; index < end; ++index) {
            if (!gather_contact(index, inputs[active])) continue;
            indices[active++] = index;
            if (active == contact_tile_width) flush();
        }
        flush();
    };
    if (cooperative && count >= 32 && omp_get_num_threads() > 1) {
        using Value = std::optional<RigidContactOutput>;
        std::vector<Value, solver_detail::CacheAlignedAllocator<Value>> values(count);
        solver_detail::evaluate_contact_ranges(static_cast<int>(count), [&](int begin, int end) {
            evaluate(begin, end, [&](std::size_t index, const auto& value) { values[index] = value; });
        }, static_cast<int>(contact_tile_width), leader_work);
        for (const auto& value : values) if (value) accumulate(*value);
    } else {
        if (leader_work) (*leader_work)();
        evaluate(0, count, [&](std::size_t, const auto& value) { accumulate(value); });
    }
    return total;
}

} // namespace ipc_simd

#pragma once

#include "resident_contact_exact.cuh"

#if defined(__CUDACC__)
#define IPC_CONTACT_COOP_HD __host__ __device__
#else
#define IPC_CONTACT_COOP_HD
#endif

namespace ipc_gpu::resident::contact_cooperative_detail {

constexpr int capacity = 256;
constexpr int active_lanes = 9;
constexpr int owner_slots = 512;
constexpr int helper_slots = 128;
constexpr int slots_per_worker = owner_slots + (active_lanes - 1) * helper_slots;

IPC_CONTACT_COOP_HD inline int self_block(unsigned mask) {
    for (int i = 0; i < 4; ++i)
        if (mask == (1u << (5 * i))) return i;
    return -1;
}

// POD only: publishing this does not copy ownership of any Integer. The
// pointed-to limbs stay immutable until all readers have completed.
struct IntegerView {
    const std::uint32_t* limb;
    int used;
    int sign;
    bool valid;
};

template<int C> IPC_CONTACT_COOP_HD inline IntegerView view(
    const resident_exact::Integer<C>& value) {
    return {value.limb, value.used, value.sign, value.valid};
}

// Arithmetic accepts const Integer references. This adapter never modifies
// the borrowed limbs and never adopts their arena slot. On CUDA its ordinary
// Integer constructor reserves a PRIVATE dummy slot; the destructor returns
// that private slot, not the source slot. The limb pointer alone is borrowed.
// Copy/move are forbidden so ownership cannot escape or change accidentally.
template<int C> class ReadOnlyInteger {
    resident_exact::Integer<C> value_;
public:
    IPC_CONTACT_COOP_HD explicit ReadOnlyInteger(IntegerView source) {
        value_.valid = value_.valid && source.valid;
        if (!value_.valid) { value_.used = 0; value_.sign = 0; return; }
#if defined(__CUDA_ARCH__)
        value_.limb = const_cast<std::uint32_t*>(source.limb);
#else
        for (int i = 0; i < source.used; ++i) value_.limb[i] = source.limb[i];
#endif
        value_.used = source.used;
        value_.sign = source.sign;
    }
    ReadOnlyInteger(const ReadOnlyInteger&) = delete;
    ReadOnlyInteger& operator=(const ReadOnlyInteger&) = delete;
    IPC_CONTACT_COOP_HD const resident_exact::Integer<C>& get() const { return value_; }
};

struct SharedGeometry {
    IntegerView residual[3];
    IntegerView squared_residual, normal_den, curvature_den, z, ww;
    IntegerView gram[2][2], derivatives[3][2];
    double distance, bp, bpp;
    int free_parameters;
};

// Immutable feature limbs plus one disjoint result per conversion lane. The
// owner must stay alive across both publication and completion barriers.
struct SharedConversions {
    IntegerView numerator[7], denominator;
    int unit;
    double values[7];
};

struct EntryResult { double value; bool ok; };

template<int C> IPC_CONTACT_COOP_HD inline EntryResult convert_geometry_component(
    const SharedConversions& shared, int component) {
    const ReadOnlyInteger<C> numerator(shared.numerator[component]), denominator(shared.denominator);
    bool okay = true;
    const double value = resident_exact::ratio_to_double(numerator.get(), denominator.get(),
        component < 3 ? shared.unit : 0, &okay);
    return {value, okay};
}

template<int C> struct GeometryOwner {
    using I = resident_exact::Integer<C>;
    resident_exact::Points<C> points;
    contact_exact_detail::Feature<C> feature;
    I gram[2][2], z, normal_den, curvature_den, ww;
    I derivatives[3][2];
    double residual[3];
    double bp = 0, bpp = 0;
    int unit = 0;
    bool arithmetic_ok = true;
    IPC_CONTACT_COOP_HD GeometryOwner() : z(1) {}
};

// Same exact expressions and conversion/FP order as evaluate_contact_exact.
// Construction/lifetime differs, so callers must replay the original scalar
// evaluator for ANY unsuccessful cooperative preparation or helper operation.
template<int C> IPC_CONTACT_COOP_HD inline bool prepare_feature(
    const ContactInput& in, GeometryOwner<C>& owner, ContactOutput& out) {
    resident_exact::Arithmetic<C> a;
    out = ContactOutput{};
    int unit = 0; bool have_unit = false;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        const auto value = resident_exact::decompose(in.positions[i][k]);
        if (!value.valid || !contact_detail::finite(in.previous[i][k])) {
            out.status = ContactStatus::InvalidInput; return false;
        }
        if (value.sign && (!have_unit || value.exponent < unit)) {
            unit = value.exponent; have_unit = true;
        }
    }
    if (!contact_detail::finite(in.d_hat) || !(in.d_hat > 0)
        || !contact_detail::finite(in.k_barrier) || in.k_barrier < 0
        || !contact_detail::finite(in.dt) || !(in.dt > 0)
        || !contact_detail::finite(in.friction) || in.friction < 0
        || !contact_detail::finite(in.eps_v) || !(in.eps_v > 0)
        || !contact_detail::finite(in.eps) || !(in.eps > 0)
        || (in.kind != ContactKind::PointTriangle && in.kind != ContactKind::EdgeEdge)
        || (in.hessian_blocks & ~0xffffu)) {
        out.status = ContactStatus::InvalidInput; return false;
    }
    if (in.k_barrier == 0) return false;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        owner.points.p[i].v[k] = resident_exact::from_double<C>(in.positions[i][k], unit);
        a.ok = a.ok && owner.points.p[i].v[k].valid;
    }
    if (!a.ok) { out.status = ContactStatus::ExactCapacityExceeded; return false; }
    owner.feature = in.kind == ContactKind::PointTriangle
        ? contact_exact_detail::point_triangle(a, owner.points)
        : contact_exact_detail::edge_edge(a, owner.points);
    const auto& f = owner.feature;
    if (!a.ok || !f.have) { out.status = ContactStatus::ExactCapacityExceeded; return false; }
    if (!a.sign(f.squared_residual)) { out.status = ContactStatus::ZeroDistance; return false; }
    owner.unit = unit;
    return true;
}

template<int C> IPC_CONTACT_COOP_HD inline void publish_conversions(
    const GeometryOwner<C>& owner, SharedConversions& shared) {
    for (int k = 0; k < 3; ++k) shared.numerator[k] = view(owner.feature.residual[k]);
    for (int i = 0; i < 4; ++i) shared.numerator[3+i] = view(owner.feature.weight[i]);
    shared.denominator = view(owner.feature.denominator);
    shared.unit = owner.unit;
}

// Called only after all seven conversions succeed, in the same floating-point
// order as the original serial preparation. No helper accesses owner storage.
template<int C> IPC_CONTACT_COOP_HD inline bool prepare_derivatives(
    const ContactInput& in, int role, GeometryOwner<C>& owner,
    ContactOutput& out, SharedGeometry& shared) {
    using I = resident_exact::Integer<C>;
    resident_exact::Arithmetic<C> a;
    const auto& f = owner.feature;
    out.distance = contact_exact_detail::stable_norm(owner.residual);
    if (!(out.distance > 0) || !contact_detail::finite(out.distance)) {
        out.status = ContactStatus::NonFiniteDerivative; return false;
    }
    for (int k = 0; k < 3; ++k) out.normal[k] = owner.residual[k] / out.distance;
    if (out.distance >= in.d_hat) return false;
    const double distance = out.distance, s = distance - in.d_hat;
    const double logratio = log(distance / in.d_hat), ratio = in.d_hat / distance;
    owner.bp = -2*s*logratio-s*s/distance;
    owner.bpp = ratio*ratio+2*ratio-3-2*logratio;
    const int n = f.free_parameters;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        owner.gram[i][j] = resident_exact::dot(a, f.tangent[i], f.tangent[j]);
    if (n == 1) owner.z = owner.gram[0][0];
    if (n == 2) owner.z = a.sub(a.mul(owner.gram[0][0], owner.gram[1][1]),
        a.mul(owner.gram[0][1], owner.gram[1][0]));
    if (!a.ok) { out.status = ContactStatus::ExactCapacityExceeded; return false; }
    if (a.sign(owner.z) <= 0) { out.status = ContactStatus::NonFiniteDerivative; return false; }
    const I p2 = a.mul(f.denominator, f.denominator);
    owner.normal_den = a.mul(p2, f.squared_residual);
    owner.curvature_den = a.mul(owner.normal_den, owner.z);
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        out.barrier_gradient[i][k] = (owner.bp*out.weights[i])*out.normal[k];
        if (i == role) for (int v = 0; v < n; ++v)
            owner.derivatives[k][v] = a.sum(a.mul(f.weight[i], f.tangent[v].v[k]),
                a.mul(I(f.sigma[v][i]), f.residual[k]));
    }
    owner.ww = a.mul(f.weight[role], f.weight[role]);
    owner.arithmetic_ok = a.ok;
    if (!a.ok) { out.status = ContactStatus::ExactCapacityExceeded; return false; }
    for (int k = 0; k < 3; ++k) {
        shared.residual[k] = view(f.residual[k]);
        for (int v = 0; v < 2; ++v) shared.derivatives[k][v] = view(owner.derivatives[k][v]);
    }
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        shared.gram[i][j] = view(owner.gram[i][j]);
    shared.squared_residual = view(f.squared_residual);
    shared.normal_den = view(owner.normal_den);
    shared.curvature_den = view(owner.curvature_den);
    shared.z = view(owner.z); shared.ww = view(owner.ww);
    shared.distance = distance; shared.bp = owner.bp; shared.bpp = owner.bpp;
    shared.free_parameters = n;
    return true;
}

// Serial preparation remains the default and the portable reference. Splitting
// it into stages does not change its conversion calls or arithmetic order.
template<int C> IPC_CONTACT_COOP_HD inline bool prepare_geometry(
    const ContactInput& in, int role, GeometryOwner<C>& owner,
    ContactOutput& out, SharedGeometry& shared) {
    if (!prepare_feature(in, owner, out)) return false;
    const auto& f = owner.feature;
    bool okay = true;
    for (int k = 0; k < 3; ++k)
        owner.residual[k] = resident_exact::ratio_to_double(f.residual[k], f.denominator, owner.unit, &okay);
    for (int i = 0; i < 4; ++i)
        out.weights[i] = resident_exact::ratio_to_double(f.weight[i], f.denominator, 0, &okay);
    if (!okay) { out.status = ContactStatus::ExactCapacityExceeded; return false; }
    return prepare_derivatives(in, role, owner, out, shared);
}

template<int C> IPC_CONTACT_COOP_HD inline EntryResult compute_entry(
    const SharedGeometry& shared, int entry) {
    using I = resident_exact::Integer<C>;
    const int k = entry / 3, l = entry % 3, n = shared.free_parameters;
    const ReadOnlyInteger<C> ww(shared.ww), z(shared.z), rk(shared.residual[k]), rl(shared.residual[l]);
    const ReadOnlyInteger<C> squared(shared.squared_residual), normal_den(shared.normal_den), curvature_den(shared.curvature_den);
    const ReadOnlyInteger<C> vi0(shared.derivatives[k][0]), vi1(shared.derivatives[k][1]);
    const ReadOnlyInteger<C> vj0(shared.derivatives[l][0]), vj1(shared.derivatives[l][1]);
    const ReadOnlyInteger<C> g00(shared.gram[0][0]), g01(shared.gram[0][1]), g11(shared.gram[1][1]);
    resident_exact::Arithmetic<C> a;
    const I normal_num = a.mul(a.mul(ww.get(), rk.get()), rl.get());
    I schur;
    if (n == 1) schur = a.mul(vi0.get(), vj0.get());
    if (n == 2) schur = a.sub(
        a.sum(a.mul(g11.get(), a.mul(vi0.get(), vj0.get())), a.mul(g00.get(), a.mul(vi1.get(), vj1.get()))),
        a.mul(g01.get(), a.sum(a.mul(vi0.get(), vj1.get()), a.mul(vi1.get(), vj0.get()))));
    const I hq_num = a.sub(k == l ? a.mul(ww.get(), z.get()) : I(0), schur);
    const I curvature_num = a.sub(a.mul(hq_num, squared.get()), a.mul(normal_num, z.get()));
    const double normal_term = resident_exact::ratio_to_double(normal_num, normal_den.get(), 0, &a.ok);
    const double curvature = resident_exact::ratio_to_double(curvature_num, curvature_den.get(), 0, &a.ok);
    return {shared.bpp*normal_term+(shared.bp/shared.distance)*curvature, a.ok};
}

// Original friction and finite checks, after every independent entry exists.
template<int C> IPC_CONTACT_COOP_HD inline void finish_output(
    const ContactInput& in, const GeometryOwner<C>& owner, ContactOutput& out) {
    if (in.friction > 0) {
        double displacement[3]{}, P[9], u[3]{};
        for (int k = 0; k < 3; ++k) for (int i = 0; i < 4; ++i)
            displacement[k] += out.weights[i]*(in.positions[i][k]-in.previous[i][k]);
        const double normal_length = contact_exact_detail::stable_norm(owner.residual);
        for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
            P[3*k+l] = (k == l ? 1.0 : 0.0)-(owner.residual[k]/normal_length)*(owner.residual[l]/normal_length);
            u[k] += P[3*k+l]*displacement[l];
        }
        const double slip = contact_detail::norm(u), eps_u = in.dt*in.eps_v;
        const double mollifier = slip < eps_u ? 2/eps_u-slip/(eps_u*eps_u) : 1/slip;
        const double scale = in.dt*in.dt*in.friction*(-in.k_barrier*owner.bp)*mollifier;
        for (int k = 0; k < 3; ++k) {
            out.friction_relative_gradient[k] = scale*u[k];
            for (int l = 0; l < 3; ++l) out.friction_relative_hessian[3*k+l] = scale*P[3*k+l];
        }
    }
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        if (!contact_detail::finite(out.barrier_gradient[i][k]) || !contact_detail::finite(out.friction_relative_gradient[k]))
            out.status = ContactStatus::NonFiniteDerivative;
        for (int j = 0; j < 4; ++j) for (int l = 0; l < 3; ++l)
            if (!contact_detail::finite(out.barrier_hessian[i][j][3*k+l]) || !contact_detail::finite(out.friction_relative_hessian[3*k+l]))
                out.status = ContactStatus::NonFiniteDerivative;
    }
    out.active = out.status == ContactStatus::Ok;
}

} // namespace ipc_gpu::resident::contact_cooperative_detail

#pragma once
// ISOLATED EXPERIMENT: no production call site includes this header.
// Caller retains the complete original feature/conversion/derivative pipeline.
#include "contact_cooperative.cuh"
static_assert(__cplusplus >= 201703L,
              "Point canonicalization arena proof requires C++17 prvalue elision");

#if defined(__CUDACC__)
#define IPC_POINT_CANON_HD __host__ __device__
#else
#define IPC_POINT_CANON_HD
#endif

namespace ipc_gpu::solver::point_hessian_canonical {
namespace ex = exact;
namespace cp = contact_cooperative_detail;
enum class Rejection : unsigned {
    None, Unsupported, InvalidFeature, WeightPattern, CommonDenominator,
    OriginalEntryCapacity, NoReduction
};
struct Gate {
    Rejection rejection = Rejection::Unsupported;
    int positive_role = -1, negative_role = -1;
    int original_denominator_bits = 0;
    IPC_POINT_CANON_HD bool eligible() const { return rejection == Rejection::None; }
};
template<int C> IPC_POINT_CANON_HD bool normalized(const ex::Integer<C>& a) {
    if (!a.valid || a.used < 0 || a.used > C) return false;
    if (a.used == 0) return a.sign == 0;
    return (a.sign == 1 || a.sign == -1) && a.limb[a.used-1] != 0;
}
template<int C> IPC_POINT_CANON_HD bool one(const ex::Integer<C>& a) {
    return normalized(a) && a.sign == 1 && a.used == 1 && a.limb[0] == 1;
}

// Allocation-free, source-bound to successful original prepare_derivatives.
// This checks eligibility, not arbitrary caller-forged polynomial identities.
template<int C> IPC_POINT_CANON_HD Gate point_gate(
    const ContactInput& in, int role, const cp::GeometryOwner<C>& owner) {
    Gate g;
    if (in.kind != ContactKind::PointTriangle || role < 0 || role >= 4
        || cp::self_block(in.hessian_blocks) != role
        || owner.feature.free_parameters != 0) return g;
    const auto& f = owner.feature;
    g.rejection = Rejection::InvalidFeature;
    if (!f.have || !owner.arithmetic_ok || !normalized(f.denominator)
        || f.denominator.sign != 1 || !normalized(f.squared_residual)
        || f.squared_residual.sign != 1 || !one(owner.z)) return g;
    for (int k = 0; k < 3; ++k) if (!normalized(f.residual[k])) return g;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        if (!normalized(owner.points.p[i].v[k])) return g;
    g.rejection = Rejection::WeightPattern;
    for (int i = 0; i < 4; ++i) {
        const auto& w = f.weight[i];
        if (!normalized(w)) return g;
        if (!w.sign) continue;
        if (ex::compare_magnitude(w, f.denominator) != 0) return g;
        if (w.sign > 0) { if (g.positive_role >= 0) return g; g.positive_role = i; }
        else { if (g.negative_role >= 0) return g; g.negative_role = i; }
    }
    if (g.positive_role < 0 || g.negative_role < 0) return g;
    g.rejection = Rejection::CommonDenominator;
    if (!normalized(owner.normal_den) || owner.normal_den.sign != 1
        || !normalized(owner.curvature_den) || owner.curvature_den.sign != 1
        || ex::compare_magnitude(owner.normal_den, owner.curvature_den) != 0
        || !normalized(owner.ww)) return g;
    g.original_denominator_bits = ex::bit_count(owner.normal_den);
    g.rejection = Rejection::OriginalEntryCapacity;
    if (g.original_denominator_bits > 32*C-54) return g;
    g.rejection = Rejection::NoReduction;
    if (one(f.denominator)) return g;
    g.rejection = Rejection::None;
    return g;
}

// Pool constraints are storage proofs, not numerical tolerances. This function
// is deliberately allocation-free. A pool must already be installed by the
// original kernel before calling it; never query uninitialized CUDA slots.
IPC_POINT_CANON_HD inline bool pool_bound(const ex::DevicePool* pool,
                                         int capacity, int needed) {
    return pool && !pool->failed && pool->words && pool->free_slots
        && pool->words_per_slot >= capacity && pool->slot_count >= needed
        && pool->available >= needed && pool->available <= pool->slot_count;
}

template<int C> struct Owner {
    ex::Integer<C> delta[3], squared, ww; // five owned slots; no borrowed owner
};

// Caller must gate >=256 free owner slots BEFORE placement construction of
// Owner<C>. Original geometry must remain immutable and alive. If this ever
// fails, retain failure and use the original scalar replay after destruction;
// do not clear pool.failed to turn this failure into a successful fast path.
template<int C> IPC_POINT_CANON_HD bool prepare(
    const cp::GeometryOwner<C>& original, int role, const Gate& gate,
    const cp::SharedGeometry& geometry, Owner<C>& canonical,
    cp::SharedGeometry& shadow) {
    if (!gate.eligible()) return false;
    ex::Arithmetic<C> a;
    canonical.squared = ex::Integer<C>(0);
    for (int k = 0; k < 3; ++k) {
        canonical.delta[k] = a.sub(original.points.p[gate.positive_role].v[k],
                                   original.points.p[gate.negative_role].v[k]);
        canonical.squared = a.sum(canonical.squared,
                                  a.mul(canonical.delta[k], canonical.delta[k]));
    }
    canonical.ww = ex::Integer<C>(role == gate.positive_role || role == gate.negative_role ? 1 : 0);
    if (!a.ok || !canonical.ww.valid || canonical.squared.sign != 1) return false;
    shadow = geometry;
    for (int k = 0; k < 3; ++k) shadow.residual[k] = cp::view(canonical.delta[k]);
    shadow.squared_residual = cp::view(canonical.squared);
    shadow.normal_den = shadow.curvature_den = cp::view(canonical.squared);
    shadow.ww = cp::view(canonical.ww);
    return true;
}

// Each helper independently retains its original entry if its private pool
// cannot prove 128 available slots. No zero-role bypass, altered ratio helper,
// approximate arithmetic, or changed FP expression appears here.
template<int C, bool InplaceRatio> IPC_POINT_CANON_HD cp::EntryResult entry(
    const cp::SharedGeometry& original, const cp::SharedGeometry& canonical,
    bool canonical_ready, int index) {
    bool use = canonical_ready;
#if defined(__CUDA_ARCH__)
    use = use && pool_bound(ex::current_pool(), C, 128);
#endif
    return cp::compute_entry<C, InplaceRatio>(use ? canonical : original, index);
}
} // namespace ipc_gpu::solver::point_hessian_canonical
#undef IPC_POINT_CANON_HD

#pragma once

#include "ccd.cuh"
#include "original_ccd.cuh"

#if defined(__CUDACC__)
#define IPC_ORIGINAL_RIGID_HD __host__ __device__
#else
#define IPC_ORIGINAL_RIGID_HD
#endif

namespace ipc_gpu::original_rigid {

// Same rigid-only squared-step cutoff as per_rigid_body_translation_safe_step.
// It skips TOI evaluation; the represented displacement is still committed.
IPC_ORIGINAL_RIGID_HD inline bool skip_translation_toi(const double delta[3]) {
    double squared=0;
    for(int k=0;k<3;++k)squared+=delta[k]*delta[k];
    return squared<1e-28;
}

// Match the CPU's unpadded world-space swept-AABB rejection BEFORE its
// relative-frame CCD query. A positive initial-distance band must not freeze
// motion whose world swept boxes are already disjoint.
IPC_ORIGINAL_RIGID_HD inline bool translation_boxes_overlap(const double start[4][3],
    bool face,bool first_moving,const double delta[3]) {
    const int split=face?1:2;
    for(int k=0;k<3;++k) {
        double lo[2]={INFINITY,INFINITY},hi[2]={-INFINITY,-INFINITY};
        for(int i=0;i<4;++i) {
            const bool moving=(i<split)==first_moving;
            const double end=start[i][k]+(moving?delta[k]:0.0);
            // Invalid/extreme data cannot be hidden by a broad rejection.
            if(!ccd::detail::finite(start[i][k])||!ccd::detail::finite(end))return true;
            const int side=i<split?0:1;
            lo[side]=fmin(lo[side],fmin(start[i][k],end));
            hi[side]=fmax(hi[side],fmax(start[i][k],end));
        }
        if(hi[0]<lo[1]||hi[1]<lo[0])return false;
    }
    return true;
}

// CPU rigid translation CCD works in a relative frame: PT always moves the
// point (with -delta for a moving triangle), and EE puts the translating edge
// first. Both endpoints of that edge receive the SAME raw displacement.
IPC_ORIGINAL_RIGID_HD inline original_ccd::Query translation_query(
    const double positions[4][3], bool face, bool first_moving,
    const double delta[3]) {
    original_ccd::Query q;
    q.vertex_face=face;
    q.eps=1e-12;
    for (int i=0;i<4;++i) for (int k=0;k<3;++k) {
        const int source=face || first_moving ? i : (i+2)%4;
        q.positions[i][k]=positions[source][k];
        q.motion[i][k]=face ? (i==0 ? (first_moving ? delta[k] : -delta[k]) : 0.0)
                           : (i<2 ? delta[k] : 0.0);
    }
    return q;
}

enum class RepresentationStatus : unsigned {
    Safe, Uncertain, InitialContact, Invalid, CapacityExceeded
};

// A rounded translation can give the vertices different EXACT end-start
// motions. Do not pass that multi-vertex path to the one-moving-vertex exact
// collision predicate, or assume rounded displacement equality proves rigid
// motion. Instead certify the actual represented affine trajectory with
// outward interval projections at ZERO gap. Subdivision only strengthens a
// proof; exhausting it is Uncertain and requires caller backtracking.
// Stationary/single-vertex cases retain the existing exact collision-only
// certificate, including explicit InitialContact and bounded resource errors.
template<int Capacity>
IPC_ORIGINAL_RIGID_HD inline RepresentationStatus represented_path(
    const double start[4][3], const double end[4][3], bool face) {
    ccd::LinearQuery q;
    q.vertex_face=face;
    int moving=0;
    for (int i=0;i<4;++i) {
        bool moved=false;
        for (int k=0;k<3;++k) {
            if (!ccd::detail::finite(start[i][k]) || !ccd::detail::finite(end[i][k]))
                return RepresentationStatus::Invalid;
            q.start[i][k]=start[i][k]; q.end[i][k]=end[i][k];
            moved=moved || start[i][k]!=end[i][k];
        }
        moving+=moved ? 1 : 0;
    }
    if (ccd::detail::separated(q,0.0,1.0,0.0)) return RepresentationStatus::Safe;
    const auto convert=[](exact::Status status) {
        return status==exact::Status::Safe ? RepresentationStatus::Safe
             : status==exact::Status::InitialContact ? RepresentationStatus::InitialContact
             : status==exact::Status::CapacityExceeded ? RepresentationStatus::CapacityExceeded
             : status==exact::Status::Invalid ? RepresentationStatus::Invalid
             : RepresentationStatus::Uncertain;
    };
    if (moving<=1) return convert(exact::validate_collision_step<Capacity>(start,end,face));
    // Check true initial membership exactly before a conservative proof can
    // request shortening. A genuinely intersecting start must never freeze
    // silently and masquerade as a successful no-op.
    const auto initial=exact::validate_collision_step<Capacity>(start,start,face);
    if (initial!=exact::Status::Safe) return convert(initial);
    constexpr int intervals=256;
    for (int i=0;i<intervals;++i)
        if (!ccd::detail::separated(q,double(i)/intervals,double(i+1)/intervals,0.0))
            return RepresentationStatus::Uncertain;
    return RepresentationStatus::Safe;
}

} // namespace ipc_gpu::original_rigid

#undef IPC_ORIGINAL_RIGID_HD

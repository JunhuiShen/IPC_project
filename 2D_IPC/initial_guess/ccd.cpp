#include "ccd.h"
#include "../ccd.h"
#include "../broad_phase.h"
#include <algorithm>
#include <vector>

double ccd_initial_guess_safe_step(const Vec& x, const Vec& v,
                                   const std::vector<NodeSegmentPair>& pairs,
                                   double dt, double eta) {
    double omega = 1.0;

    for (const auto& c : pairs) {
        Vec2 xi = get_xi(x, c.node);
        Vec2 xj = get_xi(x, c.seg0);
        Vec2 xk = get_xi(x, c.seg1);

        Vec2 vi = get_xi(v, c.node);
        Vec2 vj = get_xi(v, c.seg0);
        Vec2 vk = get_xi(v, c.seg1);

        Vec2 dxi{dt * vi.x, dt * vi.y};
        Vec2 dxj{dt * vj.x, dt * vj.y};
        Vec2 dxk{dt * vk.x, dt * vk.y};

        omega = std::min(omega, point_segment_ccd_safe_step(xi, dxi, xj, dxj, xk, dxk, eta));

        if (omega <= 0.0) return 0.0;
    }

    return omega;
}

void apply_ccd_initial_guess(const State2D& state, const RefMesh& ref_mesh,
                             Vec& xnew, double dt, double eta) {
    xnew = state.x;

    // Broad-phase the candidate node-segment pairs with the BVH (O(n log n)).
    // Pairs whose swept AABBs don't overlap over [0, dt] cannot collide, so
    // their safe step is 1.0 and omitting them leaves omega unchanged.
    BroadPhase broad_phase;
    auto pairs = broad_phase.build_ccd_candidates(state.x, state.v, ref_mesh.edges, dt);
    double omega = ccd_initial_guess_safe_step(state.x, state.v, pairs, dt, eta);

    for (int i = 0; i < state.size(); ++i) {
        Vec2 xi = get_xi(state.x, i);
        if (state.is_pinned[i]) {
            set_xi(xnew, i, xi);
            continue;
        }

        Vec2 vi = get_xi(state.v, i);
        set_xi(xnew, i, {xi.x + omega * dt * vi.x, xi.y + omega * dt * vi.y});
    }
}

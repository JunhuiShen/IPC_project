#include "trust_region.h"
#include "../broad_phase/bvh.h"
#include "../step_filter/trust_region.h"
#include <algorithm>

namespace initial_guess::trust_region {

    using namespace math;

    double global_safe_step(const Vec& x, const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt, double eta) {
        double omega = 1.0;
        for (const auto& c : pairs) {
            Vec2 xi = get_xi(x, c.node), xj = get_xi(x, c.seg0), xk = get_xi(x, c.seg1);
            Vec2 vi = get_xi(v, c.node), vj = get_xi(v, c.seg0), vk = get_xi(v, c.seg1);

            omega = std::min(omega, step_filter::trust_region::weight(
                xi, {dt*vi.x, dt*vi.y},
                xj, {dt*vj.x, dt*vj.y},
                xk, {dt*vk.x, dt*vk.y}, eta));

            if (omega <= 0.0) return 0.0;
        }
        return omega;
    }

} // namespace initial_guess::trust_region

void TrustRegionGuess::apply(Chain& left, Chain& right,
                             Vec& xnew_left, Vec& xnew_right,
                             Vec& x_combined, Vec& v_combined,
                             double dt, double eta) {
    using namespace math;

    combine_positions(x_combined, left.x, right.x, left.N, right.N);
    for (int i = 0; i < left.N;  ++i) set_xi(v_combined, i,          get_xi(left.v,  i));
    for (int i = 0; i < right.N; ++i) set_xi(v_combined, left.N + i, get_xi(right.v, i));

    double vmax = 0.0;
    for (int i = 0; i < left.N + right.N; ++i)
        vmax = std::max(vmax, norm(get_xi(v_combined, i)));

    BVHBroadPhase bp;
    double motion_pad = dt * vmax / eta;
    auto pairs = bp.build_trust_region_candidates(x_combined, v_combined,
                                                   left.N, right.N, dt, motion_pad);

    double alpha = std::max(0.0, std::min(1.0,
        initial_guess::trust_region::global_safe_step(x_combined, v_combined, pairs, dt, eta)));

    xnew_left  = left.x;
    xnew_right = right.x;
    for (int i = 0; i < left.N; ++i) {
        Vec2 xi = get_xi(left.x, i), vi = get_xi(left.v, i);
        set_xi(xnew_left,  i, {xi.x + alpha*dt*vi.x, xi.y + alpha*dt*vi.y});
    }
    for (int i = 0; i < right.N; ++i) {
        Vec2 xi = get_xi(right.x, i), vi = get_xi(right.v, i);
        set_xi(xnew_right, i, {xi.x + alpha*dt*vi.x, xi.y + alpha*dt*vi.y});
    }
}

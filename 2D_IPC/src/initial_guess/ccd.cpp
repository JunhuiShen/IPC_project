#include "ccd.h"
#include "../broad_phase/bvh.h"
#include "../step_filter/ccd.h"
#include <algorithm>
#include <vector>

namespace initial_guess::ccd {

    double global_safe_step(const Vec& x,
                            const Vec& v,
                            const std::vector<physics::NodeSegmentPair>& pairs,
                            double dt,
                            double eta) {
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

            omega = std::min(
                    omega,
                    step_filter::ccd::safe_step(xi, dxi, xj, dxj, xk, dxk, eta)
            );

            if (omega <= 0.0)
                return 0.0;
        }

        return omega;
    }

    void apply(const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined,
               const std::vector<char>& segment_valid,
               double dt,
               double eta) {

        const int total = total_nodes(blocks);
        x_combined.assign(2 * total, 0.0);
        v_combined.assign(2 * total, 0.0);

        build_x_combined_from_current_positions(x_combined, blocks);
        build_v_combined_from_chain_velocities(v_combined, blocks);

        BVHBroadPhase bp;
        auto pairs = bp.build_ccd_candidates(x_combined, v_combined, segment_valid, dt);
        double omega = global_safe_step(x_combined, v_combined, pairs, dt, eta);

        for (const auto& b : blocks) {
            Chain& c = *b.chain;
            Vec& xnew = *b.xnew;
            xnew = c.x;

            for (int i = 0; i < c.N; ++i) {
                Vec2 xi = get_xi(c.x, i);

                if (c.is_pinned[i]) {
                    set_xi(xnew, i, xi);
                    continue;
                }

                Vec2 vi = get_xi(c.v, i);
                set_xi(xnew, i, {
                        xi.x + omega * dt * vi.x,
                        xi.y + omega * dt * vi.y
                });
            }
        }

        build_x_combined_from_xnew(x_combined, blocks);
    }

} // namespace initial_guess::ccd

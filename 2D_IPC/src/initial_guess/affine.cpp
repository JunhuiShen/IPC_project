#include "affine.h"
#include <cmath>

namespace initial_guess::affine {

    Params compute_affine_params_global(const std::vector<BlockRef>& blocks) {
        Vec2 xcom{0.0, 0.0};
        double M = 0.0;

        for (const auto& b : blocks) {
            const Chain& c = *b.chain;
            for (int i = 0; i < c.N; ++i) {
                if (c.is_pinned[i]) continue;

                Vec2 xi = get_xi(c.x, i);
                xcom.x += c.mass[i] * xi.x;
                xcom.y += c.mass[i] * xi.y;
                M += c.mass[i];
            }
        }

        if (M <= 1e-12) {
            return {0.0, {0.0, 0.0}, {0.0, 0.0}};
        }

        xcom.x /= M;
        xcom.y /= M;

        double G[3][3] = {{0.0}};
        double bvec[3] = {0.0, 0.0, 0.0};

        for (const auto& br : blocks) {
            const Chain& c = *br.chain;
            for (int i = 0; i < c.N; ++i) {
                if (c.is_pinned[i]) continue;

                Vec2 Xi = get_xi(c.x, i);
                Vec2 Vi = get_xi(c.v, i);
                Vec2 d{Xi.x - xcom.x, Xi.y - xcom.y};

                Vec2 U1{-d.y, d.x};
                Vec2 U2{1.0, 0.0};
                Vec2 U3{0.0, 1.0};
                Vec2 U[3] = {U1, U2, U3};

                double w = c.mass[i];

                for (int k = 0; k < 3; ++k) {
                    bvec[k] += w * (U[k].x * Vi.x + U[k].y * Vi.y);
                    for (int j = 0; j < 3; ++j) {
                        G[k][j] += w * (U[k].x * U[j].x + U[k].y * U[j].y);
                    }
                }
            }
        }

        double omega  = (std::abs(G[0][0]) > 1e-12) ? bvec[0] / G[0][0] : 0.0;
        double vhat_x = (std::abs(G[1][1]) > 1e-12) ? bvec[1] / G[1][1] : 0.0;
        double vhat_y = (std::abs(G[2][2]) > 1e-12) ? bvec[2] / G[2][2] : 0.0;

        return {omega, {vhat_x, vhat_y}, xcom};
    }

    Vec2 velocity_at(const Params& ap, const Vec2& x) {
        Vec2 d{x.x - ap.xcom.x, x.y - ap.xcom.y};
        return {ap.vhat.x - ap.omega * d.y, ap.vhat.y + ap.omega * d.x};
    }

    void build_v_combined_from_affine(Vec& v_combined,
                                      const std::vector<BlockRef>& blocks,
                                      const Params& ap) {
        for (const auto& b : blocks) {
            const Chain& c = *b.chain;
            for (int i = 0; i < c.N; ++i) {
                Vec2 xi = get_xi(c.x, i);
                Vec2 vi = c.is_pinned[i] ? Vec2{0.0, 0.0} : velocity_at(ap, xi);
                set_xi(v_combined, b.offset + i, vi);
            }
        }
    }

    void apply_to_block(const Params& ap, const BlockRef& b, double dt) {
        Chain& c = *b.chain;
        Vec& xnew = *b.xnew;
        xnew = c.x;

        for (int i = 0; i < c.N; ++i) {
            Vec2 xi = get_xi(c.x, i);

            if (c.is_pinned[i]) {
                set_xi(xnew, i, xi);
                continue;
            }

            Vec2 v_aff = velocity_at(ap, xi);
            set_xi(xnew, i, {xi.x + dt * v_aff.x, xi.y + dt * v_aff.y});
        }
    }

} // namespace initial_guess::affine
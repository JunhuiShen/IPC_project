#include "affine.h"
#include <cmath>

namespace initial_guess::affine {

    Params compute(const Chain& A, const Chain& B) {
        Vec2 xcom{0.0, 0.0};
        double M = 0.0;

        auto accumulate_com = [&](const Chain& c) {
            for (int i = 0; i < c.N; ++i) {
                Vec2 xi = get_xi(c.x, i);
                xcom.x += c.mass[i] * xi.x;
                xcom.y += c.mass[i] * xi.y;
                M      += c.mass[i];
            }
        };
        accumulate_com(A);
        accumulate_com(B);
        xcom.x /= M;
        xcom.y /= M;

        double G[3][3] = {{0.0}};
        double b[3]    = {0.0};

        auto accumulate_ls = [&](const Chain& c, bool is_A) {
            for (int i = 0; i < c.N; ++i) {
                Vec2 Xi = get_xi(c.x, i);
                Vec2 Vi = get_xi(c.v, i);
                Vec2 d{Xi.x - xcom.x, Xi.y - xcom.y};
                Vec2 U[3] = {{-d.y, d.x}, {1.0, 0.0}, {0.0, 1.0}};

                double w = c.mass[i];
                if (i == 0) w = 0.0; // pinned nodes excluded

                for (int k = 0; k < 3; ++k) {
                    b[k] += w * (U[k].x * Vi.x + U[k].y * Vi.y);
                    for (int j = 0; j < 3; ++j)
                        G[k][j] += w * (U[k].x * U[j].x + U[k].y * U[j].y);
                }
            }
        };
        accumulate_ls(A, true);
        accumulate_ls(B, false);

        double omega  = (std::abs(G[0][0]) > 1e-12) ? b[0] / G[0][0] : 0.0;
        double vhat_x = (std::abs(G[1][1]) > 1e-12) ? b[1] / G[1][1] : 0.0;
        double vhat_y = (std::abs(G[2][2]) > 1e-12) ? b[2] / G[2][2] : 0.0;

        return {omega, {vhat_x, vhat_y}, xcom};
    }

    Vec2 velocity_at(const Params& ap, const Vec2& x) {
        Vec2 d{x.x - ap.xcom.x, x.y - ap.xcom.y};
        return {ap.vhat.x - ap.omega * d.y, ap.vhat.y + ap.omega * d.x};
    }

    void apply_to_chain(const Params& ap, const Chain& c, Vec& xnew, double dt) {
        for (int i = 0; i < c.N; ++i) {
            Vec2 xi = get_xi(c.x, i);
            Vec2 v  = velocity_at(ap, xi);
            set_xi(xnew, i, {xi.x + dt * v.x, xi.y + dt * v.y});
        }
    }

} // namespace initial_guess::affine

void AffineGuess::apply(Chain& left, Chain& right,
                        Vec& xnew_left, Vec& xnew_right,
                        Vec& x_combined, Vec& v_combined,
                        double dt, double eta) {
    using namespace initial_guess::affine;
    Params ap = compute(left, right);

    for (int i = 0; i < left.N;  ++i)
        set_xi(v_combined, i,          velocity_at(ap, get_xi(left.x,  i)));
    for (int i = 0; i < right.N; ++i)
        set_xi(v_combined, left.N + i, velocity_at(ap, get_xi(right.x, i)));

    xnew_left  = left.x;
    xnew_right = right.x;
    apply_to_chain(ap, left,  xnew_left,  dt);
    apply_to_chain(ap, right, xnew_right, dt);
    combine_positions(x_combined, left.x, right.x, left.N, right.N);
}

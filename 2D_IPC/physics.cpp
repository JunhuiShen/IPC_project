#include "physics.h"

namespace physics {
    using namespace math;

    // ======================================================
    // Incremental potential (no barrier)
    // ======================================================

    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               int rest_offset,
                               const std::vector<char> &is_pinned,
                               double dt, double k, const Vec2 &g_accel) {

        Vec2 xi = get_xi(x, i), xhi = get_xi(xhat, i);
        Vec2 gi{0.0, 0.0};

        gi.x += mass[i] * (xi.x - xhi.x);
        gi.y += mass[i] * (xi.y - xhi.y);

        Vec2 gs = local_spring_grad(i, x, k, L, rest_offset);
        gi.x += dt * dt * gs.x;
        gi.y += dt * dt * gs.y;

        gi.x -= dt * dt * mass[i] * g_accel.x;
        gi.y -= dt * dt * mass[i] * g_accel.y;

        constexpr double k_pin = 5e6;

        if (is_pinned[i]) {
            Vec2 xpi = get_xi(xpin, i);
            gi.x += dt * dt * k_pin * (xi.x - xpi.x);
            gi.y += dt * dt * k_pin * (xi.y - xpi.y);
        }

        return gi;
    }

    Mat2 local_hess_no_barrier(int i, const Vec &x,
                               const std::vector<double> &mass,
                               const std::vector<double> &L,
                               int rest_offset,
                               const std::vector<char> &is_pinned,
                               double dt, double k) {

        Mat2 H{mass[i], 0, 0, mass[i]};

        Mat2 Hs = local_spring_hess(i, x, k, L, rest_offset);
        H.a11 += dt * dt * Hs.a11;
        H.a12 += dt * dt * Hs.a12;
        H.a21 += dt * dt * Hs.a21;
        H.a22 += dt * dt * Hs.a22;

        constexpr double k_pin = 5e6;

        if (is_pinned[i]) {
            H.a11 += dt * dt * k_pin;
            H.a22 += dt * dt * k_pin;
        }

        return H;
    }

} // namespace physics

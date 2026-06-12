#include "state.h"

void build_xhat(Vec& xhat, const Vec& x, const Vec& v, double dt) {
    const int num_positions = static_cast<int>(x.size() / 2);
    xhat.resize(x.size());
    for (int i = 0; i < num_positions; ++i) {
        Vec2 xi = get_xi(x, i);
        Vec2 vi = get_xi(v, i);
        set_xi(xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
    }
}

void update_velocity(Vec& v, const Vec& xnew, const Vec& xold, double dt) {
    const int num_positions = static_cast<int>(xnew.size() / 2);
    v.resize(xnew.size());
    for (int i = 0; i < num_positions; ++i) {
        Vec2 xi_new = get_xi(xnew, i);
        Vec2 xi_old = get_xi(xold, i);
        set_xi(v, i, {
                (xi_new.x - xi_old.x) / dt,
                (xi_new.y - xi_old.y) / dt
        });
    }
}

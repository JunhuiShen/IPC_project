#include "chain.h"

Chain make_chain(Vec2 start, Vec2 end, int N, double mass_value) {
    Chain c;
    c.N = N;
    c.x.resize(2 * N);
    c.v.assign(2 * N, 0.0);
    c.xhat.assign(2 * N, 0.0);
    c.xpin.assign(2 * N, 0.0);
    c.mass.assign(N, mass_value);
    c.is_pinned.assign(N, 0);

    for (int i = 0; i < N; ++i) {
        double t = (N == 1) ? 0.0 : double(i) / (N - 1);
        Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
        set_xi(c.x, i, xi);
        set_xi(c.xpin, i, xi);
    }

    for (int i = 0; i < N - 1; ++i) {
        c.edges.emplace_back(i, i + 1);
        c.rest_lengths.push_back(math::node_distance(c.x, i, i + 1));
    }

    return c;
}

void build_xhat(Chain& c, double dt) {
    for (int i = 0; i < c.N; ++i) {
        Vec2 xi = get_xi(c.x, i);
        Vec2 vi = get_xi(c.v, i);
        set_xi(c.xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
    }
}

void update_velocity(Chain& c, const Vec& xnew, double dt) {
    for (int i = 0; i < c.N; ++i) {
        Vec2 xi_new = get_xi(xnew, i);
        Vec2 xi_old = get_xi(c.x, i);
        set_xi(c.v, i, {(xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt});
    }
    c.x = xnew;
}

void scatter_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block) {
    for (int i = 0; i < N_block; ++i)
        set_xi(x_combined, offset + i, get_xi(x_block, i));
}

void scatter_chain_positions(Vec& x_combined, const Chain& c, int offset) {
    scatter_positions(x_combined, c.x, offset, c.N);
}
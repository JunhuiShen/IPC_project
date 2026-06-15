#include "affine.h"
#include <cmath>

AffineInitialGuessParams compute_affine_initial_guess_params(
        const DeformedState& state, const RefMesh& ref_mesh, const std::vector<Pin>& pins) {
    Vec2 xcom{0.0, 0.0};
    double M = 0.0;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    for (int i = 0; i < total_nodes; ++i) {
        if (pin_map[i] >= 0) continue;

        Vec2 xi = get_xi(state.deformed_positions, i);
        const double mass = ref_mesh.mass[i];
        xcom.x += mass * xi.x;
        xcom.y += mass * xi.y;
        M += mass;
    }

    if (M <= 1e-12) {
        return {0.0, {0.0, 0.0}, {0.0, 0.0}};
    }

    xcom.x /= M;
    xcom.y /= M;

    double G[3][3] = {{0.0}};
    double bvec[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < total_nodes; ++i) {
        if (pin_map[i] >= 0) continue;

        Vec2 Xi = get_xi(state.deformed_positions, i);
        Vec2 Vi = get_xi(state.velocities, i);
        Vec2 d{Xi.x - xcom.x, Xi.y - xcom.y};

        Vec2 U1{-d.y, d.x};
        Vec2 U2{1.0, 0.0};
        Vec2 U3{0.0, 1.0};
        Vec2 U[3] = {U1, U2, U3};

        double w = ref_mesh.mass[i];

        for (int k = 0; k < 3; ++k) {
            bvec[k] += w * (U[k].x * Vi.x + U[k].y * Vi.y);
            for (int j = 0; j < 3; ++j) {
                G[k][j] += w * (U[k].x * U[j].x + U[k].y * U[j].y);
            }
        }
    }

    double omega  = (std::abs(G[0][0]) > 1e-12) ? bvec[0] / G[0][0] : 0.0;
    double vhat_x = (std::abs(G[1][1]) > 1e-12) ? bvec[1] / G[1][1] : 0.0;
    double vhat_y = (std::abs(G[2][2]) > 1e-12) ? bvec[2] / G[2][2] : 0.0;

    return {omega, {vhat_x, vhat_y}, xcom};
}

Vec2 affine_initial_guess_velocity(const AffineInitialGuessParams& params, const Vec2& x) {
    Vec2 d{x.x - params.xcom.x, x.y - params.xcom.y};
    return {
        params.vhat.x - params.omega * d.y,
        params.vhat.y + params.omega * d.x
    };
}

void apply_affine_initial_guess(const AffineInitialGuessParams& params,
                                const DeformedState& state, const std::vector<Pin>& pins,
                                Vec& xnew, double dt) {
    xnew = state.deformed_positions;
    const int total_nodes = static_cast<int>(state.deformed_positions.size());
    const PinMap pin_map = build_pin_map(pins, total_nodes);

    for (int i = 0; i < total_nodes; ++i) {
        Vec2 xi = get_xi(state.deformed_positions, i);

        if (pin_map[i] >= 0) {
            set_xi(xnew, i, pins[pin_map[i]].target_position);
            continue;
        }

        Vec2 v_aff = affine_initial_guess_velocity(params, xi);
        set_xi(xnew, i, {xi.x + dt * v_aff.x, xi.y + dt * v_aff.y});
    }
}

#pragma once

#include "ipc_math.h"
#include "sdf_penalty_energy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

enum class InitialGuessType {
    Trivial,
    CCD,
    Verlet
};

struct SimParams2D {
    double frame_dt = 1.0 / 30.0;
    int    substeps = 3;
    double k_spring = 1000.0;
    double kpin = 5e6;
    double k_barrier = 100.0;
    double k_sdf = 500.0;
    double eps_sdf = 0.002;
    std::vector<GroundSDF> sdf_grounds;
    std::vector<CircleSDF> sdf_circles;
    Vec2   gravity{0.0, -9.81};
    double d_hat = 0.005;
    double tol_abs = 1e-6;
    int    max_substep_iters = 500;
    double eta = 0.9;
    bool   use_parallel = false;
    bool   fixed_iters = false;
    double node_box_min = 0.001;
    double node_box_max = 0.01;
    double theta_box_min = 0.001;
    double theta_box_max = 0.05;
    int    node_box_update_count = 1;
    bool   use_ccd_step_policy = true;
    InitialGuessType initial_guess_type = InitialGuessType::CCD;

    double substep_dt() const {
        return frame_dt / static_cast<double>(std::max(1, substeps));
    }
};

struct Pin {
    int vertex_index = -1;
    Vec2 target_position{0.0, 0.0};
};

struct DeformedState {
    Vec deformed_positions;
    Vec velocities;

    // Rigid Bodies
    Vec x_coms; // center of mass position for each rb
    Vec v_coms; // center of mass velocity for each rb
    std::vector<double> theta; // orientation for each rb
    std::vector<double> omega; // angular velocity for each rb = [0,0,omega]

};

struct RefMesh {
    int num_positions{};
    std::vector<std::pair<int, int>> edges;
    std::vector<double> rest_lengths;
    std::vector<std::vector<int>> incident_edges;
    std::vector<double> mass;

    // Rigid Bodies
    std::vector<Vec> ref_positions; // array of particles in material space for each rb
    std::vector<Mat2> inertia_tensor; // inertia tensor for each rb
    std::vector<double> total_mass; // total_mass for each rb
    std::vector<std::vector<int>> rb_nodes; // global node indices for each rb

    inline void initialize(int n_positions, const std::vector<std::pair<int, int>>& input_edges, const Vec& rest_positions) {
        if (n_positions < 0 ||
            static_cast<int>(rest_positions.size()) != n_positions) {
            throw std::invalid_argument("RefMesh rest positions have an invalid size");
        }

        num_positions = n_positions;
        edges = input_edges;
        rest_lengths.clear();
        rest_lengths.reserve(edges.size());
        incident_edges.assign(num_positions, {});

        for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
            const auto [a, b] = edges[e];
            if (a < 0 || b < 0 || a >= num_positions || b >= num_positions || a == b) {
                throw std::invalid_argument("RefMesh contains an invalid edge");
            }
            const double rest_length = node_distance(rest_positions, a, b);
            if (!std::isfinite(rest_length) || rest_length <= 1e-12) {
                throw std::invalid_argument("RefMesh contains a degenerate edge");
            }
            rest_lengths.push_back(rest_length);
            incident_edges[a].push_back(e);
            incident_edges[b].push_back(e);
        }
    }
};

using PinMap = std::vector<int>;

inline PinMap build_pin_map(const std::vector<Pin>& pins, int num_positions) {
    PinMap map(num_positions, -1);
    for (int i = 0; i < static_cast<int>(pins.size()); ++i) {
        const int vertex = pins[i].vertex_index;
        if (vertex >= 0 && vertex < num_positions) map[vertex] = i;
    }
    return map;
}

inline void build_xhat(Vec& xhat, const Vec& x, const Vec& v, double dt) {
    const int num_positions = static_cast<int>(x.size());
    xhat.resize(x.size());
    for (int i = 0; i < num_positions; ++i) {
        Vec2 xi = get_xi(x, i);
        Vec2 vi = get_xi(v, i);
        set_xi(xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
    }
}

inline void update_velocity(Vec& v, const Vec& xnew, const Vec& xold, double dt) {
    const int num_positions = static_cast<int>(xnew.size());
    v.resize(xnew.size());
    for (int i = 0; i < num_positions; ++i) {
        Vec2 xi_new = get_xi(xnew, i);
        Vec2 xi_old = get_xi(xold, i);
        set_xi(v, i, {(xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt
        });
    }
}


Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const PinMap* pin_map,
                           double dt, double k_spring, double kpin, const Vec2 &g_accel);

Mat2 local_hess_no_barrier(int i, const Vec &x,  const RefMesh& ref_mesh, const PinMap* pin_map, double dt, double k_spring, double kpin);

void serialize_state(const std::string& dir, int frame, const DeformedState& state);
bool deserialize_state(const std::string& dir, int frame, DeformedState& state);

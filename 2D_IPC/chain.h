#pragma once

#include "ipc_math.h"
#include <utility>
#include <vector>

// ======================================================
// Chain: core data structure + construction + state update
// ======================================================

struct RefMesh {
    int num_positions{};
    std::vector<std::pair<int, int>> edges;
    std::vector<double> rest_lengths;
    std::vector<int> chain_node_offsets;
    std::vector<int> chain_rest_offsets;

    void initialize_from_chains(const std::vector<struct Chain>& chains,
                                const std::vector<int>& node_offsets);
};

struct Chain {
    int N{};
    Vec x;            // positions
    Vec v;            // velocities
    Vec xhat;         // predicted positions (linear extrapolation)
    Vec xpin;         // fixed pin targets
    std::vector<double> mass;
    std::vector<char> is_pinned;
};

// Build a uniformly-spaced chain from start to end with N nodes
Chain make_chain(Vec2 start, Vec2 end, int N, double mass_value, double thickness=0.01);

std::vector<int> compute_node_offsets(const std::vector<Chain>& chains);
RefMesh build_ref_mesh(const std::vector<Chain>& chains,
                       const std::vector<int>& node_offsets);

// xhat = x + dt * v
void build_xhat(Chain& c, double dt);

// v = (xnew - x) / dt, then x = xnew
void update_velocity(Chain& c, const Vec& xnew, double dt);

// Copy one block of node positions into the global position vector
void scatter_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block);

// Copy one chain's current positions into the global position vector
void scatter_chain_positions(Vec& x_combined, const Chain& c, int offset);

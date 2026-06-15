#pragma once

#include "ipc_math.h"
#include "mesh.h"
#include "state.h"
#include <vector>

// Chain is a scene-construction helper. Simulation uses State2D and RefMesh.
struct Chain {
    int N{};
    Vec x;
    Vec v;
    Vec xpin;
    std::vector<double> mass;
    std::vector<char> is_pinned;
};

// Build a uniformly-spaced chain from start to end with N nodes
Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness=0.001);

void assemble_chains(const std::vector<Chain>& chains, State2D& state, RefMesh& ref_mesh);

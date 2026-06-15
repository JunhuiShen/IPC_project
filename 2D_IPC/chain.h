#pragma once

#include "ipc_math.h"
#include "physics.h"
#include <vector>

// Chain is a scene-construction helper. Simulation uses DeformedState and RefMesh.
struct Chain {
    int N{};
    Vec deformed_positions;
    Vec velocities;
    std::vector<double> mass;
    std::vector<Pin> pins;
};

// Build a uniformly-spaced chain from start to end with N nodes
Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness=0.001);

void assemble_chains(const std::vector<Chain>& chains, DeformedState& state,
                     RefMesh& ref_mesh, std::vector<Pin>& pins);

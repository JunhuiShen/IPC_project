#pragma once

#include "ipc_math.h"
#include "physics.h"

#include <vector>

// A chain is one deformable 1D shape. Simulation uses global DeformedState and RefMesh.
struct Chain {
    int N{};
    Vec deformed_positions;
    Vec velocities;
    std::vector<double> mass;
    std::vector<Pin> pins;
};

Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness = 0.001);

void assemble_chains(const std::vector<Chain>& chains, DeformedState& state,
                     RefMesh& ref_mesh, std::vector<Pin>& pins);

int append_rigid_pentagon(DeformedState& state, RefMesh& ref_mesh, Vec2 center, double radius, double density,
        double thickness = 0.001, Vec2 v_com = {0.0, 0.0}, double theta = 0.0, double omega = 0.0);

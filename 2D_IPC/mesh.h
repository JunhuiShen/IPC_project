#pragma once

#include "ipc_math.h"
#include <utility>
#include <vector>

struct RefMesh {
    int num_positions{};
    std::vector<std::pair<int, int>> edges;
    std::vector<double> rest_lengths;
    std::vector<std::vector<int>> incident_edges;
};

void initialize_ref_mesh(RefMesh& ref_mesh, int num_positions,
                         const std::vector<std::pair<int, int>>& edges,
                         const Vec& rest_positions);

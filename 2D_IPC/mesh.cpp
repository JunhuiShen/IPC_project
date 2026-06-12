#include "mesh.h"
#include <cmath>
#include <stdexcept>

void initialize_ref_mesh(RefMesh& ref_mesh, int num_positions,
                         const std::vector<std::pair<int, int>>& edges,
                         const Vec& rest_positions) {
    if (num_positions < 0 ||
        static_cast<int>(rest_positions.size()) != 2 * num_positions) {
        throw std::invalid_argument("RefMesh rest positions have an invalid size");
    }

    ref_mesh.num_positions = num_positions;
    ref_mesh.edges = edges;
    ref_mesh.rest_lengths.clear();
    ref_mesh.rest_lengths.reserve(edges.size());
    ref_mesh.incident_edges.assign(num_positions, {});

    for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
        const auto [a, b] = edges[e];
        if (a < 0 || b < 0 || a >= num_positions || b >= num_positions || a == b) {
            throw std::invalid_argument("RefMesh contains an invalid edge");
        }
        const double rest_length = math::node_distance(rest_positions, a, b);
        if (!std::isfinite(rest_length) || rest_length <= 1e-12) {
            throw std::invalid_argument("RefMesh contains a degenerate edge");
        }
        ref_mesh.rest_lengths.push_back(rest_length);
        ref_mesh.incident_edges[a].push_back(e);
        ref_mesh.incident_edges[b].push_back(e);
    }
}

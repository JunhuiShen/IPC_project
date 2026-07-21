#include "mesh_utils.h"

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& ref_mesh, int tri_idx) {
    TriangleDef def;
    def.x[0] = x[tri_vertex(ref_mesh, tri_idx, 0)];
    def.x[1] = x[tri_vertex(ref_mesh, tri_idx, 1)];
    def.x[2] = x[tri_vertex(ref_mesh, tri_idx, 2)];
    return def;
}

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, std::vector<Pin>& pins) {
    X.clear();
    ref_mesh.tris.clear();
    state.deformed_positions.clear();
    state.velocities.clear();
    state.x_coms.clear();
    state.v_coms.clear();
    state.orientations.clear();
    state.omega.clear();
    ref_mesh.ref_positions.clear();
    ref_mesh.total_mass.clear();
    ref_mesh.I_hat.clear();
    ref_mesh.rb_nodes.clear();
    ref_mesh.node_to_rb.clear();
    pins.clear();
}

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vertex_index, x[vertex_index]});
}

// Builds VertexTriangleMap: each vertex maps to {triangle_index, local_node_index} pairs.
// Storing the local index (0,1,2) eliminates the linear search previously done by local_node().
VertexTriangleMap build_incident_triangle_map(const std::vector<int>& indices) {
    VertexTriangleMap result;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        const int vertex     = indices[i];
        const int tri_idx    = i / 3;
        const int local_node = i % 3;
        result[vertex].emplace_back(tri_idx, local_node);
    }
    return result;
}

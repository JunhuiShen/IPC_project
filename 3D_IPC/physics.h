#pragma once
#include "corotated_energy.h"
#include <vector>

struct Tri { int v[3]; };

struct Pin {
    int vertex_index = -1;
    Vec3 target_position = Vec3::Zero();
};

struct SimParams {
    double dt{}, mu{}, lambda{}, density{}, thickness{}, kpin{}, tol_abs{}, step_weight{};
    Vec3 gravity = Vec3::Zero();
    int max_global_iters{};
};

struct DeformedState {
    std::vector<Vec3> deformed_positions;
    std::vector<Vec3> velocities;
};

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<Tri> tris;
};

struct LumpedMass {
    std::vector<double> vertex_masses;
};


struct VertexAdjacency {
    std::vector<std::vector<int>> incident_triangle_indices;
};

double triangle_ref_area_2d(const RefMesh& ref_mesh, const Tri& tri);


LumpedMass build_lumped_mass(const RefMesh& ref_mesh, double density, double thickness);

VertexAdjacency build_vertex_adjacency(const RefMesh& ref_mesh);

double compute_incremental_potential(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const std::vector<Pin>& pins,
                                     const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                            const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Mat33 compute_local_hessian(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                            const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x);

double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                               const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);